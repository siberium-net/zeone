import asyncio
import pytest

from core.node import Node
from core.transport import Crypto
from economy.ledger import Ledger
from core.dht.node import KademliaNode
from config import config


pytestmark = pytest.mark.asyncio


async def wait_for(condition, timeout=5.0, interval=0.05):
    """Poll condition() until True or timeout."""
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        if condition():
            return True
        await asyncio.sleep(interval)
    return False


async def test_two_nodes_store_and_retrieve(tmp_path):
    # Avoid external bootstrap attempts during the test
    old_bootstrap = list(config.network.bootstrap_nodes)
    config.network.bootstrap_nodes = []
    
    alpha_ledger = Ledger(db_path=":memory:")
    beta_ledger = Ledger(db_path=":memory:")
    await asyncio.gather(alpha_ledger.initialize(), beta_ledger.initialize())
    
    alpha = Node(Crypto(), host="127.0.0.1", port=9000, ledger=alpha_ledger)
    beta = Node(Crypto(), host="127.0.0.1", port=9001, ledger=beta_ledger)
    
    alpha_dht = KademliaNode(alpha, storage_path=":memory:", use_memory_storage=True)
    beta_dht = KademliaNode(beta, storage_path=":memory:", use_memory_storage=True)
    
    try:
        # Start nodes and DHT layers
        await asyncio.gather(alpha.start(), beta.start())
        await asyncio.gather(alpha_dht.start(), beta_dht.start())
        
        if not alpha._server or not alpha._server.sockets or not beta._server or not beta._server.sockets:
            pytest.skip("TCP sockets unavailable in sandbox; skipping end-to-end P2P test")
        
        # Connect beta to alpha over TCP
        peer = await beta.connect_to_peer("127.0.0.1", 9000)
        assert peer is not None
        
        # Wait for both sides to register the connection
        connected = await wait_for(
            lambda: alpha.peer_manager.peer_count > 0 and beta.peer_manager.peer_count > 0,
            timeout=5.0,
        )
        assert connected, "Handshake between nodes did not complete in time"
        
        # Beta stores value in DHT
        await asyncio.wait_for(
            beta_dht.dht_put("mission", b"siberium_success"),
            timeout=5.0,
        )
        
        # Give a moment for STORE to propagate and be processed
        await asyncio.sleep(0.2)
        
        # Alpha retrieves the value
        value = await asyncio.wait_for(
            alpha_dht.dht_get("mission"),
            timeout=5.0,
        )
        
        assert value == b"siberium_success"
    finally:
        config.network.bootstrap_nodes = old_bootstrap
        await asyncio.gather(beta_dht.stop(), alpha_dht.stop(), return_exceptions=True)
        await asyncio.gather(beta.stop(), alpha.stop(), return_exceptions=True)
        await asyncio.gather(beta_ledger.close(), alpha_ledger.close(), return_exceptions=True)
