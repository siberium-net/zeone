import pytest

from core.dht.protocol import DHTProtocol, FindNodeRequest
from core.dht.routing import RoutingTable, NodeInfo
from core.dht.storage import DHTStorage


pytestmark = pytest.mark.asyncio


class DummyStorage:
    async def get(self, key):
        return None
    
    async def store(self, key, value, publisher_id, ttl=None):
        return True
    
    async def close(self):
        return None


async def test_handle_message_returns_response(tmp_path):
    local_id = b"a" * 20
    routing = RoutingTable(local_id, k=3)
    protocol = DHTProtocol(
        routing_table=routing,
        storage=DummyStorage(),
        local_id=local_id,
        local_host="127.0.0.1",
        local_port=9999,
    )

    # Добавляем узел, который должен вернуться в ответе
    remote = NodeInfo(node_id=b"b" * 20, host="1.1.1.1", port=1111)
    routing.add_node(remote)
    other = NodeInfo(node_id=b"d" * 20, host="2.2.2.2", port=2222)
    routing.add_node(other)

    payload = FindNodeRequest(
        target_id=b"c" * 20,
        sender_id=remote.node_id,
        sender_host=remote.host,
        sender_port=remote.port,
    ).to_dict()

    response = await protocol.handle_message(payload)

    assert response is not None, "DHT RPC handler must return response"
    assert response.get("type") == "FIND_NODE_RESPONSE"
    assert response["nodes"][0]["node_id"] == other.node_id.hex()
