"""
ZEONE Test Configuration
========================

[QA] Central pytest configuration with fixtures for all test types:
- Unit tests: Isolated, no I/O, fast
- Integration tests: Real async I/O, temp databases
- E2E tests: Full node stack with networking

[FIXTURES]
- node_factory: Spawn N nodes in memory or localhost
- mock_gpu: Stub for AI/GPU operations
- mock_chain: Fake blockchain for economy tests
- isolated_db: Per-test temporary database

Usage:
    pytest tests/unit/          # Fast unit tests
    pytest tests/integration/   # Integration tests
    pytest tests/e2e/           # End-to-end tests
    pytest --cov=zeone          # With coverage
"""

import os
import sys
import asyncio
import tempfile
import shutil
import logging
from pathlib import Path
from typing import AsyncGenerator, Generator, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from unittest.mock import MagicMock, AsyncMock, patch
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (I/O, slower)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full stack)")
    config.addinivalue_line("markers", "stress: Stress/load tests")
    config.addinivalue_line("markers", "slow: Slow tests (skip with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU (skip without hardware)")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their path."""
    for item in items:
        # Auto-add markers based on path
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Mark async tests
        if asyncio.iscoroutinefunction(item.obj):
            item.add_marker(pytest.mark.asyncio)


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="function")
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Logging Configuration
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Silence noisy loggers during tests
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.ERROR)


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp(prefix="zeone_test_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_file(temp_dir: Path) -> Callable[[str, bytes], Path]:
    """Factory to create temporary files."""
    def _create(name: str, content: bytes = b"") -> Path:
        path = temp_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path
    return _create


# ============================================================================
# Database Fixtures (Isolated)
# ============================================================================

@pytest.fixture(scope="function")
def isolated_db() -> Generator[str, None, None]:
    """
    Create isolated database for each test.
    Uses :memory: for speed, or temp file for persistence tests.
    """
    # Use in-memory SQLite
    yield ":memory:"


@pytest.fixture(scope="function")
def isolated_db_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create isolated database file for persistence tests."""
    db_path = temp_dir / "test_ledger.db"
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest_asyncio.fixture(scope="function")
async def ledger(isolated_db: str):
    """Create isolated Ledger instance."""
    from economy.ledger import Ledger
    
    ledger_instance = Ledger(isolated_db, debt_limit=1_000_000)
    await ledger_instance.initialize()
    yield ledger_instance
    await ledger_instance.close()


# ============================================================================
# Crypto Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def crypto():
    """Create fresh Crypto identity for each test."""
    from core.transport import Crypto
    return Crypto()


@pytest.fixture(scope="function")
def crypto_pair():
    """Create a pair of Crypto identities for sender/receiver tests."""
    from core.transport import Crypto
    return Crypto(), Crypto()


@pytest.fixture(scope="function")
def node_identity(temp_dir: Path):
    """Create and persist node identity."""
    from core.transport import Crypto
    
    crypto = Crypto()
    identity_file = temp_dir / "identity.key"
    identity_file.write_bytes(crypto.export_identity())
    return crypto, identity_file


# ============================================================================
# Node Factory Fixture
# ============================================================================

@dataclass
class TestNode:
    """Lightweight test node wrapper."""
    node_id: str
    host: str
    port: int
    crypto: Any
    ledger: Any
    node: Optional[Any] = None
    started: bool = False


class NodeFactory:
    """
    Factory for spawning test nodes.
    
    [USAGE]
        nodes = await node_factory.create(3)  # Create 3 nodes
        await node_factory.connect_all()       # Connect them
        await node_factory.cleanup()           # Cleanup after test
    """
    
    def __init__(self, base_port: int = 19000):
        self.base_port = base_port
        self.nodes: List[TestNode] = []
        self._port_counter = 0
    
    def _next_port(self) -> int:
        """Get next available port."""
        port = self.base_port + self._port_counter
        self._port_counter += 1
        return port
    
    async def create(
        self,
        count: int = 1,
        start: bool = False,
        in_memory_db: bool = True,
    ) -> List[TestNode]:
        """
        Create N test nodes.
        
        Args:
            count: Number of nodes to create
            start: Whether to start the nodes
            in_memory_db: Use in-memory database
        """
        from core.transport import Crypto
        from economy.ledger import Ledger
        
        created = []
        
        for i in range(count):
            crypto = Crypto()
            port = self._next_port()
            
            # Create ledger
            db_path = ":memory:" if in_memory_db else f"test_node_{port}.db"
            ledger = Ledger(db_path)
            await ledger.initialize()
            
            test_node = TestNode(
                node_id=crypto.node_id,
                host="127.0.0.1",
                port=port,
                crypto=crypto,
                ledger=ledger,
            )
            
            if start:
                await self._start_node(test_node)
            
            self.nodes.append(test_node)
            created.append(test_node)
        
        return created
    
    async def _start_node(self, test_node: TestNode) -> None:
        """Start a test node."""
        try:
            from core.node import Node
            from agents.manager import AgentManager
            
            agent_manager = AgentManager(
                ledger=test_node.ledger,
                node_id=test_node.crypto.node_id,
            )
            
            node = Node(
                crypto=test_node.crypto,
                host=test_node.host,
                port=test_node.port,
                ledger=test_node.ledger,
                agent_manager=agent_manager,
            )
            
            await node.start()
            test_node.node = node
            test_node.started = True
            
        except Exception as e:
            logging.warning(f"Failed to start test node: {e}")
    
    async def connect_all(self) -> None:
        """Connect all nodes to each other."""
        for i, node in enumerate(self.nodes):
            if not node.started or not node.node:
                continue
            
            for j, other in enumerate(self.nodes):
                if i != j and other.started:
                    try:
                        await node.node.connect_to(other.host, other.port)
                    except Exception:
                        pass
    
    async def cleanup(self) -> None:
        """Stop and cleanup all nodes."""
        for test_node in self.nodes:
            try:
                if test_node.started and test_node.node:
                    await test_node.node.stop()
                await test_node.ledger.close()
            except Exception:
                pass
        
        self.nodes.clear()
        self._port_counter = 0


@pytest_asyncio.fixture(scope="function")
async def node_factory() -> AsyncGenerator[NodeFactory, None]:
    """
    Fixture providing NodeFactory for spawning test nodes.
    
    [USAGE]
        async def test_multi_node(node_factory):
            nodes = await node_factory.create(3, start=True)
            await node_factory.connect_all()
            # Test interactions
    """
    factory = NodeFactory()
    yield factory
    await factory.cleanup()


@pytest_asyncio.fixture(scope="function")
async def single_node(node_factory: NodeFactory) -> TestNode:
    """Create a single test node (not started)."""
    nodes = await node_factory.create(1, start=False)
    return nodes[0]


# ============================================================================
# Mock GPU / AI Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_gpu():
    """
    Mock GPU/CUDA operations for tests without hardware.
    
    [MOCKS]
    - torch.cuda.is_available() -> False
    - OllamaAgent.execute() -> Mock response
    - VisionEngine -> Mock detection
    """
    mocks = {}
    
    # Mock torch.cuda
    cuda_mock = MagicMock()
    cuda_mock.is_available.return_value = False
    cuda_mock.device_count.return_value = 0
    mocks["torch.cuda"] = cuda_mock
    
    # Mock torch itself (if imported)
    torch_mock = MagicMock()
    torch_mock.cuda = cuda_mock
    torch_mock.zeros = MagicMock(return_value=MagicMock())
    torch_mock.tensor = MagicMock(return_value=MagicMock())
    mocks["torch"] = torch_mock
    
    # Patch
    patches = []
    
    try:
        patches.append(patch.dict("sys.modules", {"torch": torch_mock}))
        patches.append(patch.dict("sys.modules", {"torch.cuda": cuda_mock}))
        
        for p in patches:
            p.start()
    except Exception:
        pass
    
    yield mocks
    
    for p in patches:
        try:
            p.stop()
        except Exception:
            pass


@pytest.fixture(scope="function")
def mock_ollama():
    """
    Mock Ollama API responses.
    
    [RETURNS]
    - Pre-defined LLM responses for testing
    - No network calls
    """
    async def mock_generate(prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            "response": f"Mock response to: {prompt[:50]}...",
            "model": "mock-model",
            "eval_count": len(prompt.split()) * 2,
            "prompt_eval_count": len(prompt.split()),
        }
    
    async def mock_list_models() -> List[str]:
        return ["mock-model", "mock-model-large"]
    
    mock = MagicMock()
    mock.generate = AsyncMock(side_effect=mock_generate)
    mock.list_models = AsyncMock(side_effect=mock_list_models)
    
    return mock


@pytest.fixture(scope="function")
def mock_vision():
    """
    Mock vision/face detection operations.
    
    [RETURNS]
    - Fake face detections
    - Fake image captions
    """
    async def mock_detect_faces(image_path: str) -> List[Dict]:
        return [
            {"bbox": [100, 100, 200, 200], "confidence": 0.95},
            {"bbox": [300, 100, 400, 200], "confidence": 0.88},
        ]
    
    async def mock_caption(image_path: str) -> str:
        return "A mock description of the image"
    
    mock = MagicMock()
    mock.detect_faces = AsyncMock(side_effect=mock_detect_faces)
    mock.caption = AsyncMock(side_effect=mock_caption)
    
    return mock


# ============================================================================
# Mock Blockchain / Web3 Fixtures
# ============================================================================

@dataclass
class MockAccount:
    """Mock blockchain account."""
    address: str
    balance: int = 1000 * 10**18  # 1000 SIBR
    nonce: int = 0


@dataclass
class MockTransaction:
    """Mock blockchain transaction."""
    hash: str
    from_address: str
    to_address: str
    value: int
    status: int = 1  # 1 = success


class MockWeb3:
    """
    Mock Web3 provider for testing without real blockchain.
    
    [FEATURES]
    - Simulates account balances
    - Tracks transactions
    - Mimics Siberium chainId
    """
    
    def __init__(self, chain_id: int = 111000):
        self.chain_id = chain_id
        self.block_number = 1000
        self.accounts: Dict[str, MockAccount] = {}
        self.transactions: Dict[str, MockTransaction] = {}
        self._tx_counter = 0
        
        # Setup mock structure
        self.eth = MagicMock()
        self.eth.chain_id = chain_id
        self.eth.block_number = self.block_number
        self.eth.gas_price = 1_000_000_000  # 1 Gwei
        self.eth.get_balance = self._get_balance
        self.eth.get_transaction_count = self._get_nonce
        self.eth.send_raw_transaction = self._send_transaction
        self.eth.wait_for_transaction_receipt = self._wait_receipt
    
    def is_connected(self) -> bool:
        return True
    
    def _get_balance(self, address: str) -> int:
        address = address.lower()
        if address not in self.accounts:
            self.accounts[address] = MockAccount(address=address)
        return self.accounts[address].balance
    
    def _get_nonce(self, address: str) -> int:
        address = address.lower()
        if address not in self.accounts:
            self.accounts[address] = MockAccount(address=address)
        return self.accounts[address].nonce
    
    def _send_transaction(self, raw_tx: bytes) -> bytes:
        """Simulate sending transaction."""
        self._tx_counter += 1
        tx_hash = f"0x{self._tx_counter:064x}"
        
        # Create mock transaction
        tx = MockTransaction(
            hash=tx_hash,
            from_address="0x" + "1" * 40,
            to_address="0x" + "2" * 40,
            value=0,
        )
        self.transactions[tx_hash] = tx
        self.block_number += 1
        
        return bytes.fromhex(tx_hash[2:])
    
    def _wait_receipt(self, tx_hash: bytes, **kwargs) -> MagicMock:
        """Return mock receipt."""
        hash_hex = "0x" + tx_hash.hex()
        tx = self.transactions.get(hash_hex)
        
        receipt = MagicMock()
        receipt.status = 1 if tx else 0
        receipt.blockNumber = self.block_number
        receipt.gasUsed = 21000
        receipt.contractAddress = None
        
        return receipt
    
    def fund_account(self, address: str, amount: int) -> None:
        """Add funds to test account."""
        address = address.lower()
        if address not in self.accounts:
            self.accounts[address] = MockAccount(address=address, balance=amount)
        else:
            self.accounts[address].balance += amount


@pytest.fixture(scope="function")
def mock_chain() -> MockWeb3:
    """
    Mock blockchain for economy tests.
    
    [USAGE]
        def test_settlement(mock_chain):
            mock_chain.fund_account(address, 1000 * 10**18)
            # Test settlement logic
    """
    return MockWeb3()


@pytest.fixture(scope="function")
def mock_settlement(mock_chain: MockWeb3):
    """Mock Settlement contract."""
    contract = MagicMock()
    contract.address = "0x" + "3" * 40
    
    # Mock functions
    contract.functions = MagicMock()
    contract.functions.stakedBalance = MagicMock(return_value=MagicMock(
        call=MagicMock(return_value=0)
    ))
    contract.functions.totalStaked = MagicMock(return_value=MagicMock(
        call=MagicMock(return_value=0)
    ))
    contract.functions.deposit = MagicMock(return_value=MagicMock(
        build_transaction=MagicMock(return_value={
            "gas": 100000,
            "gasPrice": mock_chain.eth.gas_price,
        })
    ))
    
    return contract


# ============================================================================
# DHT Fixtures
# ============================================================================

@pytest_asyncio.fixture(scope="function")
async def mock_dht():
    """
    Mock DHT for testing without network.
    
    [FEATURES]
    - In-memory key-value store
    - Simulated latency
    """
    storage: Dict[bytes, bytes] = {}
    
    async def put(key: str, value: bytes) -> int:
        key_bytes = key.encode() if isinstance(key, str) else key
        storage[key_bytes] = value
        return 1  # Stored on 1 node
    
    async def get(key: str) -> Optional[bytes]:
        key_bytes = key.encode() if isinstance(key, str) else key
        return storage.get(key_bytes)
    
    async def delete(key: str) -> bool:
        key_bytes = key.encode() if isinstance(key, str) else key
        if key_bytes in storage:
            del storage[key_bytes]
            return True
        return False
    
    mock = MagicMock()
    mock.dht_put = AsyncMock(side_effect=put)
    mock.dht_get = AsyncMock(side_effect=get)
    mock.dht_delete = AsyncMock(side_effect=delete)
    mock.storage = storage
    
    return mock


# ============================================================================
# Message Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def sample_message(crypto):
    """Create a sample signed message."""
    from core.transport import Message, MessageType
    
    return Message(
        type=MessageType.DATA,
        payload={"test": "data", "value": 42},
        sender_id=crypto.node_id,
    )


@pytest.fixture(scope="function")
def binary_message_factory(crypto):
    """Factory for creating binary protocol messages."""
    def _create(msg_type: int, payload: bytes = b"") -> bytes:
        try:
            from core.wire import pack_message, WireMessageType
            return pack_message(
                msg_type=WireMessageType(msg_type),
                payload=payload,
                signing_key=crypto._signing_key,
            )
        except ImportError:
            # Fallback for tests without wire module
            return b"ZE" + bytes([1, msg_type, 0, 0]) + payload
    
    return _create


# ============================================================================
# Agent Manager Fixtures
# ============================================================================

@pytest_asyncio.fixture(scope="function")
async def agent_manager(ledger, crypto):
    """Create AgentManager with mocked AI agents."""
    from agents.manager import AgentManager
    
    manager = AgentManager(ledger=ledger, node_id=crypto.node_id)
    
    # Don't register real AI agents in tests
    return manager


# ============================================================================
# HTTP/SOCKS Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_http_response():
    """Mock HTTP response for web reader tests."""
    async def create_response(
        status: int = 200,
        body: str = "<html><body>Test</body></html>",
        headers: Optional[Dict] = None,
    ):
        response = MagicMock()
        response.status = status
        response.headers = headers or {"Content-Type": "text/html"}
        response.text = AsyncMock(return_value=body)
        response.read = AsyncMock(return_value=body.encode())
        return response
    
    return create_response


# ============================================================================
# Async Utilities
# ============================================================================

@pytest.fixture(scope="function")
def async_timeout():
    """Helper for async test timeouts."""
    async def _timeout(coro, seconds: float = 5.0):
        return await asyncio.wait_for(coro, timeout=seconds)
    return _timeout


@pytest.fixture(scope="function")
def run_async():
    """Run async function in sync context."""
    def _run(coro):
        return asyncio.get_event_loop().run_until_complete(coro)
    return _run


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture(scope="function")
def random_bytes():
    """Generate random bytes for testing."""
    import secrets
    
    def _generate(size: int = 1024) -> bytes:
        return secrets.token_bytes(size)
    
    return _generate


@pytest.fixture(scope="function")
def random_peer_id():
    """Generate random peer ID."""
    import secrets
    
    def _generate() -> str:
        return secrets.token_hex(32)
    
    return _generate


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_files():
    """Cleanup temporary test files after session."""
    yield
    
    # Cleanup any leftover test databases
    for pattern in ["test_*.db", "dht_*.db"]:
        import glob
        for f in glob.glob(str(PROJECT_ROOT / pattern)):
            try:
                os.unlink(f)
            except Exception:
                pass
