"""
Kademlia DHT Integration Tests
==============================

[INTEGRATION] Tests for DHT operations with mock network.
"""

import pytest
import pytest_asyncio


class TestMockDHT:
    """Test DHT operations with mock."""
    
    @pytest.mark.asyncio
    async def test_put_get(self, mock_dht):
        """Test basic put/get operations."""
        key = "test_key"
        value = b"test_value"
        
        # Put
        count = await mock_dht.dht_put(key, value)
        assert count >= 1
        
        # Get
        result = await mock_dht.dht_get(key)
        assert result == value
    
    @pytest.mark.asyncio
    async def test_get_missing_key(self, mock_dht):
        """Test getting non-existent key."""
        result = await mock_dht.dht_get("nonexistent_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete(self, mock_dht):
        """Test delete operation."""
        key = "key_to_delete"
        value = b"value"
        
        # Put
        await mock_dht.dht_put(key, value)
        
        # Delete
        deleted = await mock_dht.dht_delete(key)
        assert deleted is True
        
        # Verify gone
        result = await mock_dht.dht_get(key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_overwrite(self, mock_dht):
        """Test overwriting existing key."""
        key = "overwrite_key"
        
        # First put
        await mock_dht.dht_put(key, b"first_value")
        
        # Overwrite
        await mock_dht.dht_put(key, b"second_value")
        
        # Get should return new value
        result = await mock_dht.dht_get(key)
        assert result == b"second_value"


@pytest.mark.integration
class TestKademliaNode:
    """Test real Kademlia node (requires more setup)."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_kademlia_node_creation(self, node_factory):
        """Test creating Kademlia-enabled node."""
        try:
            from core.dht.node import KademliaNode
            
            # Create a node
            nodes = await node_factory.create(1, start=True)
            if not nodes or not nodes[0].started:
                pytest.skip("Node creation failed")
            
            node = nodes[0]
            
            # Create Kademlia
            kademlia = KademliaNode(node.node, storage_path=":memory:")
            await kademlia.start()
            
            # Basic check
            assert kademlia.local_id is not None
            
            await kademlia.stop()
            
        except ImportError:
            pytest.skip("KademliaNode not available")

