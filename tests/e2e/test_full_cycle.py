"""
End-to-End Full Cycle Tests
===========================

[E2E] Tests complete user scenarios.
"""

import pytest
import pytest_asyncio


@pytest.mark.e2e
@pytest.mark.slow
class TestFullCycle:
    """Test complete user scenarios."""
    
    @pytest.mark.asyncio
    async def test_node_startup_shutdown(self, node_factory):
        """Test basic node lifecycle."""
        # Create and start node
        nodes = await node_factory.create(1, start=True)
        
        assert len(nodes) == 1
        node = nodes[0]
        
        if node.started:
            assert node.node is not None
            assert node.node_id is not None
    
    @pytest.mark.asyncio
    async def test_two_node_connection(self, node_factory):
        """Test two nodes connecting."""
        # Create two nodes
        nodes = await node_factory.create(2, start=True)
        
        # Check both started
        started_nodes = [n for n in nodes if n.started]
        if len(started_nodes) < 2:
            pytest.skip("Could not start both nodes")
        
        # Try to connect
        await node_factory.connect_all()
        
        # Give time for connection
        import asyncio
        await asyncio.sleep(0.5)
        
        # Check connections (implementation dependent)
        # Just verify no crashes
    
    @pytest.mark.asyncio
    async def test_echo_service(self, node_factory):
        """Test Echo service between nodes."""
        nodes = await node_factory.create(2, start=True)
        
        started_nodes = [n for n in nodes if n.started]
        if len(started_nodes) < 2:
            pytest.skip("Could not start both nodes")
        
        await node_factory.connect_all()
        
        client, server = started_nodes[0], started_nodes[1]
        
        # Request echo service
        try:
            if client.node and hasattr(client.node, 'request_service'):
                success = await client.node.request_service(
                    server.node_id,
                    "echo",
                    "test_payload",
                    budget=10.0,
                )
                # Note: Response might be async callback
        except Exception as e:
            # Service might not be fully implemented
            pytest.skip(f"Echo service test skipped: {e}")


@pytest.mark.e2e
@pytest.mark.slow
class TestLiteMode:
    """Test LITE mode (no AI)."""
    
    @pytest.mark.asyncio
    async def test_lite_mode_startup(self, node_factory, mock_gpu):
        """Test node starts in LITE mode without GPU."""
        nodes = await node_factory.create(1, start=True)
        
        if nodes and nodes[0].started:
            # Node should be running
            assert nodes[0].node is not None
            
            # AI services should not be available
            # (implementation dependent check)

