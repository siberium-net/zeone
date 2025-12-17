"""
Cortex Visualization WebSocket Endpoint
=======================================

Provides real-time graph data for the NeuralVis frontend.

[ENDPOINTS]
- GET /vis → Static HTML visualization
- WS /ws/vis → WebSocket for real-time updates

[MESSAGES]
- get_graph → Returns full graph data
- pulse → Data transfer event
- node_update → Node state change
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class VisNode:
    """Node for visualization."""
    id: str
    role: str = "unknown"
    val: int = 10
    trustScore: float = 0.5
    currentTask: str = "Idle"
    topic: str = ""
    x: float = 0
    y: float = 0
    z: float = 0


@dataclass
class VisLink:
    """Link for visualization."""
    source: str
    target: str
    messageCount: int = 0
    balance: float = 0


class CortexVisualizer:
    """
    Manages visualization state and WebSocket connections.
    
    [USAGE]
    ```python
    vis = CortexVisualizer(cortex=cortex, node=node)
    await vis.start()
    
    # In NiceGUI or FastAPI:
    @app.websocket('/ws/vis')
    async def ws_vis(websocket):
        await vis.handle_websocket(websocket)
    ```
    """
    
    def __init__(
        self,
        cortex=None,
        node=None,
        ledger=None,
    ):
        self.cortex = cortex
        self.node = node
        self.ledger = ledger
        
        self._clients: Set[Any] = set()
        self._nodes: Dict[str, VisNode] = {}
        self._links: Dict[str, VisLink] = {}
        self._running = False
        self._update_task = None
    
    async def start(self) -> None:
        """Start the visualizer."""
        self._running = True
        
        # Build initial graph from node state
        await self._build_graph()
        
        # Start update loop
        self._update_task = asyncio.create_task(self._update_loop())
        
        logger.info("[VIS] Cortex Visualizer started")
    
    async def stop(self) -> None:
        """Stop the visualizer."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        # Close all WebSocket connections
        for client in list(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        
        logger.info("[VIS] Cortex Visualizer stopped")
    
    async def _build_graph(self) -> None:
        """Build graph from current network state."""
        self._nodes.clear()
        self._links.clear()
        
        # Add "me" node
        node_id = "me"
        if self.node:
            node_id = getattr(self.node, 'node_id', 'me')[:16]
        
        self._nodes["me"] = VisNode(
            id="me",
            role="me",
            val=25,
            trustScore=1.0,
            currentTask="Coordinating",
        )
        
        # Add peer nodes
        if self.node:
            peer_manager = getattr(self.node, 'peer_manager', None)
            if peer_manager:
                for peer in peer_manager.get_active_peers():
                    peer_short_id = peer.node_id[:16]
                    
                    # Determine role based on services or trust
                    role = "unknown"
                    trust = 0.5
                    
                    if self.ledger:
                        try:
                            trust = await self.ledger.get_trust_score(peer.node_id)
                        except RuntimeError as e:
                            # Ledger may be bound to a different event loop; skip in visualizer.
                            logger.debug(f"[VIS] Ledger trust fetch skipped: {e}")
                        except Exception as e:
                            logger.debug(f"[VIS] Ledger trust fetch error: {e}")
                    
                    # Add node
                    self._nodes[peer_short_id] = VisNode(
                        id=peer_short_id,
                        role=role,
                        val=int(trust * 20) + 5,
                        trustScore=trust,
                        currentTask="Connected",
                    )
                    
                    # Add link to me
                    link_key = f"me-{peer_short_id}"
                    balance = 0
                    if self.ledger:
                        try:
                            balance = await self.ledger.get_balance(peer.node_id)
                        except RuntimeError as e:
                            logger.debug(f"[VIS] Ledger balance fetch skipped: {e}")
                        except Exception as e:
                            logger.debug(f"[VIS] Ledger balance fetch error: {e}")
                    
                    self._links[link_key] = VisLink(
                        source="me",
                        target=peer_short_id,
                        balance=balance,
                    )
        
        # Add Cortex investigation nodes
        if self.cortex:
            automata = getattr(self.cortex, 'automata', None)
            if automata:
                for inv in automata.get_recent_investigations(5):
                    inv_id = f"inv_{inv.investigation_id[:8]}"
                    
                    status_role = {
                        "scouting": "scout",
                        "analyzing": "analyst",
                        "storing": "librarian",
                        "completed": "librarian",
                    }.get(inv.status.value, "unknown")
                    
                    self._nodes[inv_id] = VisNode(
                        id=inv_id,
                        role=status_role,
                        val=12,
                        trustScore=0.8,
                        currentTask=inv.topic[:30],
                        topic=inv.topic,
                    )
                    
                    # Link to me
                    self._links[f"me-{inv_id}"] = VisLink(
                        source="me",
                        target=inv_id,
                    )
    
    async def _update_loop(self) -> None:
        """Periodically update graph and broadcast changes."""
        while self._running:
            try:
                # Rebuild graph
                await self._build_graph()
                
                # Broadcast to all clients
                await self._broadcast({
                    "type": "graph",
                    "nodes": [asdict(n) for n in self._nodes.values()],
                    "links": [asdict(l) for l in self._links.values()],
                })
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VIS] Update error: {e}")
                await asyncio.sleep(1)
    
    async def handle_websocket(self, websocket) -> None:
        """Handle WebSocket connection."""
        self._clients.add(websocket)
        logger.debug(f"[VIS] Client connected ({len(self._clients)} total)")
        
        try:
            # Send initial graph
            await websocket.send_json({
                "type": "graph",
                "nodes": [asdict(n) for n in self._nodes.values()],
                "links": [asdict(l) for l in self._links.values()],
            })
            
            # Handle messages
            async for message in websocket.iter_json():
                await self._handle_client_message(websocket, message)
                
        except Exception as e:
            logger.debug(f"[VIS] Client error: {e}")
        finally:
            self._clients.discard(websocket)
            logger.debug(f"[VIS] Client disconnected ({len(self._clients)} total)")
    
    async def _handle_client_message(self, websocket, message: dict) -> None:
        """Handle message from client."""
        msg_type = message.get("type", "")
        
        if msg_type == "get_graph":
            await websocket.send_json({
                "type": "graph",
                "nodes": [asdict(n) for n in self._nodes.values()],
                "links": [asdict(l) for l in self._links.values()],
            })
        
        elif msg_type == "focus_node":
            node_id = message.get("nodeId")
            if node_id and node_id in self._nodes:
                await websocket.send_json({
                    "type": "node_update",
                    "node": asdict(self._nodes[node_id]),
                })
    
    async def _broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        if not self._clients:
            return
        
        for client in list(self._clients):
            try:
                await client.send_json(message)
            except Exception:
                self._clients.discard(client)
    
    # =========================================================================
    # PUBLIC API - Call these to trigger visual events
    # =========================================================================
    
    async def emit_pulse(
        self,
        from_id: str,
        to_id: str,
        value: int = 1,
    ) -> None:
        """Emit a data transfer pulse between nodes."""
        await self._broadcast({
            "type": "pulse",
            "from": from_id,
            "to": to_id,
            "value": value,
        })
    
    async def add_node(
        self,
        node_id: str,
        role: str = "unknown",
        **kwargs,
    ) -> None:
        """Add a new node to visualization."""
        node = VisNode(id=node_id, role=role, **kwargs)
        self._nodes[node_id] = node
        
        await self._broadcast({
            "type": "node_add",
            "node": asdict(node),
        })
    
    async def remove_node(self, node_id: str) -> None:
        """Remove a node from visualization."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            
            # Remove associated links
            self._links = {
                k: v for k, v in self._links.items()
                if v.source != node_id and v.target != node_id
            }
            
            await self._broadcast({
                "type": "node_remove",
                "nodeId": node_id,
            })
    
    async def update_node(
        self,
        node_id: str,
        **kwargs,
    ) -> None:
        """Update node properties."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            for k, v in kwargs.items():
                if hasattr(node, k):
                    setattr(node, k, v)
            
            await self._broadcast({
                "type": "node_update",
                "node": asdict(node),
            })
    
    async def set_node_task(
        self,
        node_id: str,
        task: str,
    ) -> None:
        """Update node's current task."""
        await self.update_node(node_id, currentTask=task)
    
    def get_graph_data(self) -> dict:
        """Get current graph data as dict."""
        return {
            "nodes": [asdict(n) for n in self._nodes.values()],
            "links": [asdict(l) for l in self._links.values()],
        }


# =========================================================================
# DEMO DATA GENERATOR
# =========================================================================

def generate_demo_graph() -> dict:
    """Generate demo graph data for testing."""
    nodes = [
        {"id": "me", "role": "me", "val": 25, "trustScore": 1.0, "currentTask": "Coordinating"},
        {"id": "scout_1", "role": "scout", "val": 15, "trustScore": 0.85, "currentTask": "Searching: quantum computing"},
        {"id": "scout_2", "role": "scout", "val": 12, "trustScore": 0.72, "currentTask": "Idle"},
        {"id": "analyst_1", "role": "analyst", "val": 18, "trustScore": 0.91, "currentTask": "Analyzing text"},
        {"id": "analyst_2", "role": "analyst", "val": 16, "trustScore": 0.88, "currentTask": "Council deliberation"},
        {"id": "analyst_3", "role": "analyst", "val": 14, "trustScore": 0.79, "currentTask": "Idle"},
        {"id": "librarian_1", "role": "librarian", "val": 20, "trustScore": 0.95, "currentTask": "Indexing: AI safety"},
        {"id": "librarian_2", "role": "librarian", "val": 17, "trustScore": 0.82, "currentTask": "DHT sync"},
        {"id": "peer_1", "role": "unknown", "val": 10, "trustScore": 0.5, "currentTask": "Unknown"},
        {"id": "peer_2", "role": "unknown", "val": 8, "trustScore": 0.45, "currentTask": "Unknown"},
    ]
    
    links = [
        {"source": "me", "target": "scout_1"},
        {"source": "me", "target": "scout_2"},
        {"source": "me", "target": "analyst_1"},
        {"source": "me", "target": "librarian_1"},
        {"source": "scout_1", "target": "analyst_1"},
        {"source": "scout_1", "target": "analyst_2"},
        {"source": "scout_2", "target": "analyst_3"},
        {"source": "analyst_1", "target": "librarian_1"},
        {"source": "analyst_2", "target": "librarian_1"},
        {"source": "analyst_3", "target": "librarian_2"},
        {"source": "librarian_1", "target": "librarian_2"},
        {"source": "peer_1", "target": "scout_1"},
        {"source": "peer_2", "target": "analyst_2"},
    ]
    
    return {"nodes": nodes, "links": links}
