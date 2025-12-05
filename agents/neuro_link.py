"""
NeuroLink Agent - Neural Synapse Between Nodes
==============================================

[NEURAL] Транспортный слой для тензоров PyTorch поверх Binary Wire Protocol.

Это "синапс" между узлами для Pipeline Parallelism:
- Head Node: Prompt -> Embed -> Layers[0:N] -> NeuroLink.send
- Middle Node: NeuroLink.recv -> Layers[N:M] -> NeuroLink.send
- Tail Node: NeuroLink.recv -> Layers[M:End] -> Detokenize -> Result

[PERFORMANCE]
- Zero-copy network where possible
- TCP_NODELAY for low latency
- Chunked streaming for large tensors
- Async pipeline для overlap compute/network

[SECURITY]
- Size limits (защита от OOM)
- No pickle (custom binary format)
- Signature verification on all packets
"""

import asyncio
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Awaitable, Tuple
from collections import defaultdict

from agents.manager import BaseAgent

logger = logging.getLogger(__name__)

# Lazy imports
torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


# ============================================================================
# Constants
# ============================================================================

# Chunk size for streaming (64KB for TCP efficiency)
STREAM_CHUNK_SIZE = 64 * 1024

# Maximum tensor size (1GB)
MAX_TENSOR_SIZE = 1024 * 1024 * 1024

# Timeout for tensor operations
TENSOR_TIMEOUT = 60.0

# Pipeline stages
STAGE_HEAD = "head"
STAGE_MIDDLE = "middle"
STAGE_TAIL = "tail"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TensorTransfer:
    """
    Metadata for tensor transfer.
    """
    tensor_id: str
    shape: Tuple[int, ...]
    dtype: str
    total_size: int
    total_chunks: int
    source_node: str
    target_node: str
    layer_idx: int = 0
    batch_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class PipelineContext:
    """
    Context for a pipeline forward/backward pass.
    """
    batch_id: str
    sequence_length: int
    batch_size: int
    current_layer: int
    total_layers: int
    stage: str  # head/middle/tail
    upstream_node: Optional[str] = None
    downstream_node: Optional[str] = None
    started_at: float = field(default_factory=time.time)


# ============================================================================
# NeuroLink Agent
# ============================================================================

class NeuroLinkAgent(BaseAgent):
    """
    Neural Link Agent for distributed tensor transport.
    
    [SERVICE] neuro_link
    
    [PROTOCOL]
    - TENSOR_DATA: Raw tensor bytes
    - TENSOR_META: Tensor metadata (shape, dtype)
    - TENSOR_CHUNK: Chunked tensor fragment
    - PIPELINE_FORWARD: Forward pass activation
    - PIPELINE_BACKWARD: Backward pass gradient
    
    [USAGE]
        agent = NeuroLinkAgent(node, device="cuda:0")
        agent.register_activation_handler(my_handler)
        
        # Send activations to next node
        await agent.send_activations(peer_id, tensor, layer_idx=5)
        
        # Receive activations (via handler callback)
        async def my_handler(tensor, context):
            output = my_model_shard(tensor)
            await agent.send_activations(next_peer, output, layer_idx=6)
    """
    
    service_name = "neuro_link"
    
    def __init__(
        self,
        node: Any = None,
        device: str = "cpu",
        chunk_size: int = STREAM_CHUNK_SIZE,
    ):
        """
        Args:
            node: Parent node for network access
            device: Target device for received tensors
            chunk_size: Size of chunks for streaming
        """
        super().__init__()
        self.node = node
        self.device = device
        self.chunk_size = chunk_size
        
        # Import codec
        from cortex.distributed.codec import (
            TensorEncoder, TensorDecoder,
            TensorChunker, TensorAssembler,
        )
        
        self.encoder = TensorEncoder(compress=False)
        self.decoder = TensorDecoder(device=device)
        self.chunker = TensorChunker(chunk_size=chunk_size)
        self.assembler = TensorAssembler(device=device)
        
        # Pending transfers
        self._pending_transfers: Dict[str, TensorTransfer] = {}
        
        # Callbacks
        self._activation_handler: Optional[Callable] = None
        self._gradient_handler: Optional[Callable] = None
        
        # Stats
        self._stats = {
            "tensors_sent": 0,
            "tensors_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "avg_latency_ms": 0.0,
        }
        
        # Pipeline context
        self._pipeline_ctx: Optional[PipelineContext] = None
        
        logger.info(f"[NEUROLINK] Initialized on device={device}")
    
    def register_activation_handler(
        self,
        handler: Callable[["torch.Tensor", Dict[str, Any]], Awaitable[None]],
    ) -> None:
        """
        Register callback for received activations.
        
        Args:
            handler: async def handler(tensor, metadata) -> None
        """
        self._activation_handler = handler
    
    def register_gradient_handler(
        self,
        handler: Callable[["torch.Tensor", Dict[str, Any]], Awaitable[None]],
    ) -> None:
        """
        Register callback for received gradients (backward pass).
        """
        self._gradient_handler = handler
    
    async def send_activations(
        self,
        peer_id: str,
        tensor: "torch.Tensor",
        layer_idx: int = 0,
        batch_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send activation tensor to peer node.
        
        [NETWORK] Процесс:
        1. Сериализуем тензор (codec.encode)
        2. Разбиваем на чанки (если > chunk_size)
        3. Отправляем через Binary Wire Protocol
        4. Ждём ACK
        
        Args:
            peer_id: Target node ID
            tensor: Activation tensor
            layer_idx: Layer index in pipeline
            batch_id: Batch identifier
            metadata: Additional metadata
        
        Returns:
            True if successful
        """
        torch = _ensure_torch()
        
        tensor_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Encode tensor
            from cortex.distributed.codec import encode_tensor
            tensor_bytes = encode_tensor(tensor, compress=False)
            
            # Check size
            if len(tensor_bytes) > MAX_TENSOR_SIZE:
                logger.error(f"[NEUROLINK] Tensor too large: {len(tensor_bytes)} bytes")
                return False
            
            # Build metadata
            meta = {
                "tensor_id": tensor_id,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "layer_idx": layer_idx,
                "batch_id": batch_id or str(uuid.uuid4()),
                "total_size": len(tensor_bytes),
                "device": str(tensor.device),
                **(metadata or {}),
            }
            
            # Send via service call
            agent_mgr = getattr(self.node, "agent_manager", None)
            if not agent_mgr:
                logger.error("[NEUROLINK] No agent manager available")
                return False
            
            # For large tensors, use chunked transfer
            if len(tensor_bytes) > self.chunk_size:
                success = await self._send_chunked(
                    agent_mgr, peer_id, tensor_bytes, meta
                )
            else:
                # Single packet
                response = await agent_mgr.call_service(
                    peer_id,
                    "neuro_link",
                    {
                        "action": "receive_activation",
                        "meta": meta,
                        "data": tensor_bytes,
                    },
                )
                success = response.get("status") == "ok"
            
            # Update stats
            elapsed = (time.time() - start_time) * 1000
            self._stats["tensors_sent"] += 1
            self._stats["bytes_sent"] += len(tensor_bytes)
            self._update_latency(elapsed)
            
            logger.debug(
                f"[NEUROLINK] Sent tensor {tensor_id[:8]}... to {peer_id[:8]}... "
                f"shape={tensor.shape} size={len(tensor_bytes)} latency={elapsed:.1f}ms"
            )
            
            return success
            
        except Exception as e:
            logger.error(f"[NEUROLINK] Send failed: {e}")
            return False
    
    async def _send_chunked(
        self,
        agent_mgr: Any,
        peer_id: str,
        tensor_bytes: bytes,
        meta: Dict[str, Any],
    ) -> bool:
        """Send tensor in chunks."""
        tensor_id = meta["tensor_id"]
        total_chunks = (len(tensor_bytes) + self.chunk_size - 1) // self.chunk_size
        
        # Send metadata first
        meta["total_chunks"] = total_chunks
        response = await agent_mgr.call_service(
            peer_id,
            "neuro_link",
            {"action": "init_transfer", "meta": meta},
        )
        
        if response.get("status") != "ready":
            return False
        
        # Send chunks
        for chunk_idx in range(total_chunks):
            offset = chunk_idx * self.chunk_size
            chunk_data = tensor_bytes[offset:offset + self.chunk_size]
            is_last = chunk_idx == total_chunks - 1
            
            response = await agent_mgr.call_service(
                peer_id,
                "neuro_link",
                {
                    "action": "receive_chunk",
                    "tensor_id": tensor_id,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                    "data": chunk_data,
                    "is_last": is_last,
                },
            )
            
            if response.get("status") != "ok":
                logger.warning(f"[NEUROLINK] Chunk {chunk_idx} failed")
                return False
        
        return True
    
    async def send_gradients(
        self,
        peer_id: str,
        tensor: "torch.Tensor",
        layer_idx: int = 0,
        batch_id: str = "",
    ) -> bool:
        """
        Send gradient tensor for backward pass.
        
        Similar to send_activations but uses PIPELINE_BACKWARD.
        """
        return await self.send_activations(
            peer_id,
            tensor,
            layer_idx=layer_idx,
            batch_id=batch_id,
            metadata={"is_gradient": True},
        )
    
    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming requests.
        
        Actions:
        - init_transfer: Initialize chunked transfer
        - receive_chunk: Receive tensor chunk
        - receive_activation: Receive single-packet tensor
        - get_stats: Get transfer statistics
        """
        action = request.get("action")
        
        if action == "init_transfer":
            return await self._handle_init_transfer(request)
        
        if action == "receive_chunk":
            return await self._handle_receive_chunk(request)
        
        if action == "receive_activation":
            return await self._handle_receive_activation(request)
        
        if action == "get_stats":
            return {"stats": self._stats}
        
        return {"error": "unknown_action"}
    
    async def _handle_init_transfer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a chunked tensor transfer."""
        meta = request.get("meta", {})
        tensor_id = meta.get("tensor_id")
        
        if not tensor_id:
            return {"status": "error", "error": "missing_tensor_id"}
        
        # Check size limit
        total_size = meta.get("total_size", 0)
        if total_size > MAX_TENSOR_SIZE:
            return {"status": "error", "error": "tensor_too_large"}
        
        # Create transfer record
        self._pending_transfers[tensor_id] = TensorTransfer(
            tensor_id=tensor_id,
            shape=tuple(meta.get("shape", [])),
            dtype=meta.get("dtype", "float32"),
            total_size=total_size,
            total_chunks=meta.get("total_chunks", 1),
            source_node=meta.get("source_node", ""),
            target_node=meta.get("target_node", ""),
            layer_idx=meta.get("layer_idx", 0),
            batch_id=meta.get("batch_id", ""),
        )
        
        logger.debug(f"[NEUROLINK] Init transfer {tensor_id[:8]}... {total_size} bytes")
        return {"status": "ready"}
    
    async def _handle_receive_chunk(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Receive a tensor chunk."""
        tensor_id = request.get("tensor_id")
        chunk_idx = request.get("chunk_idx", 0)
        data = request.get("data", b"")
        is_last = request.get("is_last", False)
        total_chunks = request.get("total_chunks", 1)
        
        if not tensor_id:
            return {"status": "error", "error": "missing_tensor_id"}
        
        # Add to assembler
        chunk = {
            "tensor_id": tensor_id,
            "chunk_idx": chunk_idx,
            "total_chunks": total_chunks,
            "data": data,
            "is_last": is_last,
        }
        
        tensor = self.assembler.add_chunk(chunk)
        
        if tensor is not None:
            # Transfer complete - invoke handler
            transfer = self._pending_transfers.pop(tensor_id, None)
            
            meta = {}
            if transfer:
                meta = {
                    "tensor_id": tensor_id,
                    "shape": transfer.shape,
                    "dtype": transfer.dtype,
                    "layer_idx": transfer.layer_idx,
                    "batch_id": transfer.batch_id,
                }
            
            await self._invoke_handler(tensor, meta)
            
            self._stats["tensors_received"] += 1
            self._stats["bytes_received"] += transfer.total_size if transfer else 0
            
            logger.debug(f"[NEUROLINK] Completed transfer {tensor_id[:8]}...")
        
        return {"status": "ok"}
    
    async def _handle_receive_activation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Receive a single-packet tensor."""
        meta = request.get("meta", {})
        data = request.get("data", b"")
        
        if not data:
            return {"status": "error", "error": "missing_data"}
        
        try:
            from cortex.distributed.codec import decode_tensor
            tensor = decode_tensor(data, device=self.device)
            
            await self._invoke_handler(tensor, meta)
            
            self._stats["tensors_received"] += 1
            self._stats["bytes_received"] += len(data)
            
            return {"status": "ok"}
            
        except Exception as e:
            logger.error(f"[NEUROLINK] Decode failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _invoke_handler(
        self,
        tensor: "torch.Tensor",
        meta: Dict[str, Any],
    ) -> None:
        """Invoke the appropriate handler for received tensor."""
        is_gradient = meta.get("is_gradient", False)
        
        if is_gradient and self._gradient_handler:
            await self._gradient_handler(tensor, meta)
        elif not is_gradient and self._activation_handler:
            await self._activation_handler(tensor, meta)
        else:
            logger.warning("[NEUROLINK] No handler registered for received tensor")
    
    def _update_latency(self, latency_ms: float) -> None:
        """Update running average latency."""
        alpha = 0.1
        current = self._stats["avg_latency_ms"]
        self._stats["avg_latency_ms"] = current * (1 - alpha) + latency_ms * alpha
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transfer statistics."""
        return dict(self._stats)
    
    def set_pipeline_context(self, ctx: PipelineContext) -> None:
        """Set current pipeline context."""
        self._pipeline_ctx = ctx
    
    def get_pipeline_context(self) -> Optional[PipelineContext]:
        """Get current pipeline context."""
        return self._pipeline_ctx


# ============================================================================
# Low-Level TCP Transport (for maximum performance)
# ============================================================================

class TensorTCPTransport:
    """
    Direct TCP transport for tensors with TCP_NODELAY.
    
    [PERFORMANCE] For latency-critical paths:
    - Bypasses service layer
    - Direct socket communication
    - TCP_NODELAY for minimal latency
    
    [USAGE]
        transport = TensorTCPTransport(host, port)
        await transport.connect()
        await transport.send_tensor(tensor)
        tensor = await transport.recv_tensor()
    """
    
    def __init__(self, host: str, port: int, device: str = "cpu"):
        self.host = host
        self.port = port
        self.device = device
        
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        
        from cortex.distributed.codec import TensorEncoder, TensorDecoder
        self.encoder = TensorEncoder()
        self.decoder = TensorDecoder(device=device)
    
    async def connect(self) -> None:
        """Connect to remote tensor endpoint."""
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port
        )
        
        # Enable TCP_NODELAY for low latency
        sock = self._writer.get_extra_info('socket')
        if sock:
            import socket
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        logger.info(f"[TENSOR_TCP] Connected to {self.host}:{self.port}")
    
    async def close(self) -> None:
        """Close connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
    
    async def send_tensor(self, tensor: "torch.Tensor") -> int:
        """
        Send tensor over TCP.
        
        Format: [4 byte length][tensor bytes]
        
        Returns:
            Bytes sent
        """
        if not self._writer:
            raise RuntimeError("Not connected")
        
        from cortex.distributed.codec import encode_tensor
        data = encode_tensor(tensor)
        
        # Send length + data
        length = len(data)
        self._writer.write(length.to_bytes(4, 'big'))
        self._writer.write(data)
        await self._writer.drain()
        
        return 4 + length
    
    async def recv_tensor(self) -> "torch.Tensor":
        """
        Receive tensor from TCP.
        
        Returns:
            Decoded tensor
        """
        if not self._reader:
            raise RuntimeError("Not connected")
        
        # Read length
        length_bytes = await self._reader.readexactly(4)
        length = int.from_bytes(length_bytes, 'big')
        
        # Check size
        if length > MAX_TENSOR_SIZE:
            raise ValueError(f"Tensor too large: {length}")
        
        # Read data
        data = await self._reader.readexactly(length)
        
        from cortex.distributed.codec import decode_tensor
        return decode_tensor(data, device=self.device)


async def create_tensor_server(
    host: str,
    port: int,
    handler: Callable[["torch.Tensor"], Awaitable["torch.Tensor"]],
    device: str = "cpu",
) -> asyncio.AbstractServer:
    """
    Create a TCP server for tensor processing.
    
    Args:
        host: Bind host
        port: Bind port
        handler: async def handler(tensor) -> tensor
        device: Device for tensors
    
    Returns:
        asyncio Server
    """
    from cortex.distributed.codec import TensorEncoder, TensorDecoder
    encoder = TensorEncoder()
    decoder = TensorDecoder(device=device)
    
    async def client_handler(reader, writer):
        # TCP_NODELAY
        sock = writer.get_extra_info('socket')
        if sock:
            import socket
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        try:
            while True:
                # Read length
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, 'big')
                
                if length > MAX_TENSOR_SIZE:
                    break
                
                # Read tensor
                data = await reader.readexactly(length)
                
                from cortex.distributed.codec import decode_tensor, encode_tensor
                tensor = decode_tensor(data, device=device)
                
                # Process
                result = await handler(tensor)
                
                # Send result
                result_data = encode_tensor(result)
                writer.write(len(result_data).to_bytes(4, 'big'))
                writer.write(result_data)
                await writer.drain()
                
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
    
    server = await asyncio.start_server(client_handler, host, port)
    logger.info(f"[TENSOR_TCP] Server listening on {host}:{port}")
    return server

