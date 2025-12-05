"""
Tensor Serialization Codec
==========================

[NEURAL] Эффективная сериализация PyTorch тензоров для передачи по сети.

[SECURITY] НЕ используем pickle - это небезопасно для сетевых данных.
Используем кастомный бинарный формат с явной структурой.

Формат TensorFrame:
==================

Header (Variable length):
┌───────────┬──────┬────────────────────────────────────────┐
│ Field     │ Size │ Description                            │
├───────────┼──────┼────────────────────────────────────────┤
│ Magic     │ 2    │ b'ZT' (0x5A54) - ZEone Tensor         │
│ Version   │ 1    │ Codec version (1)                      │
│ DType     │ 1    │ PyTorch dtype enum                     │
│ Flags     │ 1    │ 0x01=Compressed, 0x02=Contiguous      │
│ NDim      │ 1    │ Number of dimensions (max 8)           │
│ Shape     │ N*8  │ Shape as int64 array                   │
│ DataLen   │ 8    │ Compressed data length                 │
├───────────┼──────┼────────────────────────────────────────┤
│ Data      │ var  │ Raw tensor bytes (or LZ4 compressed)   │
└───────────┴──────┴────────────────────────────────────────┘

[PERFORMANCE]
- Zero-copy where possible (numpy/torch interop)
- Optional LZ4 compression for sparse tensors
- Chunk-based streaming for large tensors

[LIMITS]
- MAX_TENSOR_SIZE: 1GB (защита от OOM атак)
- MAX_NDIM: 8 dimensions
"""

import io
import struct
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from enum import IntEnum

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
torch = None
np = None
lz4_frame = None


def _ensure_torch():
    """Lazy import torch."""
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


def _ensure_numpy():
    """Lazy import numpy."""
    global np
    if np is None:
        import numpy as _np
        np = _np
    return np


def _ensure_lz4():
    """Lazy import lz4."""
    global lz4_frame
    if lz4_frame is None:
        try:
            import lz4.frame as _lz4
            lz4_frame = _lz4
        except ImportError:
            lz4_frame = None
    return lz4_frame


# ============================================================================
# Constants
# ============================================================================

TENSOR_MAGIC = b'ZT'            # ZEone Tensor
CODEC_VERSION = 1               # Current version
MAX_TENSOR_SIZE = 1024 * 1024 * 1024  # 1 GB max
MAX_NDIM = 8                    # Maximum dimensions
CHUNK_SIZE = 64 * 1024          # 64 KB chunks for streaming

# Header format: Magic(2) + Version(1) + DType(1) + Flags(1) + NDim(1) = 6 bytes
# Then: Shape(NDim * 8) + DataLen(8)
HEADER_BASE_FORMAT = '>2sBBBB'
HEADER_BASE_SIZE = 6


class TensorDType(IntEnum):
    """
    PyTorch dtype mapping.
    
    [WIRE] Stable IDs для сериализации.
    """
    FLOAT16 = 0     # torch.float16 / torch.half
    FLOAT32 = 1     # torch.float32 / torch.float
    FLOAT64 = 2     # torch.float64 / torch.double
    BFLOAT16 = 3    # torch.bfloat16
    INT8 = 4        # torch.int8
    INT16 = 5       # torch.int16
    INT32 = 6       # torch.int32
    INT64 = 7       # torch.int64
    UINT8 = 8       # torch.uint8
    BOOL = 9        # torch.bool
    COMPLEX64 = 10  # torch.complex64
    COMPLEX128 = 11 # torch.complex128


# Mapping torch dtype to TensorDType
_DTYPE_TO_ID: Dict[Any, TensorDType] = {}
_ID_TO_DTYPE: Dict[TensorDType, Any] = {}


def _init_dtype_mapping():
    """Initialize dtype mappings (lazy, after torch import)."""
    global _DTYPE_TO_ID, _ID_TO_DTYPE
    if _DTYPE_TO_ID:
        return
    
    torch = _ensure_torch()
    
    mapping = {
        torch.float16: TensorDType.FLOAT16,
        torch.float32: TensorDType.FLOAT32,
        torch.float64: TensorDType.FLOAT64,
        torch.bfloat16: TensorDType.BFLOAT16,
        torch.int8: TensorDType.INT8,
        torch.int16: TensorDType.INT16,
        torch.int32: TensorDType.INT32,
        torch.int64: TensorDType.INT64,
        torch.uint8: TensorDType.UINT8,
        torch.bool: TensorDType.BOOL,
        torch.complex64: TensorDType.COMPLEX64,
        torch.complex128: TensorDType.COMPLEX128,
    }
    
    _DTYPE_TO_ID = mapping
    _ID_TO_DTYPE = {v: k for k, v in mapping.items()}


class TensorFlags(IntEnum):
    """Flags for tensor encoding."""
    NONE = 0x00
    COMPRESSED = 0x01       # LZ4 compressed
    CONTIGUOUS = 0x02       # Memory contiguous
    ON_CUDA = 0x04          # Originally on CUDA (for info)
    REQUIRES_GRAD = 0x08    # Requires gradient


class CodecError(Exception):
    """Tensor codec error."""
    pass


class TensorTooLargeError(CodecError):
    """Tensor exceeds maximum size."""
    pass


# ============================================================================
# TensorFrame - Serialized tensor container
# ============================================================================

@dataclass
class TensorFrame:
    """
    Serialized tensor frame.
    
    [WIRE] Contains all information to reconstruct the tensor.
    """
    dtype_id: TensorDType
    shape: Tuple[int, ...]
    data: bytes
    flags: int = TensorFlags.NONE
    
    @property
    def is_compressed(self) -> bool:
        return bool(self.flags & TensorFlags.COMPRESSED)
    
    @property
    def ndim(self) -> int:
        return len(self.shape)
    
    @property
    def numel(self) -> int:
        """Number of elements."""
        result = 1
        for s in self.shape:
            result *= s
        return result
    
    def total_size(self) -> int:
        """Total serialized size in bytes."""
        header_size = HEADER_BASE_SIZE + len(self.shape) * 8 + 8
        return header_size + len(self.data)


# ============================================================================
# Encoder
# ============================================================================

class TensorEncoder:
    """
    Encodes PyTorch tensors to bytes.
    
    [PERFORMANCE]
    - Uses numpy for zero-copy conversion where possible
    - Optional LZ4 compression
    - Streaming support for large tensors
    
    [SECURITY]
    - Size limits enforced
    - No pickle/arbitrary code execution
    """
    
    def __init__(
        self,
        compress: bool = False,
        compression_threshold: int = 1024 * 1024,  # 1MB
    ):
        """
        Args:
            compress: Enable LZ4 compression
            compression_threshold: Only compress if data > threshold
        """
        self.compress = compress
        self.compression_threshold = compression_threshold
        _init_dtype_mapping()
    
    def encode(self, tensor: "torch.Tensor") -> bytes:
        """
        Encode tensor to bytes.
        
        [WIRE] Format:
            Header + Data
        
        Args:
            tensor: PyTorch tensor
        
        Returns:
            Serialized bytes
        
        Raises:
            TensorTooLargeError: If tensor > MAX_TENSOR_SIZE
            CodecError: If encoding fails
        """
        torch = _ensure_torch()
        
        # Ensure contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Get dtype ID
        dtype_id = _DTYPE_TO_ID.get(tensor.dtype)
        if dtype_id is None:
            raise CodecError(f"Unsupported dtype: {tensor.dtype}")
        
        # Check dimensions
        if tensor.ndim > MAX_NDIM:
            raise CodecError(f"Too many dimensions: {tensor.ndim} > {MAX_NDIM}")
        
        # Get raw bytes via numpy (zero-copy for CPU tensors)
        if tensor.is_cuda:
            tensor_cpu = tensor.cpu()
        else:
            tensor_cpu = tensor
        
        # Convert to bytes
        np_array = tensor_cpu.numpy()
        raw_data = np_array.tobytes()
        
        # Check size
        if len(raw_data) > MAX_TENSOR_SIZE:
            raise TensorTooLargeError(
                f"Tensor too large: {len(raw_data)} > {MAX_TENSOR_SIZE}"
            )
        
        # Compression
        flags = TensorFlags.CONTIGUOUS
        if tensor.is_cuda:
            flags |= TensorFlags.ON_CUDA
        if tensor.requires_grad:
            flags |= TensorFlags.REQUIRES_GRAD
        
        data = raw_data
        if self.compress and len(raw_data) > self.compression_threshold:
            lz4 = _ensure_lz4()
            if lz4:
                compressed = lz4.compress(raw_data)
                # Only use if actually smaller
                if len(compressed) < len(raw_data) * 0.9:
                    data = compressed
                    flags |= TensorFlags.COMPRESSED
        
        # Build header
        shape = tuple(tensor.shape)
        header = self._build_header(dtype_id, shape, len(data), flags)
        
        return header + data
    
    def _build_header(
        self,
        dtype_id: TensorDType,
        shape: Tuple[int, ...],
        data_len: int,
        flags: int,
    ) -> bytes:
        """Build tensor frame header."""
        ndim = len(shape)
        
        # Base header
        header = struct.pack(
            HEADER_BASE_FORMAT,
            TENSOR_MAGIC,
            CODEC_VERSION,
            int(dtype_id),
            flags,
            ndim,
        )
        
        # Shape (as int64 array)
        for dim in shape:
            header += struct.pack('>q', dim)
        
        # Data length
        header += struct.pack('>Q', data_len)
        
        return header
    
    def encode_to_frame(self, tensor: "torch.Tensor") -> TensorFrame:
        """
        Encode tensor to TensorFrame (for inspection/manipulation).
        """
        torch = _ensure_torch()
        
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        dtype_id = _DTYPE_TO_ID.get(tensor.dtype)
        if dtype_id is None:
            raise CodecError(f"Unsupported dtype: {tensor.dtype}")
        
        if tensor.is_cuda:
            tensor_cpu = tensor.cpu()
        else:
            tensor_cpu = tensor
        
        np_array = tensor_cpu.numpy()
        raw_data = np_array.tobytes()
        
        flags = TensorFlags.CONTIGUOUS
        if tensor.is_cuda:
            flags |= TensorFlags.ON_CUDA
        if tensor.requires_grad:
            flags |= TensorFlags.REQUIRES_GRAD
        
        data = raw_data
        if self.compress and len(raw_data) > self.compression_threshold:
            lz4 = _ensure_lz4()
            if lz4:
                compressed = lz4.compress(raw_data)
                if len(compressed) < len(raw_data) * 0.9:
                    data = compressed
                    flags |= TensorFlags.COMPRESSED
        
        return TensorFrame(
            dtype_id=dtype_id,
            shape=tuple(tensor.shape),
            data=data,
            flags=flags,
        )


# ============================================================================
# Decoder
# ============================================================================

class TensorDecoder:
    """
    Decodes bytes to PyTorch tensors.
    
    [SECURITY]
    - Validates all header fields
    - Enforces size limits
    - No arbitrary code execution
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: Target device for decoded tensors
        """
        self.device = device
        _init_dtype_mapping()
    
    def decode(self, data: bytes) -> "torch.Tensor":
        """
        Decode bytes to tensor.
        
        Args:
            data: Serialized tensor bytes
        
        Returns:
            PyTorch tensor
        
        Raises:
            CodecError: If decoding fails
        """
        frame = self._parse_frame(data)
        return self._frame_to_tensor(frame)
    
    def _parse_frame(self, data: bytes) -> TensorFrame:
        """Parse TensorFrame from bytes."""
        if len(data) < HEADER_BASE_SIZE:
            raise CodecError(f"Data too short: {len(data)} < {HEADER_BASE_SIZE}")
        
        # Parse base header
        magic, version, dtype_id, flags, ndim = struct.unpack(
            HEADER_BASE_FORMAT,
            data[:HEADER_BASE_SIZE]
        )
        
        # Validate magic
        if magic != TENSOR_MAGIC:
            raise CodecError(f"Invalid magic: {magic!r} != {TENSOR_MAGIC!r}")
        
        # Validate version
        if version != CODEC_VERSION:
            raise CodecError(f"Unsupported version: {version}")
        
        # Validate ndim
        if ndim > MAX_NDIM:
            raise CodecError(f"Too many dimensions: {ndim} > {MAX_NDIM}")
        
        # Calculate expected header size
        header_size = HEADER_BASE_SIZE + ndim * 8 + 8
        if len(data) < header_size:
            raise CodecError(f"Incomplete header: {len(data)} < {header_size}")
        
        # Parse shape
        offset = HEADER_BASE_SIZE
        shape = []
        for _ in range(ndim):
            dim, = struct.unpack('>q', data[offset:offset+8])
            shape.append(dim)
            offset += 8
        
        # Parse data length
        data_len, = struct.unpack('>Q', data[offset:offset+8])
        offset += 8
        
        # Validate data length
        if data_len > MAX_TENSOR_SIZE:
            raise TensorTooLargeError(f"Data too large: {data_len} > {MAX_TENSOR_SIZE}")
        
        expected_total = header_size + data_len
        if len(data) < expected_total:
            raise CodecError(f"Incomplete data: {len(data)} < {expected_total}")
        
        tensor_data = data[offset:offset + data_len]
        
        return TensorFrame(
            dtype_id=TensorDType(dtype_id),
            shape=tuple(shape),
            data=tensor_data,
            flags=flags,
        )
    
    def _frame_to_tensor(self, frame: TensorFrame) -> "torch.Tensor":
        """Convert TensorFrame to PyTorch tensor."""
        torch = _ensure_torch()
        np = _ensure_numpy()
        
        # Get dtype
        dtype = _ID_TO_DTYPE.get(frame.dtype_id)
        if dtype is None:
            raise CodecError(f"Unknown dtype ID: {frame.dtype_id}")
        
        # Decompress if needed
        data = frame.data
        if frame.is_compressed:
            lz4 = _ensure_lz4()
            if lz4 is None:
                raise CodecError("LZ4 not available for decompression")
            data = lz4.decompress(data)
        
        # Get numpy dtype
        np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
        
        # Create numpy array from bytes (zero-copy)
        np_array = np.frombuffer(data, dtype=np_dtype).reshape(frame.shape)
        
        # Convert to tensor
        tensor = torch.from_numpy(np_array.copy())  # copy for ownership
        
        # Move to device
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        
        return tensor


# ============================================================================
# Chunked Streaming
# ============================================================================

class TensorChunker:
    """
    Chunk large tensors for streaming.
    
    [NETWORK] Large tensors are split into chunks for:
    - Better network utilization
    - Progress tracking
    - Partial failure recovery
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.encoder = TensorEncoder(compress=False)
    
    def chunk(
        self,
        tensor: "torch.Tensor",
        tensor_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Split tensor into chunks for streaming.
        
        Returns list of chunk metadata dicts with:
        - tensor_id: Identifier
        - chunk_idx: Chunk index
        - total_chunks: Total count
        - data: Chunk bytes
        - is_last: Last chunk flag
        """
        # Encode full tensor
        full_data = self.encoder.encode(tensor)
        
        chunks = []
        offset = 0
        chunk_idx = 0
        total_chunks = (len(full_data) + self.chunk_size - 1) // self.chunk_size
        
        while offset < len(full_data):
            chunk_data = full_data[offset:offset + self.chunk_size]
            is_last = offset + self.chunk_size >= len(full_data)
            
            chunks.append({
                "tensor_id": tensor_id,
                "chunk_idx": chunk_idx,
                "total_chunks": total_chunks,
                "data": chunk_data,
                "is_last": is_last,
            })
            
            offset += self.chunk_size
            chunk_idx += 1
        
        return chunks


class TensorAssembler:
    """
    Assemble tensor from streamed chunks.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.decoder = TensorDecoder(device=device)
        self._buffers: Dict[str, Dict[int, bytes]] = {}
        self._expected: Dict[str, int] = {}
    
    def add_chunk(self, chunk: Dict[str, Any]) -> Optional["torch.Tensor"]:
        """
        Add chunk and return tensor if complete.
        
        Args:
            chunk: Chunk dict from TensorChunker
        
        Returns:
            Complete tensor if all chunks received, else None
        """
        tensor_id = chunk["tensor_id"]
        chunk_idx = chunk["chunk_idx"]
        total_chunks = chunk["total_chunks"]
        data = chunk["data"]
        
        # Initialize buffer
        if tensor_id not in self._buffers:
            self._buffers[tensor_id] = {}
            self._expected[tensor_id] = total_chunks
        
        # Store chunk
        self._buffers[tensor_id][chunk_idx] = data
        
        # Check if complete
        if len(self._buffers[tensor_id]) == self._expected[tensor_id]:
            # Assemble
            full_data = b''.join(
                self._buffers[tensor_id][i]
                for i in range(total_chunks)
            )
            
            # Cleanup
            del self._buffers[tensor_id]
            del self._expected[tensor_id]
            
            # Decode
            return self.decoder.decode(full_data)
        
        return None
    
    def pending_tensors(self) -> List[str]:
        """List of tensor IDs with pending chunks."""
        return list(self._buffers.keys())
    
    def progress(self, tensor_id: str) -> Tuple[int, int]:
        """
        Get progress for a tensor.
        
        Returns:
            (received_chunks, total_chunks)
        """
        if tensor_id not in self._buffers:
            return (0, 0)
        return (len(self._buffers[tensor_id]), self._expected[tensor_id])


# ============================================================================
# Convenience Functions
# ============================================================================

def encode_tensor(
    tensor: "torch.Tensor",
    compress: bool = False,
) -> bytes:
    """
    Encode tensor to bytes.
    
    Args:
        tensor: PyTorch tensor
        compress: Enable LZ4 compression
    
    Returns:
        Serialized bytes
    """
    encoder = TensorEncoder(compress=compress)
    return encoder.encode(tensor)


def decode_tensor(
    data: bytes,
    device: str = "cpu",
) -> "torch.Tensor":
    """
    Decode bytes to tensor.
    
    Args:
        data: Serialized bytes
        device: Target device
    
    Returns:
        PyTorch tensor
    """
    decoder = TensorDecoder(device=device)
    return decoder.decode(data)


# ============================================================================
# Alternative: torch.save to BytesIO (for compatibility)
# ============================================================================

def encode_tensor_compat(tensor: "torch.Tensor") -> bytes:
    """
    Encode tensor using torch.save (compatibility mode).
    
    [WARNING] Uses pickle internally - less secure but more compatible.
    Only use for trusted networks.
    """
    torch = _ensure_torch()
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def decode_tensor_compat(
    data: bytes,
    device: str = "cpu",
) -> "torch.Tensor":
    """
    Decode tensor using torch.load (compatibility mode).
    
    [WARNING] Uses pickle - only use for trusted data.
    """
    torch = _ensure_torch()
    buffer = io.BytesIO(data)
    tensor = torch.load(buffer, map_location=device, weights_only=True)
    return tensor

