"""
Distributed Inference Protocol - Протокол распределённого инференса
===================================================================

[PROTOCOL] Сообщения для координации pipeline:
- FORWARD: Передача активаций на следующий shard
- FORWARD_RESPONSE: Ответ с обработанными активациями
- HEALTH_CHECK: Проверка состояния shard
- SYNC: Синхронизация состояния pipeline

[MESSAGE FORMAT]
+----------+---------+------------+------------------+
| Magic(4) | Type(1) | SeqNum(4)  | Payload(var)     |
+----------+---------+------------+------------------+

[ACTIVATION FORMAT]
Quantized int8 для экономии bandwidth:
- Shape: (batch, seq_len, hidden_size)
- Scale factor для восстановления float16
"""

import struct
import time
import hashlib
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# Protocol constants
PROTOCOL_MAGIC = b"DINF"  # Distributed INFerence
PROTOCOL_VERSION = 1
MAX_ACTIVATION_SIZE = 100 * 1024 * 1024  # 100 MB


class DistributedMessageType(IntEnum):
    """Типы сообщений протокола."""
    # Inference
    FORWARD = 1              # Передача активаций вперёд
    FORWARD_RESPONSE = 2     # Ответ с активациями
    
    # Health
    HEALTH_CHECK = 10        # Запрос состояния
    HEALTH_RESPONSE = 11     # Ответ о состоянии
    
    # Pipeline
    PIPELINE_START = 20      # Начало нового запроса
    PIPELINE_END = 21        # Завершение запроса
    PIPELINE_ABORT = 22      # Прерывание запроса
    
    # Sync
    SYNC_REQUEST = 30        # Запрос синхронизации
    SYNC_RESPONSE = 31       # Ответ синхронизации
    
    # Error
    ERROR = 99               # Ошибка


@dataclass
class QuantizedActivations:
    """
    Квантованные активации для передачи по сети.
    
    [QUANTIZATION] int8 с scale factor:
    - Экономит ~50% bandwidth по сравнению с float16
    - Минимальная потеря точности для промежуточных слоёв
    
    [FORMAT]
    - data: int8 numpy array
    - scale: float32 scale factor
    - shape: original shape (batch, seq_len, hidden)
    """
    
    data: bytes  # Quantized int8 data
    scale: float
    zero_point: int
    shape: Tuple[int, ...]
    dtype: str = "int8"
    
    @property
    def size_bytes(self) -> int:
        return len(self.data)
    
    @property
    def original_size_bytes(self) -> int:
        """Размер до квантования (float16)."""
        return int(np.prod(self.shape) * 2)  # float16 = 2 bytes
    
    @property
    def compression_ratio(self) -> float:
        """Коэффициент сжатия."""
        if self.original_size_bytes == 0:
            return 1.0
        return self.original_size_bytes / self.size_bytes
    
    def to_bytes(self) -> bytes:
        """Сериализация в bytes."""
        # Header: scale(4) + zero_point(4) + ndims(4) + shape(ndims*4)
        header = struct.pack(">fii", self.scale, self.zero_point, len(self.shape))
        for dim in self.shape:
            header += struct.pack(">i", dim)
        
        return header + self.data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "QuantizedActivations":
        """Десериализация из bytes."""
        offset = 0
        
        scale, zero_point, ndims = struct.unpack(">fii", data[offset:offset+12])
        offset += 12
        
        shape = []
        for _ in range(ndims):
            dim = struct.unpack(">i", data[offset:offset+4])[0]
            shape.append(dim)
            offset += 4
        
        tensor_data = data[offset:]
        
        return cls(
            data=tensor_data,
            scale=scale,
            zero_point=zero_point,
            shape=tuple(shape),
        )


def quantize_activations(
    tensor: np.ndarray,
    bits: int = 8,
) -> QuantizedActivations:
    """
    Квантовать float16/float32 тензор в int8.
    
    [ALGORITHM]
    1. Находим min/max значения
    2. Вычисляем scale = (max - min) / 255
    3. Квантуем: q = round((x - min) / scale)
    
    Args:
        tensor: NumPy array (float16 или float32)
        bits: Количество бит (8 для int8)
    
    Returns:
        QuantizedActivations
    """
    # Приводим к float32 для вычислений
    tensor_f32 = tensor.astype(np.float32)
    
    # Находим диапазон
    min_val = tensor_f32.min()
    max_val = tensor_f32.max()
    
    # Вычисляем scale и zero_point
    if max_val == min_val:
        scale = 1.0
        zero_point = 0
        quantized = np.zeros(tensor.shape, dtype=np.int8)
    else:
        scale = (max_val - min_val) / 255.0
        zero_point = int(round(-min_val / scale))
        
        # Квантуем
        quantized = np.clip(
            np.round(tensor_f32 / scale + zero_point),
            0, 255
        ).astype(np.uint8)
        
        # Конвертируем в signed int8 (для совместимости)
        quantized = (quantized.astype(np.int16) - 128).astype(np.int8)
        zero_point -= 128
    
    return QuantizedActivations(
        data=quantized.tobytes(),
        scale=scale,
        zero_point=zero_point,
        shape=tensor.shape,
    )


def dequantize_activations(
    quantized: QuantizedActivations,
    dtype: str = "float16",
) -> np.ndarray:
    """
    Деквантовать int8 обратно в float.
    
    Args:
        quantized: Квантованные активации
        dtype: Целевой dtype ("float16" или "float32")
    
    Returns:
        NumPy array с восстановленными значениями
    """
    # Восстанавливаем массив
    q_array = np.frombuffer(quantized.data, dtype=np.int8).reshape(quantized.shape)
    
    # Деквантуем
    dequantized = (q_array.astype(np.float32) - quantized.zero_point) * quantized.scale
    
    if dtype == "float16":
        return dequantized.astype(np.float16)
    return dequantized


@dataclass
class ForwardRequest:
    """
    Запрос на forward pass через shard.
    
    [FORWARD] Содержит:
    - request_id: Уникальный ID запроса
    - shard_id: ID целевого shard
    - activations: Входные активации (квантованные)
    - position_ids: Позиции токенов (для RoPE)
    - attention_mask: Маска внимания
    """
    
    request_id: str
    shard_id: str
    activations: QuantizedActivations
    layer_idx: int
    position_ids: Optional[bytes] = None  # int64 array
    attention_mask: Optional[bytes] = None  # bool array
    is_first: bool = False  # Первый shard в pipeline
    is_last: bool = False   # Последний shard в pipeline
    timestamp: float = field(default_factory=time.time)
    
    def to_bytes(self) -> bytes:
        """Сериализация."""
        # Основные поля
        request_id_bytes = self.request_id.encode()[:64].ljust(64, b'\x00')
        shard_id_bytes = self.shard_id.encode()[:64].ljust(64, b'\x00')
        
        # Флаги
        flags = (int(self.is_first) << 0) | (int(self.is_last) << 1)
        
        # Активации
        act_bytes = self.activations.to_bytes()
        
        # Header
        header = struct.pack(
            ">64s64siiBId",
            request_id_bytes,
            shard_id_bytes,
            self.layer_idx,
            len(act_bytes),
            flags,
            len(self.position_ids or b""),
            self.timestamp,
        )
        
        result = header + act_bytes
        
        if self.position_ids:
            result += self.position_ids
        
        return result
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "ForwardRequest":
        """Десериализация."""
        # Parse header
        header_size = 64 + 64 + 4 + 4 + 1 + 4 + 8
        header = data[:header_size]
        
        (request_id_bytes, shard_id_bytes, layer_idx, 
         act_len, flags, pos_len, timestamp) = struct.unpack(
            ">64s64siiBId", header
        )
        
        request_id = request_id_bytes.rstrip(b'\x00').decode()
        shard_id = shard_id_bytes.rstrip(b'\x00').decode()
        is_first = bool(flags & 1)
        is_last = bool(flags & 2)
        
        # Parse activations
        offset = header_size
        act_data = data[offset:offset + act_len]
        activations = QuantizedActivations.from_bytes(act_data)
        offset += act_len
        
        # Parse position_ids
        position_ids = None
        if pos_len > 0:
            position_ids = data[offset:offset + pos_len]
        
        return cls(
            request_id=request_id,
            shard_id=shard_id,
            activations=activations,
            layer_idx=layer_idx,
            position_ids=position_ids,
            is_first=is_first,
            is_last=is_last,
            timestamp=timestamp,
        )


@dataclass
class ForwardResponse:
    """
    Ответ на forward pass.
    
    [RESPONSE] Содержит:
    - request_id: ID исходного запроса
    - activations: Выходные активации (квантованные)
    - logits: Если is_last, содержит logits для генерации
    - error: Сообщение об ошибке если не удалось
    """
    
    request_id: str
    success: bool
    activations: Optional[QuantizedActivations] = None
    logits: Optional[bytes] = None  # float32 array для последнего shard
    next_shard_id: Optional[str] = None
    processing_time_ms: float = 0.0
    error: str = ""
    
    def to_bytes(self) -> bytes:
        """Сериализация."""
        request_id_bytes = self.request_id.encode()[:64].ljust(64, b'\x00')
        next_shard_bytes = (self.next_shard_id or "").encode()[:64].ljust(64, b'\x00')
        error_bytes = self.error.encode()[:256].ljust(256, b'\x00')
        
        act_bytes = self.activations.to_bytes() if self.activations else b""
        logits_bytes = self.logits or b""
        
        header = struct.pack(
            ">64s?64s256sfII",
            request_id_bytes,
            self.success,
            next_shard_bytes,
            error_bytes,
            self.processing_time_ms,
            len(act_bytes),
            len(logits_bytes),
        )
        
        return header + act_bytes + logits_bytes
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "ForwardResponse":
        """Десериализация."""
        header_size = 64 + 1 + 64 + 256 + 4 + 4 + 4
        
        (request_id_bytes, success, next_shard_bytes, error_bytes,
         processing_time_ms, act_len, logits_len) = struct.unpack(
            ">64s?64s256sfII", data[:header_size]
        )
        
        request_id = request_id_bytes.rstrip(b'\x00').decode()
        next_shard_id = next_shard_bytes.rstrip(b'\x00').decode() or None
        error = error_bytes.rstrip(b'\x00').decode()
        
        offset = header_size
        
        activations = None
        if act_len > 0:
            activations = QuantizedActivations.from_bytes(data[offset:offset + act_len])
            offset += act_len
        
        logits = None
        if logits_len > 0:
            logits = data[offset:offset + logits_len]
        
        return cls(
            request_id=request_id,
            success=success,
            activations=activations,
            logits=logits,
            next_shard_id=next_shard_id,
            processing_time_ms=processing_time_ms,
            error=error,
        )


@dataclass
class ShardHealthCheck:
    """
    Проверка состояния shard.
    """
    
    shard_id: str
    node_id: str
    is_healthy: bool
    current_load: float  # 0-1
    queue_size: int
    gpu_memory_used_mb: int
    gpu_memory_total_mb: int
    requests_per_second: float
    avg_latency_ms: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "is_healthy": self.is_healthy,
            "current_load": self.current_load,
            "queue_size": self.queue_size,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "requests_per_second": self.requests_per_second,
            "avg_latency_ms": self.avg_latency_ms,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardHealthCheck":
        return cls(**data)


def create_request_id(node_id: str, seq: int) -> str:
    """Создать уникальный ID запроса."""
    data = f"{node_id}:{seq}:{time.time()}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def pack_message(
    msg_type: DistributedMessageType,
    seq_num: int,
    payload: bytes,
) -> bytes:
    """
    Упаковать сообщение протокола.
    
    Args:
        msg_type: Тип сообщения
        seq_num: Sequence number
        payload: Полезная нагрузка
    
    Returns:
        Упакованное сообщение
    """
    header = struct.pack(
        ">4sBII",
        PROTOCOL_MAGIC,
        msg_type.value,
        seq_num,
        len(payload),
    )
    return header + payload


def unpack_message(data: bytes) -> Tuple[DistributedMessageType, int, bytes]:
    """
    Распаковать сообщение протокола.
    
    Args:
        data: Сырые данные
    
    Returns:
        (msg_type, seq_num, payload)
    
    Raises:
        ValueError: Если формат неверный
    """
    if len(data) < 13:
        raise ValueError("Message too short")
    
    magic, msg_type, seq_num, payload_len = struct.unpack(">4sBII", data[:13])
    
    if magic != PROTOCOL_MAGIC:
        raise ValueError(f"Invalid magic: {magic}")
    
    if len(data) < 13 + payload_len:
        raise ValueError("Incomplete payload")
    
    payload = data[13:13 + payload_len]
    
    return DistributedMessageType(msg_type), seq_num, payload

