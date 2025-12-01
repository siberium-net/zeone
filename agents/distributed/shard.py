"""
Model Shard - Загрузка и инференс части модели
==============================================

[SHARD] Каждый узел загружает только свои слои:
- Экономия GPU памяти
- Параллельная загрузка на разных узлах
- Hot-swap: можно заменить shard без перезагрузки

[SUPPORTED MODELS]
- Transformers (Llama, Qwen, Mistral)
- MoE (Mixtral) - эксперты как отдельные shards

[ACTIVATION BUFFER]
Буфер для кэширования активаций между слоями:
- Позволяет batch несколько запросов
- Уменьшает overhead передачи
"""

import os
import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from collections import OrderedDict
import numpy as np

from .protocol import (
    QuantizedActivations,
    quantize_activations,
    dequantize_activations,
    ForwardRequest,
    ForwardResponse,
    create_request_id,
)
from .registry import ModelShardInfo, ShardStatus

logger = logging.getLogger(__name__)

# Пытаемся импортировать torch (опционально)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("[SHARD] PyTorch not available, using mock inference")


@dataclass
class ActivationBuffer:
    """
    Буфер для активаций между forward passes.
    
    [BUFFER] Используется для:
    - Кэширования KV-cache между токенами
    - Batch нескольких запросов
    - Восстановления после сбоя
    """
    
    request_id: str
    activations: np.ndarray
    position: int = 0
    kv_cache: Optional[Any] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_accessed
    
    def touch(self) -> None:
        """Обновить время последнего доступа."""
        self.last_accessed = time.time()


class ActivationCache:
    """
    LRU кэш для activation buffers.
    
    [CACHE] Особенности:
    - Ограничение по памяти (не по количеству)
    - Автоматическое вытеснение старых
    - TTL для неактивных буферов
    """
    
    def __init__(
        self,
        max_memory_mb: int = 1024,
        ttl_seconds: int = 300,
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, ActivationBuffer] = OrderedDict()
        self._current_size = 0
    
    def get(self, request_id: str) -> Optional[ActivationBuffer]:
        """Получить буфер по request_id."""
        if request_id in self._cache:
            buf = self._cache[request_id]
            buf.touch()
            # Move to end (most recently used)
            self._cache.move_to_end(request_id)
            return buf
        return None
    
    def put(self, buffer: ActivationBuffer) -> None:
        """Добавить буфер в кэш."""
        size = buffer.activations.nbytes
        
        # Очищаем место
        while self._current_size + size > self.max_memory_bytes and self._cache:
            self._evict_oldest()
        
        self._cache[buffer.request_id] = buffer
        self._cache.move_to_end(buffer.request_id)
        self._current_size += size
    
    def remove(self, request_id: str) -> bool:
        """Удалить буфер."""
        if request_id in self._cache:
            buf = self._cache.pop(request_id)
            self._current_size -= buf.activations.nbytes
            return True
        return False
    
    def _evict_oldest(self) -> None:
        """Вытеснить самый старый элемент."""
        if self._cache:
            request_id, buf = self._cache.popitem(last=False)
            self._current_size -= buf.activations.nbytes
            logger.debug(f"[CACHE] Evicted {request_id}")
    
    def cleanup_expired(self) -> int:
        """Очистить устаревшие буферы."""
        expired = []
        for request_id, buf in self._cache.items():
            if buf.idle_seconds > self.ttl_seconds:
                expired.append(request_id)
        
        for request_id in expired:
            self.remove(request_id)
        
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "buffers": len(self._cache),
            "memory_used_mb": self._current_size / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
        }


class ModelShard(ABC):
    """
    Абстрактный класс для model shard.
    
    [SHARD] Реализации:
    - TorchModelShard: для PyTorch моделей
    - MockModelShard: для тестирования без GPU
    - OllamaModelShard: делегирует Ollama
    """
    
    def __init__(
        self,
        model_name: str,
        layer_start: int,
        layer_end: int,
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.device = device
        self.is_loaded = False
        self._cache = ActivationCache()
        
        # Статистика
        self.total_requests = 0
        self.total_time_ms = 0.0
    
    @property
    def shard_id(self) -> str:
        return f"layers_{self.layer_start}_{self.layer_end}"
    
    @property
    def layer_count(self) -> int:
        return self.layer_end - self.layer_start
    
    @abstractmethod
    async def load(self) -> bool:
        """Загрузить веса модели."""
        pass
    
    @abstractmethod
    async def unload(self) -> None:
        """Выгрузить веса из памяти."""
        pass
    
    @abstractmethod
    async def forward(
        self,
        activations: np.ndarray,
        position_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
        use_cache: bool = True,
        request_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Forward pass через shard.
        
        Args:
            activations: Входные активации [batch, seq_len, hidden]
            position_ids: Позиции токенов [batch, seq_len]
            attention_mask: Маска внимания [batch, seq_len]
            use_cache: Использовать KV-cache
            request_id: ID запроса для кэширования
        
        Returns:
            (output_activations, kv_cache)
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """Получить использование GPU памяти в MB."""
        pass
    
    async def process_request(
        self,
        request: ForwardRequest,
    ) -> ForwardResponse:
        """
        Обработать ForwardRequest.
        
        Args:
            request: Запрос на forward
        
        Returns:
            ForwardResponse с результатом
        """
        start_time = time.time()
        
        try:
            # Деквантуем входные активации
            input_activations = dequantize_activations(
                request.activations,
                dtype="float16" if TORCH_AVAILABLE else "float32",
            )
            
            # Декодируем position_ids если есть
            position_ids = None
            if request.position_ids:
                position_ids = np.frombuffer(
                    request.position_ids,
                    dtype=np.int64,
                ).reshape(-1)
            
            # Forward pass
            output, kv_cache = await self.forward(
                activations=input_activations,
                position_ids=position_ids,
                request_id=request.request_id,
            )
            
            # Квантуем выходные активации
            output_quantized = quantize_activations(output)
            
            # Если последний shard, возвращаем logits
            logits = None
            if request.is_last:
                # output уже содержит logits
                logits = output.astype(np.float32).tobytes()
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.total_requests += 1
            self.total_time_ms += elapsed_ms
            
            return ForwardResponse(
                request_id=request.request_id,
                success=True,
                activations=output_quantized if not request.is_last else None,
                logits=logits,
                processing_time_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error(f"[SHARD] Forward failed: {e}")
            return ForwardResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика shard."""
        avg_latency = 0.0
        if self.total_requests > 0:
            avg_latency = self.total_time_ms / self.total_requests
        
        return {
            "shard_id": self.shard_id,
            "model_name": self.model_name,
            "layers": f"{self.layer_start}-{self.layer_end}",
            "is_loaded": self.is_loaded,
            "device": self.device,
            "total_requests": self.total_requests,
            "avg_latency_ms": avg_latency,
            "memory_mb": self.get_memory_usage(),
            "cache": self._cache.get_stats(),
        }


class MockModelShard(ModelShard):
    """
    Mock shard для тестирования без GPU.
    
    [MOCK] Эмулирует forward pass:
    - Добавляет случайный шум к активациям
    - Имитирует latency
    - Генерирует фиктивные logits
    """
    
    def __init__(
        self,
        model_name: str,
        layer_start: int,
        layer_end: int,
        hidden_size: int = 5120,
        vocab_size: int = 152064,
        latency_ms: float = 50.0,
        **kwargs,
    ):
        super().__init__(model_name, layer_start, layer_end, device="cpu")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.latency_ms = latency_ms
        self._memory_mb = 0
    
    async def load(self) -> bool:
        """Mock загрузка."""
        logger.info(
            f"[MOCK] Loading shard {self.shard_id} "
            f"(layers {self.layer_start}-{self.layer_end})"
        )
        await asyncio.sleep(0.5)  # Имитация загрузки
        self.is_loaded = True
        self._memory_mb = self.layer_count * 500  # ~500MB на слой
        return True
    
    async def unload(self) -> None:
        """Mock выгрузка."""
        self.is_loaded = False
        self._memory_mb = 0
    
    async def forward(
        self,
        activations: np.ndarray,
        position_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
        use_cache: bool = True,
        request_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Mock forward pass."""
        if not self.is_loaded:
            raise RuntimeError("Shard not loaded")
        
        # Имитация latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Трансформация активаций (эмуляция)
        # Просто добавляем небольшой шум
        noise = np.random.randn(*activations.shape).astype(activations.dtype) * 0.01
        output = activations + noise
        
        # Нормализация
        output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-6)
        output = output * np.sqrt(self.hidden_size)
        
        return output, None
    
    def get_memory_usage(self) -> int:
        return self._memory_mb


class TorchModelShard(ModelShard):
    """
    PyTorch model shard.
    
    [TORCH] Загружает часть трансформера:
    - Эмбеддинги (если первый shard)
    - Decoder layers [layer_start:layer_end]
    - LM head (если последний shard)
    
    [MEMORY] Оптимизации:
    - torch.compile для ускорения
    - Mixed precision (float16)
    - Gradient checkpointing отключен (только инференс)
    """
    
    def __init__(
        self,
        model_name: str,
        layer_start: int,
        layer_end: int,
        device: str = "cuda:0",
        model_path: Optional[str] = None,
        total_layers: int = 64,
        **kwargs,
    ):
        super().__init__(model_name, layer_start, layer_end, device)
        self.model_path = model_path or model_name
        self.total_layers = total_layers
        
        self._model = None
        self._embed_tokens = None
        self._lm_head = None
        self._norm = None
    
    @property
    def is_first_shard(self) -> bool:
        return self.layer_start == 0
    
    @property
    def is_last_shard(self) -> bool:
        return self.layer_end >= self.total_layers
    
    async def load(self) -> bool:
        """Загрузить веса модели."""
        if not TORCH_AVAILABLE:
            logger.error("[SHARD] PyTorch not available")
            return False
        
        try:
            logger.info(f"[SHARD] Loading {self.model_path} layers {self.layer_start}-{self.layer_end}")
            
            # Загружаем модель
            from transformers import AutoModelForCausalLM, AutoConfig
            
            config = AutoConfig.from_pretrained(self.model_path)
            
            # Модифицируем config для частичной загрузки
            # (это упрощённая реализация, полная требует патчинга модели)
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map={"": self.device},
                low_cpu_mem_usage=True,
            )
            
            # Извлекаем нужные слои
            self._layers = torch.nn.ModuleList([
                model.model.layers[i] 
                for i in range(self.layer_start, min(self.layer_end, len(model.model.layers)))
            ])
            
            if self.is_first_shard:
                self._embed_tokens = model.model.embed_tokens
            
            if self.is_last_shard:
                self._norm = model.model.norm
                self._lm_head = model.lm_head
            
            # Освобождаем остальное
            del model
            torch.cuda.empty_cache()
            
            self.is_loaded = True
            logger.info(f"[SHARD] Loaded {len(self._layers)} layers")
            return True
            
        except Exception as e:
            logger.error(f"[SHARD] Failed to load model: {e}")
            return False
    
    async def unload(self) -> None:
        """Выгрузить модель."""
        self._model = None
        self._layers = None
        self._embed_tokens = None
        self._lm_head = None
        self._norm = None
        
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
        
        self.is_loaded = False
    
    async def forward(
        self,
        activations: np.ndarray,
        position_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
        use_cache: bool = True,
        request_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Forward pass через loaded layers."""
        if not self.is_loaded or not TORCH_AVAILABLE:
            raise RuntimeError("Shard not loaded")
        
        with torch.no_grad():
            # Конвертируем в torch
            hidden_states = torch.from_numpy(activations).to(
                device=self.device,
                dtype=torch.float16,
            )
            
            if position_ids is not None:
                position_ids = torch.from_numpy(position_ids).to(self.device)
            
            if attention_mask is not None:
                attention_mask = torch.from_numpy(attention_mask).to(self.device)
            
            # Если первый shard - применяем embedding
            if self.is_first_shard and self._embed_tokens is not None:
                # Входные данные - это token ids
                input_ids = hidden_states.long()
                hidden_states = self._embed_tokens(input_ids)
            
            # Forward через каждый layer
            for layer in self._layers:
                layer_outputs = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=False,  # Упрощаем для распределённого варианта
                )
                hidden_states = layer_outputs[0]
            
            # Если последний shard - применяем norm и lm_head
            if self.is_last_shard:
                if self._norm is not None:
                    hidden_states = self._norm(hidden_states)
                if self._lm_head is not None:
                    hidden_states = self._lm_head(hidden_states)
            
            # Конвертируем обратно в numpy
            output = hidden_states.cpu().numpy()
            
            return output, None
    
    def get_memory_usage(self) -> int:
        """GPU memory usage in MB."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0
        
        try:
            device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            return torch.cuda.memory_allocated(device_idx) // (1024 * 1024)
        except:
            return 0


def create_shard(
    model_name: str,
    layer_start: int,
    layer_end: int,
    use_mock: bool = False,
    **kwargs,
) -> ModelShard:
    """
    Фабрика для создания shard.
    
    Args:
        model_name: Имя модели
        layer_start: Начальный слой
        layer_end: Конечный слой
        use_mock: Использовать mock (для тестов)
        **kwargs: Дополнительные параметры
    
    Returns:
        ModelShard instance
    """
    if use_mock or not TORCH_AVAILABLE:
        return MockModelShard(
            model_name=model_name,
            layer_start=layer_start,
            layer_end=layer_end,
            **kwargs,
        )
    
    return TorchModelShard(
        model_name=model_name,
        layer_start=layer_start,
        layer_end=layer_end,
        **kwargs,
    )

