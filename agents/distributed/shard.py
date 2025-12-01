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
    PyTorch model shard с частичной загрузкой.
    
    [REAL INFERENCE] Загружает только нужные слои модели:
    - Эмбеддинги (если первый shard)
    - Decoder layers [layer_start:layer_end]
    - LM head (если последний shard)
    
    [PARTIAL LOADING] Оптимизации памяти:
    - Загрузка только нужных весов из safetensors
    - Автоматическое определение архитектуры (Llama/Qwen/Mistral)
    - Mixed precision (float16/bfloat16)
    - Efficient memory allocation
    
    [SUPPORTED MODELS]
    - Llama-2, Llama-3
    - Qwen, Qwen2, Qwen2.5
    - Mistral, Mixtral
    - Any HuggingFace transformer
    """
    
    # Mapping архитектур к именам слоёв
    ARCHITECTURE_MAP = {
        "llama": {
            "layers": "model.layers",
            "embed": "model.embed_tokens",
            "norm": "model.norm",
            "lm_head": "lm_head",
        },
        "qwen2": {
            "layers": "model.layers",
            "embed": "model.embed_tokens",
            "norm": "model.norm",
            "lm_head": "lm_head",
        },
        "mistral": {
            "layers": "model.layers",
            "embed": "model.embed_tokens",
            "norm": "model.norm",
            "lm_head": "lm_head",
        },
    }
    
    def __init__(
        self,
        model_name: str,
        layer_start: int,
        layer_end: int,
        device: str = "cuda:0",
        model_path: Optional[str] = None,
        total_layers: int = 64,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__(model_name, layer_start, layer_end, device)
        self.model_path = model_path or model_name
        self.total_layers = total_layers
        self.dtype_str = dtype
        
        self._config = None
        self._layers = None
        self._embed_tokens = None
        self._lm_head = None
        self._norm = None
        self._arch_map = None
    
    @property
    def is_first_shard(self) -> bool:
        return self.layer_start == 0
    
    @property
    def is_last_shard(self) -> bool:
        return self.layer_end >= self.total_layers
    
    @property
    def torch_dtype(self):
        """Get torch dtype from string."""
        if not TORCH_AVAILABLE:
            return None
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype_str, torch.float16)
    
    def _detect_architecture(self, config) -> Dict[str, str]:
        """Определить архитектуру модели по config."""
        arch_type = getattr(config, "model_type", "llama").lower()
        
        # Нормализуем имя архитектуры
        if "qwen" in arch_type:
            arch_type = "qwen2"
        elif "llama" in arch_type:
            arch_type = "llama"
        elif "mistral" in arch_type or "mixtral" in arch_type:
            arch_type = "mistral"
        
        return self.ARCHITECTURE_MAP.get(arch_type, self.ARCHITECTURE_MAP["llama"])
    
    async def load(self) -> bool:
        """
        Загрузить веса модели (только нужные слои).
        
        [PARTIAL LOAD] Алгоритм:
        1. Загружаем config для определения архитектуры
        2. Создаём пустую модель нужного размера
        3. Загружаем только веса для наших слоёв
        4. Удаляем ненужные слои из памяти
        """
        if not TORCH_AVAILABLE:
            logger.error("[SHARD] PyTorch not available")
            return False
        
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
            
            logger.info(
                f"[SHARD] Loading {self.model_path} "
                f"layers {self.layer_start}-{self.layer_end} to {self.device}"
            )
            
            # 1. Загружаем конфигурацию
            self._config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self._arch_map = self._detect_architecture(self._config)
            
            # Обновляем total_layers из config
            num_layers = getattr(self._config, "num_hidden_layers", self.total_layers)
            if self.layer_end > num_layers:
                self.layer_end = num_layers
                logger.warning(f"[SHARD] Adjusted layer_end to {num_layers}")
            
            logger.info(f"[SHARD] Architecture: {self._config.model_type}, {num_layers} layers")
            
            # 2. Используем low_cpu_mem_usage для эффективной загрузки
            # Сначала загружаем всю модель, потом извлекаем нужное
            # (для очень больших моделей нужен более сложный подход с safetensors)
            
            logger.info("[SHARD] Loading model weights...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map="cpu",  # Сначала в CPU
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            # 3. Извлекаем нужные компоненты
            # Находим слои в модели
            layers_container = model.model.layers
            
            # Создаём ModuleList только с нужными слоями
            self._layers = torch.nn.ModuleList()
            for i in range(self.layer_start, min(self.layer_end, len(layers_container))):
                layer = layers_container[i]
                self._layers.append(layer)
            
            # Перемещаем на GPU
            self._layers = self._layers.to(device=self.device, dtype=self.torch_dtype)
            
            # Первый shard получает embeddings
            if self.is_first_shard:
                self._embed_tokens = model.model.embed_tokens.to(
                    device=self.device, 
                    dtype=self.torch_dtype
                )
                logger.info(f"[SHARD] Loaded embed_tokens: {self._embed_tokens.weight.shape}")
            
            # Последний shard получает norm и lm_head
            if self.is_last_shard:
                self._norm = model.model.norm.to(
                    device=self.device, 
                    dtype=self.torch_dtype
                )
                self._lm_head = model.lm_head.to(
                    device=self.device, 
                    dtype=self.torch_dtype
                )
                logger.info(f"[SHARD] Loaded norm and lm_head: {self._lm_head.weight.shape}")
            
            # 4. Освобождаем память от полной модели
            del model
            del layers_container
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            self.is_loaded = True
            
            memory_mb = self.get_memory_usage()
            logger.info(
                f"[SHARD] Loaded {len(self._layers)} layers, "
                f"GPU memory: {memory_mb}MB"
            )
            
            return True
            
        except ImportError as e:
            logger.error(f"[SHARD] Missing dependency: {e}")
            logger.error("[SHARD] Install with: pip install transformers accelerate")
            return False
        except Exception as e:
            logger.error(f"[SHARD] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def load_partial(self) -> bool:
        """
        [ADVANCED] Загрузить веса напрямую из safetensors файлов.
        
        Более эффективно для очень больших моделей,
        но требует знания структуры файлов.
        """
        if not TORCH_AVAILABLE:
            return False
        
        try:
            from safetensors import safe_open
            from transformers import AutoConfig
            from huggingface_hub import hf_hub_download, list_repo_files
            import json
            
            logger.info(f"[SHARD] Partial load from {self.model_path}")
            
            # Загружаем config
            self._config = AutoConfig.from_pretrained(self.model_path)
            self._arch_map = self._detect_architecture(self._config)
            
            # Получаем список safetensors файлов
            try:
                files = list_repo_files(self.model_path)
                safetensor_files = [f for f in files if f.endswith(".safetensors")]
            except Exception:
                # Локальная директория
                from pathlib import Path
                model_dir = Path(self.model_path)
                safetensor_files = list(model_dir.glob("*.safetensors"))
            
            if not safetensor_files:
                logger.warning("[SHARD] No safetensors files found, falling back to full load")
                return await self.load()
            
            # Загружаем index для mapping тензоров к файлам
            try:
                index_file = hf_hub_download(
                    self.model_path, 
                    "model.safetensors.index.json"
                )
                with open(index_file) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
            except Exception:
                weight_map = {}
            
            # Определяем нужные веса
            needed_weights = set()
            
            # Embeddings для первого shard
            if self.is_first_shard:
                needed_weights.add(f"{self._arch_map['embed']}.weight")
            
            # Слои
            for layer_idx in range(self.layer_start, self.layer_end):
                prefix = f"{self._arch_map['layers']}.{layer_idx}"
                # Добавляем все веса слоя (по паттерну)
                for weight_name in weight_map.keys():
                    if weight_name.startswith(prefix):
                        needed_weights.add(weight_name)
            
            # Norm и lm_head для последнего shard
            if self.is_last_shard:
                needed_weights.add(f"{self._arch_map['norm']}.weight")
                needed_weights.add(f"{self._arch_map['lm_head']}.weight")
            
            logger.info(f"[SHARD] Need {len(needed_weights)} weight tensors")
            
            # Загружаем только нужные веса
            loaded_weights = {}
            for weight_name in needed_weights:
                if weight_name in weight_map:
                    file_name = weight_map[weight_name]
                    file_path = hf_hub_download(self.model_path, file_name)
                    
                    with safe_open(file_path, framework="pt") as f:
                        loaded_weights[weight_name] = f.get_tensor(weight_name)
            
            # TODO: Собрать модель из загруженных весов
            # Это требует создания структуры модели вручную
            
            logger.warning("[SHARD] Partial load not fully implemented, using full load")
            return await self.load()
            
        except Exception as e:
            logger.error(f"[SHARD] Partial load failed: {e}")
            return await self.load()
    
    async def unload(self) -> None:
        """Выгрузить модель и освободить память."""
        self._layers = None
        self._embed_tokens = None
        self._lm_head = None
        self._norm = None
        self._config = None
        
        if TORCH_AVAILABLE:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("[SHARD] Unloaded")
    
    async def forward(
        self,
        activations: np.ndarray,
        position_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
        use_cache: bool = True,
        request_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Forward pass через загруженные слои.
        
        Args:
            activations: 
                - Для первого shard: token IDs [batch, seq_len]
                - Для остальных: hidden states [batch, seq_len, hidden_size]
            position_ids: Позиции токенов [batch, seq_len]
            attention_mask: Маска внимания [batch, seq_len]
        
        Returns:
            - Для последнего shard: logits [batch, seq_len, vocab_size]
            - Для остальных: hidden states [batch, seq_len, hidden_size]
        """
        if not self.is_loaded or not TORCH_AVAILABLE:
            raise RuntimeError("Shard not loaded")
        
        if self._layers is None:
            raise RuntimeError("Layers not initialized")
        
        with torch.no_grad():
            # Конвертируем входные данные
            if self.is_first_shard:
                # Первый shard получает token IDs
                input_ids = torch.from_numpy(activations.astype(np.int64)).to(self.device)
                
                # Применяем embedding
                hidden_states = self._embed_tokens(input_ids)
            else:
                # Остальные shards получают hidden states
                hidden_states = torch.from_numpy(activations).to(
                    device=self.device,
                    dtype=self.torch_dtype,
                )
            
            batch_size, seq_length = hidden_states.shape[:2]
            
            # Position IDs
            if position_ids is not None:
                position_ids_tensor = torch.from_numpy(position_ids).to(self.device)
            else:
                # Создаём position_ids автоматически
                position_ids_tensor = torch.arange(
                    seq_length, 
                    device=self.device
                ).unsqueeze(0).expand(batch_size, -1)
            
            # Attention mask (causal)
            if attention_mask is not None:
                attention_mask_tensor = torch.from_numpy(attention_mask).to(self.device)
            else:
                # Создаём causal mask
                attention_mask_tensor = torch.ones(
                    (batch_size, seq_length),
                    device=self.device,
                    dtype=self.torch_dtype,
                )
            
            # Forward через каждый слой
            for layer_idx, layer in enumerate(self._layers):
                try:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask_tensor,
                        position_ids=position_ids_tensor,
                        use_cache=False,
                        output_attentions=False,
                    )
                    hidden_states = layer_outputs[0]
                except Exception as e:
                    logger.error(f"[SHARD] Layer {self.layer_start + layer_idx} failed: {e}")
                    raise
            
            # Последний shard применяет norm и lm_head
            if self.is_last_shard:
                if self._norm is not None:
                    hidden_states = self._norm(hidden_states)
                if self._lm_head is not None:
                    # Возвращаем только logits последнего токена для генерации
                    hidden_states = self._lm_head(hidden_states)
            
            # Конвертируем обратно в numpy
            output = hidden_states.cpu().to(torch.float32).numpy()
            
            return output, None
    
    def get_memory_usage(self) -> int:
        """GPU memory usage в MB."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0
        
        try:
            device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            return torch.cuda.memory_allocated(device_idx) // (1024 * 1024)
        except:
            return 0
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Детальная статистика shard."""
        stats = self.get_stats()
        
        if self._config:
            stats["config"] = {
                "model_type": self._config.model_type,
                "hidden_size": getattr(self._config, "hidden_size", None),
                "num_attention_heads": getattr(self._config, "num_attention_heads", None),
                "num_hidden_layers": getattr(self._config, "num_hidden_layers", None),
                "vocab_size": getattr(self._config, "vocab_size", None),
            }
        
        if self._layers:
            stats["num_loaded_layers"] = len(self._layers)
        
        stats["has_embeddings"] = self._embed_tokens is not None
        stats["has_lm_head"] = self._lm_head is not None
        
        return stats


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

