"""
Model Registry - DHT реестр распределённых моделей
==================================================

[REGISTRY] Хранит информацию о доступных model shards:
- Какие модели доступны в сети
- Какие узлы обслуживают какие слои
- Health status каждого shard

[DHT KEYS]
- "model:{model_name}:info" → ModelInfo (описание модели)
- "model:{model_name}:shard:{shard_id}" → List[ModelShardInfo] (узлы с этим shard)
- "model:{model_name}:health:{node_id}" → ShardHealth (статус узла)

[REDUNDANCY]
- Каждый shard может быть на нескольких узлах
- Client выбирает узел по latency/load/trust
"""

import asyncio
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ShardStatus(Enum):
    """Статус model shard."""
    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"
    LOADING = "loading"


@dataclass
class ModelShardInfo:
    """
    Информация о model shard на конкретном узле.
    
    [SHARD] Один shard = часть модели (диапазон слоёв):
    - shard_id: уникальный ID (напр. "layers_0_10")
    - layer_start/end: диапазон слоёв
    - node_id: ID узла, обслуживающего shard
    """
    
    shard_id: str
    model_name: str
    layer_start: int
    layer_end: int
    node_id: str
    host: str
    port: int
    status: ShardStatus = ShardStatus.OFFLINE
    gpu_memory_mb: int = 0
    current_load: float = 0.0  # 0-1
    latency_ms: float = 0.0
    last_health_check: float = field(default_factory=time.time)
    trust_score: float = 0.5
    
    @property
    def layer_count(self) -> int:
        return self.layer_end - self.layer_start
    
    @property
    def is_available(self) -> bool:
        """Доступен ли shard для инференса."""
        return self.status in (ShardStatus.ONLINE, ShardStatus.BUSY) and self.current_load < 0.9
    
    @property
    def is_stale(self) -> bool:
        """Устарела ли информация о health."""
        return time.time() - self.last_health_check > 60  # 1 минута
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "model_name": self.model_name,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "gpu_memory_mb": self.gpu_memory_mb,
            "current_load": self.current_load,
            "latency_ms": self.latency_ms,
            "last_health_check": self.last_health_check,
            "trust_score": self.trust_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelShardInfo":
        return cls(
            shard_id=data["shard_id"],
            model_name=data["model_name"],
            layer_start=data["layer_start"],
            layer_end=data["layer_end"],
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            status=ShardStatus(data.get("status", "offline")),
            gpu_memory_mb=data.get("gpu_memory_mb", 0),
            current_load=data.get("current_load", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            last_health_check=data.get("last_health_check", time.time()),
            trust_score=data.get("trust_score", 0.5),
        )


@dataclass
class ModelInfo:
    """
    Информация о распределённой модели.
    
    [MODEL] Описание всей модели:
    - Имя, размер, архитектура
    - Количество слоёв и рекомендуемое разбиение
    - Требования к памяти
    """
    
    name: str
    architecture: str  # "transformer", "moe", etc.
    total_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    total_params_b: float  # billions
    recommended_shards: int  # рекомендуемое количество shards
    memory_per_shard_gb: float
    dtype: str = "float16"  # или "int8", "int4"
    
    def get_shard_layers(self, num_shards: int) -> List[Tuple[int, int]]:
        """
        Получить рекомендуемое разбиение на shards.
        
        Args:
            num_shards: Количество shards
        
        Returns:
            Список (layer_start, layer_end) для каждого shard
        """
        layers_per_shard = self.total_layers // num_shards
        extra = self.total_layers % num_shards
        
        shards = []
        start = 0
        
        for i in range(num_shards):
            # Распределяем extra слои равномерно
            end = start + layers_per_shard + (1 if i < extra else 0)
            shards.append((start, end))
            start = end
        
        return shards
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "total_layers": self.total_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "vocab_size": self.vocab_size,
            "total_params_b": self.total_params_b,
            "recommended_shards": self.recommended_shards,
            "memory_per_shard_gb": self.memory_per_shard_gb,
            "dtype": self.dtype,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        return cls(**data)


# Предопределённые модели
KNOWN_MODELS = {
    "qwen2.5-32b": ModelInfo(
        name="qwen2.5-32b",
        architecture="transformer",
        total_layers=64,
        hidden_size=5120,
        num_attention_heads=40,
        vocab_size=152064,
        total_params_b=32.5,
        recommended_shards=4,
        memory_per_shard_gb=20,
        dtype="float16",
    ),
    "llama2-70b": ModelInfo(
        name="llama2-70b",
        architecture="transformer",
        total_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        vocab_size=32000,
        total_params_b=70,
        recommended_shards=8,
        memory_per_shard_gb=20,
        dtype="float16",
    ),
    "mixtral-8x7b": ModelInfo(
        name="mixtral-8x7b",
        architecture="moe",
        total_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=32000,
        total_params_b=46.7,
        recommended_shards=4,
        memory_per_shard_gb=25,
        dtype="float16",
    ),
}


class ModelRegistry:
    """
    Реестр распределённых моделей в DHT.
    
    [REGISTRY] Функции:
    - Регистрация model shard на узле
    - Поиск узлов с нужными shards
    - Health monitoring
    - Load balancing
    
    [USAGE]
    ```python
    registry = ModelRegistry(kademlia_node)
    
    # Регистрация shard
    await registry.register_shard(shard_info)
    
    # Поиск shards для модели
    pipeline = await registry.get_model_pipeline("qwen2.5-32b")
    # → [ShardInfo(0-16), ShardInfo(16-32), ...]
    ```
    """
    
    def __init__(self, dht_node=None):
        """
        Args:
            dht_node: KademliaNode для доступа к DHT
        """
        self.dht = dht_node
        self._local_shards: Dict[str, ModelShardInfo] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 30  # секунды
    
    async def register_model_info(self, model: ModelInfo) -> bool:
        """
        Зарегистрировать информацию о модели в DHT.
        
        Args:
            model: Информация о модели
        
        Returns:
            True если успешно
        """
        if not self.dht:
            logger.warning("[REGISTRY] DHT not available")
            return False
        
        key = f"model:{model.name}:info"
        value = json.dumps(model.to_dict()).encode()
        
        try:
            await self.dht.dht_put(key, value)
            logger.info(f"[REGISTRY] Registered model info: {model.name}")
            return True
        except Exception as e:
            logger.error(f"[REGISTRY] Failed to register model: {e}")
            return False
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Получить информацию о модели.
        
        Args:
            model_name: Имя модели
        
        Returns:
            ModelInfo или None
        """
        # Сначала проверяем known models
        if model_name in KNOWN_MODELS:
            return KNOWN_MODELS[model_name]
        
        # Затем DHT
        if self.dht:
            key = f"model:{model_name}:info"
            try:
                data = await self.dht.dht_get(key)
                if data:
                    return ModelInfo.from_dict(json.loads(data.decode()))
            except Exception as e:
                logger.debug(f"[REGISTRY] DHT lookup failed: {e}")
        
        return None
    
    async def register_shard(self, shard: ModelShardInfo) -> bool:
        """
        Зарегистрировать model shard в DHT.
        
        [REDUNDANCY] Несколько узлов могут обслуживать один shard.
        
        Args:
            shard: Информация о shard
        
        Returns:
            True если успешно
        """
        # Сохраняем локально
        self._local_shards[shard.shard_id] = shard
        
        if not self.dht:
            return True
        
        # Регистрируем в DHT
        key = f"model:{shard.model_name}:shard:{shard.shard_id}:{shard.node_id}"
        value = json.dumps(shard.to_dict()).encode()
        
        try:
            await self.dht.dht_put(key, value)
            logger.info(
                f"[REGISTRY] Registered shard: {shard.model_name} "
                f"layers {shard.layer_start}-{shard.layer_end} on {shard.node_id[:16]}..."
            )
            return True
        except Exception as e:
            logger.error(f"[REGISTRY] Failed to register shard: {e}")
            return False
    
    async def unregister_shard(self, shard_id: str) -> bool:
        """Отменить регистрацию shard."""
        shard = self._local_shards.pop(shard_id, None)
        
        if shard and self.dht:
            # В DHT нет delete, просто обновляем статус
            shard.status = ShardStatus.OFFLINE
            key = f"model:{shard.model_name}:shard:{shard.shard_id}:{shard.node_id}"
            value = json.dumps(shard.to_dict()).encode()
            await self.dht.dht_put(key, value)
        
        return shard is not None
    
    async def find_shards(
        self,
        model_name: str,
        layer_start: int,
        layer_end: int,
    ) -> List[ModelShardInfo]:
        """
        Найти узлы с shards, покрывающими указанные слои.
        
        Args:
            model_name: Имя модели
            layer_start: Начальный слой
            layer_end: Конечный слой
        
        Returns:
            Список доступных shards
        """
        shards = []
        
        # Проверяем локальные
        for shard in self._local_shards.values():
            if (shard.model_name == model_name and 
                shard.layer_start <= layer_start and 
                shard.layer_end >= layer_end):
                shards.append(shard)
        
        # TODO: Поиск в DHT по префиксу
        # Пока используем известные shard_id
        
        return shards
    
    async def get_model_pipeline(
        self,
        model_name: str,
        prefer_local: bool = True,
    ) -> Optional[List[ModelShardInfo]]:
        """
        Получить полный pipeline для модели.
        
        [PIPELINE] Возвращает упорядоченный список shards,
        покрывающих все слои модели.
        
        Args:
            model_name: Имя модели
            prefer_local: Предпочитать локальные shards
        
        Returns:
            Список shards в порядке слоёв или None если недостаточно shards
        """
        model_info = await self.get_model_info(model_name)
        if not model_info:
            logger.error(f"[REGISTRY] Unknown model: {model_name}")
            return None
        
        # Собираем все доступные shards
        all_shards: Dict[str, List[ModelShardInfo]] = {}
        
        # Из DHT (упрощённо - ищем по известным shard_id)
        num_shards = model_info.recommended_shards
        shard_ranges = model_info.get_shard_layers(num_shards)
        
        for i, (start, end) in enumerate(shard_ranges):
            shard_id = f"layers_{start}_{end}"
            shards = await self.find_shards(model_name, start, end)
            
            if shards:
                all_shards[shard_id] = shards
        
        # Проверяем, есть ли все shards
        if len(all_shards) < num_shards:
            logger.warning(
                f"[REGISTRY] Incomplete pipeline for {model_name}: "
                f"found {len(all_shards)}/{num_shards} shards"
            )
            return None
        
        # Выбираем лучший узел для каждого shard
        pipeline = []
        for i, (start, end) in enumerate(shard_ranges):
            shard_id = f"layers_{start}_{end}"
            candidates = all_shards.get(shard_id, [])
            
            if not candidates:
                return None
            
            # Выбираем по: availability > trust > latency > load
            best = self._select_best_shard(candidates, prefer_local)
            pipeline.append(best)
        
        logger.info(
            f"[REGISTRY] Built pipeline for {model_name}: "
            f"{len(pipeline)} shards across {len(set(s.node_id for s in pipeline))} nodes"
        )
        
        return pipeline
    
    def _select_best_shard(
        self,
        candidates: List[ModelShardInfo],
        prefer_local: bool,
    ) -> ModelShardInfo:
        """Выбрать лучший shard из кандидатов."""
        # Фильтруем доступные
        available = [s for s in candidates if s.is_available]
        if not available:
            available = candidates  # Берём что есть
        
        # Сортируем по критериям
        def score(shard: ModelShardInfo) -> Tuple:
            return (
                shard.is_available,
                shard.trust_score,
                -shard.latency_ms,
                -shard.current_load,
            )
        
        available.sort(key=score, reverse=True)
        return available[0]
    
    async def update_shard_health(
        self,
        shard_id: str,
        status: ShardStatus,
        load: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Обновить health status shard.
        
        Args:
            shard_id: ID shard
            status: Новый статус
            load: Текущая нагрузка (0-1)
            latency_ms: Latency в мс
        """
        if shard_id in self._local_shards:
            shard = self._local_shards[shard_id]
            shard.status = status
            shard.current_load = load
            shard.latency_ms = latency_ms
            shard.last_health_check = time.time()
            
            # Обновляем в DHT
            if self.dht:
                await self.register_shard(shard)
    
    def get_local_shards(self) -> List[ModelShardInfo]:
        """Получить локальные shards."""
        return list(self._local_shards.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика реестра."""
        return {
            "local_shards": len(self._local_shards),
            "shards": [s.to_dict() for s in self._local_shards.values()],
            "dht_available": self.dht is not None,
        }

