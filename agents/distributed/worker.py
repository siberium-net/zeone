"""
Inference Worker - Сервис обработки на GPU узле
===============================================

[WORKER] Запускается на каждом GPU узле:
- Загружает и обслуживает model shard
- Обрабатывает ForwardRequest от других узлов
- Публикует health status в DHT
- Динамически переключает shards при необходимости

[QUEUE] Очередь запросов:
- Приоритизация по budget
- Batch обработка когда возможно
- Timeout для долгих запросов

[HEALTH] Мониторинг:
- GPU utilization
- Memory usage
- Queue length
- Latency
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable, Tuple
from collections import deque
import heapq

from .shard import ModelShard, create_shard, ActivationCache
from .protocol import (
    ForwardRequest,
    ForwardResponse,
    ShardHealthCheck,
    DistributedMessageType,
    pack_message,
)
from .registry import ModelRegistry, ModelShardInfo, ShardStatus, KNOWN_MODELS

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Состояние worker."""
    IDLE = auto()        # Ожидание
    LOADING = auto()     # Загрузка модели
    READY = auto()       # Готов к работе
    BUSY = auto()        # Обрабатывает запрос
    ERROR = auto()       # Ошибка
    SHUTTING_DOWN = auto()


@dataclass(order=True)
class QueuedRequest:
    """Запрос в очереди с приоритетом."""
    priority: float
    request: ForwardRequest = field(compare=False)
    future: asyncio.Future = field(compare=False)
    created_at: float = field(default_factory=time.time, compare=False)


class RequestQueue:
    """
    Приоритетная очередь запросов.
    
    [PRIORITY] Приоритет определяется:
    - Budget запроса (больше = выше приоритет)
    - Trust score отправителя
    - Время ожидания
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._heap: List[QueuedRequest] = []
        self._lock = asyncio.Lock()
    
    async def put(
        self,
        request: ForwardRequest,
        priority: float = 1.0,
    ) -> asyncio.Future:
        """
        Добавить запрос в очередь.
        
        Args:
            request: Forward запрос
            priority: Приоритет (больше = важнее)
        
        Returns:
            Future с результатом
        """
        async with self._lock:
            if len(self._heap) >= self.max_size:
                raise asyncio.QueueFull("Request queue is full")
            
            future = asyncio.get_event_loop().create_future()
            queued = QueuedRequest(
                priority=-priority,  # Min-heap, инвертируем для max
                request=request,
                future=future,
            )
            heapq.heappush(self._heap, queued)
            
            return future
    
    async def get(self) -> Optional[QueuedRequest]:
        """Получить следующий запрос."""
        async with self._lock:
            if self._heap:
                return heapq.heappop(self._heap)
            return None
    
    def __len__(self) -> int:
        return len(self._heap)
    
    @property
    def is_empty(self) -> bool:
        return len(self._heap) == 0


class InferenceWorker:
    """
    Worker для обработки distributed inference.
    
    [WORKER] Основной сервис на GPU узле:
    
    ```python
    worker = InferenceWorker(
        node_id=crypto.node_id,
        registry=model_registry,
    )
    
    # Загрузить shard
    await worker.load_shard("qwen2.5-32b", layer_start=0, layer_end=16)
    
    # Запустить обработку
    await worker.start()
    
    # Обработать запрос
    response = await worker.process(forward_request)
    ```
    
    [FEATURES]
    - Асинхронная очередь запросов
    - Health monitoring
    - Dynamic shard switching
    - Graceful shutdown
    """
    
    def __init__(
        self,
        node_id: str,
        registry: Optional[ModelRegistry] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_queue_size: int = 100,
        health_interval: float = 30.0,
        on_forward_to_next: Optional[Callable] = None,
    ):
        """
        Args:
            node_id: ID текущего узла
            registry: ModelRegistry для публикации shards
            host: Адрес для прослушивания
            port: Порт
            max_queue_size: Максимальный размер очереди
            health_interval: Интервал health check в секундах
            on_forward_to_next: Callback для передачи на следующий shard
        """
        self.node_id = node_id
        self.registry = registry
        self.host = host
        self.port = port
        self.health_interval = health_interval
        self.on_forward_to_next = on_forward_to_next
        
        self.state = WorkerState.IDLE
        self._shard: Optional[ModelShard] = None
        self._shard_info: Optional[ModelShardInfo] = None
        self._queue = RequestQueue(max_queue_size)
        
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        
        # Статистика
        self._stats = {
            "requests_processed": 0,
            "requests_failed": 0,
            "total_time_ms": 0.0,
            "last_request_time": 0.0,
        }
    
    @property
    def is_loaded(self) -> bool:
        return self._shard is not None and self._shard.is_loaded
    
    @property
    def current_shard_id(self) -> Optional[str]:
        return self._shard.shard_id if self._shard else None
    
    async def load_shard(
        self,
        model_name: str,
        layer_start: int,
        layer_end: int,
        use_mock: bool = False,
        **kwargs,
    ) -> bool:
        """
        Загрузить model shard.
        
        Args:
            model_name: Имя модели
            layer_start: Начальный слой
            layer_end: Конечный слой
            use_mock: Использовать mock shard
            **kwargs: Дополнительные параметры
        
        Returns:
            True если успешно загружен
        """
        if self._shard and self._shard.is_loaded:
            logger.info("[WORKER] Unloading current shard")
            await self.unload_shard()
        
        self.state = WorkerState.LOADING
        
        try:
            # Получаем информацию о модели
            model_info = None
            if self.registry:
                model_info = await self.registry.get_model_info(model_name)
            
            if not model_info and model_name in KNOWN_MODELS:
                model_info = KNOWN_MODELS[model_name]
            
            # Создаём shard
            self._shard = create_shard(
                model_name=model_name,
                layer_start=layer_start,
                layer_end=layer_end,
                use_mock=use_mock,
                hidden_size=model_info.hidden_size if model_info else 5120,
                vocab_size=model_info.vocab_size if model_info else 152064,
                total_layers=model_info.total_layers if model_info else 64,
                **kwargs,
            )
            
            # Загружаем веса
            success = await self._shard.load()
            
            if success:
                self.state = WorkerState.READY
                
                # Создаём ShardInfo
                self._shard_info = ModelShardInfo(
                    shard_id=self._shard.shard_id,
                    model_name=model_name,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    node_id=self.node_id,
                    host=self.host,
                    port=self.port,
                    status=ShardStatus.ONLINE,
                    gpu_memory_mb=self._shard.get_memory_usage(),
                )
                
                # Регистрируем в DHT
                if self.registry:
                    await self.registry.register_shard(self._shard_info)
                
                logger.info(
                    f"[WORKER] Loaded shard: {model_name} "
                    f"layers {layer_start}-{layer_end}"
                )
                return True
            else:
                self.state = WorkerState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"[WORKER] Failed to load shard: {e}")
            self.state = WorkerState.ERROR
            return False
    
    async def unload_shard(self) -> None:
        """Выгрузить текущий shard."""
        if self._shard:
            await self._shard.unload()
            
            # Обновляем статус в registry
            if self.registry and self._shard_info:
                self._shard_info.status = ShardStatus.OFFLINE
                await self.registry.register_shard(self._shard_info)
            
            self._shard = None
            self._shard_info = None
        
        self.state = WorkerState.IDLE
    
    async def start(self) -> None:
        """Запустить worker."""
        if self._running:
            return
        
        self._running = True
        
        # Запускаем обработку очереди
        self._process_task = asyncio.create_task(self._process_loop())
        
        # Запускаем health monitoring
        self._health_task = asyncio.create_task(self._health_loop())
        
        logger.info("[WORKER] Started")
    
    async def stop(self) -> None:
        """Остановить worker."""
        self._running = False
        self.state = WorkerState.SHUTTING_DOWN
        
        # Отменяем задачи
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        # Выгружаем shard
        await self.unload_shard()
        
        logger.info("[WORKER] Stopped")
    
    async def process(
        self,
        request: ForwardRequest,
        priority: float = 1.0,
    ) -> ForwardResponse:
        """
        Обработать ForwardRequest.
        
        Args:
            request: Запрос на forward
            priority: Приоритет запроса
        
        Returns:
            ForwardResponse
        """
        if not self.is_loaded:
            return ForwardResponse(
                request_id=request.request_id,
                success=False,
                error="Shard not loaded",
            )
        
        # Добавляем в очередь
        try:
            future = await self._queue.put(request, priority)
            
            # Ждём результат с timeout
            response = await asyncio.wait_for(future, timeout=300.0)
            return response
            
        except asyncio.QueueFull:
            return ForwardResponse(
                request_id=request.request_id,
                success=False,
                error="Worker queue is full",
            )
        except asyncio.TimeoutError:
            return ForwardResponse(
                request_id=request.request_id,
                success=False,
                error="Request timeout",
            )
    
    async def _process_loop(self) -> None:
        """Цикл обработки очереди."""
        while self._running:
            try:
                # Получаем следующий запрос
                queued = await self._queue.get()
                
                if queued is None:
                    await asyncio.sleep(0.01)
                    continue
                
                self.state = WorkerState.BUSY
                
                # Обновляем статус
                if self._shard_info:
                    self._shard_info.status = ShardStatus.BUSY
                    self._shard_info.current_load = len(self._queue) / self._queue.max_size
                
                # Обрабатываем запрос
                try:
                    response = await self._shard.process_request(queued.request)
                    
                    self._stats["requests_processed"] += 1
                    self._stats["total_time_ms"] += response.processing_time_ms
                    self._stats["last_request_time"] = time.time()
                    
                    # Если есть следующий shard, передаём туда
                    if (response.success and 
                        not queued.request.is_last and 
                        self.on_forward_to_next):
                        await self.on_forward_to_next(response)
                    
                except Exception as e:
                    logger.error(f"[WORKER] Process failed: {e}")
                    response = ForwardResponse(
                        request_id=queued.request.request_id,
                        success=False,
                        error=str(e),
                    )
                    self._stats["requests_failed"] += 1
                
                # Возвращаем результат
                if not queued.future.done():
                    queued.future.set_result(response)
                
                self.state = WorkerState.READY
                if self._shard_info:
                    self._shard_info.status = ShardStatus.ONLINE
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WORKER] Loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _health_loop(self) -> None:
        """Цикл обновления health status."""
        while self._running:
            try:
                await asyncio.sleep(self.health_interval)
                
                if not self._shard_info or not self.registry:
                    continue
                
                # Обновляем метрики
                self._shard_info.gpu_memory_mb = (
                    self._shard.get_memory_usage() if self._shard else 0
                )
                self._shard_info.current_load = len(self._queue) / self._queue.max_size
                
                if self._stats["requests_processed"] > 0:
                    self._shard_info.latency_ms = (
                        self._stats["total_time_ms"] / 
                        self._stats["requests_processed"]
                    )
                
                # Публикуем в DHT
                await self.registry.update_shard_health(
                    self._shard_info.shard_id,
                    self._shard_info.status,
                    self._shard_info.current_load,
                    self._shard_info.latency_ms,
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[WORKER] Health loop error: {e}")
    
    def get_health(self) -> ShardHealthCheck:
        """Получить текущий health status."""
        rps = 0.0
        if self._stats["last_request_time"] > 0:
            elapsed = time.time() - self._stats["last_request_time"]
            if elapsed > 0 and elapsed < 60:
                rps = self._stats["requests_processed"] / elapsed
        
        avg_latency = 0.0
        if self._stats["requests_processed"] > 0:
            avg_latency = (
                self._stats["total_time_ms"] / 
                self._stats["requests_processed"]
            )
        
        return ShardHealthCheck(
            shard_id=self.current_shard_id or "",
            node_id=self.node_id,
            is_healthy=self.state in (WorkerState.READY, WorkerState.BUSY),
            current_load=len(self._queue) / self._queue.max_size,
            queue_size=len(self._queue),
            gpu_memory_used_mb=self._shard.get_memory_usage() if self._shard else 0,
            gpu_memory_total_mb=24000,  # TODO: Получать реальное значение
            requests_per_second=rps,
            avg_latency_ms=avg_latency,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Полная статистика worker."""
        return {
            "node_id": self.node_id[:16] + "...",
            "state": self.state.name,
            "shard": self._shard.get_stats() if self._shard else None,
            "queue_size": len(self._queue),
            "stats": self._stats.copy(),
            "health": self.get_health().to_dict(),
        }

