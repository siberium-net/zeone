"""
Pipeline Coordinator - Координация распределённого инференса
============================================================

[PIPELINE] Управляет потоком данных между shards:
1. Получает prompt от клиента
2. Строит route через shards (из Registry)
3. Передаёт активации последовательно через все shards
4. Возвращает результат клиенту

[FAULT TOLERANCE]
- Если узел недоступен → переключение на резервный
- Retry с exponential backoff
- Timeout на каждый hop

[OPTIMIZATION]
- Prefetch: начинаем следующий токен пока текущий обрабатывается
- Batching: объединяем запросы для одного shard
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable, Tuple
import hashlib
import numpy as np

from .registry import ModelRegistry, ModelShardInfo, ShardStatus
from .protocol import (
    ForwardRequest,
    ForwardResponse,
    QuantizedActivations,
    quantize_activations,
    dequantize_activations,
    create_request_id,
)

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Состояние pipeline."""
    IDLE = auto()
    BUILDING = auto()      # Построение маршрута
    RUNNING = auto()       # Выполнение
    RECOVERING = auto()    # Восстановление после сбоя
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class PipelineHop:
    """
    Один hop в pipeline.
    
    [HOP] Содержит информацию о shard и резервных узлах.
    """
    shard_info: ModelShardInfo
    backup_shards: List[ModelShardInfo] = field(default_factory=list)
    attempts: int = 0
    last_error: str = ""
    latency_ms: float = 0.0


@dataclass
class PipelineRequest:
    """
    Запрос на distributed inference.
    """
    request_id: str
    model_name: str
    input_ids: np.ndarray  # [batch, seq_len] token IDs
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    created_at: float = field(default_factory=time.time)
    
    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]
    
    @property
    def seq_length(self) -> int:
        return self.input_ids.shape[1]


@dataclass
class PipelineResult:
    """
    Результат distributed inference.
    """
    request_id: str
    success: bool
    output_ids: Optional[np.ndarray] = None  # Generated tokens
    logits: Optional[np.ndarray] = None      # Raw logits
    total_time_ms: float = 0.0
    hops_completed: int = 0
    total_hops: int = 0
    tokens_generated: int = 0
    error: str = ""


class PipelineCoordinator:
    """
    Координатор распределённого inference pipeline.
    
    [COORDINATOR] Управляет всем процессом:
    
    ```python
    coordinator = PipelineCoordinator(
        node_id=crypto.node_id,
        registry=model_registry,
        send_forward=node.send_forward_request,
    )
    
    # Выполнить inference
    result = await coordinator.run(
        model_name="qwen2.5-32b",
        input_ids=tokenized_input,
        max_new_tokens=100,
    )
    ```
    
    [FEATURES]
    - Автоматическое построение pipeline
    - Fault tolerance с резервными узлами
    - Streaming generation (token by token)
    - Batching запросов
    """
    
    def __init__(
        self,
        node_id: str,
        registry: ModelRegistry,
        send_forward: Optional[Callable] = None,
        max_retries: int = 3,
        hop_timeout: float = 60.0,
        prefer_local: bool = True,
    ):
        """
        Args:
            node_id: ID текущего узла
            registry: ModelRegistry для поиска shards
            send_forward: Callback для отправки ForwardRequest
            max_retries: Максимум попыток на hop
            hop_timeout: Timeout на один hop в секундах
            prefer_local: Предпочитать локальные shards
        """
        self.node_id = node_id
        self.registry = registry
        self.send_forward = send_forward
        self.max_retries = max_retries
        self.hop_timeout = hop_timeout
        self.prefer_local = prefer_local
        
        self._state = PipelineState.IDLE
        self._active_pipelines: Dict[str, "ActivePipeline"] = {}
        
        # Pending responses (request_id -> Future)
        self._pending_responses: Dict[str, asyncio.Future] = {}
    
    @property
    def state(self) -> PipelineState:
        return self._state
    
    async def run(
        self,
        model_name: str,
        input_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        on_token: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Запустить distributed inference.
        
        Args:
            model_name: Имя модели
            input_ids: Токенизированный вход [batch, seq_len]
            max_new_tokens: Максимум новых токенов
            temperature: Температура сэмплирования
            top_p: Top-p sampling
            do_sample: Использовать сэмплирование
            on_token: Callback для каждого нового токена
        
        Returns:
            PipelineResult с результатом
        """
        start_time = time.time()
        request_id = create_request_id(self.node_id, len(self._active_pipelines))
        
        request = PipelineRequest(
            request_id=request_id,
            model_name=model_name,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        
        logger.info(
            f"[PIPELINE] Starting inference: {model_name}, "
            f"seq_len={request.seq_length}, max_tokens={max_new_tokens}"
        )
        
        try:
            # 1. Строим pipeline
            self._state = PipelineState.BUILDING
            pipeline = await self._build_pipeline(model_name)
            
            if not pipeline:
                return PipelineResult(
                    request_id=request_id,
                    success=False,
                    error="Failed to build pipeline: not enough shards available",
                    total_time_ms=(time.time() - start_time) * 1000,
                )
            
            logger.info(
                f"[PIPELINE] Built pipeline with {len(pipeline)} hops: "
                f"{' -> '.join(h.shard_info.node_id[:8] + '...' for h in pipeline)}"
            )
            
            # 2. Создаём active pipeline
            active = ActivePipeline(
                request=request,
                hops=pipeline,
                on_token=on_token,
            )
            self._active_pipelines[request_id] = active
            
            # 3. Запускаем generation loop
            self._state = PipelineState.RUNNING
            result = await self._generation_loop(active)
            
            result.total_time_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"[PIPELINE] Error: {e}")
            return PipelineResult(
                request_id=request_id,
                success=False,
                error=str(e),
                total_time_ms=(time.time() - start_time) * 1000,
            )
        finally:
            self._active_pipelines.pop(request_id, None)
            self._state = PipelineState.IDLE
    
    async def _build_pipeline(
        self,
        model_name: str,
    ) -> Optional[List[PipelineHop]]:
        """
        Построить pipeline для модели.
        
        Returns:
            Список hops или None если невозможно
        """
        # Получаем pipeline из registry
        shards = await self.registry.get_model_pipeline(
            model_name,
            prefer_local=self.prefer_local,
        )
        
        if not shards:
            return None
        
        # Создаём hops
        hops = []
        for shard in shards:
            # Ищем backup shards
            backups = await self.registry.find_shards(
                model_name,
                shard.layer_start,
                shard.layer_end,
            )
            # Убираем основной из backups
            backups = [s for s in backups if s.node_id != shard.node_id]
            
            hop = PipelineHop(
                shard_info=shard,
                backup_shards=backups[:2],  # Макс 2 backup
            )
            hops.append(hop)
        
        return hops
    
    async def _generation_loop(
        self,
        active: "ActivePipeline",
    ) -> PipelineResult:
        """
        Основной цикл генерации.
        
        [GENERATION] Autoregressive generation:
        1. Forward через все shards
        2. Sample next token
        3. Append to sequence
        4. Repeat until EOS or max_tokens
        """
        request = active.request
        current_ids = request.input_ids.copy()
        generated_tokens = []
        
        for step in range(request.max_new_tokens):
            # Forward через весь pipeline
            logits = await self._forward_pipeline(
                active,
                current_ids,
                is_first_step=(step == 0),
            )
            
            if logits is None:
                return PipelineResult(
                    request_id=request.request_id,
                    success=False,
                    error=f"Forward failed at step {step}",
                    hops_completed=active.completed_hops,
                    total_hops=len(active.hops),
                )
            
            # Sample next token
            next_token = self._sample_token(
                logits,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
            )
            
            generated_tokens.append(next_token)
            
            # Callback
            if active.on_token:
                try:
                    await active.on_token(next_token, step)
                except Exception as e:
                    logger.warning(f"[PIPELINE] Token callback error: {e}")
            
            # Check EOS
            if next_token == 2:  # Typical EOS token
                break
            
            # Append to sequence
            current_ids = np.concatenate([
                current_ids,
                np.array([[next_token]], dtype=np.int64),
            ], axis=1)
        
        output_ids = np.array(generated_tokens, dtype=np.int64)
        
        return PipelineResult(
            request_id=request.request_id,
            success=True,
            output_ids=output_ids,
            hops_completed=active.completed_hops,
            total_hops=len(active.hops),
            tokens_generated=len(generated_tokens),
        )
    
    async def _forward_pipeline(
        self,
        active: "ActivePipeline",
        input_ids: np.ndarray,
        is_first_step: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Forward через весь pipeline.
        
        Returns:
            Logits или None если failed
        """
        # Начальные активации = input_ids
        current_activations = input_ids.astype(np.float32)
        
        for hop_idx, hop in enumerate(active.hops):
            is_first = (hop_idx == 0)
            is_last = (hop_idx == len(active.hops) - 1)
            
            # Forward через один hop с retry
            result = await self._forward_hop(
                hop=hop,
                activations=current_activations,
                layer_idx=hop.shard_info.layer_start,
                request_id=active.request.request_id,
                is_first=is_first,
                is_last=is_last,
            )
            
            if result is None:
                return None
            
            current_activations = result
            active.completed_hops = hop_idx + 1
        
        return current_activations
    
    async def _forward_hop(
        self,
        hop: PipelineHop,
        activations: np.ndarray,
        layer_idx: int,
        request_id: str,
        is_first: bool,
        is_last: bool,
    ) -> Optional[np.ndarray]:
        """
        Forward через один hop с fault tolerance.
        
        [FAULT TOLERANCE]
        1. Пробуем основной узел
        2. При ошибке → retry
        3. После max_retries → backup узел
        """
        shards_to_try = [hop.shard_info] + hop.backup_shards
        
        for shard in shards_to_try:
            for attempt in range(self.max_retries):
                try:
                    hop.attempts += 1
                    start_time = time.time()
                    
                    # Квантуем активации
                    quantized = quantize_activations(activations)
                    
                    # Создаём запрос
                    forward_request = ForwardRequest(
                        request_id=request_id,
                        shard_id=shard.shard_id,
                        activations=quantized,
                        layer_idx=layer_idx,
                        is_first=is_first,
                        is_last=is_last,
                    )
                    
                    # Отправляем
                    response = await self._send_and_wait(
                        shard=shard,
                        request=forward_request,
                    )
                    
                    hop.latency_ms = (time.time() - start_time) * 1000
                    
                    if response.success:
                        # Деквантуем результат
                        if is_last:
                            # Logits уже в float32
                            return np.frombuffer(
                                response.logits,
                                dtype=np.float32,
                            ).reshape(-1)
                        else:
                            return dequantize_activations(
                                response.activations,
                                dtype="float32",
                            )
                    else:
                        hop.last_error = response.error
                        logger.warning(
                            f"[PIPELINE] Hop failed: {shard.node_id[:8]}... "
                            f"attempt {attempt+1}/{self.max_retries}: {response.error}"
                        )
                    
                except asyncio.TimeoutError:
                    hop.last_error = "Timeout"
                    logger.warning(
                        f"[PIPELINE] Hop timeout: {shard.node_id[:8]}... "
                        f"attempt {attempt+1}/{self.max_retries}"
                    )
                except Exception as e:
                    hop.last_error = str(e)
                    logger.error(f"[PIPELINE] Hop error: {e}")
                
                # Exponential backoff
                await asyncio.sleep(0.1 * (2 ** attempt))
            
            # Все попытки исчерпаны для этого shard
            logger.warning(
                f"[PIPELINE] Shard exhausted: {shard.node_id[:8]}..., "
                f"trying backup"
            )
        
        # Все shards failed
        logger.error(
            f"[PIPELINE] All shards failed for layers "
            f"{hop.shard_info.layer_start}-{hop.shard_info.layer_end}"
        )
        return None
    
    async def _send_and_wait(
        self,
        shard: ModelShardInfo,
        request: ForwardRequest,
    ) -> ForwardResponse:
        """
        Отправить запрос и дождаться ответа.
        """
        if not self.send_forward:
            raise RuntimeError("send_forward callback not configured")
        
        # Создаём future для ответа
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[request.request_id] = future
        
        try:
            # Отправляем запрос
            await self.send_forward(shard, request)
            
            # Ждём ответ
            response = await asyncio.wait_for(
                future,
                timeout=self.hop_timeout,
            )
            
            return response
            
        finally:
            self._pending_responses.pop(request.request_id, None)
    
    def handle_response(self, response: ForwardResponse) -> None:
        """
        Обработать входящий ForwardResponse.
        
        Вызывается из Node при получении ответа.
        """
        future = self._pending_responses.get(response.request_id)
        if future and not future.done():
            future.set_result(response)
    
    def _sample_token(
        self,
        logits: np.ndarray,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> int:
        """
        Sample next token from logits.
        
        [SAMPLING]
        - Greedy: argmax
        - Temperature: scale logits
        - Top-p: nucleus sampling
        """
        if not do_sample:
            return int(np.argmax(logits))
        
        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Top-p filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            
            # Find cutoff
            cutoff_idx = np.searchsorted(cumsum, top_p)
            mask = np.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx + 1]] = 1
            
            probs = probs * mask
            probs = probs / np.sum(probs)
        
        # Sample
        return int(np.random.choice(len(probs), p=probs))
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика coordinator."""
        return {
            "state": self._state.name,
            "active_pipelines": len(self._active_pipelines),
            "pending_responses": len(self._pending_responses),
        }


@dataclass
class ActivePipeline:
    """
    Активный pipeline.
    """
    request: PipelineRequest
    hops: List[PipelineHop]
    on_token: Optional[Callable] = None
    completed_hops: int = 0
    generated_tokens: List[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

