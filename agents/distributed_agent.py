"""
Distributed LLM Agent - Агент для распределённого инференса
==========================================================

[AGENT] Предоставляет LLM сервис через распределённый pipeline:
- Автоматический поиск shards в сети
- Fallback на локальные ресурсы
- Биллинг по токенам

[PRICING]
- Distributed: 20 units per 1000 tokens (дешевле, используем чужие ресурсы)
- Local shard: 30 units per 1000 tokens (свои ресурсы)
- Ollama fallback: 50 units per 1000 tokens (полная нагрузка)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple

from .manager import BaseAgent

logger = logging.getLogger(__name__)


class DistributedLlmAgent(BaseAgent):
    """
    Агент для распределённого LLM инференса.
    
    [SERVICE] "llm_distributed"
    
    [FEATURES]
    - Pipeline parallelism через P2P сеть
    - Динамический выбор shards
    - Streaming output
    - Fallback на Ollama
    
    [USAGE]
    ```python
    agent = DistributedLlmAgent(
        distributed_client=client,
        default_model="qwen2.5-32b",
    )
    
    result, cost, error = await agent.execute({
        "prompt": "Explain distributed computing",
        "max_tokens": 100,
    })
    ```
    """
    
    service_name: str = "llm_distributed"
    
    # Pricing per 1000 tokens
    price_distributed: float = 20.0
    price_local: float = 30.0
    price_ollama: float = 50.0
    error_price: float = 2.0
    
    def __init__(
        self,
        distributed_client=None,
        default_model: str = "qwen2.5-32b",
        timeout: float = 300.0,
    ):
        """
        Args:
            distributed_client: DistributedInferenceClient instance
            default_model: Модель по умолчанию
            timeout: Таймаут в секундах
        """
        super().__init__()
        self.distributed_client = distributed_client
        self.default_model = default_model
        self.timeout = timeout
        
        # Статистика
        self._requests = 0
        self._distributed_count = 0
        self._local_count = 0
        self._ollama_count = 0
    
    @property
    def price_per_unit(self) -> float:
        """Средняя цена (будет варьироваться по результату)."""
        return self.price_distributed
    
    async def execute(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """
        Выполнить distributed inference.
        
        Args:
            payload: {
                "prompt": str,
                "model": str (optional),
                "max_tokens": int (optional),
                "temperature": float (optional),
                "top_p": float (optional),
                "system_prompt": str (optional),
            }
        
        Returns:
            (result, cost, error)
        """
        prompt = payload.get("prompt")
        if not prompt:
            return {}, self.error_price, "Prompt not provided"
        
        if not self.distributed_client:
            return {}, self.error_price, "Distributed client not configured"
        
        model = payload.get("model", self.default_model)
        max_tokens = payload.get("max_tokens", 100)
        temperature = payload.get("temperature", 0.7)
        top_p = payload.get("top_p", 0.9)
        system_prompt = payload.get("system_prompt")
        
        logger.info(
            f"[DIST_LLM] Request: model={model}, "
            f"prompt_len={len(prompt)}, max_tokens={max_tokens}"
        )
        
        try:
            # Запускаем инференс с таймаутом
            result = await asyncio.wait_for(
                self.distributed_client.generate(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    system_prompt=system_prompt,
                ),
                timeout=self.timeout,
            )
            
            self._requests += 1
            
            if not result.success:
                return {}, self.error_price, result.error
            
            # Вычисляем стоимость
            tokens = result.tokens_generated
            
            if result.distributed:
                cost = (tokens / 1000) * self.price_distributed
                self._distributed_count += 1
            elif "ollama" in result.model_used:
                cost = (tokens / 1000) * self.price_ollama
                self._ollama_count += 1
            else:
                cost = (tokens / 1000) * self.price_local
                self._local_count += 1
            
            # Минимальная стоимость
            cost = max(cost, 1.0)
            
            response = {
                "response": result.text,
                "model": result.model_used,
                "tokens_generated": result.tokens_generated,
                "tokens_per_second": result.tokens_per_second,
                "distributed": result.distributed,
                "shards_used": result.shards_used,
                "total_time_ms": result.total_time_ms,
            }
            
            logger.info(
                f"[DIST_LLM] Completed: {tokens} tokens, "
                f"{result.tokens_per_second:.1f} t/s, cost={cost:.2f}"
            )
            
            return response, cost, None
            
        except asyncio.TimeoutError:
            logger.warning(f"[DIST_LLM] Request timeout ({self.timeout}s)")
            return {}, self.error_price, f"Request timeout ({self.timeout}s)"
        except Exception as e:
            logger.error(f"[DIST_LLM] Error: {e}")
            return {}, self.error_price, str(e)
    
    async def check_models(self) -> Dict[str, Any]:
        """Проверить доступные модели."""
        if not self.distributed_client:
            return {"error": "Client not configured"}
        
        return await self.distributed_client.check_availability(self.default_model)
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика агента."""
        return {
            "service_name": self.service_name,
            "default_model": self.default_model,
            "requests": self._requests,
            "distributed_count": self._distributed_count,
            "local_count": self._local_count,
            "ollama_count": self._ollama_count,
            "client_stats": (
                self.distributed_client.get_stats() 
                if self.distributed_client else None
            ),
        }


class ShardProviderAgent(BaseAgent):
    """
    Агент для предоставления model shard в сеть.
    
    [SERVICE] "shard_provider"
    
    [PRICING]
    - Per forward pass based on tokens processed
    - Bonus for low latency
    
    [USAGE]
    Запускается на GPU узлах для участия в distributed inference.
    """
    
    service_name: str = "shard_provider"
    price_per_unit: float = 10.0  # Per 1000 tokens processed
    
    def __init__(
        self,
        worker=None,
        registry=None,
    ):
        """
        Args:
            worker: InferenceWorker instance
            registry: ModelRegistry for publishing shard
        """
        super().__init__()
        self.worker = worker
        self.registry = registry
    
    async def execute(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """
        Обработать forward request.
        
        Обычно вызывается через протокол, не напрямую как сервис.
        """
        if not self.worker or not self.worker.is_loaded:
            return {}, 0, "Worker not loaded"
        
        # Return worker stats
        return self.worker.get_stats(), 0, None
    
    async def load_shard(
        self,
        model_name: str,
        layer_start: int,
        layer_end: int,
        use_mock: bool = False,
    ) -> bool:
        """Загрузить shard."""
        if not self.worker:
            return False
        return await self.worker.load_shard(
            model_name, layer_start, layer_end, use_mock
        )
    
    async def unload_shard(self) -> None:
        """Выгрузить shard."""
        if self.worker:
            await self.worker.unload_shard()
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика."""
        return {
            "service_name": self.service_name,
            "worker": self.worker.get_stats() if self.worker else None,
        }

