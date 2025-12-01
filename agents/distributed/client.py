"""
Distributed Inference Client - Клиент для распределённого инференса
===================================================================

[CLIENT] Высокоуровневый API для пользователей:
- Простой интерфейс generate(prompt) -> response
- Автоматическая токенизация/детокенизация
- Streaming output
- Fallback на локальный Ollama если нет shards

[USAGE]
```python
client = DistributedInferenceClient(
    coordinator=pipeline_coordinator,
    registry=model_registry,
)

# Простой запрос
response = await client.generate(
    model="qwen2.5-32b",
    prompt="Explain quantum computing",
    max_tokens=100,
)

# Streaming
async for token in client.stream(
    model="qwen2.5-32b",
    prompt="Write a story",
):
    print(token, end="", flush=True)
```
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, AsyncIterator, Callable
import numpy as np

from .pipeline import PipelineCoordinator, PipelineResult
from .registry import ModelRegistry, KNOWN_MODELS
from .worker import InferenceWorker

logger = logging.getLogger(__name__)


# Простой tokenizer (для demo без зависимости от transformers)
# В production использовать AutoTokenizer
class SimpleTokenizer:
    """
    Упрощённый tokenizer для демонстрации.
    
    [NOTE] В production использовать:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    """
    
    def __init__(self, vocab_size: int = 152064):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        # Простая char-level токенизация
        self._char_to_id = {}
        self._id_to_char = {}
        
        # Специальные токены
        self._id_to_char[0] = "<pad>"
        self._id_to_char[1] = "<bos>"
        self._id_to_char[2] = "<eos>"
        self._id_to_char[3] = "<unk>"
        
        # ASCII chars starting from id 4
        for i in range(128):
            self._char_to_id[chr(i)] = i + 4
            self._id_to_char[i + 4] = chr(i)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to token IDs."""
        ids = []
        
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        for char in text:
            ids.append(self._char_to_id.get(char, self.unk_token_id))
        
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        chars = []
        
        for token_id in ids:
            if skip_special_tokens and token_id in (0, 1, 2, 3):
                continue
            
            char = self._id_to_char.get(token_id, "")
            chars.append(char)
        
        return "".join(chars)


@dataclass
class InferenceResult:
    """
    Результат инференса.
    """
    success: bool
    text: str = ""
    tokens_generated: int = 0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    model_used: str = ""
    distributed: bool = False
    shards_used: int = 0
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "total_time_ms": self.total_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "model_used": self.model_used,
            "distributed": self.distributed,
            "shards_used": self.shards_used,
            "error": self.error,
        }


class DistributedInferenceClient:
    """
    Клиент для распределённого инференса.
    
    [CLIENT] Предоставляет простой API:
    
    ```python
    client = DistributedInferenceClient(coordinator, registry)
    
    # Генерация текста
    result = await client.generate("qwen2.5-32b", "Hello, how are you?")
    print(result.text)
    
    # Streaming
    async for token in client.stream("qwen2.5-32b", "Write code:"):
        print(token, end="")
    ```
    
    [FALLBACK]
    Если distributed pipeline недоступен:
    1. Проверяем локальный worker
    2. Пробуем Ollama
    3. Возвращаем ошибку
    """
    
    def __init__(
        self,
        coordinator: Optional[PipelineCoordinator] = None,
        registry: Optional[ModelRegistry] = None,
        local_worker: Optional[InferenceWorker] = None,
        ollama_fallback: bool = True,
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
    ):
        """
        Args:
            coordinator: Pipeline coordinator
            registry: Model registry
            local_worker: Локальный inference worker
            ollama_fallback: Использовать Ollama как fallback
            ollama_host: Хост Ollama
            ollama_port: Порт Ollama
        """
        self.coordinator = coordinator
        self.registry = registry
        self.local_worker = local_worker
        self.ollama_fallback = ollama_fallback
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        
        self._tokenizer = SimpleTokenizer()
        
        # Статистика
        self._stats = {
            "distributed_requests": 0,
            "local_requests": 0,
            "ollama_requests": 0,
            "failed_requests": 0,
        }
    
    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
    ) -> InferenceResult:
        """
        Сгенерировать текст.
        
        Args:
            model: Имя модели
            prompt: Входной текст
            max_tokens: Максимум токенов
            temperature: Температура
            top_p: Top-p sampling
            system_prompt: Системный промпт
        
        Returns:
            InferenceResult
        """
        start_time = time.time()
        full_prompt = prompt
        
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        # Пробуем distributed
        if self.coordinator and self.registry:
            result = await self._try_distributed(
                model, full_prompt, max_tokens, temperature, top_p
            )
            if result.success:
                result.total_time_ms = (time.time() - start_time) * 1000
                if result.tokens_generated > 0:
                    result.tokens_per_second = (
                        result.tokens_generated / (result.total_time_ms / 1000)
                    )
                self._stats["distributed_requests"] += 1
                return result
        
        # Пробуем локальный worker
        if self.local_worker and self.local_worker.is_loaded:
            result = await self._try_local_worker(
                model, full_prompt, max_tokens, temperature
            )
            if result.success:
                result.total_time_ms = (time.time() - start_time) * 1000
                self._stats["local_requests"] += 1
                return result
        
        # Fallback на Ollama
        if self.ollama_fallback:
            result = await self._try_ollama(
                model, full_prompt, max_tokens, temperature
            )
            if result.success:
                result.total_time_ms = (time.time() - start_time) * 1000
                self._stats["ollama_requests"] += 1
                return result
        
        self._stats["failed_requests"] += 1
        return InferenceResult(
            success=False,
            error="No inference backend available",
            total_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        """
        Streaming генерация.
        
        Yields:
            Токены по мере генерации
        """
        # Для distributed используем callback
        if self.coordinator and self.registry:
            tokens_queue: asyncio.Queue = asyncio.Queue()
            
            async def on_token(token_id: int, step: int):
                token_str = self._tokenizer.decode([token_id])
                await tokens_queue.put(token_str)
            
            # Запускаем генерацию в background
            task = asyncio.create_task(
                self._run_distributed_with_callback(
                    model, prompt, max_tokens, temperature, top_p, on_token
                )
            )
            
            try:
                while not task.done():
                    try:
                        token = await asyncio.wait_for(
                            tokens_queue.get(),
                            timeout=0.1,
                        )
                        yield token
                    except asyncio.TimeoutError:
                        continue
                
                # Собираем оставшиеся токены
                while not tokens_queue.empty():
                    yield await tokens_queue.get()
                    
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        # Fallback: non-streaming
        else:
            result = await self.generate(
                model, prompt, max_tokens, temperature, top_p
            )
            if result.success:
                yield result.text
    
    async def _try_distributed(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> InferenceResult:
        """Попытка distributed inference."""
        try:
            # Проверяем доступность pipeline
            pipeline = await self.registry.get_model_pipeline(model)
            if not pipeline:
                return InferenceResult(
                    success=False,
                    error="No shards available for model",
                )
            
            # Токенизируем
            input_ids = np.array(
                [self._tokenizer.encode(prompt)],
                dtype=np.int64,
            )
            
            # Запускаем pipeline
            result = await self.coordinator.run(
                model_name=model,
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            if result.success:
                # Декодируем
                text = self._tokenizer.decode(
                    result.output_ids.tolist(),
                    skip_special_tokens=True,
                )
                
                return InferenceResult(
                    success=True,
                    text=text,
                    tokens_generated=result.tokens_generated,
                    model_used=model,
                    distributed=True,
                    shards_used=result.total_hops,
                )
            else:
                return InferenceResult(
                    success=False,
                    error=result.error,
                )
                
        except Exception as e:
            logger.error(f"[CLIENT] Distributed inference failed: {e}")
            return InferenceResult(
                success=False,
                error=str(e),
            )
    
    async def _run_distributed_with_callback(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        on_token: Callable,
    ) -> None:
        """Run distributed with token callback."""
        input_ids = np.array(
            [self._tokenizer.encode(prompt)],
            dtype=np.int64,
        )
        
        await self.coordinator.run(
            model_name=model,
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            on_token=on_token,
        )
    
    async def _try_local_worker(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> InferenceResult:
        """Попытка через локальный worker."""
        try:
            # Создаём mock forward request
            input_ids = np.array(
                [self._tokenizer.encode(prompt)],
                dtype=np.float32,
            )
            
            from .protocol import (
                ForwardRequest,
                quantize_activations,
                create_request_id,
            )
            
            request = ForwardRequest(
                request_id=create_request_id("local", 0),
                shard_id=self.local_worker.current_shard_id or "",
                activations=quantize_activations(input_ids),
                layer_idx=0,
                is_first=True,
                is_last=True,
            )
            
            response = await self.local_worker.process(request)
            
            if response.success and response.logits:
                # Simple greedy decode
                logits = np.frombuffer(response.logits, dtype=np.float32)
                token_id = int(np.argmax(logits))
                text = self._tokenizer.decode([token_id])
                
                return InferenceResult(
                    success=True,
                    text=text,
                    tokens_generated=1,
                    model_used=model,
                    distributed=False,
                )
            
            return InferenceResult(
                success=False,
                error=response.error or "Local worker failed",
            )
            
        except Exception as e:
            return InferenceResult(
                success=False,
                error=str(e),
            )
    
    async def _try_ollama(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> InferenceResult:
        """Fallback на Ollama."""
        try:
            import aiohttp
            
            # Map model names to Ollama models
            ollama_model = self._map_to_ollama(model)
            
            url = f"http://{self.ollama_host}:{self.ollama_port}/api/generate"
            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        return InferenceResult(
                            success=True,
                            text=data.get("response", ""),
                            tokens_generated=data.get("eval_count", 0),
                            model_used=f"ollama:{ollama_model}",
                            distributed=False,
                        )
                    else:
                        return InferenceResult(
                            success=False,
                            error=f"Ollama error: {resp.status}",
                        )
                        
        except Exception as e:
            return InferenceResult(
                success=False,
                error=f"Ollama fallback failed: {e}",
            )
    
    def _map_to_ollama(self, model: str) -> str:
        """Map distributed model name to Ollama model."""
        mappings = {
            "qwen2.5-32b": "qwen2.5:32b",
            "qwen2.5-7b": "qwen2.5:7b",
            "llama2-70b": "llama2:70b",
            "llama2-7b": "llama2:7b",
            "mixtral-8x7b": "mixtral:8x7b",
        }
        return mappings.get(model, model)
    
    async def check_availability(self, model: str) -> Dict[str, Any]:
        """
        Проверить доступность модели.
        
        Returns:
            Словарь с информацией о доступности
        """
        result = {
            "model": model,
            "distributed_available": False,
            "local_available": False,
            "ollama_available": False,
            "shards": [],
        }
        
        # Distributed
        if self.registry:
            pipeline = await self.registry.get_model_pipeline(model)
            if pipeline:
                result["distributed_available"] = True
                result["shards"] = [
                    {
                        "shard_id": s.shard_id,
                        "node_id": s.node_id[:16] + "...",
                        "layers": f"{s.layer_start}-{s.layer_end}",
                        "status": s.status.value,
                    }
                    for s in pipeline
                ]
        
        # Local
        if self.local_worker and self.local_worker.is_loaded:
            result["local_available"] = True
        
        # Ollama
        if self.ollama_fallback:
            try:
                import aiohttp
                url = f"http://{self.ollama_host}:{self.ollama_port}/api/tags"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            result["ollama_available"] = True
            except:
                pass
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика клиента."""
        return {
            "requests": self._stats.copy(),
            "coordinator": (
                self.coordinator.get_stats() if self.coordinator else None
            ),
            "local_worker": (
                self.local_worker.get_stats() if self.local_worker else None
            ),
        }


# Convenience function
async def distributed_generate(
    prompt: str,
    model: str = "qwen2.5-32b",
    coordinator: Optional[PipelineCoordinator] = None,
    registry: Optional[ModelRegistry] = None,
    **kwargs,
) -> InferenceResult:
    """
    Convenience function для быстрой генерации.
    
    ```python
    result = await distributed_generate(
        "Explain AI",
        model="qwen2.5-32b",
        coordinator=coordinator,
        registry=registry,
    )
    ```
    """
    client = DistributedInferenceClient(
        coordinator=coordinator,
        registry=registry,
    )
    return await client.generate(model, prompt, **kwargs)

