"""
Local LLM Agent - Ollama (GPU Inference)
========================================

[MARKET] Услуга "llm_local":
- Запускает инференс на локальном Ollama
- Работает с GPU (RTX 3090 и др.)
- Дешевле облака, но нагружает железо

[ECONOMY] Цена: 30 единиц за успешный запрос
- Дешевле облачного API
- При ошибке деньги не списываются

[CONFIG] Параметры:
- OLLAMA_HOST: хост Ollama (по умолчанию localhost)
- OLLAMA_PORT: порт Ollama (по умолчанию 11434)
- OLLAMA_MODEL: модель по умолчанию (по умолчанию qwen2.5:32b)
"""

import asyncio
import os
import logging
from typing import Any, Tuple, Optional, Dict, List

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from .manager import BaseAgent

logger = logging.getLogger(__name__)


# Конфигурация по умолчанию
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 11434
DEFAULT_MODEL = "qwen3:32b"
REQUEST_TIMEOUT = 300  # 5 минут для больших моделей
ERROR_PRICE = 0  # При ошибке деньги не списываются


class OllamaAgent(BaseAgent):
    """
    Агент локального LLM через Ollama.
    
    [MARKET] Услуга "llm_local":
    - Принимает промпт
    - Запускает инференс на локальном GPU
    - Возвращает текстовый ответ
    
    [ECONOMY] Ценообразование:
    - Успешный запрос: 30 единиц
    - Ошибка: 0 единиц (деньги не списываются)
    
    [CONFIG] Переменные окружения:
    - OLLAMA_HOST: хост (по умолчанию localhost)
    - OLLAMA_PORT: порт (по умолчанию 11434)
    - OLLAMA_MODEL: модель (по умолчанию qwen2.5:32b)
    
    [EXAMPLE]
    Request: {"prompt": "В чем смысл жизни?"}
    или
    Request: "В чем смысл жизни?"
    
    Response: {
        "response": "Смысл жизни в...",
        "model": "qwen2.5:32b",
        "eval_count": 150
    }
    """
    
    SUCCESS_PRICE = 30  # Цена за успешный запрос
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        """
        Инициализация агента.
        
        Args:
            host: Хост Ollama (или из OLLAMA_HOST)
            port: Порт Ollama (или из OLLAMA_PORT)
            model_name: Модель по умолчанию (или из OLLAMA_MODEL)
        """
        self.host = host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)
        self.port = port or int(os.getenv("OLLAMA_PORT", str(DEFAULT_PORT)))
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        
        self.base_url = f"http://{self.host}:{self.port}"
        
        logger.info(f"[OLLAMA_AGENT] Configured: {self.base_url}, model: {self.model_name}")
    
    @property
    def service_name(self) -> str:
        return "llm_local"
    
    @property
    def price_per_unit(self) -> float:
        return self.SUCCESS_PRICE
    
    @property
    def description(self) -> str:
        return (
            f"Local LLM (Ollama): {self.model_name} on {self.base_url}. "
            f"Price: {self.SUCCESS_PRICE} on success, {ERROR_PRICE} on error."
        )
    
    async def execute(self, payload: Any) -> Tuple[Any, float]:
        """
        Выполнить запрос к Ollama.
        
        Args:
            payload: Строка (промпт) или dict с ключом "prompt"
        
        Returns:
            (result, units) - результат и количество единиц работы
        """
        if not AIOHTTP_AVAILABLE:
            return {
                "error": "aiohttp not available",
            }, 0
        
        # Извлекаем промпт
        prompt = self._extract_prompt(payload)
        
        if not prompt:
            return {
                "error": "Missing or invalid prompt in payload",
            }, 0
        
        # Определяем модель (из payload или default)
        model = self.model_name
        if isinstance(payload, dict) and "model" in payload:
            model = payload["model"]
        
        # Дополнительные параметры
        system_prompt = None
        if isinstance(payload, dict):
            system_prompt = payload.get("system")
        
        # Выполняем запрос
        try:
            result = await self._call_ollama(prompt, model, system_prompt)
            
            if "error" in result:
                # Ошибка - деньги не списываем
                return result, 0
            
            # Успешный запрос - полная плата
            logger.info(f"[OLLAMA_AGENT] Success: {model}, {result.get('eval_count', 0)} tokens")
            return result, 1.0  # 1 единица работы = SUCCESS_PRICE
            
        except Exception as e:
            logger.error(f"[OLLAMA_AGENT] Unexpected error: {e}")
            return {
                "error": f"Unexpected error: {str(e)}",
            }, 0
    
    def _extract_prompt(self, payload: Any) -> Optional[str]:
        """Извлечь промпт из payload."""
        if isinstance(payload, str):
            return payload.strip()
        
        if isinstance(payload, dict):
            # Промпт
            if "prompt" in payload:
                prompt = payload["prompt"]
                if isinstance(prompt, str):
                    return prompt.strip()
            
            # Текст
            if "text" in payload:
                text = payload["text"]
                if isinstance(text, str):
                    return text.strip()
            
            # Сообщения (берем последнее от user)
            if "messages" in payload and isinstance(payload["messages"], list):
                for msg in reversed(payload["messages"]):
                    if msg.get("role") == "user":
                        return msg.get("content", "").strip()
        
        return None
    
    async def _call_ollama(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Вызов Ollama API.
        
        Returns:
            dict с результатом или ошибкой
        """
        url = f"{self.base_url}/api/generate"
        
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Важно: получаем ответ целиком
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body, timeout=timeout) as response:
                    # Проверяем статус
                    if response.status != 200:
                        text = await response.text()
                        return {
                            "error": f"Ollama error ({response.status}): {text}",
                            "status": response.status,
                        }
                    
                    data = await response.json()
                    
                    # Извлекаем ответ
                    response_text = data.get("response", "")
                    
                    if not response_text:
                        return {"error": "Empty response from Ollama"}
                    
                    return {
                        "response": response_text,
                        "model": data.get("model", model),
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration": data.get("eval_duration", 0),
                        "total_duration": data.get("total_duration", 0),
                    }
        
        except asyncio.TimeoutError:
            return {
                "error": f"Request timeout after {REQUEST_TIMEOUT}s - model may be too slow",
            }
        except aiohttp.ClientConnectorError:
            return {
                "error": f"GPU Node offline - Ollama not running at {self.base_url}",
            }
        except aiohttp.ClientError as e:
            return {
                "error": f"Connection error: {str(e)}",
            }
    
    async def check_health(self) -> Tuple[bool, str]:
        """
        Проверить доступность Ollama.
        
        Returns:
            (is_healthy, message)
        """
        if not AIOHTTP_AVAILABLE:
            return False, "aiohttp not available"
        
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        return True, f"Online, models: {', '.join(models[:5])}"
                    return False, f"HTTP {response.status}"
        except aiohttp.ClientConnectorError:
            return False, "GPU Node offline"
        except Exception as e:
            return False, str(e)
    
    async def list_models(self) -> List[str]:
        """Получить список доступных моделей."""
        if not AIOHTTP_AVAILABLE:
            return []
        
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []
    
    def estimate_cost(self, payload: Any) -> float:
        """Оценить стоимость запроса."""
        return self.SUCCESS_PRICE

