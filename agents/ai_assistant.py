"""
AI Assistant Agent - Облачный LLM (OpenAI-compatible API)
=========================================================

[MARKET] Услуга "llm_prompt":
- Отправляет промпт к облачному LLM API
- Поддерживает OpenAI-совместимые API (OpenAI, Claude, etc.)
- Конфигурируется через переменные окружения

[ECONOMY] Цена: 50 единиц за успешный запрос
- Дорогой ресурс (облачные API)
- При ошибке API деньги не списываются

[SECURITY] Настройки:
- API ключ читается из переменных окружения
- Таймаут запроса: 120 секунд
- Максимальная длина ответа: 4096 токенов
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
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 120  # секунды
MAX_TOKENS = 4096
ERROR_PRICE = 0  # При ошибке API деньги не списываются


class LlmAgent(BaseAgent):
    """
    Агент облачного LLM (OpenAI-compatible API).
    
    [MARKET] Услуга "llm_prompt":
    - Принимает промпт или историю сообщений
    - Отправляет запрос к облачному LLM API
    - Возвращает текстовый ответ
    
    [ECONOMY] Ценообразование:
    - Успешный запрос: 50 единиц
    - Ошибка: 0 единиц (деньги не списываются)
    
    [CONFIG] Переменные окружения:
    - LLM_API_KEY: API ключ (обязательно)
    - LLM_API_BASE: Базовый URL API (по умолчанию OpenAI)
    - LLM_MODEL: Название модели (по умолчанию gpt-4o-mini)
    
    [EXAMPLE]
    Request: {"prompt": "What is the meaning of life?"}
    или
    Request: {"messages": [{"role": "user", "content": "Hello!"}]}
    
    Response: {
        "response": "The meaning of life is...",
        "model": "gpt-4o-mini",
        "tokens_used": 150
    }
    """
    
    SUCCESS_PRICE = 50  # Цена за успешный запрос
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Инициализация агента.
        
        Args:
            api_key: API ключ (или из LLM_API_KEY)
            api_base: Базовый URL API (или из LLM_API_BASE)
            model: Название модели (или из LLM_MODEL)
        """
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.api_base = api_base or os.getenv("LLM_API_BASE", DEFAULT_API_BASE)
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        
        # Убираем trailing slash
        self.api_base = self.api_base.rstrip("/")
        
        if not self.api_key:
            logger.warning("[LLM_AGENT] LLM_API_KEY not set - agent will return errors")
    
    @property
    def service_name(self) -> str:
        return "llm_prompt"
    
    @property
    def price_per_unit(self) -> float:
        return self.SUCCESS_PRICE
    
    @property
    def description(self) -> str:
        return (
            f"Cloud LLM: send prompt to {self.model}. "
            f"Price: {self.SUCCESS_PRICE} on success, {ERROR_PRICE} on error."
        )
    
    async def execute(self, payload: Any) -> Tuple[Any, float]:
        """
        Выполнить запрос к LLM API.
        
        Args:
            payload: Строка (промпт) или dict с ключом "prompt" или "messages"
        
        Returns:
            (result, units) - результат и количество единиц работы
        """
        if not AIOHTTP_AVAILABLE:
            return {
                "error": "aiohttp not available",
            }, ERROR_PRICE / self.price_per_unit if self.price_per_unit else 0
        
        if not self.api_key:
            return {
                "error": "LLM_API_KEY not configured",
            }, ERROR_PRICE / self.price_per_unit if self.price_per_unit else 0
        
        # Извлекаем промпт или сообщения
        messages = self._extract_messages(payload)
        
        if not messages:
            return {
                "error": "Missing or invalid prompt/messages in payload",
            }, ERROR_PRICE / self.price_per_unit if self.price_per_unit else 0
        
        # Определяем модель (из payload или default)
        model = self.model
        if isinstance(payload, dict) and "model" in payload:
            model = payload["model"]
        
        # Выполняем запрос
        try:
            result = await self._call_api(messages, model)
            
            if "error" in result:
                # Ошибка API - деньги не списываем
                return result, ERROR_PRICE / self.price_per_unit if self.price_per_unit else 0
            
            # Успешный запрос - полная плата
            logger.info(f"[LLM_AGENT] Success: {model}, {result.get('tokens_used', 0)} tokens")
            return result, 1.0  # 1 единица работы = SUCCESS_PRICE
            
        except Exception as e:
            logger.error(f"[LLM_AGENT] Unexpected error: {e}")
            return {
                "error": f"Unexpected error: {str(e)}",
            }, ERROR_PRICE / self.price_per_unit if self.price_per_unit else 0
    
    def _extract_messages(self, payload: Any) -> Optional[List[Dict[str, str]]]:
        """Извлечь сообщения из payload."""
        # Строка -> простой промпт
        if isinstance(payload, str):
            return [{"role": "user", "content": payload}]
        
        if isinstance(payload, dict):
            # История сообщений
            if "messages" in payload and isinstance(payload["messages"], list):
                return payload["messages"]
            
            # Простой промпт
            if "prompt" in payload:
                prompt = payload["prompt"]
                if isinstance(prompt, str):
                    return [{"role": "user", "content": prompt}]
            
            # Текст
            if "text" in payload:
                text = payload["text"]
                if isinstance(text, str):
                    return [{"role": "user", "content": text}]
        
        return None
    
    async def _call_api(self, messages: List[Dict[str, str]], model: str) -> dict:
        """
        Вызов OpenAI-compatible API.
        
        Returns:
            dict с результатом или ошибкой
        """
        url = f"{self.api_base}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=body) as response:
                    data = await response.json()
                    
                    if response.status != 200:
                        error_msg = data.get("error", {}).get("message", str(data))
                        return {
                            "error": f"API error ({response.status}): {error_msg}",
                            "status": response.status,
                        }
                    
                    # Извлекаем ответ
                    choices = data.get("choices", [])
                    if not choices:
                        return {"error": "No response from API"}
                    
                    response_text = choices[0].get("message", {}).get("content", "")
                    
                    # Извлекаем информацию о токенах
                    usage = data.get("usage", {})
                    tokens_used = usage.get("total_tokens", 0)
                    
                    return {
                        "response": response_text,
                        "model": model,
                        "tokens_used": tokens_used,
                    }
        
        except asyncio.TimeoutError:
            return {
                "error": f"Request timeout after {REQUEST_TIMEOUT}s",
            }
        except aiohttp.ClientError as e:
            return {
                "error": f"Connection error: {str(e)}",
            }
    
    def estimate_cost(self, payload: Any) -> float:
        """Оценить стоимость запроса."""
        return self.SUCCESS_PRICE

