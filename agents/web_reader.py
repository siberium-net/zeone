"""
Web Reader Agent - Чтение веб-страниц
=====================================

[MARKET] Услуга "web_read":
- Скачивает HTML по URL
- Очищает от тегов (HTML/JS/CSS)
- Возвращает чистый текст

[ECONOMY] Цена: 10 единиц за успешный запрос
- Это дороже Echo, так как тратит внешний трафик
- При ошибке берется минимальная плата (1 единица) за попытку

[SECURITY] Ограничения:
- Таймаут запроса: 30 секунд
- Максимальный размер страницы: 5 MB
- Возвращается только первые 2000 символов текста
"""

import asyncio
import logging
from typing import Any, Tuple, Optional
from urllib.parse import urlparse

try:
    import aiohttp
    from bs4 import BeautifulSoup
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    _IMPORT_ERROR = str(e)

from .manager import BaseAgent

logger = logging.getLogger(__name__)


# Конфигурация
REQUEST_TIMEOUT = 30  # секунды
MAX_CONTENT_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_TEXT_LENGTH = 2000  # символов в ответе
MIN_FEE_ON_ERROR = 1  # минимальная плата за попытку при ошибке

# User-Agent для запросов
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class ReaderAgent(BaseAgent):
    """
    Агент чтения веб-страниц.
    
    [MARKET] Услуга "web_read":
    - Принимает URL в payload
    - Скачивает страницу через aiohttp
    - Парсит HTML через BeautifulSoup
    - Возвращает заголовок и чистый текст
    
    [ECONOMY] Ценообразование:
    - Успешный запрос: 10 единиц
    - Ошибка: 1 единица (за попытку)
    
    [EXAMPLE]
    Request: {"url": "https://example.com"}
    Response: {
        "title": "Example Domain",
        "text": "This domain is for use in illustrative examples...",
        "url": "https://example.com",
        "content_length": 1256
    }
    """
    
    SUCCESS_PRICE = 10  # Цена за успешный запрос
    ERROR_PRICE = 1     # Цена за попытку при ошибке
    
    @property
    def service_name(self) -> str:
        return "web_read"
    
    @property
    def price_per_unit(self) -> float:
        return self.SUCCESS_PRICE
    
    @property
    def description(self) -> str:
        return (
            f"Web Reader: fetch URL and return clean text. "
            f"Price: {self.SUCCESS_PRICE} on success, {self.ERROR_PRICE} on error. "
            f"Max text: {MAX_TEXT_LENGTH} chars"
        )
    
    async def execute(self, payload: Any) -> Tuple[Any, float]:
        """
        Выполнить запрос веб-страницы.
        
        Args:
            payload: URL строка или dict с ключом "url"
        
        Returns:
            (result, units) - результат и количество единиц работы
        """
        # Проверяем доступность зависимостей
        if not DEPENDENCIES_AVAILABLE:
            return {
                "error": f"Dependencies not available: {_IMPORT_ERROR}",
                "url": None,
            }, self.ERROR_PRICE / self.price_per_unit
        
        # Извлекаем URL из payload
        url = self._extract_url(payload)
        
        if not url:
            return {
                "error": "Missing or invalid URL in payload",
                "url": None,
            }, self.ERROR_PRICE / self.price_per_unit
        
        # Валидируем URL
        if not self._validate_url(url):
            return {
                "error": f"Invalid URL format: {url}",
                "url": url,
            }, self.ERROR_PRICE / self.price_per_unit
        
        # Выполняем запрос
        try:
            result = await self._fetch_and_parse(url)
            
            if "error" in result:
                # Ошибка при запросе - минимальная плата
                return result, self.ERROR_PRICE / self.price_per_unit
            
            # Успешный запрос - полная плата
            logger.info(f"[WEB_READ] Success: {url}, {result.get('content_length', 0)} chars")
            return result, 1.0  # 1 единица работы = SUCCESS_PRICE
            
        except Exception as e:
            logger.error(f"[WEB_READ] Unexpected error for {url}: {e}")
            return {
                "error": f"Unexpected error: {str(e)}",
                "url": url,
            }, self.ERROR_PRICE / self.price_per_unit
    
    def _extract_url(self, payload: Any) -> Optional[str]:
        """Извлечь URL из payload."""
        if isinstance(payload, str):
            return payload.strip()
        elif isinstance(payload, dict):
            url = payload.get("url")
            if isinstance(url, str):
                return url.strip()
        return None
    
    def _validate_url(self, url: str) -> bool:
        """Проверить валидность URL."""
        try:
            parsed = urlparse(url)
            # Должен быть http или https
            if parsed.scheme not in ("http", "https"):
                return False
            # Должен быть хост
            if not parsed.netloc:
                return False
            return True
        except Exception:
            return False
    
    async def _fetch_and_parse(self, url: str) -> dict:
        """
        Скачать и распарсить веб-страницу.
        
        Returns:
            dict с результатом или ошибкой
        """
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        headers = {"User-Agent": USER_AGENT}
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, ssl=False) as response:
                    # Проверяем статус
                    if response.status != 200:
                        return {
                            "error": f"HTTP {response.status}: {response.reason}",
                            "url": url,
                            "status": response.status,
                        }
                    
                    # Проверяем размер
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > MAX_CONTENT_SIZE:
                        return {
                            "error": f"Content too large: {content_length} bytes",
                            "url": url,
                        }
                    
                    # Читаем контент
                    html = await response.text(errors="ignore")
                    
                    if len(html) > MAX_CONTENT_SIZE:
                        return {
                            "error": f"Content too large: {len(html)} bytes",
                            "url": url,
                        }
        
        except asyncio.TimeoutError:
            return {
                "error": f"Request timeout after {REQUEST_TIMEOUT}s",
                "url": url,
            }
        except aiohttp.ClientError as e:
            return {
                "error": f"Connection error: {str(e)}",
                "url": url,
            }
        
        # Парсим HTML
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Удаляем скрипты и стили
            for tag in soup(["script", "style", "noscript", "meta", "link"]):
                tag.decompose()
            
            # Извлекаем заголовок
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            
            # Извлекаем текст
            text = soup.get_text(separator=" ", strip=True)
            
            # Нормализуем пробелы
            text = " ".join(text.split())
            
            # Обрезаем до максимальной длины
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH] + "..."
            
            return {
                "title": title,
                "text": text,
                "url": url,
                "content_length": len(text),
            }
            
        except Exception as e:
            return {
                "error": f"Parse error: {str(e)}",
                "url": url,
            }
    
    def estimate_cost(self, payload: Any) -> float:
        """
        Оценить стоимость запроса.
        
        Всегда возвращаем максимальную стоимость,
        так как реальная стоимость зависит от успеха запроса.
        """
        return self.SUCCESS_PRICE

