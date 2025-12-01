"""
Cortex Roles - Ролевая модель специализированных агентов
========================================================

[ROLES] Три специализации:
- Scout: Разведчик - поиск и сбор информации из веба
- Analyst: Аналитик - анализ текста, извлечение структурированных данных
- Librarian: Библиотекарь - индексация и хранение знаний в DHT

[DESIGN] Агенты stateful в рамках задачи, но stateless глобально.
"""

import asyncio
import json
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from agents.web_reader import ReaderAgent
    from agents.local_llm import OllamaAgent
    from agents.distributed_agent import DistributedLlmAgent
    from core.dht import DHTStorage

from .prompts import (
    format_analyst_prompt,
    format_scout_prompt,
    format_librarian_prompt,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class TaskStatus(Enum):
    """Статус выполнения задачи."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Задача для роли."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    requester_id: str = ""
    budget: float = 0.0
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class RoleResult:
    """Результат выполнения роли."""
    success: bool
    data: Dict[str, Any]
    cost: float = 0.0
    error: Optional[str] = None
    execution_time: float = 0.0
    role_name: str = ""


@dataclass
class ScoutResult:
    """Результат работы Scout."""
    urls: List[str]
    raw_texts: List[str]
    sources: List[Dict[str, str]]  # {"url": ..., "title": ..., "snippet": ...}
    total_chars: int = 0


@dataclass
class AnalysisResult:
    """Результат работы Analyst."""
    summary: str
    sentiment: str  # positive, negative, neutral, mixed
    key_facts: List[str]
    entities: Dict[str, List[str]]
    confidence: float
    topics: List[str]
    raw_json: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LibrarianResult:
    """Результат работы Librarian."""
    cid: str  # Content ID (hash)
    topic_key: str  # DHT key for topic index
    keywords: List[str]
    stored: bool = False


# =============================================================================
# BASE ROLE
# =============================================================================

class BaseRole(ABC):
    """
    Базовый класс для ролей Cortex.
    
    [INTERFACE]
    - execute(task) -> RoleResult
    - estimate_cost(task) -> float
    """
    
    role_name: str = "base"
    
    def __init__(self):
        self._execution_count = 0
        self._total_cost = 0.0
    
    @abstractmethod
    async def execute(self, task: Task) -> RoleResult:
        """Выполнить задачу."""
        pass
    
    def estimate_cost(self, task: Task) -> float:
        """Оценить стоимость выполнения задачи."""
        return 0.0
    
    def _create_result(
        self,
        success: bool,
        data: Dict[str, Any],
        cost: float = 0.0,
        error: Optional[str] = None,
        start_time: float = 0.0,
    ) -> RoleResult:
        """Создать результат выполнения."""
        execution_time = time.time() - start_time if start_time else 0.0
        self._execution_count += 1
        self._total_cost += cost
        
        return RoleResult(
            success=success,
            data=data,
            cost=cost,
            error=error,
            execution_time=execution_time,
            role_name=self.role_name,
        )


# =============================================================================
# SCOUT ROLE
# =============================================================================

class Scout(BaseRole):
    """
    Разведчик - поиск и сбор информации.
    
    [USES] ReaderAgent для чтения веб-страниц
    
    [INPUT] Task с payload:
        - topic: str - тема для поиска
        - urls: List[str] - конкретные URL (опционально)
        - max_sources: int - максимум источников (default: 5)
    
    [OUTPUT] ScoutResult:
        - urls: найденные URL
        - raw_texts: текст со страниц
        - sources: метаданные источников
    """
    
    role_name = "scout"
    
    # Базовые URL для поиска (можно расширить)
    SEARCH_ENGINES = [
        "https://en.wikipedia.org/wiki/{topic}",
        "https://www.britannica.com/search?query={topic}",
    ]
    
    def __init__(
        self,
        reader_agent: Optional["ReaderAgent"] = None,
        llm_agent: Optional["OllamaAgent"] = None,
    ):
        """
        Args:
            reader_agent: ReaderAgent для чтения страниц
            llm_agent: LLM для генерации поисковых запросов (опционально)
        """
        super().__init__()
        self.reader = reader_agent
        self.llm = llm_agent
    
    async def execute(self, task: Task) -> RoleResult:
        """
        Выполнить разведку по теме.
        
        Args:
            task: Task с topic или urls
        
        Returns:
            RoleResult с ScoutResult в data
        """
        start_time = time.time()
        
        topic = task.payload.get("topic", "")
        urls = task.payload.get("urls", [])
        max_sources = task.payload.get("max_sources", 5)
        
        if not topic and not urls:
            return self._create_result(
                success=False,
                data={},
                error="Missing topic or urls in payload",
                start_time=start_time,
            )
        
        # Если нет конкретных URL - генерируем из темы
        if not urls and topic:
            urls = await self._generate_urls(topic, max_sources)
        
        # Собираем информацию с URL
        sources = []
        raw_texts = []
        total_cost = 0.0
        
        for url in urls[:max_sources]:
            try:
                result = await self._fetch_url(url)
                if result:
                    sources.append({
                        "url": url,
                        "title": result.get("title", ""),
                        "snippet": result.get("text", "")[:500],
                    })
                    raw_texts.append(result.get("text", ""))
                    total_cost += result.get("cost", 0.0)
            except Exception as e:
                logger.warning(f"[SCOUT] Failed to fetch {url}: {e}")
        
        if not raw_texts:
            return self._create_result(
                success=False,
                data={},
                cost=total_cost,
                error="Could not fetch any sources",
                start_time=start_time,
            )
        
        scout_result = ScoutResult(
            urls=[s["url"] for s in sources],
            raw_texts=raw_texts,
            sources=sources,
            total_chars=sum(len(t) for t in raw_texts),
        )
        
        logger.info(f"[SCOUT] Collected {len(sources)} sources, {scout_result.total_chars} chars")
        
        return self._create_result(
            success=True,
            data={
                "urls": scout_result.urls,
                "raw_texts": scout_result.raw_texts,
                "sources": scout_result.sources,
                "total_chars": scout_result.total_chars,
            },
            cost=total_cost,
            start_time=start_time,
        )
    
    async def _generate_urls(self, topic: str, max_urls: int) -> List[str]:
        """Генерировать URL для темы."""
        urls = []
        
        # Базовые URL из шаблонов
        topic_formatted = topic.replace(" ", "_")
        topic_query = topic.replace(" ", "+")
        
        for template in self.SEARCH_ENGINES:
            try:
                url = template.format(topic=topic_formatted)
                urls.append(url)
            except Exception:
                pass
        
        # Если есть LLM - можно сгенерировать дополнительные запросы
        if self.llm and len(urls) < max_urls:
            try:
                prompts = format_scout_prompt(topic)
                result, _ = await self.llm.execute({
                    "prompt": prompts["user"],
                    "system": prompts["system"],
                })
                
                if isinstance(result, dict) and "response" in result:
                    response = result["response"]
                    # Пытаемся распарсить JSON
                    try:
                        data = json.loads(response)
                        queries = data.get("queries", [])
                        # Конвертируем запросы в URL (упрощённо)
                        for q in queries[:max_urls - len(urls)]:
                            wiki_url = f"https://en.wikipedia.org/wiki/{q.replace(' ', '_')}"
                            urls.append(wiki_url)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.warning(f"[SCOUT] LLM query generation failed: {e}")
        
        return urls[:max_urls]
    
    async def _fetch_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Получить контент с URL."""
        if not self.reader:
            logger.warning("[SCOUT] No reader agent configured")
            return None
        
        try:
            result, cost = await self.reader.execute({"url": url})
            
            if isinstance(result, dict) and "error" not in result:
                return {
                    "title": result.get("title", ""),
                    "text": result.get("text", ""),
                    "url": url,
                    "cost": cost * self.reader.price_per_unit,
                }
        except Exception as e:
            logger.error(f"[SCOUT] Fetch error for {url}: {e}")
        
        return None
    
    def estimate_cost(self, task: Task) -> float:
        """Оценить стоимость."""
        max_sources = task.payload.get("max_sources", 5)
        # Примерная стоимость: 10 единиц за источник
        return max_sources * 10.0


# =============================================================================
# ANALYST ROLE
# =============================================================================

class Analyst(BaseRole):
    """
    Аналитик - анализ текста, извлечение структурированных данных.
    
    [USES] OllamaAgent или DistributedLlmAgent
    
    [INPUT] Task с payload:
        - text: str - текст для анализа
        - topic: str - тема (для контекста)
    
    [OUTPUT] AnalysisResult:
        - summary: краткое содержание
        - sentiment: тональность
        - key_facts: ключевые факты
        - entities: извлечённые сущности
        - confidence: уверенность анализа
        - topics: темы для индексации
    """
    
    role_name = "analyst"
    
    def __init__(
        self,
        llm_agent: Optional["OllamaAgent"] = None,
        distributed_agent: Optional["DistributedLlmAgent"] = None,
        prefer_distributed: bool = False,
    ):
        """
        Args:
            llm_agent: OllamaAgent для локального инференса
            distributed_agent: DistributedLlmAgent для распределённого
            prefer_distributed: Предпочитать распределённый инференс
        """
        super().__init__()
        self.llm = llm_agent
        self.distributed = distributed_agent
        self.prefer_distributed = prefer_distributed
    
    async def execute(self, task: Task) -> RoleResult:
        """
        Выполнить анализ текста.
        
        Args:
            task: Task с text
        
        Returns:
            RoleResult с AnalysisResult в data
        """
        start_time = time.time()
        
        text = task.payload.get("text", "")
        topic = task.payload.get("topic", "")
        
        if not text:
            return self._create_result(
                success=False,
                data={},
                error="Missing text in payload",
                start_time=start_time,
            )
        
        # Выбираем агента
        agent = self._select_agent()
        if not agent:
            return self._create_result(
                success=False,
                data={},
                error="No LLM agent available",
                start_time=start_time,
            )
        
        # Формируем промпт
        prompts = format_analyst_prompt(text)
        
        try:
            # Запрос к LLM
            result, units = await agent.execute({
                "prompt": prompts["user"],
                "system": prompts["system"],
            })
            
            cost = units * agent.price_per_unit
            
            # Парсим ответ
            if isinstance(result, dict) and "response" in result:
                response_text = result["response"]
                analysis = self._parse_analysis(response_text)
                
                if analysis:
                    logger.info(f"[ANALYST] Analysis complete: {analysis.summary[:100]}...")
                    
                    return self._create_result(
                        success=True,
                        data={
                            "summary": analysis.summary,
                            "sentiment": analysis.sentiment,
                            "key_facts": analysis.key_facts,
                            "entities": analysis.entities,
                            "confidence": analysis.confidence,
                            "topics": analysis.topics,
                            "raw_json": analysis.raw_json,
                        },
                        cost=cost,
                        start_time=start_time,
                    )
                else:
                    return self._create_result(
                        success=False,
                        data={"raw_response": response_text},
                        cost=cost,
                        error="Failed to parse LLM response as JSON",
                        start_time=start_time,
                    )
            else:
                return self._create_result(
                    success=False,
                    data={"raw_result": result},
                    cost=cost,
                    error=result.get("error", "Unknown LLM error"),
                    start_time=start_time,
                )
                
        except Exception as e:
            logger.error(f"[ANALYST] Analysis failed: {e}")
            return self._create_result(
                success=False,
                data={},
                error=str(e),
                start_time=start_time,
            )
    
    def _select_agent(self):
        """Выбрать LLM агента."""
        if self.prefer_distributed and self.distributed:
            return self.distributed
        if self.llm:
            return self.llm
        if self.distributed:
            return self.distributed
        return None
    
    def _parse_analysis(self, response: str) -> Optional[AnalysisResult]:
        """Парсить JSON ответ LLM."""
        import re
        
        try:
            text = response.strip()
            
            # Убираем <think>...</think> блоки (qwen3 и другие reasoning модели)
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = text.strip()
            
            # Убираем markdown если есть
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # Пытаемся найти JSON объект с вложенными структурами
            # Ищем от первой { до последней }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                text = text[start:end+1]
            
            data = json.loads(text)
            logger.debug(f"[ANALYST] Parsed JSON keys: {list(data.keys())}")
            
            return AnalysisResult(
                summary=data.get("summary", ""),
                sentiment=data.get("sentiment", "neutral"),
                key_facts=data.get("key_facts", []),
                entities=data.get("entities", {}),
                confidence=float(data.get("confidence", 0.5)),
                topics=data.get("topics", []),
                raw_json=data,
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[ANALYST] JSON parse error: {e}")
            return None
    
    def estimate_cost(self, task: Task) -> float:
        """Оценить стоимость."""
        # Примерная стоимость анализа
        if self.prefer_distributed and self.distributed:
            return 20.0  # Distributed дешевле
        return 30.0  # Ollama


# =============================================================================
# LIBRARIAN ROLE
# =============================================================================

class Librarian(BaseRole):
    """
    Библиотекарь - индексация и хранение знаний в DHT.
    
    [USES] core/dht для хранения
    
    [INPUT] Task с payload:
        - analysis: Dict - результат анализа (AnalysisResult)
        - topic: str - основная тема
    
    [OUTPUT] LibrarianResult:
        - cid: Content ID (hash)
        - topic_key: DHT key
        - keywords: извлечённые ключевые слова
        - stored: успешно ли сохранено
    """
    
    role_name = "librarian"
    
    # Префиксы для DHT ключей
    TOPIC_PREFIX = "topic:"
    CONTENT_PREFIX = "content:"
    INDEX_PREFIX = "index:"
    
    def __init__(
        self,
        dht_storage: Optional["DHTStorage"] = None,
        kademlia_node=None,
        llm_agent: Optional["OllamaAgent"] = None,
    ):
        """
        Args:
            dht_storage: DHTStorage для локального хранения
            kademlia_node: KademliaNode для сетевого хранения
            llm_agent: LLM для извлечения ключевых слов (опционально)
        """
        super().__init__()
        self.storage = dht_storage
        self.kademlia = kademlia_node
        self.llm = llm_agent
    
    async def execute(self, task: Task) -> RoleResult:
        """
        Сохранить анализ в DHT и обновить индекс.
        
        Args:
            task: Task с analysis и topic
        
        Returns:
            RoleResult с LibrarianResult
        """
        start_time = time.time()
        
        analysis = task.payload.get("analysis", {})
        topic = task.payload.get("topic", "")
        
        if not analysis:
            return self._create_result(
                success=False,
                data={},
                error="Missing analysis in payload",
                start_time=start_time,
            )
        
        # Генерируем CID (Content ID)
        content_bytes = json.dumps(analysis, sort_keys=True).encode()
        cid = self._generate_cid(content_bytes)
        
        # Извлекаем ключевые слова
        keywords = await self._extract_keywords(analysis, topic)
        
        # Генерируем ключ темы
        topic_key = self._generate_topic_key(topic or keywords[0] if keywords else "unknown")
        
        # Сохраняем в DHT
        stored = False
        cost = 0.0
        
        try:
            # 1. Сохраняем контент
            content_key = f"{self.CONTENT_PREFIX}{cid}"
            await self._store(content_key, content_bytes)
            
            # 2. Обновляем индекс темы
            await self._update_topic_index(topic_key, cid)
            
            # 3. Индексируем по ключевым словам
            for kw in keywords[:10]:  # Максимум 10 ключевых слов
                kw_key = self._generate_topic_key(kw)
                await self._update_topic_index(kw_key, cid)
            
            stored = True
            logger.info(f"[LIBRARIAN] Stored CID {cid[:16]}... under topic '{topic}'")
            
        except Exception as e:
            logger.error(f"[LIBRARIAN] Storage error: {e}")
            return self._create_result(
                success=False,
                data={"cid": cid},
                error=str(e),
                start_time=start_time,
            )
        
        return self._create_result(
            success=True,
            data={
                "cid": cid,
                "topic_key": topic_key,
                "keywords": keywords,
                "stored": stored,
            },
            cost=cost,
            start_time=start_time,
        )
    
    def _generate_cid(self, content: bytes) -> str:
        """Генерировать Content ID (SHA-256 hash)."""
        return hashlib.sha256(content).hexdigest()
    
    def _generate_topic_key(self, topic: str) -> str:
        """Генерировать ключ темы для DHT."""
        normalized = topic.lower().strip().replace(" ", "_")
        return f"{self.TOPIC_PREFIX}{normalized}"
    
    async def _extract_keywords(
        self,
        analysis: Dict[str, Any],
        topic: str,
    ) -> List[str]:
        """Извлечь ключевые слова из анализа."""
        keywords = []
        
        # Из topics в анализе
        keywords.extend(analysis.get("topics", []))
        
        # Из entities
        entities = analysis.get("entities", {})
        for entity_type, entities_list in entities.items():
            keywords.extend(entities_list[:3])  # Макс 3 от каждого типа
        
        # Основная тема
        if topic:
            keywords.insert(0, topic)
        
        # Если есть LLM - можно улучшить
        if self.llm and len(keywords) < 5:
            try:
                prompts = format_librarian_prompt(json.dumps(analysis))
                result, _ = await self.llm.execute({
                    "prompt": prompts["user"],
                    "system": prompts["system"],
                })
                
                if isinstance(result, dict) and "response" in result:
                    try:
                        data = json.loads(result["response"])
                        keywords.extend(data.get("primary_keywords", []))
                        keywords.extend(data.get("secondary_keywords", []))
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.warning(f"[LIBRARIAN] Keyword extraction failed: {e}")
        
        # Дедупликация и нормализация
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_norm = kw.lower().strip()
            if kw_norm and kw_norm not in seen:
                seen.add(kw_norm)
                unique_keywords.append(kw_norm)
        
        return unique_keywords[:20]  # Максимум 20
    
    async def _store(self, key: str, value: bytes) -> None:
        """Сохранить в DHT."""
        if self.kademlia:
            await self.kademlia.dht_put(key, value)
        elif self.storage:
            self.storage.store(key, value)
        else:
            logger.warning("[LIBRARIAN] No storage backend configured")
    
    async def _update_topic_index(self, topic_key: str, cid: str) -> None:
        """Обновить индекс темы (добавить CID в список)."""
        index_key = f"{self.INDEX_PREFIX}{topic_key}"
        
        # Получаем текущий индекс
        current_index = []
        try:
            if self.kademlia:
                data = await self.kademlia.dht_get(index_key)
                if data:
                    current_index = json.loads(data.decode())
            elif self.storage:
                data = self.storage.get(index_key)
                if data:
                    current_index = json.loads(data.decode())
        except Exception:
            pass
        
        # Добавляем CID если его нет
        if cid not in current_index:
            current_index.insert(0, cid)  # Новые вначале
            current_index = current_index[:100]  # Максимум 100 CID на тему
        
        # Сохраняем обновлённый индекс
        index_bytes = json.dumps(current_index).encode()
        await self._store(index_key, index_bytes)
    
    async def get_topic_cids(self, topic: str) -> List[str]:
        """Получить список CID для темы."""
        topic_key = self._generate_topic_key(topic)
        index_key = f"{self.INDEX_PREFIX}{topic_key}"
        
        try:
            if self.kademlia:
                data = await self.kademlia.dht_get(index_key)
                if data:
                    return json.loads(data.decode())
            elif self.storage:
                data = self.storage.get(index_key)
                if data:
                    return json.loads(data.decode())
        except Exception as e:
            logger.warning(f"[LIBRARIAN] Get index error: {e}")
        
        return []
    
    async def get_content(self, cid: str) -> Optional[Dict[str, Any]]:
        """Получить контент по CID."""
        content_key = f"{self.CONTENT_PREFIX}{cid}"
        
        try:
            if self.kademlia:
                data = await self.kademlia.dht_get(content_key)
                if data:
                    return json.loads(data.decode())
            elif self.storage:
                data = self.storage.get(content_key)
                if data:
                    return json.loads(data.decode())
        except Exception as e:
            logger.warning(f"[LIBRARIAN] Get content error: {e}")
        
        return None
    
    def estimate_cost(self, task: Task) -> float:
        """Оценить стоимость."""
        # DHT операции относительно дешёвые
        return 5.0

