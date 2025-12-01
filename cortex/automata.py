"""
Cortex Automata - Автономный цикл жизни
=======================================

[AUTOMATA] Фоновый процесс "мышления":
- Мониторинг трендов (RSS, API, заглушки)
- Автоматическое исследование новых тем
- Заполнение библиотеки знаниями до того, как спросят

[FLOW]
thought_loop() -> monitor_trends() -> check_library() -> 
-> investigate() -> Scout -> Analyst -> Librarian -> DHT
"""

import asyncio
import json
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, TYPE_CHECKING
from enum import Enum
import hashlib

if TYPE_CHECKING:
    from .roles import Scout, Analyst, Librarian, Task
    from .library import SemanticLibrary
    from .consilium import Consilium

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class InvestigationStatus(Enum):
    """Статус расследования."""
    PENDING = "pending"
    SCOUTING = "scouting"
    ANALYZING = "analyzing"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrendTopic:
    """Трендовая тема."""
    topic: str
    urgency: str  # high, medium, low
    source: str
    reason: str
    discovered_at: float = field(default_factory=time.time)
    
    def priority_score(self) -> float:
        """Вычислить приоритет."""
        urgency_scores = {"high": 1.0, "medium": 0.5, "low": 0.2}
        return urgency_scores.get(self.urgency, 0.1)


@dataclass
class Investigation:
    """Расследование темы."""
    investigation_id: str
    topic: str
    status: InvestigationStatus
    started_at: float
    scout_result: Optional[Dict[str, Any]] = None
    analysis_result: Optional[Dict[str, Any]] = None
    library_cid: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None


# =============================================================================
# TREND SOURCES
# =============================================================================

class TrendSource:
    """Базовый класс источника трендов."""
    
    name: str = "base"
    
    async def fetch_trends(self) -> List[TrendTopic]:
        """Получить текущие тренды."""
        return []


class StubTrendSource(TrendSource):
    """Заглушка источника трендов для разработки."""
    
    name = "stub"
    
    # Список тем для исследования
    STUB_TOPICS = [
        ("quantum computing", "high", "Emerging technology with rapid advancement"),
        ("artificial general intelligence", "high", "Critical topic in AI development"),
        ("blockchain scalability", "medium", "Technical challenge in crypto space"),
        ("neural interfaces", "medium", "Brain-computer interface developments"),
        ("fusion energy progress", "low", "Long-term energy solution updates"),
        ("CRISPR applications", "medium", "Gene editing breakthroughs"),
        ("autonomous vehicles", "low", "Self-driving car developments"),
        ("space colonization", "low", "Mars and Moon base planning"),
        ("post-quantum cryptography", "high", "Security in quantum era"),
        ("synthetic biology", "medium", "Engineered organisms research"),
    ]
    
    def __init__(self, randomize: bool = True):
        self.randomize = randomize
        self._used_topics: Set[str] = set()
    
    async def fetch_trends(self) -> List[TrendTopic]:
        """Вернуть заглушечные тренды."""
        available = [
            t for t in self.STUB_TOPICS
            if t[0] not in self._used_topics
        ]
        
        if not available:
            # Сброс если всё использовано
            self._used_topics.clear()
            available = list(self.STUB_TOPICS)
        
        if self.randomize:
            random.shuffle(available)
        
        # Возвращаем 2-3 темы
        count = random.randint(2, 3) if self.randomize else 2
        selected = available[:count]
        
        topics = []
        for topic, urgency, reason in selected:
            self._used_topics.add(topic)
            topics.append(TrendTopic(
                topic=topic,
                urgency=urgency,
                source=self.name,
                reason=reason,
            ))
        
        return topics


class RSSFeedSource(TrendSource):
    """RSS источник трендов."""
    
    name = "rss"
    
    # Популярные технические RSS
    DEFAULT_FEEDS = [
        "https://news.ycombinator.com/rss",
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    ]
    
    def __init__(self, feeds: Optional[List[str]] = None):
        self.feeds = feeds or self.DEFAULT_FEEDS
        self._llm = None  # Можно добавить LLM для анализа
    
    async def fetch_trends(self) -> List[TrendTopic]:
        """Получить тренды из RSS."""
        try:
            import aiohttp
            import xml.etree.ElementTree as ET
        except ImportError:
            logger.warning("[AUTOMATA] aiohttp not available for RSS")
            return []
        
        headlines = []
        
        for feed_url in self.feeds[:3]:  # Лимит на количество фидов
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(feed_url, timeout=10) as response:
                        if response.status == 200:
                            xml_text = await response.text()
                            root = ET.fromstring(xml_text)
                            
                            # Парсим RSS
                            for item in root.findall(".//item")[:10]:
                                title = item.find("title")
                                if title is not None and title.text:
                                    headlines.append(title.text)
            except Exception as e:
                logger.debug(f"[AUTOMATA] RSS fetch error: {e}")
        
        # Конвертируем заголовки в темы (упрощённо)
        topics = []
        for headline in headlines[:5]:
            # Извлекаем ключевые слова (упрощённая логика)
            words = headline.lower().split()
            keywords = [w for w in words if len(w) > 4][:3]
            
            if keywords:
                topic = " ".join(keywords)
                topics.append(TrendTopic(
                    topic=topic,
                    urgency="medium",
                    source="rss",
                    reason=f"From headline: {headline[:50]}...",
                ))
        
        return topics


# =============================================================================
# AUTOMATA
# =============================================================================

class Automata:
    """
    Автономный мыслящий агент.
    
    [LIFECYCLE]
    1. thought_loop() запускается как фоновая задача
    2. Периодически проверяет тренды
    3. Для новых тем запускает расследование
    4. Результаты сохраняются в библиотеку
    
    [USAGE]
    ```python
    automata = Automata(
        scout=scout,
        analyst=analyst,
        librarian=librarian,
        library=library,
    )
    await automata.start()
    ```
    """
    
    # Configuration
    THOUGHT_INTERVAL = 300  # 5 минут между циклами мышления
    MAX_CONCURRENT_INVESTIGATIONS = 2
    INVESTIGATION_TIMEOUT = 600  # 10 минут на расследование
    
    def __init__(
        self,
        scout: Optional["Scout"] = None,
        analyst: Optional["Analyst"] = None,
        librarian: Optional["Librarian"] = None,
        library: Optional["SemanticLibrary"] = None,
        consilium: Optional["Consilium"] = None,
        trend_sources: Optional[List[TrendSource]] = None,
        node_id: str = "",
    ):
        """
        Args:
            scout: Scout роль для разведки
            analyst: Analyst роль для анализа
            librarian: Librarian роль для хранения
            library: SemanticLibrary для проверки/хранения
            consilium: Consilium для сложных задач
            trend_sources: Источники трендов
            node_id: ID текущего узла
        """
        self.scout = scout
        self.analyst = analyst
        self.librarian = librarian
        self.library = library
        self.consilium = consilium
        self.node_id = node_id
        
        # Trend sources
        self.trend_sources = trend_sources or [StubTrendSource()]
        
        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._investigations: Dict[str, Investigation] = {}
        self._investigated_topics: Set[str] = set()
        
        # Callbacks
        self._on_investigation_start: List[Callable] = []
        self._on_investigation_complete: List[Callable] = []
        
        # Stats
        self._total_thoughts = 0
        self._successful_investigations = 0
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    async def start(self) -> None:
        """Запустить автономный цикл."""
        if self._running:
            logger.warning("[AUTOMATA] Already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._thought_loop())
        logger.info("[AUTOMATA] Started autonomous thought loop")
    
    async def stop(self) -> None:
        """Остановить автономный цикл."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("[AUTOMATA] Stopped")
    
    # =========================================================================
    # THOUGHT LOOP
    # =========================================================================
    
    async def _thought_loop(self) -> None:
        """
        Основной цикл мышления.
        
        Периодически:
        1. Проверяет тренды
        2. Фильтрует уже исследованные темы
        3. Запускает расследования для новых тем
        """
        logger.info("[AUTOMATA] Thought loop started")
        
        while self._running:
            try:
                self._total_thoughts += 1
                logger.debug(f"[AUTOMATA] Thought cycle #{self._total_thoughts}")
                
                # 1. Получаем тренды
                trends = await self._monitor_trends()
                
                if trends:
                    logger.info(f"[AUTOMATA] Found {len(trends)} trending topics")
                    
                    # 2. Фильтруем известные темы
                    new_topics = await self._filter_new_topics(trends)
                    
                    if new_topics:
                        logger.info(f"[AUTOMATA] {len(new_topics)} new topics to investigate")
                        
                        # 3. Запускаем расследования
                        for topic in new_topics[:self.MAX_CONCURRENT_INVESTIGATIONS]:
                            asyncio.create_task(self._investigate(topic))
                
                # Ждём до следующего цикла
                await asyncio.sleep(self.THOUGHT_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AUTOMATA] Thought loop error: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке
        
        logger.info("[AUTOMATA] Thought loop ended")
    
    async def _monitor_trends(self) -> List[TrendTopic]:
        """Мониторинг трендов из всех источников."""
        all_trends = []
        
        for source in self.trend_sources:
            try:
                trends = await source.fetch_trends()
                all_trends.extend(trends)
                logger.debug(f"[AUTOMATA] Got {len(trends)} trends from {source.name}")
            except Exception as e:
                logger.warning(f"[AUTOMATA] Trend source {source.name} error: {e}")
        
        # Сортируем по приоритету
        all_trends.sort(key=lambda t: t.priority_score(), reverse=True)
        
        return all_trends
    
    async def _filter_new_topics(
        self,
        trends: List[TrendTopic],
    ) -> List[TrendTopic]:
        """Отфильтровать уже известные темы."""
        new_topics = []
        
        for trend in trends:
            topic_normalized = trend.topic.lower().strip()
            
            # Уже расследовали?
            if topic_normalized in self._investigated_topics:
                continue
            
            # Есть в библиотеке?
            if self.library:
                has_topic = await self.library.has_topic(trend.topic)
                if has_topic:
                    logger.debug(f"[AUTOMATA] Topic '{trend.topic}' already in library")
                    self._investigated_topics.add(topic_normalized)
                    continue
            
            new_topics.append(trend)
        
        return new_topics
    
    # =========================================================================
    # INVESTIGATION
    # =========================================================================
    
    async def investigate(self, topic: str) -> Optional[Investigation]:
        """
        Публичный метод для запуска расследования.
        
        Args:
            topic: Тема для расследования
        
        Returns:
            Investigation или None
        """
        trend = TrendTopic(
            topic=topic,
            urgency="high",
            source="manual",
            reason="Manual investigation request",
        )
        return await self._investigate(trend)
    
    async def _investigate(self, trend: TrendTopic) -> Optional[Investigation]:
        """
        Провести расследование темы.
        
        Flow: Scout -> Analyst -> Librarian
        """
        investigation_id = hashlib.sha256(
            f"{trend.topic}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        investigation = Investigation(
            investigation_id=investigation_id,
            topic=trend.topic,
            status=InvestigationStatus.PENDING,
            started_at=time.time(),
        )
        
        self._investigations[investigation_id] = investigation
        
        logger.info(f"[AUTOMATA] Starting investigation: '{trend.topic}'")
        
        # Notify callbacks
        for callback in self._on_investigation_start:
            try:
                await callback(investigation)
            except Exception:
                pass
        
        try:
            # STEP 1: SCOUTING
            investigation.status = InvestigationStatus.SCOUTING
            
            if not self.scout:
                raise ValueError("Scout not configured")
            
            from .roles import Task
            
            scout_task = Task(
                task_id=f"scout_{investigation_id}",
                task_type="scout",
                payload={
                    "topic": trend.topic,
                    "max_sources": 3,
                },
            )
            
            scout_result = await asyncio.wait_for(
                self.scout.execute(scout_task),
                timeout=self.INVESTIGATION_TIMEOUT / 3,
            )
            
            if not scout_result.success:
                raise ValueError(f"Scout failed: {scout_result.error}")
            
            investigation.scout_result = scout_result.data
            raw_texts = scout_result.data.get("raw_texts", [])
            
            if not raw_texts:
                raise ValueError("No text collected by Scout")
            
            logger.debug(f"[AUTOMATA] Scout collected {len(raw_texts)} texts")
            
            # STEP 2: ANALYZING
            investigation.status = InvestigationStatus.ANALYZING
            
            if not self.analyst:
                raise ValueError("Analyst not configured")
            
            # Объединяем тексты
            combined_text = "\n\n---\n\n".join(raw_texts)[:15000]  # Лимит
            
            analysis_task = Task(
                task_id=f"analysis_{investigation_id}",
                task_type="analysis",
                payload={
                    "text": combined_text,
                    "topic": trend.topic,
                },
            )
            
            analysis_result = await asyncio.wait_for(
                self.analyst.execute(analysis_task),
                timeout=self.INVESTIGATION_TIMEOUT / 3,
            )
            
            if not analysis_result.success:
                raise ValueError(f"Analyst failed: {analysis_result.error}")
            
            investigation.analysis_result = analysis_result.data
            
            logger.debug(
                f"[AUTOMATA] Analysis complete: "
                f"confidence={analysis_result.data.get('confidence', 0)}"
            )
            
            # STEP 3: STORING
            investigation.status = InvestigationStatus.STORING
            
            cid = None
            
            if self.library:
                # Сохраняем через Library
                cid = await self.library.store(
                    topic=trend.topic,
                    analysis=analysis_result.data,
                    author_id=self.node_id,
                )
            elif self.librarian:
                # Или через Librarian роль
                store_task = Task(
                    task_id=f"store_{investigation_id}",
                    task_type="store",
                    payload={
                        "analysis": analysis_result.data,
                        "topic": trend.topic,
                    },
                )
                
                store_result = await asyncio.wait_for(
                    self.librarian.execute(store_task),
                    timeout=60,
                )
                
                if store_result.success:
                    cid = store_result.data.get("cid")
            
            investigation.library_cid = cid
            investigation.status = InvestigationStatus.COMPLETED
            investigation.completed_at = time.time()
            
            # Mark topic as investigated
            self._investigated_topics.add(trend.topic.lower().strip())
            self._successful_investigations += 1
            
            logger.info(
                f"[AUTOMATA] Investigation complete: '{trend.topic}' -> CID={cid[:16] if cid else 'N/A'}..."
            )
            
            # Notify callbacks
            for callback in self._on_investigation_complete:
                try:
                    await callback(investigation)
                except Exception:
                    pass
            
            return investigation
            
        except asyncio.TimeoutError:
            investigation.status = InvestigationStatus.FAILED
            investigation.error = "Investigation timeout"
            investigation.completed_at = time.time()
            logger.warning(f"[AUTOMATA] Investigation timeout: '{trend.topic}'")
            
        except Exception as e:
            investigation.status = InvestigationStatus.FAILED
            investigation.error = str(e)
            investigation.completed_at = time.time()
            logger.error(f"[AUTOMATA] Investigation failed: '{trend.topic}': {e}")
        
        return investigation
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_investigation(self, investigation_id: str) -> Optional[Investigation]:
        """Получить расследование по ID."""
        return self._investigations.get(investigation_id)
    
    def get_active_investigations(self) -> List[Investigation]:
        """Получить активные расследования."""
        return [
            inv for inv in self._investigations.values()
            if inv.status not in (
                InvestigationStatus.COMPLETED,
                InvestigationStatus.FAILED,
            )
        ]
    
    def get_recent_investigations(self, limit: int = 10) -> List[Investigation]:
        """Получить последние расследования."""
        investigations = list(self._investigations.values())
        investigations.sort(key=lambda i: i.started_at, reverse=True)
        return investigations[:limit]
    
    def on_investigation_start(self, callback: Callable) -> None:
        """Регистрация callback на начало расследования."""
        self._on_investigation_start.append(callback)
    
    def on_investigation_complete(self, callback: Callable) -> None:
        """Регистрация callback на завершение расследования."""
        self._on_investigation_complete.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            "running": self._running,
            "total_thoughts": self._total_thoughts,
            "total_investigations": len(self._investigations),
            "successful_investigations": self._successful_investigations,
            "active_investigations": len(self.get_active_investigations()),
            "investigated_topics": len(self._investigated_topics),
        }
    
    @property
    def is_running(self) -> bool:
        """Проверить запущен ли цикл."""
        return self._running


# =============================================================================
# THOUGHT LOOP FUNCTION
# =============================================================================

async def thought_loop(
    scout: Optional["Scout"] = None,
    analyst: Optional["Analyst"] = None,
    librarian: Optional["Librarian"] = None,
    library: Optional["SemanticLibrary"] = None,
    interval: float = 300,
) -> None:
    """
    Автономный цикл мышления (функциональный интерфейс).
    
    Args:
        scout: Scout роль
        analyst: Analyst роль
        librarian: Librarian роль
        library: SemanticLibrary
        interval: Интервал между циклами
    """
    automata = Automata(
        scout=scout,
        analyst=analyst,
        librarian=librarian,
        library=library,
    )
    automata.THOUGHT_INTERVAL = interval
    
    await automata.start()
    
    # Ждём бесконечно
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await automata.stop()

