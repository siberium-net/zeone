"""
Cortex Library - Децентрализованная Библиотека Смыслов
======================================================

[LIBRARY] Semantic Library поверх Kademlia DHT:
- Topic Index: hash("topic:X") -> List[CID]
- Content Store: hash("content:CID") -> Report JSON
- Bounty System: Автоматическое создание задач на исследование

[RACE] Гонка за знаниями:
- Если тема не найдена - создаётся Bounty
- Первые качественные отчёты получают оплату
"""

import asyncio
import json
import hashlib
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from core.dht import DHTStorage
    from economy.ledger import Ledger

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class BountyStatus(Enum):
    """Статус Bounty."""
    OPEN = "open"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class Report:
    """Аналитический отчёт в библиотеке."""
    cid: str
    topic: str
    summary: str
    sentiment: str
    key_facts: List[str]
    topics: List[str]
    confidence: float
    author_id: str
    created_at: float
    quality_score: float = 0.0
    views: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь."""
        return {
            "cid": self.cid,
            "topic": self.topic,
            "summary": self.summary,
            "sentiment": self.sentiment,
            "key_facts": self.key_facts,
            "topics": self.topics,
            "confidence": self.confidence,
            "author_id": self.author_id,
            "created_at": self.created_at,
            "quality_score": self.quality_score,
            "views": self.views,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Report":
        """Создать из словаря."""
        return cls(
            cid=data.get("cid", ""),
            topic=data.get("topic", ""),
            summary=data.get("summary", ""),
            sentiment=data.get("sentiment", "neutral"),
            key_facts=data.get("key_facts", []),
            topics=data.get("topics", []),
            confidence=data.get("confidence", 0.0),
            author_id=data.get("author_id", ""),
            created_at=data.get("created_at", time.time()),
            quality_score=data.get("quality_score", 0.0),
            views=data.get("views", 0),
        )


@dataclass
class Bounty:
    """Bounty-задача на исследование темы."""
    bounty_id: str
    topic: str
    reward: float
    creator_id: str
    created_at: float
    expires_at: float
    status: BountyStatus = BountyStatus.OPEN
    claimed_by: Optional[str] = None
    result_cid: Optional[str] = None
    min_confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь."""
        return {
            "bounty_id": self.bounty_id,
            "topic": self.topic,
            "reward": self.reward,
            "creator_id": self.creator_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "claimed_by": self.claimed_by,
            "result_cid": self.result_cid,
            "min_confidence": self.min_confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bounty":
        """Создать из словаря."""
        return cls(
            bounty_id=data.get("bounty_id", ""),
            topic=data.get("topic", ""),
            reward=data.get("reward", 0.0),
            creator_id=data.get("creator_id", ""),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at", time.time() + 3600),
            status=BountyStatus(data.get("status", "open")),
            claimed_by=data.get("claimed_by"),
            result_cid=data.get("result_cid"),
            min_confidence=data.get("min_confidence", 0.5),
        )
    
    def is_expired(self) -> bool:
        """Проверить истёк ли срок."""
        return time.time() > self.expires_at


@dataclass
class TopicIndex:
    """Индекс темы в библиотеке."""
    topic: str
    topic_key: str
    cids: List[str]
    last_updated: float
    total_reports: int
    best_cid: Optional[str] = None
    best_quality: float = 0.0


# =============================================================================
# SEMANTIC LIBRARY
# =============================================================================

class SemanticLibrary:
    """
    Децентрализованная Библиотека Смыслов.
    
    [STORAGE] Три типа данных в DHT:
    - topic:X -> TopicIndex (список CID отчётов)
    - content:CID -> Report (полный отчёт)
    - bounty:ID -> Bounty (задача на исследование)
    
    [OPERATIONS]
    - search(topic) -> List[Report]
    - store(topic, report) -> CID
    - has_topic(topic) -> bool
    - create_bounty(topic, reward) -> Bounty
    - claim_bounty(bounty_id, report) -> bool
    """
    
    # DHT Key Prefixes
    TOPIC_PREFIX = "topic:"
    CONTENT_PREFIX = "content:"
    BOUNTY_PREFIX = "bounty:"
    BOUNTY_LIST_KEY = "bounties:active"
    
    # Configuration
    DEFAULT_BOUNTY_TTL = 3600  # 1 hour
    MAX_REPORTS_PER_TOPIC = 100
    MIN_QUALITY_THRESHOLD = 0.3
    
    def __init__(
        self,
        kademlia_node=None,
        dht_storage: Optional["DHTStorage"] = None,
        ledger: Optional["Ledger"] = None,
        node_id: str = "",
    ):
        """
        Args:
            kademlia_node: KademliaNode для сетевого хранения
            dht_storage: DHTStorage для локального хранения
            ledger: Ledger для транзакций (IOU)
            node_id: ID текущего узла
        """
        self.kademlia = kademlia_node
        self.storage = dht_storage
        self.ledger = ledger
        self.node_id = node_id
        
        # Локальный кэш
        self._topic_cache: Dict[str, TopicIndex] = {}
        self._bounty_cache: Dict[str, Bounty] = {}
        
        # Callbacks
        self._on_bounty_created: List[Callable] = []
        self._on_report_added: List[Callable] = []
    
    # =========================================================================
    # SEARCH & RETRIEVAL
    # =========================================================================
    
    async def search(self, topic: str, limit: int = 10) -> List[Report]:
        """
        Поиск отчётов по теме.
        
        Args:
            topic: Тема для поиска
            limit: Максимум отчётов
        
        Returns:
            Список Report, отсортированный по quality_score
        """
        topic_key = self._make_topic_key(topic)
        
        # Получаем индекс темы
        index = await self._get_topic_index(topic_key)
        
        if not index or not index.cids:
            logger.debug(f"[LIBRARY] No reports found for topic '{topic}'")
            return []
        
        # Загружаем отчёты
        reports = []
        for cid in index.cids[:limit * 2]:  # Загружаем больше для фильтрации
            report = await self.get_report(cid)
            if report and report.quality_score >= self.MIN_QUALITY_THRESHOLD:
                reports.append(report)
            if len(reports) >= limit:
                break
        
        # Сортируем по качеству
        reports.sort(key=lambda r: r.quality_score, reverse=True)
        
        logger.info(f"[LIBRARY] Found {len(reports)} reports for '{topic}'")
        return reports[:limit]
    
    async def get_report(self, cid: str) -> Optional[Report]:
        """Получить отчёт по CID."""
        content_key = f"{self.CONTENT_PREFIX}{cid}"
        
        try:
            data = await self._get(content_key)
            if data:
                report_dict = json.loads(data.decode())
                return Report.from_dict(report_dict)
        except Exception as e:
            logger.warning(f"[LIBRARY] Get report error: {e}")
        
        return None
    
    async def has_topic(self, topic: str) -> bool:
        """Проверить есть ли отчёты по теме."""
        topic_key = self._make_topic_key(topic)
        index = await self._get_topic_index(topic_key)
        return index is not None and len(index.cids) > 0
    
    async def get_topic_stats(self, topic: str) -> Optional[TopicIndex]:
        """Получить статистику по теме."""
        topic_key = self._make_topic_key(topic)
        return await self._get_topic_index(topic_key)
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    async def store(
        self,
        topic: str,
        analysis: Dict[str, Any],
        author_id: str = "",
    ) -> Optional[str]:
        """
        Сохранить анализ в библиотеку.
        
        Args:
            topic: Тема отчёта
            analysis: Результат анализа (от Analyst)
            author_id: ID автора
        
        Returns:
            CID сохранённого отчёта или None
        """
        # Создаём Report
        report = Report(
            cid="",  # Будет вычислен
            topic=topic,
            summary=analysis.get("summary", ""),
            sentiment=analysis.get("sentiment", "neutral"),
            key_facts=analysis.get("key_facts", []),
            topics=analysis.get("topics", []),
            confidence=analysis.get("confidence", 0.5),
            author_id=author_id or self.node_id,
            created_at=time.time(),
            quality_score=self._calculate_quality(analysis),
        )
        
        # Генерируем CID
        report_bytes = json.dumps(report.to_dict(), sort_keys=True).encode()
        cid = hashlib.sha256(report_bytes).hexdigest()
        report.cid = cid
        
        # Обновляем bytes с CID
        report_bytes = json.dumps(report.to_dict(), sort_keys=True).encode()
        
        try:
            # 1. Сохраняем контент
            content_key = f"{self.CONTENT_PREFIX}{cid}"
            await self._put(content_key, report_bytes)
            
            # 2. Обновляем индекс основной темы
            topic_key = self._make_topic_key(topic)
            await self._update_topic_index(topic_key, cid, report.quality_score)
            
            # 3. Индексируем по дополнительным темам
            for t in report.topics[:5]:
                if t.lower() != topic.lower():
                    t_key = self._make_topic_key(t)
                    await self._update_topic_index(t_key, cid, report.quality_score)
            
            logger.info(f"[LIBRARY] Stored report CID={cid[:16]}... topic='{topic}' quality={report.quality_score:.2f}")
            
            # Notify callbacks
            for callback in self._on_report_added:
                try:
                    await callback(report)
                except Exception as e:
                    logger.warning(f"[LIBRARY] Callback error: {e}")
            
            return cid
            
        except Exception as e:
            logger.error(f"[LIBRARY] Store error: {e}")
            return None
    
    def _calculate_quality(self, analysis: Dict[str, Any]) -> float:
        """Вычислить качество отчёта."""
        score = 0.0
        
        # Confidence от LLM
        confidence = analysis.get("confidence", 0.5)
        score += confidence * 0.4
        
        # Количество фактов
        facts = analysis.get("key_facts", [])
        facts_score = min(len(facts) / 5.0, 1.0)  # Норма: 5 фактов
        score += facts_score * 0.3
        
        # Наличие summary
        summary = analysis.get("summary", "")
        if len(summary) > 50:
            score += 0.15
        
        # Наличие entities
        entities = analysis.get("entities", {})
        if entities:
            score += 0.15
        
        return min(score, 1.0)
    
    # =========================================================================
    # BOUNTY SYSTEM
    # =========================================================================
    
    async def create_bounty(
        self,
        topic: str,
        reward: float,
        ttl: float = DEFAULT_BOUNTY_TTL,
        min_confidence: float = 0.5,
    ) -> Optional[Bounty]:
        """
        Создать Bounty-задачу на исследование темы.
        
        Args:
            topic: Тема для исследования
            reward: Награда за выполнение
            ttl: Время жизни bounty в секундах
            min_confidence: Минимальная уверенность отчёта
        
        Returns:
            Bounty или None
        """
        # Генерируем ID
        bounty_id = hashlib.sha256(
            f"{topic}:{self.node_id}:{time.time()}".encode()
        ).hexdigest()[:32]
        
        bounty = Bounty(
            bounty_id=bounty_id,
            topic=topic,
            reward=reward,
            creator_id=self.node_id,
            created_at=time.time(),
            expires_at=time.time() + ttl,
            status=BountyStatus.OPEN,
            min_confidence=min_confidence,
        )
        
        try:
            # Сохраняем bounty
            bounty_key = f"{self.BOUNTY_PREFIX}{bounty_id}"
            await self._put(bounty_key, json.dumps(bounty.to_dict()).encode())
            
            # Добавляем в список активных
            await self._add_to_bounty_list(bounty_id)
            
            # Кэшируем
            self._bounty_cache[bounty_id] = bounty
            
            logger.info(f"[LIBRARY] Created bounty {bounty_id[:16]}... topic='{topic}' reward={reward}")
            
            # Notify callbacks
            for callback in self._on_bounty_created:
                try:
                    await callback(bounty)
                except Exception:
                    pass
            
            return bounty
            
        except Exception as e:
            logger.error(f"[LIBRARY] Create bounty error: {e}")
            return None
    
    async def get_bounty(self, bounty_id: str) -> Optional[Bounty]:
        """Получить bounty по ID."""
        # Проверяем кэш
        if bounty_id in self._bounty_cache:
            bounty = self._bounty_cache[bounty_id]
            if bounty.is_expired() and bounty.status == BountyStatus.OPEN:
                bounty.status = BountyStatus.EXPIRED
            return bounty
        
        # Загружаем из DHT
        bounty_key = f"{self.BOUNTY_PREFIX}{bounty_id}"
        try:
            data = await self._get(bounty_key)
            if data:
                bounty = Bounty.from_dict(json.loads(data.decode()))
                if bounty.is_expired() and bounty.status == BountyStatus.OPEN:
                    bounty.status = BountyStatus.EXPIRED
                self._bounty_cache[bounty_id] = bounty
                return bounty
        except Exception as e:
            logger.warning(f"[LIBRARY] Get bounty error: {e}")
        
        return None
    
    async def get_open_bounties(self, limit: int = 50) -> List[Bounty]:
        """Получить список открытых bounties."""
        bounties = []
        
        try:
            data = await self._get(self.BOUNTY_LIST_KEY)
            if data:
                bounty_ids = json.loads(data.decode())
                
                for bid in bounty_ids[:limit * 2]:
                    bounty = await self.get_bounty(bid)
                    if bounty and bounty.status == BountyStatus.OPEN and not bounty.is_expired():
                        bounties.append(bounty)
                    if len(bounties) >= limit:
                        break
        except Exception as e:
            logger.warning(f"[LIBRARY] Get bounties error: {e}")
        
        # Сортируем по награде
        bounties.sort(key=lambda b: b.reward, reverse=True)
        return bounties
    
    async def claim_bounty(
        self,
        bounty_id: str,
        report_cid: str,
        claimer_id: str,
    ) -> bool:
        """
        Выполнить bounty - предоставить отчёт.
        
        Args:
            bounty_id: ID bounty
            report_cid: CID отчёта
            claimer_id: ID исполнителя
        
        Returns:
            True если успешно
        """
        bounty = await self.get_bounty(bounty_id)
        
        if not bounty:
            logger.warning(f"[LIBRARY] Bounty {bounty_id} not found")
            return False
        
        if bounty.status != BountyStatus.OPEN:
            logger.warning(f"[LIBRARY] Bounty {bounty_id} is not open: {bounty.status}")
            return False
        
        if bounty.is_expired():
            logger.warning(f"[LIBRARY] Bounty {bounty_id} expired")
            return False
        
        # Проверяем качество отчёта
        report = await self.get_report(report_cid)
        if not report:
            logger.warning(f"[LIBRARY] Report {report_cid} not found")
            return False
        
        if report.confidence < bounty.min_confidence:
            logger.warning(
                f"[LIBRARY] Report confidence {report.confidence} < min {bounty.min_confidence}"
            )
            return False
        
        # Обновляем bounty
        bounty.status = BountyStatus.COMPLETED
        bounty.claimed_by = claimer_id
        bounty.result_cid = report_cid
        
        try:
            # Сохраняем обновлённый bounty
            bounty_key = f"{self.BOUNTY_PREFIX}{bounty_id}"
            await self._put(bounty_key, json.dumps(bounty.to_dict()).encode())
            
            # Выплачиваем награду через IOU
            if self.ledger and bounty.reward > 0:
                await self.ledger.create_iou(
                    debtor_id=bounty.creator_id,
                    creditor_id=claimer_id,
                    amount=bounty.reward,
                    description=f"Bounty reward: {bounty.topic}",
                )
            
            logger.info(
                f"[LIBRARY] Bounty {bounty_id[:16]}... completed by {claimer_id[:16]}... "
                f"reward={bounty.reward}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[LIBRARY] Claim bounty error: {e}")
            return False
    
    async def search_or_bounty(
        self,
        topic: str,
        reward: float = 10.0,
    ) -> Tuple[List[Report], Optional[Bounty]]:
        """
        Поиск темы с автоматическим созданием bounty если не найдено.
        
        "Гонка за знаниями" - если тема не найдена, создаётся bounty.
        
        Args:
            topic: Тема для поиска
            reward: Награда за bounty если создаётся
        
        Returns:
            (reports, bounty) - отчёты и/или созданный bounty
        """
        # Сначала ищем
        reports = await self.search(topic)
        
        if reports:
            return reports, None
        
        # Проверяем нет ли уже bounty на эту тему
        existing_bounties = await self.get_open_bounties()
        for b in existing_bounties:
            if b.topic.lower() == topic.lower():
                logger.info(f"[LIBRARY] Bounty already exists for '{topic}'")
                return [], b
        
        # Создаём bounty
        bounty = await self.create_bounty(topic, reward)
        
        return [], bounty
    
    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================
    
    def _make_topic_key(self, topic: str) -> str:
        """Создать ключ темы."""
        normalized = topic.lower().strip().replace(" ", "_")
        return f"{self.TOPIC_PREFIX}{normalized}"
    
    async def _get_topic_index(self, topic_key: str) -> Optional[TopicIndex]:
        """Получить индекс темы."""
        # Проверяем кэш
        if topic_key in self._topic_cache:
            return self._topic_cache[topic_key]
        
        try:
            data = await self._get(topic_key)
            if data:
                index_dict = json.loads(data.decode())
                index = TopicIndex(
                    topic=index_dict.get("topic", ""),
                    topic_key=topic_key,
                    cids=index_dict.get("cids", []),
                    last_updated=index_dict.get("last_updated", 0),
                    total_reports=index_dict.get("total_reports", 0),
                    best_cid=index_dict.get("best_cid"),
                    best_quality=index_dict.get("best_quality", 0.0),
                )
                self._topic_cache[topic_key] = index
                return index
        except Exception as e:
            logger.warning(f"[LIBRARY] Get topic index error: {e}")
        
        return None
    
    async def _update_topic_index(
        self,
        topic_key: str,
        cid: str,
        quality: float,
    ) -> None:
        """Обновить индекс темы."""
        index = await self._get_topic_index(topic_key)
        
        if index is None:
            # Создаём новый индекс
            topic_name = topic_key.replace(self.TOPIC_PREFIX, "").replace("_", " ")
            index = TopicIndex(
                topic=topic_name,
                topic_key=topic_key,
                cids=[],
                last_updated=time.time(),
                total_reports=0,
            )
        
        # Добавляем CID если его нет
        if cid not in index.cids:
            index.cids.insert(0, cid)
            index.total_reports += 1
        
        # Обновляем best если нужно
        if quality > index.best_quality:
            index.best_cid = cid
            index.best_quality = quality
        
        # Ограничиваем размер
        index.cids = index.cids[:self.MAX_REPORTS_PER_TOPIC]
        index.last_updated = time.time()
        
        # Сохраняем
        index_dict = {
            "topic": index.topic,
            "cids": index.cids,
            "last_updated": index.last_updated,
            "total_reports": index.total_reports,
            "best_cid": index.best_cid,
            "best_quality": index.best_quality,
        }
        
        await self._put(topic_key, json.dumps(index_dict).encode())
        
        # Обновляем кэш
        self._topic_cache[topic_key] = index
    
    async def _add_to_bounty_list(self, bounty_id: str) -> None:
        """Добавить bounty в список активных."""
        try:
            data = await self._get(self.BOUNTY_LIST_KEY)
            bounty_ids = json.loads(data.decode()) if data else []
        except Exception:
            bounty_ids = []
        
        if bounty_id not in bounty_ids:
            bounty_ids.insert(0, bounty_id)
            bounty_ids = bounty_ids[:500]  # Максимум 500
        
        await self._put(self.BOUNTY_LIST_KEY, json.dumps(bounty_ids).encode())
    
    async def _get(self, key: str) -> Optional[bytes]:
        """Получить из DHT."""
        if self.kademlia:
            return await self.kademlia.dht_get(key)
        if self.storage:
            return self.storage.get(key)
        return None
    
    async def _put(self, key: str, value: bytes) -> None:
        """Сохранить в DHT."""
        if self.kademlia:
            await self.kademlia.dht_put(key, value)
        elif self.storage:
            self.storage.store(key, value)
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def on_bounty_created(self, callback: Callable) -> None:
        """Регистрация callback на создание bounty."""
        self._on_bounty_created.append(callback)
    
    def on_report_added(self, callback: Callable) -> None:
        """Регистрация callback на добавление отчёта."""
        self._on_report_added.append(callback)
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получить статистику библиотеки."""
        return {
            "cached_topics": len(self._topic_cache),
            "cached_bounties": len(self._bounty_cache),
            "open_bounties": len([
                b for b in self._bounty_cache.values()
                if b.status == BountyStatus.OPEN and not b.is_expired()
            ]),
        }


# Type alias for backwards compatibility
from typing import Tuple

