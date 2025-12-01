"""
Optimized Discovery - Улучшенный механизм обнаружения узлов
===========================================================

[OPTIMIZATION] Улучшения для больших сетей:
- Bloom filter для уже известных узлов (уменьшает дублирование)
- Exponential backoff при повторных запросах
- Ограничение глубины gossip (TTL на сообщениях)
- Приоритизация узлов по Trust Score

[BLOOM] Bloom Filter:
- Вероятностная структура данных
- Быстрая проверка "возможно есть" / "точно нет"
- Экономит память при большом количестве узлов
- False positive rate ~1%

[GOSSIP] Gossip Protocol:
- Периодически обмениваемся информацией о пирах
- TTL ограничивает распространение
- Экспоненциальный backoff предотвращает флуд
"""

import asyncio
import time
import hashlib
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple, TYPE_CHECKING
from collections import defaultdict
import math

if TYPE_CHECKING:
    from core.node import Node, Peer
    from economy.ledger import Ledger

logger = logging.getLogger(__name__)


# Константы
BLOOM_SIZE = 10000  # Ожидаемое количество узлов
BLOOM_FP_RATE = 0.01  # False positive rate 1%
GOSSIP_TTL = 3  # Максимальная глубина распространения
GOSSIP_INTERVAL = 30  # Секунды между gossip раундами
MAX_PEERS_PER_GOSSIP = 10  # Максимум узлов в одном gossip сообщении
BACKOFF_BASE = 2  # База для экспоненциального backoff
BACKOFF_MAX = 3600  # Максимальный backoff (1 час)
DISCOVERY_TIMEOUT = 5  # Таймаут на discovery запрос


class BloomFilter:
    """
    Простая реализация Bloom Filter.
    
    [BLOOM] Вероятностная структура данных:
    - add(item): Добавить элемент
    - contains(item): Проверить наличие
    - False positives возможны
    - False negatives невозможны
    
    [MATH] Оптимальные параметры:
    - m = -n * ln(p) / (ln(2)^2) - размер битового массива
    - k = m/n * ln(2) - количество хэш-функций
    """
    
    def __init__(self, expected_items: int = BLOOM_SIZE, fp_rate: float = BLOOM_FP_RATE):
        """
        Args:
            expected_items: Ожидаемое количество элементов
            fp_rate: Желаемый false positive rate
        """
        # Вычисляем оптимальный размер
        self.size = self._optimal_size(expected_items, fp_rate)
        self.num_hashes = self._optimal_hashes(self.size, expected_items)
        
        # Битовый массив (используем set для простоты)
        self._bits: Set[int] = set()
        self._count = 0
        
        logger.debug(
            f"[BLOOM] Created: size={self.size}, hashes={self.num_hashes}, "
            f"expected_items={expected_items}, fp_rate={fp_rate}"
        )
    
    def _optimal_size(self, n: int, p: float) -> int:
        """Вычислить оптимальный размер битового массива."""
        if n <= 0 or p <= 0 or p >= 1:
            return 10000
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(m) + 1
    
    def _optimal_hashes(self, m: int, n: int) -> int:
        """Вычислить оптимальное количество хэш-функций."""
        if n <= 0:
            return 3
        k = (m / n) * math.log(2)
        return max(1, int(k))
    
    def _hashes(self, item: bytes) -> List[int]:
        """Вычислить k хэшей для элемента."""
        hashes = []
        
        # Используем двойное хэширование: h(i) = h1 + i*h2
        h1 = int.from_bytes(hashlib.sha256(item).digest()[:8], 'big')
        h2 = int.from_bytes(hashlib.sha256(item + b'\x00').digest()[:8], 'big')
        
        for i in range(self.num_hashes):
            h = (h1 + i * h2) % self.size
            hashes.append(h)
        
        return hashes
    
    def add(self, item: bytes) -> None:
        """Добавить элемент в фильтр."""
        for h in self._hashes(item):
            self._bits.add(h)
        self._count += 1
    
    def contains(self, item: bytes) -> bool:
        """
        Проверить наличие элемента.
        
        Returns:
            True = возможно есть (или false positive)
            False = точно нет
        """
        for h in self._hashes(item):
            if h not in self._bits:
                return False
        return True
    
    def __contains__(self, item: bytes) -> bool:
        return self.contains(item)
    
    def __len__(self) -> int:
        return self._count
    
    def clear(self) -> None:
        """Очистить фильтр."""
        self._bits.clear()
        self._count = 0
    
    @property
    def fill_ratio(self) -> float:
        """Процент заполнения."""
        return len(self._bits) / self.size if self.size > 0 else 0


@dataclass
class PeerRecord:
    """
    Запись о пире для Discovery.
    
    [DISCOVERY] Расширенная информация:
    - Базовые данные (host, port, node_id)
    - Время обнаружения и последнего контакта
    - Backoff для повторных запросов
    - Trust Score для приоритизации
    """
    
    node_id: str
    host: str
    port: int
    discovered_at: float = field(default_factory=time.time)
    last_contact: float = field(default_factory=time.time)
    last_attempt: float = 0
    failed_attempts: int = 0
    trust_score: float = 0.5
    
    @property
    def backoff_time(self) -> float:
        """Вычислить время backoff."""
        if self.failed_attempts == 0:
            return 0
        backoff = BACKOFF_BASE ** min(self.failed_attempts, 10)
        return min(backoff, BACKOFF_MAX)
    
    @property
    def can_contact(self) -> bool:
        """Можно ли сейчас связаться с пиром."""
        if self.failed_attempts == 0:
            return True
        return time.time() - self.last_attempt >= self.backoff_time
    
    def mark_success(self) -> None:
        """Отметить успешный контакт."""
        self.last_contact = time.time()
        self.failed_attempts = 0
    
    def mark_failure(self) -> None:
        """Отметить неудачную попытку."""
        self.last_attempt = time.time()
        self.failed_attempts += 1
    
    def to_dict(self) -> Dict:
        """Сериализация для gossip."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "trust_score": self.trust_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PeerRecord":
        """Десериализация."""
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            trust_score=data.get("trust_score", 0.5),
        )


@dataclass
class GossipMessage:
    """
    Gossip сообщение для обмена информацией о пирах.
    
    [GOSSIP] Содержит:
    - Список известных пиров
    - TTL для ограничения распространения
    - ID отправителя для дедупликации
    """
    
    peers: List[Dict]
    ttl: int = GOSSIP_TTL
    sender_id: str = ""
    message_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.message_id:
            # Генерируем уникальный ID сообщения
            data = f"{self.sender_id}:{self.timestamp}:{len(self.peers)}"
            self.message_id = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def decrement_ttl(self) -> "GossipMessage":
        """Создать копию с уменьшенным TTL."""
        return GossipMessage(
            peers=self.peers,
            ttl=self.ttl - 1,
            sender_id=self.sender_id,
            message_id=self.message_id,
            timestamp=self.timestamp,
        )
    
    def to_dict(self) -> Dict:
        return {
            "type": "GOSSIP",
            "peers": self.peers,
            "ttl": self.ttl,
            "sender_id": self.sender_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GossipMessage":
        return cls(
            peers=data.get("peers", []),
            ttl=data.get("ttl", GOSSIP_TTL),
            sender_id=data.get("sender_id", ""),
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", time.time()),
        )


class OptimizedDiscovery:
    """
    Оптимизированный механизм обнаружения узлов.
    
    [OPTIMIZATION] Улучшения:
    1. Bloom filter - быстрая проверка известных узлов
    2. Exponential backoff - защита от флуда
    3. TTL на gossip - ограничение распространения
    4. Trust-based priority - приоритет надёжным узлам
    
    [USAGE]
    ```python
    discovery = OptimizedDiscovery(node, ledger)
    await discovery.start()
    
    # Bootstrap
    await discovery.bootstrap([("127.0.0.1", 8468)])
    
    # Получить пиров для подключения
    peers = discovery.get_peers_to_connect(count=5)
    ```
    """
    
    def __init__(
        self,
        node: 'Node',
        ledger: Optional['Ledger'] = None,
    ):
        """
        Args:
            node: Базовый P2P узел
            ledger: Ledger для получения Trust Score (опционально)
        """
        self.node = node
        self.ledger = ledger
        
        # Bloom filter для известных узлов
        self.known_bloom = BloomFilter(BLOOM_SIZE, BLOOM_FP_RATE)
        
        # Полный реестр известных пиров
        self._peers: Dict[str, PeerRecord] = {}
        
        # Bloom filter для обработанных gossip сообщений
        self._seen_gossip = BloomFilter(1000, 0.01)
        
        # Фоновые задачи
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        logger.info("[DISCOVERY] OptimizedDiscovery created")
    
    async def start(self) -> None:
        """Запустить discovery."""
        if self._running:
            return
        
        self._running = True
        
        # Запускаем фоновый gossip
        self._tasks.append(asyncio.create_task(self._gossip_loop()))
        
        # Запускаем периодическую очистку
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))
        
        logger.info("[DISCOVERY] Started")
    
    async def stop(self) -> None:
        """Остановить discovery."""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("[DISCOVERY] Stopped")
    
    # =========================================================================
    # Peer Management
    # =========================================================================
    
    def add_peer(self, node_id: str, host: str, port: int, trust_score: float = 0.5) -> bool:
        """
        Добавить пира в реестр.
        
        [BLOOM] Используем bloom filter для быстрой проверки.
        
        Returns:
            True если пир новый
        """
        peer_key = f"{host}:{port}".encode()
        
        # Быстрая проверка через Bloom filter
        if peer_key in self.known_bloom:
            # Возможно уже есть - проверяем точно
            if node_id in self._peers:
                # Обновляем существующего
                self._peers[node_id].trust_score = trust_score
                return False
        
        # Новый пир
        self.known_bloom.add(peer_key)
        self._peers[node_id] = PeerRecord(
            node_id=node_id,
            host=host,
            port=port,
            trust_score=trust_score,
        )
        
        logger.debug(f"[DISCOVERY] Added peer: {node_id[:16]}... @ {host}:{port}")
        return True
    
    def remove_peer(self, node_id: str) -> bool:
        """Удалить пира из реестра."""
        if node_id in self._peers:
            del self._peers[node_id]
            return True
        return False
    
    def get_peer(self, node_id: str) -> Optional[PeerRecord]:
        """Получить информацию о пире."""
        return self._peers.get(node_id)
    
    def mark_peer_success(self, node_id: str) -> None:
        """Отметить успешный контакт с пиром."""
        if node_id in self._peers:
            self._peers[node_id].mark_success()
    
    def mark_peer_failure(self, node_id: str) -> None:
        """Отметить неудачную попытку связи."""
        if node_id in self._peers:
            self._peers[node_id].mark_failure()
    
    def get_peers_to_connect(self, count: int = 5) -> List[PeerRecord]:
        """
        Получить пиров для подключения.
        
        [PRIORITY] Приоритет:
        1. Высокий Trust Score
        2. Можно связаться (backoff прошёл)
        3. Давно не связывались
        
        Args:
            count: Количество пиров
        
        Returns:
            Список PeerRecord для подключения
        """
        # Фильтруем: можно связаться и не подключены
        active_ids = {p.node_id for p in self.node.peer_manager.get_active_peers()}
        
        candidates = [
            p for p in self._peers.values()
            if p.can_contact and p.node_id not in active_ids
        ]
        
        # Сортируем по Trust Score (убывание) и времени последнего контакта
        candidates.sort(
            key=lambda p: (p.trust_score, -p.last_contact),
            reverse=True
        )
        
        return candidates[:count]
    
    def get_random_peers(self, count: int = MAX_PEERS_PER_GOSSIP) -> List[PeerRecord]:
        """Получить случайных пиров для gossip."""
        peers = list(self._peers.values())
        if len(peers) <= count:
            return peers
        return random.sample(peers, count)
    
    # =========================================================================
    # Bootstrap
    # =========================================================================
    
    async def bootstrap(self, nodes: List[Tuple[str, int]]) -> int:
        """
        Bootstrap с известными узлами.
        
        Args:
            nodes: Список (host, port) bootstrap узлов
        
        Returns:
            Количество успешных подключений
        """
        success_count = 0
        
        for host, port in nodes:
            try:
                peer = await asyncio.wait_for(
                    self.node.connect_to_peer(host, port),
                    timeout=DISCOVERY_TIMEOUT
                )
                
                if peer:
                    self.add_peer(
                        node_id=peer.node_id,
                        host=host,
                        port=port,
                    )
                    success_count += 1
                    logger.info(f"[DISCOVERY] Bootstrap success: {host}:{port}")
                    
            except asyncio.TimeoutError:
                logger.debug(f"[DISCOVERY] Bootstrap timeout: {host}:{port}")
            except Exception as e:
                logger.debug(f"[DISCOVERY] Bootstrap failed {host}:{port}: {e}")
        
        return success_count
    
    # =========================================================================
    # Gossip Protocol
    # =========================================================================
    
    async def handle_gossip(self, message: GossipMessage) -> Optional[GossipMessage]:
        """
        Обработать входящее gossip сообщение.
        
        [GOSSIP] Логика:
        1. Проверяем, не видели ли мы это сообщение
        2. Добавляем новых пиров
        3. Если TTL > 0, пересылаем дальше
        
        Args:
            message: Входящее gossip сообщение
        
        Returns:
            Сообщение для пересылки или None
        """
        msg_id = message.message_id.encode()
        
        # Проверяем через Bloom filter
        if msg_id in self._seen_gossip:
            return None
        
        self._seen_gossip.add(msg_id)
        
        # Добавляем пиров из сообщения
        new_count = 0
        for peer_data in message.peers:
            try:
                if self.add_peer(
                    node_id=peer_data["node_id"],
                    host=peer_data["host"],
                    port=peer_data["port"],
                    trust_score=peer_data.get("trust_score", 0.5),
                ):
                    new_count += 1
            except (KeyError, TypeError):
                continue
        
        if new_count > 0:
            logger.debug(f"[DISCOVERY] Gossip: added {new_count} new peers")
        
        # Пересылаем если TTL > 0
        if message.ttl > 1:
            return message.decrement_ttl()
        
        return None
    
    async def send_gossip(self) -> int:
        """
        Отправить gossip сообщение случайным пирам.
        
        Returns:
            Количество отправленных сообщений
        """
        # Получаем подключенных пиров
        active_peers = self.node.peer_manager.get_active_peers()
        
        if not active_peers:
            return 0
        
        # Создаём gossip сообщение
        peers_to_share = self.get_random_peers(MAX_PEERS_PER_GOSSIP)
        
        message = GossipMessage(
            peers=[p.to_dict() for p in peers_to_share],
            ttl=GOSSIP_TTL,
            sender_id=self.node.node_id,
        )
        
        # Отправляем нескольким случайным пирам
        targets = random.sample(
            active_peers,
            min(3, len(active_peers))
        )
        
        sent = 0
        for peer in targets:
            try:
                from core.transport import Message, MessageType
                
                msg = Message(
                    type=MessageType.DISCOVER,
                    payload=message.to_dict(),
                    sender_id=self.node.node_id,
                )
                
                success = await peer.send(msg)
                if success:
                    sent += 1
                    
            except Exception as e:
                logger.debug(f"[DISCOVERY] Gossip send failed: {e}")
        
        if sent > 0:
            logger.debug(f"[DISCOVERY] Sent gossip to {sent} peers")
        
        return sent
    
    # =========================================================================
    # Background Tasks
    # =========================================================================
    
    async def _gossip_loop(self) -> None:
        """Фоновая задача: периодический gossip."""
        while self._running:
            try:
                await asyncio.sleep(GOSSIP_INTERVAL)
                
                if not self._running:
                    break
                
                await self.send_gossip()
                
                # Пытаемся подключиться к новым пирам
                candidates = self.get_peers_to_connect(3)
                for peer_record in candidates:
                    if not self._running:
                        break
                    
                    try:
                        peer = await asyncio.wait_for(
                            self.node.connect_to_peer(
                                peer_record.host,
                                peer_record.port
                            ),
                            timeout=DISCOVERY_TIMEOUT
                        )
                        
                        if peer:
                            peer_record.mark_success()
                        else:
                            peer_record.mark_failure()
                            
                    except asyncio.TimeoutError:
                        peer_record.mark_failure()
                    except Exception:
                        peer_record.mark_failure()
                    
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[DISCOVERY] Gossip loop error: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Фоновая задача: очистка старых данных."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Каждый час
                
                if not self._running:
                    break
                
                # Удаляем пиров, которые не отвечают долго
                stale_threshold = time.time() - 86400  # 24 часа
                to_remove = []
                
                for node_id, peer in self._peers.items():
                    if peer.last_contact < stale_threshold and peer.failed_attempts > 5:
                        to_remove.append(node_id)
                
                for node_id in to_remove:
                    del self._peers[node_id]
                
                if to_remove:
                    logger.info(f"[DISCOVERY] Cleaned up {len(to_remove)} stale peers")
                
                # Пересоздаём Bloom filter если слишком заполнен
                if self._seen_gossip.fill_ratio > 0.5:
                    self._seen_gossip.clear()
                    logger.debug("[DISCOVERY] Reset gossip bloom filter")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[DISCOVERY] Cleanup error: {e}")
    
    # =========================================================================
    # Stats
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Получить статистику discovery."""
        active_peers = len(self.node.peer_manager.get_active_peers())
        
        return {
            "known_peers": len(self._peers),
            "active_peers": active_peers,
            "bloom_fill_ratio": self.known_bloom.fill_ratio,
            "gossip_bloom_fill_ratio": self._seen_gossip.fill_ratio,
            "running": self._running,
        }
    
    async def update_trust_scores(self) -> None:
        """Обновить Trust Score из Ledger."""
        if not self.ledger:
            return
        
        for node_id, peer in self._peers.items():
            try:
                score = await self.ledger.get_trust_score(node_id)
                peer.trust_score = score
            except Exception:
                pass

