"""
Ledger - Экономический модуль P2P сети
======================================

[DECENTRALIZATION] Этот модуль реализует децентрализованную экономику:
- Локальная база данных для каждого узла
- Trust Score на основе поведения пиров
- IOU (долговые расписки) с криптографическими подписями

[SECURITY] Все транзакции подписываются приватным ключом.
Каждый узел может независимо верифицировать подписи.
"""

import asyncio
import time
import json
import hashlib
import base64
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import aiosqlite

from config import config


@dataclass
class Transaction:
    """
    Транзакция в сети.
    
    [DECENTRALIZATION] Транзакции хранятся локально на каждом узле.
    Нет центрального реестра - согласованность достигается через
    gossip-протокол и подписи.
    """
    
    id: str  # SHA256 хеш содержимого
    from_id: str  # node_id отправителя
    to_id: str  # node_id получателя
    amount: float  # Количество "кредитов"
    timestamp: float
    signature: str  # Подпись отправителя
    tx_type: str = "transfer"  # Тип транзакции
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "tx_type": self.tx_type,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transaction":
        return cls(
            id=data["id"],
            from_id=data["from_id"],
            to_id=data["to_id"],
            amount=data["amount"],
            timestamp=data["timestamp"],
            signature=data["signature"],
            tx_type=data.get("tx_type", "transfer"),
            metadata=data.get("metadata", {}),
        )
    
    def get_signing_data(self) -> bytes:
        """Данные для подписи (все кроме id и signature)."""
        data = {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "tx_type": self.tx_type,
            "metadata": self.metadata,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    
    @staticmethod
    def compute_id(signing_data: bytes) -> str:
        """Вычислить ID транзакции как SHA256 хеш."""
        return hashlib.sha256(signing_data).hexdigest()


@dataclass 
class IOU:
    """
    IOU (I Owe You) - Долговая расписка.
    
    [DECENTRALIZATION] IOU позволяет узлам обмениваться "кредитами"
    без центрального банка:
    
    1. Узел A выполняет работу для узла B (передает трафик)
    2. Узел B создает IOU: "Я должен A 10 кредитов"
    3. IOU подписывается приватным ключом B
    4. A может предъявить IOU для получения услуг от B
    5. При погашении IOU помечается как redeemed
    
    [SECURITY] IOU защищен криптографически:
    - Подпись гарантирует, что IOU создан владельцем ключа
    - ID = хеш содержимого, исключает подделку
    - Нельзя погасить дважды (redeemed flag)
    """
    
    id: str  # SHA256 хеш содержимого
    debtor_id: str  # Кто должен (создатель IOU)
    creditor_id: str  # Кому должен
    amount: float  # Сумма долга
    created_at: float  # Время создания
    expires_at: Optional[float]  # Срок действия (None = бессрочно)
    signature: str  # Подпись должника
    redeemed: bool = False  # Погашено ли
    redeemed_at: Optional[float] = None  # Когда погашено
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "debtor_id": self.debtor_id,
            "creditor_id": self.creditor_id,
            "amount": self.amount,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "signature": self.signature,
            "redeemed": self.redeemed,
            "redeemed_at": self.redeemed_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IOU":
        return cls(
            id=data["id"],
            debtor_id=data["debtor_id"],
            creditor_id=data["creditor_id"],
            amount=data["amount"],
            created_at=data["created_at"],
            expires_at=data.get("expires_at"),
            signature=data["signature"],
            redeemed=data.get("redeemed", False),
            redeemed_at=data.get("redeemed_at"),
        )
    
    def get_signing_data(self) -> bytes:
        """Данные для подписи."""
        data = {
            "debtor_id": self.debtor_id,
            "creditor_id": self.creditor_id,
            "amount": self.amount,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    
    @staticmethod
    def compute_id(signing_data: bytes) -> str:
        """Вычислить ID как SHA256 хеш."""
        return hashlib.sha256(signing_data).hexdigest()
    
    def is_expired(self) -> bool:
        """Проверить, истек ли срок действия."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def is_valid(self) -> bool:
        """Проверить, валиден ли IOU для погашения."""
        return not self.redeemed and not self.is_expired()


class TrustScore:
    """
    Система Trust Score (репутации).
    
    [DECENTRALIZATION] Trust Score позволяет узлам оценивать
    надежность друг друга без центрального арбитра:
    
    - Каждый узел ведет свой учет репутации пиров
    - Score увеличивается за успешные взаимодействия
    - Score уменьшается за неудачи/обман
    - Узлы предпочитают работать с высоким Trust Score
    
    [SECURITY] Trust Score защищает от:
    - Sybil-атак (новые узлы имеют низкий score)
    - Недобросовестных узлов (score падает)
    - Eclipse-атак (выбираем diverse пиров)
    """
    
    # Веса для разных событий
    WEIGHTS = {
        "successful_transfer": 0.01,      # Успешная передача данных
        "failed_transfer": -0.05,         # Неудачная передача
        "valid_message": 0.001,           # Валидное сообщение
        "invalid_message": -0.02,         # Невалидное сообщение
        "iou_created": 0.005,             # Создан IOU
        "iou_redeemed": 0.02,             # IOU погашен
        "iou_defaulted": -0.1,            # IOU просрочен
        "ping_responded": 0.001,          # Ответил на PING
        "ping_timeout": -0.01,            # Не ответил на PING
    }
    
    @classmethod
    def calculate_adjustment(cls, event: str, magnitude: float = 1.0) -> float:
        """
        Рассчитать изменение Trust Score.
        
        Args:
            event: Тип события
            magnitude: Множитель (например, размер транзакции)
        
        Returns:
            Изменение score
        """
        weight = cls.WEIGHTS.get(event, 0)
        return weight * magnitude
    
    @staticmethod
    def clamp(score: float) -> float:
        """Ограничить score в диапазоне [0, 1]."""
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def decay(score: float, days_inactive: float) -> float:
        """
        Применить decay к score за неактивность.
        
        [DECENTRALIZATION] Decay важен для:
        - Удаления "мертвых" узлов из сети
        - Поддержания актуальности репутации
        """
        # Экспоненциальный decay: score *= 0.99^days
        decay_factor = 0.99 ** days_inactive
        return score * decay_factor


class Ledger:
    """
    Локальный реестр транзакций.
    
    [DECENTRALIZATION] Каждый узел хранит свою копию реестра:
    - Транзакции с участием этого узла
    - IOU где узел является debtor или creditor
    - Trust Score для известных пиров
    
    Нет "главной" копии - каждый узел авторитетен для своих данных.
    Согласованность достигается через криптографические подписи.
    """
    
    def __init__(self, db_path: str = "ledger.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """
        Инициализировать базу данных.
        
        Создает таблицы если они не существуют.
        """
        self._db = await aiosqlite.connect(self.db_path)
        
        # Включаем WAL mode для лучшей производительности
        await self._db.execute("PRAGMA journal_mode=WAL")
        
        # Таблица пиров и их Trust Score
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS peers (
                node_id TEXT PRIMARY KEY,
                public_key TEXT,
                trust_score REAL DEFAULT 0.5,
                total_sent REAL DEFAULT 0,
                total_received REAL DEFAULT 0,
                first_seen REAL,
                last_seen REAL,
                metadata TEXT
            )
        """)
        
        # Таблица транзакций
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                amount REAL NOT NULL,
                timestamp REAL NOT NULL,
                signature TEXT NOT NULL,
                tx_type TEXT DEFAULT 'transfer',
                metadata TEXT,
                FOREIGN KEY (from_id) REFERENCES peers(node_id),
                FOREIGN KEY (to_id) REFERENCES peers(node_id)
            )
        """)
        
        # Таблица IOU
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS ious (
                id TEXT PRIMARY KEY,
                debtor_id TEXT NOT NULL,
                creditor_id TEXT NOT NULL,
                amount REAL NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                signature TEXT NOT NULL,
                redeemed INTEGER DEFAULT 0,
                redeemed_at REAL,
                FOREIGN KEY (debtor_id) REFERENCES peers(node_id),
                FOREIGN KEY (creditor_id) REFERENCES peers(node_id)
            )
        """)
        
        # Индексы для быстрого поиска
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_to ON transactions(to_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_iou_debtor ON ious(debtor_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_iou_creditor ON ious(creditor_id)"
        )
        
        await self._db.commit()
    
    async def close(self) -> None:
        """Закрыть соединение с базой данных."""
        if self._db:
            await self._db.close()
            self._db = None
    
    # --- Peer Management ---
    
    async def get_or_create_peer(
        self,
        node_id: str,
        public_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Получить или создать запись о пире.
        
        [DECENTRALIZATION] Новые пиры начинают с базовым Trust Score.
        Репутация строится со временем.
        """
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT * FROM peers WHERE node_id = ?",
                (node_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                return {
                    "node_id": row[0],
                    "public_key": row[1],
                    "trust_score": row[2],
                    "total_sent": row[3],
                    "total_received": row[4],
                    "first_seen": row[5],
                    "last_seen": row[6],
                }
            
            # Создаем новую запись
            now = time.time()
            await self._db.execute(
                """
                INSERT INTO peers (node_id, public_key, trust_score, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
                """,
                (node_id, public_key, config.ledger.initial_trust_score, now, now)
            )
            await self._db.commit()
            
            return {
                "node_id": node_id,
                "public_key": public_key,
                "trust_score": config.ledger.initial_trust_score,
                "total_sent": 0,
                "total_received": 0,
                "first_seen": now,
                "last_seen": now,
            }
    
    async def update_trust_score(
        self,
        node_id: str,
        event: str,
        magnitude: float = 1.0,
    ) -> float:
        """
        Обновить Trust Score пира.
        
        Returns:
            Новый Trust Score
        """
        async with self._lock:
            # Получаем текущий score
            cursor = await self._db.execute(
                "SELECT trust_score FROM peers WHERE node_id = ?",
                (node_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return config.ledger.initial_trust_score
            
            current_score = row[0]
            adjustment = TrustScore.calculate_adjustment(event, magnitude)
            new_score = TrustScore.clamp(current_score + adjustment)
            
            await self._db.execute(
                "UPDATE peers SET trust_score = ?, last_seen = ? WHERE node_id = ?",
                (new_score, time.time(), node_id)
            )
            await self._db.commit()
            
            return new_score
    
    async def get_trust_score(self, node_id: str) -> float:
        """Получить Trust Score пира."""
        cursor = await self._db.execute(
            "SELECT trust_score FROM peers WHERE node_id = ?",
            (node_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else config.ledger.initial_trust_score
    
    async def get_peers_by_trust(self, min_score: float = 0.0, limit: int = 100) -> List[Dict[str, Any]]:
        """Получить пиров с Trust Score выше порога."""
        cursor = await self._db.execute(
            """
            SELECT node_id, trust_score, last_seen 
            FROM peers 
            WHERE trust_score >= ?
            ORDER BY trust_score DESC
            LIMIT ?
            """,
            (min_score, limit)
        )
        rows = await cursor.fetchall()
        return [
            {"node_id": r[0], "trust_score": r[1], "last_seen": r[2]}
            for r in rows
        ]
    
    # --- Transaction Management ---
    
    async def add_transaction(self, tx: Transaction) -> bool:
        """
        Добавить транзакцию в реестр.
        
        [SECURITY] Транзакция должна быть подписана отправителем.
        Валидация подписи выполняется на уровне протокола.
        
        Returns:
            True если транзакция добавлена
        """
        async with self._lock:
            try:
                await self._db.execute(
                    """
                    INSERT INTO transactions 
                    (id, from_id, to_id, amount, timestamp, signature, tx_type, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tx.id,
                        tx.from_id,
                        tx.to_id,
                        tx.amount,
                        tx.timestamp,
                        tx.signature,
                        tx.tx_type,
                        json.dumps(tx.metadata),
                    )
                )
                
                # Обновляем статистику пиров
                await self._db.execute(
                    "UPDATE peers SET total_sent = total_sent + ? WHERE node_id = ?",
                    (tx.amount, tx.from_id)
                )
                await self._db.execute(
                    "UPDATE peers SET total_received = total_received + ? WHERE node_id = ?",
                    (tx.amount, tx.to_id)
                )
                
                await self._db.commit()
                return True
                
            except aiosqlite.IntegrityError:
                # Транзакция уже существует
                return False
    
    async def get_transaction(self, tx_id: str) -> Optional[Transaction]:
        """Получить транзакцию по ID."""
        cursor = await self._db.execute(
            "SELECT * FROM transactions WHERE id = ?",
            (tx_id,)
        )
        row = await cursor.fetchone()
        
        if not row:
            return None
        
        return Transaction(
            id=row[0],
            from_id=row[1],
            to_id=row[2],
            amount=row[3],
            timestamp=row[4],
            signature=row[5],
            tx_type=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
        )
    
    async def get_transactions_for_peer(
        self,
        node_id: str,
        limit: int = 100,
    ) -> List[Transaction]:
        """Получить транзакции с участием пира."""
        cursor = await self._db.execute(
            """
            SELECT * FROM transactions 
            WHERE from_id = ? OR to_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (node_id, node_id, limit)
        )
        rows = await cursor.fetchall()
        
        return [
            Transaction(
                id=row[0],
                from_id=row[1],
                to_id=row[2],
                amount=row[3],
                timestamp=row[4],
                signature=row[5],
                tx_type=row[6],
                metadata=json.loads(row[7]) if row[7] else {},
            )
            for row in rows
        ]
    
    # --- IOU Management ---
    
    async def create_iou(
        self,
        debtor_id: str,
        creditor_id: str,
        amount: float,
        signature: str,
        expires_at: Optional[float] = None,
    ) -> IOU:
        """
        Создать IOU.
        
        [SECURITY] IOU подписывается должником (debtor).
        Подпись доказывает, что должник признает долг.
        
        Returns:
            Созданный IOU
        """
        created_at = time.time()
        
        # Создаем IOU для вычисления ID
        iou = IOU(
            id="",  # Будет вычислен
            debtor_id=debtor_id,
            creditor_id=creditor_id,
            amount=amount,
            created_at=created_at,
            expires_at=expires_at,
            signature=signature,
        )
        
        # Вычисляем ID
        iou.id = IOU.compute_id(iou.get_signing_data())
        
        # Сохраняем в базу
        async with self._lock:
            await self._db.execute(
                """
                INSERT INTO ious 
                (id, debtor_id, creditor_id, amount, created_at, expires_at, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iou.id,
                    iou.debtor_id,
                    iou.creditor_id,
                    iou.amount,
                    iou.created_at,
                    iou.expires_at,
                    iou.signature,
                )
            )
            await self._db.commit()
        
        return iou
    
    async def get_iou(self, iou_id: str) -> Optional[IOU]:
        """Получить IOU по ID."""
        cursor = await self._db.execute(
            "SELECT * FROM ious WHERE id = ?",
            (iou_id,)
        )
        row = await cursor.fetchone()
        
        if not row:
            return None
        
        return IOU(
            id=row[0],
            debtor_id=row[1],
            creditor_id=row[2],
            amount=row[3],
            created_at=row[4],
            expires_at=row[5],
            signature=row[6],
            redeemed=bool(row[7]),
            redeemed_at=row[8],
        )
    
    async def redeem_iou(self, iou_id: str) -> bool:
        """
        Погасить IOU.
        
        [SECURITY] После погашения IOU помечается как redeemed
        и не может быть использован повторно.
        
        Returns:
            True если IOU успешно погашен
        """
        async with self._lock:
            # Проверяем, что IOU существует и не погашен
            cursor = await self._db.execute(
                "SELECT redeemed, expires_at FROM ious WHERE id = ?",
                (iou_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return False
            
            if row[0]:  # Уже погашен
                return False
            
            if row[1] and time.time() > row[1]:  # Истек срок
                return False
            
            # Помечаем как погашенный
            await self._db.execute(
                "UPDATE ious SET redeemed = 1, redeemed_at = ? WHERE id = ?",
                (time.time(), iou_id)
            )
            await self._db.commit()
            
            return True
    
    async def get_ious_as_creditor(
        self,
        creditor_id: str,
        only_valid: bool = True,
    ) -> List[IOU]:
        """Получить IOU где узел является кредитором."""
        if only_valid:
            cursor = await self._db.execute(
                """
                SELECT * FROM ious 
                WHERE creditor_id = ? AND redeemed = 0 
                AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY created_at DESC
                """,
                (creditor_id, time.time())
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM ious WHERE creditor_id = ? ORDER BY created_at DESC",
                (creditor_id,)
            )
        
        rows = await cursor.fetchall()
        return [
            IOU(
                id=row[0],
                debtor_id=row[1],
                creditor_id=row[2],
                amount=row[3],
                created_at=row[4],
                expires_at=row[5],
                signature=row[6],
                redeemed=bool(row[7]),
                redeemed_at=row[8],
            )
            for row in rows
        ]
    
    async def get_ious_as_debtor(
        self,
        debtor_id: str,
        only_valid: bool = True,
    ) -> List[IOU]:
        """Получить IOU где узел является должником."""
        if only_valid:
            cursor = await self._db.execute(
                """
                SELECT * FROM ious 
                WHERE debtor_id = ? AND redeemed = 0
                AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY created_at DESC
                """,
                (debtor_id, time.time())
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM ious WHERE debtor_id = ? ORDER BY created_at DESC",
                (debtor_id,)
            )
        
        rows = await cursor.fetchall()
        return [
            IOU(
                id=row[0],
                debtor_id=row[1],
                creditor_id=row[2],
                amount=row[3],
                created_at=row[4],
                expires_at=row[5],
                signature=row[6],
                redeemed=bool(row[7]),
                redeemed_at=row[8],
            )
            for row in rows
        ]
    
    async def get_total_debt(self, node_id: str) -> Tuple[float, float]:
        """
        Получить общий долг узла.
        
        Returns:
            (owed_to_others, owed_by_others) - должен другим, должны узлу
        """
        # Сколько должен узел другим
        cursor = await self._db.execute(
            """
            SELECT COALESCE(SUM(amount), 0) FROM ious 
            WHERE debtor_id = ? AND redeemed = 0
            AND (expires_at IS NULL OR expires_at > ?)
            """,
            (node_id, time.time())
        )
        row = await cursor.fetchone()
        owed_to_others = row[0]
        
        # Сколько должны узлу
        cursor = await self._db.execute(
            """
            SELECT COALESCE(SUM(amount), 0) FROM ious 
            WHERE creditor_id = ? AND redeemed = 0
            AND (expires_at IS NULL OR expires_at > ?)
            """,
            (node_id, time.time())
        )
        row = await cursor.fetchone()
        owed_by_others = row[0]
        
        return (owed_to_others, owed_by_others)
    
    # --- Statistics ---
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получить статистику реестра."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM peers")
        peer_count = (await cursor.fetchone())[0]
        
        cursor = await self._db.execute("SELECT COUNT(*) FROM transactions")
        tx_count = (await cursor.fetchone())[0]
        
        cursor = await self._db.execute("SELECT COUNT(*) FROM ious WHERE redeemed = 0")
        active_ious = (await cursor.fetchone())[0]
        
        cursor = await self._db.execute("SELECT SUM(amount) FROM ious WHERE redeemed = 0")
        total_debt = (await cursor.fetchone())[0] or 0
        
        return {
            "peer_count": peer_count,
            "transaction_count": tx_count,
            "active_ious": active_ious,
            "total_outstanding_debt": total_debt,
        }

