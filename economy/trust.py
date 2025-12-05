"""
Weighted Trust Score System
===========================

[SECURITY] Защита от Sybil-атак через экономическое взвешивание.

Формула эффективного доверия:
    T_effective = T_behavior * log10(1 + Stake / BaseStake)

Где:
- T_behavior: Скользящее среднее (EMA) успешных взаимодействий [0.0 - 1.0]
- Stake: Баланс узла (IOU + on-chain ZEO)
- BaseStake: Базовый порог стейкинга (по умолчанию 100 ZEO)

[ECONOMY] Dust Limit:
- Если баланс < DUST_LIMIT (10 ZEO), множитель стремится к 0.1
- Это делает Sybil-атаку экономически невыгодной

[SLASHING] При обнаружении INVALID_MERKLE_PROOF:
- Trust Score обнуляется мгновенно
- Peer добавляется в blacklist
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from enum import Enum, auto

logger = logging.getLogger(__name__)


class TrustEvent(Enum):
    """События, влияющие на Trust Score."""
    
    # Позитивные события
    SUCCESSFUL_TRANSFER = auto()      # +0.01
    VALID_MESSAGE = auto()            # +0.001
    IOU_CREATED = auto()              # +0.005
    IOU_REDEEMED = auto()             # +0.02
    PING_RESPONDED = auto()           # +0.001
    DEBT_REPAID = auto()              # +0.02
    VALID_MERKLE_PROOF = auto()       # +0.005
    
    # Негативные события
    FAILED_TRANSFER = auto()          # -0.05
    INVALID_MESSAGE = auto()          # -0.02
    IOU_DEFAULTED = auto()            # -0.1
    PING_TIMEOUT = auto()             # -0.01
    EXCESSIVE_DEBT = auto()           # -0.03
    
    # [CRITICAL] Slashing events - мгновенное обнуление
    INVALID_MERKLE_PROOF = auto()     # -> 0.0 (SLASH)
    DOUBLE_SPEND_ATTEMPT = auto()     # -> 0.0 (SLASH)
    SIGNATURE_FORGERY = auto()        # -> 0.0 (SLASH)


# Веса событий для EMA
EVENT_WEIGHTS: Dict[TrustEvent, float] = {
    # Позитивные
    TrustEvent.SUCCESSFUL_TRANSFER: 0.01,
    TrustEvent.VALID_MESSAGE: 0.001,
    TrustEvent.IOU_CREATED: 0.005,
    TrustEvent.IOU_REDEEMED: 0.02,
    TrustEvent.PING_RESPONDED: 0.001,
    TrustEvent.DEBT_REPAID: 0.02,
    TrustEvent.VALID_MERKLE_PROOF: 0.005,
    
    # Негативные
    TrustEvent.FAILED_TRANSFER: -0.05,
    TrustEvent.INVALID_MESSAGE: -0.02,
    TrustEvent.IOU_DEFAULTED: -0.1,
    TrustEvent.PING_TIMEOUT: -0.01,
    TrustEvent.EXCESSIVE_DEBT: -0.03,
    
    # Slashing (специальная обработка)
    TrustEvent.INVALID_MERKLE_PROOF: -1.0,
    TrustEvent.DOUBLE_SPEND_ATTEMPT: -1.0,
    TrustEvent.SIGNATURE_FORGERY: -1.0,
}

# События, требующие мгновенного слэшинга
SLASHING_EVENTS = {
    TrustEvent.INVALID_MERKLE_PROOF,
    TrustEvent.DOUBLE_SPEND_ATTEMPT,
    TrustEvent.SIGNATURE_FORGERY,
}


@dataclass
class PeerTrustState:
    """
    Состояние доверия для конкретного пира.
    
    [PERSISTENCE] Сохраняется в SQLite через Ledger.
    """
    
    peer_id: str
    behavior_score: float = 0.5       # T_behavior [0.0 - 1.0]
    stake_balance: float = 0.0        # Баланс (IOU + on-chain)
    interaction_count: int = 0         # Количество взаимодействий
    last_interaction: float = field(default_factory=time.time)
    slashed: bool = False              # Был ли слэшинг
    slash_reason: Optional[str] = None
    slash_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "behavior_score": self.behavior_score,
            "stake_balance": self.stake_balance,
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction,
            "slashed": self.slashed,
            "slash_reason": self.slash_reason,
            "slash_timestamp": self.slash_timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeerTrustState":
        return cls(
            peer_id=data["peer_id"],
            behavior_score=data.get("behavior_score", 0.5),
            stake_balance=data.get("stake_balance", 0.0),
            interaction_count=data.get("interaction_count", 0),
            last_interaction=data.get("last_interaction", time.time()),
            slashed=data.get("slashed", False),
            slash_reason=data.get("slash_reason"),
            slash_timestamp=data.get("slash_timestamp"),
        )


class WeightedTrustScore:
    """
    Система взвешенного доверия.
    
    [SECURITY] Формула:
        T_effective = T_behavior * log10(1 + Stake / BaseStake)
    
    [ECONOMY] Параметры:
    - BASE_STAKE: Базовый порог стейкинга (100 ZEO)
    - DUST_LIMIT: Минимальный баланс для участия (10 ZEO)
    - DUST_MULTIPLIER: Множитель для пылевых балансов (0.1)
    - EMA_ALPHA: Коэффициент сглаживания EMA (0.1)
    - DECAY_RATE: Скорость затухания за неактивность (0.99/день)
    """
    
    # Экономические константы
    BASE_STAKE: float = 100.0        # Базовый порог в ZEO
    DUST_LIMIT: float = 10.0         # Минимальный порог в ZEO
    DUST_MULTIPLIER: float = 0.1     # Множитель для пылевых балансов
    
    # EMA параметры
    EMA_ALPHA: float = 0.1           # Коэффициент сглаживания
    INITIAL_SCORE: float = 0.5       # Начальный behavior score
    
    # Decay параметры
    DECAY_RATE: float = 0.99         # Множитель за день неактивности
    DECAY_INTERVAL: float = 86400.0  # 1 день в секундах
    
    def __init__(self, ledger: Optional[Any] = None):
        """
        Args:
            ledger: Экземпляр Ledger для получения балансов
        """
        self.ledger = ledger
        self._cache: Dict[str, PeerTrustState] = {}
        self._blacklist: Dict[str, str] = {}  # peer_id -> reason
    
    def calculate_effective_trust(
        self,
        behavior_score: float,
        stake: float,
    ) -> float:
        """
        Рассчитать эффективный Trust Score.
        
        Formula: T_effective = T_behavior * log10(1 + Stake / BaseStake)
        
        [SECURITY] Dust limit handling:
        - Если stake < DUST_LIMIT, применяется DUST_MULTIPLIER
        
        Args:
            behavior_score: Поведенческий скор [0.0 - 1.0]
            stake: Баланс пира в ZEO
        
        Returns:
            Эффективный Trust Score
        """
        # Clamp behavior score
        t_behavior = max(0.0, min(1.0, behavior_score))
        
        # Dust limit check
        if stake < self.DUST_LIMIT:
            stake_multiplier = self.DUST_MULTIPLIER
        else:
            # log10(1 + Stake / BaseStake)
            stake_multiplier = math.log10(1.0 + stake / self.BASE_STAKE)
        
        # Clamp multiplier to reasonable range [0.1, 3.0]
        stake_multiplier = max(0.1, min(3.0, stake_multiplier))
        
        return t_behavior * stake_multiplier
    
    def update_behavior_score(
        self,
        current_score: float,
        event: TrustEvent,
        magnitude: float = 1.0,
    ) -> float:
        """
        Обновить поведенческий скор используя EMA.
        
        Formula: T_new = T_old * (1 - alpha) + R * alpha
        
        Где R = текущий результат взаимодействия.
        
        [SLASHING] Если событие в SLASHING_EVENTS, score обнуляется.
        
        Args:
            current_score: Текущий behavior score
            event: Тип события
            magnitude: Множитель величины события
        
        Returns:
            Новый behavior score
        """
        # [CRITICAL] Проверка на slashing
        if event in SLASHING_EVENTS:
            return 0.0
        
        # Получаем вес события
        weight = EVENT_WEIGHTS.get(event, 0.0)
        adjustment = weight * magnitude
        
        # EMA update
        # R = 1.0 для позитивного события, 0.0 для негативного
        if adjustment > 0:
            result = 1.0
        elif adjustment < 0:
            result = 0.0
        else:
            return current_score  # Нет изменений
        
        # Масштабируем alpha по величине события
        effective_alpha = self.EMA_ALPHA * abs(adjustment) * 10
        effective_alpha = min(1.0, effective_alpha)
        
        new_score = current_score * (1 - effective_alpha) + result * effective_alpha
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, new_score))
    
    def apply_decay(
        self,
        score: float,
        last_interaction: float,
    ) -> float:
        """
        Применить decay за неактивность.
        
        Formula: score *= DECAY_RATE ^ days_inactive
        
        Args:
            score: Текущий скор
            last_interaction: Timestamp последнего взаимодействия
        
        Returns:
            Скор после decay
        """
        elapsed = time.time() - last_interaction
        days_inactive = elapsed / self.DECAY_INTERVAL
        
        if days_inactive < 0.1:  # Меньше 2.4 часов - без decay
            return score
        
        decay_factor = self.DECAY_RATE ** days_inactive
        return score * decay_factor
    
    async def get_peer_state(self, peer_id: str) -> PeerTrustState:
        """
        Получить состояние доверия пира.
        
        Загружает из кэша или создаёт новое состояние.
        """
        if peer_id in self._cache:
            return self._cache[peer_id]
        
        # Создаём новое состояние
        state = PeerTrustState(
            peer_id=peer_id,
            behavior_score=self.INITIAL_SCORE,
        )
        
        # Получаем баланс из ledger если доступен
        if self.ledger:
            state.stake_balance = await self._get_total_balance(peer_id)
        
        self._cache[peer_id] = state
        return state
    
    async def _get_total_balance(self, peer_id: str) -> float:
        """
        Получить общий баланс пира (IOU + on-chain).
        
        [ECONOMY] Сумма:
        - Локальные IOU (быстрые/бесплатные)
        - On-chain ZEO токены (медленные/надежные)
        """
        if not self.ledger:
            return 0.0
        
        try:
            # Получаем баланс из IOU
            balance_info = await self.ledger.get_balance_info(peer_id)
            iou_balance = max(0.0, balance_info.get("balance", 0.0))
            
            # TODO: Добавить on-chain баланс через ChainManager
            # on_chain_balance = await self.chain_manager.get_balance(peer_id)
            on_chain_balance = 0.0
            
            return iou_balance + on_chain_balance
        except Exception as e:
            logger.warning(f"[TRUST] Failed to get balance for {peer_id[:8]}...: {e}")
            return 0.0
    
    async def record_event(
        self,
        peer_id: str,
        event: TrustEvent,
        magnitude: float = 1.0,
    ) -> float:
        """
        Записать событие и обновить Trust Score.
        
        [SLASHING] При событиях из SLASHING_EVENTS:
        - Score обнуляется мгновенно
        - Peer добавляется в blacklist
        
        Args:
            peer_id: ID пира
            event: Тип события
            magnitude: Множитель величины
        
        Returns:
            Новый эффективный Trust Score
        """
        state = await self.get_peer_state(peer_id)
        
        # [CRITICAL] Slashing check
        if event in SLASHING_EVENTS:
            state.slashed = True
            state.slash_reason = event.name
            state.slash_timestamp = time.time()
            state.behavior_score = 0.0
            self._blacklist[peer_id] = event.name
            
            logger.warning(
                f"[TRUST] SLASHED peer {peer_id[:8]}... reason={event.name}"
            )
            
            # Обновляем в ledger если доступен
            if self.ledger:
                try:
                    await self.ledger.update_trust_score(peer_id, event.name, -1.0)
                except Exception as e:
                    logger.error(f"[TRUST] Failed to update ledger: {e}")
            
            return 0.0
        
        # Обновляем behavior score
        new_behavior = self.update_behavior_score(
            state.behavior_score,
            event,
            magnitude,
        )
        state.behavior_score = new_behavior
        state.interaction_count += 1
        state.last_interaction = time.time()
        
        # Обновляем stake balance
        if self.ledger:
            state.stake_balance = await self._get_total_balance(peer_id)
        
        # Рассчитываем эффективный скор
        effective = self.calculate_effective_trust(
            state.behavior_score,
            state.stake_balance,
        )
        
        # Обновляем в ledger
        if self.ledger:
            try:
                await self.ledger.update_trust_score(peer_id, event.name, magnitude)
            except Exception as e:
                logger.debug(f"[TRUST] Ledger update skipped: {e}")
        
        return effective
    
    async def get_effective_trust(self, peer_id: str) -> float:
        """
        Получить текущий эффективный Trust Score пира.
        
        Учитывает:
        - Behavior score
        - Stake balance
        - Decay за неактивность
        - Slashing status
        """
        # Проверка blacklist
        if peer_id in self._blacklist:
            return 0.0
        
        state = await self.get_peer_state(peer_id)
        
        # Проверка slashing
        if state.slashed:
            return 0.0
        
        # Применяем decay
        decayed_behavior = self.apply_decay(
            state.behavior_score,
            state.last_interaction,
        )
        
        # Рассчитываем эффективный скор
        return self.calculate_effective_trust(
            decayed_behavior,
            state.stake_balance,
        )
    
    def is_peer_trusted(self, peer_id: str, min_trust: float = 0.1) -> bool:
        """
        Проверить, является ли пир доверенным.
        
        [SECURITY] Быстрая синхронная проверка для hot path.
        """
        if peer_id in self._blacklist:
            return False
        
        if peer_id in self._cache:
            state = self._cache[peer_id]
            if state.slashed:
                return False
            
            # Грубая оценка без async
            effective = self.calculate_effective_trust(
                state.behavior_score,
                state.stake_balance,
            )
            return effective >= min_trust
        
        # Новый пир - даём шанс
        return True
    
    def is_blacklisted(self, peer_id: str) -> bool:
        """Проверить, в blacklist ли пир."""
        return peer_id in self._blacklist
    
    def get_blacklist(self) -> Dict[str, str]:
        """Получить список заблокированных пиров."""
        return dict(self._blacklist)
    
    async def slash_peer(self, peer_id: str, reason: str) -> None:
        """
        Явный слэшинг пира.
        
        [SECURITY] Используется при обнаружении:
        - INVALID_MERKLE_PROOF
        - DOUBLE_SPEND_ATTEMPT
        - SIGNATURE_FORGERY
        """
        state = await self.get_peer_state(peer_id)
        state.slashed = True
        state.slash_reason = reason
        state.slash_timestamp = time.time()
        state.behavior_score = 0.0
        self._blacklist[peer_id] = reason
        
        logger.warning(f"[TRUST] Manually slashed peer {peer_id[:8]}... reason={reason}")
    
    async def update_stake_balance(self, peer_id: str, balance: float) -> None:
        """Обновить баланс стейка пира."""
        state = await self.get_peer_state(peer_id)
        state.stake_balance = balance
    
    def clear_cache(self) -> None:
        """Очистить кэш состояний (не blacklist)."""
        self._cache.clear()


# Singleton instance for global access
_trust_system: Optional[WeightedTrustScore] = None


def get_trust_system(ledger: Optional[Any] = None) -> WeightedTrustScore:
    """
    Получить глобальный экземпляр системы доверия.
    
    [USAGE]
        trust = get_trust_system(ledger)
        score = await trust.get_effective_trust(peer_id)
    """
    global _trust_system
    if _trust_system is None:
        _trust_system = WeightedTrustScore(ledger)
    elif ledger is not None and _trust_system.ledger is None:
        _trust_system.ledger = ledger
    return _trust_system

