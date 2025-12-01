"""
Rate Limiter - Ограничение частоты запросов
==========================================

[ALGORITHM] Token Bucket с sliding window:
- Каждый peer имеет bucket с токенами
- Токены добавляются с фиксированной скоростью
- Операция потребляет токен
- Нет токенов = rate limited

[FEATURES]
- Per-peer rate limiting
- Per-action rate limiting
- Global rate limiting
- Burst allowance
- Automatic cleanup
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Правило rate limiting."""
    name: str
    tokens_per_second: float  # Скорость восполнения
    bucket_size: int          # Максимум токенов (burst)
    cost: int = 1             # Стоимость операции
    
    # Опционально: cooldown после исчерпания
    cooldown_seconds: float = 0.0


class RateLimitResult(Enum):
    """Результат проверки rate limit."""
    ALLOWED = "allowed"
    RATE_LIMITED = "rate_limited"
    COOLDOWN = "cooldown"
    BANNED = "banned"


@dataclass
class TokenBucket:
    """Token bucket для rate limiting."""
    tokens: float
    max_tokens: int
    last_update: float
    tokens_per_second: float
    
    # Cooldown state
    cooldown_until: float = 0.0
    
    # Stats
    total_allowed: int = 0
    total_denied: int = 0
    
    def refill(self) -> None:
        """Пополнить токены."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.max_tokens,
            self.tokens + elapsed * self.tokens_per_second
        )
        self.last_update = now
    
    def consume(self, cost: int = 1) -> bool:
        """Потребить токены. Возвращает True если успешно."""
        self.refill()
        
        if self.tokens >= cost:
            self.tokens -= cost
            self.total_allowed += 1
            return True
        
        self.total_denied += 1
        return False
    
    @property
    def is_in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until


class RateLimiter:
    """
    Rate limiter с поддержкой per-peer и per-action лимитов.
    
    [USAGE]
    ```python
    limiter = RateLimiter()
    
    # Добавить правило
    limiter.add_rule(RateLimitRule(
        name="messages",
        tokens_per_second=10,
        bucket_size=100,
    ))
    
    # Проверить
    result = limiter.check("peer123", "messages")
    if result == RateLimitResult.ALLOWED:
        # Обработать
        pass
    else:
        # Отклонить
        pass
    ```
    """
    
    # Стандартные правила
    DEFAULT_RULES = {
        "connect": RateLimitRule("connect", tokens_per_second=0.2, bucket_size=5, cooldown_seconds=60),
        "message": RateLimitRule("message", tokens_per_second=20, bucket_size=100),
        "ping": RateLimitRule("ping", tokens_per_second=1, bucket_size=10),
        "discovery": RateLimitRule("discovery", tokens_per_second=0.5, bucket_size=10),
        "dht_store": RateLimitRule("dht_store", tokens_per_second=2, bucket_size=20),
        "dht_find": RateLimitRule("dht_find", tokens_per_second=5, bucket_size=50),
        "service_request": RateLimitRule("service_request", tokens_per_second=2, bucket_size=10),
        "forward": RateLimitRule("forward", tokens_per_second=10, bucket_size=50),
    }
    
    def __init__(
        self,
        default_tokens_per_second: float = 10.0,
        default_bucket_size: int = 100,
        cleanup_interval: float = 300.0,
        max_peers: int = 10000,
    ):
        """
        Args:
            default_tokens_per_second: Дефолтная скорость восполнения
            default_bucket_size: Дефолтный размер bucket
            cleanup_interval: Интервал очистки неактивных
            max_peers: Максимум отслеживаемых пиров
        """
        self.default_tokens_per_second = default_tokens_per_second
        self.default_bucket_size = default_bucket_size
        self.cleanup_interval = cleanup_interval
        self.max_peers = max_peers
        
        # Rules
        self._rules: Dict[str, RateLimitRule] = dict(self.DEFAULT_RULES)
        
        # Buckets: peer_id -> action -> bucket
        self._buckets: Dict[str, Dict[str, TokenBucket]] = {}
        
        # Global buckets (per action)
        self._global_buckets: Dict[str, TokenBucket] = {}
        
        # Banned peers
        self._banned: Dict[str, float] = {}  # peer_id -> ban_until
        
        # Whitelist (не rate limit)
        self._whitelist: Set[str] = set()
        
        # Stats
        self._stats = {
            "total_checks": 0,
            "total_allowed": 0,
            "total_denied": 0,
            "total_banned": 0,
        }
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def add_rule(self, rule: RateLimitRule) -> None:
        """Добавить правило."""
        self._rules[rule.name] = rule
        logger.debug(f"[RATE] Added rule: {rule.name} ({rule.tokens_per_second}/s, burst={rule.bucket_size})")
    
    def remove_rule(self, name: str) -> None:
        """Удалить правило."""
        self._rules.pop(name, None)
    
    def whitelist_add(self, peer_id: str) -> None:
        """Добавить в whitelist."""
        self._whitelist.add(peer_id)
    
    def whitelist_remove(self, peer_id: str) -> None:
        """Удалить из whitelist."""
        self._whitelist.discard(peer_id)
    
    def check(
        self,
        peer_id: str,
        action: str = "message",
        cost: int = 1,
    ) -> RateLimitResult:
        """
        Проверить rate limit.
        
        Args:
            peer_id: ID пира
            action: Тип действия
            cost: Стоимость операции
        
        Returns:
            RateLimitResult
        """
        self._stats["total_checks"] += 1
        
        # Whitelist
        if peer_id in self._whitelist:
            self._stats["total_allowed"] += 1
            return RateLimitResult.ALLOWED
        
        # Banned
        if peer_id in self._banned:
            if time.time() < self._banned[peer_id]:
                return RateLimitResult.BANNED
            else:
                del self._banned[peer_id]
        
        # Get rule
        rule = self._rules.get(action)
        if not rule:
            rule = RateLimitRule(
                name=action,
                tokens_per_second=self.default_tokens_per_second,
                bucket_size=self.default_bucket_size,
            )
        
        # Get or create bucket
        bucket = self._get_bucket(peer_id, action, rule)
        
        # Check cooldown
        if bucket.is_in_cooldown:
            self._stats["total_denied"] += 1
            return RateLimitResult.COOLDOWN
        
        # Consume tokens
        if bucket.consume(cost):
            self._stats["total_allowed"] += 1
            return RateLimitResult.ALLOWED
        
        # Rate limited - apply cooldown if configured
        if rule.cooldown_seconds > 0:
            bucket.cooldown_until = time.time() + rule.cooldown_seconds
        
        self._stats["total_denied"] += 1
        return RateLimitResult.RATE_LIMITED
    
    def check_global(self, action: str, cost: int = 1) -> RateLimitResult:
        """Проверить глобальный rate limit (не per-peer)."""
        rule = self._rules.get(action)
        if not rule:
            return RateLimitResult.ALLOWED
        
        if action not in self._global_buckets:
            self._global_buckets[action] = TokenBucket(
                tokens=rule.bucket_size,
                max_tokens=rule.bucket_size,
                last_update=time.time(),
                tokens_per_second=rule.tokens_per_second * 10,  # Global = 10x
            )
        
        bucket = self._global_buckets[action]
        if bucket.consume(cost):
            return RateLimitResult.ALLOWED
        return RateLimitResult.RATE_LIMITED
    
    def ban(self, peer_id: str, duration: float = 3600.0) -> None:
        """Забанить пира."""
        self._banned[peer_id] = time.time() + duration
        self._stats["total_banned"] += 1
        logger.warning(f"[RATE] Banned peer {peer_id[:16]}... for {duration}s")
    
    def unban(self, peer_id: str) -> None:
        """Разбанить пира."""
        self._banned.pop(peer_id, None)
    
    def get_remaining(self, peer_id: str, action: str) -> Tuple[float, float]:
        """
        Получить оставшиеся токены и время до восполнения.
        
        Returns:
            (tokens, seconds_until_refill)
        """
        rule = self._rules.get(action)
        if not rule:
            return (float('inf'), 0.0)
        
        bucket = self._get_bucket(peer_id, action, rule)
        bucket.refill()
        
        if bucket.tokens >= 1:
            return (bucket.tokens, 0.0)
        
        # Time to get 1 token
        time_to_refill = (1 - bucket.tokens) / rule.tokens_per_second
        return (bucket.tokens, time_to_refill)
    
    def get_peer_stats(self, peer_id: str) -> Dict[str, Any]:
        """Статистика по пиру."""
        if peer_id not in self._buckets:
            return {}
        
        result = {}
        for action, bucket in self._buckets[peer_id].items():
            bucket.refill()
            result[action] = {
                "tokens": bucket.tokens,
                "max_tokens": bucket.max_tokens,
                "allowed": bucket.total_allowed,
                "denied": bucket.total_denied,
                "cooldown": bucket.is_in_cooldown,
            }
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Общая статистика."""
        return {
            **self._stats,
            "tracked_peers": len(self._buckets),
            "banned_peers": len(self._banned),
            "whitelisted": len(self._whitelist),
            "rules": list(self._rules.keys()),
        }
    
    async def start(self) -> None:
        """Запустить фоновую очистку."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(self.cleanup_interval)
                self._cleanup()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def stop(self) -> None:
        """Остановить."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    def _get_bucket(
        self,
        peer_id: str,
        action: str,
        rule: RateLimitRule,
    ) -> TokenBucket:
        """Получить или создать bucket."""
        if peer_id not in self._buckets:
            self._buckets[peer_id] = {}
        
        if action not in self._buckets[peer_id]:
            self._buckets[peer_id][action] = TokenBucket(
                tokens=rule.bucket_size,
                max_tokens=rule.bucket_size,
                last_update=time.time(),
                tokens_per_second=rule.tokens_per_second,
            )
        
        return self._buckets[peer_id][action]
    
    def _cleanup(self) -> None:
        """Очистить неактивные buckets."""
        now = time.time()
        stale_threshold = self.cleanup_interval * 2
        
        # Cleanup peer buckets
        stale_peers = []
        for peer_id, buckets in self._buckets.items():
            # Если все buckets полные и старые - удалить
            all_full_and_stale = all(
                b.tokens >= b.max_tokens and (now - b.last_update) > stale_threshold
                for b in buckets.values()
            )
            if all_full_and_stale:
                stale_peers.append(peer_id)
        
        for peer_id in stale_peers:
            del self._buckets[peer_id]
        
        # Cleanup expired bans
        expired_bans = [
            peer_id for peer_id, until in self._banned.items()
            if now > until
        ]
        for peer_id in expired_bans:
            del self._banned[peer_id]
        
        # Enforce max peers
        if len(self._buckets) > self.max_peers:
            # Remove oldest
            sorted_peers = sorted(
                self._buckets.items(),
                key=lambda x: max(b.last_update for b in x[1].values()),
            )
            to_remove = len(self._buckets) - self.max_peers
            for peer_id, _ in sorted_peers[:to_remove]:
                del self._buckets[peer_id]
        
        if stale_peers or expired_bans:
            logger.debug(f"[RATE] Cleanup: {len(stale_peers)} peers, {len(expired_bans)} bans")

