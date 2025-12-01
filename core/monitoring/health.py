"""
Health Checker - Проверка здоровья узла
======================================

[HEALTH CHECKS]
- Liveness: узел жив и отвечает
- Readiness: узел готов принимать трафик
- Component checks: состояние подсистем

[ENDPOINTS] Для Kubernetes/Docker:
- /health/live - liveness probe
- /health/ready - readiness probe
- /health/components - детали по компонентам
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Awaitable
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Статус здоровья."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Здоровье компонента."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: float = field(default_factory=time.time)
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check,
            "response_time_ms": self.response_time_ms,
            "details": self.details,
        }


class HealthChecker:
    """
    Проверка здоровья узла.
    
    [USAGE]
    ```python
    checker = HealthChecker()
    
    # Регистрация проверок
    checker.register("database", check_db)
    checker.register("network", check_network)
    
    # Запуск
    await checker.start()
    
    # Получить статус
    status = await checker.get_status()
    is_ready = await checker.is_ready()
    ```
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        timeout: float = 5.0,
        unhealthy_threshold: int = 3,
    ):
        """
        Args:
            check_interval: Интервал проверок (секунды)
            timeout: Таймаут на проверку
            unhealthy_threshold: Сколько failed checks до unhealthy
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self.unhealthy_threshold = unhealthy_threshold
        
        # Registered checks: name -> async checker function
        self._checks: Dict[str, Callable[[], Awaitable[ComponentHealth]]] = {}
        
        # Results
        self._results: Dict[str, ComponentHealth] = {}
        self._failure_counts: Dict[str, int] = {}
        
        # State
        self._started = False
        self._check_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._on_status_change: List[Callable[[str, HealthStatus, HealthStatus], None]] = []
    
    def register(
        self,
        name: str,
        check_func: Callable[[], Awaitable[ComponentHealth]],
    ) -> None:
        """
        Зарегистрировать проверку компонента.
        
        Args:
            name: Имя компонента
            check_func: Async функция проверки
        """
        self._checks[name] = check_func
        self._results[name] = ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            message="Not checked yet",
        )
        self._failure_counts[name] = 0
        logger.debug(f"[HEALTH] Registered check: {name}")
    
    def register_simple(
        self,
        name: str,
        check_func: Callable[[], bool],
    ) -> None:
        """
        Зарегистрировать простую проверку (sync, returns bool).
        
        Args:
            name: Имя компонента
            check_func: Функция проверки (True = healthy)
        """
        async def wrapper() -> ComponentHealth:
            try:
                result = check_func()
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="OK" if result else "Check failed",
                )
            except Exception as e:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )
        
        self.register(name, wrapper)
    
    async def start(self) -> None:
        """Запустить периодические проверки."""
        if self._started:
            return
        
        self._started = True
        
        # Initial check
        await self._run_all_checks()
        
        # Start background checks
        async def check_loop():
            while self._started:
                await asyncio.sleep(self.check_interval)
                await self._run_all_checks()
        
        self._check_task = asyncio.create_task(check_loop())
        logger.info(f"[HEALTH] Started (interval={self.check_interval}s)")
    
    async def stop(self) -> None:
        """Остановить проверки."""
        self._started = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
    
    async def check_now(self, name: Optional[str] = None) -> None:
        """Выполнить проверку сейчас."""
        if name:
            if name in self._checks:
                await self._run_check(name)
        else:
            await self._run_all_checks()
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Получить полный статус.
        
        Returns:
            {
                "status": "healthy|degraded|unhealthy",
                "timestamp": ...,
                "components": {...}
            }
        """
        overall = self._calculate_overall_status()
        
        return {
            "status": overall.value,
            "timestamp": time.time(),
            "uptime": time.time() - getattr(self, '_start_time', time.time()),
            "components": {
                name: result.to_dict()
                for name, result in self._results.items()
            },
        }
    
    async def is_live(self) -> bool:
        """Liveness check - узел жив."""
        return self._started
    
    async def is_ready(self) -> bool:
        """Readiness check - узел готов."""
        overall = self._calculate_overall_status()
        return overall in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    
    def get_component_status(self, name: str) -> Optional[ComponentHealth]:
        """Получить статус компонента."""
        return self._results.get(name)
    
    def on_status_change(
        self,
        callback: Callable[[str, HealthStatus, HealthStatus], None],
    ) -> None:
        """Callback при изменении статуса."""
        self._on_status_change.append(callback)
    
    async def _run_all_checks(self) -> None:
        """Запустить все проверки."""
        tasks = [
            self._run_check(name)
            for name in self._checks
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_check(self, name: str) -> None:
        """Запустить одну проверку."""
        check_func = self._checks.get(name)
        if not check_func:
            return
        
        start = time.time()
        old_status = self._results[name].status
        
        try:
            # Run with timeout
            result = await asyncio.wait_for(
                check_func(),
                timeout=self.timeout,
            )
            result.response_time_ms = (time.time() - start) * 1000
            result.last_check = time.time()
            
            # Update failure count
            if result.status == HealthStatus.HEALTHY:
                self._failure_counts[name] = 0
            else:
                self._failure_counts[name] += 1
            
            # Check threshold
            if self._failure_counts[name] >= self.unhealthy_threshold:
                result.status = HealthStatus.UNHEALTHY
                result.message = f"Failed {self._failure_counts[name]} times: {result.message}"
            
            self._results[name] = result
            
        except asyncio.TimeoutError:
            self._failure_counts[name] += 1
            self._results[name] = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Timeout after {self.timeout}s",
                response_time_ms=self.timeout * 1000,
            )
            
        except Exception as e:
            self._failure_counts[name] += 1
            self._results[name] = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {e}",
                response_time_ms=(time.time() - start) * 1000,
            )
        
        # Notify on change
        new_status = self._results[name].status
        if old_status != new_status:
            for callback in self._on_status_change:
                try:
                    callback(name, old_status, new_status)
                except Exception as e:
                    logger.error(f"[HEALTH] Callback error: {e}")
    
    def _calculate_overall_status(self) -> HealthStatus:
        """Вычислить общий статус."""
        if not self._results:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in self._results.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            # Check if critical components are down
            return HealthStatus.UNHEALTHY
        
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        
        return HealthStatus.UNKNOWN


# Standard health check implementations

async def check_memory(threshold_percent: float = 90.0) -> ComponentHealth:
    """Проверка памяти."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        status = HealthStatus.HEALTHY
        if memory.percent > threshold_percent:
            status = HealthStatus.UNHEALTHY
        elif memory.percent > threshold_percent * 0.8:
            status = HealthStatus.DEGRADED
        
        return ComponentHealth(
            name="memory",
            status=status,
            message=f"{memory.percent:.1f}% used",
            details={
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
            },
        )
    except ImportError:
        return ComponentHealth(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message="psutil not available",
        )


async def check_disk(path: str = "/", threshold_percent: float = 90.0) -> ComponentHealth:
    """Проверка диска."""
    try:
        import psutil
        disk = psutil.disk_usage(path)
        
        status = HealthStatus.HEALTHY
        if disk.percent > threshold_percent:
            status = HealthStatus.UNHEALTHY
        elif disk.percent > threshold_percent * 0.8:
            status = HealthStatus.DEGRADED
        
        return ComponentHealth(
            name="disk",
            status=status,
            message=f"{disk.percent:.1f}% used",
            details={
                "total": disk.total,
                "free": disk.free,
                "percent": disk.percent,
            },
        )
    except ImportError:
        return ComponentHealth(
            name="disk",
            status=HealthStatus.UNKNOWN,
            message="psutil not available",
        )


async def check_cpu(threshold_percent: float = 90.0) -> ComponentHealth:
    """Проверка CPU."""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        status = HealthStatus.HEALTHY
        if cpu_percent > threshold_percent:
            status = HealthStatus.UNHEALTHY
        elif cpu_percent > threshold_percent * 0.8:
            status = HealthStatus.DEGRADED
        
        return ComponentHealth(
            name="cpu",
            status=status,
            message=f"{cpu_percent:.1f}% used",
            details={
                "percent": cpu_percent,
                "cores": psutil.cpu_count(),
            },
        )
    except ImportError:
        return ComponentHealth(
            name="cpu",
            status=HealthStatus.UNKNOWN,
            message="psutil not available",
        )

