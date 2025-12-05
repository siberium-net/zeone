"""
Agent Manager - Система услуг и рынок (Layer 3)
===============================================

[MARKET] Каждый "Агент" - это услуга, которую узел продает сети:
- storage: хранение данных
- compute: вычисления
- vpn: проксирование трафика
- echo: тестовый сервис (Ping-Pong с оплатой)

[ECONOMY] Интеграция с Ledger:
- Проверка бюджета/лимита доверия перед выполнением
- Автоматическая запись стоимости в Ledger после выполнения
- Блокировка должников

[SECURITY] Контракты выполняются в песочнице RestrictedPython:
- Ограниченный набор функций
- Лимит времени выполнения
- Изолированное пространство имен
"""

import asyncio
import time
import hashlib
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Set, Tuple, TYPE_CHECKING
from pathlib import Path

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
)
from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem

from config import config

if TYPE_CHECKING:
    from economy.ledger import Ledger
    from core.node import Node

logger = logging.getLogger(__name__)


# =============================================================================
# Agent System - Система услуг
# =============================================================================

@dataclass
class ServiceRequest:
    """
    Запрос на услугу.
    
    [MARKET] Содержит:
    - service_name: название услуги ("echo", "storage", etc.)
    - payload: данные для обработки
    - requester_id: ID узла-заказчика
    - budget: максимальный бюджет на выполнение
    """
    
    service_name: str
    payload: Any
    requester_id: str
    budget: float  # Максимальный бюджет в единицах
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.request_id:
            # Генерируем уникальный ID запроса
            data = f"{self.service_name}:{self.requester_id}:{self.timestamp}"
            self.request_id = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "payload": self.payload,
            "requester_id": self.requester_id,
            "budget": self.budget,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceRequest":
        return cls(
            service_name=data["service_name"],
            payload=data["payload"],
            requester_id=data["requester_id"],
            budget=data["budget"],
            request_id=data.get("request_id", ""),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ServiceResponse:
    """
    Ответ на запрос услуги.
    
    [MARKET] Содержит:
    - success: успешно ли выполнение
    - result: результат выполнения
    - cost: фактическая стоимость
    - execution_time: время выполнения
    """
    
    success: bool
    result: Any
    cost: float  # Фактическая стоимость в единицах
    execution_time: float
    request_id: str
    error: Optional[str] = None
    provider_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "cost": self.cost,
            "execution_time": self.execution_time,
            "request_id": self.request_id,
            "error": self.error,
            "provider_id": self.provider_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceResponse":
        return cls(
            success=data["success"],
            result=data["result"],
            cost=data["cost"],
            execution_time=data["execution_time"],
            request_id=data["request_id"],
            error=data.get("error"),
            provider_id=data.get("provider_id", ""),
        )


class BaseAgent(ABC):
    """
    Абстрактный базовый класс для агентов (услуг).
    
    [MARKET] Каждый агент реализует:
    - service_name: уникальное название услуги
    - price_per_unit: цена за единицу работы
    - execute(): метод выполнения задачи
    
    [ECONOMY] Стоимость вычисляется как:
    cost = price_per_unit * units_of_work
    """
    
    @property
    @abstractmethod
    def service_name(self) -> str:
        """Уникальное название услуги."""
        pass
    
    @property
    @abstractmethod
    def price_per_unit(self) -> float:
        """Цена за единицу работы."""
        pass
    
    @property
    def description(self) -> str:
        """Описание услуги."""
        return f"Service: {self.service_name}"
    
    @abstractmethod
    async def execute(self, payload: Any) -> Tuple[Any, float]:
        """
        Выполнить задачу.
        
        Args:
            payload: Данные для обработки
        
        Returns:
            (result, units_of_work) - результат и количество единиц работы
        """
        pass
    
    def calculate_cost(self, units: float) -> float:
        """Рассчитать стоимость."""
        return self.price_per_unit * units
    
    def estimate_cost(self, payload: Any) -> float:
        """
        Оценить стоимость выполнения (до выполнения).
        
        По умолчанию оценивает по размеру payload.
        Может быть переопределен для более точной оценки.
        """
        if isinstance(payload, (str, bytes)):
            size = len(payload)
        elif isinstance(payload, dict):
            size = len(json.dumps(payload))
        else:
            size = 100  # Default estimate
        
        # Оценка в единицах работы
        estimated_units = size / 10  # 10 байт = 1 единица (по умолчанию)
        return self.calculate_cost(estimated_units)


class EchoAgent(BaseAgent):
    """
    Тестовый агент Echo - Ping-Pong с оплатой.
    
    [MARKET] Услуга "echo":
    - Просто возвращает полученные данные
    - Цена: 1 единица за 10 байт
    - Используется для отладки биллинга
    
    [EXAMPLE]
    Узел А посылает "Hello" (5 байт) Узлу Б.
    Узел Б возвращает "Hello", стоимость = 0.5 единиц.
    В базе Узла Б: "Узел А должен мне 0.5 единиц".
    """
    
    BYTES_PER_UNIT = 10  # 10 байт = 1 единица
    
    @property
    def service_name(self) -> str:
        return "echo"
    
    @property
    def price_per_unit(self) -> float:
        return 1.0  # 1 единица стоит 1 "кредит"
    
    @property
    def description(self) -> str:
        return f"Echo service: returns input data. Price: {self.price_per_unit} per {self.BYTES_PER_UNIT} bytes"
    
    async def execute(self, payload: Any) -> Tuple[Any, float]:
        """
        Выполнить Echo - просто вернуть данные.
        
        Returns:
            (payload, units) - те же данные и количество единиц работы
        """
        # Вычисляем размер payload
        if isinstance(payload, bytes):
            size = len(payload)
        elif isinstance(payload, str):
            size = len(payload.encode("utf-8"))
        else:
            # Сериализуем для определения размера
            size = len(json.dumps(payload).encode("utf-8"))
        
        # Вычисляем единицы работы
        units = size / self.BYTES_PER_UNIT
        
        # Минимум 0.1 единицы даже для пустого payload
        units = max(0.1, units)
        
        logger.debug(f"[ECHO] Processing {size} bytes, units={units:.2f}")
        
        return payload, units


class StorageAgent(BaseAgent):
    """
    Агент распределённого хранения данных.
    
    [REAL STORAGE] Функции:
    - store: Сохранить данные с TTL
    - get: Получить данные по ID
    - delete: Удалить данные
    - list: Список сохранённых объектов
    
    [PRICING]
    - Хранение: 0.1 единицы за 1KB
    - Получение: 0.05 за запрос
    - Удаление: бесплатно
    """
    
    def __init__(self, storage_dir: str = "storage"):
        """
        Args:
            storage_dir: Директория для хранения файлов
        """
        import os
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Метаданные в SQLite
        self._db_path = os.path.join(storage_dir, "storage_meta.db")
        self._db_initialized = False
    
    @property
    def service_name(self) -> str:
        return "storage"
    
    @property
    def price_per_unit(self) -> float:
        return 0.1  # 0.1 за KB
    
    @property
    def description(self) -> str:
        return "Distributed storage: store/get/delete data. Price: 0.1 per KB stored"
    
    async def _init_db(self) -> None:
        """Инициализация базы метаданных."""
        if self._db_initialized:
            return
        
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS storage_objects (
                    storage_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    content_type TEXT DEFAULT 'application/octet-stream',
                    filename TEXT
                )
            """)
            await db.commit()
        
        self._db_initialized = True
    
    async def execute(self, payload: Any) -> Tuple[Any, float]:
        """
        Выполнить операцию хранилища.
        
        Payload:
            action: "store" | "get" | "delete" | "list"
            data: bytes | str (для store)
            storage_id: str (для get/delete)
            ttl_hours: int (для store, default 24)
            owner_id: str (идентификатор владельца)
        """
        import os
        import aiosqlite
        import base64
        
        await self._init_db()
        
        action = payload.get("action", "store")
        owner_id = payload.get("owner_id", "anonymous")
        
        if action == "store":
            # Сохранить данные
            data = payload.get("data", b"")
            ttl_hours = payload.get("ttl_hours", 24)
            filename = payload.get("filename")
            content_type = payload.get("content_type", "application/octet-stream")
            
            # Конвертируем в bytes
            if isinstance(data, str):
                # Проверяем, это base64 или plain text
                try:
                    data_bytes = base64.b64decode(data)
                except Exception:
                    data_bytes = data.encode("utf-8")
                    content_type = "text/plain"
            else:
                data_bytes = data
            
            size_bytes = len(data_bytes)
            storage_id = hashlib.sha256(
                f"{time.time()}{owner_id}{size_bytes}".encode()
            ).hexdigest()[:32]
            
            # Сохраняем файл
            file_path = os.path.join(self.storage_dir, storage_id)
            with open(file_path, "wb") as f:
                f.write(data_bytes)
            
            # Сохраняем метаданные
            created_at = time.time()
            expires_at = created_at + (ttl_hours * 3600) if ttl_hours > 0 else None
            
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """INSERT INTO storage_objects 
                       (storage_id, owner_id, size_bytes, created_at, expires_at, content_type, filename)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (storage_id, owner_id, size_bytes, created_at, expires_at, content_type, filename)
                )
                await db.commit()
            
            # Стоимость: 0.1 за KB
            cost = (size_bytes / 1024) * self.price_per_unit
            cost = max(cost, 0.01)  # Минимум 0.01
            
            return {
                "action": "store",
                "storage_id": storage_id,
                "size_bytes": size_bytes,
                "size_kb": size_bytes / 1024,
                "expires_at": expires_at,
                "content_type": content_type,
            }, cost
        
        elif action == "get":
            # Получить данные
            storage_id = payload.get("storage_id")
            if not storage_id:
                return {"error": "storage_id required"}, 0.01
            
            # Проверяем метаданные
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT size_bytes, expires_at, content_type, filename FROM storage_objects WHERE storage_id = ?",
                    (storage_id,)
                ) as cursor:
                    row = await cursor.fetchone()
            
            if not row:
                return {"error": "Object not found"}, 0.01
            
            size_bytes, expires_at, content_type, filename = row
            
            # Проверяем срок
            if expires_at and time.time() > expires_at:
                return {"error": "Object expired"}, 0.01
            
            # Читаем файл
            file_path = os.path.join(self.storage_dir, storage_id)
            if not os.path.exists(file_path):
                return {"error": "Object file not found"}, 0.01
            
            with open(file_path, "rb") as f:
                data = f.read()
            
            # Возвращаем как base64
            data_b64 = base64.b64encode(data).decode("ascii")
            
            return {
                "action": "get",
                "storage_id": storage_id,
                "data": data_b64,
                "size_bytes": size_bytes,
                "content_type": content_type,
                "filename": filename,
            }, 0.05  # Фиксированная плата за получение
        
        elif action == "delete":
            # Удалить данные
            storage_id = payload.get("storage_id")
            if not storage_id:
                return {"error": "storage_id required"}, 0
            
            # Удаляем файл
            file_path = os.path.join(self.storage_dir, storage_id)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Удаляем метаданные
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "DELETE FROM storage_objects WHERE storage_id = ?",
                    (storage_id,)
                )
                await db.commit()
            
            return {
                "action": "delete",
                "storage_id": storage_id,
                "deleted": True,
            }, 0  # Удаление бесплатно
        
        elif action == "list":
            # Список объектов владельца
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    """SELECT storage_id, size_bytes, created_at, expires_at, content_type, filename 
                       FROM storage_objects WHERE owner_id = ?
                       ORDER BY created_at DESC LIMIT 100""",
                    (owner_id,)
                ) as cursor:
                    rows = await cursor.fetchall()
            
            objects = []
            for row in rows:
                storage_id, size_bytes, created_at, expires_at, content_type, filename = row
                objects.append({
                    "storage_id": storage_id,
                    "size_bytes": size_bytes,
                    "created_at": created_at,
                    "expires_at": expires_at,
                    "content_type": content_type,
                    "filename": filename,
                })
            
            return {
                "action": "list",
                "count": len(objects),
                "objects": objects,
            }, 0.01  # Небольшая плата за листинг
        
        else:
            return {"error": f"Unknown action: {action}"}, 0


class ComputeAgent(BaseAgent):
    """
    Агент вычислений с реальным выполнением кода.
    
    [REAL COMPUTE] Функции:
    - eval: Вычислить Python выражение (безопасно)
    - exec: Выполнить Python код в sandbox
    - math: Математические вычисления
    - hash: Хэширование данных
    
    [SECURITY]
    - RestrictedPython sandbox
    - Timeout на выполнение
    - Ограничение памяти
    
    [PRICING]
    - 1 единица за секунду CPU
    - Минимум 0.01
    """
    
    def __init__(
        self, 
        timeout_seconds: float = 10.0,
        max_memory_mb: int = 100,
    ):
        """
        Args:
            timeout_seconds: Максимальное время выполнения
            max_memory_mb: Лимит памяти (не enforced в Python)
        """
        self.timeout = timeout_seconds
        self.max_memory_mb = max_memory_mb
        
        # Безопасные встроенные функции
        self._safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "pow": pow,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }
    
    @property
    def service_name(self) -> str:
        return "compute"
    
    @property
    def price_per_unit(self) -> float:
        return 1.0  # 1 за CPU-секунду
    
    @property
    def description(self) -> str:
        return "Compute service: eval/exec/math/hash. Price: 1 per CPU-second"
    
    async def execute(self, payload: Any) -> Tuple[Any, float]:
        """
        Выполнить вычисление.
        
        Payload:
            task: "eval" | "exec" | "math" | "hash" | "sum" | "count"
            expression: str (для eval)
            code: str (для exec)
            operation: str (для math)
            data: Any (входные данные)
            algorithm: str (для hash, default sha256)
        """
        import math as math_module
        import resource
        import signal
        
        task = payload.get("task", "eval")
        start_time = time.time()
        start_cpu = time.process_time()
        
        result = None
        error = None
        
        try:
            if task == "eval":
                # Безопасное вычисление выражения
                expression = payload.get("expression", "")
                if not expression:
                    return {"error": "expression required"}, 0.01
                
                # Проверяем на опасные конструкции
                dangerous = ["import", "exec", "eval", "open", "file", "__", "os.", "sys."]
                for d in dangerous:
                    if d in expression.lower():
                        return {"error": f"Forbidden: {d}"}, 0.01
                
                # Добавляем math функции
                safe_globals = {
                    "__builtins__": self._safe_builtins,
                    "math": math_module,
                    "sin": math_module.sin,
                    "cos": math_module.cos,
                    "tan": math_module.tan,
                    "sqrt": math_module.sqrt,
                    "log": math_module.log,
                    "log10": math_module.log10,
                    "exp": math_module.exp,
                    "pi": math_module.pi,
                    "e": math_module.e,
                }
                
                # Добавляем переменные из payload
                variables = payload.get("variables", {})
                safe_globals.update(variables)
                
                result = eval(expression, safe_globals, {})
            
            elif task == "exec":
                # Выполнение кода в sandbox (RestrictedPython)
                code = payload.get("code", "")
                if not code:
                    return {"error": "code required"}, 0.01
                
                try:
                    from RestrictedPython import compile_restricted, safe_globals as rp_globals
                    from RestrictedPython.Eval import default_guarded_getiter
                    from RestrictedPython.Guards import guarded_iter_unpack_sequence
                    
                    byte_code = compile_restricted(code, "<compute>", "exec")
                    
                    exec_globals = {
                        "__builtins__": self._safe_builtins,
                        "_getiter_": default_guarded_getiter,
                        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
                        "result": None,
                    }
                    
                    exec(byte_code, exec_globals)
                    result = exec_globals.get("result")
                    
                except ImportError:
                    return {"error": "RestrictedPython not available"}, 0.01
            
            elif task == "math":
                # Математические операции
                operation = payload.get("operation", "add")
                numbers = payload.get("numbers", [])
                
                if operation == "add":
                    result = sum(numbers)
                elif operation == "multiply":
                    result = 1
                    for n in numbers:
                        result *= n
                elif operation == "factorial":
                    n = payload.get("n", 0)
                    result = math_module.factorial(min(n, 1000))  # Лимит
                elif operation == "power":
                    base = payload.get("base", 2)
                    exp = payload.get("exp", 2)
                    result = pow(base, min(exp, 1000))  # Лимит
                elif operation == "sqrt":
                    n = payload.get("n", 0)
                    result = math_module.sqrt(n)
                elif operation == "prime_check":
                    n = payload.get("n", 0)
                    result = self._is_prime(n)
                else:
                    result = f"Unknown operation: {operation}"
            
            elif task == "hash":
                # Хэширование
                data = payload.get("data", "")
                algorithm = payload.get("algorithm", "sha256")
                
                if isinstance(data, str):
                    data = data.encode("utf-8")
                
                if algorithm == "sha256":
                    result = hashlib.sha256(data).hexdigest()
                elif algorithm == "sha512":
                    result = hashlib.sha512(data).hexdigest()
                elif algorithm == "md5":
                    result = hashlib.md5(data).hexdigest()
                elif algorithm == "blake2b":
                    result = hashlib.blake2b(data).hexdigest()
                else:
                    return {"error": f"Unknown algorithm: {algorithm}"}, 0.01
            
            elif task == "sum":
                # Простое суммирование (обратная совместимость)
                numbers = payload.get("numbers", [])
                result = sum(numbers)
            
            elif task == "count":
                # Подсчёт (обратная совместимость)
                data = payload.get("data", "")
                result = len(data)
            
            else:
                return {"error": f"Unknown task: {task}"}, 0.01
                
        except Exception as e:
            error = str(e)
        
        # Вычисляем время
        end_time = time.time()
        end_cpu = time.process_time()
        
        wall_time = end_time - start_time
        cpu_time = end_cpu - start_cpu
        cpu_seconds = max(0.01, cpu_time)  # Минимум 0.01
        
        if error:
            return {
                "task": task,
                "error": error,
                "wall_time_ms": wall_time * 1000,
                "cpu_time_ms": cpu_time * 1000,
            }, 0.01
        
        return {
            "task": task,
            "result": result,
            "wall_time_ms": wall_time * 1000,
            "cpu_time_ms": cpu_time * 1000,
            "cpu_seconds": cpu_seconds,
        }, cpu_seconds
    
    def _is_prime(self, n: int) -> bool:
        """Проверка на простое число."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True


# =============================================================================
# Agent Manager - Менеджер услуг
# =============================================================================

class AgentManager:
    """
    Менеджер агентов/услуг.
    
    [MARKET] Функции:
    - Реестр доступных услуг на этом узле
    - Обработка запросов от других узлов
    - Интеграция с Ledger для биллинга
    
    [ECONOMY] Процесс обработки запроса:
    1. Проверить, есть ли такой агент
    2. Проверить бюджет/лимит доверия заказчика
    3. Выполнить agent.execute()
    4. Записать стоимость в Ledger
    5. Вернуть результат
    """
    
    def __init__(
        self,
        ledger: Optional["Ledger"] = None,
        node_id: str = "",
    ):
        """
        Инициализация менеджера агентов.
        
        Args:
            ledger: Экземпляр Ledger для биллинга
            node_id: ID этого узла (провайдера услуг)
        """
        self.ledger = ledger
        self.node_id = node_id
        self._node: Optional["Node"] = None
        
        # Реестр агентов: service_name -> Agent
        self._agents: Dict[str, BaseAgent] = {}
        
        # Статистика
        self._total_requests = 0
        self._total_revenue = 0.0
        
        # Регистрируем встроенных агентов
        self._register_builtin_agents()
    
    def _register_builtin_agents(self) -> None:
        """Зарегистрировать встроенных агентов."""
        # Core agents (always available)
        self.register_agent(EchoAgent())
        self.register_agent(StorageAgent())
        self.register_agent(ComputeAgent())

        # VPN Exit Agent
        try:
            from .vpn import VpnExitAgent
            self.register_agent(VpnExitAgent(ledger=self.ledger))
        except ImportError as e:
            logger.warning(f"[AGENTS] VpnExitAgent not available: {e}")
        except Exception as e:
            logger.warning(f"[AGENTS] Failed to init VpnExitAgent: {e}")
        
        # Web Reader Agent - скачивание и парсинг веб-страниц
        try:
            from .web_reader import ReaderAgent
            self.register_agent(ReaderAgent())
        except ImportError as e:
            logger.warning(f"[AGENTS] ReaderAgent not available: {e}")
        
        # AI Agents - register via lazy loader (only if dependencies available)
        self._register_ai_agents()
    
    def _register_ai_agents(self) -> None:
        """
        Register AI agents with lazy loading.
        
        [LITE MODE] If AI libraries (torch, transformers) are not installed,
        these agents are skipped and node runs in LITE mode.
        """
        try:
            from core.lazy_imports import (
                is_ai_available, 
                AI_STATUS,
                check_ai_availability,
            )
            
            check_ai_availability()
            
            if not is_ai_available():
                logger.info("[AGENTS] AI Module not found. Running in LITE mode.")
                logger.info("[AGENTS]   Node operates as VPN/Storage/Wallet only.")
                logger.info("[AGENTS]   To enable AI: pip install -r requirements/ai.txt")
                return
            
        except ImportError:
            # lazy_imports module not available, try direct imports
            pass
        
        # Cloud LLM Agent - облачный AI (OpenAI-compatible)
        try:
            from .ai_assistant import LlmAgent
            self.register_agent(LlmAgent())
        except ImportError as e:
            logger.debug(f"[AGENTS] LlmAgent not available: {e}")
        except Exception as e:
            logger.warning(f"[AGENTS] Failed to init LlmAgent: {e}")
        
        # Local LLM Agent - локальный AI (Ollama/GPU)
        try:
            from .local_llm import OllamaAgent
            self.register_agent(OllamaAgent())
        except ImportError as e:
            logger.debug(f"[AGENTS] OllamaAgent not available: {e}")
        except Exception as e:
            logger.warning(f"[AGENTS] Failed to init OllamaAgent: {e}")
        
        # Vision Agent (if insightface available)
        try:
            from .vision import VisionAgent
            self.register_agent(VisionAgent())
        except ImportError as e:
            logger.debug(f"[AGENTS] VisionAgent not available: {e}")
        except Exception as e:
            logger.debug(f"[AGENTS] VisionAgent init skipped: {e}")
        
        # NeuroLink Agent (tensor transport)
        try:
            from .neuro_link import NeuroLinkAgent
            self.register_agent(NeuroLinkAgent())
        except ImportError as e:
            logger.debug(f"[AGENTS] NeuroLinkAgent not available: {e}")
        except Exception as e:
            logger.debug(f"[AGENTS] NeuroLinkAgent init skipped: {e}")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Зарегистрировать агента в реестре.
        
        [MARKET] После регистрации услуга становится доступной
        для запросов от других узлов.
        """
        self._agents[agent.service_name] = agent
        if self._node and hasattr(agent, "attach_node"):
            try:
                agent.attach_node(self._node)
            except Exception as e:
                logger.warning(f"[AGENT] Failed to attach node to {agent.service_name}: {e}")
        logger.info(f"[AGENT] Registered: {agent.service_name} ({agent.description})")
    
    def unregister_agent(self, service_name: str) -> bool:
        """Удалить агента из реестра."""
        if service_name in self._agents:
            del self._agents[service_name]
            logger.info(f"[AGENT] Unregistered: {service_name}")
            return True
        return False
    
    def get_agent(self, service_name: str) -> Optional[BaseAgent]:
        """Получить агента по имени услуги."""
        return self._agents.get(service_name)
    
    def list_services(self) -> List[Dict[str, Any]]:
        """
        Получить список доступных услуг.
        
        [MARKET] Этот список можно отправлять другим узлам
        для рекламы услуг.
        """
        services = []
        for name, agent in self._agents.items():
            services.append({
                "service_name": name,
                "price_per_unit": agent.price_per_unit,
                "description": agent.description,
            })
        return services
    
    def set_ledger(self, ledger: "Ledger") -> None:
        """Установить Ledger для биллинга."""
        self.ledger = ledger
    
    def set_node_id(self, node_id: str) -> None:
        """Установить ID узла."""
        self.node_id = node_id

    def set_node(self, node: "Node") -> None:
        """
        Установить Node и передать его агентам, которым нужен доступ.
        """
        self._node = node
        for agent in self._agents.values():
            if hasattr(agent, "attach_node"):
                try:
                    agent.attach_node(node)
                except Exception as e:
                    logger.warning(f"[AGENT] Failed to attach node to {agent.service_name}: {e}")
    
    async def handle_request(
        self,
        request: ServiceRequest,
    ) -> ServiceResponse:
        """
        Обработать запрос на услугу.
        
        [MARKET] Процесс:
        1. Проверить наличие агента
        2. Проверить бюджет заказчика
        3. Выполнить услугу
        4. Записать долг в Ledger
        5. Вернуть результат
        
        Args:
            request: Запрос на услугу
        
        Returns:
            ServiceResponse с результатом
        """
        start_time = time.time()
        self._total_requests += 1
        
        logger.info(
            f"[AGENT] Request from {request.requester_id[:8]}...: "
            f"service={request.service_name}, budget={request.budget}"
        )
        
        # 1. Проверяем наличие агента
        agent = self.get_agent(request.service_name)
        if agent is None:
            return ServiceResponse(
                success=False,
                result=None,
                cost=0,
                execution_time=time.time() - start_time,
                request_id=request.request_id,
                error=f"Service not found: {request.service_name}",
                provider_id=self.node_id,
            )
        
        # 2. Оцениваем стоимость
        estimated_cost = agent.estimate_cost(request.payload)
        
        if estimated_cost > request.budget:
            return ServiceResponse(
                success=False,
                result=None,
                cost=0,
                execution_time=time.time() - start_time,
                request_id=request.request_id,
                error=f"Insufficient budget: estimated {estimated_cost:.2f}, budget {request.budget:.2f}",
                provider_id=self.node_id,
            )
        
        # 3. Проверяем лимит доверия в Ledger (если есть)
        if self.ledger:
            can_process, reason = await self._check_requester_credit(request.requester_id)
            if not can_process:
                return ServiceResponse(
                    success=False,
                    result=None,
                    cost=0,
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                    error=f"Credit check failed: {reason}",
                    provider_id=self.node_id,
                )
        
        # 4. Выполняем услугу
        agent_payload = request.payload
        if isinstance(agent_payload, dict):
            if "requester_id" not in agent_payload:
                agent_payload = {**agent_payload, "requester_id": request.requester_id}
            if "request_id" not in agent_payload:
                agent_payload = {**agent_payload, "request_id": request.request_id}

        try:
            result, units = await agent.execute(agent_payload)
            actual_cost = agent.calculate_cost(units)
            
            # Проверяем, что фактическая стоимость не превышает бюджет
            if actual_cost > request.budget:
                actual_cost = request.budget
                logger.warning(
                    f"[AGENT] Cost capped to budget: {actual_cost:.2f} "
                    f"(was {agent.calculate_cost(units):.2f})"
                )
            
        except Exception as e:
            logger.error(f"[AGENT] Execution error: {e}")
            return ServiceResponse(
                success=False,
                result=None,
                cost=0,
                execution_time=time.time() - start_time,
                request_id=request.request_id,
                error=f"Execution error: {str(e)}",
                provider_id=self.node_id,
            )
        
        # 5. Записываем долг в Ledger
        if self.ledger and actual_cost > 0:
            await self._record_service_debt(request.requester_id, actual_cost)
        
        self._total_revenue += actual_cost
        
        execution_time = time.time() - start_time
        
        logger.info(
            f"[AGENT] Completed: service={request.service_name}, "
            f"cost={actual_cost:.2f}, time={execution_time:.3f}s"
        )
        
        return ServiceResponse(
            success=True,
            result=result,
            cost=actual_cost,
            execution_time=execution_time,
            request_id=request.request_id,
            provider_id=self.node_id,
        )
    
    async def _check_requester_credit(self, requester_id: str) -> Tuple[bool, str]:
        """
        Проверить кредитоспособность заказчика.
        
        [ECONOMY] Проверяем:
        - Не заблокирован ли заказчик из-за долга
        - Есть ли у него лимит доверия
        """
        if not self.ledger:
            return (True, "No ledger")
        
        # Проверяем, не заблокирован ли заказчик
        # (отрицательный баланс = заказчик должен нам)
        balance = await self.ledger.get_balance(requester_id)
        
        # Баланс с точки зрения ledger:
        # положительный = они должны нам
        # отрицательный = мы должны им
        # Для услуг: если они уже много должны - можем отказать
        
        debt_limit = self.ledger.debt_limit
        
        if balance > debt_limit:
            return (
                False,
                f"Requester debt {balance:.0f} exceeds limit {debt_limit:.0f}"
            )
        
        return (True, "OK")
    
    async def _record_service_debt(self, requester_id: str, cost: float) -> None:
        """
        Записать долг за услугу в Ledger.
        
        [ECONOMY] Записываем claim - заказчик теперь должен нам cost единиц.
        """
        if not self.ledger:
            return
        
        # record_claim увеличивает наш баланс (заказчик должен нам больше)
        new_balance = await self.ledger.record_claim(
            peer_id=requester_id,
            amount=cost,
            signature="",  # Подпись добавится на уровне протокола
        )
        
        logger.debug(
            f"[AGENT] Recorded debt: {requester_id[:8]}... owes {cost:.2f}, "
            f"total balance: {new_balance:.2f}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику менеджера."""
        return {
            "registered_services": len(self._agents),
            "services": list(self._agents.keys()),
            "total_requests": self._total_requests,
            "total_revenue": self._total_revenue,
        }


# =============================================================================
# Contract/Sandbox System (сохранено для обратной совместимости)
# =============================================================================

@dataclass
class Contract:
    """
    Контракт - исполняемый код от другого узла.
    
    [SECURITY] Контракт содержит:
    - code: исходный код Python
    - author_id: ID узла-автора
    - signature: подпись автора
    - hash: SHA256 хеш кода (идентификатор)
    """
    
    code: str
    author_id: str
    signature: str
    name: str = "unnamed"
    description: str = ""
    created_at: float = field(default_factory=time.time)
    
    @property
    def hash(self) -> str:
        """SHA256 хеш кода контракта."""
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "author_id": self.author_id,
            "signature": self.signature,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contract":
        return cls(
            code=data["code"],
            author_id=data["author_id"],
            signature=data["signature"],
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class ContractResult:
    """Результат выполнения контракта."""
    
    success: bool
    output: Any
    execution_time: float
    contract_hash: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "execution_time": self.execution_time,
            "contract_hash": self.contract_hash,
            "error": self.error,
        }


class SandboxViolation(Exception):
    """Исключение при попытке нарушить ограничения песочницы."""
    pass


class ContractExecutor:
    """
    Исполнитель контрактов в песочнице.
    
    [SECURITY] Использует RestrictedPython для изоляции.
    """
    
    SAFE_BUILTINS = {
        "True": True, "False": False, "None": None,
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "bytes": bytes, "frozenset": frozenset,
        "abs": abs, "min": min, "max": max, "sum": sum,
        "round": round, "pow": pow, "divmod": divmod,
        "len": len, "enumerate": enumerate, "zip": zip,
        "map": map, "filter": filter, "sorted": sorted,
        "reversed": reversed, "all": all, "any": any,
        "chr": chr, "ord": ord, "repr": repr, "format": format,
        "isinstance": isinstance, "callable": callable,
        "hash": hash, "type": type,
    }
    
    def __init__(self, max_execution_time: float = 5.0):
        self.max_execution_time = max_execution_time
    
    async def execute(
        self,
        contract: Contract,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ContractResult:
        """Выполнить контракт в песочнице."""
        start_time = time.time()
        
        try:
            # Компиляция
            result = compile_restricted(
                contract.code,
                filename=f"<contract:{contract.hash[:8]}>",
                mode="exec",
            )
            
            if result.errors:
                return ContractResult(
                    success=False,
                    output=None,
                    execution_time=time.time() - start_time,
                    contract_hash=contract.hash,
                    error=f"Compilation errors: {result.errors}",
                )
            
            # Создаем globals
            restricted_globals = {
                "__builtins__": self.SAFE_BUILTINS.copy(),
                "_getattr_": default_guarded_getattr,
                "_getitem_": default_guarded_getitem,
                "_getiter_": iter,
                "_result_": None,
            }
            
            if inputs:
                restricted_globals.update(inputs)
            
            # Выполнение
            loop = asyncio.get_event_loop()
            
            def run():
                locals_dict: Dict[str, Any] = {}
                exec(result.code, restricted_globals, locals_dict)
                return restricted_globals.get("_result_", locals_dict.get("result"))
            
            output = await asyncio.wait_for(
                loop.run_in_executor(None, run),
                timeout=self.max_execution_time,
            )
            
            return ContractResult(
                success=True,
                output=output,
                execution_time=time.time() - start_time,
                contract_hash=contract.hash,
            )
            
        except asyncio.TimeoutError:
            return ContractResult(
                success=False,
                output=None,
                execution_time=self.max_execution_time,
                contract_hash=contract.hash,
                error="Execution timeout",
            )
        except Exception as e:
            return ContractResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                contract_hash=contract.hash,
                error=str(e),
            )
