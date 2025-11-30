"""
Agents Module (Layer 3: Market)
===============================
Система услуг и рынок:
- BaseAgent: абстрактный класс для услуг
- AgentManager: реестр и обработка запросов
- EchoAgent: тестовый сервис для биллинга
- ReaderAgent: чтение веб-страниц

Sandbox:
- Contract: исполняемый код в песочнице
- ContractExecutor: RestrictedPython изоляция
"""

from .manager import (
    AgentManager,
    BaseAgent,
    EchoAgent,
    StorageAgent,
    ComputeAgent,
    ServiceRequest,
    ServiceResponse,
    Contract,
    ContractResult,
    ContractExecutor,
)

# ReaderAgent может быть недоступен если нет aiohttp/beautifulsoup4
try:
    from .web_reader import ReaderAgent
    _READER_AVAILABLE = True
except ImportError:
    ReaderAgent = None  # type: ignore
    _READER_AVAILABLE = False

__all__ = [
    "AgentManager",
    "BaseAgent",
    "EchoAgent",
    "StorageAgent",
    "ComputeAgent",
    "ReaderAgent",
    "ServiceRequest",
    "ServiceResponse",
    "Contract",
    "ContractResult",
    "ContractExecutor",
]

