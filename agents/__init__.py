"""
Agents Module (Layer 3: Market)
===============================
Система услуг и рынок:
- BaseAgent: абстрактный класс для услуг
- AgentManager: реестр и обработка запросов
- EchoAgent: тестовый сервис для биллинга

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

__all__ = [
    "AgentManager",
    "BaseAgent",
    "EchoAgent",
    "StorageAgent",
    "ComputeAgent",
    "ServiceRequest",
    "ServiceResponse",
    "Contract",
    "ContractResult",
    "ContractExecutor",
]

