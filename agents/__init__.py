"""
Agents Module
=============
Система исполнения контрактов в изолированной среде:
- AgentManager: загрузка и выполнение скриптов
- Sandbox: RestrictedPython изоляция
"""

from .manager import AgentManager, ContractResult

__all__ = ["AgentManager", "ContractResult"]

