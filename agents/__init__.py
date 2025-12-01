"""
Agents Module (Layer 3: Market)
===============================
Система услуг и рынок:
- BaseAgent: абстрактный класс для услуг
- AgentManager: реестр и обработка запросов
- EchoAgent: тестовый сервис для биллинга
- ReaderAgent: чтение веб-страниц
- LlmAgent: облачный LLM (OpenAI-compatible)
- OllamaAgent: локальный LLM (Ollama/GPU)
- DistributedLlmAgent: распределённый инференс через P2P

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

# LlmAgent (облачный) - требует aiohttp
try:
    from .ai_assistant import LlmAgent
    _LLM_AVAILABLE = True
except ImportError:
    LlmAgent = None  # type: ignore
    _LLM_AVAILABLE = False

# OllamaAgent (локальный GPU) - требует aiohttp
try:
    from .local_llm import OllamaAgent
    _OLLAMA_AVAILABLE = True
except ImportError:
    OllamaAgent = None  # type: ignore
    _OLLAMA_AVAILABLE = False

# Distributed LLM Agent - требует numpy
try:
    from .distributed_agent import DistributedLlmAgent, ShardProviderAgent
    _DISTRIBUTED_AVAILABLE = True
except ImportError:
    DistributedLlmAgent = None  # type: ignore
    ShardProviderAgent = None  # type: ignore
    _DISTRIBUTED_AVAILABLE = False

__all__ = [
    "AgentManager",
    "BaseAgent",
    "EchoAgent",
    "StorageAgent",
    "ComputeAgent",
    "ReaderAgent",
    "LlmAgent",
    "OllamaAgent",
    "DistributedLlmAgent",
    "ShardProviderAgent",
    "ServiceRequest",
    "ServiceResponse",
    "Contract",
    "ContractResult",
    "ContractExecutor",
]

