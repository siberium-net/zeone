"""
Cortex - Автономная Система Знаний
==================================

Cortex - это децентрализованная система автоматического поиска,
анализа и генерации знаний поверх P2P сети.

[ARCHITECTURE]
┌─────────────────────────────────────────────────────────────────┐
│                        CortexService                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Scout     │  │  Analyst    │  │      Librarian          │  │
│  │ (WebReader) │  │ (LLM/Ollama)│  │  (DHT + Topic Index)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Consilium                                 ││
│  │   [Analyst A] ──┐                                           ││
│  │   [Analyst B] ──┼──► [Judge LLM] ──► Final Answer           ││
│  │   [Analyst C] ──┘                                           ││
│  └─────────────────────────────────────────────────────────────┘│
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Automata (thought_loop)                     ││
│  │   Monitor Trends → Scout → Analyst → Librarian → DHT       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

[COMPONENTS]

Roles (roles.py):
- Scout: Разведчик - поиск и сбор информации из веба
- Analyst: Аналитик - анализ текста через LLM
- Librarian: Библиотекарь - хранение в DHT

Library (library.py):
- SemanticLibrary: Децентрализованный индекс знаний
- Topic Index: hash("topic:X") -> List[CID]
- Bounty System: Задачи на исследование

Consilium (consilium.py):
- Consilium: Оркестратор консенсуса
- convene_council(): Созыв совета аналитиков
- Judge LLM: Синтез ответов

Automata (automata.py):
- Automata: Автономный мыслящий агент
- thought_loop(): Фоновый цикл исследований
- Trend Sources: RSS, API, заглушки

Service (service.py):
- CortexService: Главный сервис интеграции
- create_cortex_service(): Factory function

[USAGE]

```python
from cortex import CortexService, create_cortex_service

# Создание через factory
cortex = create_cortex_service(
    node=node,
    ledger=ledger,
    agent_manager=agent_manager,
    kademlia=kademlia,
)

# Запуск
await cortex.start()

# Исследование темы
result = await cortex.investigate("quantum computing")

# Созыв совета
council = await cortex.convene_council("AI safety", text, budget=100)

# Поиск в библиотеке
reports = await cortex.search("machine learning")

# Остановка
await cortex.stop()
```
"""

# Roles
from .roles import (
    BaseRole,
    Scout,
    Analyst,
    Librarian,
    Task,
    TaskStatus,
    RoleResult,
    ScoutResult,
    AnalysisResult,
    LibrarianResult,
)

# Library
from .library import (
    SemanticLibrary,
    Report,
    Bounty,
    BountyStatus,
    TopicIndex,
)

# Consilium
from .consilium import (
    Consilium,
    CouncilResult,
    CouncilStatus,
    CouncilSession,
    AnalystCandidate,
    AnalystResponse,
    convene_council,
)

# Pathfinder / smart routing
from .pathfinder import VpnPathfinder

# Amplifier (traffic deduplication)
from .amplifier import Amplifier, AmplifierCache

# Automata
from .automata import (
    Automata,
    Investigation,
    InvestigationStatus,
    TrendTopic,
    TrendSource,
    StubTrendSource,
    RSSFeedSource,
    thought_loop,
)

# Service
from .service import (
    CortexService,
    CortexConfig,
    create_cortex_service,
)

# Prompts
from .prompts import (
    ANALYST_SYSTEM,
    ANALYST_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
    format_analyst_prompt,
    format_judge_prompt,
    format_scout_prompt,
    format_librarian_prompt,
)

__all__ = [
    # Roles
    "BaseRole",
    "Scout",
    "Analyst",
    "Librarian",
    "Task",
    "TaskStatus",
    "RoleResult",
    "ScoutResult",
    "AnalysisResult",
    "LibrarianResult",
    # Library
    "SemanticLibrary",
    "Report",
    "Bounty",
    "BountyStatus",
    "TopicIndex",
    # Consilium
    "Consilium",
    "CouncilResult",
    "CouncilStatus",
    "CouncilSession",
    "AnalystCandidate",
    "AnalystResponse",
    "convene_council",
    # Automata
    "Automata",
    "Investigation",
    "InvestigationStatus",
    "TrendTopic",
    "TrendSource",
    "StubTrendSource",
    "RSSFeedSource",
    "thought_loop",
    # Service
    "CortexService",
    "CortexConfig",
    "create_cortex_service",
    # Prompts
    "ANALYST_SYSTEM",
    "ANALYST_USER",
    "JUDGE_SYSTEM",
    "JUDGE_USER",
    "format_analyst_prompt",
    "format_judge_prompt",
    "format_scout_prompt",
    "format_librarian_prompt",
]

__version__ = "0.1.0"
