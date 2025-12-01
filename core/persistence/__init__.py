"""
Persistence Module - Сохранение и восстановление состояния
==========================================================

[PRODUCTION] Обеспечивает:
- Сохранение списка известных пиров
- Восстановление DHT после перезапуска
- Сохранение сессий и балансов
- Graceful recovery

[COMPONENTS]
- StateManager: Централизованное управление состоянием
- PeerStore: Хранение информации о пирах
- SessionStore: Сохранение активных сессий
"""

from .state_manager import (
    StateManager,
    NodeState,
    PeerRecord,
)

from .peer_store import (
    PeerStore,
    PeerInfo,
)

__all__ = [
    "StateManager",
    "NodeState",
    "PeerRecord",
    "PeerStore",
    "PeerInfo",
]

