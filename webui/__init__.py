"""
P2P Node Web UI
===============

Веб-интерфейс для управления P2P узлом.

[FEATURES]
- Dashboard: статус узла, peers, статистика
- Services: управление услугами
- AI: LLM запросы, distributed inference
- DHT: распределённое хранилище
- Economy: балансы и транзакции
- Storage: файловое хранилище
- Compute: вычислительные задачи

[TECH]
- NiceGUI: Python web framework
- WebSocket: real-time updates
- Async: неблокирующий I/O
"""

from .app import P2PWebUI, create_webui

__all__ = ["P2PWebUI", "create_webui"]

