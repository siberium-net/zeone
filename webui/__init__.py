"""
P2P Node Web UI
===============

Веб-интерфейс для управления P2P узлом.

[FEATURES]
- Dashboard: статус узла, peers, статистика
- Services: управление услугами
- AI: LLM запросы, distributed inference
- Cortex: автономная система знаний
- DHT: распределённое хранилище
- Economy: балансы и транзакции
- Storage: файловое хранилище
- Compute: вычислительные задачи
- Neural Visualization: 3D граф сети

[TECH]
- NiceGUI: Python web framework
- WebSocket: real-time updates
- Three.js + 3d-force-graph: 3D visualization
- Async: неблокирующий I/O
"""

from .app import P2PWebUI, create_webui
from .vis_endpoint import CortexVisualizer, generate_demo_graph

__all__ = [
    "P2PWebUI",
    "create_webui",
    "CortexVisualizer",
    "generate_demo_graph",
]
