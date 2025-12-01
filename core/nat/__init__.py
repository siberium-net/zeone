"""
NAT Traversal Module
====================

Реализация NAT traversal для P2P сети:
- STUN: Обнаружение публичного IP и типа NAT
- Hole Punching: UDP и TCP hole punching
- P2P Relay: Любой узел с публичным IP может быть relay
- ICE: Координация всех методов подключения

[NAT TYPES]
- Full Cone: Любой внешний хост может отправить пакет
- Restricted Cone: Только хосты, которым мы отправляли
- Port Restricted: Только хосты+порты, которым мы отправляли  
- Symmetric: Разный mapping для каждого destination

[CONNECTION PRIORITY]
1. Direct (если оба узла имеют публичный IP)
2. Hole Punch (UDP, затем TCP)
3. P2P Relay (через узел с публичным IP)
"""

from .stun import STUNClient, NATType, MappedAddress
from .candidates import Candidate, CandidateType, CandidateGatherer
from .hole_punch import HolePuncher, HolePunchResult
from .relay import RelayServer, RelayClient, RelayConnection
from .ice import ICEAgent, ICEConnection, ICEState

__all__ = [
    # STUN
    "STUNClient",
    "NATType",
    "MappedAddress",
    # Candidates
    "Candidate",
    "CandidateType",
    "CandidateGatherer",
    # Hole Punch
    "HolePuncher",
    "HolePunchResult",
    # Relay
    "RelayServer",
    "RelayClient",
    "RelayConnection",
    # ICE
    "ICEAgent",
    "ICEConnection",
    "ICEState",
]

