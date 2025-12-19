#!/usr/bin/env python3
"""
P2P Network Node - Production Ready
===================================

[DECENTRALIZATION] Этот скрипт запускает полностью децентрализованный узел:
- Генерирует или загружает криптографическую идентичность
- Запускает TCP сервер для входящих соединений
- Подключается к bootstrap-узлам
- Участвует в discovery для расширения сети

[ECONOMY] Layer 2 - Экономика и репутация:
- Автоматический учет трафика (debt/claim)
- Блокировка leechers при превышении лимита долга
- Обмен балансом при handshake

[MARKET] Layer 3 - Рынок услуг:
- Регистрация услуг (агентов) на узле
- Обработка SERVICE_REQUEST от других узлов
- Биллинг через Ledger

[PRODUCTION] Production-ready features:
- Persistence: сохранение состояния между перезапусками
- Security: rate limiting, DoS protection, ban system
- Monitoring: health checks, metrics export

[WEBUI] Web Interface:
- Dashboard, Peers, Services, AI, DHT, Economy
- Real-time updates
- Modern UI

Использование:
    python main.py [--port PORT] [--bootstrap HOST:PORT] [--identity FILE]
    python main.py --webui  # Запуск с Web UI

Примеры:
    # Запуск первого узла (bootstrap)
    python main.py --port 8468
    
    # Подключение к существующему узлу
    python main.py --port 8469 --bootstrap 127.0.0.1:8468
    
    # Запуск с Web UI
    python main.py --port 8468 --webui --webui-port 8080
    
    # Production mode с метриками
    python main.py --port 8468 --metrics --health-port 9090
    
    # Тест Echo сервиса (в консоли)
    >>> echo Hello World
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
from contextlib import suppress
from collections import deque

# Configure portable environment before heavy imports
ROOT_DIR = Path(__file__).parent
from core.env_setup import configure_environment
configure_environment(ROOT_DIR)
from core.logger import UIStreamHandler
from core.socks_proxy import SocksServer

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не установлен

from config import config, get_current_network, ZEONE_NETWORK
from core.node import Node
from core.transport import Crypto, Message, MessageType
from economy.ledger import Ledger, DEFAULT_DEBT_LIMIT_BYTES
from agents.manager import AgentManager, ServiceRequest, ServiceResponse
from cortex.pathfinder import VpnPathfinder
from cortex.amplifier import Amplifier

# Updater (optional, requires GitPython)
try:
    from core.updater import UpdateManager
    UPDATER_AVAILABLE = True
except ImportError:
    UpdateManager = None  # type: ignore[assignment]
    UPDATER_AVAILABLE = False

# Production imports (optional)
try:
    from core.persistence import StateManager, NodeState, PeerRecord as PersistentPeerRecord
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False

try:
    from core.security import RateLimiter, DoSProtector, RateLimitResult
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

try:
    from core.monitoring import HealthChecker, HealthStatus, MetricsCollector, get_metrics
    from core.monitoring.health import check_memory, check_disk, check_cpu
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Cortex - Autonomous Knowledge System
try:
    from cortex import CortexService, create_cortex_service
    CORTEX_AVAILABLE = True
except ImportError:
    CORTEX_AVAILABLE = False

# AI Module Lazy Loading
try:
    from core.lazy_imports import (
        is_ai_available, 
        get_ai_mode, 
        log_ai_status,
        AI_STATUS,
    )
    LAZY_IMPORTS_AVAILABLE = True
except ImportError:
    LAZY_IMPORTS_AVAILABLE = False
    def is_ai_available(): return False
    def get_ai_mode(): return "LITE"
    def log_ai_status(): pass


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("p2p")

# UI log handler buffer
ui_log_buffer = deque(maxlen=1000)
ui_handler = UIStreamHandler(ui_log_buffer)
ui_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(ui_handler)

def format_bytes(num_bytes: float) -> str:
    """Форматировать байты в читаемый вид."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def load_or_create_identity(identity_file: str) -> Crypto:
    """
    Загрузить или создать криптографическую идентичность узла.
    """
    path = Path(identity_file)
    
    if path.exists():
        logger.info(f"[IDENTITY] Loading from {identity_file}")
        key_bytes = path.read_bytes()
        crypto = Crypto.import_identity(key_bytes)
        logger.info(f"[IDENTITY] Loaded: {crypto.node_id[:16]}...")
    else:
        logger.info(f"[IDENTITY] Generating new identity...")
        crypto = Crypto()
        path.write_bytes(crypto.export_identity())
        logger.info(f"[IDENTITY] Created: {crypto.node_id[:16]}...")
        logger.info(f"[IDENTITY] Saved to {identity_file}")
    
    return crypto


def parse_bootstrap(bootstrap_str: str) -> List[Tuple[str, int]]:
    """Парсить строку bootstrap узлов."""
    nodes = []
    for item in bootstrap_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            host, port_str = item.rsplit(":", 1)
            port = int(port_str)
        else:
            host = item
            port = config.network.default_port
        nodes.append((host, port))
    return nodes


async def interactive_shell(
    node: Node,
    ledger: Ledger,
    agent_manager: AgentManager,
    kademlia = None,
    rate_limiter = None,
    dos_protector = None,
    health_checker = None,
    metrics_collector = None,
    shutdown_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Интерактивная оболочка для взаимодействия с узлом.
    """
    print("\n" + "=" * 60)
    print("P2P Node Interactive Shell (Production Ready)")
    print(f"Node ID: {node.node_id[:32]}...")
    print(f"Listening on: {node.host}:{node.port}")
    print(f"Services: {', '.join(agent_manager._agents.keys())}")
    if kademlia:
        print(f"DHT: Active (ID: {kademlia.local_id.hex()[:16]}...)")
    else:
        print("DHT: Not available")
    # Production status
    prod_features = []
    if rate_limiter:
        prod_features.append("RateLimit")
    if dos_protector:
        prod_features.append("DoS")
    if health_checker:
        prod_features.append("Health")
    if metrics_collector:
        prod_features.append("Metrics")
    if prod_features:
        print(f"Production: {', '.join(prod_features)}")
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 60 + "\n")
    
    async def _read_input(prompt: str) -> Optional[str]:
        try:
            from aioconsole import ainput  # type: ignore
        except Exception:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: input(prompt))

        input_task = asyncio.create_task(ainput(prompt))
        if shutdown_event is None:
            return await input_task

        stop_task = asyncio.create_task(shutdown_event.wait())
        done, _pending = await asyncio.wait(
            {input_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task in done:
            input_task.cancel()
            with suppress(asyncio.CancelledError):
                await input_task
            return None
        stop_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task
        return input_task.result()

    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            break
        try:
            line = await _read_input(">>> ")
            if line is None:
                break
            line = line.strip()
            
            if not line:
                continue
            
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd == "quit" or cmd == "exit":
                print("[INFO] Shutting down...")
                break
            
            elif cmd == "help":
                print("""
Available commands:
  peers             - List connected peers
  known             - List known peers  
  ping <peer_id>    - Send PING to peer
  broadcast <msg>   - Broadcast message to all peers
  stats             - Show node statistics
  trust <peer_id>   - Show Trust Score
  balance [peer_id] - Show balance with peer
  balances          - Show all balances
  ledger            - Show ledger statistics
  
[MARKET] Service commands:
  services          - List available services on this node
  echo <peer_id> <text>  - Send Echo request to peer (test billing)
  request <peer_id> <service> <data> <budget> - Send service request
  
[AI] LLM commands:
  ask <prompt>           - Ask local Ollama (llm_local service)
  ask <peer_id> <prompt> - Ask peer's LLM service
  models                 - List available Ollama models (local)

[DISTRIBUTED AI] Distributed Inference:
  dist generate <prompt> - Generate using distributed pipeline
  dist models            - List available distributed models
  dist shards            - Show local model shards
  dist load <model> <start> <end> - Load model shard (mock)
  dist unload            - Unload current shard
  dist status            - Show distributed inference status

[DHT] Distributed Hash Table:
  dht put <key> <value>  - Store data in DHT
  dht get <key>          - Retrieve data from DHT
  dht del <key>          - Delete data from local DHT storage
  dht info               - Show DHT statistics
  routing                - Show routing table info

[NAT] NAT Traversal:
  nat info               - Show NAT type and public address
  nat candidates         - Show local ICE candidates
  nat relay start        - Start P2P relay server (if public IP)
  nat relay stop         - Stop relay server
  nat connect <peer_id>  - Connect to peer using ICE

[PRODUCTION] Health & Monitoring:
  health              - Show health check status
  metrics             - Show collected metrics
  security            - Show security stats (rate limits, bans)
  banned              - List banned peers
  ban <peer_id> [sec] - Ban peer for N seconds
  unban <peer_id>     - Unban peer
  
Other:
  id                - Show this node's ID
  help              - Show this help
  quit              - Exit

[ECONOMY] Balance: + = they owe us, - = we owe them
""")
            
            elif cmd == "id":
                print(f"Node ID: {node.node_id}")
            
            elif cmd == "peers":
                peers = node.peer_manager.get_active_peers()
                if not peers:
                    print("[INFO] No connected peers")
                else:
                    print(f"[INFO] Connected peers ({len(peers)}):")
                    for peer in peers:
                        direction = "OUT" if peer.is_outbound else "IN"
                        blocked = "[BLOCKED]" if peer.blocked else ""
                        balance = await ledger.get_balance(peer.node_id)
                        print(
                            f"  - {peer.node_id[:12]}... @ {peer.host}:{peer.port} "
                            f"[{direction}] bal:{balance:+.0f} {blocked}"
                        )
            
            elif cmd == "known":
                known = node.peer_manager._known_peers
                if not known:
                    print("[INFO] No known peers")
                else:
                    print(f"[INFO] Known peers ({len(known)}):")
                    for info in known.values():
                        print(f"  - {info.node_id[:16]}... @ {info.host}:{info.port}")
            
            elif cmd == "ping":
                if not args:
                    print("[ERROR] Usage: ping <node_id_prefix>")
                    continue
                
                target_peer = None
                for peer in node.peer_manager.get_active_peers():
                    if peer.node_id.startswith(args) or peer.node_id[:16].startswith(args):
                        target_peer = peer
                        break
                
                if not target_peer:
                    print(f"[ERROR] Peer not found: {args}")
                    continue
                
                from core.protocol import PingPongHandler
                ping = PingPongHandler.create_ping(node.crypto)
                success, sent, _ = await target_peer.send_with_accounting(
                    ping, ledger, node.use_masking
                )
                
                if success:
                    print(f"[OK] PING sent to {target_peer.node_id[:16]}... ({sent} bytes)")
                else:
                    print(f"[ERROR] Failed to send PING")
            
            elif cmd == "broadcast":
                if not args:
                    print("[ERROR] Usage: broadcast <message>")
                    continue
                
                message = Message(
                    type=MessageType.DATA,
                    payload={"text": args},
                    sender_id=node.node_id,
                )
                count = await node.broadcast(message, with_accounting=True)
                print(f"[OK] Broadcast sent to {count} peers")
            
            elif cmd == "stats":
                peers = node.peer_manager.get_active_peers()
                known = node.peer_manager._known_peers
                
                total_sent = sum(p.bytes_sent for p in peers)
                total_recv = sum(p.bytes_received for p in peers)
                blocked_count = sum(1 for p in peers if p.blocked)
                
                ledger_stats = await ledger.get_stats()
                agent_stats = agent_manager.get_stats()
                
                print(f"""
Node Statistics:
  Connected peers: {len(peers)} ({blocked_count} blocked)
  Known peers: {len(known)}

Traffic (this session):
  Total sent: {format_bytes(total_sent)}
  Total received: {format_bytes(total_recv)}
  
Economy:
  Debt limit: {format_bytes(ledger.debt_limit)}
  Total owed to us: {format_bytes(ledger_stats['total_owed_to_us'])}
  Total we owe: {format_bytes(ledger_stats['total_we_owe'])}

Market:
  Registered services: {agent_stats['registered_services']}
  Total requests handled: {agent_stats['total_requests']}
  Total revenue: {agent_stats['total_revenue']:.2f} units
""")
            
            elif cmd == "trust":
                if not args:
                    print("[ERROR] Usage: trust <node_id_prefix>")
                    continue
                
                found = False
                for peer in node.peer_manager.get_active_peers():
                    if peer.node_id.startswith(args) or peer.node_id[:16].startswith(args):
                        score = await ledger.get_trust_score(peer.node_id)
                        print(f"Trust Score for {peer.node_id[:16]}...: {score:.4f}")
                        found = True
                        break
                
                if not found:
                    print(f"[ERROR] Peer not found: {args}")
            
            elif cmd == "balance":
                if args:
                    found = False
                    for peer in node.peer_manager.get_active_peers():
                        if peer.node_id.startswith(args) or peer.node_id[:16].startswith(args):
                            info = await ledger.get_balance_info(peer.node_id)
                            print(f"""
Balance with {peer.node_id[:16]}...:
  Current balance: {info['balance']:+.0f} ({format_bytes(abs(info['balance']))})
  Total sent: {format_bytes(info['total_sent'])}
  Total received: {format_bytes(info['total_received'])}
  Status: {"BLOCKED" if peer.blocked else "OK"}
  Meaning: {"They owe us" if info['balance'] > 0 else "We owe them" if info['balance'] < 0 else "Even"}
""")
                            found = True
                            break
                    
                    if not found:
                        print(f"[ERROR] Peer not found: {args}")
                else:
                    balances = await ledger.get_all_balances()
                    if not balances:
                        print("[INFO] No balance records")
                    else:
                        print(f"[INFO] All balances ({len(balances)}):")
                        for b in balances[:20]:
                            sign = "+" if b['balance'] > 0 else ""
                            status = "BLOCKED" if ledger.is_peer_blocked(b['peer_id']) else ""
                            print(
                                f"  {b['peer_id'][:12]}... "
                                f"bal:{sign}{b['balance']:.0f} {status}"
                            )
            
            elif cmd == "balances":
                balances = await ledger.get_all_balances()
                if not balances:
                    print("[INFO] No balance records")
                else:
                    total_positive = sum(b['balance'] for b in balances if b['balance'] > 0)
                    total_negative = sum(b['balance'] for b in balances if b['balance'] < 0)
                    
                    print(f"""
Balance Summary:
  Peers with balance: {len(balances)}
  Total owed to us: {format_bytes(total_positive)}
  Total we owe: {format_bytes(abs(total_negative))}
  Net position: {format_bytes(total_positive + total_negative)}
""")
            
            elif cmd == "ledger":
                stats = await ledger.get_stats()
                print(f"""
Ledger Statistics:
  Known peers: {stats['peer_count']}
  Transactions: {stats['transaction_count']}
  Active IOUs: {stats['active_ious']}
  Total owed to us: {format_bytes(stats['total_owed_to_us'])}
  Total we owe: {format_bytes(stats['total_we_owe'])}
  Debt limit: {format_bytes(stats['debt_limit'])}
""")
            
            # [MARKET] Service commands
            elif cmd == "services":
                services = agent_manager.list_services()
                if not services:
                    print("[INFO] No services registered")
                else:
                    print(f"[INFO] Available services ({len(services)}):")
                    for svc in services:
                        print(f"  - {svc['service_name']}: {svc['description']}")
                        print(f"    Price: {svc['price_per_unit']} per unit")
            
            elif cmd == "echo":
                # echo <peer_id> <text>
                parts = args.split(maxsplit=1)
                if len(parts) < 2:
                    print("[ERROR] Usage: echo <peer_id_prefix> <text>")
                    continue
                
                peer_prefix, text = parts
                
                target_peer = None
                for peer in node.peer_manager.get_active_peers():
                    if peer.node_id.startswith(peer_prefix) or peer.node_id[:16].startswith(peer_prefix):
                        target_peer = peer
                        break
                
                if not target_peer:
                    print(f"[ERROR] Peer not found: {peer_prefix}")
                    continue
                
                # Отправляем Echo запрос
                budget = len(text) / 10 + 1  # Примерный бюджет
                success = await node.request_service(
                    target_peer.node_id,
                    "echo",
                    text,
                    budget,
                )
                
                if success:
                    print(f"[OK] Echo request sent to {target_peer.node_id[:16]}...")
                    print(f"     Payload: '{text}' ({len(text)} bytes)")
                    print(f"     Budget: {budget:.2f} units")
                    print("     Waiting for response...")
                else:
                    print(f"[ERROR] Failed to send echo request")
            
            elif cmd == "request":
                # request <peer_id> <service> <payload> <budget>
                parts = args.split(maxsplit=3)
                if len(parts) < 4:
                    print("[ERROR] Usage: request <peer_id> <service_name> <payload> <budget>")
                    continue
                
                peer_prefix, service_name, payload, budget_str = parts
                
                try:
                    budget = float(budget_str)
                except ValueError:
                    print(f"[ERROR] Invalid budget: {budget_str}")
                    continue
                
                target_peer = None
                for peer in node.peer_manager.get_active_peers():
                    if peer.node_id.startswith(peer_prefix) or peer.node_id[:16].startswith(peer_prefix):
                        target_peer = peer
                        break
                
                if not target_peer:
                    print(f"[ERROR] Peer not found: {peer_prefix}")
                    continue
                
                success = await node.request_service(
                    target_peer.node_id,
                    service_name,
                    payload,
                    budget,
                )
                
                if success:
                    print(f"[OK] Service request sent: {service_name}")
                else:
                    print(f"[ERROR] Failed to send request")
            
            # [AI] LLM commands
            elif cmd == "ask":
                # ask <prompt> - локальный запрос к Ollama
                # ask <peer_id> <prompt> - запрос к пиру
                if not args:
                    print("[ERROR] Usage: ask <prompt> OR ask <peer_id> <prompt>")
                    continue
                
                parts = args.split(maxsplit=1)
                
                # Проверяем, похоже ли первое слово на peer_id
                potential_peer = parts[0]
                is_peer_id = False
                target_peer = None
                
                if len(potential_peer) >= 8:  # Минимум 8 символов для ID
                    for peer in node.peer_manager.get_active_peers():
                        if peer.node_id.startswith(potential_peer) or peer.node_id[:16].startswith(potential_peer):
                            is_peer_id = True
                            target_peer = peer
                            break
                
                if is_peer_id and target_peer and len(parts) > 1:
                    # Запрос к пиру
                    prompt = parts[1]
                    print(f"[AI] Sending prompt to peer {target_peer.node_id[:16]}...")
                    print(f"     Prompt: '{prompt[:50]}...' ({len(prompt)} chars)")
                    
                    success = await node.request_service(
                        target_peer.node_id,
                        "llm_local",  # Или "llm_prompt" для облачного
                        {"prompt": prompt},
                        50,  # Бюджет
                    )
                    
                    if success:
                        print("[OK] LLM request sent, waiting for response...")
                    else:
                        print("[ERROR] Failed to send LLM request")
                else:
                    # Локальный запрос к Ollama
                    prompt = args
                    
                    # Проверяем, есть ли OllamaAgent
                    ollama_agent = agent_manager._agents.get("llm_local")
                    if not ollama_agent:
                        print("[ERROR] OllamaAgent not available on this node")
                        continue
                    
                    print(f"[AI] Asking local Ollama...")
                    print(f"     Prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
                    
                    try:
                        result, units = await ollama_agent.execute({"prompt": prompt})
                        
                        if "error" in result:
                            print(f"[ERROR] Ollama: {result['error']}")
                        else:
                            response = result.get("response", "")
                            model = result.get("model", "unknown")
                            eval_count = result.get("eval_count", 0)
                            
                            print(f"\n[AI Response] ({model}, {eval_count} tokens)")
                            print("-" * 40)
                            print(response)
                            print("-" * 40)
                            print(f"Cost: {units * ollama_agent.price_per_unit:.1f} units")
                    except Exception as e:
                        print(f"[ERROR] {e}")
            
            elif cmd == "models":
                # Показать доступные модели Ollama
                ollama_agent = agent_manager._agents.get("llm_local")
                if not ollama_agent:
                    print("[ERROR] OllamaAgent not available on this node")
                    continue
                
                print("[AI] Checking Ollama models...")
                try:
                    models = await ollama_agent.list_models()
                    if models:
                        print(f"[OK] Available models ({len(models)}):")
                        for m in models:
                            default_marker = " [DEFAULT]" if m == ollama_agent.model_name else ""
                            print(f"  - {m}{default_marker}")
                    else:
                        print("[WARN] No models found or Ollama offline")
                except Exception as e:
                    print(f"[ERROR] {e}")
            
            # [DISTRIBUTED AI] Distributed Inference commands
            elif cmd == "dist":
                if not args:
                    print("[ERROR] Usage: dist <generate|models|shards|load|unload|status> [args]")
                    continue
                
                parts = args.split(maxsplit=2)
                subcmd = parts[0].lower()
                
                # Получаем distributed компоненты (если доступны)
                dist_client = getattr(node, '_dist_client', None)
                dist_worker = getattr(node, '_dist_worker', None)
                dist_registry = getattr(node, '_dist_registry', None)
                
                if subcmd == "generate":
                    prompt = parts[1] if len(parts) > 1 else ""
                    if not prompt:
                        print("[ERROR] Usage: dist generate <prompt>")
                        continue
                    
                    if not dist_client:
                        # Fallback на локальный Ollama
                        print("[INFO] Distributed client not available, using local Ollama")
                        ollama_agent = agent_manager._agents.get("llm_local")
                        if ollama_agent:
                            print(f"[AI] Generating with local Ollama...")
                            result, cost, error = await ollama_agent.execute({"prompt": prompt})
                            if error:
                                print(f"[ERROR] {error}")
                            else:
                                print(f"\n[RESPONSE]\n{result.get('response', '')}")
                                print(f"\n[INFO] Tokens: {result.get('eval_count', 0)}, Cost: {cost:.2f}")
                        else:
                            print("[ERROR] No inference backend available")
                        continue
                    
                    print(f"[DIST] Generating with distributed pipeline...")
                    try:
                        result = await dist_client.generate(
                            model="qwen2.5-32b",
                            prompt=prompt,
                            max_tokens=100,
                        )
                        if result.success:
                            print(f"\n[RESPONSE]\n{result.text}")
                            print(f"\n[INFO] Model: {result.model_used}")
                            print(f"       Tokens: {result.tokens_generated}")
                            print(f"       Speed: {result.tokens_per_second:.1f} t/s")
                            print(f"       Distributed: {result.distributed}")
                            if result.distributed:
                                print(f"       Shards: {result.shards_used}")
                        else:
                            print(f"[ERROR] {result.error}")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                
                elif subcmd == "models":
                    print("[DIST] Known distributed models:")
                    try:
                        from agents.distributed.registry import KNOWN_MODELS
                        for name, info in KNOWN_MODELS.items():
                            print(f"  - {name}")
                            print(f"      Layers: {info.total_layers}")
                            print(f"      Params: {info.total_params_b}B")
                            print(f"      Shards: {info.recommended_shards}")
                            print(f"      Memory/shard: {info.memory_per_shard_gb}GB")
                    except ImportError:
                        print("[ERROR] Distributed module not available")
                
                elif subcmd == "shards":
                    if dist_registry:
                        shards = dist_registry.get_local_shards()
                        if shards:
                            print(f"[DIST] Local shards ({len(shards)}):")
                            for s in shards:
                                print(f"  - {s.model_name} layers {s.layer_start}-{s.layer_end}")
                                print(f"      Status: {s.status.value}")
                                print(f"      Memory: {s.gpu_memory_mb}MB")
                        else:
                            print("[INFO] No local shards loaded")
                    else:
                        print("[INFO] No shards loaded (registry not initialized)")
                
                elif subcmd == "load":
                    if len(parts) < 4:
                        print("[ERROR] Usage: dist load <model> <layer_start> <layer_end>")
                        print("        Example: dist load qwen2.5-32b 0 16")
                        continue
                    
                    model_name = parts[1]
                    try:
                        # Парсим layer_start и layer_end из оставшихся аргументов
                        remaining = parts[2] if len(parts) > 2 else ""
                        layer_parts = remaining.split()
                        if len(layer_parts) < 2:
                            print("[ERROR] Usage: dist load <model> <layer_start> <layer_end>")
                            continue
                        layer_start = int(layer_parts[0])
                        layer_end = int(layer_parts[1])
                    except ValueError:
                        print("[ERROR] layer_start and layer_end must be integers")
                        continue
                    
                    print(f"[DIST] Loading shard: {model_name} layers {layer_start}-{layer_end}")
                    print("       (Using mock shard for demo)")
                    
                    try:
                        from agents.distributed.worker import InferenceWorker
                        from agents.distributed.registry import ModelRegistry
                        
                        # Инициализируем worker если нужно
                        if not dist_worker:
                            dist_registry = ModelRegistry()
                            dist_worker = InferenceWorker(
                                node_id=node.node_id,
                                registry=dist_registry,
                                host=node.host,
                                port=node.port,
                            )
                            node._dist_worker = dist_worker
                            node._dist_registry = dist_registry
                        
                        success = await dist_worker.load_shard(
                            model_name=model_name,
                            layer_start=layer_start,
                            layer_end=layer_end,
                            use_mock=True,  # Mock для demo
                        )
                        
                        if success:
                            await dist_worker.start()
                            print(f"[OK] Shard loaded and worker started")
                            print(f"     Shard ID: {dist_worker.current_shard_id}")
                        else:
                            print("[ERROR] Failed to load shard")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                
                elif subcmd == "unload":
                    if dist_worker:
                        await dist_worker.unload_shard()
                        print("[OK] Shard unloaded")
                    else:
                        print("[INFO] No shard loaded")
                
                elif subcmd == "status":
                    print("[DIST] Distributed Inference Status:")
                    print(f"  Client: {'Active' if dist_client else 'Not initialized'}")
                    print(f"  Worker: {'Active' if dist_worker else 'Not initialized'}")
                    print(f"  Registry: {'Active' if dist_registry else 'Not initialized'}")
                    
                    if dist_worker:
                        stats = dist_worker.get_stats()
                        print(f"\n  Worker State: {stats.get('state', 'Unknown')}")
                        if stats.get('shard'):
                            shard = stats['shard']
                            print(f"  Shard: {shard.get('shard_id', 'None')}")
                            print(f"  Loaded: {shard.get('is_loaded', False)}")
                            print(f"  Memory: {shard.get('memory_mb', 0)}MB")
                            print(f"  Requests: {shard.get('total_requests', 0)}")
                else:
                    print("[ERROR] Unknown subcommand. Use: generate, models, shards, load, unload, status")
            
            # [DHT] Distributed Hash Table commands
            elif cmd == "dht":
                if not kademlia:
                    print("[ERROR] DHT not available")
                    continue
                
                if not args:
                    print("[ERROR] Usage: dht <put|get|del|info> [args]")
                    continue
                
                parts = args.split(maxsplit=2)
                subcmd = parts[0].lower()
                
                if subcmd == "put":
                    if len(parts) < 3:
                        print("[ERROR] Usage: dht put <key> <value>")
                        continue
                    
                    key = parts[1]
                    value = parts[2].encode('utf-8')
                    
                    print(f"[DHT] Storing '{key}'...")
                    try:
                        count = await kademlia.dht_put(key, value)
                        print(f"[OK] Stored on {count} nodes")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                
                elif subcmd == "get":
                    if len(parts) < 2:
                        print("[ERROR] Usage: dht get <key>")
                        continue
                    
                    key = parts[1]
                    
                    print(f"[DHT] Looking up '{key}'...")
                    try:
                        value = await kademlia.dht_get(key)
                        if value:
                            try:
                                text = value.decode('utf-8')
                                print(f"[OK] Found: {text}")
                            except UnicodeDecodeError:
                                print(f"[OK] Found: {value.hex()} ({len(value)} bytes)")
                        else:
                            print(f"[WARN] Key '{key}' not found in DHT")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                
                elif subcmd == "del":
                    if len(parts) < 2:
                        print("[ERROR] Usage: dht del <key>")
                        continue
                    
                    key = parts[1]
                    
                    try:
                        deleted = await kademlia.dht_delete(key)
                        if deleted:
                            print(f"[OK] Deleted '{key}' from local storage")
                        else:
                            print(f"[WARN] Key '{key}' not found locally")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                
                elif subcmd == "info":
                    try:
                        stats = await kademlia.get_full_stats()
                        print(f"""
[DHT] Statistics:
  Local ID: {stats['local_id'][:32]}...
  Running: {stats['running']}

  Routing Table:
    Total nodes: {stats['routing_table']['total_nodes']}
    Non-empty buckets: {stats['routing_table']['non_empty_buckets']}/{stats['routing_table']['total_buckets']}
    K (bucket size): {stats['routing_table']['k']}

  Storage:
    Active entries: {stats['storage']['active_entries']}
    Total size: {stats['storage']['total_size_bytes']} bytes
""")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                
                else:
                    print(f"[ERROR] Unknown DHT command: {subcmd}")
                    print("  Available: put, get, del, info")
            
            elif cmd == "routing":
                if not kademlia:
                    print("[ERROR] DHT not available")
                    continue
                
                stats = kademlia.routing_table.get_stats()
                print(f"""
[DHT] Routing Table:
  Local ID: {stats['local_id'][:32]}...
  Total nodes: {stats['total_nodes']}
  Non-empty buckets: {stats['non_empty_buckets']}/{stats['total_buckets']}
  K (bucket size): {stats['k']}
  
  Bucket sizes (first 10 non-empty):
    {stats['bucket_sizes']}
""")
            
            # [NAT] NAT Traversal commands
            elif cmd == "nat":
                if not args:
                    print("[ERROR] Usage: nat <info|candidates|relay|connect> [args]")
                    continue
                
                parts = args.split(maxsplit=1)
                subcmd = parts[0].lower()
                subargs = parts[1] if len(parts) > 1 else ""
                
                if subcmd == "info":
                    # Показать информацию о NAT
                    print("[NAT] Detecting NAT type...")
                    try:
                        from core.nat import STUNClient
                        stun = STUNClient()
                        mapped = await stun.get_mapped_address(node.port)
                        
                        if mapped:
                            nat_type = await stun.detect_nat_type(node.port)
                            print(f"""
[NAT] Information:
  Local address: {mapped.local_ip}:{mapped.local_port}
  Public address: {mapped.ip}:{mapped.port}
  NAT type: {nat_type.name}
  Is public: {mapped.is_public}
""")
                        else:
                            print("[WARN] Could not determine public address")
                            print("       STUN servers may be unreachable")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                
                elif subcmd == "candidates":
                    # Показать ICE candidates
                    print("[NAT] Gathering ICE candidates...")
                    try:
                        from core.nat import CandidateGatherer
                        gatherer = CandidateGatherer(node.port)
                        candidates = await gatherer.gather()
                        
                        print(f"\n[NAT] ICE Candidates ({len(candidates)}):")
                        for c in candidates:
                            pub = "[PUBLIC]" if c.is_public else "[PRIVATE]"
                            print(f"  {c.type.value:6} {c.transport.value:3} {c.ip}:{c.port} {pub}")
                        
                        # Публикуем в DHT если доступен
                        if kademlia and candidates:
                            candidates_data = [c.to_dict() for c in candidates]
                            import json
                            await kademlia.dht_put(
                                f"ice:{node.node_id[:32]}",
                                json.dumps(candidates_data).encode()
                            )
                            print(f"\n[OK] Published candidates to DHT")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                
                elif subcmd == "relay":
                    # Управление relay сервером
                    relay_cmd = subargs.lower() if subargs else "status"
                    
                    if relay_cmd == "start":
                        if hasattr(node, '_relay_server') and node._relay_server:
                            print("[WARN] Relay server already running")
                        else:
                            try:
                                from core.nat import RelayServer
                                relay = RelayServer(
                                    host="0.0.0.0",
                                    port=node.port + 1,
                                    node_id=node.node_id,
                                )
                                await relay.start()
                                node._relay_server = relay
                                
                                print(f"[OK] Relay server started on port {node.port + 1}")
                                
                                # Публикуем в DHT
                                if kademlia:
                                    from core.nat import STUNClient
                                    stun = STUNClient()
                                    mapped = await stun.get_mapped_address(node.port + 1)
                                    if mapped and mapped.is_public:
                                        import json
                                        relay_info = {
                                            "node_id": node.node_id,
                                            "ip": mapped.ip,
                                            "port": mapped.port,
                                            "capacity": relay.available_slots,
                                        }
                                        await kademlia.dht_put(
                                            f"relay:{node.node_id[:16]}",
                                            json.dumps(relay_info).encode()
                                        )
                                        print(f"[OK] Published relay to DHT: {mapped.ip}:{mapped.port}")
                            except Exception as e:
                                print(f"[ERROR] {e}")
                    
                    elif relay_cmd == "stop":
                        if hasattr(node, '_relay_server') and node._relay_server:
                            await node._relay_server.stop()
                            node._relay_server = None
                            print("[OK] Relay server stopped")
                        else:
                            print("[WARN] Relay server not running")
                    
                    else:  # status
                        if hasattr(node, '_relay_server') and node._relay_server:
                            stats = node._relay_server.get_stats()
                            print(f"""
[NAT] Relay Server:
  Running: {stats['running']}
  Port: {stats['port']}
  Connected peers: {stats['connected_peers']}/{stats['max_peers']}
  Total relayed: {stats['total_bytes_relayed']} bytes
""")
                        else:
                            print("[NAT] Relay server not running")
                
                elif subcmd == "connect":
                    # Подключиться к пиру через ICE
                    if not subargs:
                        print("[ERROR] Usage: nat connect <peer_id_prefix>")
                        continue
                    
                    peer_prefix = subargs
                    
                    # Ищем peer в DHT
                    if not kademlia:
                        print("[ERROR] DHT not available for peer lookup")
                        continue
                    
                    print(f"[NAT] Looking up ICE candidates for {peer_prefix}...")
                    
                    # Пробуем найти candidates в DHT
                    import json
                    candidates_raw = await kademlia.dht_get(f"ice:{peer_prefix}")
                    
                    if not candidates_raw:
                        print(f"[ERROR] No ICE candidates found for {peer_prefix}")
                        continue
                    
                    try:
                        remote_candidates_data = json.loads(candidates_raw.decode())
                        from core.nat import Candidate, ICEAgent
                        
                        remote_candidates = [
                            Candidate.from_dict(c) for c in remote_candidates_data
                        ]
                        
                        print(f"[NAT] Found {len(remote_candidates)} remote candidates")
                        
                        # Создаём ICE agent и подключаемся
                        ice = ICEAgent(
                            node_id=node.node_id,
                            local_port=node.port,
                        )
                        
                        print("[NAT] Establishing ICE connection...")
                        connection = await ice.connect(remote_candidates, peer_prefix)
                        
                        if connection:
                            print(f"""
[OK] ICE Connection established!
  Type: {connection.connection_type}
  Local: {connection.local_candidate.ip}:{connection.local_candidate.port}
  Remote: {connection.remote_candidate.ip}:{connection.remote_candidate.port}
  Latency: {connection.latency_ms:.1f}ms
""")
                        else:
                            print("[ERROR] Failed to establish ICE connection")
                            print("        Try: nat relay start (on a node with public IP)")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                
                else:
                    print(f"[ERROR] Unknown NAT command: {subcmd}")
                    print("  Available: info, candidates, relay, connect")
            
            # [PRODUCTION] Health & Monitoring commands
            elif cmd == "health":
                if not health_checker:
                    print("[WARN] Health checker not available (start with --metrics)")
                else:
                    status = await health_checker.get_status()
                    print(f"\n[HEALTH] Overall: {status['status'].upper()}")
                    print(f"  Uptime: {status.get('uptime', 0):.0f}s")
                    print("\n  Components:")
                    for name, comp in status.get('components', {}).items():
                        status_str = comp['status'].upper()
                        msg = comp.get('message', '')
                        rt = comp.get('response_time_ms', 0)
                        print(f"    {name}: {status_str} ({msg}, {rt:.1f}ms)")
            
            elif cmd == "metrics":
                if not metrics_collector:
                    print("[WARN] Metrics collector not available (start with --metrics)")
                else:
                    data = metrics_collector.export_json()
                    print("\n[METRICS]")
                    print("\n  Counters:")
                    for name, value in data.get('counters', {}).items():
                        if isinstance(value, (int, float)):
                            print(f"    {name}: {value}")
                    print("\n  Gauges:")
                    for name, value in data.get('gauges', {}).items():
                        if isinstance(value, (int, float)):
                            print(f"    {name}: {value}")
            
            elif cmd == "security":
                if not rate_limiter and not dos_protector:
                    print("[WARN] Security modules not available (start without --no-security)")
                else:
                    print("\n[SECURITY]")
                    if rate_limiter:
                        stats = rate_limiter.get_stats()
                        print(f"\n  Rate Limiter:")
                        print(f"    Total checks: {stats.get('total_checks', 0)}")
                        print(f"    Allowed: {stats.get('total_allowed', 0)}")
                        print(f"    Denied: {stats.get('total_denied', 0)}")
                        print(f"    Tracked peers: {stats.get('tracked_peers', 0)}")
                        print(f"    Banned: {stats.get('banned_peers', 0)}")
                    if dos_protector:
                        stats = dos_protector.get_stats()
                        print(f"\n  DoS Protector:")
                        print(f"    Tracked peers: {stats.get('tracked_peers', 0)}")
                        print(f"    Temp banned: {stats.get('temp_banned', 0)}")
                        print(f"    Perm banned: {stats.get('perm_banned', 0)}")
                        print(f"    Global conn/min: {stats.get('global_connection_rate', 0):.1f}")
                        print(f"    Global msg/sec: {stats.get('global_message_rate', 0):.1f}")
            
            elif cmd == "banned":
                if not dos_protector:
                    print("[WARN] DoS protector not available")
                else:
                    stats = dos_protector.get_stats()
                    temp = stats.get('temp_banned', 0)
                    perm = stats.get('perm_banned', 0)
                    if temp == 0 and perm == 0:
                        print("[INFO] No banned peers")
                    else:
                        print(f"\n[BANNED] Temp: {temp}, Permanent: {perm}")
                        events = dos_protector.get_recent_events(20)
                        if events:
                            print("\n  Recent threats:")
                            for event in events[-10:]:
                                print(f"    {event['peer_id']} - {event['attack_type']} ({event['threat_level']})")
            
            elif cmd == "ban":
                if not dos_protector:
                    print("[WARN] DoS protector not available")
                else:
                    ban_args = args.split()
                    if len(ban_args) < 1:
                        print("[ERROR] Usage: ban <peer_id> [duration_seconds]")
                    else:
                        peer_prefix = ban_args[0]
                        duration = float(ban_args[1]) if len(ban_args) > 1 else 3600.0
                        
                        # Find full peer_id
                        full_id = None
                        for p in node.peer_manager.get_active_peers():
                            if p.node_id.startswith(peer_prefix):
                                full_id = p.node_id
                                break
                        
                        if full_id:
                            dos_protector.ban(full_id, duration=duration)
                            print(f"[OK] Banned {full_id[:16]}... for {duration}s")
                        else:
                            # Ban by prefix anyway
                            dos_protector.ban(peer_prefix, duration=duration)
                            print(f"[OK] Banned {peer_prefix}... for {duration}s")
            
            elif cmd == "unban":
                if not dos_protector:
                    print("[WARN] DoS protector not available")
                else:
                    if not args:
                        print("[ERROR] Usage: unban <peer_id>")
                    else:
                        dos_protector.unban(args)
                        print(f"[OK] Unbanned {args[:16]}...")
            
            else:
                print(f"[ERROR] Unknown command: {cmd}. Type 'help' for commands.")
                
        except EOFError:
            print("\n[INFO] EOF received, shutting down...")
            break
        except EOFError:
            print("\n[INFO] EOF received, shutting down...")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


async def main() -> None:
    """
    Главная функция - точка входа.
    """
    # Log active blockchain network (kept inside main to avoid noise when main.py is imported)
    _net = get_current_network()
    logger.info(
        "[INFO] Starting in NETWORK: %s (%s)",
        _net["name"],
        _net["chain_id"],
    )
    parser = argparse.ArgumentParser(
        description="P2P Network Node (Layer 3: Market)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start as bootstrap node
  python main.py --port 8468
  
  # Connect to existing network
  python main.py --port 8469 --bootstrap 127.0.0.1:8468
  
  # Test Echo service between two nodes:
  # Terminal 1: python main.py --port 8468
  # Terminal 2: python main.py --port 8469 --bootstrap 127.0.0.1:8468
  # In Terminal 2: echo <peer_id> Hello World
""",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=config.network.default_port,
        help=f"Port to listen on (default: {config.network.default_port})",
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--bootstrap", "-b",
        type=str,
        default="",
        help="Bootstrap nodes (comma-separated host:port)",
    )
    parser.add_argument(
        "--identity", "-i",
        type=str,
        default=config.crypto.identity_file,
        help=f"Identity file path (default: {config.crypto.identity_file})",
    )
    parser.add_argument(
        "--db", "-d",
        type=str,
        default=config.ledger.database_path,
        help=f"Database file path (default: {config.ledger.database_path})",
    )
    parser.add_argument(
        "--debt-limit",
        type=int,
        default=DEFAULT_DEBT_LIMIT_BYTES,
        help=f"Debt limit in bytes (default: {DEFAULT_DEBT_LIMIT_BYTES})",
    )
    parser.add_argument(
        "--masking", "-m",
        action="store_true",
        help="Enable HTTP traffic masking",
    )
    parser.add_argument(
        "--no-shell",
        action="store_true",
        help="Disable interactive shell",
    )
    parser.add_argument(
        "--webui", "-w",
        action="store_true",
        help="Enable Web UI (default port: 8080)",
    )
    parser.add_argument(
        "--webui-port",
        type=int,
        default=8080,
        help="Web UI port (default: 8080)",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Enable MCP SSE server (default port: 8090)",
    )
    parser.add_argument(
        "--mcp-host",
        type=str,
        default="0.0.0.0",
        help="MCP server bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8090,
        help="MCP server port (default: 8090)",
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="Enable auto-update from git with periodic checks",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--genesis",
        action="store_true",
        help="Run Genesis Protocol on start (auto-adapt to hardware)",
    )
    parser.add_argument(
        "--genesis-epochs",
        type=int,
        default=10,
        help="Genesis epochs (default: 10)",
    )
    parser.add_argument(
        "--genesis-population",
        type=int,
        default=20,
        help="Genesis population size (default: 20)",
    )
    parser.add_argument(
        "--genesis-data-dir",
        type=str,
        default="data",
        help="Genesis output directory (default: data)",
    )
    parser.add_argument(
        "--genesis-niche",
        type=str,
        default="",
        help="Force niche for Genesis (e.g. neural_miner, traffic_weaver, storage_keeper, chain_weaver)",
    )
    
    # Production arguments
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Enable metrics collection and export",
    )
    parser.add_argument(
        "--exit-node",
        action="store_true",
        help="Run as VPN exit node (advertise in DHT and serve vpn_exit)",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=0,
        help="HTTP port for health/metrics endpoints (0=disabled)",
    )
    parser.add_argument(
        "--no-persistence",
        action="store_true",
        help="Disable state persistence",
    )
    parser.add_argument(
        "--no-security",
        action="store_true",
        help="Disable rate limiting and DoS protection",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # [GENESIS] Controlled bootstrap (diagnostics + evolution).
    # Run before any networking / persistence so it works in restricted environments.
    if args.genesis:
        try:
            from cortex.genesis import run_genesis

            result = await run_genesis(
                epochs=args.genesis_epochs,
                population=args.genesis_population,
                data_dir=args.genesis_data_dir,
                niche=(args.genesis_niche or None),
            )
            logger.info("[GENESIS] Complete: %s", result.get("niche"))
        except Exception as e:
            logger.error("[GENESIS] Failed: %s", e)
        return
    
    # Парсим bootstrap узлы
    bootstrap_nodes = parse_bootstrap(args.bootstrap)
    logger.info(f"[BOOTSTRAP] Parsed bootstrap args: {bootstrap_nodes}")
    if bootstrap_nodes:
        config.network.bootstrap_nodes = bootstrap_nodes
    
    # Загружаем идентичность
    crypto = load_or_create_identity(args.identity)
    
    # [PRODUCTION] Initialize StateManager for persistence
    state_manager = None
    saved_state = None
    if PERSISTENCE_AVAILABLE and not args.no_persistence:
        state_manager = StateManager(
            db_path=config.persistence.state_db_path,
            auto_save_interval=config.persistence.auto_save_interval,
        )
        await state_manager.initialize()
        saved_state = await state_manager.load_state()
        if saved_state:
            logger.info(f"[PERSISTENCE] Restored state: {len(saved_state.peers)} peers")
        else:
            logger.info("[PERSISTENCE] No saved state found, starting fresh")
    
    # [PRODUCTION] Initialize Security modules
    rate_limiter = None
    dos_protector = None
    if SECURITY_AVAILABLE and not args.no_security:
        rate_limiter = RateLimiter(
            default_tokens_per_second=config.security.rate_limit_messages_per_sec,
            default_bucket_size=config.security.rate_limit_burst_size,
        )
        await rate_limiter.start()
        
        dos_protector = DoSProtector(
            max_connections_per_minute=config.security.rate_limit_connections_per_min,
            max_messages_per_second=config.security.rate_limit_messages_per_sec,
            max_bandwidth_per_second=config.security.dos_max_bandwidth_per_sec,
            temp_ban_duration=config.security.dos_temp_ban_duration,
            perm_ban_threshold=config.security.dos_perm_ban_threshold,
        )
        await dos_protector.start()
        
        # Whitelist bootstrap nodes
        for node_id in config.security.whitelist:
            rate_limiter.whitelist_add(node_id)
            dos_protector.whitelist_add(node_id)
        
        logger.info("[SECURITY] Rate limiter and DoS protector started")
    
    # [PRODUCTION] Initialize Monitoring
    health_checker = None
    metrics_collector = None
    if MONITORING_AVAILABLE and (args.metrics or args.health_port > 0):
        metrics_collector = get_metrics()
        
        health_checker = HealthChecker(
            check_interval=config.monitoring.health_check_interval,
            timeout=config.monitoring.health_check_timeout,
        )
        
        # Register health checks
        health_checker.register("memory", check_memory)
        health_checker.register("disk", check_disk)
        health_checker.register("cpu", check_cpu)
        
        await health_checker.start()
        logger.info("[MONITORING] Health checker and metrics collector started")
    
    # Инициализируем Ledger
    ledger = Ledger(args.db, debt_limit=args.debt_limit)
    await ledger.initialize()
    logger.info(f"[LEDGER] Initialized: {args.db}")
    
    # [AI] Log AI module availability
    if LAZY_IMPORTS_AVAILABLE:
        log_ai_status()
    
    # [MARKET] Инициализируем AgentManager с Ledger и node_id
    agent_manager = AgentManager(ledger=ledger, node_id=crypto.node_id)
    logger.info(f"[AGENTS] Initialized with services: {list(agent_manager._agents.keys())}")
    logger.info(f"[MODE] Running in {get_ai_mode()} mode")
    try:
        from agents.cache_provider import CacheProviderAgent
        cache_agent = CacheProviderAgent(amplifier=None)
    except Exception as e:
        cache_agent = None
        logger.warning(f"[AGENTS] CacheProviderAgent unavailable: {e}")

    # Vector store / background ingestion (optional)
    idle_worker = None
    try:
        from cortex.archivist.vector_store import VectorStore
        from cortex.background import IdleWorker

        vector_store = VectorStore()
        idle_worker = IdleWorker(ledger=ledger, vector_store=vector_store)
        await idle_worker.start()
    except Exception as e:
        idle_worker = None
        logger.warning(f"[CORTEX] IdleWorker disabled: {e}")
    
    # Создаем узел с интегрированным ledger и agent_manager
    node = Node(
        crypto=crypto,
        host=args.host,
        port=args.port,
        use_masking=args.masking,
        ledger=ledger,
        agent_manager=agent_manager,
        amplifier=None,  # set after amplifier init
    )
    
    socks_server: Optional[SocksServer] = None
    pathfinder: Optional[VpnPathfinder] = None
    amplifier: Optional[Amplifier] = None
    vpn_tasks: List[asyncio.Task] = []
    
    # Callbacks
    async def on_peer_connected(peer):
        await ledger.get_or_create_peer(peer.node_id)
        await ledger.update_trust_score(peer.node_id, "ping_responded")
    
    async def on_peer_disconnected(peer):
        await ledger.update_trust_score(peer.node_id, "ping_timeout")
        balance = await ledger.get_balance(peer.node_id)
        logger.info(f"[ECONOMY] Peer disconnected: {peer.node_id[:12]}..., balance={balance:+.0f}")
    
    async def on_message(message, peer):
        if crypto.verify_signature(message):
            await ledger.update_trust_score(peer.node_id, "valid_message")
        else:
            await ledger.update_trust_score(peer.node_id, "invalid_message")
    
    async def on_service_response(response: ServiceResponse, sender_id: str):
        """Callback для ответов на запросы услуг."""
        if response.success:
            print(f"\n[SERVICE] Response from {sender_id[:12]}...")
            print(f"  Result: {response.result}")
            print(f"  Cost: {response.cost:.2f} units")
            print(f"  Time: {response.execution_time:.3f}s")
        else:
            print(f"\n[SERVICE] Error from {sender_id[:12]}...: {response.error}")
        print(">>> ", end="", flush=True)  # Восстанавливаем промпт
    
    node.on_peer_connected(on_peer_connected)
    node.on_peer_disconnected(on_peer_disconnected)
    node.on_message(on_message)
    node.on_service_response(on_service_response)
    
    # Запускаем узел
    logger.info(f"[BOOTSTRAP] Using peers: {config.network.bootstrap_nodes}")
    await node.start()

    # Periodic auto-update loop
    if args.auto_update:
        if not UPDATER_AVAILABLE or UpdateManager is None:
            logger.warning("[UPDATER] GitPython not installed, auto-update disabled (pip install gitpython)")
        else:
            updater = UpdateManager(str(ROOT_DIR))

            async def _auto_update_loop():
                while True:
                    await asyncio.sleep(21600)  # 6 hours
                    has_update, log = updater.check_update_available()
                    if has_update:
                        logger.info(f"[UPDATER] Update available:\n" + "\n".join(log))
                        if updater.perform_update():
                            logger.info("[UPDATER] Update applied, restarting...")
                            updater.restart_node()

            asyncio.create_task(_auto_update_loop())
    
    # [DHT] Инициализируем Kademlia DHT
    kademlia = None
    try:
        from core.dht.node import KademliaNode
        kademlia = KademliaNode(node, storage_path=f"dht_{args.port}.db")
        await kademlia.start()
        if bootstrap_nodes:
            logger.info(f"[DHT] Bootstrapping Kademlia with {bootstrap_nodes}")
            await kademlia.bootstrap(bootstrap_nodes)
        logger.info(f"[DHT] Kademlia started: {kademlia.local_id.hex()[:16]}...")
    except ImportError as e:
        logger.warning(f"[DHT] Not available: {e}")
    except Exception as e:
        logger.error(f"[DHT] Failed to start: {e}")
    
    # VPN helpers (Pathfinder / Amplifier)
    if kademlia:
        pathfinder = VpnPathfinder(node=node, kademlia=kademlia, ledger=ledger)
    else:
        pathfinder = None
    amplifier = Amplifier(node=node, kademlia=kademlia, ledger=ledger)
    node.amplifier = amplifier
    if cache_agent:
        cache_agent.attach_amplifier(amplifier)
        agent_manager.register_agent(cache_agent)

    # VPN Exit mode setup
    vpn_exit_enabled = args.exit_node or getattr(config, "vpn_mode", "off") == "exit"
    if vpn_exit_enabled:
        exit_agent = agent_manager.get_agent("vpn_exit")
        if exit_agent:
            try:
                exit_agent._price_per_mb = getattr(config, "vpn_exit_price", 0.1)
                exit_agent.metadata["price_per_mb"] = getattr(config, "vpn_exit_price", 0.1)
                exit_agent.metadata["country"] = getattr(config, "vpn_exit_country", "UN")
            except Exception:
                pass

    async def _publish_exit_loop() -> None:
        if not (vpn_exit_enabled and pathfinder):
            return
        # Initial publish
        try:
            await pathfinder.publish_exit(
                ip=args.host,
                geo=getattr(config, "vpn_exit_country", "UN"),
                price=getattr(config, "vpn_exit_price", 0.1),
            )
        except Exception as e:
            logger.warning(f"[VPN] Initial exit publish failed: {e}")
        while True:
            await asyncio.sleep(600)
            try:
                await pathfinder.publish_exit(
                    ip=args.host,
                    geo=getattr(config, "vpn_exit_country", "UN"),
                    price=getattr(config, "vpn_exit_price", 0.1),
                )
            except Exception as e:
                logger.debug(f"[VPN] Exit publish tick failed: {e}")

    if vpn_exit_enabled and pathfinder:
        vpn_tasks.append(asyncio.create_task(_publish_exit_loop()))

    # VPN client controls
    vpn_client_state: Dict[str, Any] = {"exit_id": None, "latency": None}

    async def stop_vpn_client() -> Dict[str, Any]:
        nonlocal socks_server
        if socks_server:
            await socks_server.stop()
            socks_server = None
        vpn_client_state["exit_id"] = None
        vpn_client_state["latency"] = None
        return {"ok": True, "message": "VPN client stopped"}

    async def start_vpn_client(
        strategy: str = "fastest",
        region: str = "any",
        enable_accel: bool = True,
    ) -> Dict[str, Any]:
        nonlocal socks_server
        await stop_vpn_client()

        if not pathfinder:
            return {"ok": False, "message": "Pathfinder unavailable"}

        selected = await pathfinder.pick_exit(
            target_country=(region or "any").lower(),
            strategy=strategy or "fastest",
        )
        if not selected or not selected.get("node_id"):
            return {"ok": False, "message": "No exit node found"}

        amp = amplifier if enable_accel else None
        listen_port = getattr(config, "socks_port", 1080)
        try:
            socks = SocksServer(
                node=node,
                exit_peer_id=selected["node_id"],
                listen_port=listen_port,
                amplifier=amp,
            )
            await socks.start()
            socks_server = socks
            vpn_client_state["exit_id"] = selected["node_id"]
            vpn_client_state["latency"] = selected.get("latency")
            return {
                "ok": True,
                "message": "Connected",
                "exit_id": selected["node_id"],
                "latency": selected.get("latency"),
            }
        except Exception as e:
            logger.warning(f"[VPN] Failed to start SOCKS client: {e}")
            return {"ok": False, "message": str(e)}
    
    # [CORTEX] Инициализируем Cortex - Автономную Систему Знаний
    cortex = None
    if CORTEX_AVAILABLE:
        try:
            cortex = create_cortex_service(
                node=node,
                ledger=ledger,
                agent_manager=agent_manager,
                kademlia=kademlia,
                enable_automata=True,  # Включаем автономный цикл
            )
            await cortex.start()
            logger.info("[CORTEX] Autonomous Knowledge System started")
        except Exception as e:
            logger.warning(f"[CORTEX] Failed to start: {e}")
            cortex = None
    else:
        logger.info("[CORTEX] Not available (module not installed)")

    # [MCP] Optional MCP SSE server
    mcp_server = None
    mcp_task: Optional[asyncio.Task] = None
    if args.mcp:
        try:
            from core.mcp_server import ZeoneMCPServer, MCPConfig

            mcp_server = ZeoneMCPServer(
                node=node,
                ledger=ledger,
                agent_manager=agent_manager,
                cortex=cortex,
                log_buffer=ui_log_buffer,
                mcp_config=MCPConfig(
                    host=args.mcp_host,
                    port=args.mcp_port,
                ),
            )
            mcp_task = mcp_server.start_sse()
            logger.info(
                "[MCP] SSE server running on http://%s:%s/mcp/sse",
                args.mcp_host,
                args.mcp_port,
            )
        except Exception as e:
            logger.warning(f"[MCP] Failed to start: {e}")
    
    # Graceful shutdown
    shutdown_event = asyncio.Event()
    
    signal_state = {"sigint_count": 0}

    def signal_handler():
        signal_state["sigint_count"] += 1
        if signal_state["sigint_count"] == 1:
            print("\nнажмите еще раз Ctrl+C чтобы завершить работу")
            return
        logger.info("[MAIN] Received shutdown signal")
        shutdown_event.set()

    def term_handler():
        logger.info("[MAIN] Received shutdown signal")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            if sig == signal.SIGINT:
                loop.add_signal_handler(sig, signal_handler)
            else:
                loop.add_signal_handler(sig, term_handler)
        except NotImplementedError:
            pass
    
    try:
        if args.webui:
            # Запуск Web UI
            try:
                from webui import create_webui
                
                webui = create_webui(
                    node=node,
                    ledger=ledger,
                    agent_manager=agent_manager,
                    kademlia=kademlia,
                    cortex=cortex,
                    idle_worker=idle_worker,
                    start_vpn_client=start_vpn_client,
                    stop_vpn_client=stop_vpn_client,
                    title=f"P2P Node - {crypto.node_id[:16]}...",
                )
                
                logger.info(f"[WEBUI] Starting on http://0.0.0.0:{args.webui_port}")
                logger.info("[WEBUI] Interactive shell disabled when WebUI is active")
                
                # NiceGUI блокирует, поэтому shell недоступен
                await webui.start(host="0.0.0.0", port=args.webui_port)
                
            except ImportError as e:
                logger.error(f"[WEBUI] Failed to import: {e}")
                logger.error("[WEBUI] Install with: pip install nicegui")
                await interactive_shell(
                    node, ledger, agent_manager, kademlia,
                    rate_limiter=rate_limiter,
                    dos_protector=dos_protector,
                    health_checker=health_checker,
                    metrics_collector=metrics_collector,
                    shutdown_event=shutdown_event,
                )
        elif args.no_shell:
            logger.info("[MAIN] Running in daemon mode. Press Ctrl+C to stop.")
            await shutdown_event.wait()
        else:
            await interactive_shell(
                node, ledger, agent_manager, kademlia,
                rate_limiter=rate_limiter,
                dos_protector=dos_protector,
                health_checker=health_checker,
                metrics_collector=metrics_collector,
                shutdown_event=shutdown_event,
            )
    finally:
        # [PRODUCTION] Save state before shutdown
        if state_manager:
            logger.info("[PERSISTENCE] Saving state...")
            current_state = NodeState(
                node_id=crypto.node_id,
                host=args.host,
                port=args.port,
            )
            # Add connected peers to state
            for peer in node.peer_manager.get_active_peers():
                peer_record = PersistentPeerRecord(
                    node_id=peer.node_id,
                    host=peer.host,
                    port=peer.port,
                    last_connected=time.time(),
                )
                current_state.peers[peer.node_id] = peer_record
            await state_manager.save_state(current_state)
            await state_manager.stop_auto_save()
        
        # Stop Cortex
        if cortex:
            await cortex.stop()
            logger.info("[CORTEX] Stopped")
        
        # Stop production modules
        if health_checker:
            await health_checker.stop()
        if rate_limiter:
            await rate_limiter.stop()
        if dos_protector:
            await dos_protector.stop()
        if socks_server:
            await socks_server.stop()
        for t in vpn_tasks:
            t.cancel()
            with suppress(asyncio.CancelledError):
                await t
        
        if kademlia:
            await kademlia.stop()
        if idle_worker:
            await idle_worker.stop()
        if mcp_task:
            mcp_task.cancel()
            with suppress(asyncio.CancelledError):
                await mcp_task
        await node.stop()
        await ledger.close()
        logger.info("[MAIN] Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
