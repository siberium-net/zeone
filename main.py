#!/usr/bin/env python3
"""
P2P Network Node - Точка входа (Layer 3: Market)
================================================

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

Использование:
    python main.py [--port PORT] [--bootstrap HOST:PORT] [--identity FILE]

Примеры:
    # Запуск первого узла (bootstrap)
    python main.py --port 8468
    
    # Подключение к существующему узлу
    python main.py --port 8469 --bootstrap 127.0.0.1:8468
    
    # Тест Echo сервиса
    >>> echo Hello World
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не установлен

from config import config
from core.node import Node
from core.transport import Crypto, Message, MessageType
from economy.ledger import Ledger, DEFAULT_DEBT_LIMIT_BYTES
from agents.manager import AgentManager, ServiceRequest, ServiceResponse


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("p2p")


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
) -> None:
    """
    Интерактивная оболочка для взаимодействия с узлом.
    """
    print("\n" + "=" * 60)
    print("P2P Node Interactive Shell (Layer 3: Market + DHT)")
    print(f"Node ID: {node.node_id[:32]}...")
    print(f"Listening on: {node.host}:{node.port}")
    print(f"Services: {', '.join(agent_manager._agents.keys())}")
    if kademlia:
        print(f"DHT: Active (ID: {kademlia.local_id.hex()[:16]}...)")
    else:
        print("DHT: Not available")
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 60 + "\n")
    
    while True:
        try:
            loop = asyncio.get_event_loop()
            line = await loop.run_in_executor(None, lambda: input(">>> ").strip())
            
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

[DHT] Distributed Hash Table:
  dht put <key> <value>  - Store data in DHT
  dht get <key>          - Retrieve data from DHT
  dht del <key>          - Delete data from local DHT storage
  dht info               - Show DHT statistics
  routing                - Show routing table info
  
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
            
            else:
                print(f"[ERROR] Unknown command: {cmd}. Type 'help' for commands.")
                
        except EOFError:
            print("\n[INFO] EOF received, shutting down...")
            break
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted, shutting down...")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


async def main() -> None:
    """
    Главная функция - точка входа.
    """
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Парсим bootstrap узлы
    bootstrap_nodes = parse_bootstrap(args.bootstrap)
    if bootstrap_nodes:
        config.network.bootstrap_nodes = bootstrap_nodes
    
    # Загружаем идентичность
    crypto = load_or_create_identity(args.identity)
    
    # Инициализируем Ledger
    ledger = Ledger(args.db, debt_limit=args.debt_limit)
    await ledger.initialize()
    logger.info(f"[LEDGER] Initialized: {args.db}")
    
    # [MARKET] Инициализируем AgentManager с Ledger и node_id
    agent_manager = AgentManager(ledger=ledger, node_id=crypto.node_id)
    logger.info(f"[AGENTS] Initialized with services: {list(agent_manager._agents.keys())}")
    
    # Создаем узел с интегрированным ledger и agent_manager
    node = Node(
        crypto=crypto,
        host=args.host,
        port=args.port,
        use_masking=args.masking,
        ledger=ledger,
        agent_manager=agent_manager,
    )
    
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
    await node.start()
    
    # [DHT] Инициализируем Kademlia DHT
    kademlia = None
    try:
        from core.dht.node import KademliaNode
        kademlia = KademliaNode(node, storage_path=f"dht_{args.port}.db")
        await kademlia.start()
        logger.info(f"[DHT] Kademlia started: {kademlia.local_id.hex()[:16]}...")
    except ImportError as e:
        logger.warning(f"[DHT] Not available: {e}")
    except Exception as e:
        logger.error(f"[DHT] Failed to start: {e}")
    
    # Graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        logger.info("[MAIN] Received shutdown signal")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    try:
        if args.no_shell:
            logger.info("[MAIN] Running in daemon mode. Press Ctrl+C to stop.")
            await shutdown_event.wait()
        else:
            await interactive_shell(node, ledger, agent_manager, kademlia)
    finally:
        if kademlia:
            await kademlia.stop()
        await node.stop()
        await ledger.close()
        logger.info("[MAIN] Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
