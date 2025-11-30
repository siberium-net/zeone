#!/usr/bin/env python3
"""
P2P Network Node - Точка входа (Layer 2: Economy)
=================================================

[DECENTRALIZATION] Этот скрипт запускает полностью децентрализованный узел:
- Генерирует или загружает криптографическую идентичность
- Запускает TCP сервер для входящих соединений
- Подключается к bootstrap-узлам
- Участвует в discovery для расширения сети

[ECONOMY] Layer 2 - Экономика и репутация:
- Автоматический учет трафика (debt/claim)
- Блокировка leechers при превышении лимита долга
- Обмен балансом при handshake

Использование:
    python main.py [--port PORT] [--bootstrap HOST:PORT] [--identity FILE]

Примеры:
    # Запуск первого узла (bootstrap)
    python main.py --port 8468
    
    # Подключение к существующему узлу
    python main.py --port 8469 --bootstrap 127.0.0.1:8468
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from config import config
from core.node import Node
from core.transport import Crypto, Message, MessageType
from economy.ledger import Ledger, DEFAULT_DEBT_LIMIT_BYTES
from agents.manager import AgentManager


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
    
    [SECURITY] Приватный ключ - это идентичность узла.
    Потеря ключа = потеря идентичности.
    Компрометация ключа = возможность выдавать себя за узел.
    
    [DECENTRALIZATION] Каждый узел генерирует свой ключ локально.
    Нет центра регистрации или сертификации.
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


async def interactive_shell(node: Node, ledger: Ledger, agent_manager: AgentManager) -> None:
    """
    Интерактивная оболочка для взаимодействия с узлом.
    
    Команды:
    - peers: показать подключенных пиров
    - ping <node_id>: отправить PING пиру
    - broadcast <message>: отправить сообщение всем
    - stats: показать статистику
    - trust <node_id>: показать Trust Score
    - balance [node_id]: показать баланс
    - balances: показать все балансы
    - help: показать справку
    - quit: выйти
    """
    print("\n" + "=" * 60)
    print("P2P Node Interactive Shell (Layer 2: Economy)")
    print(f"Node ID: {node.node_id[:32]}...")
    print(f"Listening on: {node.host}:{node.port}")
    print(f"Debt limit: {format_bytes(ledger.debt_limit)}")
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # Используем asyncio для неблокирующего ввода
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
  peers             - List connected peers with traffic stats
  known             - List known (not connected) peers  
  ping <node_id>    - Send PING to peer (use first 8 chars of ID)
  broadcast <msg>   - Broadcast message to all peers
  stats             - Show node statistics
  trust <node_id>   - Show Trust Score for peer
  balance [node_id] - Show balance with peer (or all if no arg)
  balances          - Show all balances
  ledger            - Show ledger statistics
  id                - Show this node's ID
  help              - Show this help
  quit              - Exit the node
  
[ECONOMY] Balance interpretation:
  Positive (+) = peer owes us (we sent more data)
  Negative (-) = we owe peer (we received more data)
  Blocked = peer exceeded debt limit, no more data sent to them
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
                        balance_str = f"{balance:+.0f}" if balance != 0 else "0"
                        print(
                            f"  - {peer.node_id[:12]}... @ {peer.host}:{peer.port} "
                            f"[{direction}] sent:{format_bytes(peer.bytes_sent)} "
                            f"recv:{format_bytes(peer.bytes_received)} "
                            f"bal:{balance_str} {blocked}"
                        )
            
            elif cmd == "known":
                known = node.peer_manager._known_peers
                if not known:
                    print("[INFO] No known peers")
                else:
                    print(f"[INFO] Known peers ({len(known)}):")
                    for info in known.values():
                        print(f"  - {info.node_id[:16]}... @ {info.host}:{info.port} (trust: {info.trust_score:.2f})")
            
            elif cmd == "ping":
                if not args:
                    print("[ERROR] Usage: ping <node_id_prefix>")
                    continue
                
                # Ищем пира по префиксу ID
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
                
                # Используем send_with_accounting для учета
                success, sent, reason = await target_peer.send_with_accounting(
                    ping, ledger, node.use_masking
                )
                
                if success:
                    print(f"[OK] PING sent to {target_peer.node_id[:16]}... ({sent} bytes)")
                else:
                    print(f"[ERROR] Failed to send PING: {reason}")
            
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
                print(f"[OK] Broadcast sent to {count} peers (with accounting)")
            
            elif cmd == "stats":
                peers = node.peer_manager.get_active_peers()
                known = node.peer_manager._known_peers
                
                total_sent = sum(p.bytes_sent for p in peers)
                total_recv = sum(p.bytes_received for p in peers)
                blocked_count = sum(1 for p in peers if p.blocked)
                
                ledger_stats = await ledger.get_stats()
                
                print(f"""
Node Statistics:
  Connected peers: {len(peers)} ({blocked_count} blocked)
  Known peers: {len(known)}
  Inbound connections: {sum(1 for p in peers if not p.is_outbound)}
  Outbound connections: {sum(1 for p in peers if p.is_outbound)}

Traffic (this session):
  Total sent: {format_bytes(total_sent)}
  Total received: {format_bytes(total_recv)}
  
Economy:
  Debt limit: {format_bytes(ledger.debt_limit)}
  Total owed to us: {format_bytes(ledger_stats['total_owed_to_us'])}
  Total we owe: {format_bytes(ledger_stats['total_we_owe'])}
  Peers with balance: {ledger_stats['peers_with_balance']}
""")
            
            elif cmd == "trust":
                if not args:
                    print("[ERROR] Usage: trust <node_id_prefix>")
                    continue
                
                # Ищем пира по префиксу
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
                    # Баланс с конкретным пиром
                    found = False
                    for peer in node.peer_manager.get_active_peers():
                        if peer.node_id.startswith(args) or peer.node_id[:16].startswith(args):
                            info = await ledger.get_balance_info(peer.node_id)
                            print(f"""
Balance with {peer.node_id[:16]}...:
  Current balance: {info['balance']:+.0f} bytes ({format_bytes(abs(info['balance']))})
  Total sent: {format_bytes(info['total_sent'])}
  Total received: {format_bytes(info['total_received'])}
  Status: {"BLOCKED" if peer.blocked else "OK"}
  Interpretation: {"They owe us" if info['balance'] > 0 else "We owe them" if info['balance'] < 0 else "Even"}
""")
                            found = True
                            break
                    
                    if not found:
                        print(f"[ERROR] Peer not found: {args}")
                else:
                    # Показать все балансы
                    balances = await ledger.get_all_balances()
                    if not balances:
                        print("[INFO] No balance records")
                    else:
                        print(f"[INFO] All balances ({len(balances)}):")
                        for b in balances[:20]:  # Показываем топ 20
                            sign = "+" if b['balance'] > 0 else ""
                            status = "BLOCKED" if ledger.is_peer_blocked(b['peer_id']) else ""
                            print(
                                f"  {b['peer_id'][:12]}... "
                                f"bal:{sign}{b['balance']:.0f} "
                                f"sent:{format_bytes(b['total_sent'])} "
                                f"recv:{format_bytes(b['total_received'])} "
                                f"{status}"
                            )
            
            elif cmd == "balances":
                balances = await ledger.get_all_balances()
                if not balances:
                    print("[INFO] No balance records")
                else:
                    total_positive = sum(b['balance'] for b in balances if b['balance'] > 0)
                    total_negative = sum(b['balance'] for b in balances if b['balance'] < 0)
                    
                    print(f"""
[INFO] Balance Summary:
  Peers with balance: {len(balances)}
  Total owed to us: {format_bytes(total_positive)}
  Total we owe: {format_bytes(abs(total_negative))}
  Net position: {format_bytes(total_positive + total_negative)}
  
Top debtors (owe us):""")
                    for b in sorted(balances, key=lambda x: x['balance'], reverse=True)[:5]:
                        if b['balance'] > 0:
                            print(f"    {b['peer_id'][:12]}... owes {format_bytes(b['balance'])}")
                    
                    print("\nTop creditors (we owe):")
                    for b in sorted(balances, key=lambda x: x['balance'])[:5]:
                        if b['balance'] < 0:
                            print(f"    {b['peer_id'][:12]}... owed {format_bytes(abs(b['balance']))}")
            
            elif cmd == "ledger":
                stats = await ledger.get_stats()
                print(f"""
Ledger Statistics:
  Known peers: {stats['peer_count']}
  Transactions: {stats['transaction_count']}
  Active IOUs: {stats['active_ious']}
  Outstanding IOU debt: {format_bytes(stats['total_outstanding_debt'])}
  
Balance Statistics:
  Total owed to us: {format_bytes(stats['total_owed_to_us'])}
  Total we owe: {format_bytes(stats['total_we_owe'])}
  Peers with balance: {stats['peers_with_balance']}
  Debt limit: {format_bytes(stats['debt_limit'])}
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
    
    [DECENTRALIZATION] Запускает полностью автономный узел:
    1. Загружает/создает идентичность
    2. Инициализирует все модули
    3. Запускает сервер
    4. Подключается к сети
    5. Обрабатывает события
    
    [ECONOMY] Layer 2:
    - Инициализирует Ledger для учета трафика
    - Передает Ledger в Node для автоматического учета
    - Регистрирует callbacks для обновления Trust Score
    """
    parser = argparse.ArgumentParser(
        description="P2P Network Node (Layer 2: Economy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start as bootstrap node
  python main.py --port 8468
  
  # Connect to existing network
  python main.py --port 8469 --bootstrap 127.0.0.1:8468
  
  # Use custom identity file
  python main.py --identity my_node.key
  
  # Set custom debt limit (50MB)
  python main.py --debt-limit 52428800
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
        help=f"Debt limit in bytes before blocking peer (default: {DEFAULT_DEBT_LIMIT_BYTES} = 100MB)",
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
    
    # Инициализируем Ledger с лимитом долга
    ledger = Ledger(args.db, debt_limit=args.debt_limit)
    await ledger.initialize()
    logger.info(f"[LEDGER] Initialized: {args.db}, debt_limit={format_bytes(args.debt_limit)}")
    
    agent_manager = AgentManager()
    logger.info(f"[AGENTS] Initialized: {agent_manager.contracts_dir}")
    
    # Создаем узел с интегрированным ledger
    node = Node(
        crypto=crypto,
        host=args.host,
        port=args.port,
        use_masking=args.masking,
        ledger=ledger,  # [ECONOMY] Передаем ledger для автоматического учета
    )
    
    # Регистрируем callback для обновления Trust Score
    async def on_peer_connected(peer):
        await ledger.get_or_create_peer(peer.node_id)
        await ledger.update_trust_score(peer.node_id, "ping_responded")
        logger.info(f"[ECONOMY] Peer connected: {peer.node_id[:12]}...")
    
    async def on_peer_disconnected(peer):
        await ledger.update_trust_score(peer.node_id, "ping_timeout")
        balance = await ledger.get_balance(peer.node_id)
        logger.info(
            f"[ECONOMY] Peer disconnected: {peer.node_id[:12]}..., "
            f"final_balance={balance:+.0f}"
        )
    
    async def on_message(message, peer):
        if crypto.verify_signature(message):
            await ledger.update_trust_score(peer.node_id, "valid_message")
        else:
            await ledger.update_trust_score(peer.node_id, "invalid_message")
    
    async def on_balance_received(peer_id: str, their_balance: float):
        our_balance = await ledger.get_balance(peer_id)
        logger.debug(
            f"[ECONOMY] Balance exchange with {peer_id[:12]}...: "
            f"ours={our_balance:+.0f}, theirs={their_balance:+.0f}"
        )
    
    node.on_peer_connected(on_peer_connected)
    node.on_peer_disconnected(on_peer_disconnected)
    node.on_message(on_message)
    node.on_balance_received(on_balance_received)
    
    # Запускаем узел
    await node.start()
    
    # Настраиваем graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        logger.info("[MAIN] Received shutdown signal")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows не поддерживает add_signal_handler
            pass
    
    try:
        if args.no_shell:
            # Работаем как демон
            logger.info("[MAIN] Running in daemon mode. Press Ctrl+C to stop.")
            await shutdown_event.wait()
        else:
            # Запускаем интерактивную оболочку
            await interactive_shell(node, ledger, agent_manager)
    finally:
        # Останавливаем узел
        await node.stop()
        await ledger.close()
        logger.info("[MAIN] Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
