#!/usr/bin/env python3
"""
P2P Network Node - Точка входа
==============================

[DECENTRALIZATION] Этот скрипт запускает полностью децентрализованный узел:
- Генерирует или загружает криптографическую идентичность
- Запускает TCP сервер для входящих соединений
- Подключается к bootstrap-узлам
- Участвует в discovery для расширения сети

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
from economy.ledger import Ledger
from agents.manager import AgentManager


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("p2p")


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
    - help: показать справку
    - quit: выйти
    """
    print("\n" + "=" * 60)
    print("P2P Node Interactive Shell")
    print(f"Node ID: {node.node_id[:32]}...")
    print(f"Listening on: {node.host}:{node.port}")
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
  peers           - List connected peers
  known           - List known (not connected) peers  
  ping <node_id>  - Send PING to peer (use first 8 chars of ID)
  broadcast <msg> - Broadcast message to all peers
  stats           - Show node statistics
  trust <node_id> - Show Trust Score for peer
  ledger          - Show ledger statistics
  id              - Show this node's ID
  help            - Show this help
  quit            - Exit the node
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
                        direction = "outbound" if peer.is_outbound else "inbound"
                        print(f"  - {peer.node_id[:16]}... @ {peer.host}:{peer.port} ({direction})")
            
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
                if await target_peer.send(ping):
                    print(f"[OK] PING sent to {target_peer.node_id[:16]}...")
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
                count = await node.broadcast(message)
                print(f"[OK] Broadcast sent to {count} peers")
            
            elif cmd == "stats":
                peers = node.peer_manager.get_active_peers()
                known = node.peer_manager._known_peers
                print(f"""
Node Statistics:
  Connected peers: {len(peers)}
  Known peers: {len(known)}
  Inbound connections: {sum(1 for p in peers if not p.is_outbound)}
  Outbound connections: {sum(1 for p in peers if p.is_outbound)}
""")
            
            elif cmd == "trust":
                if not args:
                    print("[ERROR] Usage: trust <node_id_prefix>")
                    continue
                
                # Ищем пира по префиксу
                for peer in node.peer_manager.get_active_peers():
                    if peer.node_id.startswith(args) or peer.node_id[:16].startswith(args):
                        score = await ledger.get_trust_score(peer.node_id)
                        print(f"Trust Score for {peer.node_id[:16]}...: {score:.4f}")
                        break
                else:
                    print(f"[ERROR] Peer not found: {args}")
            
            elif cmd == "ledger":
                stats = await ledger.get_stats()
                print(f"""
Ledger Statistics:
  Known peers: {stats['peer_count']}
  Transactions: {stats['transaction_count']}
  Active IOUs: {stats['active_ious']}
  Outstanding debt: {stats['total_outstanding_debt']:.2f}
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
    """
    parser = argparse.ArgumentParser(
        description="P2P Network Node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start as bootstrap node
  python main.py --port 8468
  
  # Connect to existing network
  python main.py --port 8469 --bootstrap 127.0.0.1:8468
  
  # Use custom identity file
  python main.py --identity my_node.key
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
    
    # Инициализируем модули
    ledger = Ledger(args.db)
    await ledger.initialize()
    logger.info(f"[LEDGER] Initialized: {args.db}")
    
    agent_manager = AgentManager()
    logger.info(f"[AGENTS] Initialized: {agent_manager.contracts_dir}")
    
    # Создаем узел
    node = Node(
        crypto=crypto,
        host=args.host,
        port=args.port,
        use_masking=args.masking,
    )
    
    # Регистрируем callback для обновления Trust Score
    async def on_peer_connected(peer):
        await ledger.get_or_create_peer(peer.node_id)
        await ledger.update_trust_score(peer.node_id, "ping_responded")
    
    async def on_peer_disconnected(peer):
        await ledger.update_trust_score(peer.node_id, "ping_timeout")
    
    async def on_message(message, peer):
        if crypto.verify_signature(message):
            await ledger.update_trust_score(peer.node_id, "valid_message")
        else:
            await ledger.update_trust_score(peer.node_id, "invalid_message")
    
    node.on_peer_connected(on_peer_connected)
    node.on_peer_disconnected(on_peer_disconnected)
    node.on_message(on_message)
    
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

