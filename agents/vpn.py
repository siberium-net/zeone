"""
VPN Exit Agent
==============

Реализует шлюзовую логику для децентрализованного VPN:
- Принимает запросы VPN_CONNECT и открывает реальное TCP соединение
- Проксирует потоковые данные VPN_DATA в обе стороны
- Учитывает трафик и списывает стоимость через ledger
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple, TYPE_CHECKING

from agents.manager import BaseAgent
from core.transport import Message, MessageType, StreamingMessage, StreamingBuffer
from config import config

if TYPE_CHECKING:
    from economy.ledger import Ledger
    from core.node import Node, Peer

logger = logging.getLogger(__name__)


@dataclass
class VpnSession:
    """Состояние активного VPN туннеля."""

    session_id: str
    target_host: str
    target_port: int
    client_peer_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    client_peer: Optional["Peer"] = None
    inbound_buffer: StreamingBuffer = field(default_factory=StreamingBuffer)
    outbound_seq: int = 0
    closed: bool = False
    bytes_from_client: int = 0  # -> реальный интернет
    bytes_to_client: int = 0    # <- обратно в P2P


class VpnExitAgent(BaseAgent):
    """
    Агенти-шлюз для выхода в интернет.

    [VPN] Поддерживает подключение и проксирование потоковых данных.
    """

    def __init__(
        self,
        ledger: Optional["Ledger"] = None,
        node: Optional["Node"] = None,
        country: str = "DE",
        bandwidth: str = "100mbps",
        price_per_mb: float = 0.5,
        chunk_size: Optional[int] = None,
    ):
        self.ledger = ledger
        self._node: Optional["Node"] = node
        self._price_per_mb = price_per_mb
        self.metadata = {
            "country": country,
            "bandwidth": bandwidth,
            "price_per_mb": price_per_mb,
        }
        self.chunk_size = chunk_size or config.network.buffer_size
        self.sessions: Dict[str, VpnSession] = {}
        self._lock = asyncio.Lock()
        self._attached = False

        if node:
            self.attach_node(node)

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------
    @property
    def service_name(self) -> str:
        return "vpn_exit"

    @property
    def price_per_unit(self) -> float:
        # price_per_unit трактуем как стоимость за 1 MB трафика
        return self._price_per_mb

    @property
    def description(self) -> str:
        return (
            f"VPN exit node ({self.metadata['country']}, "
            f"{self.metadata['bandwidth']}, {self._price_per_mb} per MB)"
        )

    async def execute(self, payload: Any) -> Tuple[Any, float]:
        """
        Обрабатывает ServiceRequest для совместимости с рынком услуг.

        Ожидает payload вида:
        {
            "action": "connect",
            "target_host": "...",
            "target_port": 443,
            "session_id": "...",
            "requester_id": "<peer_id>"
        }
        """
        if not isinstance(payload, dict):
            return {"ok": False, "error": "Invalid payload"}, 0.0

        if payload.get("action", "connect") != "connect":
            return {"ok": False, "error": "Unsupported action"}, 0.0

        session_id = payload.get("session_id") or uuid.uuid4().hex
        target_host = payload.get("target_host")
        target_port = int(payload.get("target_port", 0))
        requester_id = payload.get("requester_id", "")

        ok, error = await self._start_session(
            session_id=session_id,
            target_host=target_host,
            target_port=target_port,
            client_peer_id=requester_id,
        )

        result = {
            "session_id": session_id,
            "ok": ok,
            "error": error,
            "metadata": self.metadata,
        }
        # Стоимость = объем в MB (отложенно учитывается при стриминге)
        return result, 0.1

    # ------------------------------------------------------------------
    # Node integration
    # ------------------------------------------------------------------
    def attach_node(self, node: "Node") -> None:
        """Подключить Node и подписаться на сообщения."""
        if self._attached:
            return
        self._node = node
        node.on_message(self._on_node_message)
        node.on_peer_disconnected(self._on_peer_disconnected)
        self._attached = True
        logger.info("[VPN] VpnExitAgent attached to node callbacks")

    async def _on_node_message(self, message: Message, peer: "Peer") -> None:
        """Получение P2P сообщений VPN_*."""
        if message.type == MessageType.VPN_CONNECT:
            await self._handle_connect_message(message, peer)
        elif message.type == MessageType.VPN_DATA:
            await self._handle_stream_message(message, peer)
        elif message.type == MessageType.VPN_CLOSE:
            await self._handle_close_message(message, peer)

    async def _on_peer_disconnected(self, peer: "Peer") -> None:
        """Чистим сессии, привязанные к пиру."""
        stale = [
            sid for sid, sess in self.sessions.items()
            if sess.client_peer_id == peer.node_id
        ]
        for sid in stale:
            await self._close_session(sid, reason="peer_disconnected")

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------
    async def _handle_connect_message(self, message: Message, peer: "Peer") -> None:
        if self._node and not self._node.crypto.verify_signature(message):
            logger.warning(f"[VPN] Dropping unsigned VPN_CONNECT from {peer.node_id[:8]}...")
            return

        payload = message.payload or {}
        session_id = payload.get("session_id") or uuid.uuid4().hex
        target_host = payload.get("target_host")
        target_port = int(payload.get("target_port", 0))

        await self._start_session(
            session_id=session_id,
            target_host=target_host,
            target_port=target_port,
            client_peer_id=peer.node_id,
            client_peer=peer,
        )

    async def _handle_stream_message(self, message: Message, peer: "Peer") -> None:
        if self._node and not self._node.crypto.verify_signature(message):
            logger.warning(f"[VPN] Dropping unsigned VPN_DATA from {peer.node_id[:8]}...")
            return

        payload = message.payload or {}
        session_id = payload.get("session_id") or payload.get("stream_id")
        if not session_id:
            return

        session = self.sessions.get(session_id)
        if not session or session.closed or session.client_peer_id != peer.node_id:
            return
        if session.client_peer and peer is not session.client_peer:
            return

        stream_msg = StreamingMessage.from_payload(
            {**payload, "stream_id": session_id}
        )

        ready_chunks, finished = session.inbound_buffer.add(stream_msg)

        if ready_chunks:
            session.bytes_from_client += sum(len(c) for c in ready_chunks)
            await self._bill_usage(session.client_peer_id, sum(len(c) for c in ready_chunks))
            try:
                for chunk in ready_chunks:
                    session.writer.write(chunk)
                await session.writer.drain()
            except Exception as e:
                logger.warning(f"[VPN] Write to target failed ({session_id}): {e}")
                await self._close_session(session_id, reason="write_failed")
                return

        if finished:
            await self._close_session(session_id, reason="client_eof")

    async def _handle_close_message(self, message: Message, peer: "Peer") -> None:
        if self._node and not self._node.crypto.verify_signature(message):
            logger.warning(f"[VPN] Dropping unsigned VPN_CLOSE from {peer.node_id[:8]}...")
            return

        payload = message.payload or {}
        session_id = payload.get("session_id")
        if not session_id:
            return
        session = self.sessions.get(session_id)
        if not session or session.client_peer_id != peer.node_id:
            return
        await self._close_session(session_id, reason=payload.get("reason", "remote_close"))

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    async def _start_session(
        self,
        session_id: str,
        target_host: Optional[str],
        target_port: int,
        client_peer_id: str,
        client_peer: Optional["Peer"] = None,
    ) -> Tuple[bool, str]:
        if not self._node:
            return False, "Node not attached"

        if not target_host or target_port <= 0:
            await self._send_connect_result(client_peer_id, session_id, False, "Invalid target")
            return False, "Invalid target"

        async with self._lock:
            if session_id in self.sessions:
                return True, ""

        try:
            connect_coro = asyncio.open_connection(target_host, target_port)
            reader, writer = await asyncio.wait_for(
                connect_coro,
                timeout=config.network.connection_timeout,
            )
        except Exception as e:
            error = str(e)
            await self._send_connect_result(client_peer_id, session_id, False, error)
            logger.warning(f"[VPN] Connect failed {target_host}:{target_port} - {error}")
            return False, error

        session = VpnSession(
            session_id=session_id,
            target_host=target_host,
            target_port=target_port,
            client_peer_id=client_peer_id,
            client_peer=client_peer,
            reader=reader,
            writer=writer,
        )

        async with self._lock:
            self.sessions[session_id] = session

        await self._send_connect_result(client_peer_id, session_id, True, "", client_peer=client_peer)

        asyncio.create_task(self._pipe_remote_to_peer(session))
        logger.info(
            f"[VPN] Session started {session_id[:8]}... to {target_host}:{target_port} for {client_peer_id[:8]}..."
        )
        return True, ""

    async def _pipe_remote_to_peer(self, session: VpnSession) -> None:
        """Читает данные из реального сокета и отправляет в P2P."""
        try:
            while not session.closed:
                chunk = await session.reader.read(self.chunk_size)
                if not chunk:
                    break

                stream_msg = StreamingMessage(
                    stream_id=session.session_id,
                    seq=session.outbound_seq,
                    data=chunk,
                    eof=False,
                )
                session.outbound_seq += 1
                sent = await self._send_stream(session.client_peer_id, stream_msg, session.client_peer)
                if sent:
                    session.bytes_to_client += len(chunk)
                    await self._bill_usage(session.client_peer_id, len(chunk))
        except Exception as e:
            logger.warning(f"[VPN] Pipe error ({session.session_id}): {e}")
        finally:
            await self._send_close_message(session.client_peer_id, session.session_id, "remote_closed", session.client_peer)
            await self._close_session(session.session_id, reason="remote_closed")

    async def _send_stream(self, peer_id: str, stream: StreamingMessage, peer: Optional["Peer"] = None) -> bool:
        if not self._node:
            return False
        message = Message(
            type=MessageType.VPN_DATA,
            payload={**stream.to_payload(), "session_id": stream.stream_id},
            sender_id=self._node.node_id,
        )
        if peer and peer.is_connected:
            signed = self._node.crypto.sign_message(message)
            return await peer.send(signed, self._node.use_masking)
        return await self._node.send_to(peer_id, message, with_accounting=False)

    async def _send_connect_result(
        self,
        peer_id: str,
        session_id: str,
        ok: bool,
        error: str,
        client_peer: Optional["Peer"] = None,
    ) -> None:
        if not self._node:
            return
        message = Message(
            type=MessageType.VPN_CONNECT_RESULT,
            payload={
                "session_id": session_id,
                "ok": ok,
                "error": error,
                "metadata": self.metadata,
            },
            sender_id=self._node.node_id,
        )
        if client_peer and client_peer.is_connected:
            signed = self._node.crypto.sign_message(message)
            await client_peer.send(signed, self._node.use_masking)
            return
        await self._node.send_to(peer_id, message, with_accounting=False)

    async def _send_close_message(
        self,
        peer_id: str,
        session_id: str,
        reason: str,
        client_peer: Optional["Peer"] = None,
    ) -> None:
        if not self._node:
            return
        message = Message(
            type=MessageType.VPN_CLOSE,
            payload={"session_id": session_id, "reason": reason},
            sender_id=self._node.node_id,
        )
        if client_peer and client_peer.is_connected:
            signed = self._node.crypto.sign_message(message)
            await client_peer.send(signed, self._node.use_masking)
            return
        await self._node.send_to(peer_id, message, with_accounting=False)

    async def _close_session(self, session_id: str, reason: str = "") -> None:
        """Закрыть и удалить сессию."""
        async with self._lock:
            session = self.sessions.pop(session_id, None)
        if not session or session.closed:
            return

        session.closed = True
        try:
            session.writer.close()
            await session.writer.wait_closed()
        except Exception:
            pass

        logger.info(f"[VPN] Session closed {session_id[:8]}... ({reason})")

    async def _bill_usage(self, peer_id: str, num_bytes: int) -> None:
        """Списать стоимость трафика с клиента."""
        if not self.ledger or num_bytes <= 0:
            return
        tokens = (num_bytes / (1024 * 1024)) * self._price_per_mb
        if tokens <= 0:
            tokens = 0.0001  # минимальный шаг
        try:
            await self.ledger.record_claim(peer_id, tokens)
        except Exception as e:
            logger.warning(f"[VPN] Billing failed for {peer_id[:8]}...: {e}")
