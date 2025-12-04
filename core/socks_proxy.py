"""
Локальный SOCKS5 сервер, пробрасывающий трафик через P2P VPN.
"""

import asyncio
import ipaddress
import logging
import socket
import struct
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

from core.transport import Message, MessageType, StreamingMessage, StreamingBuffer

if TYPE_CHECKING:
    from core.node import Node, Peer

logger = logging.getLogger(__name__)


# SOCKS5 constants
SOCKS_VERSION = 5
CMD_CONNECT = 1
ATYP_IPV4 = 1
ATYP_DOMAIN = 3
ATYP_IPV6 = 4


@dataclass
class SocksSession:
    """Локальное состояние SOCKS подключения."""

    session_id: str
    target_host: str
    target_port: int
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    outbound_seq: int = 0
    inbound_buffer: StreamingBuffer = field(default_factory=StreamingBuffer)
    closed: bool = False
    close_event: asyncio.Event = field(default_factory=asyncio.Event)


class SocksServer:
    """
    Простая реализация SOCKS5 (CONNECT) поверх P2P VPN.
    """

    def __init__(
        self,
        node: "Node",
        exit_peer_id: str,
        listen_host: str = "127.0.0.1",
        listen_port: int = 1080,
        chunk_size: int = 16_384,
        connect_timeout: float = 10.0,
    ):
        self.node = node
        self.exit_peer_id = exit_peer_id
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.chunk_size = chunk_size
        self.connect_timeout = connect_timeout
        self._server: Optional[asyncio.AbstractServer] = None
        self._sessions: Dict[str, SocksSession] = {}
        self._pending_connect: Dict[str, asyncio.Future] = {}

        self.node.on_message(self._on_p2p_message)

    async def start(self) -> None:
        """Запуск локального SOCKS сервера."""
        if self._server:
            return
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.listen_host,
            port=self.listen_port,
        )
        logger.info(f"[SOCKS] Listening on {self.listen_host}:{self.listen_port} (exit={self.exit_peer_id[:12]}...)")

    async def stop(self) -> None:
        """Остановить сервер и закрыть все сессии."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for session_id in list(self._sessions.keys()):
            await self._cleanup_session(session_id, reason="server_stop")
        logger.info("[SOCKS] Stopped")

    # ------------------------------------------------------------------
    # Client handling
    # ------------------------------------------------------------------
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            if not await self._handshake(reader, writer):
                return

            request = await reader.readexactly(4)
            ver, cmd, _, atyp = request
            if ver != SOCKS_VERSION or cmd != CMD_CONNECT:
                await self._send_reply(writer, rep=0x07)  # Command not supported
                return

            target_host = await self._read_address(reader, atyp)
            target_port_bytes = await reader.readexactly(2)
            target_port = struct.unpack("!H", target_port_bytes)[0]

            session_id = uuid.uuid4().hex
            ok, error = await self._initiate_tunnel(session_id, target_host, target_port)
            if not ok:
                logger.warning(f"[SOCKS] VPN connect failed: {error}")
                await self._send_reply(writer, rep=0x05)  # Connection refused
                return

            await self._send_reply(writer, rep=0x00)

            session = SocksSession(
                session_id=session_id,
                target_host=target_host,
                target_port=target_port,
                reader=reader,
                writer=writer,
            )
            self._sessions[session_id] = session

            logger.info(f"[SOCKS] Tunnel ready {target_host}:{target_port} (session {session_id[:8]}...)")

            await self._pipe_client(session)
        except asyncio.IncompleteReadError:
            return
        except Exception as e:
            logger.warning(f"[SOCKS] Client handling error: {e}")
        finally:
            await self._cleanup_session_by_writer(writer)

    async def _handshake(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> bool:
        """SOCKS5 handshake (без аутентификации)."""
        data = await reader.readexactly(2)
        ver, nmethods = data[0], data[1]
        methods = await reader.readexactly(nmethods)

        if ver != SOCKS_VERSION or 0 not in methods:
            writer.write(bytes([SOCKS_VERSION, 0xFF]))
            await writer.drain()
            return False

        writer.write(bytes([SOCKS_VERSION, 0x00]))  # no auth
        await writer.drain()
        return True

    async def _read_address(self, reader: asyncio.StreamReader, atyp: int) -> str:
        """Считать адрес в зависимости от ATYP."""
        if atyp == ATYP_IPV4:
            addr = await reader.readexactly(4)
            return socket.inet_ntoa(addr)
        if atyp == ATYP_DOMAIN:
            length = await reader.readexactly(1)
            domain = await reader.readexactly(length[0])
            return domain.decode("utf-8")
        if atyp == ATYP_IPV6:
            addr = await reader.readexactly(16)
            return str(ipaddress.IPv6Address(addr))
        raise ValueError(f"Unsupported ATYP {atyp}")

    async def _pipe_client(self, session: SocksSession) -> None:
        """Чтение из локального клиента и отправка в P2P."""
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

                message = Message(
                    type=MessageType.VPN_DATA,
                    payload={**stream_msg.to_payload(), "session_id": session.session_id},
                    sender_id=self.node.node_id,
                )
                success = await self.node.send_to(
                    self.exit_peer_id,
                    message,
                    with_accounting=False,
                )
                if not success:
                    logger.warning("[SOCKS] Failed to push data to exit node")
                    break
        except Exception as e:
            logger.warning(f"[SOCKS] Client pipe error: {e}")
        finally:
            await self._send_close(session.session_id, reason="client_closed")
            await self._cleanup_session(session.session_id, reason="client_closed")

    # ------------------------------------------------------------------
    # P2P side
    # ------------------------------------------------------------------
    async def _on_p2p_message(self, message: Message, peer: "Peer") -> None:
        if peer.node_id != self.exit_peer_id:
            return

        if message.type == MessageType.VPN_CONNECT_RESULT:
            payload = message.payload or {}
            session_id = payload.get("session_id")
            future = self._pending_connect.get(session_id)
            if future and not future.done():
                future.set_result(payload)
            return

        if message.type == MessageType.VPN_DATA:
            await self._handle_data(message)
        elif message.type == MessageType.VPN_CLOSE:
            await self._handle_close(message)

    async def _handle_data(self, message: Message) -> None:
        payload = message.payload or {}
        session_id = payload.get("session_id") or payload.get("stream_id")
        if not session_id:
            return
        session = self._sessions.get(session_id)
        if not session or session.closed:
            return

        stream_msg = StreamingMessage.from_payload({**payload, "stream_id": session_id})
        ready_chunks, finished = session.inbound_buffer.add(stream_msg)

        if ready_chunks:
            try:
                for chunk in ready_chunks:
                    session.writer.write(chunk)
                await session.writer.drain()
            except Exception as e:
                logger.warning(f"[SOCKS] Write to client failed: {e}")
                await self._cleanup_session(session_id, reason="write_failed")
                return

        if finished:
            await self._cleanup_session(session_id, reason="remote_eof")

    async def _handle_close(self, message: Message) -> None:
        payload = message.payload or {}
        session_id = payload.get("session_id")
        if not session_id:
            return
        await self._cleanup_session(session_id, reason=payload.get("reason", "remote_close"))

    async def _initiate_tunnel(self, session_id: str, host: str, port: int) -> tuple:
        """Отправить запрос VPN_CONNECT и ждать результата."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_connect[session_id] = future

        message = Message(
            type=MessageType.VPN_CONNECT,
            payload={
                "session_id": session_id,
                "target_host": host,
                "target_port": port,
            },
            sender_id=self.node.node_id,
        )

        sent = await self.node.send_to(self.exit_peer_id, message, with_accounting=False)
        if not sent:
            self._pending_connect.pop(session_id, None)
            return False, "send_failed"

        try:
            result = await asyncio.wait_for(future, timeout=self.connect_timeout)
            ok = bool(result.get("ok"))
            error = result.get("error", "")
            return ok, error
        except asyncio.TimeoutError:
            return False, "connect_timeout"
        finally:
            self._pending_connect.pop(session_id, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _send_reply(self, writer: asyncio.StreamWriter, rep: int) -> None:
        """Отправить SOCKS5 ответ клиенту."""
        reply = bytes([
            SOCKS_VERSION,
            rep,
            0x00,  # RSV
            ATYP_IPV4,
        ]) + b"\x00\x00\x00\x00" + struct.pack("!H", 0)
        writer.write(reply)
        await writer.drain()

    async def _send_close(self, session_id: str, reason: str) -> None:
        message = Message(
            type=MessageType.VPN_CLOSE,
            payload={"session_id": session_id, "reason": reason},
            sender_id=self.node.node_id,
        )
        await self.node.send_to(self.exit_peer_id, message, with_accounting=False)

    async def _cleanup_session(self, session_id: str, reason: str = "") -> None:
        session = self._sessions.pop(session_id, None)
        if not session or session.closed:
            return
        session.closed = True
        session.close_event.set()
        try:
            session.writer.close()
            await session.writer.wait_closed()
        except Exception:
            pass
        logger.info(f"[SOCKS] Session closed {session_id[:8]}... ({reason})")

    async def _cleanup_session_by_writer(self, writer: asyncio.StreamWriter) -> None:
        for session_id, session in list(self._sessions.items()):
            if session.writer is writer:
                await self._cleanup_session(session_id, reason="client_closed")
