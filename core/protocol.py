"""
Protocol Layer - Обработчики сообщений и Ping-Pong протокол
==========================================================

[SECURITY] Этот модуль реализует:
1. Ping-Pong протокол с верификацией подписей
2. Discovery протокол для поиска пиров
3. Базовые обработчики сообщений
4. [NEW] Интеграция с ReplayProtector (persistent nonce storage)
5. [NEW] Интеграция с WeightedTrustScore (slash on invalid merkle)

[DECENTRALIZATION] Все сообщения верифицируются криптографически.
PONG возвращается ТОЛЬКО если подпись PING валидна.
Это защищает от спуфинга и replay-атак.

[HARD FORK] Wire Protocol V1:
- Бинарный формат заголовка (98 bytes)
- Magic b'ZE' обязателен
- См. core/wire.py для спецификации
"""

import time
import os
import base64
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Awaitable, Any, TYPE_CHECKING
from abc import ABC, abstractmethod

from .transport import Message, MessageType, Crypto

if TYPE_CHECKING:
    from .node import Peer
    from core.security.replay import ReplayProtector
    from economy.trust import WeightedTrustScore

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Информация о пире для Discovery."""
    
    node_id: str
    host: str
    port: int
    trust_score: float = 0.5
    last_seen: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "trust_score": self.trust_score,
            "last_seen": self.last_seen,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeerInfo":
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            trust_score=data.get("trust_score", 0.5),
            last_seen=data.get("last_seen", time.time()),
        )


class MessageHandler(ABC):
    """
    Базовый класс для обработчиков сообщений.
    
    [DECENTRALIZATION] Каждый тип сообщения обрабатывается
    независимо. Узел может выбирать, какие обработчики
    использовать.
    """
    
    @property
    @abstractmethod
    def message_type(self) -> MessageType:
        """Тип сообщения, который обрабатывает этот handler."""
        pass
    
    @abstractmethod
    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        """
        Обработать входящее сообщение.
        
        Args:
            message: Входящее сообщение
            crypto: Криптомодуль узла
            context: Контекст (peer info, etc.)
        
        Returns:
            Ответное сообщение или None
        """
        pass


class PingPongHandler(MessageHandler):
    """
    Обработчик Ping-Pong протокола.
    
    [SECURITY] Протокол проверки связи с верификацией:
    
    1. Узел A отправляет PING:
       - nonce: случайное значение
       - timestamp: время отправки
       - signature: подпись (nonce + timestamp + sender_id)
    
    2. Узел B проверяет:
       - Подпись валидна
       - timestamp не слишком старый (защита от replay)
       - sender_id соответствует подписи
       - [NEW] nonce не использовался ранее (persistent replay protection)
    
    3. Если все проверки пройдены, узел B отправляет PONG:
       - original_nonce: nonce из PING
       - timestamp: текущее время
       - signature: подпись ответа
    
    [DECENTRALIZATION] Этот протокол позволяет узлам:
    - Проверить, что пир владеет заявленным приватным ключом
    - Измерить latency соединения
    - Обнаружить "мертвые" узлы
    
    [PERSISTENCE] Nonces сохраняются в SQLite через ReplayProtector.
    Защита работает даже после перезагрузки узла.
    """
    
    # Максимальный возраст PING сообщения (секунды)
    MAX_PING_AGE = 60.0
    
    @property
    def message_type(self) -> MessageType:
        return MessageType.PING
    
    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        """
        Обработать PING и вернуть PONG если подпись валидна.
        
        [SECURITY] PONG возвращается ТОЛЬКО если:
        1. Подпись PING валидна
        2. Timestamp не слишком старый
        3. [NEW] Nonce не использовался ранее (persistent check)
        
        Это предотвращает:
        - Спуфинг (подделка отправителя)
        - Replay-атаки (повторная отправка старых сообщений)
        """
        # Проверка 1: Валидность подписи
        if not crypto.verify_signature(message):
            # [SECURITY] Отклоняем сообщение с невалидной подписью
            # Не отправляем ответ - это может быть попытка атаки
            logger.warning(f"[PING] Invalid signature from {message.sender_id[:8]}...")
            return None
        
        # Проверка 2: Возраст сообщения
        age = time.time() - message.timestamp
        if age > self.MAX_PING_AGE:
            # [SECURITY] Сообщение слишком старое
            # Возможная replay-атака
            logger.warning(f"[PING] Message too old ({age:.1f}s) from {message.sender_id[:8]}...")
            return None
        
        # Проверка 3: [NEW] Persistent replay protection
        replay_protector: Optional["ReplayProtector"] = context.get("replay_protector")
        if replay_protector and message.nonce:
            nonce_bytes = base64.b64decode(message.nonce)
            if not await replay_protector.is_nonce_fresh(nonce_bytes):
                # [SECURITY] Replay attack detected!
                logger.warning(f"[PING] Replay attack detected from {message.sender_id[:8]}...")
                
                # Update trust score if available
                trust_system: Optional["WeightedTrustScore"] = context.get("trust_system")
                if trust_system:
                    from economy.trust import TrustEvent
                    await trust_system.record_event(
                        message.sender_id,
                        TrustEvent.INVALID_MESSAGE,
                        magnitude=2.0,  # Double penalty for replay
                    )
                
                return None
        
        # Все проверки пройдены - создаем PONG
        pong = Message(
            type=MessageType.PONG,
            payload={
                "original_nonce": message.nonce,
                "ping_timestamp": message.timestamp,
            },
            sender_id=crypto.node_id,
        )
        
        # Update trust score for successful ping
        trust_system: Optional["WeightedTrustScore"] = context.get("trust_system")
        if trust_system:
            from economy.trust import TrustEvent
            await trust_system.record_event(
                message.sender_id,
                TrustEvent.PING_RESPONDED,
            )
        
        # Подписываем ответ
        return crypto.sign_message(pong)
    
    @staticmethod
    def create_ping(crypto: Crypto) -> Message:
        """
        Создать PING сообщение.
        
        [SECURITY] PING содержит:
        - nonce: уникальное случайное значение
        - timestamp: текущее время
        - подпись: доказательство владения ключом
        """
        ping = Message(
            type=MessageType.PING,
            payload={},
            sender_id=crypto.node_id,
            nonce=base64.b64encode(os.urandom(16)).decode("ascii"),
        )
        return crypto.sign_message(ping)
    
    @staticmethod
    def verify_pong(ping: Message, pong: Message, crypto: Crypto) -> bool:
        """
        Проверить что PONG соответствует нашему PING.
        
        [SECURITY] Проверяем:
        1. Подпись PONG валидна
        2. original_nonce совпадает с nonce из PING
        """
        # Проверяем подпись
        if not crypto.verify_signature(pong):
            return False
        
        # Проверяем nonce
        if pong.payload.get("original_nonce") != ping.nonce:
            return False
        
        return True


class PongHandler(MessageHandler):
    """Обработчик PONG сообщений (для логирования и обновления RTT)."""
    
    @property
    def message_type(self) -> MessageType:
        return MessageType.PONG
    
    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        """
        Обработать PONG.
        
        PONG не требует ответа, но мы можем:
        - Обновить RTT для пира
        - Подтвердить что пир жив
        """
        # Проверяем подпись
        if not crypto.verify_signature(message):
            return None
        
        # Обновляем информацию о пире в контексте
        if "peer" in context:
            peer = context["peer"]
            ping_timestamp = message.payload.get("ping_timestamp", 0)
            if ping_timestamp:
                rtt = time.time() - ping_timestamp
                # RTT можно использовать для оценки качества соединения
                context.setdefault("rtt_history", []).append(rtt)
        
        return None


class DiscoverHandler(MessageHandler):
    """
    Обработчик Discovery протокола.
    
    [DECENTRALIZATION] Discovery позволяет узлам находить друг друга
    без центрального реестра:
    
    1. Новый узел подключается к bootstrap-узлу
    2. Отправляет DISCOVER запрос
    3. Получает PEER_LIST со списком известных пиров
    4. Подключается к некоторым из них
    5. Повторяет процесс для расширения сети
    
    Это создает mesh-топологию где каждый узел знает о части сети.
    """
    
    @property
    def message_type(self) -> MessageType:
        return MessageType.DISCOVER
    
    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        """
        Обработать DISCOVER запрос.
        
        Возвращает список известных пиров.
        """
        # Проверяем подпись
        if not crypto.verify_signature(message):
            return None
        
        # Получаем список пиров из контекста
        peer_manager = context.get("peer_manager")
        if not peer_manager:
            return None
        
        # Собираем информацию о пирах
        peers_info = []
        for peer in peer_manager.get_active_peers():
            peers_info.append(PeerInfo(
                node_id=peer.node_id,
                host=peer.host,
                port=peer.port,
                trust_score=peer.trust_score,
            ).to_dict())
        
        # Создаем ответ
        response = Message(
            type=MessageType.PEER_LIST,
            payload={
                "peers": peers_info,
                "total_known": len(peers_info),
            },
            sender_id=crypto.node_id,
        )
        
        return crypto.sign_message(response)
    
    @staticmethod
    def create_discover(crypto: Crypto) -> Message:
        """Создать DISCOVER запрос."""
        discover = Message(
            type=MessageType.DISCOVER,
            payload={
                "max_peers": 20,  # Максимум пиров в ответе
            },
            sender_id=crypto.node_id,
        )
        return crypto.sign_message(discover)


class PeerListHandler(MessageHandler):
    """Обработчик PEER_LIST сообщений."""
    
    @property
    def message_type(self) -> MessageType:
        return MessageType.PEER_LIST
    
    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        """
        Обработать PEER_LIST.
        
        Добавляет новых пиров в список для подключения.
        """
        if not crypto.verify_signature(message):
            return None
        
        # Извлекаем информацию о пирах
        peers_data = message.payload.get("peers", [])
        new_peers = [PeerInfo.from_dict(p) for p in peers_data]
        
        # Сохраняем в контексте для дальнейшей обработки
        context["discovered_peers"] = new_peers
        
        return None


class DataHandler(MessageHandler):
    """
    Обработчик DATA сообщений.
    
    [DECENTRALIZATION] DATA сообщения могут содержать
    произвольные данные для передачи между узлами.
    """
    
    @property
    def message_type(self) -> MessageType:
        return MessageType.DATA
    
    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        """Обработать DATA сообщение."""
        if not crypto.verify_signature(message):
            return None
        
        # Данные передаются callback-у в контексте
        data_callback = context.get("on_data")
        if data_callback:
            await data_callback(message.payload, message.sender_id)
        
        return None


class ServiceRequestHandler(MessageHandler):
    """
    Обработчик SERVICE_REQUEST сообщений.
    
    [MARKET] Layer 3 - Рынок услуг:
    - Получает запрос на услугу от другого узла
    - Передает запрос в AgentManager
    - Возвращает результат и записывает долг в Ledger
    
    [ECONOMY] Процесс:
    1. Узел A отправляет SERVICE_REQUEST с payload и budget
    2. Узел B проверяет подпись и наличие услуги
    3. AgentManager выполняет услугу
    4. Стоимость записывается в Ledger (A должен B)
    5. SERVICE_RESPONSE отправляется обратно A
    """
    
    @property
    def message_type(self) -> MessageType:
        return MessageType.SERVICE_REQUEST
    
    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        """
        Обработать SERVICE_REQUEST.
        
        [ECONOMY] При успешном выполнении:
        - Стоимость записывается в Ledger автоматически через AgentManager
        - Результат возвращается отправителю
        """
        # Проверяем подпись
        if not crypto.verify_signature(message):
            return self._create_error_response(
                crypto, message, "Invalid signature"
            )
        
        # Получаем AgentManager из контекста
        agent_manager = context.get("agent_manager")
        if not agent_manager:
            return self._create_error_response(
                crypto, message, "No agent manager available"
            )
        
        # Извлекаем данные запроса
        payload = message.payload
        service_name = payload.get("service_name", "")
        service_payload = payload.get("payload")
        budget = payload.get("budget", 0)
        request_id = payload.get("request_id", "")
        
        if not service_name:
            return self._create_error_response(
                crypto, message, "Missing service_name"
            )
        
        # Импортируем здесь чтобы избежать циклических импортов
        from agents.manager import ServiceRequest
        
        # Создаем запрос
        request = ServiceRequest(
            service_name=service_name,
            payload=service_payload,
            requester_id=message.sender_id,
            budget=budget,
            request_id=request_id,
        )
        
        # Обрабатываем запрос через AgentManager
        response = await agent_manager.handle_request(request)
        
        # Формируем ответное сообщение
        reply = Message(
            type=MessageType.SERVICE_RESPONSE,
            payload=response.to_dict(),
            sender_id=crypto.node_id,
        )
        
        return crypto.sign_message(reply)
    
    def _create_error_response(
        self,
        crypto: Crypto,
        request: Message,
        error: str
    ) -> Message:
        """Создать ответ с ошибкой."""
        reply = Message(
            type=MessageType.SERVICE_RESPONSE,
            payload={
                "success": False,
                "result": None,
                "cost": 0,
                "execution_time": 0,
                "request_id": request.payload.get("request_id", ""),
                "error": error,
                "provider_id": crypto.node_id,
            },
            sender_id=crypto.node_id,
        )
        return crypto.sign_message(reply)
    
    @staticmethod
    def create_service_request(
        crypto: Crypto,
        service_name: str,
        payload: Any,
        budget: float,
    ) -> Message:
        """
        Создать SERVICE_REQUEST сообщение.
        
        Args:
            crypto: Криптомодуль для подписи
            service_name: Название услуги ("echo", "storage", etc.)
            payload: Данные для обработки
            budget: Максимальный бюджет
        
        Returns:
            Подписанное сообщение SERVICE_REQUEST
        """
        import hashlib
        
        request_id = hashlib.sha256(
            f"{service_name}:{crypto.node_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        msg = Message(
            type=MessageType.SERVICE_REQUEST,
            payload={
                "service_name": service_name,
                "payload": payload,
                "budget": budget,
                "request_id": request_id,
            },
            sender_id=crypto.node_id,
        )
        return crypto.sign_message(msg)


class ServiceResponseHandler(MessageHandler):
    """
    Обработчик SERVICE_RESPONSE сообщений.
    
    [MARKET] Обрабатывает ответы на наши запросы услуг.
    """
    
    @property
    def message_type(self) -> MessageType:
        return MessageType.SERVICE_RESPONSE
    
    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        """Обработать SERVICE_RESPONSE."""
        if not crypto.verify_signature(message):
            return None
        
        # Сохраняем ответ в контексте для обработки
        context["service_response"] = message.payload
        
        # Callback если есть
        callback = context.get("on_service_response")
        if callback:
            from agents.manager import ServiceResponse
            response = ServiceResponse.from_dict(message.payload)
            await callback(response, message.sender_id)
        
        return None


class ProtocolRouter:
    """
    Маршрутизатор протокола.
    
    Направляет входящие сообщения соответствующим обработчикам.
    """
    
    def __init__(self):
        self.handlers: Dict[MessageType, MessageHandler] = {}
        
        # Регистрируем стандартные обработчики
        self.register(PingPongHandler())
        self.register(PongHandler())
        self.register(DiscoverHandler())
        self.register(PeerListHandler())
        self.register(DataHandler())
        
        # [MARKET] Layer 3 - обработчики услуг
        self.register(ServiceRequestHandler())
        self.register(ServiceResponseHandler())
        self.register(CacheRequestHandler())
        self.register(CacheResponseHandler())
    
    def register(self, handler: MessageHandler) -> None:
        """Зарегистрировать обработчик."""
        self.handlers[handler.message_type] = handler
    
    async def route(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        """
        Маршрутизировать сообщение к обработчику.
        
        Returns:
            Ответное сообщение или None
        """
        handler = self.handlers.get(message.type)
        if not handler:
            return None
        
        return await handler.handle(message, crypto, context)


class CacheRequestHandler(MessageHandler):
    """Обработчик запросов на кэшированные чанки."""

    @property
    def message_type(self) -> MessageType:
        return MessageType.CACHE_REQUEST

    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        if not crypto.verify_signature(message):
            return None

        payload = message.payload or {}
        chunk_hash = payload.get("hash")
        if not chunk_hash:
            return None

        data_bytes: Optional[bytes] = None

        amplifier = context.get("amplifier")
        if amplifier:
            data_bytes = await amplifier.get_chunk(chunk_hash)

        if data_bytes is None:
            agent_manager = context.get("agent_manager")
            if agent_manager:
                agent = agent_manager.get_agent("cache_provider")
                if agent and hasattr(agent, "get_chunk"):
                    try:
                        data_bytes = await agent.get_chunk(chunk_hash)
                    except Exception:
                        data_bytes = None

        response_payload: Dict[str, Any] = {"found": data_bytes is not None, "hash": chunk_hash}
        if data_bytes is not None:
            response_payload["data"] = base64.b64encode(data_bytes).decode("ascii")

        response = Message(
            type=MessageType.CACHE_RESPONSE,
            payload=response_payload,
            sender_id=crypto.node_id,
        )
        return crypto.sign_message(response)


class CacheResponseHandler(MessageHandler):
    """
    Обработчик ответов на кэш-запросы.
    
    [SECURITY] Интеграция с Merkle verification и Trust slashing:
    - При получении чанка проверяется Merkle proof
    - При INVALID_MERKLE_PROOF -> мгновенный slashing пира
    """

    @property
    def message_type(self) -> MessageType:
        return MessageType.CACHE_RESPONSE

    async def handle(
        self,
        message: Message,
        crypto: Crypto,
        context: Dict[str, Any]
    ) -> Optional[Message]:
        if not crypto.verify_signature(message):
            return None

        amplifier = context.get("amplifier")
        if amplifier:
            payload = message.payload or {}
            
            # [SECURITY] Check for merkle verification result
            merkle_valid = payload.get("merkle_valid")
            if merkle_valid is False:
                # SLASHING: Invalid merkle proof detected
                logger.error(
                    f"[CACHE] INVALID_MERKLE_PROOF from {message.sender_id[:8]}... "
                    f"chunk_hash={payload.get('hash', 'unknown')}"
                )
                
                # Slash the peer
                trust_system: Optional["WeightedTrustScore"] = context.get("trust_system")
                if trust_system:
                    from economy.trust import TrustEvent
                    await trust_system.record_event(
                        message.sender_id,
                        TrustEvent.INVALID_MERKLE_PROOF,
                    )
                    logger.warning(f"[CACHE] Peer {message.sender_id[:8]}... SLASHED for invalid merkle proof")
                
                # Do NOT process the chunk
                return None
            
            await amplifier.handle_cache_response(payload)
        return None


# ============================================================================
# Helper functions for security integration
# ============================================================================

async def verify_message_security(
    message: Message,
    crypto: Crypto,
    context: Dict[str, Any],
) -> bool:
    """
    Комплексная проверка безопасности сообщения.
    
    [SECURITY] Проверяет:
    1. Подпись сообщения
    2. Возраст сообщения (MAX_AGE = 60s)
    3. Replay protection (persistent nonce check)
    4. Trust score пира (не в blacklist)
    
    Args:
        message: Сообщение для проверки
        crypto: Криптомодуль
        context: Контекст с replay_protector и trust_system
    
    Returns:
        True если сообщение прошло все проверки
    """
    MAX_MESSAGE_AGE = 60.0
    
    # Check 1: Signature
    if not crypto.verify_signature(message):
        logger.warning(f"[SECURITY] Invalid signature from {message.sender_id[:8]}...")
        return False
    
    # Check 2: Age
    age = time.time() - message.timestamp
    if age > MAX_MESSAGE_AGE:
        logger.warning(f"[SECURITY] Message too old ({age:.1f}s) from {message.sender_id[:8]}...")
        return False
    
    # Check 3: Replay protection
    replay_protector: Optional["ReplayProtector"] = context.get("replay_protector")
    if replay_protector and message.nonce:
        try:
            nonce_bytes = base64.b64decode(message.nonce)
            if not await replay_protector.is_nonce_fresh(nonce_bytes):
                logger.warning(f"[SECURITY] Replay attack from {message.sender_id[:8]}...")
                return False
        except Exception as e:
            logger.error(f"[SECURITY] Nonce check error: {e}")
    
    # Check 4: Trust blacklist
    trust_system: Optional["WeightedTrustScore"] = context.get("trust_system")
    if trust_system:
        if trust_system.is_blacklisted(message.sender_id):
            logger.warning(f"[SECURITY] Blacklisted peer {message.sender_id[:8]}...")
            return False
    
    return True


async def slash_peer_for_merkle_violation(
    peer_id: str,
    context: Dict[str, Any],
    chunk_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Слэшинг пира за нарушение Merkle verification.
    
    [SECURITY] Вызывается из P2PLoader при обнаружении
    невалидного Merkle proof.
    
    Args:
        peer_id: ID нарушившего пира
        context: Контекст с trust_system
        chunk_info: Информация о чанке (для логирования)
    """
    trust_system: Optional["WeightedTrustScore"] = context.get("trust_system")
    if trust_system:
        from economy.trust import TrustEvent
        await trust_system.record_event(
            peer_id,
            TrustEvent.INVALID_MERKLE_PROOF,
        )
        
        chunk_desc = ""
        if chunk_info:
            chunk_desc = f" chunk={chunk_info.get('index', '?')}"
        
        logger.error(
            f"[SECURITY] SLASHED peer {peer_id[:8]}... "
            f"for INVALID_MERKLE_PROOF{chunk_desc}"
        )
