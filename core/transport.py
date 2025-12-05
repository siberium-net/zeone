"""
Transport Layer - Шифрование и маскировка трафика
=================================================

[SECURITY] Этот модуль обеспечивает:
1. E2E шифрование через NaCl Box (Curve25519 + XSalsa20 + Poly1305)
2. Цифровые подписи через Ed25519
3. Маскировку трафика под HTTP/WebSocket для обхода DPI

[DECENTRALIZATION] Шифрование происходит между пирами напрямую,
без посредников. Каждый узел имеет свою ключевую пару.
"""

import base64
import json
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, List

from nacl.public import PrivateKey, PublicKey, Box
from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import Base64Encoder
from nacl.exceptions import CryptoError


class MessageType(Enum):
    """Типы сообщений в протоколе P2P сети."""
    
    # Базовые сообщения
    PING = auto()           # Проверка связи
    PONG = auto()           # Ответ на PING
    
    # Discovery
    DISCOVER = auto()       # Запрос списка пиров
    PEER_LIST = auto()      # Ответ со списком пиров
    
    # Данные
    DATA = auto()           # Произвольные данные
    
    # Экономика
    IOU = auto()            # Долговая расписка
    IOU_ACK = auto()        # Подтверждение IOU
    BALANCE_CLAIM = auto()  # Заявление о балансе при handshake
    BALANCE_ACK = auto()    # Подтверждение баланса
    
    # Агенты/Услуги (Layer 3: Market)
    CONTRACT = auto()           # Код контракта
    CONTRACT_RESULT = auto()    # Результат выполнения
    SERVICE_REQUEST = auto()    # Запрос услуги
    SERVICE_RESPONSE = auto()   # Ответ на запрос услуги
    SERVICE_LIST = auto()       # Список доступных услуг

    # VPN / Streaming
    VPN_CONNECT = auto()        # Запрос на установление туннеля
    VPN_CONNECT_RESULT = auto() # Результат подключения
    VPN_DATA = auto()           # Потоковые данные VPN
    VPN_CLOSE = auto()          # Закрытие туннеля
    STREAM = auto()             # Общие потоковые данные с seq
    CACHE_REQUEST = auto()      # Запрос кэшированного чанка
    CACHE_RESPONSE = auto()     # Ответ с чанком


@dataclass
class Message:
    """
    Обертка для сообщений в P2P сети.
    
    [SECURITY] Каждое сообщение содержит:
    - type: тип сообщения
    - payload: полезная нагрузка (может быть зашифрована)
    - sender_id: ID отправителя (публичный ключ в hex)
    - timestamp: время создания (для защиты от replay-атак)
    - signature: подпись всего сообщения
    - nonce: случайное значение для уникальности
    """
    
    type: MessageType
    payload: Dict[str, Any]
    sender_id: str
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    nonce: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "type": self.type.name,
            "payload": self.payload,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "nonce": self.nonce,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Десериализация из словаря."""
        return cls(
            type=MessageType[data["type"]],
            payload=data["payload"],
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            signature=data.get("signature"),
            nonce=data.get("nonce"),
        )
    
    def to_json(self) -> str:
        """Сериализация в JSON строку."""
        return json.dumps(self.to_dict(), separators=(",", ":"))
    
    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Десериализация из JSON строки."""
        return cls.from_dict(json.loads(json_str))
    
    def get_signing_data(self) -> bytes:
        """
        Получить данные для подписи.
        
        [SECURITY] Подписываются все поля кроме самой подписи.
        Это гарантирует целостность сообщения.
        """
        data = {
            "type": self.type.name,
            "payload": self.payload,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }
        # Детерминированная сериализация для консистентных подписей
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class StreamingMessage:
    """
    Потоковое сообщение с порядковым номером.

    [STREAMING] Используется для передачи больших данных по кускам
    с сохранением порядка доставки.
    """

    stream_id: str
    seq: int
    data: bytes
    eof: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        """Преобразовать в payload для Message."""
        payload: Dict[str, Any] = {
            "stream_id": self.stream_id,
            "seq": self.seq,
            "data": base64.b64encode(self.data).decode("ascii"),
            "eof": self.eof,
        }
        if self.metadata:
            payload["meta"] = self.metadata
        return payload

    def to_message(
        self,
        sender_id: str,
        message_type: MessageType = MessageType.STREAM,
    ) -> Message:
        """Обертка в общий Message для передачи по сети."""
        return Message(
            type=message_type,
            payload=self.to_payload(),
            sender_id=sender_id,
        )

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "StreamingMessage":
        """Создать StreamingMessage из payload."""
        data_b64 = payload.get("data", "")
        raw = base64.b64decode(data_b64) if data_b64 else b""
        return cls(
            stream_id=payload["stream_id"],
            seq=int(payload.get("seq", 0)),
            data=raw,
            eof=bool(payload.get("eof", False)),
            metadata=payload.get("meta"),
        )

    @classmethod
    def from_message(cls, message: Message) -> "StreamingMessage":
        """Создать StreamingMessage из Message."""
        return cls.from_payload(message.payload)

    @staticmethod
    def chunk_bytes(
        data: bytes,
        stream_id: str,
        start_seq: int = 0,
        chunk_size: int = 16_384,
        eof: bool = True,
    ) -> List["StreamingMessage"]:
        """
        Разбить байты на последовательность StreamingMessage.

        Args:
            data: исходные байты
            stream_id: ID потока
            start_seq: с какого порядкового номера начинать
            chunk_size: размер одного блока
            eof: помечать последний блок как eof
        """
        messages: List[StreamingMessage] = []
        seq = start_seq
        for offset in range(0, len(data), chunk_size):
            chunk = data[offset:offset + chunk_size]
            is_last = offset + chunk_size >= len(data)
            messages.append(
                StreamingMessage(
                    stream_id=stream_id,
                    seq=seq,
                    data=chunk,
                    eof=eof and is_last,
                )
            )
            seq += 1
        if not messages and eof:
            messages.append(
                StreamingMessage(
                    stream_id=stream_id,
                    seq=start_seq,
                    data=b"",
                    eof=True,
                )
            )
        return messages


class StreamingBuffer:
    """
    Буфер сборки потоковых сообщений по порядку seq.

    [STREAMING] Позволяет принимать сообщения вне порядка и
    возвращать готовые байты в правильной последовательности.
    """

    def __init__(self):
        self._buffers: Dict[str, Dict[str, Any]] = {}

    def add(self, message: StreamingMessage) -> Tuple[List[bytes], bool]:
        """
        Добавить сообщение в буфер.

        Returns:
            (готовые_чанки, stream_closed)
        """
        state = self._buffers.setdefault(
            message.stream_id,
            {"expected": 0, "pending": {}},
        )

        if message.seq < state["expected"]:
            return ([], message.eof)

        state["pending"][message.seq] = message

        ready: List[bytes] = []
        finished = False

        while state["expected"] in state["pending"]:
            next_msg: StreamingMessage = state["pending"].pop(state["expected"])
            ready.append(next_msg.data)
            finished = finished or next_msg.eof
            state["expected"] += 1
            if next_msg.eof:
                self._buffers.pop(message.stream_id, None)
                break

        return ready, finished


class Crypto:
    """
    Криптографический модуль на базе PyNaCl.
    
    [SECURITY] Использует:
    - Ed25519 для подписей (SigningKey/VerifyKey)
    - Curve25519 для обмена ключами (PrivateKey/PublicKey)
    - XSalsa20-Poly1305 для шифрования (Box)
    
    [DECENTRALIZATION] Каждый узел генерирует свою ключевую пару.
    Идентичность узла = его публичный ключ.
    Нет центра сертификации - доверие строится на поведении.
    """
    
    def __init__(self, signing_key: Optional[SigningKey] = None):
        """
        Инициализация криптографического модуля.
        
        Args:
            signing_key: Существующий ключ подписи или None для генерации нового
        """
        # Ключ для подписей (Ed25519)
        self.signing_key: SigningKey = signing_key or SigningKey.generate()
        self.verify_key: VerifyKey = self.signing_key.verify_key
        
        # Ключ для шифрования (Curve25519)
        # Конвертируем Ed25519 ключ в Curve25519 для шифрования
        self.private_key: PrivateKey = self.signing_key.to_curve25519_private_key()
        self.public_key: PublicKey = self.private_key.public_key
        
        # ID узла = hex публичного ключа подписи
        self.node_id: str = self.verify_key.encode(encoder=Base64Encoder).decode("ascii")
    
    @classmethod
    def from_seed(cls, seed: bytes) -> "Crypto":
        """
        Создать криптомодуль из seed (32 байта).
        
        [SECURITY] Seed должен быть криптографически случайным.
        Используется для восстановления идентичности узла.
        """
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")
        signing_key = SigningKey(seed)
        return cls(signing_key)
    
    def export_identity(self) -> bytes:
        """Экспорт приватного ключа для сохранения."""
        return bytes(self.signing_key)
    
    @classmethod
    def import_identity(cls, key_bytes: bytes) -> "Crypto":
        """Импорт приватного ключа из байтов."""
        signing_key = SigningKey(key_bytes)
        return cls(signing_key)
    
    def sign_message(self, message: Message) -> Message:
        """
        Подписать сообщение.
        
        [SECURITY] Подпись создается над всеми полями сообщения,
        гарантируя, что сообщение не было изменено.
        """
        import os
        # Генерируем nonce если его нет
        if message.nonce is None:
            message.nonce = base64.b64encode(os.urandom(16)).decode("ascii")
        
        # Получаем данные для подписи
        signing_data = message.get_signing_data()
        
        # Создаем подпись
        signed = self.signing_key.sign(signing_data)
        message.signature = base64.b64encode(signed.signature).decode("ascii")
        
        return message
    
    def verify_signature(self, message: Message) -> bool:
        """
        Проверить подпись сообщения.
        
        [SECURITY] Возвращает True только если:
        1. Подпись валидна
        2. Подпись соответствует sender_id
        
        [DECENTRALIZATION] Любой узел может проверить подпись,
        не обращаясь к центральному серверу.
        """
        if message.signature is None:
            return False
        
        try:
            # Получаем публичный ключ отправителя из sender_id
            verify_key = VerifyKey(
                message.sender_id.encode("ascii"),
                encoder=Base64Encoder
            )
            
            # Получаем подпись
            signature = base64.b64decode(message.signature)
            
            # Получаем данные для проверки
            signing_data = message.get_signing_data()
            
            # Проверяем подпись
            verify_key.verify(signing_data, signature)
            return True
            
        except (CryptoError, Exception):
            return False
    
    def encrypt_payload(
        self,
        payload: Dict[str, Any],
        recipient_public_key: PublicKey
    ) -> str:
        """
        Зашифровать payload для конкретного получателя.
        
        [SECURITY] Использует NaCl Box (Curve25519-XSalsa20-Poly1305):
        - Curve25519: ECDH для получения общего секрета
        - XSalsa20: потоковое шифрование
        - Poly1305: аутентификация (MAC)
        
        [DECENTRALIZATION] Только отправитель и получатель могут
        прочитать сообщение. Промежуточные узлы видят только
        зашифрованные данные.
        """
        # Создаем Box для шифрования между нами и получателем
        box = Box(self.private_key, recipient_public_key)
        
        # Сериализуем payload
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        
        # Шифруем (nonce генерируется автоматически и добавляется к ciphertext)
        encrypted = box.encrypt(payload_bytes, encoder=Base64Encoder)
        
        return encrypted.decode("ascii")
    
    def decrypt_payload(
        self,
        encrypted_payload: str,
        sender_public_key: PublicKey
    ) -> Dict[str, Any]:
        """
        Расшифровать payload от конкретного отправителя.
        
        [SECURITY] Расшифровка возможна только если:
        1. У нас есть правильный приватный ключ
        2. Сообщение было зашифровано для нас
        3. Сообщение не было изменено (MAC проверка)
        """
        # Создаем Box для расшифровки
        box = Box(self.private_key, sender_public_key)
        
        # Расшифровываем
        encrypted_bytes = encrypted_payload.encode("ascii")
        decrypted = box.decrypt(encrypted_bytes, encoder=Base64Encoder)
        
        # Десериализуем
        return json.loads(decrypted.decode("utf-8"))
    
    @staticmethod
    def public_key_from_id(node_id: str) -> PublicKey:
        """Получить PublicKey для шифрования из node_id."""
        # node_id это VerifyKey в Base64
        verify_key = VerifyKey(node_id.encode("ascii"), encoder=Base64Encoder)
        # Конвертируем в Curve25519 PublicKey для шифрования
        return verify_key.to_curve25519_public_key()


class TrafficMasker:
    """
    Маскировка трафика под HTTP/WebSocket.
    
    [SECURITY] Цель - затруднить идентификацию P2P трафика
    системами DPI (Deep Packet Inspection).
    
    Формат:
    - Заголовки выглядят как обычный HTTP запрос/ответ
    - Payload кодируется в Base64 и передается в теле
    - WebSocket upgrade для постоянных соединений
    """
    
    # Фейковые HTTP заголовки для маскировки
    HTTP_REQUEST_TEMPLATE = (
        "POST /api/v1/sync HTTP/1.1\r\n"
        "Host: {host}\r\n"
        "Content-Type: application/json\r\n"
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\r\n"
        "Accept: application/json\r\n"
        "Content-Length: {length}\r\n"
        "Connection: keep-alive\r\n"
        "\r\n"
        "{payload}"
    )
    
    HTTP_RESPONSE_TEMPLATE = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Server: nginx/1.18.0\r\n"
        "Content-Length: {length}\r\n"
        "Connection: keep-alive\r\n"
        "\r\n"
        "{payload}"
    )
    
    @classmethod
    def mask_as_http_request(cls, message: Message, host: str = "api.example.com") -> bytes:
        """
        Замаскировать сообщение под HTTP POST запрос.
        
        [SECURITY] Для внешнего наблюдателя трафик выглядит
        как обычные API вызовы к веб-серверу.
        """
        # Сериализуем и кодируем сообщение
        payload = message.to_json()
        encoded_payload = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        
        # Оборачиваем в JSON чтобы выглядело как обычный API запрос
        body = json.dumps({"data": encoded_payload, "v": "1.0"})
        
        # Формируем HTTP запрос
        http_request = cls.HTTP_REQUEST_TEMPLATE.format(
            host=host,
            length=len(body),
            payload=body
        )
        
        return http_request.encode("utf-8")
    
    @classmethod
    def mask_as_http_response(cls, message: Message) -> bytes:
        """
        Замаскировать сообщение под HTTP ответ.
        """
        payload = message.to_json()
        encoded_payload = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        
        body = json.dumps({"data": encoded_payload, "status": "ok"})
        
        http_response = cls.HTTP_RESPONSE_TEMPLATE.format(
            length=len(body),
            payload=body
        )
        
        return http_response.encode("utf-8")
    
    @classmethod
    def unmask_from_http(cls, data: bytes) -> Optional[Message]:
        """
        Извлечь сообщение из HTTP запроса/ответа.
        
        [SECURITY] Парсит как запросы, так и ответы.
        Возвращает None если формат не распознан.
        """
        try:
            text = data.decode("utf-8")
            
            # Находим тело (после двойного CRLF)
            body_start = text.find("\r\n\r\n")
            if body_start == -1:
                return None
            
            body = text[body_start + 4:]
            
            # Парсим JSON тело
            body_json = json.loads(body)
            
            # Извлекаем закодированный payload
            encoded_payload = body_json.get("data")
            if not encoded_payload:
                return None
            
            # Декодируем и десериализуем
            payload_json = base64.b64decode(encoded_payload).decode("utf-8")
            return Message.from_json(payload_json)
            
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            return None
    
    @classmethod
    def is_http_masked(cls, data: bytes) -> bool:
        """Проверить, является ли данные HTTP-замаскированными."""
        try:
            text = data.decode("utf-8")
            return (
                text.startswith("POST /api/") or 
                text.startswith("GET /api/") or
                text.startswith("HTTP/1.")
            )
        except UnicodeDecodeError:
            return False


# Простой формат без маскировки для локальных сетей
# [DEPRECATED] Используй BinaryTransport для production
class SimpleTransport:
    """
    Простой транспорт без HTTP маскировки.
    
    [DEPRECATED] Используется только для legacy/тестов.
    Для production используй BinaryTransport.
    
    Формат: 4 байта длины (big-endian) + JSON payload
    """
    
    @staticmethod
    def pack(message: Message) -> bytes:
        """Упаковать сообщение для передачи."""
        payload = message.to_json().encode("utf-8")
        length = len(payload)
        return length.to_bytes(4, "big") + payload
    
    @staticmethod
    def unpack_length(header: bytes) -> int:
        """Получить длину сообщения из заголовка (4 байта)."""
        if len(header) < 4:
            raise ValueError("Header too short")
        return int.from_bytes(header[:4], "big")
    
    @staticmethod
    def unpack(data: bytes) -> Message:
        """Распаковать сообщение."""
        payload = data.decode("utf-8")
        return Message.from_json(payload)
    
    @staticmethod
    def get_payload_size(data: bytes) -> int:
        """Получить размер payload в байтах."""
        return len(data)


# ============================================================================
# Binary Wire Protocol V1 - PRODUCTION TRANSPORT
# ============================================================================

# Import wire protocol components
from core.wire import (
    MAGIC,
    HEADER_SIZE,
    PROTOCOL_VERSION,
    WireMessageType,
    WireHeader,
    WireMessage,
    WireCodec,
    WireStreamReader,
    WireStreamWriter,
    WireError,
    InvalidMagicError,
    InvalidSignatureError,
    PayloadTooLargeError,
    FLAG_ENCRYPTED,
    FLAG_COMPRESSED,
)


class BinaryTransport:
    """
    Binary Wire Protocol V1 Transport.
    
    [HARD FORK] Полный отказ от JSON в заголовках.
    
    Спецификация: см. core/wire.py
    
    [SECURITY]
    - Magic b'ZE' обязателен - иначе закрываем сокет
    - Ed25519 подпись покрывает header + payload
    - NaCl Box для E2E шифрования
    
    [USAGE]
        transport = BinaryTransport(crypto)
        
        # Pack
        wire_bytes = transport.pack(msg_type, payload, recipient_key)
        
        # Unpack
        message = transport.unpack(wire_bytes, sender_keys)
    """
    
    def __init__(self, crypto: Crypto):
        """
        Args:
            crypto: Криптомодуль узла
        """
        self.crypto = crypto
        self.codec = WireCodec(
            signing_key=crypto.signing_key,
            private_key=crypto.private_key,
        )
        self.stream_reader = WireStreamReader(self.codec)
        self.stream_writer = WireStreamWriter(self.codec)
    
    def pack(
        self,
        msg_type: WireMessageType,
        payload: bytes,
        recipient_public_key: Optional[PublicKey] = None,
        flags: int = 0,
    ) -> bytes:
        """
        Упаковать сообщение в бинарный формат.
        
        [WIRE] Формат:
            [Header 98 bytes][Encrypted Payload N bytes]
        
        Args:
            msg_type: Тип сообщения
            payload: Сырой payload (msgpack/json bytes)
            recipient_public_key: Ключ получателя для шифрования
            flags: Дополнительные флаги
        
        Returns:
            Wire bytes
        """
        return self.codec.encode(msg_type, payload, recipient_public_key, flags)
    
    def unpack(
        self,
        data: bytes,
        sender_verify_key: VerifyKey,
        sender_public_key: Optional[PublicKey] = None,
    ) -> WireMessage:
        """
        Распаковать сообщение из бинарного формата.
        
        [SECURITY]
        - Проверяет Magic (InvalidMagicError -> close socket)
        - Верифицирует подпись
        - Дешифрует payload
        
        Args:
            data: Wire bytes
            sender_verify_key: Ключ для проверки подписи
            sender_public_key: Ключ для дешифрования
        
        Returns:
            WireMessage
        
        Raises:
            InvalidMagicError: Закрыть соединение!
            InvalidSignatureError: Невалидная подпись
        """
        return self.codec.decode(data, sender_verify_key, sender_public_key)
    
    def unpack_header(self, data: bytes) -> WireHeader:
        """
        Распаковать только заголовок.
        
        Используется для чтения длины payload перед
        полным чтением сообщения.
        
        [SECURITY] Проверяет Magic!
        
        Args:
            data: Минимум 98 байт
        
        Returns:
            WireHeader
        
        Raises:
            InvalidMagicError: Закрыть соединение!
        """
        return self.codec.decode_header_only(data)
    
    async def read_message(
        self,
        reader,  # asyncio.StreamReader
        sender_verify_key: VerifyKey,
        sender_public_key: Optional[PublicKey] = None,
    ) -> WireMessage:
        """
        Прочитать сообщение из потока.
        
        [SECURITY] При InvalidMagicError вызывающий код
        ОБЯЗАН закрыть соединение.
        """
        return await self.stream_reader.read_message(
            reader, sender_verify_key, sender_public_key
        )
    
    async def write_message(
        self,
        writer,  # asyncio.StreamWriter
        msg_type: WireMessageType,
        payload: bytes,
        recipient_public_key: Optional[PublicKey] = None,
        flags: int = 0,
    ) -> int:
        """
        Записать сообщение в поток.
        
        Returns:
            Количество записанных байт
        """
        return await self.stream_writer.write_message(
            writer, msg_type, payload, recipient_public_key, flags
        )
    
    @staticmethod
    def is_binary_protocol(data: bytes) -> bool:
        """
        Проверить, является ли данные бинарным протоколом.
        
        [MIGRATION] Используется для определения формата
        входящего соединения.
        
        Returns:
            True если начинается с Magic b'ZE'
        """
        return len(data) >= 2 and data[:2] == MAGIC
    
    @staticmethod
    def message_type_from_legacy(legacy_type: MessageType) -> WireMessageType:
        """
        Конвертировать legacy MessageType в WireMessageType.
        
        [MIGRATION] Для совместимости при переходе.
        """
        mapping = {
            MessageType.PING: WireMessageType.PING,
            MessageType.PONG: WireMessageType.PONG,
            MessageType.DISCOVER: WireMessageType.DISCOVER,
            MessageType.PEER_LIST: WireMessageType.PEER_LIST,
            MessageType.DATA: WireMessageType.DATA,
            MessageType.STREAM: WireMessageType.STREAM,
            MessageType.CACHE_REQUEST: WireMessageType.CACHE_REQUEST,
            MessageType.CACHE_RESPONSE: WireMessageType.CACHE_RESPONSE,
            MessageType.IOU: WireMessageType.IOU,
            MessageType.IOU_ACK: WireMessageType.IOU_ACK,
            MessageType.BALANCE_CLAIM: WireMessageType.BALANCE_CLAIM,
            MessageType.BALANCE_ACK: WireMessageType.BALANCE_ACK,
            MessageType.SERVICE_REQUEST: WireMessageType.SERVICE_REQUEST,
            MessageType.SERVICE_RESPONSE: WireMessageType.SERVICE_RESPONSE,
            MessageType.SERVICE_LIST: WireMessageType.SERVICE_LIST,
            MessageType.CONTRACT: WireMessageType.CONTRACT,
            MessageType.CONTRACT_RESULT: WireMessageType.CONTRACT_RESULT,
            MessageType.VPN_CONNECT: WireMessageType.VPN_CONNECT,
            MessageType.VPN_CONNECT_RESULT: WireMessageType.VPN_CONNECT_RESULT,
            MessageType.VPN_DATA: WireMessageType.VPN_DATA,
            MessageType.VPN_CLOSE: WireMessageType.VPN_CLOSE,
        }
        return mapping.get(legacy_type, WireMessageType.DATA)


class BlockingTransport:
    """
    Транспорт с блокировкой для защиты от leechers.
    
    [ECONOMY] BLOCKING MECHANISM:
    Перед отправкой данных проверяет баланс пира.
    Если peer имеет слишком большой долг - блокирует отправку.
    
    Это защита от "пиявок" (Leechers) которые только потребляют
    и не раздают данные.
    """
    
    def __init__(self, ledger: "Ledger"):  # type: ignore
        """
        Args:
            ledger: Экземпляр Ledger для проверки балансов
        """
        self.ledger = ledger
    
    async def can_send_to(self, peer_id: str) -> tuple:
        """
        Проверить, можно ли отправлять данные пиру.
        
        [ECONOMY] Блокировка включается если:
        - Баланс пира превышает лимит (peer должен нам слишком много)
        
        Returns:
            (can_send: bool, reason: str)
        """
        return await self.ledger.check_can_send(peer_id)
    
    def is_blocked(self, peer_id: str) -> bool:
        """
        Быстрая синхронная проверка блокировки (из кэша).
        
        Returns:
            True если peer заблокирован
        """
        return self.ledger.is_peer_blocked(peer_id)
    
    async def pack_with_accounting(
        self,
        message: Message,
        peer_id: str,
    ) -> tuple:
        """
        Упаковать сообщение с учетом в ledger.
        
        [ECONOMY] Автоматически записывает claim (peer будет должен нам).
        
        Returns:
            (data: bytes, size: int, blocked: bool, reason: str)
        """
        # Проверяем блокировку
        can_send, reason = await self.can_send_to(peer_id)
        
        if not can_send:
            return (b"", 0, True, reason)
        
        # Упаковываем
        data = SimpleTransport.pack(message)
        size = len(data)
        
        # Записываем claim - peer будет должен нам за эти данные
        await self.ledger.record_claim(peer_id, size)
        
        return (data, size, False, "OK")
    
    async def unpack_with_accounting(
        self,
        data: bytes,
        peer_id: str,
    ) -> Message:
        """
        Распаковать сообщение с учетом в ledger.
        
        [ECONOMY] Автоматически записывает debt (мы должны peer).
        
        Returns:
            Message
        """
        # Записываем debt - мы получили данные, должны peer
        size = len(data)
        await self.ledger.record_debt(peer_id, size)
        
        # Распаковываем
        return SimpleTransport.unpack(data)
