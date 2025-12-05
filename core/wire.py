"""
Binary Wire Protocol V1 - Hard Fork
====================================

[BREAKING CHANGE] Полный отказ от JSON в заголовках.
Бинарный протокол на базе struct для максимальной производительности.

Спецификация заголовка (Fixed Header: 98 bytes):
=================================================

Format String: `>2sBBHI24s64s` (Big-Endian)

| Field     | Type | Size | Description                           |
|-----------|------|------|---------------------------------------|
| Magic     | 2s   | 2    | b'ZE' (0x5A45) - идентификатор        |
| Version   | B    | 1    | 1 - версия протокола                  |
| MsgType   | B    | 1    | MessageType enum value                |
| Flags     | H    | 2    | 0x01=Encrypted, 0x02=Compressed       |
| Length    | I    | 4    | Размер payload                        |
| Nonce     | 24s  | 24   | Уникальный nonce для шифрования       |
| Signature | 64s  | 64   | Ed25519 подпись (header + payload)    |
|-----------|------|------|---------------------------------------|
| TOTAL     |      | 98   |                                       |

[SECURITY] Подпись покрывает:
- Все поля заголовка (кроме самой подписи)
- Весь зашифрованный payload

[MIGRATION] Hard Fork:
- Любые данные без Magic b'ZE' -> немедленное закрытие сокета
- Нет обратной совместимости с JSON протоколом
"""

import struct
import os
import logging
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

from nacl.signing import SigningKey, VerifyKey
from nacl.public import PrivateKey, PublicKey, Box
from nacl.encoding import RawEncoder
from nacl.exceptions import CryptoError

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol Constants
# ============================================================================

MAGIC = b'ZE'                    # Protocol identifier (0x5A45)
PROTOCOL_VERSION = 1             # Current protocol version
HEADER_SIZE = 98                 # Fixed header size in bytes
HEADER_FORMAT = '>2sBBHI24s64s'  # Big-endian struct format

# Flags
FLAG_ENCRYPTED = 0x0001          # Payload is encrypted
FLAG_COMPRESSED = 0x0002         # Payload is compressed (reserved)

# Limits
MAX_PAYLOAD_SIZE = 16 * 1024 * 1024  # 16 MB max payload


class WireMessageType(IntEnum):
    """
    Типы сообщений в бинарном протоколе.
    
    [WIRE] Значения должны быть стабильны между версиями.
    Новые типы добавляются в конец, старые не удаляются.
    """
    
    # Базовые (0-15)
    PING = 0
    PONG = 1
    HANDSHAKE = 2
    HANDSHAKE_ACK = 3
    
    # Discovery (16-31)
    DISCOVER = 16
    PEER_LIST = 17
    
    # Data (32-47)
    DATA = 32
    STREAM = 33
    CACHE_REQUEST = 34
    CACHE_RESPONSE = 35
    
    # Economy (48-63)
    IOU = 48
    IOU_ACK = 49
    BALANCE_CLAIM = 50
    BALANCE_ACK = 51
    
    # Services (64-79)
    SERVICE_REQUEST = 64
    SERVICE_RESPONSE = 65
    SERVICE_LIST = 66
    CONTRACT = 67
    CONTRACT_RESULT = 68
    
    # VPN (80-95)
    VPN_CONNECT = 80
    VPN_CONNECT_RESULT = 81
    VPN_DATA = 82
    VPN_CLOSE = 83
    
    # DHT (96-111)
    DHT_FIND_NODE = 96
    DHT_FIND_VALUE = 97
    DHT_STORE = 98
    DHT_RESPONSE = 99


class WireError(Exception):
    """Ошибки wire протокола."""
    pass


class InvalidMagicError(WireError):
    """Неверный Magic - не наш протокол."""
    pass


class InvalidVersionError(WireError):
    """Неподдерживаемая версия протокола."""
    pass


class InvalidSignatureError(WireError):
    """Невалидная подпись."""
    pass


class PayloadTooLargeError(WireError):
    """Payload превышает лимит."""
    pass


# ============================================================================
# Wire Header
# ============================================================================

@dataclass
class WireHeader:
    """
    Заголовок бинарного сообщения.
    
    [WIRE] Fixed 98 bytes, Big-Endian.
    """
    
    magic: bytes = MAGIC
    version: int = PROTOCOL_VERSION
    msg_type: int = 0
    flags: int = 0
    length: int = 0
    nonce: bytes = b'\x00' * 24
    signature: bytes = b'\x00' * 64
    
    def pack_unsigned(self) -> bytes:
        """
        Упаковать заголовок без подписи (для подписания).
        
        Returns:
            34 bytes: magic(2) + version(1) + type(1) + flags(2) + length(4) + nonce(24)
        """
        return struct.pack(
            '>2sBBHI24s',
            self.magic,
            self.version,
            self.msg_type,
            self.flags,
            self.length,
            self.nonce,
        )
    
    def pack(self) -> bytes:
        """
        Упаковать полный заголовок.
        
        Returns:
            98 bytes
        """
        return struct.pack(
            HEADER_FORMAT,
            self.magic,
            self.version,
            self.msg_type,
            self.flags,
            self.length,
            self.nonce,
            self.signature,
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> "WireHeader":
        """
        Распаковать заголовок из байтов.
        
        Args:
            data: Ровно 98 байт
        
        Returns:
            WireHeader
        
        Raises:
            InvalidMagicError: Неверный Magic
            InvalidVersionError: Неподдерживаемая версия
        """
        if len(data) < HEADER_SIZE:
            raise WireError(f"Header too short: {len(data)} < {HEADER_SIZE}")
        
        magic, version, msg_type, flags, length, nonce, signature = struct.unpack(
            HEADER_FORMAT,
            data[:HEADER_SIZE]
        )
        
        # [SECURITY] Проверка Magic
        if magic != MAGIC:
            raise InvalidMagicError(f"Invalid magic: {magic!r} != {MAGIC!r}")
        
        # Проверка версии
        if version != PROTOCOL_VERSION:
            raise InvalidVersionError(f"Unsupported version: {version}")
        
        return cls(
            magic=magic,
            version=version,
            msg_type=msg_type,
            flags=flags,
            length=length,
            nonce=nonce,
            signature=signature,
        )
    
    @property
    def is_encrypted(self) -> bool:
        return bool(self.flags & FLAG_ENCRYPTED)
    
    @property
    def is_compressed(self) -> bool:
        return bool(self.flags & FLAG_COMPRESSED)


# ============================================================================
# Wire Message
# ============================================================================

@dataclass
class WireMessage:
    """
    Полное сообщение с заголовком и payload.
    
    [WIRE] Формат на проводе:
        [Header 98 bytes][Encrypted Payload N bytes]
    """
    
    header: WireHeader
    payload: bytes  # Decrypted payload (msgpack или raw bytes)
    
    @property
    def msg_type(self) -> WireMessageType:
        return WireMessageType(self.header.msg_type)
    
    @property
    def total_size(self) -> int:
        return HEADER_SIZE + len(self.payload)


# ============================================================================
# Wire Codec
# ============================================================================

class WireCodec:
    """
    Кодек для бинарного протокола.
    
    [SECURITY] Использует:
    - Ed25519 для подписей
    - NaCl Box (XSalsa20-Poly1305) для шифрования
    
    [USAGE]
        codec = WireCodec(signing_key, private_key)
        
        # Encode
        wire_bytes = codec.encode(msg_type, payload, recipient_pubkey)
        
        # Decode
        message = codec.decode(wire_bytes, sender_pubkey)
    """
    
    def __init__(
        self,
        signing_key: SigningKey,
        private_key: PrivateKey,
    ):
        """
        Args:
            signing_key: Ключ для подписей (Ed25519)
            private_key: Ключ для шифрования (Curve25519)
        """
        self.signing_key = signing_key
        self.verify_key = signing_key.verify_key
        self.private_key = private_key
        self.public_key = private_key.public_key
    
    def encode(
        self,
        msg_type: WireMessageType,
        payload: bytes,
        recipient_public_key: Optional[PublicKey] = None,
        flags: int = 0,
    ) -> bytes:
        """
        Закодировать сообщение в wire format.
        
        [SECURITY] Процесс:
        1. Генерируем nonce
        2. Шифруем payload (если recipient указан)
        3. Формируем заголовок (без подписи)
        4. Подписываем header + encrypted_payload
        5. Вставляем подпись в заголовок
        
        Args:
            msg_type: Тип сообщения
            payload: Сырой payload (bytes)
            recipient_public_key: Ключ получателя (для шифрования)
            flags: Дополнительные флаги
        
        Returns:
            Wire bytes: header(98) + encrypted_payload(N)
        
        Raises:
            PayloadTooLargeError: Payload превышает MAX_PAYLOAD_SIZE
        """
        # Генерируем nonce
        nonce = os.urandom(24)
        
        # Шифруем если указан получатель
        if recipient_public_key is not None:
            box = Box(self.private_key, recipient_public_key)
            encrypted_payload = box.encrypt(payload, nonce, encoder=RawEncoder).ciphertext
            flags |= FLAG_ENCRYPTED
        else:
            encrypted_payload = payload
        
        # Проверяем размер
        if len(encrypted_payload) > MAX_PAYLOAD_SIZE:
            raise PayloadTooLargeError(
                f"Payload too large: {len(encrypted_payload)} > {MAX_PAYLOAD_SIZE}"
            )
        
        # Формируем заголовок (без подписи)
        header = WireHeader(
            magic=MAGIC,
            version=PROTOCOL_VERSION,
            msg_type=int(msg_type),
            flags=flags,
            length=len(encrypted_payload),
            nonce=nonce,
            signature=b'\x00' * 64,  # Placeholder
        )
        
        # Данные для подписи: header_unsigned + encrypted_payload
        header_unsigned = header.pack_unsigned()
        signing_data = header_unsigned + encrypted_payload
        
        # Подписываем
        signed = self.signing_key.sign(signing_data, encoder=RawEncoder)
        signature = signed.signature
        
        # Вставляем подпись
        header.signature = signature
        
        return header.pack() + encrypted_payload
    
    def decode(
        self,
        data: bytes,
        sender_verify_key: VerifyKey,
        sender_public_key: Optional[PublicKey] = None,
    ) -> WireMessage:
        """
        Декодировать сообщение из wire format.
        
        [SECURITY] Процесс:
        1. Распаковываем заголовок
        2. Проверяем Magic
        3. Читаем payload
        4. Верифицируем подпись
        5. Дешифруем payload (если зашифрован)
        
        Args:
            data: Wire bytes
            sender_verify_key: Ключ отправителя для проверки подписи
            sender_public_key: Ключ отправителя для дешифрования
        
        Returns:
            WireMessage
        
        Raises:
            InvalidMagicError: Неверный Magic -> закрыть сокет
            InvalidSignatureError: Невалидная подпись
            WireError: Другие ошибки протокола
        """
        # Распаковываем заголовок
        header = WireHeader.unpack(data)
        
        # Извлекаем encrypted payload
        payload_start = HEADER_SIZE
        payload_end = payload_start + header.length
        
        if len(data) < payload_end:
            raise WireError(
                f"Incomplete payload: need {payload_end} bytes, got {len(data)}"
            )
        
        encrypted_payload = data[payload_start:payload_end]
        
        # Верифицируем подпись
        header_unsigned = header.pack_unsigned()
        signing_data = header_unsigned + encrypted_payload
        
        try:
            sender_verify_key.verify(signing_data, header.signature, encoder=RawEncoder)
        except CryptoError as e:
            raise InvalidSignatureError(f"Signature verification failed: {e}")
        
        # Дешифруем если нужно
        if header.is_encrypted:
            if sender_public_key is None:
                raise WireError("Encrypted message but no sender public key")
            
            box = Box(self.private_key, sender_public_key)
            try:
                payload = box.decrypt(encrypted_payload, header.nonce, encoder=RawEncoder)
            except CryptoError as e:
                raise WireError(f"Decryption failed: {e}")
        else:
            payload = encrypted_payload
        
        return WireMessage(header=header, payload=payload)
    
    def decode_header_only(self, data: bytes) -> WireHeader:
        """
        Декодировать только заголовок (для чтения длины payload).
        
        Args:
            data: Минимум 98 байт
        
        Returns:
            WireHeader
        
        Raises:
            InvalidMagicError: Неверный Magic -> CLOSE SOCKET
        """
        return WireHeader.unpack(data)


# ============================================================================
# Stream Reader/Writer
# ============================================================================

class WireStreamReader:
    """
    Потоковый читатель wire сообщений.
    
    [USAGE]
        reader = WireStreamReader(codec)
        
        # В цикле чтения
        header = reader.read_header(stream_reader)
        payload = await stream_reader.readexactly(header.length)
        message = reader.decode_payload(header, payload, sender_keys)
    """
    
    def __init__(self, codec: WireCodec):
        self.codec = codec
    
    async def read_message(
        self,
        reader,  # asyncio.StreamReader
        sender_verify_key: VerifyKey,
        sender_public_key: Optional[PublicKey] = None,
    ) -> WireMessage:
        """
        Прочитать полное сообщение из потока.
        
        [SECURITY] При InvalidMagicError вызывающий код
        должен закрыть соединение.
        """
        # Читаем заголовок
        header_bytes = await reader.readexactly(HEADER_SIZE)
        header = WireHeader.unpack(header_bytes)
        
        # Читаем payload
        if header.length > MAX_PAYLOAD_SIZE:
            raise PayloadTooLargeError(f"Payload too large: {header.length}")
        
        encrypted_payload = await reader.readexactly(header.length)
        
        # Верифицируем и дешифруем
        full_data = header_bytes + encrypted_payload
        return self.codec.decode(full_data, sender_verify_key, sender_public_key)


class WireStreamWriter:
    """
    Потоковый писатель wire сообщений.
    
    [USAGE]
        writer = WireStreamWriter(codec)
        await writer.write_message(stream_writer, msg_type, payload, recipient_key)
    """
    
    def __init__(self, codec: WireCodec):
        self.codec = codec
    
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
        wire_bytes = self.codec.encode(msg_type, payload, recipient_public_key, flags)
        writer.write(wire_bytes)
        await writer.drain()
        return len(wire_bytes)


# ============================================================================
# Connection Handler
# ============================================================================

async def handle_incoming_connection(
    reader,  # asyncio.StreamReader
    writer,  # asyncio.StreamWriter
    codec: WireCodec,
    on_message,  # Callable[[WireMessage], Awaitable[Optional[WireMessage]]]
    get_peer_keys,  # Callable[[str], Tuple[VerifyKey, PublicKey]]
):
    """
    Обработчик входящего соединения.
    
    [SECURITY] При InvalidMagicError соединение закрывается немедленно.
    
    Args:
        reader: asyncio StreamReader
        writer: asyncio StreamWriter
        codec: WireCodec для декодирования
        on_message: Callback для обработки сообщений
        get_peer_keys: Функция получения ключей пира по ID
    """
    peer_addr = writer.get_extra_info('peername')
    stream_reader = WireStreamReader(codec)
    stream_writer = WireStreamWriter(codec)
    
    # Peer keys будут установлены после handshake
    peer_verify_key: Optional[VerifyKey] = None
    peer_public_key: Optional[PublicKey] = None
    
    try:
        while True:
            # Пробуем прочитать первые байты для проверки Magic
            peek = await reader.read(2)
            if not peek:
                break  # Connection closed
            
            if peek != MAGIC:
                # [SECURITY] Hard Fork - немедленно закрываем
                logger.warning(
                    f"[WIRE] Invalid magic from {peer_addr}: {peek!r}. Closing connection."
                )
                break
            
            # Читаем остаток заголовка
            header_rest = await reader.readexactly(HEADER_SIZE - 2)
            header_bytes = peek + header_rest
            header = WireHeader.unpack(header_bytes)
            
            # Читаем payload
            encrypted_payload = await reader.readexactly(header.length)
            
            # Полное сообщение
            full_data = header_bytes + encrypted_payload
            
            # Если это handshake - особая обработка
            if header.msg_type == WireMessageType.HANDSHAKE:
                # Handshake payload содержит публичный ключ
                # Верифицируем с ключом из payload
                # TODO: Implement proper handshake
                pass
            
            # Декодируем
            if peer_verify_key is None:
                # Пока нет ключей - пропускаем верификацию
                # (должен быть handshake первым)
                message = WireMessage(
                    header=header,
                    payload=encrypted_payload,  # Не дешифруем
                )
            else:
                message = codec.decode(full_data, peer_verify_key, peer_public_key)
            
            # Обрабатываем
            response = await on_message(message)
            
            if response:
                await stream_writer.write_message(
                    writer,
                    response.msg_type,
                    response.payload,
                    peer_public_key,
                )
    
    except InvalidMagicError as e:
        logger.warning(f"[WIRE] Invalid magic from {peer_addr}: {e}")
    except InvalidSignatureError as e:
        logger.warning(f"[WIRE] Invalid signature from {peer_addr}: {e}")
    except Exception as e:
        logger.error(f"[WIRE] Connection error from {peer_addr}: {e}")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

