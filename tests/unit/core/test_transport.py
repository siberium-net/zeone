"""
Transport Layer Unit Tests
==========================

[CRITICAL] 100% coverage for core/transport.py
Network and cryptography are critical paths.
"""

import pytest
import json
import time
import base64
from pathlib import Path

from nacl.public import PrivateKey, PublicKey, Box
from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import Base64Encoder


# ============================================================================
# Message Tests
# ============================================================================

class TestMessageType:
    """Test MessageType enum."""
    
    def test_all_message_types_exist(self):
        """Verify all expected message types exist."""
        from core.transport import MessageType
        
        expected = [
            "PING", "PONG", "DISCOVER", "PEER_LIST", "DATA",
            "IOU", "IOU_ACK", "BALANCE_CLAIM", "BALANCE_ACK",
            "CONTRACT", "CONTRACT_RESULT", "SERVICE_REQUEST",
            "SERVICE_RESPONSE", "SERVICE_LIST", "VPN_CONNECT",
            "VPN_CONNECT_RESULT", "VPN_DATA", "VPN_CLOSE",
            "STREAM", "CACHE_REQUEST", "CACHE_RESPONSE"
        ]
        
        for name in expected:
            assert hasattr(MessageType, name), f"Missing MessageType.{name}"


class TestMessage:
    """Test Message dataclass."""
    
    def test_create_message(self):
        """Test message creation."""
        from core.transport import Message, MessageType
        
        msg = Message(
            type=MessageType.PING,
            payload={"test": "data"},
            sender_id="test_sender",
        )
        
        assert msg.type == MessageType.PING
        assert msg.payload == {"test": "data"}
        assert msg.sender_id == "test_sender"
        assert msg.timestamp > 0
        assert msg.signature is None
        assert msg.nonce is None
    
    def test_message_to_dict(self):
        """Test message serialization to dict."""
        from core.transport import Message, MessageType
        
        msg = Message(
            type=MessageType.DATA,
            payload={"key": "value"},
            sender_id="sender123",
            timestamp=1234567890.0,
            signature="sig123",
            nonce="nonce123",
        )
        
        d = msg.to_dict()
        
        assert d["type"] == "DATA"
        assert d["payload"] == {"key": "value"}
        assert d["sender_id"] == "sender123"
        assert d["timestamp"] == 1234567890.0
        assert d["signature"] == "sig123"
        assert d["nonce"] == "nonce123"
    
    def test_message_from_dict(self):
        """Test message deserialization from dict."""
        from core.transport import Message, MessageType
        
        data = {
            "type": "PONG",
            "payload": {"response": True},
            "sender_id": "responder",
            "timestamp": 1234567890.0,
            "signature": "sig",
            "nonce": "nc",
        }
        
        msg = Message.from_dict(data)
        
        assert msg.type == MessageType.PONG
        assert msg.payload == {"response": True}
        assert msg.sender_id == "responder"
        assert msg.timestamp == 1234567890.0
        assert msg.signature == "sig"
        assert msg.nonce == "nc"
    
    def test_message_to_json(self):
        """Test message serialization to JSON."""
        from core.transport import Message, MessageType
        
        msg = Message(
            type=MessageType.DATA,
            payload={"x": 1},
            sender_id="s1",
        )
        
        json_str = msg.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["type"] == "DATA"
        assert parsed["payload"] == {"x": 1}
    
    def test_message_from_json(self):
        """Test message deserialization from JSON."""
        from core.transport import Message, MessageType
        
        json_str = '{"type":"PING","payload":{},"sender_id":"s","timestamp":0,"signature":null,"nonce":null}'
        msg = Message.from_json(json_str)
        
        assert msg.type == MessageType.PING
        assert msg.sender_id == "s"
    
    def test_message_roundtrip(self):
        """Test message JSON roundtrip."""
        from core.transport import Message, MessageType
        
        original = Message(
            type=MessageType.SERVICE_REQUEST,
            payload={"service": "echo", "data": "hello"},
            sender_id="client",
            timestamp=time.time(),
        )
        
        json_str = original.to_json()
        restored = Message.from_json(json_str)
        
        assert restored.type == original.type
        assert restored.payload == original.payload
        assert restored.sender_id == original.sender_id
        assert restored.timestamp == original.timestamp
    
    def test_get_signing_data(self):
        """Test signing data generation."""
        from core.transport import Message, MessageType
        
        msg = Message(
            type=MessageType.DATA,
            payload={"a": 1},
            sender_id="sender",
            timestamp=1000.0,
            nonce="abc123",
        )
        
        signing_data = msg.get_signing_data()
        
        assert isinstance(signing_data, bytes)
        # Should be deterministic
        assert signing_data == msg.get_signing_data()
        # Should not include signature
        assert b"signature" not in signing_data


# ============================================================================
# StreamingMessage Tests
# ============================================================================

class TestStreamingMessage:
    """Test StreamingMessage for chunked data transfer."""
    
    def test_create_streaming_message(self):
        """Test creating a streaming message."""
        from core.transport import StreamingMessage
        
        msg = StreamingMessage(
            stream_id="stream_001",
            seq=0,
            data=b"chunk data",
            eof=False,
        )
        
        assert msg.stream_id == "stream_001"
        assert msg.seq == 0
        assert msg.data == b"chunk data"
        assert msg.eof is False
    
    def test_to_payload(self):
        """Test converting to payload dict."""
        from core.transport import StreamingMessage
        
        msg = StreamingMessage(
            stream_id="s1",
            seq=5,
            data=b"hello",
            eof=True,
            metadata={"key": "value"},
        )
        
        payload = msg.to_payload()
        
        assert payload["stream_id"] == "s1"
        assert payload["seq"] == 5
        assert payload["eof"] is True
        assert payload["meta"] == {"key": "value"}
        # Data should be base64 encoded
        assert base64.b64decode(payload["data"]) == b"hello"
    
    def test_from_payload(self):
        """Test creating from payload dict."""
        from core.transport import StreamingMessage
        
        payload = {
            "stream_id": "test",
            "seq": 3,
            "data": base64.b64encode(b"test data").decode("ascii"),
            "eof": False,
            "meta": {"info": "test"},
        }
        
        msg = StreamingMessage.from_payload(payload)
        
        assert msg.stream_id == "test"
        assert msg.seq == 3
        assert msg.data == b"test data"
        assert msg.eof is False
        assert msg.metadata == {"info": "test"}
    
    def test_from_payload_empty_data(self):
        """Test handling empty data in payload."""
        from core.transport import StreamingMessage
        
        payload = {"stream_id": "s", "seq": 0}
        msg = StreamingMessage.from_payload(payload)
        
        assert msg.data == b""
    
    def test_to_message(self):
        """Test converting to Message wrapper."""
        from core.transport import StreamingMessage, MessageType
        
        smsg = StreamingMessage(
            stream_id="s1",
            seq=0,
            data=b"data",
        )
        
        msg = smsg.to_message(sender_id="sender")
        
        assert msg.type == MessageType.STREAM
        assert msg.sender_id == "sender"
        assert "stream_id" in msg.payload
    
    def test_from_message(self):
        """Test extracting from Message."""
        from core.transport import StreamingMessage, Message, MessageType
        
        original = StreamingMessage(
            stream_id="stream",
            seq=1,
            data=b"bytes",
            eof=False,
        )
        
        msg = original.to_message("sender")
        restored = StreamingMessage.from_message(msg)
        
        assert restored.stream_id == original.stream_id
        assert restored.seq == original.seq
        assert restored.data == original.data
    
    def test_chunk_bytes(self):
        """Test splitting bytes into chunks."""
        from core.transport import StreamingMessage
        
        data = b"A" * 100
        chunks = StreamingMessage.chunk_bytes(
            data,
            stream_id="test",
            chunk_size=30,
        )
        
        assert len(chunks) == 4  # 30 + 30 + 30 + 10
        assert chunks[0].seq == 0
        assert chunks[1].seq == 1
        assert chunks[-1].eof is True
        
        # Reconstruct data
        reconstructed = b"".join(c.data for c in chunks)
        assert reconstructed == data
    
    def test_chunk_bytes_empty(self):
        """Test chunking empty data."""
        from core.transport import StreamingMessage
        
        chunks = StreamingMessage.chunk_bytes(
            b"",
            stream_id="empty",
            eof=True,
        )
        
        assert len(chunks) == 1
        assert chunks[0].data == b""
        assert chunks[0].eof is True
    
    def test_chunk_bytes_exact_multiple(self):
        """Test chunking when data is exact multiple of chunk size."""
        from core.transport import StreamingMessage
        
        data = b"X" * 30
        chunks = StreamingMessage.chunk_bytes(data, "s", chunk_size=10)
        
        assert len(chunks) == 3
        assert all(len(c.data) == 10 for c in chunks)
        assert chunks[-1].eof is True
    
    def test_chunk_bytes_custom_start_seq(self):
        """Test chunking with custom starting sequence number."""
        from core.transport import StreamingMessage
        
        chunks = StreamingMessage.chunk_bytes(
            b"data",
            "s",
            start_seq=100,
        )
        
        assert chunks[0].seq == 100


class TestStreamingBuffer:
    """Test StreamingBuffer for reassembling streams."""
    
    def test_add_in_order(self):
        """Test adding messages in order."""
        from core.transport import StreamingMessage, StreamingBuffer
        
        buffer = StreamingBuffer()
        
        msg0 = StreamingMessage("s1", 0, b"first", eof=False)
        ready, finished = buffer.add(msg0)
        assert ready == [b"first"]
        assert finished is False
        
        msg1 = StreamingMessage("s1", 1, b"second", eof=True)
        ready, finished = buffer.add(msg1)
        assert ready == [b"second"]
        assert finished is True
    
    def test_add_out_of_order(self):
        """Test adding messages out of order."""
        from core.transport import StreamingMessage, StreamingBuffer
        
        buffer = StreamingBuffer()
        
        # Add seq 1 first
        msg1 = StreamingMessage("s1", 1, b"second", eof=False)
        ready, _ = buffer.add(msg1)
        assert ready == []  # Can't deliver yet
        
        # Add seq 0
        msg0 = StreamingMessage("s1", 0, b"first", eof=False)
        ready, _ = buffer.add(msg0)
        assert ready == [b"first", b"second"]  # Both delivered
    
    def test_duplicate_seq_ignored(self):
        """Test that duplicate sequence numbers are ignored."""
        from core.transport import StreamingMessage, StreamingBuffer
        
        buffer = StreamingBuffer()
        
        msg0 = StreamingMessage("s1", 0, b"data", eof=False)
        buffer.add(msg0)
        
        # Add same seq again
        msg0_dup = StreamingMessage("s1", 0, b"duplicate", eof=False)
        ready, _ = buffer.add(msg0_dup)
        assert ready == []  # Ignored
    
    def test_multiple_streams(self):
        """Test handling multiple independent streams."""
        from core.transport import StreamingMessage, StreamingBuffer
        
        buffer = StreamingBuffer()
        
        # Stream 1
        buffer.add(StreamingMessage("s1", 0, b"s1_0"))
        
        # Stream 2
        buffer.add(StreamingMessage("s2", 0, b"s2_0"))
        
        # Both should work independently
        ready1, _ = buffer.add(StreamingMessage("s1", 1, b"s1_1", eof=True))
        assert ready1 == [b"s1_1"]


# ============================================================================
# Crypto Tests
# ============================================================================

class TestCrypto:
    """Test Crypto class - cryptographic operations."""
    
    def test_create_crypto(self):
        """Test creating Crypto instance."""
        from core.transport import Crypto
        
        crypto = Crypto()
        
        assert crypto.signing_key is not None
        assert crypto.verify_key is not None
        assert crypto.private_key is not None
        assert crypto.public_key is not None
        assert crypto.node_id is not None
        assert len(crypto.node_id) == 44  # Base64 encoded 32-byte key
    
    def test_crypto_uniqueness(self):
        """Test that each Crypto instance has unique keys."""
        from core.transport import Crypto
        
        c1 = Crypto()
        c2 = Crypto()
        
        assert c1.node_id != c2.node_id
        assert bytes(c1.signing_key) != bytes(c2.signing_key)
    
    def test_from_seed(self):
        """Test creating Crypto from seed."""
        from core.transport import Crypto
        import os
        
        seed = os.urandom(32)
        c1 = Crypto.from_seed(seed)
        c2 = Crypto.from_seed(seed)
        
        # Same seed should produce same keys
        assert c1.node_id == c2.node_id
        assert bytes(c1.signing_key) == bytes(c2.signing_key)
    
    def test_from_seed_invalid_length(self):
        """Test that invalid seed length raises error."""
        from core.transport import Crypto
        
        with pytest.raises(ValueError, match="32 bytes"):
            Crypto.from_seed(b"too_short")
    
    def test_export_import_identity(self):
        """Test exporting and importing identity."""
        from core.transport import Crypto
        
        original = Crypto()
        exported = original.export_identity()
        
        assert isinstance(exported, bytes)
        assert len(exported) == 32  # Ed25519 seed size
        
        restored = Crypto.import_identity(exported)
        
        assert restored.node_id == original.node_id
    
    def test_sign_message(self):
        """Test message signing."""
        from core.transport import Crypto, Message, MessageType
        
        crypto = Crypto()
        msg = Message(
            type=MessageType.PING,
            payload={},
            sender_id=crypto.node_id,
        )
        
        signed = crypto.sign_message(msg)
        
        assert signed.signature is not None
        assert signed.nonce is not None
    
    def test_verify_signature_valid(self):
        """Test verifying valid signature."""
        from core.transport import Crypto, Message, MessageType
        
        crypto = Crypto()
        msg = Message(
            type=MessageType.DATA,
            payload={"test": "data"},
            sender_id=crypto.node_id,
        )
        
        signed = crypto.sign_message(msg)
        is_valid = crypto.verify_signature(signed)
        
        assert is_valid is True
    
    def test_verify_signature_invalid_no_signature(self):
        """Test that message without signature fails verification."""
        from core.transport import Crypto, Message, MessageType
        
        crypto = Crypto()
        msg = Message(
            type=MessageType.PING,
            payload={},
            sender_id=crypto.node_id,
            signature=None,
        )
        
        is_valid = crypto.verify_signature(msg)
        assert is_valid is False
    
    def test_verify_signature_tampered_payload(self):
        """Test that tampered message fails verification."""
        from core.transport import Crypto, Message, MessageType
        
        crypto = Crypto()
        msg = Message(
            type=MessageType.DATA,
            payload={"original": True},
            sender_id=crypto.node_id,
        )
        
        signed = crypto.sign_message(msg)
        
        # Tamper with payload
        signed.payload = {"tampered": True}
        
        is_valid = crypto.verify_signature(signed)
        assert is_valid is False
    
    def test_verify_signature_wrong_sender(self):
        """Test verification with wrong sender_id."""
        from core.transport import Crypto, Message, MessageType
        
        crypto1 = Crypto()
        crypto2 = Crypto()
        
        msg = Message(
            type=MessageType.DATA,
            payload={},
            sender_id=crypto1.node_id,
        )
        
        # Sign with crypto1
        signed = crypto1.sign_message(msg)
        
        # Change sender_id to crypto2
        signed.sender_id = crypto2.node_id
        
        # Should fail - signature doesn't match new sender
        is_valid = crypto1.verify_signature(signed)
        assert is_valid is False
    
    def test_encrypt_decrypt_payload(self):
        """Test payload encryption and decryption."""
        from core.transport import Crypto
        
        sender = Crypto()
        receiver = Crypto()
        
        original_payload = {"secret": "message", "number": 42}
        
        # Encrypt for receiver
        encrypted = sender.encrypt_payload(
            original_payload,
            receiver.public_key
        )
        
        assert isinstance(encrypted, str)
        assert "secret" not in encrypted
        
        # Decrypt
        decrypted = receiver.decrypt_payload(
            encrypted,
            sender.public_key
        )
        
        assert decrypted == original_payload
    
    def test_encrypt_decrypt_wrong_key(self):
        """Test that decryption with wrong key fails."""
        from core.transport import Crypto
        from nacl.exceptions import CryptoError
        
        sender = Crypto()
        receiver = Crypto()
        wrong_receiver = Crypto()
        
        encrypted = sender.encrypt_payload(
            {"data": "test"},
            receiver.public_key
        )
        
        # Try to decrypt with wrong key
        with pytest.raises(CryptoError):
            wrong_receiver.decrypt_payload(encrypted, sender.public_key)
    
    def test_public_key_from_id(self):
        """Test getting PublicKey from node_id."""
        from core.transport import Crypto
        
        crypto = Crypto()
        
        pub_key = Crypto.public_key_from_id(crypto.node_id)
        
        assert isinstance(pub_key, PublicKey)
        # Should be usable for encryption
        assert pub_key is not None


# ============================================================================
# TrafficMasker Tests
# ============================================================================

class TestTrafficMasker:
    """Test HTTP traffic masking."""
    
    def test_mask_as_http_request(self):
        """Test masking message as HTTP request."""
        from core.transport import Message, MessageType, TrafficMasker
        
        msg = Message(
            type=MessageType.PING,
            payload={},
            sender_id="sender",
        )
        
        masked = TrafficMasker.mask_as_http_request(msg, "api.example.com")
        
        assert isinstance(masked, bytes)
        assert b"POST /api/v1/sync HTTP/1.1" in masked
        assert b"Host: api.example.com" in masked
        assert b"Content-Type: application/json" in masked
    
    def test_mask_as_http_response(self):
        """Test masking message as HTTP response."""
        from core.transport import Message, MessageType, TrafficMasker
        
        msg = Message(
            type=MessageType.PONG,
            payload={"ok": True},
            sender_id="responder",
        )
        
        masked = TrafficMasker.mask_as_http_response(msg)
        
        assert isinstance(masked, bytes)
        assert b"HTTP/1.1 200 OK" in masked
        assert b"Content-Type: application/json" in masked
    
    def test_unmask_from_http_request(self):
        """Test unmasking from HTTP request."""
        from core.transport import Message, MessageType, TrafficMasker
        
        original = Message(
            type=MessageType.DATA,
            payload={"key": "value"},
            sender_id="client",
            timestamp=1000.0,
        )
        
        masked = TrafficMasker.mask_as_http_request(original)
        unmasked = TrafficMasker.unmask_from_http(masked)
        
        assert unmasked is not None
        assert unmasked.type == original.type
        assert unmasked.payload == original.payload
        assert unmasked.sender_id == original.sender_id
    
    def test_unmask_from_http_response(self):
        """Test unmasking from HTTP response."""
        from core.transport import Message, MessageType, TrafficMasker
        
        original = Message(
            type=MessageType.SERVICE_RESPONSE,
            payload={"result": "success"},
            sender_id="server",
            timestamp=2000.0,
        )
        
        masked = TrafficMasker.mask_as_http_response(original)
        unmasked = TrafficMasker.unmask_from_http(masked)
        
        assert unmasked is not None
        assert unmasked.type == original.type
        assert unmasked.payload == original.payload
    
    def test_unmask_invalid_data(self):
        """Test unmasking returns None for invalid data."""
        from core.transport import TrafficMasker
        
        # No body
        result = TrafficMasker.unmask_from_http(b"HTTP/1.1 200 OK")
        assert result is None
        
        # Invalid JSON
        result = TrafficMasker.unmask_from_http(b"HTTP/1.1 200 OK\r\n\r\nnot json")
        assert result is None
        
        # Missing data field
        result = TrafficMasker.unmask_from_http(b"HTTP/1.1 200 OK\r\n\r\n{\"other\":1}")
        assert result is None
    
    def test_is_http_masked_true(self):
        """Test detection of HTTP masked data."""
        from core.transport import TrafficMasker
        
        assert TrafficMasker.is_http_masked(b"POST /api/v1/test HTTP/1.1\r\n") is True
        assert TrafficMasker.is_http_masked(b"GET /api/data HTTP/1.1\r\n") is True
        assert TrafficMasker.is_http_masked(b"HTTP/1.1 200 OK\r\n") is True
    
    def test_is_http_masked_false(self):
        """Test detection of non-HTTP data."""
        from core.transport import TrafficMasker
        
        assert TrafficMasker.is_http_masked(b"ZE\x01\x01") is False  # Binary
        assert TrafficMasker.is_http_masked(b"{\"json\":1}") is False
        assert TrafficMasker.is_http_masked(b"\x00\x00\x00\x10") is False
    
    def test_is_http_masked_binary_utf8_error(self):
        """Test handling of non-UTF8 data."""
        from core.transport import TrafficMasker
        
        # Invalid UTF-8 sequence
        assert TrafficMasker.is_http_masked(b"\xff\xfe\x00\x01") is False


# ============================================================================
# SimpleTransport Tests
# ============================================================================

class TestSimpleTransport:
    """Test SimpleTransport (legacy JSON transport)."""
    
    def test_pack(self):
        """Test packing message."""
        from core.transport import Message, MessageType, SimpleTransport
        
        msg = Message(
            type=MessageType.PING,
            payload={},
            sender_id="s",
            timestamp=0,
        )
        
        packed = SimpleTransport.pack(msg)
        
        assert isinstance(packed, bytes)
        # First 4 bytes are length
        length = int.from_bytes(packed[:4], "big")
        assert length == len(packed) - 4
    
    def test_unpack_length(self):
        """Test unpacking length from header."""
        from core.transport import SimpleTransport
        
        # Create header with length 100
        header = (100).to_bytes(4, "big")
        length = SimpleTransport.unpack_length(header)
        
        assert length == 100
    
    def test_unpack_length_short_header(self):
        """Test error on too short header."""
        from core.transport import SimpleTransport
        
        with pytest.raises(ValueError, match="too short"):
            SimpleTransport.unpack_length(b"\x00\x00")
    
    def test_unpack(self):
        """Test unpacking message."""
        from core.transport import Message, MessageType, SimpleTransport
        
        original = Message(
            type=MessageType.DATA,
            payload={"test": 123},
            sender_id="sender",
            timestamp=1234.5,
        )
        
        packed = SimpleTransport.pack(original)
        # Skip length prefix
        payload_data = packed[4:]
        
        unpacked = SimpleTransport.unpack(payload_data)
        
        assert unpacked.type == original.type
        assert unpacked.payload == original.payload
        assert unpacked.sender_id == original.sender_id
    
    def test_pack_unpack_roundtrip(self):
        """Test full pack/unpack cycle."""
        from core.transport import Message, MessageType, SimpleTransport
        
        original = Message(
            type=MessageType.SERVICE_REQUEST,
            payload={"service": "compute", "data": [1, 2, 3]},
            sender_id="client",
            timestamp=time.time(),
            signature="sig",
            nonce="nc",
        )
        
        packed = SimpleTransport.pack(original)
        length = SimpleTransport.unpack_length(packed[:4])
        unpacked = SimpleTransport.unpack(packed[4:4+length])
        
        assert unpacked.type == original.type
        assert unpacked.payload == original.payload
    
    def test_get_payload_size(self):
        """Test getting payload size."""
        from core.transport import SimpleTransport
        
        data = b"test payload data"
        size = SimpleTransport.get_payload_size(data)
        
        assert size == len(data)


# ============================================================================
# BinaryTransport Tests
# ============================================================================

class TestBinaryTransport:
    """Test BinaryTransport (production wire protocol)."""
    
    def test_create_binary_transport(self):
        """Test creating BinaryTransport."""
        from core.transport import Crypto, BinaryTransport
        
        crypto = Crypto()
        transport = BinaryTransport(crypto)
        
        assert transport.crypto is crypto
        assert transport.codec is not None
    
    def test_is_binary_protocol_true(self):
        """Test detecting binary protocol."""
        from core.transport import BinaryTransport
        
        assert BinaryTransport.is_binary_protocol(b"ZE\x01\x01") is True
        assert BinaryTransport.is_binary_protocol(b"ZE" + b"\x00" * 96) is True
    
    def test_is_binary_protocol_false(self):
        """Test detecting non-binary data."""
        from core.transport import BinaryTransport
        
        assert BinaryTransport.is_binary_protocol(b"XX\x01\x01") is False
        assert BinaryTransport.is_binary_protocol(b"Z") is False
        assert BinaryTransport.is_binary_protocol(b"") is False
    
    def test_message_type_from_legacy(self):
        """Test converting legacy MessageType to WireMessageType."""
        from core.transport import BinaryTransport, MessageType
        from core.wire import WireMessageType
        
        # Test mappings
        assert BinaryTransport.message_type_from_legacy(MessageType.PING) == WireMessageType.PING
        assert BinaryTransport.message_type_from_legacy(MessageType.PONG) == WireMessageType.PONG
        assert BinaryTransport.message_type_from_legacy(MessageType.DATA) == WireMessageType.DATA
        assert BinaryTransport.message_type_from_legacy(MessageType.IOU) == WireMessageType.IOU
    
    def test_pack_unpack(self):
        """Test binary pack and unpack."""
        from core.transport import Crypto, BinaryTransport
        from core.wire import WireMessageType
        
        sender = Crypto()
        receiver = Crypto()
        
        transport = BinaryTransport(sender)
        
        payload = b"test payload"
        packed = transport.pack(
            WireMessageType.DATA,
            payload,
            recipient_public_key=receiver.public_key,
        )
        
        assert isinstance(packed, bytes)
        assert packed[:2] == b"ZE"  # Magic
        
        # Unpack
        receiver_transport = BinaryTransport(receiver)
        message = receiver_transport.unpack(
            packed,
            sender_verify_key=sender.verify_key,
            sender_public_key=sender.public_key,
        )
        
        assert message.msg_type == WireMessageType.DATA
    
    def test_unpack_header(self):
        """Test unpacking only header."""
        from core.transport import Crypto, BinaryTransport
        from core.wire import WireMessageType
        
        crypto = Crypto()
        transport = BinaryTransport(crypto)
        
        packed = transport.pack(WireMessageType.PING, b"")
        
        header = transport.unpack_header(packed[:98])
        
        assert header.magic == b"ZE"
        assert header.version == 1
        assert header.msg_type == WireMessageType.PING


# ============================================================================
# BlockingTransport Tests  
# ============================================================================

class TestBlockingTransport:
    """Test BlockingTransport with ledger integration."""
    
    @pytest.mark.asyncio
    async def test_create_blocking_transport(self, ledger):
        """Test creating BlockingTransport."""
        from core.transport import BlockingTransport
        
        transport = BlockingTransport(ledger)
        assert transport.ledger is ledger
    
    @pytest.mark.asyncio
    async def test_can_send_to_allowed(self, ledger, random_peer_id):
        """Test that sending is allowed for good peers."""
        from core.transport import BlockingTransport
        
        transport = BlockingTransport(ledger)
        peer_id = random_peer_id()
        
        can_send, reason = await transport.can_send_to(peer_id)
        assert can_send is True
    
    @pytest.mark.asyncio
    async def test_is_blocked(self, ledger, random_peer_id):
        """Test synchronous blocked check."""
        from core.transport import BlockingTransport
        
        transport = BlockingTransport(ledger)
        peer_id = random_peer_id()
        
        # New peer should not be blocked
        assert transport.is_blocked(peer_id) is False
    
    @pytest.mark.asyncio
    async def test_pack_with_accounting(self, ledger, random_peer_id):
        """Test packing with ledger accounting."""
        from core.transport import BlockingTransport, Message, MessageType
        
        transport = BlockingTransport(ledger)
        peer_id = random_peer_id()
        
        msg = Message(
            type=MessageType.DATA,
            payload={"test": "data"},
            sender_id="me",
        )
        
        data, size, blocked, reason = await transport.pack_with_accounting(msg, peer_id)
        
        assert len(data) > 0
        assert size > 0
        assert blocked is False
        assert reason == "OK"
    
    @pytest.mark.asyncio
    async def test_unpack_with_accounting(self, ledger, random_peer_id):
        """Test unpacking with ledger accounting."""
        from core.transport import BlockingTransport, Message, MessageType, SimpleTransport
        
        transport = BlockingTransport(ledger)
        peer_id = random_peer_id()
        
        # Create packed message
        original = Message(
            type=MessageType.PONG,
            payload={"response": True},
            sender_id=peer_id,
        )
        packed = SimpleTransport.pack(original)
        payload = packed[4:]  # Skip length
        
        # Record initial balance
        initial_balance = await ledger.get_balance(peer_id)
        
        # Unpack with accounting
        unpacked = await transport.unpack_with_accounting(payload, peer_id)
        
        assert unpacked.type == original.type
        
        # Balance should have changed (debt recorded)
        new_balance = await ledger.get_balance(peer_id)
        assert new_balance < initial_balance  # We owe them now

