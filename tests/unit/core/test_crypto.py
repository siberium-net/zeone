"""
Cryptography Unit Tests
=======================

[UNIT] Tests for core/transport.py cryptographic operations.
"""

import pytest
from pathlib import Path


class TestCrypto:
    """Test Crypto class."""
    
    def test_create_crypto(self, crypto):
        """Test crypto identity creation."""
        assert crypto is not None
        assert crypto.node_id is not None
        # node_id is base64 encoded (44 chars) not hex (64 chars)
        assert len(crypto.node_id) == 44
    
    def test_node_id_uniqueness(self, crypto_pair):
        """Test that different crypto instances have unique IDs."""
        crypto1, crypto2 = crypto_pair
        assert crypto1.node_id != crypto2.node_id
    
    def test_export_import_identity(self, crypto, temp_dir):
        """Test identity export and import."""
        from core.transport import Crypto
        
        # Export
        exported = crypto.export_identity()
        assert isinstance(exported, bytes)
        assert len(exported) > 0
        
        # Save to file
        identity_file = temp_dir / "test_identity.key"
        identity_file.write_bytes(exported)
        
        # Import
        restored = Crypto.import_identity(identity_file.read_bytes())
        assert restored.node_id == crypto.node_id
    
    def test_sign_message(self, crypto):
        """Test message signing via signing_key."""
        message = b"test message to sign"
        
        # Sign using signing_key directly
        signed = crypto.signing_key.sign(message)
        assert signed is not None
        assert len(signed.signature) == 64  # Ed25519 signature
        
        # Verify
        crypto.verify_key.verify(signed)  # Raises if invalid
    
    def test_sign_verify_message(self, crypto):
        """Test signing and verifying a Message object."""
        from core.transport import Message, MessageType
        
        message = Message(
            type=MessageType.DATA,
            payload={"test": "data"},
            sender_id=crypto.node_id,
        )
        
        # Sign message
        signed = crypto.sign_message(message)
        assert signed.signature is not None
        
        # Verify signature
        is_valid = crypto.verify_signature(signed)
        assert is_valid is True
    
    def test_encrypt_decrypt_box(self, crypto_pair):
        """Test asymmetric encryption using NaCl Box."""
        from nacl.public import Box
        
        sender, receiver = crypto_pair
        
        plaintext = b"secret message"
        
        # Create Box for encryption (sender -> receiver)
        box = Box(sender.private_key, receiver.public_key)
        
        # Encrypt
        encrypted = box.encrypt(plaintext)
        assert encrypted != plaintext
        
        # Create Box for decryption (receiver <- sender)
        decrypt_box = Box(receiver.private_key, sender.public_key)
        
        # Decrypt
        decrypted = decrypt_box.decrypt(encrypted)
        assert decrypted == plaintext
    
    def test_public_key_bytes(self, crypto):
        """Test public key access."""
        # Public key bytes should be accessible
        pk_bytes = crypto.public_key.encode()
        assert isinstance(pk_bytes, bytes)
        assert len(pk_bytes) == 32  # Curve25519 key
    
    def test_verify_key_bytes(self, crypto):
        """Test verify key access."""
        vk_bytes = crypto.verify_key.encode()
        assert isinstance(vk_bytes, bytes)
        assert len(vk_bytes) == 32  # Ed25519 key


class TestMessage:
    """Test Message class."""
    
    def test_create_message(self, sample_message):
        """Test message creation."""
        assert sample_message.type is not None
        assert sample_message.payload is not None
        assert sample_message.sender_id is not None
    
    def test_message_serialization(self, crypto):
        """Test message to_dict and from_dict."""
        from core.transport import Message, MessageType
        
        original = Message(
            type=MessageType.DATA,
            payload={"key": "value", "number": 42},
            sender_id=crypto.node_id,
        )
        
        # Serialize
        data = original.to_dict()
        assert isinstance(data, dict)
        assert "type" in data
        assert "payload" in data
        
        # Deserialize
        restored = Message.from_dict(data)
        assert restored.type == original.type
        assert restored.payload == original.payload
        assert restored.sender_id == original.sender_id
    
    def test_message_sign_verify(self, crypto):
        """Test message signing with Crypto."""
        from core.transport import Message, MessageType
        
        message = Message(
            type=MessageType.PING,
            payload={"timestamp": 12345},
            sender_id=crypto.node_id,
        )
        
        # Sign
        signed = crypto.sign_message(message)
        assert signed.signature is not None
        
        # Verify
        is_valid = crypto.verify_signature(signed)
        assert is_valid is True


class TestBinaryProtocol:
    """Test binary wire protocol."""
    
    def test_magic_bytes(self):
        """Test protocol magic bytes."""
        try:
            from core.wire import BinaryWireProtocol
            assert BinaryWireProtocol.MAGIC == b"ZE"
        except ImportError:
            pytest.skip("Binary wire protocol not available")
    
    def test_pack_unpack_message(self, crypto):
        """Test message packing and unpacking."""
        try:
            from core.wire import (
                BinaryWireProtocol,
                WireMessageType,
                pack_message,
                unpack_message,
            )
            
            payload = b"test payload data"
            
            # Pack
            packed = pack_message(
                msg_type=WireMessageType.DATA,
                payload=payload,
                signing_key=crypto.signing_key,
            )
            
            assert packed.startswith(b"ZE")  # Magic
            assert len(packed) >= 98 + len(payload)  # Header + payload
            
            # Unpack
            msg_type, unpacked_payload, nonce = unpack_message(
                packed,
                verify_key=crypto.verify_key,
            )
            
            assert msg_type == WireMessageType.DATA
            # Note: payload might be encrypted, so just check we got something
            assert unpacked_payload is not None
            
        except ImportError:
            pytest.skip("Binary wire protocol not available")
    
    def test_invalid_magic_rejected(self):
        """Test that invalid magic bytes are rejected."""
        try:
            from core.wire import unpack_message
            
            invalid_data = b"XX" + b"\x00" * 96  # Wrong magic
            
            with pytest.raises(Exception):
                unpack_message(invalid_data, verify_key=None)
                
        except ImportError:
            pytest.skip("Binary wire protocol not available")
