"""
Integration test for tensor handoff over BinaryWireProtocol.
"""

import asyncio

import pytest

from core.transport import Crypto
from core.wire import WireCodec, WireMessageType, WireHeader, HEADER_SIZE


class DummyPartA:
    """Head: multiplies input by 2 and sends to tail."""

    def __init__(self, codec: WireCodec, peer_verify):
        self.codec = codec
        self.peer_verify = peer_verify  # tail verify key

    def forward(self, tensor):
        doubled = [x * 2 for x in tensor]
        payload = bytes(repr(doubled), encoding="utf-8")
        message = self.codec.encode(
            msg_type=WireMessageType.TENSOR_DATA,
            payload=payload,
            recipient_public_key=None,
        )
        return message

    def receive(self, raw_bytes):
        header = WireHeader.unpack(raw_bytes[:HEADER_SIZE])
        assert header.msg_type == WireMessageType.TENSOR_DATA
        decoded = self.codec.decode(raw_bytes, self.peer_verify, None)
        result = eval(decoded.payload.decode("utf-8"))
        return result


class DummyPartB:
    """Tail: adds 1 and returns back."""

    def __init__(self, codec: WireCodec, peer_verify):
        self.codec = codec
        self.peer_verify = peer_verify  # head verify key

    def handle(self, raw_bytes):
        decoded = self.codec.decode(raw_bytes, self.peer_verify, None)
        incoming = eval(decoded.payload.decode("utf-8"))
        out = [x + 1 for x in incoming]
        payload = bytes(repr(out), encoding="utf-8")
        message = self.codec.encode(
            msg_type=WireMessageType.TENSOR_DATA,
            payload=payload,
            recipient_public_key=None,
        )
        return message


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_neuro_link_tensor_flow():
    """
    Simulate split model: head (x*2) -> tail (+1) over wire protocol.
    """
    head_crypto = Crypto()
    tail_crypto = Crypto()
    head_codec = WireCodec(head_crypto.signing_key, head_crypto.private_key, head_crypto.verify_key)
    tail_codec = WireCodec(tail_crypto.signing_key, tail_crypto.private_key, tail_crypto.verify_key)

    part_a = DummyPartA(codec=head_codec, peer_verify=tail_crypto.verify_key)
    part_b = DummyPartB(codec=tail_codec, peer_verify=head_crypto.verify_key)

    # Head sends activations
    tensor_in = [1.0, 2.0]
    msg_to_tail = part_a.forward(tensor_in)

    # Tail processes and responds
    msg_to_head = part_b.handle(msg_to_tail)

    # Head receives final output
    result = part_a.receive(msg_to_head)
    assert result == [3.0, 5.0]
    # ensure header was tensor type
    header_back = WireHeader.unpack(msg_to_head[:HEADER_SIZE])
    assert header_back.msg_type == WireMessageType.TENSOR_DATA
