"""
Human Link unit tests.
"""

from __future__ import annotations

from agents.bridge.human_link import HumanLinkAgent, HumanRequest, RequestStatus


def test_human_link_init():
    link = HumanLinkAgent(bot_token="")
    assert link.is_human_online() is False


def test_human_request():
    req = HumanRequest(
        request_id="test123",
        message="Test request",
        options=["Yes", "No"],
    )

    assert req.status == RequestStatus.PENDING
    assert len(req.options) == 2

