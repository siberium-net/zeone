"""
Ledger Unit Tests - 100% Coverage
=================================

[CRITICAL] Complete coverage for economy/ledger.py - financial operations.
"""

import pytest
import pytest_asyncio
import time
import json
import os
import tempfile
from pathlib import Path


# ============================================================================
# Transaction Tests
# ============================================================================

class TestTransaction:
    """Test Transaction dataclass."""
    
    def test_create_transaction(self):
        """Test transaction creation."""
        from economy.ledger import Transaction
        
        tx = Transaction(
            id="tx123",
            from_id="alice",
            to_id="bob",
            amount=100.0,
            timestamp=time.time(),
            signature="sig",
            tx_type="transfer",
        )
        
        assert tx.id == "tx123"
        assert tx.from_id == "alice"
        assert tx.to_id == "bob"
        assert tx.amount == 100.0
    
    def test_transaction_to_dict(self):
        """Test transaction serialization."""
        from economy.ledger import Transaction
        
        tx = Transaction(
            id="tx1",
            from_id="a",
            to_id="b",
            amount=50.0,
            timestamp=1000.0,
            signature="s",
            metadata={"note": "test"},
        )
        
        d = tx.to_dict()
        
        assert d["id"] == "tx1"
        assert d["amount"] == 50.0
        assert d["metadata"] == {"note": "test"}
    
    def test_transaction_from_dict(self):
        """Test transaction deserialization."""
        from economy.ledger import Transaction
        
        data = {
            "id": "tx2",
            "from_id": "x",
            "to_id": "y",
            "amount": 25.0,
            "timestamp": 2000.0,
            "signature": "sig2",
            "tx_type": "debt",
            "metadata": {"key": "value"},
        }
        
        tx = Transaction.from_dict(data)
        
        assert tx.id == "tx2"
        assert tx.tx_type == "debt"
        assert tx.metadata == {"key": "value"}
    
    def test_transaction_signing_data(self):
        """Test signing data generation."""
        from economy.ledger import Transaction
        
        tx = Transaction(
            id="ignore",
            from_id="sender",
            to_id="receiver",
            amount=100.0,
            timestamp=12345.0,
            signature="also_ignore",
        )
        
        data = tx.get_signing_data()
        
        assert isinstance(data, bytes)
        # id and signature should not be in signing data
        assert b"ignore" not in data or b"id" not in data
    
    def test_transaction_compute_id(self):
        """Test ID computation from signing data."""
        from economy.ledger import Transaction
        
        signing_data = b"test data for hashing"
        tx_id = Transaction.compute_id(signing_data)
        
        assert isinstance(tx_id, str)
        assert len(tx_id) == 64  # SHA256 hex


# ============================================================================
# IOU Tests
# ============================================================================

class TestIOU:
    """Test IOU dataclass."""
    
    def test_create_iou(self):
        """Test IOU creation."""
        from economy.ledger import IOU
        
        iou = IOU(
            id="iou1",
            debtor_id="debtor",
            creditor_id="creditor",
            amount=500.0,
            created_at=time.time(),
            expires_at=None,
            signature="sig",
        )
        
        assert iou.id == "iou1"
        assert iou.amount == 500.0
        assert iou.redeemed is False
    
    def test_iou_to_dict(self):
        """Test IOU serialization."""
        from economy.ledger import IOU
        
        iou = IOU(
            id="iou2",
            debtor_id="d",
            creditor_id="c",
            amount=100.0,
            created_at=1000.0,
            expires_at=2000.0,
            signature="s",
            redeemed=True,
            redeemed_at=1500.0,
        )
        
        d = iou.to_dict()
        
        assert d["redeemed"] is True
        assert d["redeemed_at"] == 1500.0
    
    def test_iou_from_dict(self):
        """Test IOU deserialization."""
        from economy.ledger import IOU
        
        data = {
            "id": "iou3",
            "debtor_id": "db",
            "creditor_id": "cr",
            "amount": 200.0,
            "created_at": 3000.0,
            "expires_at": 4000.0,
            "signature": "sig3",
            "redeemed": False,
        }
        
        iou = IOU.from_dict(data)
        
        assert iou.id == "iou3"
        assert iou.expires_at == 4000.0
    
    def test_iou_is_expired_not_set(self):
        """Test that IOU without expiry is never expired."""
        from economy.ledger import IOU
        
        iou = IOU(
            id="i",
            debtor_id="d",
            creditor_id="c",
            amount=10.0,
            created_at=1.0,
            expires_at=None,
            signature="s",
        )
        
        assert iou.is_expired() is False
    
    def test_iou_is_expired_future(self):
        """Test IOU with future expiry is not expired."""
        from economy.ledger import IOU
        
        iou = IOU(
            id="i",
            debtor_id="d",
            creditor_id="c",
            amount=10.0,
            created_at=1.0,
            expires_at=time.time() + 3600,  # 1 hour from now
            signature="s",
        )
        
        assert iou.is_expired() is False
    
    def test_iou_is_expired_past(self):
        """Test IOU with past expiry is expired."""
        from economy.ledger import IOU
        
        iou = IOU(
            id="i",
            debtor_id="d",
            creditor_id="c",
            amount=10.0,
            created_at=1.0,
            expires_at=time.time() - 1,  # 1 second ago
            signature="s",
        )
        
        assert iou.is_expired() is True
    
    def test_iou_is_valid(self):
        """Test IOU validity check."""
        from economy.ledger import IOU
        
        # Valid IOU
        valid = IOU(
            id="v",
            debtor_id="d",
            creditor_id="c",
            amount=10.0,
            created_at=1.0,
            expires_at=time.time() + 3600,
            signature="s",
            redeemed=False,
        )
        assert valid.is_valid() is True
        
        # Redeemed IOU
        redeemed = IOU(
            id="r",
            debtor_id="d",
            creditor_id="c",
            amount=10.0,
            created_at=1.0,
            expires_at=time.time() + 3600,
            signature="s",
            redeemed=True,
        )
        assert redeemed.is_valid() is False


# ============================================================================
# TrustScore Tests
# ============================================================================

class TestTrustScoreClass:
    """Test TrustScore class methods."""
    
    def test_calculate_adjustment_known_event(self):
        """Test adjustment calculation for known events."""
        from economy.ledger import TrustScore
        
        adj = TrustScore.calculate_adjustment("successful_transfer")
        assert adj == 0.01
        
        adj = TrustScore.calculate_adjustment("failed_transfer")
        assert adj == -0.05
    
    def test_calculate_adjustment_magnitude(self):
        """Test adjustment with magnitude multiplier."""
        from economy.ledger import TrustScore
        
        adj = TrustScore.calculate_adjustment("ping_responded", magnitude=10.0)
        assert adj == 0.001 * 10.0
    
    def test_calculate_adjustment_unknown_event(self):
        """Test adjustment for unknown event returns 0."""
        from economy.ledger import TrustScore
        
        adj = TrustScore.calculate_adjustment("unknown_event")
        assert adj == 0.0
    
    def test_clamp(self):
        """Test score clamping to [0, 1]."""
        from economy.ledger import TrustScore
        
        assert TrustScore.clamp(-0.5) == 0.0
        assert TrustScore.clamp(0.5) == 0.5
        assert TrustScore.clamp(1.5) == 1.0
    
    def test_decay(self):
        """Test score decay for inactivity."""
        from economy.ledger import TrustScore
        
        original = 0.8
        
        # No decay for 0 days
        assert TrustScore.decay(original, 0.0) == original
        
        # Decay for 10 days
        decayed = TrustScore.decay(original, 10.0)
        assert decayed < original
        assert decayed > 0


# ============================================================================
# Ledger Tests
# ============================================================================

class TestLedger:
    """Test Ledger class - core financial operations."""
    
    @pytest.mark.asyncio
    async def test_create_ledger(self, ledger):
        """Test ledger creation."""
        assert ledger is not None
        assert ledger.debt_limit > 0
    
    @pytest.mark.asyncio
    async def test_get_or_create_peer(self, ledger, random_peer_id):
        """Test peer creation."""
        peer_id = random_peer_id()
        
        peer = await ledger.get_or_create_peer(peer_id)
        assert peer is not None
        assert peer["node_id"] == peer_id
        
        # Second call returns existing
        peer2 = await ledger.get_or_create_peer(peer_id)
        assert peer2 is not None
    
    @pytest.mark.asyncio
    async def test_get_or_create_peer_with_key(self, ledger, random_peer_id):
        """Test peer creation with public key."""
        peer_id = random_peer_id()
        
        peer = await ledger.get_or_create_peer(peer_id, public_key="pubkey123")
        assert peer["public_key"] == "pubkey123"
    
    @pytest.mark.asyncio
    async def test_record_debt(self, ledger, random_peer_id):
        """Test recording debt (we owe them)."""
        peer_id = random_peer_id()
        
        # Record debt
        new_balance = await ledger.record_debt(peer_id, 100)
        assert new_balance == -100  # Negative = we owe them
        
        # Record more debt
        new_balance = await ledger.record_debt(peer_id, 50)
        assert new_balance == -150
        
        # Check balance
        balance = await ledger.get_balance(peer_id)
        assert balance == -150
    
    @pytest.mark.asyncio
    async def test_record_claim(self, ledger, random_peer_id):
        """Test recording claim (they owe us)."""
        peer_id = random_peer_id()
        
        # Record claim
        new_balance = await ledger.record_claim(peer_id, 100, signature="")
        assert new_balance == 100  # Positive = they owe us
        
        # Check balance
        balance = await ledger.get_balance(peer_id)
        assert balance == 100
    
    @pytest.mark.asyncio
    async def test_record_debt_and_claim(self, ledger, random_peer_id):
        """Test combined debt and claim operations."""
        peer_id = random_peer_id()
        
        await ledger.record_debt(peer_id, 100)  # -100
        await ledger.record_claim(peer_id, 70)  # +70
        
        balance = await ledger.get_balance(peer_id)
        assert balance == -30  # Net: -100 + 70 = -30
    
    @pytest.mark.asyncio
    async def test_debt_limit_blocking(self, ledger, random_peer_id):
        """Test that exceeding debt limit blocks peer."""
        peer_id = random_peer_id()
        
        # Record debt exceeding limit
        await ledger.record_claim(peer_id, ledger.debt_limit + 100)
        
        # Check blocked status
        is_blocked = ledger.is_peer_blocked(peer_id)
        assert is_blocked is True
    
    @pytest.mark.asyncio
    async def test_not_blocked_under_limit(self, ledger, random_peer_id):
        """Test that peer under limit is not blocked."""
        peer_id = random_peer_id()
        
        await ledger.record_claim(peer_id, ledger.debt_limit // 2)
        
        is_blocked = ledger.is_peer_blocked(peer_id)
        assert is_blocked is False
    
    @pytest.mark.asyncio
    async def test_check_can_send(self, ledger, random_peer_id):
        """Test can_send check."""
        peer_id = random_peer_id()
        
        # New peer - can send
        can_send, reason = await ledger.check_can_send(peer_id)
        assert can_send is True
        assert reason == "OK"
        
        # Exceed limit
        await ledger.record_claim(peer_id, ledger.debt_limit + 1)
        
        can_send, reason = await ledger.check_can_send(peer_id)
        assert can_send is False
        assert "exceeds limit" in reason
    
    @pytest.mark.asyncio
    async def test_balance_info(self, ledger, random_peer_id):
        """Test getting balance info."""
        peer_id = random_peer_id()
        
        await ledger.record_debt(peer_id, 100)
        await ledger.record_claim(peer_id, 50)
        
        info = await ledger.get_balance_info(peer_id)
        
        assert "balance" in info
        assert info["balance"] == -50  # Net
        assert "peer_id" in info
    
    @pytest.mark.asyncio
    async def test_balance_info_unknown_peer(self, ledger, random_peer_id):
        """Test balance info for unknown peer."""
        peer_id = random_peer_id()
        
        info = await ledger.get_balance_info(peer_id)
        
        assert info["balance"] == 0.0
        assert info["total_sent"] == 0.0
        assert info["total_received"] == 0.0
    
    @pytest.mark.asyncio
    async def test_all_balances(self, ledger, random_peer_id):
        """Test getting all balances."""
        # Create multiple peers
        for i in range(3):
            peer_id = random_peer_id()
            await ledger.record_debt(peer_id, 50 * (i + 1))
        
        balances = await ledger.get_all_balances()
        assert len(balances) >= 3
        
        for b in balances:
            assert "peer_id" in b
            assert "balance" in b
    
    @pytest.mark.asyncio
    async def test_reset_balance(self, ledger, random_peer_id):
        """Test resetting balance."""
        peer_id = random_peer_id()
        
        await ledger.record_claim(peer_id, 1000)
        assert await ledger.get_balance(peer_id) == 1000
        
        await ledger.reset_balance(peer_id)
        assert await ledger.get_balance(peer_id) == 0.0
    
    @pytest.mark.asyncio
    async def test_trust_score_update(self, ledger, random_peer_id):
        """Test trust score updates."""
        peer_id = random_peer_id()
        
        # Create peer
        await ledger.get_or_create_peer(peer_id)
        
        # Get initial score
        score = await ledger.get_trust_score(peer_id)
        assert score > 0  # Default initial score
    
    @pytest.mark.asyncio
    async def test_get_trust_score_unknown_peer(self, ledger, random_peer_id):
        """Test trust score for unknown peer returns default."""
        peer_id = random_peer_id()
        score = await ledger.get_trust_score(peer_id)
        assert score == 0.5  # Default from config
    
    @pytest.mark.asyncio
    async def test_get_total_balance(self, ledger, random_peer_id):
        """Test get_total_balance for weighted trust."""
        peer_id = random_peer_id()
        
        # Record some claims (positive balance = they owe us)
        await ledger.record_claim(peer_id, 1024 * 1024 * 1024)  # 1 GB
        
        total = await ledger.get_total_balance(peer_id)
        assert total >= 1.0  # At least 1 ZEO (1 GB)
    
    @pytest.mark.asyncio
    async def test_get_total_balance_negative_ignored(self, ledger, random_peer_id):
        """Test that negative balance doesn't count in total."""
        peer_id = random_peer_id()
        
        # Record debt (negative balance)
        await ledger.record_debt(peer_id, 1024 * 1024 * 1024)
        
        total = await ledger.get_total_balance(peer_id)
        assert total == 0.0  # Negative balance ignored
    
    @pytest.mark.asyncio
    async def test_balance_claim_handshake(self, ledger, random_peer_id):
        """Test balance claim generation for handshake."""
        peer_id = random_peer_id()
        
        await ledger.record_claim(peer_id, 500)
        
        claim = await ledger.get_balance_claim(peer_id)
        
        assert "claimed_balance" in claim
        assert claim["claimed_balance"] == 500
        assert "timestamp" in claim
    
    @pytest.mark.asyncio
    async def test_reconcile_balance_agreed(self, ledger, random_peer_id):
        """Test balance reconciliation when agreed."""
        peer_id = random_peer_id()
        
        await ledger.record_claim(peer_id, 1000)
        
        # Peer claims they owe us 1000 (inverted: -1000)
        result = await ledger.reconcile_balance(peer_id, -1000)
        
        assert result["status"] == "agreed"
    
    @pytest.mark.asyncio
    async def test_reconcile_balance_disputed(self, ledger, random_peer_id):
        """Test balance reconciliation when disputed."""
        peer_id = random_peer_id()
        
        await ledger.record_claim(peer_id, 10_000_000)
        
        # Peer claims a very different amount
        result = await ledger.reconcile_balance(peer_id, 0)
        
        assert result["status"] == "disputed"
        assert "action" in result


# ============================================================================
# Knowledge Base Tests
# ============================================================================

class TestKnowledgeBase:
    """Test Ledger knowledge base operations."""
    
    @pytest.mark.asyncio
    async def test_add_knowledge_entry(self, ledger):
        """Test adding knowledge entry."""
        entry_id = await ledger.add_knowledge_entry(
            cid="QmTest123",
            path="/test/file.txt",
            summary="Test document",
            tags="test,unit",
            size=1024,
            metadata={"author": "test"},
        )
        
        assert entry_id >= 0
    
    @pytest.mark.asyncio
    async def test_get_knowledge_entries(self, ledger):
        """Test retrieving knowledge entries."""
        # Add entries
        await ledger.add_knowledge_entry(
            cid="cid1", path="/p1", summary="s1", tags="t1", size=100, metadata={}
        )
        await ledger.add_knowledge_entry(
            cid="cid2", path="/p2", summary="s2", tags="t2", size=200, metadata={}
        )
        
        entries = await ledger.get_knowledge_entries(limit=10)
        assert len(entries) >= 2


# ============================================================================
# WeightedTrustScore Tests  
# ============================================================================

class TestWeightedTrustScore:
    """Test WeightedTrustScore class from economy.trust."""
    
    def test_constants_defined(self):
        """Test that all constants are defined."""
        from economy.trust import WeightedTrustScore
        
        assert WeightedTrustScore.BASE_STAKE > 0
        assert WeightedTrustScore.DUST_LIMIT > 0
        assert 0 < WeightedTrustScore.DUST_MULTIPLIER < 1
        assert 0 < WeightedTrustScore.EMA_ALPHA < 1
        assert 0 < WeightedTrustScore.DECAY_RATE <= 1
    
    def test_calculate_effective_trust_high_stake(self):
        """Test effective trust with high stake."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore()
        
        # High behavior, high stake
        score = wts.calculate_effective_trust(behavior_score=0.9, stake=10000)
        
        assert score > 0.5
    
    def test_calculate_effective_trust_low_stake(self):
        """Test effective trust with low stake."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore()
        
        # High behavior, low stake
        score = wts.calculate_effective_trust(behavior_score=0.9, stake=1)
        
        # Should be penalized
        assert score < 0.5
    
    def test_dust_limit_penalty(self):
        """Test dust limit penalty application."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore()
        
        # Below dust limit
        score_dust = wts.calculate_effective_trust(1.0, stake=5)
        
        # Above dust limit
        score_normal = wts.calculate_effective_trust(1.0, stake=100)
        
        assert score_dust < score_normal
    
    def test_behavior_score_clamping(self):
        """Test that behavior score is clamped."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore()
        
        # Over 1.0 should be clamped
        score_over = wts.calculate_effective_trust(1.5, stake=100)
        score_one = wts.calculate_effective_trust(1.0, stake=100)
        
        assert score_over == score_one
        
        # Under 0.0 should be clamped
        score_under = wts.calculate_effective_trust(-0.5, stake=100)
        score_zero = wts.calculate_effective_trust(0.0, stake=100)
        
        assert score_under == score_zero
    
    def test_update_behavior_score_positive(self):
        """Test behavior score update for positive event."""
        from economy.trust import WeightedTrustScore, TrustEvent
        
        wts = WeightedTrustScore()
        
        new_score = wts.update_behavior_score(0.5, TrustEvent.VALID_MESSAGE)
        
        assert new_score >= 0.5  # Should increase or stay same
    
    def test_update_behavior_score_negative(self):
        """Test behavior score update for negative event."""
        from economy.trust import WeightedTrustScore, TrustEvent
        
        wts = WeightedTrustScore()
        
        new_score = wts.update_behavior_score(0.5, TrustEvent.INVALID_MESSAGE)
        
        assert new_score <= 0.5  # Should decrease or stay same
    
    def test_update_behavior_score_slashing(self):
        """Test that slashing events return 0."""
        from economy.trust import WeightedTrustScore, TrustEvent
        
        wts = WeightedTrustScore()
        
        new_score = wts.update_behavior_score(0.9, TrustEvent.INVALID_MERKLE_PROOF)
        
        assert new_score == 0.0
    
    def test_apply_decay(self):
        """Test decay application."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore()
        
        # Recent interaction - no decay
        recent = time.time() - 3600  # 1 hour ago
        score = wts.apply_decay(0.8, recent)
        assert score == 0.8
        
        # Old interaction - decay applied
        old = time.time() - 86400 * 10  # 10 days ago
        score_decayed = wts.apply_decay(0.8, old)
        assert score_decayed < 0.8
    
    @pytest.mark.asyncio
    async def test_get_peer_state(self, ledger):
        """Test getting peer state."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore(ledger=ledger)
        
        state = await wts.get_peer_state("test_peer")
        
        assert state.peer_id == "test_peer"
        assert state.behavior_score == 0.5  # Initial
        assert state.slashed is False
    
    @pytest.mark.asyncio
    async def test_get_peer_state_cached(self, ledger):
        """Test that peer state is cached."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore(ledger=ledger)
        
        state1 = await wts.get_peer_state("cached_peer")
        state2 = await wts.get_peer_state("cached_peer")
        
        assert state1 is state2  # Same object
    
    @pytest.mark.asyncio
    async def test_record_event_positive(self, ledger):
        """Test recording positive event."""
        from economy.trust import WeightedTrustScore, TrustEvent
        
        wts = WeightedTrustScore(ledger=ledger)
        
        peer_id = "positive_peer"
        
        initial = await wts.get_effective_trust(peer_id)
        
        await wts.record_event(peer_id, TrustEvent.SUCCESSFUL_TRANSFER)
        
        after = await wts.get_effective_trust(peer_id)
        assert after >= initial
    
    @pytest.mark.asyncio
    async def test_record_event_slashing(self, ledger):
        """Test slashing via record_event."""
        from economy.trust import WeightedTrustScore, TrustEvent
        
        wts = WeightedTrustScore(ledger=ledger)
        
        peer_id = "slash_test_peer"
        
        score = await wts.record_event(peer_id, TrustEvent.INVALID_MERKLE_PROOF)
        
        assert score == 0.0
        assert wts.is_blacklisted(peer_id)
    
    @pytest.mark.asyncio
    async def test_get_effective_trust_blacklisted(self, ledger):
        """Test effective trust for blacklisted peer."""
        from economy.trust import WeightedTrustScore, TrustEvent
        
        wts = WeightedTrustScore(ledger=ledger)
        
        peer_id = "blacklisted_peer"
        
        # Slash the peer
        await wts.record_event(peer_id, TrustEvent.DOUBLE_SPEND_ATTEMPT)
        
        score = await wts.get_effective_trust(peer_id)
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_get_effective_trust_with_decay(self, ledger):
        """Test effective trust applies decay."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore(ledger=ledger)
        
        peer_id = "decay_test_peer"
        state = await wts.get_peer_state(peer_id)
        
        # Set last interaction to 10 days ago
        state.last_interaction = time.time() - 86400 * 10
        
        score = await wts.get_effective_trust(peer_id)
        # Should be decayed from initial
        assert score < wts.calculate_effective_trust(0.5, 0.0)
    
    def test_is_peer_trusted_new_peer(self):
        """Test that new peer is trusted by default."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore()
        
        # New peer (not in cache) is given a chance
        assert wts.is_peer_trusted("new_unknown_peer") is True
    
    def test_is_peer_trusted_blacklisted(self):
        """Test that blacklisted peer is not trusted."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore()
        wts._blacklist["bad_peer"] = "test"
        
        assert wts.is_peer_trusted("bad_peer") is False
    
    @pytest.mark.asyncio
    async def test_is_peer_trusted_slashed(self, ledger):
        """Test that slashed peer is not trusted."""
        from economy.trust import WeightedTrustScore, TrustEvent
        
        wts = WeightedTrustScore(ledger=ledger)
        
        peer_id = "slashed_untrusted"
        await wts.record_event(peer_id, TrustEvent.SIGNATURE_FORGERY)
        
        assert wts.is_peer_trusted(peer_id) is False
    
    def test_is_blacklisted(self):
        """Test blacklist check."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore()
        
        assert wts.is_blacklisted("unknown") is False
        
        wts._blacklist["known_bad"] = "reason"
        assert wts.is_blacklisted("known_bad") is True
    
    def test_get_blacklist(self):
        """Test getting blacklist."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore()
        wts._blacklist["peer1"] = "reason1"
        wts._blacklist["peer2"] = "reason2"
        
        bl = wts.get_blacklist()
        
        assert "peer1" in bl
        assert "peer2" in bl
        assert bl["peer1"] == "reason1"
    
    @pytest.mark.asyncio
    async def test_slash_peer_explicit(self, ledger):
        """Test explicit slashing."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore(ledger=ledger)
        
        peer_id = "explicit_slash"
        
        await wts.slash_peer(peer_id, "manual_test")
        
        state = await wts.get_peer_state(peer_id)
        assert state.slashed is True
        assert state.slash_reason == "manual_test"
        assert wts.is_blacklisted(peer_id)
    
    @pytest.mark.asyncio
    async def test_update_stake_balance(self, ledger):
        """Test updating stake balance."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore(ledger=ledger)
        
        peer_id = "stake_update"
        
        await wts.update_stake_balance(peer_id, 1000.0)
        
        state = await wts.get_peer_state(peer_id)
        assert state.stake_balance == 1000.0
    
    def test_clear_cache(self):
        """Test clearing cache."""
        from economy.trust import WeightedTrustScore, PeerTrustState
        
        wts = WeightedTrustScore()
        wts._cache["peer1"] = PeerTrustState(peer_id="peer1")
        wts._blacklist["bad"] = "reason"
        
        wts.clear_cache()
        
        assert len(wts._cache) == 0
        # Blacklist is NOT cleared
        assert len(wts._blacklist) == 1


# ============================================================================
# TrustEvent Tests
# ============================================================================

class TestTrustEvent:
    """Test TrustEvent enum."""
    
    def test_positive_events_exist(self):
        """Test positive events are defined."""
        from economy.trust import TrustEvent
        
        assert TrustEvent.SUCCESSFUL_TRANSFER
        assert TrustEvent.VALID_MESSAGE
        assert TrustEvent.PING_RESPONDED
    
    def test_negative_events_exist(self):
        """Test negative events are defined."""
        from economy.trust import TrustEvent
        
        assert TrustEvent.FAILED_TRANSFER
        assert TrustEvent.INVALID_MESSAGE
        assert TrustEvent.PING_TIMEOUT
    
    def test_slashing_events_exist(self):
        """Test slashing events are defined."""
        from economy.trust import TrustEvent, SLASHING_EVENTS
        
        assert TrustEvent.INVALID_MERKLE_PROOF in SLASHING_EVENTS
        assert TrustEvent.DOUBLE_SPEND_ATTEMPT in SLASHING_EVENTS
        assert TrustEvent.SIGNATURE_FORGERY in SLASHING_EVENTS


# ============================================================================
# PeerTrustState Tests
# ============================================================================

class TestPeerTrustState:
    """Test PeerTrustState dataclass."""
    
    def test_create_state(self):
        """Test creating peer trust state."""
        from economy.trust import PeerTrustState
        
        state = PeerTrustState(peer_id="test")
        
        assert state.peer_id == "test"
        assert state.behavior_score == 0.5
        assert state.slashed is False
    
    def test_to_dict(self):
        """Test serialization."""
        from economy.trust import PeerTrustState
        
        state = PeerTrustState(
            peer_id="p1",
            behavior_score=0.8,
            stake_balance=100.0,
            slashed=True,
            slash_reason="test",
        )
        
        d = state.to_dict()
        
        assert d["peer_id"] == "p1"
        assert d["behavior_score"] == 0.8
        assert d["slashed"] is True
    
    def test_from_dict(self):
        """Test deserialization."""
        from economy.trust import PeerTrustState
        
        data = {
            "peer_id": "p2",
            "behavior_score": 0.6,
            "stake_balance": 50.0,
            "interaction_count": 10,
            "slashed": False,
        }
        
        state = PeerTrustState.from_dict(data)
        
        assert state.peer_id == "p2"
        assert state.interaction_count == 10


# ============================================================================
# Global Trust System Tests
# ============================================================================

class TestGetTrustSystem:
    """Test global trust system accessor."""
    
    def test_get_trust_system(self):
        """Test getting trust system singleton."""
        from economy.trust import get_trust_system, _trust_system
        
        # Clear singleton for test
        import economy.trust
        economy.trust._trust_system = None
        
        ts1 = get_trust_system()
        ts2 = get_trust_system()
        
        assert ts1 is ts2  # Same instance
    
    def test_get_trust_system_with_ledger(self, ledger):
        """Test getting trust system with ledger."""
        from economy.trust import get_trust_system
        import economy.trust
        
        # Clear singleton
        economy.trust._trust_system = None
        
        ts = get_trust_system(ledger)
        assert ts.ledger is ledger


# ============================================================================
# File-Based Ledger Tests (Database Operations)
# ============================================================================

class TestFileLedger:
    """Test Ledger with file-based database for full coverage."""
    
    @pytest_asyncio.fixture
    async def file_ledger(self, temp_dir):
        """Create file-based ledger."""
        from economy.ledger import Ledger
        
        db_path = str(temp_dir / "test_ledger.db")
        ledger = Ledger(db_path, debt_limit=1_000_000)
        await ledger.initialize()
        yield ledger
        await ledger.close()
    
    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, file_ledger):
        """Test database initialization creates tables."""
        assert file_ledger._db is not None
        
        # Check tables exist
        cursor = await file_ledger._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        
        assert "peers" in tables
        assert "transactions" in tables
        assert "ious" in tables
        assert "balances" in tables
        assert "knowledge_base" in tables
    
    @pytest.mark.asyncio
    async def test_record_debt_file_mode(self, file_ledger, random_peer_id):
        """Test recording debt in file mode."""
        peer_id = random_peer_id()
        
        balance = await file_ledger.record_debt(peer_id, 500, signature="sig")
        assert balance == -500
        
        # Verify in database
        cursor = await file_ledger._db.execute(
            "SELECT balance FROM balances WHERE peer_id = ?", (peer_id,)
        )
        row = await cursor.fetchone()
        assert row[0] == -500
    
    @pytest.mark.asyncio
    async def test_record_claim_file_mode(self, file_ledger, random_peer_id):
        """Test recording claim in file mode."""
        peer_id = random_peer_id()
        
        balance = await file_ledger.record_claim(peer_id, 300, signature="sig")
        assert balance == 300
        
        # Update existing
        balance = await file_ledger.record_claim(peer_id, 200, signature="sig2")
        assert balance == 500
    
    @pytest.mark.asyncio
    async def test_get_balance_from_db(self, file_ledger, random_peer_id):
        """Test getting balance from database."""
        peer_id = random_peer_id()
        
        await file_ledger.record_claim(peer_id, 100)
        
        # Clear cache to force DB read
        file_ledger._balance_cache.clear()
        file_ledger._cache_valid = False
        
        balance = await file_ledger.get_balance(peer_id)
        assert balance == 100
    
    @pytest.mark.asyncio
    async def test_get_balance_info_file_mode(self, file_ledger, random_peer_id):
        """Test getting balance info from file database."""
        peer_id = random_peer_id()
        
        await file_ledger.record_debt(peer_id, 100)
        await file_ledger.record_claim(peer_id, 50)
        
        info = await file_ledger.get_balance_info(peer_id)
        
        assert info["balance"] == -50
        assert info["total_sent"] > 0 or info["total_received"] > 0
    
    @pytest.mark.asyncio
    async def test_get_all_balances_file_mode(self, file_ledger, random_peer_id):
        """Test getting all balances from file database."""
        for _ in range(3):
            await file_ledger.record_debt(random_peer_id(), 100)
        
        balances = await file_ledger.get_all_balances()
        assert len(balances) >= 3
    
    @pytest.mark.asyncio
    async def test_check_can_send_file_mode(self, file_ledger, random_peer_id):
        """Test can_send in file mode."""
        peer_id = random_peer_id()
        
        can, reason = await file_ledger.check_can_send(peer_id)
        assert can is True
    
    @pytest.mark.asyncio
    async def test_reset_balance_file_mode(self, file_ledger, random_peer_id):
        """Test resetting balance in file mode."""
        peer_id = random_peer_id()
        
        await file_ledger.record_claim(peer_id, 1000)
        await file_ledger.reset_balance(peer_id)
        
        balance = await file_ledger.get_balance(peer_id)
        assert balance == 0.0
    
    @pytest.mark.asyncio
    async def test_get_or_create_peer_file_mode(self, file_ledger, random_peer_id):
        """Test peer creation in file mode."""
        peer_id = random_peer_id()
        
        peer = await file_ledger.get_or_create_peer(peer_id, public_key="pubkey")
        assert peer["node_id"] == peer_id
        
        # Get existing
        peer2 = await file_ledger.get_or_create_peer(peer_id)
        assert peer2["node_id"] == peer_id
    
    @pytest.mark.asyncio
    async def test_update_trust_score_file_mode(self, file_ledger, random_peer_id):
        """Test trust score update in file mode."""
        peer_id = random_peer_id()
        
        # Create peer first
        await file_ledger.get_or_create_peer(peer_id)
        
        # Update score
        score = await file_ledger.update_trust_score(peer_id, "ping_responded", 1.0)
        assert score >= 0
    
    @pytest.mark.asyncio
    async def test_get_trust_score_file_mode(self, file_ledger, random_peer_id):
        """Test getting trust score in file mode."""
        peer_id = random_peer_id()
        
        await file_ledger.get_or_create_peer(peer_id)
        
        score = await file_ledger.get_trust_score(peer_id)
        assert score > 0
    
    @pytest.mark.asyncio
    async def test_get_peers_by_trust(self, file_ledger, random_peer_id):
        """Test getting peers by trust score."""
        # Create some peers
        for _ in range(3):
            await file_ledger.get_or_create_peer(random_peer_id())
        
        peers = await file_ledger.get_peers_by_trust(min_score=0.0, limit=10)
        assert len(peers) >= 3
    
    @pytest.mark.asyncio
    async def test_knowledge_base_file_mode(self, file_ledger):
        """Test knowledge base in file mode."""
        entry_id = await file_ledger.add_knowledge_entry(
            cid="Qm123",
            path="/test/path",
            summary="Test summary",
            tags="test,tags",
            size=1024,
            metadata={"key": "value"},
        )
        
        assert entry_id > 0
        
        entries = await file_ledger.get_knowledge_entries(limit=10)
        assert len(entries) >= 1


# ============================================================================
# IOU Tests (File-Based)
# ============================================================================

class TestIOUOperations:
    """Test IOU CRUD operations with file-based ledger."""
    
    @pytest_asyncio.fixture
    async def iou_ledger(self, temp_dir):
        """Create ledger for IOU tests."""
        from economy.ledger import Ledger
        
        db_path = str(temp_dir / "iou_test.db")
        ledger = Ledger(db_path)
        await ledger.initialize()
        yield ledger
        await ledger.close()
    
    @pytest.mark.asyncio
    async def test_create_iou(self, iou_ledger):
        """Test IOU creation."""
        iou = await iou_ledger.create_iou(
            debtor_id="debtor1",
            creditor_id="creditor1",
            amount=1000.0,
            signature="test_sig",
        )
        
        assert iou.id is not None
        assert len(iou.id) == 64  # SHA256 hex
        assert iou.amount == 1000.0
        assert iou.redeemed is False
    
    @pytest.mark.asyncio
    async def test_create_iou_with_expiry(self, iou_ledger):
        """Test IOU creation with expiry."""
        expires = time.time() + 3600
        
        iou = await iou_ledger.create_iou(
            debtor_id="d",
            creditor_id="c",
            amount=100.0,
            signature="s",
            expires_at=expires,
        )
        
        assert iou.expires_at == expires
    
    @pytest.mark.asyncio
    async def test_get_iou(self, iou_ledger):
        """Test getting IOU by ID."""
        created = await iou_ledger.create_iou(
            debtor_id="d",
            creditor_id="c",
            amount=50.0,
            signature="s",
        )
        
        fetched = await iou_ledger.get_iou(created.id)
        
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.amount == 50.0
    
    @pytest.mark.asyncio
    async def test_get_iou_not_found(self, iou_ledger):
        """Test getting non-existent IOU."""
        result = await iou_ledger.get_iou("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_redeem_iou(self, iou_ledger):
        """Test redeeming IOU."""
        iou = await iou_ledger.create_iou(
            debtor_id="d",
            creditor_id="c",
            amount=100.0,
            signature="s",
        )
        
        result = await iou_ledger.redeem_iou(iou.id)
        assert result is True
        
        # Verify redeemed
        fetched = await iou_ledger.get_iou(iou.id)
        assert fetched.redeemed is True
    
    @pytest.mark.asyncio
    async def test_redeem_iou_not_found(self, iou_ledger):
        """Test redeeming non-existent IOU."""
        result = await iou_ledger.redeem_iou("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_redeem_iou_already_redeemed(self, iou_ledger):
        """Test redeeming already redeemed IOU."""
        iou = await iou_ledger.create_iou(
            debtor_id="d",
            creditor_id="c",
            amount=100.0,
            signature="s",
        )
        
        # Redeem once
        await iou_ledger.redeem_iou(iou.id)
        
        # Try to redeem again
        result = await iou_ledger.redeem_iou(iou.id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_redeem_iou_expired(self, iou_ledger):
        """Test redeeming expired IOU."""
        iou = await iou_ledger.create_iou(
            debtor_id="d",
            creditor_id="c",
            amount=100.0,
            signature="s",
            expires_at=time.time() - 1,  # Already expired
        )
        
        result = await iou_ledger.redeem_iou(iou.id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_ious_as_creditor(self, iou_ledger):
        """Test getting IOUs where node is creditor."""
        creditor_id = "test_creditor"
        
        # Create multiple IOUs
        for i in range(3):
            await iou_ledger.create_iou(
                debtor_id=f"debtor_{i}",
                creditor_id=creditor_id,
                amount=float(100 * (i + 1)),
                signature="s",
            )
        
        ious = await iou_ledger.get_ious_as_creditor(creditor_id)
        assert len(ious) == 3
    
    @pytest.mark.asyncio
    async def test_get_ious_as_creditor_only_valid(self, iou_ledger):
        """Test getting only valid IOUs as creditor."""
        creditor_id = "cred"
        
        # Create valid IOU
        await iou_ledger.create_iou(
            debtor_id="d1",
            creditor_id=creditor_id,
            amount=100.0,
            signature="s",
        )
        
        # Create expired IOU
        await iou_ledger.create_iou(
            debtor_id="d2",
            creditor_id=creditor_id,
            amount=50.0,
            signature="s",
            expires_at=time.time() - 100,
        )
        
        # Get only valid
        ious = await iou_ledger.get_ious_as_creditor(creditor_id, only_valid=True)
        assert len(ious) == 1
        
        # Get all
        ious_all = await iou_ledger.get_ious_as_creditor(creditor_id, only_valid=False)
        assert len(ious_all) == 2
    
    @pytest.mark.asyncio
    async def test_get_ious_as_debtor(self, iou_ledger):
        """Test getting IOUs where node is debtor."""
        debtor_id = "test_debtor"
        
        for i in range(2):
            await iou_ledger.create_iou(
                debtor_id=debtor_id,
                creditor_id=f"creditor_{i}",
                amount=100.0,
                signature="s",
            )
        
        ious = await iou_ledger.get_ious_as_debtor(debtor_id)
        assert len(ious) == 2
    
    @pytest.mark.asyncio
    async def test_get_ious_as_debtor_only_valid(self, iou_ledger):
        """Test getting only valid IOUs as debtor."""
        debtor_id = "deb"
        
        # Create valid IOU
        iou = await iou_ledger.create_iou(
            debtor_id=debtor_id,
            creditor_id="c1",
            amount=100.0,
            signature="s",
        )
        
        # Redeem it
        await iou_ledger.redeem_iou(iou.id)
        
        # Create another valid IOU
        await iou_ledger.create_iou(
            debtor_id=debtor_id,
            creditor_id="c2",
            amount=50.0,
            signature="s",
        )
        
        # Get only valid
        ious = await iou_ledger.get_ious_as_debtor(debtor_id, only_valid=True)
        assert len(ious) == 1
        
        # Get all
        ious_all = await iou_ledger.get_ious_as_debtor(debtor_id, only_valid=False)
        assert len(ious_all) == 2
    
    @pytest.mark.asyncio
    async def test_get_total_debt(self, iou_ledger):
        """Test calculating total debt."""
        node_id = "test_node"
        
        # Node owes others
        await iou_ledger.create_iou(
            debtor_id=node_id,
            creditor_id="c1",
            amount=100.0,
            signature="s",
        )
        
        # Others owe node
        await iou_ledger.create_iou(
            debtor_id="d1",
            creditor_id=node_id,
            amount=50.0,
            signature="s",
        )
        
        owed_to_others, owed_by_others = await iou_ledger.get_total_debt(node_id)
        
        assert owed_to_others == 100.0
        assert owed_by_others == 50.0


# ============================================================================
# Transaction Tests (File-Based)
# ============================================================================

class TestTransactionOperations:
    """Test Transaction CRUD operations."""
    
    @pytest_asyncio.fixture
    async def tx_ledger(self, temp_dir):
        """Create ledger for transaction tests."""
        from economy.ledger import Ledger
        
        db_path = str(temp_dir / "tx_test.db")
        ledger = Ledger(db_path)
        await ledger.initialize()
        yield ledger
        await ledger.close()
    
    @pytest.mark.asyncio
    async def test_add_transaction(self, tx_ledger):
        """Test adding transaction."""
        from economy.ledger import Transaction
        
        # Create peers first
        await tx_ledger.get_or_create_peer("sender")
        await tx_ledger.get_or_create_peer("receiver")
        
        tx = Transaction(
            id="tx123",
            from_id="sender",
            to_id="receiver",
            amount=500.0,
            timestamp=time.time(),
            signature="sig",
            tx_type="transfer",
            metadata={"note": "test"},
        )
        
        result = await tx_ledger.add_transaction(tx)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_add_transaction_duplicate(self, tx_ledger):
        """Test adding duplicate transaction fails."""
        from economy.ledger import Transaction
        
        await tx_ledger.get_or_create_peer("s")
        await tx_ledger.get_or_create_peer("r")
        
        tx = Transaction(
            id="dup_tx",
            from_id="s",
            to_id="r",
            amount=100.0,
            timestamp=time.time(),
            signature="s",
        )
        
        # First add succeeds
        result1 = await tx_ledger.add_transaction(tx)
        assert result1 is True
        
        # Second add fails (duplicate ID)
        result2 = await tx_ledger.add_transaction(tx)
        assert result2 is False
    
    @pytest.mark.asyncio
    async def test_get_transaction(self, tx_ledger):
        """Test getting transaction by ID."""
        from economy.ledger import Transaction
        
        await tx_ledger.get_or_create_peer("a")
        await tx_ledger.get_or_create_peer("b")
        
        tx = Transaction(
            id="get_tx",
            from_id="a",
            to_id="b",
            amount=200.0,
            timestamp=time.time(),
            signature="sig",
        )
        
        await tx_ledger.add_transaction(tx)
        
        fetched = await tx_ledger.get_transaction("get_tx")
        assert fetched is not None
        assert fetched.amount == 200.0
    
    @pytest.mark.asyncio
    async def test_get_transaction_not_found(self, tx_ledger):
        """Test getting non-existent transaction."""
        result = await tx_ledger.get_transaction("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_transactions_for_peer(self, tx_ledger):
        """Test getting transactions involving a peer."""
        from economy.ledger import Transaction
        
        peer = "target_peer"
        await tx_ledger.get_or_create_peer(peer)
        await tx_ledger.get_or_create_peer("other")
        
        # Outgoing transaction
        tx1 = Transaction(
            id="tx_out",
            from_id=peer,
            to_id="other",
            amount=100.0,
            timestamp=time.time(),
            signature="s",
        )
        await tx_ledger.add_transaction(tx1)
        
        # Incoming transaction
        tx2 = Transaction(
            id="tx_in",
            from_id="other",
            to_id=peer,
            amount=50.0,
            timestamp=time.time(),
            signature="s",
        )
        await tx_ledger.add_transaction(tx2)
        
        transactions = await tx_ledger.get_transactions_for_peer(peer)
        assert len(transactions) == 2


# ============================================================================
# Statistics Tests
# ============================================================================

class TestLedgerStats:
    """Test Ledger statistics."""
    
    @pytest_asyncio.fixture
    async def stats_ledger(self, temp_dir):
        """Create ledger for stats tests."""
        from economy.ledger import Ledger
        
        db_path = str(temp_dir / "stats_test.db")
        ledger = Ledger(db_path)
        await ledger.initialize()
        yield ledger
        await ledger.close()
    
    @pytest.mark.asyncio
    async def test_get_stats_empty(self, stats_ledger):
        """Test getting stats from empty ledger."""
        stats = await stats_ledger.get_stats()
        
        assert "peer_count" in stats
        assert "transaction_count" in stats
        assert "active_ious" in stats
        assert "debt_limit" in stats
        assert stats["debt_limit"] > 0
    
    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, stats_ledger, random_peer_id):
        """Test getting stats with data."""
        # Add peers
        for _ in range(5):
            await stats_ledger.get_or_create_peer(random_peer_id())
        
        # Add IOUs
        await stats_ledger.create_iou("d1", "c1", 100.0, "s")
        await stats_ledger.create_iou("d2", "c2", 200.0, "s")
        
        # Add balances
        await stats_ledger.record_claim(random_peer_id(), 1000)
        await stats_ledger.record_debt(random_peer_id(), 500)
        
        stats = await stats_ledger.get_stats()
        
        assert stats["peer_count"] >= 5
        assert stats["active_ious"] == 2
        assert stats["total_outstanding_debt"] == 300.0


# ============================================================================
# Settlement Tests
# ============================================================================

class TestSettlement:
    """Test settlement/payment operations."""
    
    @pytest_asyncio.fixture
    async def settlement_ledger(self, temp_dir):
        """Create ledger for settlement tests."""
        from economy.ledger import Ledger
        
        db_path = str(temp_dir / "settlement_test.db")
        ledger = Ledger(db_path)
        await ledger.initialize()
        yield ledger
        await ledger.close()
    
    @pytest.mark.asyncio
    async def test_record_payment(self, settlement_ledger, random_peer_id):
        """Test recording payment (settlement)."""
        peer_id = random_peer_id()
        
        # First create some debt
        await settlement_ledger.record_debt(peer_id, 1000)
        
        # Record payment to offset
        new_balance = await settlement_ledger.record_payment(peer_id, 500, tx_hash="0x123")
        
        assert new_balance == -500  # -1000 + 500
    
    @pytest.mark.asyncio
    async def test_record_settlement_alias(self, settlement_ledger, random_peer_id):
        """Test that record_settlement is alias for record_payment."""
        peer_id = random_peer_id()
        
        await settlement_ledger.record_debt(peer_id, 100)
        balance = await settlement_ledger.record_settlement(peer_id, 100, tx_hash="0xabc")
        
        assert balance == 0.0


# ============================================================================
# Additional Coverage Tests
# ============================================================================

class TestTrustAdditionalCoverage:
    """Additional tests for economy/trust.py coverage."""
    
    def test_update_behavior_score_no_change(self):
        """Test behavior score with no change (unknown event)."""
        from economy.trust import WeightedTrustScore, TrustEvent
        
        wts = WeightedTrustScore()
        
        # Try with an event that has 0 weight (if any)
        # Use a very small magnitude to test edge case
        current = 0.5
        
        # Test that no-change path returns current score
        # This happens when adjustment is exactly 0
        score = wts.update_behavior_score(current, TrustEvent.VALID_MESSAGE, magnitude=0.0)
        assert score == current  # No change
    
    @pytest.mark.asyncio
    async def test_get_total_balance_no_ledger(self):
        """Test _get_total_balance without ledger."""
        from economy.trust import WeightedTrustScore
        
        wts = WeightedTrustScore(ledger=None)
        
        balance = await wts._get_total_balance("test_peer")
        assert balance == 0.0
    
    @pytest.mark.asyncio
    async def test_record_event_updates_stake(self, ledger, random_peer_id):
        """Test that record_event updates stake balance."""
        from economy.trust import WeightedTrustScore, TrustEvent
        
        wts = WeightedTrustScore(ledger=ledger)
        peer_id = random_peer_id()
        
        # Add claim to ledger
        await ledger.record_claim(peer_id, 1024 * 1024 * 1024)  # 1 GB
        
        # Record event should pick up new balance
        await wts.record_event(peer_id, TrustEvent.VALID_MESSAGE)
        
        state = await wts.get_peer_state(peer_id)
        assert state.stake_balance > 0
    
    @pytest.mark.asyncio
    async def test_is_peer_trusted_cached_good_score(self, ledger):
        """Test is_peer_trusted with cached peer having good score."""
        from economy.trust import WeightedTrustScore, PeerTrustState
        
        wts = WeightedTrustScore(ledger=ledger)
        
        # Pre-cache a peer with good score
        wts._cache["good_peer"] = PeerTrustState(
            peer_id="good_peer",
            behavior_score=0.8,
            stake_balance=100.0,
        )
        
        assert wts.is_peer_trusted("good_peer", min_trust=0.1) is True
    
    @pytest.mark.asyncio
    async def test_is_peer_trusted_cached_low_score(self, ledger):
        """Test is_peer_trusted with cached peer having low score."""
        from economy.trust import WeightedTrustScore, PeerTrustState
        
        wts = WeightedTrustScore(ledger=ledger)
        
        # Pre-cache a peer with very low score
        wts._cache["low_peer"] = PeerTrustState(
            peer_id="low_peer",
            behavior_score=0.01,
            stake_balance=0.0,
        )
        
        # With high min_trust requirement
        assert wts.is_peer_trusted("low_peer", min_trust=0.5) is False


class TestLedgerEdgeCases:
    """Edge case tests for ledger."""
    
    @pytest_asyncio.fixture
    async def edge_ledger(self, temp_dir):
        """Create ledger for edge case tests."""
        from economy.ledger import Ledger
        
        db_path = str(temp_dir / "edge_test.db")
        ledger = Ledger(db_path)
        await ledger.initialize()
        yield ledger
        await ledger.close()
    
    @pytest.mark.asyncio
    async def test_update_trust_score_unknown_peer(self, edge_ledger, random_peer_id):
        """Test updating trust score for unknown peer."""
        peer_id = random_peer_id()
        
        # Don't create peer first
        score = await edge_ledger.update_trust_score(peer_id, "ping_responded")
        
        # Should return default score since peer doesn't exist
        assert score == 0.5  # Default from config
    
    @pytest.mark.asyncio
    async def test_get_peers_by_trust_empty(self, edge_ledger):
        """Test getting peers by trust from empty ledger."""
        peers = await edge_ledger.get_peers_by_trust(min_score=0.9)
        assert peers == []
    
    @pytest.mark.asyncio
    async def test_reconcile_with_our_balance_provided(self, edge_ledger, random_peer_id):
        """Test reconcile when our_balance is explicitly provided."""
        peer_id = random_peer_id()
        
        result = await edge_ledger.reconcile_balance(
            peer_id=peer_id,
            peer_claimed_balance=-100,
            our_balance=100,  # Explicitly provided
        )
        
        assert result["our_balance"] == 100
    
    @pytest.mark.asyncio
    async def test_get_balance_cache_invalid(self, edge_ledger, random_peer_id):
        """Test get_balance when cache is invalid."""
        peer_id = random_peer_id()
        
        await edge_ledger.record_claim(peer_id, 500)
        
        # Invalidate cache
        edge_ledger._cache_valid = False
        edge_ledger._balance_cache.clear()
        
        balance = await edge_ledger.get_balance(peer_id)
        assert balance == 500


class TestTransportAdditionalCoverage:
    """Additional transport tests for edge cases."""
    
    @pytest.mark.asyncio
    async def test_blocking_transport_blocked(self, ledger, random_peer_id):
        """Test blocking transport when peer is blocked."""
        from core.transport import BlockingTransport, Message, MessageType
        
        transport = BlockingTransport(ledger)
        peer_id = random_peer_id()
        
        # Make peer exceed debt limit
        await ledger.record_claim(peer_id, ledger.debt_limit + 1000)
        
        msg = Message(
            type=MessageType.DATA,
            payload={},
            sender_id="me",
        )
        
        data, size, blocked, reason = await transport.pack_with_accounting(msg, peer_id)
        
        assert blocked is True
        assert data == b""
        assert size == 0
        assert "exceeds limit" in reason


class TestTrustExceptionHandling:
    """Test exception handling in trust module."""
    
    @pytest.mark.asyncio
    async def test_get_total_balance_with_ledger_error(self):
        """Test _get_total_balance handles ledger errors."""
        from economy.trust import WeightedTrustScore
        from unittest.mock import AsyncMock, MagicMock
        
        # Create mock ledger that raises exception
        mock_ledger = MagicMock()
        mock_ledger.get_balance_info = AsyncMock(side_effect=Exception("DB error"))
        
        wts = WeightedTrustScore(ledger=mock_ledger)
        
        # Should return 0 and not raise
        balance = await wts._get_total_balance("peer")
        assert balance == 0.0
    
    @pytest.mark.asyncio
    async def test_record_event_with_ledger_error(self):
        """Test record_event handles ledger errors gracefully."""
        from economy.trust import WeightedTrustScore, TrustEvent
        from unittest.mock import AsyncMock, MagicMock
        
        # Create mock ledger that raises exception on update
        mock_ledger = MagicMock()
        mock_ledger.get_balance_info = AsyncMock(return_value={"balance": 0.0})
        mock_ledger.update_trust_score = AsyncMock(side_effect=Exception("Update failed"))
        
        wts = WeightedTrustScore(ledger=mock_ledger)
        
        # Should not raise, just log and continue
        score = await wts.record_event("peer", TrustEvent.VALID_MESSAGE)
        assert score >= 0  # Some score returned
    
    @pytest.mark.asyncio
    async def test_record_event_slashing_with_ledger_error(self):
        """Test slashing event handles ledger error gracefully."""
        from economy.trust import WeightedTrustScore, TrustEvent
        from unittest.mock import AsyncMock, MagicMock
        
        mock_ledger = MagicMock()
        mock_ledger.update_trust_score = AsyncMock(side_effect=Exception("Fail"))
        
        wts = WeightedTrustScore(ledger=mock_ledger)
        
        # Slashing should still work
        score = await wts.record_event("peer", TrustEvent.INVALID_MERKLE_PROOF)
        assert score == 0.0
        assert wts.is_blacklisted("peer")
    
    @pytest.mark.asyncio
    async def test_get_effective_trust_slashed_state(self, ledger):
        """Test get_effective_trust for slashed peer (not via blacklist)."""
        from economy.trust import WeightedTrustScore, PeerTrustState
        
        wts = WeightedTrustScore(ledger=ledger)
        
        # Cache a slashed peer (but not in blacklist)
        peer_id = "slashed_not_blacklisted"
        wts._cache[peer_id] = PeerTrustState(
            peer_id=peer_id,
            slashed=True,
            behavior_score=0.0,
        )
        
        score = await wts.get_effective_trust(peer_id)
        assert score == 0.0
    
    def test_is_peer_trusted_cached_slashed(self):
        """Test is_peer_trusted for cached slashed peer."""
        from economy.trust import WeightedTrustScore, PeerTrustState
        
        wts = WeightedTrustScore()
        
        wts._cache["slashed_peer"] = PeerTrustState(
            peer_id="slashed_peer",
            slashed=True,
        )
        
        assert wts.is_peer_trusted("slashed_peer") is False


class TestTrustSystemSingleton:
    """Test trust system singleton edge cases."""
    
    def test_get_trust_system_update_ledger(self, ledger):
        """Test that get_trust_system updates ledger if initially None."""
        import economy.trust
        
        # Create system without ledger
        economy.trust._trust_system = None
        ts = economy.trust.get_trust_system(None)
        assert ts.ledger is None
        
        # Now update with ledger
        ts2 = economy.trust.get_trust_system(ledger)
        assert ts2 is ts  # Same instance
        assert ts2.ledger is ledger  # Ledger updated
