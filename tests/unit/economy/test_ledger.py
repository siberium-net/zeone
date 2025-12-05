"""
Ledger Unit Tests
=================

[UNIT] Tests for economy/ledger.py operations.
"""

import pytest
import pytest_asyncio


class TestLedger:
    """Test Ledger class."""
    
    @pytest.mark.asyncio
    async def test_create_ledger(self, ledger):
        """Test ledger creation."""
        assert ledger is not None
        assert ledger.debt_limit > 0
    
    @pytest.mark.asyncio
    async def test_get_or_create_peer(self, ledger, random_peer_id):
        """Test peer creation."""
        peer_id = random_peer_id()
        
        # First call creates
        peer = await ledger.get_or_create_peer(peer_id)
        assert peer is not None
        
        # Second call returns existing
        peer2 = await ledger.get_or_create_peer(peer_id)
        assert peer2 is not None
    
    @pytest.mark.asyncio
    async def test_record_debt(self, ledger, random_peer_id):
        """Test recording debt (we owe them)."""
        peer_id = random_peer_id()
        
        # Record debt
        new_balance = await ledger.record_debt(peer_id, 100)
        assert new_balance == -100  # Negative = we owe them
        
        # Check balance
        balance = await ledger.get_balance(peer_id)
        assert balance == -100
    
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
    async def test_debt_limit_blocking(self, ledger, random_peer_id):
        """Test that exceeding debt limit blocks peer."""
        peer_id = random_peer_id()
        
        # Record debt exceeding limit
        await ledger.record_claim(peer_id, ledger.debt_limit + 100, signature="")
        
        # Check blocked status
        is_blocked = ledger.is_peer_blocked(peer_id)
        assert is_blocked is True
    
    @pytest.mark.asyncio
    async def test_balance_info(self, ledger, random_peer_id):
        """Test getting balance info."""
        peer_id = random_peer_id()
        
        # Record some transactions
        await ledger.record_debt(peer_id, 100)
        await ledger.record_claim(peer_id, 50, signature="")
        
        # Get info
        info = await ledger.get_balance_info(peer_id)
        
        assert "balance" in info
        assert info["balance"] == -50  # Net: owed 100, claimed 50
    
    @pytest.mark.asyncio
    async def test_all_balances(self, ledger, random_peer_id):
        """Test getting all balances."""
        # Create multiple peers
        for _ in range(3):
            peer_id = random_peer_id()
            await ledger.record_debt(peer_id, 50)
        
        balances = await ledger.get_all_balances()
        assert len(balances) >= 3
    
    @pytest.mark.asyncio
    async def test_trust_score_update(self, ledger, random_peer_id):
        """Test trust score updates."""
        peer_id = random_peer_id()
        
        # Initial score
        score = await ledger.get_trust_score(peer_id)
        initial_score = score
        
        # Positive interaction
        await ledger.update_trust_score(peer_id, "ping_responded")
        score = await ledger.get_trust_score(peer_id)
        assert score >= initial_score
        
        # Negative interaction
        await ledger.update_trust_score(peer_id, "invalid_message")
        score2 = await ledger.get_trust_score(peer_id)
        assert score2 <= score


class TestWeightedTrustScore:
    """Test WeightedTrustScore class."""
    
    def test_weighted_trust_formula(self):
        """Test weighted trust score calculation."""
        try:
            from economy.trust import WeightedTrustScore
            
            wts = WeightedTrustScore()
            
            # Low stake, high behavior
            score1 = wts.calculate_effective_trust(behavior_score=0.9, stake=10)
            
            # High stake, high behavior
            score2 = wts.calculate_effective_trust(behavior_score=0.9, stake=10000)
            
            # Higher stake should give higher effective score
            assert score2 > score1
            
        except ImportError:
            pytest.skip("WeightedTrustScore not available")
    
    def test_dust_limit(self):
        """Test dust limit penalty."""
        try:
            from economy.trust import WeightedTrustScore
            
            wts = WeightedTrustScore()
            
            # Below dust limit
            score = wts.calculate_effective_trust(behavior_score=1.0, stake=1)
            
            # Should be significantly penalized
            assert score < 0.5
            
        except ImportError:
            pytest.skip("WeightedTrustScore not available")
    
    @pytest.mark.asyncio
    async def test_critical_slash(self, ledger):
        """Test critical slashing for violations."""
        try:
            from economy.trust import WeightedTrustScore, TrustEvent
            
            wts = WeightedTrustScore(ledger=ledger)
            
            peer_id = "test_peer_for_slash"
            
            # Record slashing event
            score = await wts.record_event(peer_id, TrustEvent.INVALID_MERKLE_PROOF)
            
            # Score should be zero after slash
            assert score == 0.0
            
            # Peer should be blacklisted
            assert peer_id in wts._blacklist
            
        except ImportError:
            pytest.skip("WeightedTrustScore not available")
    
    @pytest.mark.asyncio
    async def test_behavior_score_update(self, ledger):
        """Test behavior score EMA update."""
        try:
            from economy.trust import WeightedTrustScore, TrustEvent
            
            wts = WeightedTrustScore(ledger=ledger)
            
            peer_id = "test_peer_123"
            
            # Get initial state
            state = await wts.get_peer_state(peer_id)
            initial_behavior = state.behavior_score
            
            # Successful interaction
            await wts.record_event(peer_id, TrustEvent.VALID_MESSAGE)
            state = await wts.get_peer_state(peer_id)
            
            # Behavior should increase
            assert state.behavior_score >= initial_behavior
            
        except ImportError:
            pytest.skip("WeightedTrustScore not available")
    
    def test_constants(self):
        """Test trust score constants are defined."""
        try:
            from economy.trust import WeightedTrustScore
            
            assert WeightedTrustScore.BASE_STAKE > 0
            assert WeightedTrustScore.DUST_LIMIT > 0
            assert 0 < WeightedTrustScore.EMA_ALPHA < 1
            assert 0 < WeightedTrustScore.DECAY_RATE <= 1
            
        except ImportError:
            pytest.skip("WeightedTrustScore not available")
