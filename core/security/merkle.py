"""
Merkle Tree Implementation for Chunk Verification
==================================================

[SECURITY] This module provides cryptographic verification of data chunks
using Merkle Trees. It enables instant verification of individual chunks
without requiring the complete file.

Key Features:
- Build tree from list of chunk hashes
- Generate inclusion proofs for any chunk
- Verify chunk against root hash using proof

Algorithm:
- Hash function: SHA-256
- Pair hashing: H(left || right) 
- Odd leaf count: duplicate last hash
- Proof format: [(sibling_hash, is_right), ...] from leaf to root
"""

import hashlib
from typing import List, Tuple, Optional


class MerkleTree:
    """
    Merkle Tree for chunk-level data verification.
    
    [SECURITY] Allows verification of individual data chunks without
    downloading the entire file. A malicious chunk is detected instantly.
    
    Example:
        # Build tree from chunk hashes
        chunk_hashes = [sha256(chunk) for chunk in chunks]
        tree = MerkleTree(chunk_hashes)
        
        # Get proof for chunk 5
        proof = tree.get_proof(5)
        
        # Verify chunk (on receiving end)
        is_valid = MerkleTree.verify_chunk(
            chunk_data, proof, tree.root, chunk_index=5, total_chunks=10
        )
    """
    
    def __init__(self, chunk_hashes: List[bytes]):
        """
        Build Merkle Tree from list of chunk hashes.
        
        Args:
            chunk_hashes: List of 32-byte SHA-256 hashes, one per chunk.
                          Order must match chunk indices.
        
        Raises:
            ValueError: If chunk_hashes is empty
        """
        if not chunk_hashes:
            raise ValueError("Cannot build Merkle tree from empty list")
        
        self._leaf_count = len(chunk_hashes)
        self._levels: List[List[bytes]] = []
        
        # Build tree bottom-up
        self._build(chunk_hashes)
    
    def _build(self, leaves: List[bytes]) -> None:
        """
        Build tree levels from leaves to root.
        
        [ALGORITHM]
        1. Start with leaf hashes as level 0
        2. If odd number of nodes, duplicate last node
        3. Pair adjacent nodes: H(left || right)
        4. Repeat until single root remains
        """
        current_level = list(leaves)
        self._levels.append(current_level)
        
        while len(current_level) > 1:
            next_level = []
            
            # Handle odd number of nodes by duplicating last
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])
            
            # Pair and hash
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1]
                parent = self._hash_pair(left, right)
                next_level.append(parent)
            
            self._levels.append(next_level)
            current_level = next_level
    
    @staticmethod
    def _hash_pair(left: bytes, right: bytes) -> bytes:
        """Hash two nodes together: SHA256(left || right)."""
        return hashlib.sha256(left + right).digest()
    
    @staticmethod
    def hash_chunk(data: bytes) -> bytes:
        """Compute SHA-256 hash of chunk data."""
        return hashlib.sha256(data).digest()
    
    @property
    def root(self) -> bytes:
        """
        Get the Merkle root hash.
        
        Returns:
            32-byte root hash that commits to all chunks
        """
        return self._levels[-1][0]
    
    @property
    def root_hex(self) -> str:
        """Get root hash as hex string."""
        return self.root.hex()
    
    @property
    def leaf_count(self) -> int:
        """Number of leaf nodes (chunks)."""
        return self._leaf_count
    
    def get_proof(self, chunk_index: int) -> List[Tuple[bytes, bool]]:
        """
        Generate Merkle proof for a chunk.
        
        Args:
            chunk_index: Index of the chunk (0-based)
        
        Returns:
            List of (sibling_hash, is_right) tuples from leaf to root.
            is_right indicates if sibling is on the right side.
        
        Raises:
            IndexError: If chunk_index out of range
        
        [ALGORITHM]
        Walk from leaf to root, collecting sibling at each level:
        - If index is even, sibling is at index+1 (right)
        - If index is odd, sibling is at index-1 (left)
        """
        if chunk_index < 0 or chunk_index >= self._leaf_count:
            raise IndexError(f"Chunk index {chunk_index} out of range [0, {self._leaf_count})")
        
        proof: List[Tuple[bytes, bool]] = []
        index = chunk_index
        
        for level in self._levels[:-1]:  # Exclude root level
            # Handle duplicated last node for odd-length levels
            level_len = len(level)
            if level_len % 2 == 1:
                level = level + [level[-1]]
            
            if index % 2 == 0:
                # Sibling is on the right
                sibling_index = index + 1
                is_right = True
            else:
                # Sibling is on the left
                sibling_index = index - 1
                is_right = False
            
            sibling_hash = level[sibling_index]
            proof.append((sibling_hash, is_right))
            
            # Move to parent index
            index = index // 2
        
        return proof
    
    def get_proof_hex(self, chunk_index: int) -> List[Tuple[str, bool]]:
        """Get proof with hashes as hex strings (for JSON serialization)."""
        proof = self.get_proof(chunk_index)
        return [(h.hex(), is_right) for h, is_right in proof]
    
    @staticmethod
    def verify_chunk(
        chunk_data: bytes,
        proof: List[Tuple[bytes, bool]],
        root_hash: bytes,
        chunk_index: int,
        total_chunks: int,
    ) -> bool:
        """
        Verify a chunk belongs to the tree with given root.
        
        Args:
            chunk_data: Raw chunk bytes to verify
            proof: Merkle proof [(sibling_hash, is_right), ...]
            root_hash: Expected Merkle root (32 bytes)
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks in file
        
        Returns:
            True if chunk is valid, False if corrupted/tampered
        
        [SECURITY] This method enables instant verification of individual
        chunks. A corrupted chunk will fail verification immediately,
        without needing to download the entire file.
        """
        if chunk_index < 0 or chunk_index >= total_chunks:
            return False
        
        # Start with hash of the chunk data
        current_hash = MerkleTree.hash_chunk(chunk_data)
        
        # Walk up the tree using the proof
        for sibling_hash, is_right in proof:
            if is_right:
                # Sibling is on right, we are on left
                current_hash = MerkleTree._hash_pair(current_hash, sibling_hash)
            else:
                # Sibling is on left, we are on right
                current_hash = MerkleTree._hash_pair(sibling_hash, current_hash)
        
        # Compare computed root with expected root
        return current_hash == root_hash
    
    @staticmethod
    def verify_chunk_hex(
        chunk_data: bytes,
        proof_hex: List[Tuple[str, bool]],
        root_hash_hex: str,
        chunk_index: int,
        total_chunks: int,
    ) -> bool:
        """
        Verify chunk with hex-encoded proof and root.
        
        Convenience method for verifying after JSON deserialization.
        """
        proof = [(bytes.fromhex(h), is_right) for h, is_right in proof_hex]
        root_hash = bytes.fromhex(root_hash_hex)
        return MerkleTree.verify_chunk(chunk_data, proof, root_hash, chunk_index, total_chunks)
    
    def get_leaf_hash(self, chunk_index: int) -> bytes:
        """Get the hash of a specific leaf (chunk)."""
        if chunk_index < 0 or chunk_index >= self._leaf_count:
            raise IndexError(f"Chunk index {chunk_index} out of range")
        return self._levels[0][chunk_index]


class SecurityError(Exception):
    """
    Raised when chunk verification fails.
    
    [SECURITY] This exception indicates potential malicious activity.
    The receiving node should:
    1. NOT write the chunk to disk
    2. Log the incident
    3. Consider banning the peer
    """
    
    def __init__(self, message: str, peer_id: str = "", chunk_index: int = -1):
        super().__init__(message)
        self.peer_id = peer_id
        self.chunk_index = chunk_index

