"""
Distribution Module - File chunking and Merkle Tree manifest
============================================================

[SECURITY] This module provides secure file distribution with
Merkle Tree based chunk verification. Each chunk can be verified
independently without downloading the entire file.

Key Features:
- 1MB chunk size for parallel downloads
- Merkle root in manifest (not full hash list)
- Per-chunk verification via Merkle proofs
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.security.merkle import MerkleTree

CHUNK_SIZE = 1024 * 1024  # 1 MB


def sha256_chunk(data: bytes) -> bytes:
    """Compute SHA-256 hash of chunk data (returns bytes)."""
    return hashlib.sha256(data).digest()


def sha256_file(path: Path) -> str:
    """
    Compute SHA-256 hash of entire file.
    
    [LEGACY] This function is kept for backwards compatibility.
    New code should use Merkle tree verification instead.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_chunk_hashes(path: Path) -> List[bytes]:
    """
    Compute SHA-256 hashes for all chunks in a file.
    
    Args:
        path: Path to the file
    
    Returns:
        List of 32-byte hashes, one per chunk
    
    [SECURITY] These hashes form the leaves of the Merkle tree.
    """
    hashes: List[bytes] = []
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hashes.append(sha256_chunk(chunk))
    return hashes


def build_file_merkle_tree(path: Path) -> MerkleTree:
    """
    Build a Merkle tree for a file's chunks.
    
    Args:
        path: Path to the file
    
    Returns:
        MerkleTree instance with all chunk hashes
    
    [SECURITY] The tree's root hash commits to all chunks.
    Any modification to any chunk will change the root.
    """
    chunk_hashes = compute_chunk_hashes(path)
    return MerkleTree(chunk_hashes)


def get_chunk_count(file_size: int) -> int:
    """Calculate number of chunks for a given file size."""
    return (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE


def build_manifest(model_id: str, base_dir: Path) -> Dict[str, Any]:
    """
    Scan model directory and build manifest with Merkle roots.
    
    Args:
        model_id: Identifier for the model
        base_dir: Base directory containing model directories
    
    Returns:
        Manifest dict with structure:
        {
            "model_id": "...",
            "files": [
                {
                    "name": "relative/path.bin",
                    "size": 1234567890,
                    "chunk_size": 1048576,
                    "merkle_root": "hex_string",
                    "total_chunks": 1234
                },
                ...
            ]
        }
    
    [SECURITY] The merkle_root allows verification of individual
    chunks without needing to store all chunk hashes in the manifest.
    """
    root = base_dir / model_id
    files: List[Dict[str, Any]] = []
    
    if not root.exists():
        return {"model_id": model_id, "files": []}
    
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            size = p.stat().st_size
            total_chunks = get_chunk_count(size)
            
            # Build Merkle tree and get root
            tree = build_file_merkle_tree(p)
            
            files.append({
                "name": rel,
                "size": size,
                "chunk_size": CHUNK_SIZE,
                "merkle_root": tree.root_hex,
                "total_chunks": total_chunks,
            })
    
    return {"model_id": model_id, "files": files}


def read_chunk(path: Path, chunk_index: int) -> bytes:
    """
    Read a specific chunk from a file.
    
    Args:
        path: Path to the file
        chunk_index: 0-based chunk index
    
    Returns:
        Chunk data (up to CHUNK_SIZE bytes, may be less for last chunk)
    """
    with path.open("rb") as f:
        f.seek(chunk_index * CHUNK_SIZE)
        return f.read(CHUNK_SIZE)


def write_chunk(path: Path, chunk_index: int, data: bytes) -> None:
    """
    Write a chunk to a file at the correct offset.
    
    Args:
        path: Path to the file
        chunk_index: 0-based chunk index
        data: Chunk data to write
    
    [NOTE] Creates parent directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r+b" if path.exists() else "wb") as f:
        f.seek(chunk_index * CHUNK_SIZE)
        f.write(data)


def read_chunk_with_proof(
    path: Path,
    chunk_index: int,
    tree: Optional[MerkleTree] = None,
) -> Dict[str, Any]:
    """
    Read a chunk along with its Merkle proof.
    
    Args:
        path: Path to the file
        chunk_index: 0-based chunk index
        tree: Pre-built MerkleTree (builds new one if None)
    
    Returns:
        Dict with:
        {
            "data": bytes,
            "proof": [(hash_hex, is_right), ...],
            "total_chunks": int,
            "chunk_index": int
        }
    
    [SECURITY] The proof allows the receiver to verify the chunk
    against the merkle_root without trusting the sender.
    """
    # Build tree if not provided
    if tree is None:
        tree = build_file_merkle_tree(path)
    
    # Read chunk data
    data = read_chunk(path, chunk_index)
    
    # Get Merkle proof (hex format for JSON serialization)
    proof = tree.get_proof_hex(chunk_index)
    
    return {
        "data": data,
        "proof": proof,
        "total_chunks": tree.leaf_count,
        "chunk_index": chunk_index,
    }
