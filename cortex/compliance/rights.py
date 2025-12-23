import base64
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:
    import chromadb  # type: ignore
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]

from cortex.dedup import phash_image, hamming_distance
from core.transport import Crypto

if TYPE_CHECKING:
    from core.dht.node import KademliaNode
    from core.dht.storage import DHTStorage

try:
    from core.dht.storage import string_to_key
except Exception:  # pragma: no cover
    string_to_key = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class FingerprintRecord:
    record_id: str
    content_id: str
    phash: str
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass
class RightsMatchResult:
    status: str = "UNLICENSED"
    similarity: float = 0.0
    matched_content_id: Optional[str] = None
    matched_record_id: Optional[str] = None
    beneficiary_address: Optional[str] = None
    revenue_share: bool = False


@dataclass
class MagnetMetadata:
    info_hash: str
    name: str
    size_bytes: int
    piece_length: int
    num_pieces: int
    magnet_uri: Optional[str] = None
    storyboard: List[Dict[str, Any]] = field(default_factory=list)
    rights_status: str = "UNKNOWN"
    beneficiary_address: Optional[str] = None
    discoverer_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    signature: Optional[str] = None

    def to_dict(self, include_signature: bool = True) -> Dict[str, Any]:
        data = {
            "info_hash": self.info_hash,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "piece_length": self.piece_length,
            "num_pieces": self.num_pieces,
            "magnet_uri": self.magnet_uri,
            "storyboard": self.storyboard,
            "rights_status": self.rights_status,
            "beneficiary_address": self.beneficiary_address,
            "discoverer_id": self.discoverer_id,
            "created_at": self.created_at,
        }
        if include_signature:
            data["signature"] = self.signature
        return data

    def signing_payload(self) -> bytes:
        return json.dumps(self.to_dict(include_signature=False), sort_keys=True).encode("utf-8")


class RightsManager:
    """Decentralized Content ID and rights resolver backed by ChromaDB."""

    BENEFICIARY_PREFIX = "rights:beneficiary:"

    def __init__(
        self,
        persist_path: str = "vector_store/rights",
        kademlia: Optional["KademliaNode"] = None,
        dht_storage: Optional["DHTStorage"] = None,
        beneficiary_resolver: Optional[Any] = None,
    ):
        self.kademlia = kademlia
        self.storage = dht_storage
        self.beneficiary_resolver = beneficiary_resolver
        self.safe_mode = False
        self._memory_store: List[FingerprintRecord] = []

        if chromadb is None:
            self.safe_mode = True
            self.client = None
            self.collection = None
            logger.warning("[RIGHTS] chromadb not installed; using in-memory fingerprints")
            return

        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            "content_rights",
            metadata={"hnsw:space": "cosine"},
        )

    async def register_protected_content(
        self,
        content_id: str,
        frame_paths: Sequence[str],
        beneficiary_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        count = 0
        for idx, frame_path in enumerate(frame_paths):
            try:
                phash = phash_image(frame_path)
                embedding = self._phash_to_vector(phash)
                record_id = f"{content_id}:{idx}"
                record_meta = {
                    "content_id": content_id,
                    "phash": phash,
                }
                if beneficiary_address:
                    record_meta["beneficiary_address"] = beneficiary_address
                if metadata:
                    record_meta.update(metadata)
                self._store_record(record_id, content_id, phash, embedding, record_meta)
                count += 1
            except Exception as e:
                logger.warning(f"[RIGHTS] Failed to fingerprint {frame_path}: {e}")

        if beneficiary_address:
            await self._publish_beneficiary(content_id, beneficiary_address)
        return count

    async def match_frames(
        self,
        frame_paths: Sequence[str],
        similarity_threshold: float = 0.9,
    ) -> RightsMatchResult:
        best = RightsMatchResult()
        for frame_path in frame_paths:
            try:
                phash = phash_image(frame_path)
                embedding = self._phash_to_vector(phash)
                match = self._query_best_match(embedding)
                if not match:
                    continue
                match_id, match_meta = match
                match_phash = match_meta.get("phash", "")
                similarity = self._phash_similarity(phash, match_phash)
                if similarity > best.similarity:
                    best = RightsMatchResult(
                        status="LICENSED" if similarity >= similarity_threshold else "UNLICENSED",
                        similarity=similarity,
                        matched_content_id=match_meta.get("content_id"),
                        matched_record_id=match_id,
                        beneficiary_address=match_meta.get("beneficiary_address"),
                        revenue_share=similarity >= similarity_threshold,
                    )
            except Exception as e:
                logger.warning(f"[RIGHTS] Match failed for {frame_path}: {e}")

        if best.revenue_share and best.matched_content_id:
            beneficiary = await self.resolve_beneficiary(
                best.matched_content_id,
                fallback=best.beneficiary_address,
            )
            best.beneficiary_address = beneficiary
        return best

    async def resolve_beneficiary(
        self,
        content_id: str,
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        if self.beneficiary_resolver:
            try:
                result = self.beneficiary_resolver(content_id)
                if hasattr(result, "__await__"):
                    result = await result
                if result:
                    return str(result)
            except Exception as e:
                logger.warning(f"[RIGHTS] Beneficiary resolver failed: {e}")

        key = f"{self.BENEFICIARY_PREFIX}{content_id}"
        if self.kademlia:
            try:
                raw = await self.kademlia.dht_get(key)
                if raw:
                    data = json.loads(raw.decode("utf-8"))
                    return data.get("beneficiary_address")
            except Exception as e:
                logger.warning(f"[RIGHTS] DHT beneficiary lookup failed: {e}")

        if self.storage and string_to_key is not None:
            try:
                stored = await self.storage.get(string_to_key(key))
                if stored:
                    data = json.loads(stored.value.decode("utf-8"))
                    return data.get("beneficiary_address")
            except Exception as e:
                logger.warning(f"[RIGHTS] Local beneficiary lookup failed: {e}")

        return fallback

    def _store_record(
        self,
        record_id: str,
        content_id: str,
        phash: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        if self.safe_mode or not self.collection:
            self._memory_store.append(
                FingerprintRecord(
                    record_id=record_id,
                    content_id=content_id,
                    phash=phash,
                    embedding=embedding,
                    metadata=metadata,
                )
            )
            return

        self.collection.add(
            ids=[record_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def _query_best_match(self, embedding: List[float]) -> Optional[Tuple[str, Dict[str, Any]]]:
        if self.safe_mode or not self.collection:
            best: Optional[Tuple[str, Dict[str, Any], float]] = None
            for record in self._memory_store:
                similarity = self._cosine_similarity(embedding, record.embedding)
                if best is None or similarity > best[2]:
                    best = (record.record_id, record.metadata, similarity)
            if best:
                return best[0], best[1]
            return None

        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["metadatas", "ids"],
        )
        if not result or not result.get("ids"):
            return None
        match_id = result["ids"][0][0]
        match_meta = result.get("metadatas", [[{}]])[0][0]
        return match_id, match_meta

    async def _publish_beneficiary(self, content_id: str, beneficiary_address: str) -> None:
        payload = json.dumps({"beneficiary_address": beneficiary_address}).encode("utf-8")
        key = f"{self.BENEFICIARY_PREFIX}{content_id}"
        if self.kademlia:
            try:
                await self.kademlia.dht_put(key, payload)
                return
            except Exception as e:
                logger.warning(f"[RIGHTS] Failed to store beneficiary in DHT: {e}")
        if self.storage and string_to_key is not None:
            try:
                await self.storage.store(string_to_key(key), payload, string_to_key("rights"))
            except Exception as e:
                logger.warning(f"[RIGHTS] Failed to store beneficiary locally: {e}")

    def _phash_to_vector(self, phash: str) -> List[float]:
        if not phash:
            return []
        bits = bin(int(phash, 16))[2:].zfill(len(phash) * 4)
        return [1.0 if b == "1" else 0.0 for b in bits]

    def _phash_similarity(self, phash_a: str, phash_b: str) -> float:
        if not phash_a or not phash_b:
            return 0.0
        bit_len = max(len(phash_a), len(phash_b)) * 4
        if bit_len == 0:
            return 0.0
        dist = hamming_distance(phash_a, phash_b)
        return max(0.0, 1.0 - (dist / float(bit_len)))

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class IndexPublisher:
    """Publishes signed MagnetMetadata payloads to the DHT."""

    def __init__(
        self,
        kademlia: Optional["KademliaNode"] = None,
        dht_storage: Optional["DHTStorage"] = None,
        identity_path: str = "identity.key",
    ):
        self.kademlia = kademlia
        self.storage = dht_storage
        self.crypto = self._load_identity(identity_path)

    async def publish(self, metadata: MagnetMetadata) -> str:
        if not metadata.discoverer_id:
            metadata.discoverer_id = self.crypto.node_id
        payload = metadata.signing_payload()
        metadata.signature = self._sign_payload(payload)
        encoded = json.dumps(metadata.to_dict(), sort_keys=True).encode("utf-8")
        key = f"magnet:index:{metadata.info_hash}"
        await self._dht_put(key, encoded)
        return key

    def _sign_payload(self, payload: bytes) -> str:
        signed = self.crypto.signing_key.sign(payload)
        return base64.b64encode(signed.signature).decode("ascii")

    def _load_identity(self, identity_path: str) -> Crypto:
        path = Path(identity_path)
        if path.exists():
            try:
                key_bytes = path.read_bytes()
                if len(key_bytes) == 32:
                    return Crypto.import_identity(key_bytes)
            except Exception as e:
                logger.warning(f"[INDEX] Failed to load identity key: {e}")
        return Crypto()

    async def _dht_put(self, key: str, value: bytes) -> None:
        if self.kademlia:
            await self.kademlia.dht_put(key, value)
            return
        if self.storage and string_to_key is not None:
            publisher_id = string_to_key(self.crypto.node_id)
            await self.storage.store(string_to_key(key), value, publisher_id)
            return
        logger.warning("[INDEX] No DHT backend available for publishing")
