import cv2
import imagehash
from PIL import Image
from pathlib import Path
from typing import List, Tuple


def phash_image(path: str) -> str:
    """Compute perceptual hash for an image."""
    img = Image.open(path).convert("RGB")
    return str(imagehash.phash(img))


def phash_video(path: str, fps_sample: float = 1.0) -> List[str]:
    """
    Compute video signature: pHash per keyframe (sampled every ~1 second).
    Returns list of hash strings.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    stride = max(int(fps // fps_sample), 1)
    hashes = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frame_path = Path(path).parent / f".tmp_frame_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            try:
                h = phash_image(str(frame_path))
                hashes.append(h)
            finally:
                frame_path.unlink(missing_ok=True)
        idx += 1
    cap.release()
    return hashes


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between hex string hashes."""
    a = int(hash_a, 16)
    b = int(hash_b, 16)
    return bin(a ^ b).count("1")


def is_duplicate(hash_a: str, hash_b: str, threshold: int = 5) -> bool:
    """True if perceptual hashes are within threshold."""
    return hamming_distance(hash_a, hash_b) < threshold


def find_overlap(sig_a: List[str], sig_b: List[str], min_match: int = 3) -> Tuple[int, int]:
    """
    Find overlap between two video signatures (list of frame hashes).
    Returns (start_index_in_b, match_length). If no overlap, (-1, 0).
    Simple substring search on hashes.
    """
    if not sig_a or not sig_b:
        return -1, 0
    len_a, len_b = len(sig_a), len(sig_b)
    for i in range(len_b - len_a + 1):
        match_len = 0
        for j in range(len_a):
            if i + j >= len_b:
                break
            if is_duplicate(sig_a[j], sig_b[i + j], threshold=5):
                match_len += 1
            else:
                break
        if match_len >= min_match:
            return i, match_len
    return -1, 0
