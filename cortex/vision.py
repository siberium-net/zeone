import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from PIL import Image

from cortex.dedup import phash_image, is_duplicate

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HF_AVAILABLE = False


class VisionEngine:
    """Image analysis via Florence-2-large."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        if not _HF_AVAILABLE:
            logger.warning("[VISION] transformers not available; vision disabled")
            return
        try:
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")
            self.model = AutoModelForVision2Seq.from_pretrained(
                "microsoft/Florence-2-large",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)
            self.model.eval()
            logger.info(f"[VISION] Florence loaded on {self.device}")
        except Exception as e:
            logger.warning(f"[VISION] Failed to load Florence: {e}")
            self.model = None
            self.processor = None

    def _run_task(self, image: Image.Image, task: str) -> str:
        if not self.model or not self.processor:
            return ""
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=3,
            )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def analyze_image(
        self,
        image_path: str,
        known_hashes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run multi-task analysis (caption, OD, OCR, brands) with dedup check."""
        img = Image.open(Path(image_path)).convert("RGB")
        img_hash = phash_image(str(image_path))
        if known_hashes:
            for h in known_hashes:
                if is_duplicate(img_hash, h):
                    return {"duplicate": True, "phash": img_hash}

        description = self._run_task(img, "<MORE_DETAILED_CAPTION>")
        objects_raw = self._run_task(img, "<OD>")
        text_raw = self._run_task(img, "<OCR>")
        brands_raw = self._run_task(img, "<OCR_WITH_REGION>")

        tags: List[str] = []
        if objects_raw:
            tags = [t.strip() for t in objects_raw.replace("\n", " ").split(",") if t.strip()]

        brands: List[str] = []
        if brands_raw:
            brands = [b.strip() for b in brands_raw.replace("\n", " ").split(",") if b.strip()]

        return {
            "description": description,
            "objects": objects_raw,
            "text": text_raw,
            "tags": tags,
            "brands": brands,
            "phash": img_hash,
        }
