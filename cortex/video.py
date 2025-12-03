import logging
from pathlib import Path
from typing import List, Dict, Any

import cv2

from cortex.vision import VisionEngine
from agents.local_llm import OllamaAgent

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Extract key frames and summarize video content."""

    def __init__(self, vision: VisionEngine, llm: OllamaAgent):
        self.vision = vision
        self.llm = llm

    def extract_smart_frames(self, video_path: str, stride_seconds: float = 3.0) -> List[str]:
        """Save sampled frames to temp files and return their paths."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        stride = max(int(fps * stride_seconds), 1)

        frames_dir = Path(video_path).parent / ".frames"
        frames_dir.mkdir(exist_ok=True, parents=True)

        saved = []
        idx = 0
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frame_path = frames_dir / f"frame_{frame_id:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved.append(str(frame_path))
                frame_id += 1
            idx += 1
        cap.release()
        logger.info(f"[VIDEO] Extracted {len(saved)} frames from {video_path}")
        return saved

    async def process_video(self, video_path: str) -> Dict[str, Any]:
        """Extract frames, run vision, summarize with LLM."""
        frame_paths = self.extract_smart_frames(video_path)
        descriptions = []
        for fp in frame_paths:
            try:
                res = self.vision.analyze_image(fp)
                desc = res.get("description", "")
                if desc:
                    descriptions.append(desc)
            except Exception as e:
                logger.warning(f"[VIDEO] Frame {fp} failed: {e}")
        joined = "\n".join(descriptions)
        prompt = f"Summarize this video based on these frame descriptions:\n{joined}"
        result, _ = await self.llm.execute({"prompt": prompt})
        return {
            "frame_descriptions": descriptions,
            "summary": result.get("response") if isinstance(result, dict) else result,
        }
