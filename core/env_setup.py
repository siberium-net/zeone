import os
from pathlib import Path


def configure_environment(root_path: Path) -> None:
    """
    Configure portable cache/model directories before heavy imports.
    All downloaded artifacts stay under ./data to avoid permission issues.
    """
    data_dir = root_path / "data"
    models_dir = data_dir / "models"
    cache_dir = data_dir / "cache"
    tmp_dir = data_dir / "tmp"

    for path in (models_dir, cache_dir, tmp_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(models_dir / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(models_dir / "sentence_transformers"))
    os.environ.setdefault("TORCH_HOME", str(models_dir / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
