"""Environment file utilities.

[CONFIG] Update .env values programmatically without destroying comments.
"""

from pathlib import Path
from typing import Optional


def update_env_variable(key: str, value: str, env_path: Optional[str] = None) -> None:
    """
    Update or append KEY=value in .env file, preserving comments and other lines.
    
    Args:
        key: Environment variable name (without '=')
        value: New value to set (string)
        env_path: Optional path to .env (defaults to project root/.env)
    """
    path = Path(env_path) if env_path else Path(".env")
    lines = []
    found = False

    if path.exists():
        raw = path.read_text().splitlines()
        for line in raw:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines.append(line)
                continue
            if stripped.startswith(f"{key}="):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        if lines and lines[-1] != "":
            lines.append("")  # ensure newline before append
        lines.append(f"{key}={value}")

    path.write_text("\n".join(lines) + "\n")

