#!/usr/bin/env python3
"""
Build ZEONE documentation (HTML and PDF).
"""

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent
SOURCE = ROOT / "docs" / "source"
BUILD_HTML = ROOT / "docs" / "build" / "html"
BUILD_PDF = ROOT / "docs" / "build" / "pdf"


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=ROOT)


def ensure_deps() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "sphinx", "sphinx-rtd-theme", "myst-parser"]
    )


def build_html() -> int:
    BUILD_HTML.mkdir(parents=True, exist_ok=True)
    return run(["sphinx-build", "-b", "html", str(SOURCE), str(BUILD_HTML)])


def build_pdf() -> int:
    if not shutil.which("pdflatex"):
        print("[WARN] pdflatex not found, skipping PDF build")
        return 0
    BUILD_PDF.mkdir(parents=True, exist_ok=True)
    return run(["sphinx-build", "-b", "latexpdf", str(SOURCE), str(BUILD_PDF)])


def main() -> None:
    ensure_deps()
    rc_html = build_html()
    rc_pdf = build_pdf()
    if rc_html != 0 or rc_pdf != 0:
        sys.exit(rc_html or rc_pdf)
    print(f"Documentation ready at {BUILD_HTML / 'index.html'}")


if __name__ == "__main__":
    main()
