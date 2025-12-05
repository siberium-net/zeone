#!/usr/bin/env python3
"""
ZEONE Distribution Builder
==========================

[BUILD] Automated build script for standalone distributions.

Targets:
    core    - Lightweight P2P node (VPN/Storage/Wallet) ~50-100MB
    full    - Full node with AI capabilities ~3-5GB

Usage:
    python scripts/build_dist.py [core|full|all] [--no-clean]

Output:
    dist/zeone-v{VERSION}-{OS}-{ARCH}.zip
"""

import os
import sys
import shutil
import platform
import subprocess
import argparse
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"
SPEC_FILE = BUILD_DIR / "zeone.spec"

# Version from package or default
VERSION = "1.0.0"
try:
    # Try to get version from git tag
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode == 0:
        VERSION = result.stdout.strip().lstrip("v")
except Exception:
    pass

# Platform detection
OS_NAME = platform.system().lower()  # linux, darwin, windows
ARCH = platform.machine().lower()  # x86_64, arm64, aarch64

# Normalize architecture names
if ARCH in ("x86_64", "amd64"):
    ARCH = "x64"
elif ARCH in ("arm64", "aarch64"):
    ARCH = "arm64"


# ============================================================================
# Console Helpers
# ============================================================================

class Console:
    """Formatted console output."""
    
    @staticmethod
    def header(text: str):
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def info(text: str):
        print(f"[INFO] {text}")
    
    @staticmethod
    def success(text: str):
        print(f"[OK] {text}")
    
    @staticmethod
    def warning(text: str):
        print(f"[WARN] {text}")
    
    @staticmethod
    def error(text: str):
        print(f"[ERROR] {text}", file=sys.stderr)
    
    @staticmethod
    def step(num: int, total: int, text: str):
        print(f"\n[{num}/{total}] {text}")
        print("-" * 40)


# ============================================================================
# Build Functions
# ============================================================================

def check_prerequisites() -> bool:
    """Check if build prerequisites are met."""
    Console.info("Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        Console.error(f"Python 3.9+ required, got {sys.version}")
        return False
    
    # Check PyInstaller
    try:
        import PyInstaller
        Console.info(f"PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        Console.error("PyInstaller not found. Install with: pip install pyinstaller")
        return False
    
    # Check spec file
    if not SPEC_FILE.exists():
        Console.error(f"Spec file not found: {SPEC_FILE}")
        return False
    
    Console.success("Prerequisites OK")
    return True


def install_requirements(target: str) -> bool:
    """Install requirements for the target build."""
    Console.info(f"Installing requirements for {target}...")
    
    req_files = [PROJECT_ROOT / "requirements" / "core.txt"]
    
    if target == "full":
        req_files.append(PROJECT_ROOT / "requirements" / "ai.txt")
    
    for req_file in req_files:
        if not req_file.exists():
            Console.warning(f"Requirements file not found: {req_file}")
            continue
        
        Console.info(f"Installing {req_file.name}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            Console.error(f"Failed to install {req_file.name}")
            Console.error(result.stderr)
            return False
    
    Console.success("Requirements installed")
    return True


def clean_build() -> None:
    """Clean previous build artifacts."""
    Console.info("Cleaning previous builds...")
    
    # Clean PyInstaller cache
    for cache_dir in ["__pycache__", "build", "dist"]:
        cache_path = PROJECT_ROOT / cache_dir
        if cache_path.exists() and cache_dir != "build":  # Keep our build/ dir
            shutil.rmtree(cache_path)
    
    # Clean PyInstaller work dir
    pyinstaller_build = PROJECT_ROOT / "build" / "zeone-core"
    if pyinstaller_build.exists():
        shutil.rmtree(pyinstaller_build)
    
    pyinstaller_build = PROJECT_ROOT / "build" / "zeone-full"
    if pyinstaller_build.exists():
        shutil.rmtree(pyinstaller_build)
    
    Console.success("Cleaned")


def run_pyinstaller(target: str) -> Tuple[bool, Path]:
    """Run PyInstaller for the specified target."""
    Console.info(f"Building {target}...")
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(SPEC_FILE),
        "--",
        f"--target={target}",
    ]
    
    Console.info(f"Command: {' '.join(cmd)}")
    
    # Run PyInstaller
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_ROOT,
    )
    
    # Stream output
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            # Filter verbose output
            if "INFO:" in line or "WARNING:" in line or "ERROR:" in line:
                print(f"  {line.rstrip()}")
    
    if process.returncode != 0:
        Console.error(f"PyInstaller failed with code {process.returncode}")
        return False, Path()
    
    # Find output
    if target == "core":
        output_path = DIST_DIR / f"zeone-{target}"
        if not output_path.exists():
            output_path = DIST_DIR / f"zeone-{target}.exe"  # Windows
    else:
        output_path = DIST_DIR / f"zeone-{target}"  # Directory for full
    
    if not output_path.exists():
        Console.error(f"Output not found: {output_path}")
        return False, Path()
    
    Console.success(f"Built: {output_path}")
    return True, output_path


def create_archive(source: Path, target: str) -> Optional[Path]:
    """Create distribution archive."""
    Console.info("Creating archive...")
    
    # Archive name
    archive_name = f"zeone-v{VERSION}-{OS_NAME}-{ARCH}-{target}"
    
    if OS_NAME == "windows":
        archive_path = DIST_DIR / f"{archive_name}.zip"
    else:
        archive_path = DIST_DIR / f"{archive_name}.tar.gz"
    
    # Remove existing archive
    if archive_path.exists():
        archive_path.unlink()
    
    # Create archive
    if source.is_file():
        # Single file (core target)
        if OS_NAME == "windows":
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(source, source.name)
        else:
            with tarfile.open(archive_path, "w:gz") as tf:
                tf.add(source, arcname=source.name)
    else:
        # Directory (full target)
        if OS_NAME == "windows":
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in source.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(source.parent)
                        zf.write(file_path, arcname)
        else:
            with tarfile.open(archive_path, "w:gz") as tf:
                tf.add(source, arcname=source.name)
    
    # Get size
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    
    Console.success(f"Archive created: {archive_path.name} ({size_mb:.1f} MB)")
    return archive_path


def build_target(target: str, no_clean: bool = False) -> bool:
    """Build a specific target."""
    Console.header(f"Building ZEONE {target.upper()}")
    
    total_steps = 4 if not no_clean else 3
    step = 0
    
    # Step 1: Clean (optional)
    if not no_clean:
        step += 1
        Console.step(step, total_steps, "Cleaning")
        clean_build()
    
    # Step 2: Install requirements
    step += 1
    Console.step(step, total_steps, "Installing Requirements")
    if not install_requirements(target):
        return False
    
    # Step 3: Build
    step += 1
    Console.step(step, total_steps, "Running PyInstaller")
    success, output_path = run_pyinstaller(target)
    if not success:
        return False
    
    # Step 4: Archive
    step += 1
    Console.step(step, total_steps, "Creating Archive")
    archive_path = create_archive(output_path, target)
    if not archive_path:
        return False
    
    Console.header("Build Complete")
    Console.info(f"Target: {target}")
    Console.info(f"Version: {VERSION}")
    Console.info(f"Platform: {OS_NAME}-{ARCH}")
    Console.info(f"Archive: {archive_path}")
    
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build ZEONE standalone distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Targets:
  core    Lightweight P2P node (no AI) - Single executable ~50-100MB
  full    Full node with AI - Directory ~3-5GB
  all     Build both targets

Examples:
  python scripts/build_dist.py core
  python scripts/build_dist.py full --no-clean
  python scripts/build_dist.py all
""",
    )
    
    parser.add_argument(
        "target",
        choices=["core", "full", "all"],
        default="core",
        nargs="?",
        help="Build target (default: core)",
    )
    
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning previous builds",
    )
    
    parser.add_argument(
        "--version",
        type=str,
        default=VERSION,
        help=f"Version string (default: {VERSION})",
    )
    
    args = parser.parse_args()
    
    global VERSION
    VERSION = args.version
    
    Console.header("ZEONE Distribution Builder")
    Console.info(f"Version: {VERSION}")
    Console.info(f"Platform: {OS_NAME}-{ARCH}")
    Console.info(f"Python: {sys.version.split()[0]}")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Ensure dist directory exists
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build targets
    targets = ["core", "full"] if args.target == "all" else [args.target]
    
    results = {}
    for target in targets:
        success = build_target(target, args.no_clean)
        results[target] = success
        
        if not success:
            Console.error(f"Build failed: {target}")
            if len(targets) > 1:
                Console.warning("Continuing with other targets...")
    
    # Summary
    Console.header("Build Summary")
    for target, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        Console.info(f"{target}: {status}")
    
    # Exit code
    if all(results.values()):
        Console.success("All builds completed successfully")
        sys.exit(0)
    else:
        Console.error("Some builds failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

