# -*- mode: python ; coding: utf-8 -*-
"""
ZEONE PyInstaller Specification
===============================

Build targets:
  - zeone-core: Lightweight P2P node (VPN/Storage/Wallet)
  - zeone-full: Full node with AI capabilities

Usage:
    # Build core (lightweight)
    pyinstaller build/zeone.spec -- --target=core
    
    # Build full (with AI)
    pyinstaller build/zeone.spec -- --target=full

[NOTE] This is a template. Use scripts/build_dist.py for automated builds.
"""

import sys
import os
from pathlib import Path

# Get build target from command line
BUILD_TARGET = "core"  # Default: lightweight
for i, arg in enumerate(sys.argv):
    if arg == "--target" and i + 1 < len(sys.argv):
        BUILD_TARGET = sys.argv[i + 1]
    elif arg.startswith("--target="):
        BUILD_TARGET = arg.split("=")[1]

print(f"[BUILD] Target: {BUILD_TARGET}")

# Paths
PROJECT_ROOT = Path(SPECPATH).parent
MAIN_SCRIPT = PROJECT_ROOT / "main.py"

# ============================================================================
# Common Analysis
# ============================================================================

# Hidden imports that PyInstaller often misses
HIDDEN_IMPORTS_CORE = [
    # Async
    "asyncio",
    "aiohttp",
    "aiosqlite",
    "aiofiles",
    
    # Web3
    "web3",
    "web3.providers.rpc",
    "web3.middleware",
    "eth_account",
    "eth_utils",
    "eth_abi",
    
    # Crypto
    "nacl",
    "nacl.signing",
    "nacl.public",
    "nacl.secret",
    
    # Network
    "dns",
    "dns.resolver",
    
    # WebUI (NiceGUI)
    "nicegui",
    "engineio",
    "socketio",
    "uvicorn",
    "starlette",
    "fastapi",
    
    # Database
    "sqlite3",
    
    # Standard
    "json",
    "hashlib",
    "base64",
    "logging",
    "argparse",
    "pathlib",
    "dataclasses",
    
    # Our modules
    "core",
    "core.transport",
    "core.node",
    "core.protocol",
    "core.dht",
    "core.security",
    "economy",
    "economy.ledger",
    "economy.chain",
    "agents",
    "agents.manager",
]

HIDDEN_IMPORTS_AI = [
    # PyTorch
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.cuda",
    
    # Transformers
    "transformers",
    "transformers.models",
    "transformers.pipelines",
    
    # Sentence Transformers
    "sentence_transformers",
    
    # ChromaDB
    "chromadb",
    "chromadb.config",
    
    # Vision
    "cv2",
    "PIL",
    "PIL.Image",
    "insightface",
    
    # Scientific
    "numpy",
    "scipy",
    "sklearn",
    
    # Our AI modules
    "cortex",
    "cortex.distributed",
    "cortex.archivist",
]

# Data files to include
DATAS_CORE = [
    # Config files
    (str(PROJECT_ROOT / "config.py"), "."),
    
    # Data directory (create placeholder)
    (str(PROJECT_ROOT / "data"), "data"),
]

# Check if webui/static exists
WEBUI_STATIC = PROJECT_ROOT / "webui" / "static"
if WEBUI_STATIC.exists():
    DATAS_CORE.append((str(WEBUI_STATIC), "webui/static"))

# Check if docs/build exists
DOCS_BUILD = PROJECT_ROOT / "docs" / "build" / "html"
if DOCS_BUILD.exists():
    DATAS_CORE.append((str(DOCS_BUILD), "docs"))

# Contracts (for settlement)
CONTRACTS_DIR = PROJECT_ROOT / "contracts"
if CONTRACTS_DIR.exists():
    DATAS_CORE.append((str(CONTRACTS_DIR), "contracts"))

# Excludes for lightweight build
EXCLUDES_CORE = [
    # AI/ML libraries (large)
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "sentence_transformers",
    "chromadb",
    "insightface",
    "onnxruntime",
    "onnxruntime_gpu",
    
    # NVIDIA/CUDA
    "nvidia",
    "cuda",
    "cudnn",
    "cublas",
    "cufft",
    "curand",
    "cusparse",
    "nvrtc",
    
    # Scientific (heavy parts)
    "scipy.spatial.transform",
    "scipy.ndimage",
    
    # Testing
    "pytest",
    "unittest",
    "_pytest",
    
    # Development
    "IPython",
    "jupyter",
    "notebook",
    
    # Documentation
    "sphinx",
    "docutils",
]

EXCLUDES_FULL = [
    # Still exclude development tools
    "pytest",
    "unittest",
    "_pytest",
    "IPython",
    "jupyter",
    "notebook",
    "sphinx",
    "docutils",
]

# ============================================================================
# Build Configuration
# ============================================================================

if BUILD_TARGET == "core":
    NAME = "zeone-core"
    HIDDEN_IMPORTS = HIDDEN_IMPORTS_CORE
    EXCLUDES = EXCLUDES_CORE
    DATAS = DATAS_CORE
    ONEFILE = True  # Single executable
    CONSOLE = True
    
elif BUILD_TARGET == "full":
    NAME = "zeone-full"
    HIDDEN_IMPORTS = HIDDEN_IMPORTS_CORE + HIDDEN_IMPORTS_AI
    EXCLUDES = EXCLUDES_FULL
    DATAS = DATAS_CORE
    ONEFILE = False  # Directory mode (too large for onefile)
    CONSOLE = True
    
else:
    raise ValueError(f"Unknown target: {BUILD_TARGET}")

# ============================================================================
# Analysis
# ============================================================================

a = Analysis(
    [str(MAIN_SCRIPT)],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ============================================================================
# PYZ (Python archive)
# ============================================================================

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

# ============================================================================
# EXE
# ============================================================================

if ONEFILE:
    # Single file executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=CONSOLE,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    # Directory mode (onedir)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,
        upx=True,
        console=CONSOLE,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=True,
        upx=True,
        upx_exclude=[],
        name=NAME,
    )

