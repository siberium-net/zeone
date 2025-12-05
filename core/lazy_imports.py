"""
Lazy Import System
==================

[LITE MODE] Provides safe lazy imports for optional heavy modules.
Node starts even if AI libraries (torch, transformers) are not installed.

Usage:
    from core.lazy_imports import LazyModule, is_ai_available, AI_MODE

    torch = LazyModule("torch")
    if torch.available:
        tensor = torch.module.zeros(10)
"""

import sys
import logging
from typing import Optional, Any, List, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# Global State
# ============================================================================

@dataclass
class AIStatus:
    """Status of AI module availability."""
    available: bool = False
    torch_available: bool = False
    transformers_available: bool = False
    chromadb_available: bool = False
    insightface_available: bool = False
    missing_modules: List[str] = field(default_factory=list)
    mode: str = "LITE"  # "FULL" or "LITE"


AI_STATUS = AIStatus()


# ============================================================================
# Lazy Module Wrapper
# ============================================================================

class LazyModule:
    """
    Lazy import wrapper for optional modules.
    
    [USAGE]
        torch = LazyModule("torch")
        if torch.available:
            # Safe to use
            tensor = torch.module.zeros(10)
        else:
            logger.warning("torch not available")
    """
    
    _cache: dict = {}
    
    def __init__(self, module_name: str, required_for: str = "AI"):
        self.module_name = module_name
        self.required_for = required_for
        self._module: Optional[Any] = None
        self._attempted: bool = False
        self._error: Optional[str] = None
    
    @property
    def available(self) -> bool:
        """Check if module is importable."""
        if not self._attempted:
            self._try_import()
        return self._module is not None
    
    @property
    def module(self) -> Any:
        """Get the actual module (raises ImportError if not available)."""
        if not self._attempted:
            self._try_import()
        if self._module is None:
            raise ImportError(
                f"Module '{self.module_name}' is not available. "
                f"Required for: {self.required_for}. "
                f"Install with: pip install -r requirements/ai.txt"
            )
        return self._module
    
    @property
    def error(self) -> Optional[str]:
        """Get import error message if any."""
        if not self._attempted:
            self._try_import()
        return self._error
    
    def _try_import(self) -> None:
        """Attempt to import the module."""
        self._attempted = True
        
        # Check cache first
        if self.module_name in LazyModule._cache:
            self._module = LazyModule._cache[self.module_name]
            return
        
        try:
            import importlib
            self._module = importlib.import_module(self.module_name)
            LazyModule._cache[self.module_name] = self._module
            logger.debug(f"[LAZY] Loaded: {self.module_name}")
        except ImportError as e:
            self._error = str(e)
            if self.module_name not in AI_STATUS.missing_modules:
                AI_STATUS.missing_modules.append(self.module_name)
            logger.debug(f"[LAZY] Not available: {self.module_name} ({e})")
    
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the actual module."""
        return getattr(self.module, name)
    
    def __repr__(self) -> str:
        status = "available" if self.available else "not available"
        return f"<LazyModule '{self.module_name}' ({status})>"


# ============================================================================
# Pre-defined Lazy Modules
# ============================================================================

# PyTorch
torch = LazyModule("torch", "Neural networks")
torch_nn = LazyModule("torch.nn", "Neural network layers")

# Transformers
transformers = LazyModule("transformers", "LLM inference")

# Sentence Transformers (embeddings)
sentence_transformers = LazyModule("sentence_transformers", "Text embeddings")

# ChromaDB (vector store)
chromadb = LazyModule("chromadb", "Vector database")

# InsightFace (face recognition)
insightface = LazyModule("insightface", "Face recognition")

# Computer Vision
cv2 = LazyModule("cv2", "Image processing")
PIL = LazyModule("PIL", "Image loading")

# LZ4 compression
lz4 = LazyModule("lz4.frame", "Tensor compression")


# ============================================================================
# Detection Functions
# ============================================================================

def check_ai_availability() -> AIStatus:
    """
    Check availability of AI modules and update global status.
    
    Returns:
        AIStatus with all availability flags
    """
    global AI_STATUS
    
    AI_STATUS.torch_available = torch.available
    AI_STATUS.transformers_available = transformers.available
    AI_STATUS.chromadb_available = chromadb.available
    AI_STATUS.insightface_available = insightface.available
    
    # Full AI mode requires at least torch + transformers
    if AI_STATUS.torch_available and AI_STATUS.transformers_available:
        AI_STATUS.available = True
        AI_STATUS.mode = "FULL"
    else:
        AI_STATUS.available = False
        AI_STATUS.mode = "LITE"
    
    return AI_STATUS


def is_ai_available() -> bool:
    """Check if AI modules are available."""
    if AI_STATUS.mode == "LITE" and not AI_STATUS.missing_modules:
        # First check
        check_ai_availability()
    return AI_STATUS.available


def get_ai_mode() -> str:
    """Get current AI mode (FULL or LITE)."""
    if AI_STATUS.mode == "LITE" and not AI_STATUS.missing_modules:
        check_ai_availability()
    return AI_STATUS.mode


def log_ai_status() -> None:
    """Log AI module availability status."""
    check_ai_availability()
    
    if AI_STATUS.available:
        logger.info("[AI] Running in FULL mode - all AI features available")
        logger.info(f"[AI]   torch: {AI_STATUS.torch_available}")
        logger.info(f"[AI]   transformers: {AI_STATUS.transformers_available}")
        logger.info(f"[AI]   chromadb: {AI_STATUS.chromadb_available}")
        logger.info(f"[AI]   insightface: {AI_STATUS.insightface_available}")
    else:
        logger.info("[AI] Running in LITE mode - AI features disabled")
        if AI_STATUS.missing_modules:
            logger.info(f"[AI]   Missing modules: {', '.join(AI_STATUS.missing_modules)}")
        logger.info("[AI]   Node operates as VPN/Storage/Wallet only")
        logger.info("[AI]   To enable AI: pip install -r requirements/ai.txt")


# ============================================================================
# Safe Import Decorators
# ============================================================================

def requires_ai(func):
    """
    Decorator that marks a function as requiring AI modules.
    Returns None and logs warning if AI not available.
    """
    def wrapper(*args, **kwargs):
        if not is_ai_available():
            logger.warning(
                f"[AI] Function {func.__name__} requires AI modules "
                f"(missing: {', '.join(AI_STATUS.missing_modules)})"
            )
            return None
        return func(*args, **kwargs)
    return wrapper


def requires_torch(func):
    """Decorator requiring torch specifically."""
    def wrapper(*args, **kwargs):
        if not torch.available:
            logger.warning(f"[AI] Function {func.__name__} requires torch")
            return None
        return func(*args, **kwargs)
    return wrapper


# ============================================================================
# Agent Registration Helper
# ============================================================================

def register_ai_agents(agent_manager) -> List[str]:
    """
    Register AI agents only if dependencies are available.
    
    Args:
        agent_manager: AgentManager instance
    
    Returns:
        List of registered AI agent names
    """
    registered = []
    
    check_ai_availability()
    
    if not AI_STATUS.available:
        logger.info("[AI] AI Module not found. Running in LITE mode.")
        return registered
    
    # Try to register LLM agents
    if AI_STATUS.transformers_available:
        try:
            from agents.ai_assistant import LlmAgent
            agent_manager.register_agent(LlmAgent())
            registered.append("llm_cloud")
        except Exception as e:
            logger.warning(f"[AI] LlmAgent registration failed: {e}")
        
        try:
            from agents.local_llm import OllamaAgent
            agent_manager.register_agent(OllamaAgent())
            registered.append("llm_local")
        except Exception as e:
            logger.warning(f"[AI] OllamaAgent registration failed: {e}")
    
    # Try to register Vision agents
    if AI_STATUS.insightface_available:
        try:
            from agents.vision import VisionAgent
            agent_manager.register_agent(VisionAgent())
            registered.append("vision")
        except Exception as e:
            logger.warning(f"[AI] VisionAgent registration failed: {e}")
    
    # Try to register NeuroLink (tensor transport)
    if AI_STATUS.torch_available:
        try:
            from agents.neuro_link import NeuroLinkAgent
            agent_manager.register_agent(NeuroLinkAgent())
            registered.append("neuro_link")
        except Exception as e:
            logger.debug(f"[AI] NeuroLinkAgent registration skipped: {e}")
    
    if registered:
        logger.info(f"[AI] Registered AI agents: {', '.join(registered)}")
    
    return registered

