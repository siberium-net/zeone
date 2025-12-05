"""
ZEONE WebUI Package
===================

NiceGUI-based web interface for ZEONE network management.
"""

try:
    from .app import create_webui, P2PWebUI
    __all__ = ["create_webui", "P2PWebUI", "tabs"]
except ImportError as e:
    # NiceGUI not installed
    import logging
    logging.getLogger(__name__).debug(f"WebUI not available: {e}")
    
    def create_webui(*args, **kwargs):
        raise ImportError(
            "WebUI requires NiceGUI. Install with: pip install nicegui"
        )
    
    __all__ = ["tabs"]
