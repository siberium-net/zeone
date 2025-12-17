"""
WebUI Tabs Package
==================

NiceGUI tabs for ZEONE web interface.
"""

from .wallet import WalletTab
from .settings import SettingsTab
from .genesis import GenesisTab
from .learning import LearningTab

__all__ = ["WalletTab", "SettingsTab", "GenesisTab", "LearningTab"]
