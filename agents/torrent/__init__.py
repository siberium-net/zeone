"""Torrent integration package."""

from .client import TorrentManager
from .smart_client import SmartTorrentClient, RightsInfo

__all__ = ["TorrentManager", "SmartTorrentClient", "RightsInfo"]
