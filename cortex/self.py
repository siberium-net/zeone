"""
Self-Awareness Module
=====================
Hardware/network diagnostics to determine a node's niche.

This module is designed to be safe in constrained environments:
- Optional GPU detection (pynvml/torch) with graceful fallback
- Optional STUN-based public IP/NAT discovery (disabled by default)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class NodeNiche(Enum):
    """Possible node niches."""

    NEURAL_MINER = "neural_miner"
    TRAFFIC_WEAVER = "traffic_weaver"
    STORAGE_KEEPER = "storage_keeper"
    ARBITRAGEUR = "arbitrageur"
    CHAIN_WEAVER = "chain_weaver"


@dataclass
class HardwareProfile:
    """Node hardware profile."""

    cpu_cores: int = 0
    cpu_freq_mhz: float = 0.0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_speed_mbps: float = 0.0

    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_compute_capability: str = ""


@dataclass
class NetworkProfile:
    """Node network profile."""

    public_ip: str = ""
    local_ip: str = ""
    nat_type: str = "unknown"
    download_mbps: float = 0.0
    upload_mbps: float = 0.0
    latency_ms: float = 0.0
    is_public: bool = False


@dataclass
class NodeProfile:
    """Full node profile."""

    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    network: NetworkProfile = field(default_factory=NetworkProfile)
    niches: List[NodeNiche] = field(default_factory=list)
    dominant_niche: Optional[NodeNiche] = None
    tags: List[str] = field(default_factory=list)


class SelfAwareness:
    """
    Diagnostics helper.

    Example:
        awareness = SelfAwareness()
        profile = await awareness.diagnose()
        print(profile.dominant_niche)
    """

    def __init__(
        self,
        *,
        enable_stun: Optional[bool] = None,
        stun_timeout_s: float = 2.0,
    ):
        env_flag = os.getenv("ZEONE_SELF_ENABLE_STUN", "").strip().lower()
        self._enable_stun = enable_stun if enable_stun is not None else env_flag in {"1", "true", "yes", "on"}
        self._stun_timeout_s = float(stun_timeout_s)
        self._profile: Optional[NodeProfile] = None

    async def diagnose(self) -> NodeProfile:
        """Run full diagnostics and return a NodeProfile."""
        profile = NodeProfile()

        logger.info("[SELF] Diagnosing hardware...")
        profile.hardware = await self._scan_hardware()
        logger.info("[SELF] Diagnosing network...")
        profile.network = await self._scan_network()
        profile.tags = self._compute_tags(profile)
        profile.niches = self._compute_niches(profile)
        profile.dominant_niche = self._select_dominant_niche(profile)

        self._profile = profile
        logger.info("[SELF] Diagnosis complete: %s", profile.dominant_niche)
        return profile

    async def _scan_hardware(self) -> HardwareProfile:
        hw = HardwareProfile()

        try:
            import psutil  # type: ignore
        except Exception:
            psutil = None  # type: ignore

        # CPU
        if psutil is not None:
            hw.cpu_cores = psutil.cpu_count(logical=True) or 1
            freq = psutil.cpu_freq()
            hw.cpu_freq_mhz = float(freq.current) if freq else 0.0
        else:
            hw.cpu_cores = os.cpu_count() or 1
            hw.cpu_freq_mhz = 0.0

        # RAM
        if psutil is not None:
            mem = psutil.virtual_memory()
            hw.ram_total_gb = float(mem.total) / (1024**3)
            hw.ram_available_gb = float(mem.available) / (1024**3)
        else:
            total_b = 0
            avail_b = 0
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_b = int(line.split()[1]) * 1024
                        elif line.startswith("MemAvailable:"):
                            avail_b = int(line.split()[1]) * 1024
                if total_b > 0:
                    hw.ram_total_gb = float(total_b) / (1024**3)
                if avail_b > 0:
                    hw.ram_available_gb = float(avail_b) / (1024**3)
            except Exception:
                pass
        
        # Disk
        if psutil is not None:
            disk = psutil.disk_usage(os.path.abspath(os.sep))
            hw.disk_total_gb = float(disk.total) / (1024**3)
            hw.disk_free_gb = float(disk.free) / (1024**3)
        else:
            try:
                total, _used, free = shutil.disk_usage(os.path.abspath(os.sep))
                hw.disk_total_gb = float(total) / (1024**3)
                hw.disk_free_gb = float(free) / (1024**3)
            except Exception:
                pass

        # GPU
        (
            hw.gpu_available,
            hw.gpu_name,
            hw.gpu_vram_gb,
            hw.gpu_compute_capability,
        ) = await self._detect_gpu()

        return hw

    async def _detect_gpu(self) -> tuple[bool, str, float, str]:
        """Return (available, name, vram_gb, compute_capability)."""
        # NVIDIA (pynvml)
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode(errors="ignore")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = float(mem_info.total) / (1024**3)
            try:
                cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{cc[0]}.{cc[1]}" if isinstance(cc, tuple) else ""
            except Exception:
                compute_capability = ""
            pynvml.nvmlShutdown()
            return True, str(name), vram_gb, compute_capability
        except Exception:
            pass

        # Torch fallback
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = float(props.total_memory) / (1024**3)
                compute_capability = ""
                try:
                    compute_capability = f"{props.major}.{props.minor}"
                except Exception:
                    compute_capability = ""
                return True, str(name), vram_gb, compute_capability
        except Exception:
            pass

        # nvidia-smi fallback (no Python deps)
        try:
            # Example line:
            # "NVIDIA GeForce RTX 3090, 24576, 8.6"
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if proc.returncode == 0:
                line = (proc.stdout or "").strip().splitlines()[0].strip()
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        vram_gb = float(parts[1]) / 1024.0
                    except Exception:
                        vram_gb = 0.0
                    compute_capability = parts[2] if len(parts) >= 3 else ""
                    return True, name, vram_gb, compute_capability
        except Exception:
            pass

        return False, "", 0.0, ""

    async def _scan_network(self) -> NetworkProfile:
        net = NetworkProfile()

        net.local_ip = self._get_local_ipv4()

        # Optional STUN (may require external network; disabled by default)
        if self._enable_stun:
            try:
                from core.nat import STUNClient  # Local dependency (core)

                stun = STUNClient()
                mapped = await asyncio.wait_for(stun.get_mapped_address(0), timeout=self._stun_timeout_s)
                if mapped:
                    net.public_ip = mapped.ip
                    net.is_public = mapped.is_public
                    try:
                        nat_type = await asyncio.wait_for(stun.detect_nat_type(0), timeout=self._stun_timeout_s)
                        net.nat_type = nat_type.name.lower() if nat_type else "unknown"
                    except Exception:
                        net.nat_type = "unknown"
            except Exception as e:
                logger.debug("[SELF] STUN disabled/failed: %s", e)

        return net

    @staticmethod
    def _get_local_ipv4() -> str:
        try:
            import psutil  # type: ignore

            for addrs in psutil.net_if_addrs().values():
                for addr in addrs:
                    if getattr(addr, "family", None) == socket.AF_INET:
                        ip = getattr(addr, "address", "") or ""
                        if ip and not ip.startswith("127."):
                            return ip
        except Exception:
            pass

        try:
            ip = socket.gethostbyname(socket.gethostname())
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass

        return "127.0.0.1"

    def _compute_tags(self, profile: NodeProfile) -> List[str]:
        tags: List[str] = []
        hw = profile.hardware
        net = profile.network

        if hw.gpu_available:
            if hw.gpu_vram_gb >= 16:
                tags.append("HIGH_GPU")
            elif hw.gpu_vram_gb >= 8:
                tags.append("MEDIUM_GPU")
            else:
                tags.append("LOW_GPU")

        if hw.ram_total_gb >= 32:
            tags.append("HIGH_RAM")

        if hw.disk_free_gb >= 500:
            tags.append("STORAGE_HEAVY")

        if net.upload_mbps >= 100:
            tags.append("BANDWIDTH_HEAVY")
        if net.is_public:
            tags.append("PUBLIC_IP")
        if 0 < net.latency_ms < 50:
            tags.append("LOW_LATENCY")

        return tags

    @staticmethod
    def _compute_niches(profile: NodeProfile) -> List[NodeNiche]:
        niches: List[NodeNiche] = []
        tags = set(profile.tags)

        if {"HIGH_GPU", "MEDIUM_GPU"} & tags:
            niches.append(NodeNiche.NEURAL_MINER)

        if "BANDWIDTH_HEAVY" in tags or "PUBLIC_IP" in tags:
            niches.append(NodeNiche.TRAFFIC_WEAVER)

        if "STORAGE_HEAVY" in tags:
            niches.append(NodeNiche.STORAGE_KEEPER)

        niches.append(NodeNiche.ARBITRAGEUR)
        niches.append(NodeNiche.CHAIN_WEAVER)

        return niches

    @staticmethod
    def _select_dominant_niche(profile: NodeProfile) -> NodeNiche:
        priority = [
            NodeNiche.NEURAL_MINER,
            NodeNiche.TRAFFIC_WEAVER,
            NodeNiche.STORAGE_KEEPER,
            NodeNiche.ARBITRAGEUR,
        ]
        for niche in priority:
            if niche in profile.niches:
                return niche
        return NodeNiche.ARBITRAGEUR

    @property
    def profile(self) -> Optional[NodeProfile]:
        return self._profile
