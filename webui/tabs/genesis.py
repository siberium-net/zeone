"""
Genesis Tab
===========

Visualizes node self-awareness and Genesis Protocol:
- Live hardware monitor (CPU/RAM/Disk/GPU)
- Niche diagnosis + re-evaluate
- Evolution dashboard (fitness chart + [EVO] log stream)
- Alpha agent code viewer
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nicegui import ui

from cortex.genesis import run_genesis
from cortex.self import SelfAwareness

logger = logging.getLogger(__name__)


def _read_text(path: Path, limit: int = 200_000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
        if len(data) > limit:
            return data[-limit:]
        return data
    except Exception:
        return ""


def _nvidia_smi_stats() -> Dict[str, Any]:
    """
    Return a small GPU stats snapshot using `nvidia-smi` (no Python deps).
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if proc.returncode != 0:
            return {}
        line = (proc.stdout or "").strip().splitlines()[0].strip()
        util, mem_used, mem_total, temp, name = [p.strip() for p in line.split(",", 4)]
        return {
            "name": name,
            "util_pct": float(util),
            "mem_used_mb": float(mem_used),
            "mem_total_mb": float(mem_total),
            "temp_c": float(temp),
        }
    except Exception:
        return {}


class GenesisTab:
    def __init__(self, parent=None):
        self.parent = parent
        self._hw_labels: Dict[str, Any] = {}
        self._disk_io_prev: Optional[Tuple[float, int, int]] = None  # (ts, read_bytes, write_bytes)
        self._niche_label = None
        self._tags_label = None
        self._alpha_code = None
        self._evo_log = None
        self._chart = None
        self._fitness: List[Tuple[int, float, float]] = []  # (gen, max, avg)

    def create_page(self, parent) -> None:
        self.parent = parent

        @ui.page("/genesis")
        async def genesis():
            await parent._create_header()
            await parent._create_sidebar()

            with ui.row().classes("w-full p-4 gap-6"):
                with ui.column().classes("w-1/3 gap-4"):
                    ui.label("Genesis").classes("text-2xl font-bold")
                    ui.label("Self-awareness + evolutionary bootstrap").classes("text-sm text-gray-400")

                    with ui.card().classes("w-full"):
                        ui.label("Hardware Monitor").classes("text-lg font-semibold")
                        self._hw_labels["cpu"] = ui.label("CPU: -").classes("text-sm font-mono")
                        self._hw_labels["ram"] = ui.label("RAM: -").classes("text-sm font-mono")
                        self._hw_labels["disk"] = ui.label("Disk: -").classes("text-sm font-mono")
                        self._hw_labels["disk_io"] = ui.label("Disk IO: -").classes("text-sm font-mono")
                        self._hw_labels["gpu"] = ui.label("GPU: -").classes("text-sm font-mono")

                    with ui.card().classes("w-full"):
                        ui.label("Niche Diagnosis").classes("text-lg font-semibold")
                        self._niche_label = ui.label("Niche: -").classes("text-xl font-bold")
                        self._tags_label = ui.label("Tags: -").classes("text-sm text-gray-400")
                        ui.button("Re-evaluate Niche", icon="refresh", on_click=self._on_recheck_niche).props("color=primary")

                    with ui.card().classes("w-full"):
                        ui.label("Run Genesis").classes("text-lg font-semibold")
                        epochs = ui.number("Epochs", value=10, min=1, max=200).classes("w-full")
                        pop = ui.number("Population", value=20, min=2, max=200).classes("w-full")
                        niche = ui.input("Niche override (optional)", placeholder="neural_miner").classes("w-full")
                        ui.button(
                            "Run",
                            icon="play_arrow",
                            on_click=lambda: self._on_run_genesis(
                                int(epochs.value or 10),
                                int(pop.value or 20),
                                str(niche.value or "").strip() or None,
                            ),
                        ).props("color=positive").classes("w-full")

                with ui.column().classes("w-2/3 gap-4"):
                    ui.label("Evolution Dashboard").classes("text-2xl font-bold")
                    self._chart = ui.echart(self._chart_options()).classes("w-full h-64")
                    self._evo_log = ui.log(max_lines=500).classes("w-full h-48")

                    with ui.card().classes("w-full"):
                        ui.label("Alpha Agent Code").classes("text-lg font-semibold mb-2")
                        self._alpha_code = ui.code("", language="python").classes("w-full h-96 overflow-auto")

            ui.timer(1.0, self._tick)
            asyncio.create_task(self._recheck_niche())

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        # keep timer callback sync; run async work in task
        asyncio.create_task(self._refresh_hw())
        self._refresh_evo_from_logs()
        self._refresh_alpha()

    async def _refresh_hw(self) -> None:
        # CPU/RAM/Disk via psutil if available
        try:
            import psutil  # type: ignore

            cpu = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            du = psutil.disk_usage("/")
            io = psutil.disk_io_counters()
            cpu_s = f"CPU: {cpu:.0f}%"
            ram_s = f"RAM: {vm.percent:.0f}% ({vm.available/1024**3:.1f} GB free)"
            disk_s = f"Disk: {du.percent:.0f}% ({du.free/1024**3:.1f} GB free)"
            disk_io_s = "Disk IO: -"
            if io is not None:
                now = asyncio.get_running_loop().time()
                if self._disk_io_prev is None:
                    self._disk_io_prev = (now, int(io.read_bytes), int(io.write_bytes))
                else:
                    prev_ts, prev_r, prev_w = self._disk_io_prev
                    dt = max(0.001, now - prev_ts)
                    r_mb_s = (int(io.read_bytes) - prev_r) / dt / (1024**2)
                    w_mb_s = (int(io.write_bytes) - prev_w) / dt / (1024**2)
                    disk_io_s = f"Disk IO: {r_mb_s:.1f} MB/s read | {w_mb_s:.1f} MB/s write"
                    self._disk_io_prev = (now, int(io.read_bytes), int(io.write_bytes))
        except Exception:
            cpu_s, ram_s, disk_s, disk_io_s = "CPU: -", "RAM: -", "Disk: -", "Disk IO: -"

        gpu = _nvidia_smi_stats()
        if gpu:
            used_gb = float(gpu["mem_used_mb"]) / 1024.0
            total_gb = float(gpu["mem_total_mb"]) / 1024.0
            gpu_s = f"GPU: {gpu['name']} | {gpu['util_pct']:.0f}% | VRAM {used_gb:.1f}/{total_gb:.1f} GB | {gpu['temp_c']:.0f}°C"
        else:
            gpu_s = "GPU: -"

        for k, v in (("cpu", cpu_s), ("ram", ram_s), ("disk", disk_s), ("disk_io", disk_io_s), ("gpu", gpu_s)):
            el = self._hw_labels.get(k)
            if el is not None:
                el.text = v

    def _refresh_alpha(self) -> None:
        if not self._alpha_code:
            return
        code = _read_text(Path("data/alpha_agent.py"))
        if code and getattr(self._alpha_code, "value", None) != code:
            self._alpha_code.value = code

    def _refresh_evo_from_logs(self) -> None:
        if not self.parent:
            return

        evo_lines: List[str] = []
        try:
            buf = list(getattr(self.parent, "_log_buffer", []))
        except Exception:
            buf = []

        for line in buf:
            if "[EVO]" in line:
                evo_lines.append(str(line))
        evo_lines = evo_lines[-200:]

        # Parse fitness points from logs.
        points: List[Tuple[int, float, float]] = []
        rx = re.compile(r"\\[EVO\\]\\s+Gen\\s+(\\d+)\\s*:\\s*Max Fitness\\s+([0-9.]+),\\s*Avg\\s+([0-9.]+)")
        for ln in evo_lines:
            m = rx.search(ln)
            if not m:
                continue
            try:
                points.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
            except Exception:
                continue
        if points:
            # de-dup by generation (keep last)
            by_gen: Dict[int, Tuple[int, float, float]] = {g: (g, mx, av) for g, mx, av in points}
            self._fitness = [by_gen[g] for g in sorted(by_gen)]
            self._update_chart()

        if self._evo_log:
            self._evo_log.clear()
            for ln in evo_lines:
                self._evo_log.push(ln)

    def _update_chart(self) -> None:
        if not self._chart:
            return
        gens = [g for g, _mx, _av in self._fitness]
        maxes = [mx for _g, mx, _av in self._fitness]
        avgs = [av for _g, _mx, av in self._fitness]

        opt = self._chart_options()
        opt["xAxis"]["data"] = gens
        opt["series"][0]["data"] = maxes
        opt["series"][1]["data"] = avgs
        self._chart.options = opt
        self._chart.update()

    def _chart_options(self) -> Dict[str, Any]:
        return {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Max Fitness", "Avg Fitness"], "textStyle": {"color": "#a3a3a3"}},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {"type": "category", "data": [], "axisLabel": {"color": "#a3a3a3"}},
            "yAxis": {"type": "value", "axisLabel": {"color": "#a3a3a3"}},
            "series": [
                {"name": "Max Fitness", "type": "line", "smooth": True, "data": []},
                {"name": "Avg Fitness", "type": "line", "smooth": True, "data": []},
            ],
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def _recheck_niche(self) -> None:
        try:
            profile = await SelfAwareness(enable_stun=False).diagnose()
            if self._niche_label:
                self._niche_label.text = f"Niche: {profile.dominant_niche.name if profile.dominant_niche else '-'}"
            if self._tags_label:
                self._tags_label.text = f"Tags: {', '.join(profile.tags) if profile.tags else '-'}"
        except Exception as e:
            logger.warning("[WEBUI] Niche re-eval failed: %s", e)

    def _on_recheck_niche(self) -> None:
        asyncio.create_task(self._recheck_niche())

    def _on_run_genesis(self, epochs: int, population: int, niche: Optional[str]) -> None:
        asyncio.create_task(self._run_genesis_bg(epochs, population, niche))

    async def _run_genesis_bg(self, epochs: int, population: int, niche: Optional[str]) -> None:
        try:
            if self.parent and hasattr(self.parent, "notify"):
                self.parent.notify("Genesis started...", type="info")
            else:
                ui.notify("Genesis started...", type="info")
            await run_genesis(epochs=epochs, population=population, data_dir="data", niche=niche)
            if self.parent and hasattr(self.parent, "notify"):
                self.parent.notify("Genesis complete", type="positive")
            else:
                ui.notify("Genesis complete", type="positive")
        except Exception as e:
            if self.parent and hasattr(self.parent, "notify"):
                self.parent.notify(f"Genesis failed: {e}", type="negative")
            else:
                ui.notify(f"Genesis failed: {e}", type="negative")


__all__ = ["GenesisTab"]
