"""
Learning Tab
============

UI for PrivacyCollector (federated learning signals):
- enable/disable toggle (requires explicit consent)
- aggregated stats board
- privacy indicator (session hash, anonymization notes)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

from nicegui import ui

from cortex.learning.runtime import get_collector


class LearningTab:
    def __init__(self, parent=None):
        self.parent = parent
        self._enabled_switch = None
        self._session_label = None
        self._stats_cards: Dict[str, Any] = {}
        self._top_accept_table = None
        self._top_reject_table = None

    def create_page(self, parent) -> None:
        self.parent = parent

        @ui.page("/learning")
        async def learning():
            await parent._create_header()
            await parent._create_sidebar()

            collector = get_collector()

            with ui.column().classes("w-full p-4 gap-4"):
                ui.label("Learning").classes("text-2xl font-bold")
                ui.label("Privacy-preserving local signals (disabled by default).").classes("text-sm text-gray-400")

                with ui.card().classes("w-full"):
                    ui.label("Controls").classes("text-lg font-semibold")
                    self._enabled_switch = ui.switch(
                        "Enable Learning (requires consent)",
                        value=collector.enabled,
                        on_change=lambda e: self._toggle_learning(bool(e.value)),
                    )
                    self._session_label = ui.label(f"Session hash: {collector.session_hash}").classes("text-sm font-mono text-gray-400")
                    ui.label(
                        "Privacy: raw data never leaves the node; long strings are hashed; only aggregates are displayed."
                    ).classes("text-sm text-gray-400")

                with ui.row().classes("w-full gap-4 flex-wrap"):
                    self._stats_cards["total"] = self._stat_card("Total signals", "-")
                    self._stats_cards["rate"] = self._stat_card("Acceptance rate", "-")
                    self._stats_cards["corr"] = self._stat_card("Avg correction time (ms)", "-")
                    self._stats_cards["edit"] = self._stat_card("Avg edit distance", "-")

                with ui.row().classes("w-full gap-4 flex-wrap"):
                    with ui.card().classes("flex-1 min-w-[420px]"):
                        ui.label("Top Accepted Actions").classes("text-lg font-semibold mb-2")
                        self._top_accept_table = ui.table(
                            columns=[
                                {"name": "type", "label": "Type", "field": "type"},
                                {"name": "count", "label": "Count", "field": "count"},
                            ],
                            rows=[],
                            row_key="type",
                            pagination=10,
                        ).classes("w-full")

                    with ui.card().classes("flex-1 min-w-[420px]"):
                        ui.label("Top Rejections").classes("text-lg font-semibold mb-2")
                        self._top_reject_table = ui.table(
                            columns=[
                                {"name": "type", "label": "Type", "field": "type"},
                                {"name": "count", "label": "Count", "field": "count"},
                            ],
                            rows=[],
                            row_key="type",
                            pagination=10,
                        ).classes("w-full")

            ui.timer(1.0, self._tick)

    def _stat_card(self, title: str, value: str):
        with ui.card().classes("w-64"):
            ui.label(title).classes("text-sm text-gray-400")
            lbl = ui.label(value).classes("text-3xl font-bold")
        return lbl

    def _toggle_learning(self, enabled: bool) -> None:
        collector = get_collector()
        if enabled:
            collector.enable()
            if self.parent and hasattr(self.parent, "notify"):
                self.parent.notify("Learning enabled (with consent)", type="positive")
            else:
                ui.notify("Learning enabled (with consent)", type="positive")
        else:
            collector.disable()
            if self.parent and hasattr(self.parent, "notify"):
                self.parent.notify("Learning disabled", type="info")
            else:
                ui.notify("Learning disabled", type="info")
        if self._enabled_switch:
            self._enabled_switch.value = collector.enabled

    def _tick(self) -> None:
        collector = get_collector()
        stats = collector.get_aggregated_stats()
        if not stats:
            self._set_stats("-", "-", "-", "-")
            self._set_tables([], [])
            return

        total = str(stats.get("total_signals", 0))
        rate = stats.get("acceptance_rate", 0.0)
        corr = stats.get("avg_correction_time_ms", 0.0)
        edit = stats.get("avg_edit_distance", 0.0)
        self._set_stats(total, f"{float(rate)*100:.1f}%", f"{float(corr):.0f}", f"{float(edit):.2f}")

        acc = stats.get("acceptance_by_type") or {}
        rej = stats.get("rejection_by_type") or {}
        accept_rows = self._top_rows(acc)
        reject_rows = self._top_rows(rej)
        self._set_tables(accept_rows, reject_rows)

        if self._session_label:
            self._session_label.text = f"Session hash: {collector.session_hash}"

    def _set_stats(self, total: str, rate: str, corr: str, edit: str) -> None:
        if self._stats_cards.get("total"):
            self._stats_cards["total"].text = total
        if self._stats_cards.get("rate"):
            self._stats_cards["rate"].text = rate
        if self._stats_cards.get("corr"):
            self._stats_cards["corr"].text = corr
        if self._stats_cards.get("edit"):
            self._stats_cards["edit"].text = edit

    def _top_rows(self, d: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Tuple[str, int]] = []
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(k, str):
                    try:
                        items.append((k, int(v)))
                    except Exception:
                        pass
        items.sort(key=lambda kv: kv[1], reverse=True)
        return [{"type": k, "count": v} for k, v in items[:20]]

    def _set_tables(self, accept_rows: List[Dict[str, Any]], reject_rows: List[Dict[str, Any]]) -> None:
        if self._top_accept_table:
            self._top_accept_table.rows = accept_rows
            self._top_accept_table.update()
        if self._top_reject_table:
            self._top_reject_table.rows = reject_rows
            self._top_reject_table.update()


__all__ = ["LearningTab"]
