"""
Human Link status widget.

Shows Telegram bridge connectivity in the header/footer.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from nicegui import ui

from agents.bridge.human_link import get_human_link


def _mask_chat_id(chat_id: Optional[int]) -> str:
    if chat_id is None:
        return "-"
    s = str(chat_id)
    if len(s) <= 3:
        return s
    return f"***{s[-3:]}"


class HumanStatusWidget:
    def __init__(self) -> None:
        self._badge = None
        self._label = None
        self._pending = None
        self._dialog = None

    def mount(self) -> None:
        with ui.row().classes("items-center gap-2"):
            self._badge = ui.badge("Human: Disconnected", color="red")
            self._label = ui.label("Telegram: -").classes("text-sm text-gray-300")
            self._pending = ui.badge("0", color="grey")
            ui.button(icon="link", on_click=self._open_dialog).props("flat dense")

        ui.timer(1.0, self._tick)

    def _tick(self) -> None:
        link = get_human_link()
        st: Dict[str, Any] = {}
        try:
            st = link.status_snapshot()
        except Exception:
            st = {"online": link.is_human_online(), "chat_id": None, "auth_code": None, "has_token": False, "pending": 0}

        online = bool(st.get("online"))
        chat_id = st.get("chat_id")
        pending = int(st.get("pending") or 0)
        if self._badge:
            self._badge.text = "Human: Connected" if online else "Human: Disconnected"
            self._badge.props(f"color={'green' if online else 'red'}")
        if self._label:
            self._label.text = f"Telegram ID: {_mask_chat_id(chat_id)}"
        if self._pending:
            self._pending.text = str(pending)

    def _open_dialog(self) -> None:
        link = get_human_link()
        st = link.status_snapshot()

        if self._dialog is None:
            self._dialog = ui.dialog()

        self._dialog.clear()
        with self._dialog, ui.card().classes("w-[520px]"):
            ui.label("Human Link (Telegram)").classes("text-xl font-bold")
            ui.label("Bridge for human-in-the-loop approvals.").classes("text-sm text-gray-400")

            ui.separator()

            ui.label(f"Status: {'CONNECTED' if st.get('online') else 'DISCONNECTED'}").classes("font-mono")
            ui.label(f"Chat: {_mask_chat_id(st.get('chat_id'))}").classes("font-mono")
            ui.label(f"Pending: {st.get('pending', 0)}").classes("font-mono")
            ui.label(f"Token present: {bool(st.get('has_token'))}").classes("font-mono")

            code = st.get("auth_code") or ""
            if code:
                ui.label("Link command:").classes("text-sm text-gray-400 mt-2")
                ui.code(f"/auth {code}", language="text").classes("w-full")
            else:
                ui.label("Auth code appears in node logs after bot start.").classes("text-sm text-gray-400 mt-2")

            with ui.row().classes("gap-2 mt-2"):
                ui.button("Start bot", icon="play_arrow", on_click=lambda: asyncio.create_task(self._start_bot())).props(
                    "color=primary"
                )
                ui.button("Close", icon="close", on_click=self._dialog.close).props("flat")

        self._dialog.open()

    async def _start_bot(self) -> None:
        link = get_human_link()
        try:
            await link.start()
            ui.notify("Human Link started (check logs for /auth code)", type="positive")
        except Exception as e:
            ui.notify(f"Failed to start Human Link: {e}", type="negative")


__all__ = ["HumanStatusWidget"]

