"""
Human Link Agent
================
Telegram bot for human-in-the-loop confirmations.

[NOTE]
This module is optional: it runs in offline mode if `aiogram` and/or
`TELEGRAM_BOT_TOKEN` are unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class HumanRequest:
    """A single request awaiting human response."""

    request_id: str
    message: str
    options: list[str]
    status: RequestStatus = RequestStatus.PENDING
    response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HumanLinkAgent:
    """
    Telegram bot for human approvals.

    Requires `TELEGRAM_BOT_TOKEN` env var or explicit `bot_token`.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        timeout: float = 300.0,
    ):
        self.bot_token = bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.timeout = float(timeout)

        self._chat_id: Optional[int] = None
        self._auth_code: Optional[str] = None
        self._pending: Dict[str, HumanRequest] = {}
        self._bot = None
        self._polling_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start bot polling (no-op in offline mode)."""
        if not self.bot_token:
            logger.warning("[HUMAN_LINK] No TELEGRAM_BOT_TOKEN, running in offline mode")
            return

        try:
            from aiogram import Bot, Dispatcher  # type: ignore
            from aiogram import F  # type: ignore
            from aiogram.filters import Command  # type: ignore
            from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message  # type: ignore

            self._bot = Bot(token=self.bot_token)
            dp = Dispatcher()

            self._auth_code = secrets.token_hex(4).upper()
            logger.info("[HUMAN_LINK] Auth code: %s", self._auth_code)

            @dp.message(Command("start"))
            async def cmd_start(message: Message) -> None:
                await message.answer(
                    "ZEONE Human Link\n\n"
                    "To link this chat, send: /auth <code>\n"
                    "Get the code from node logs."
                )

            @dp.message(Command("auth"))
            async def cmd_auth(message: Message) -> None:
                text = message.text or ""
                parts = text.split()
                if len(parts) < 2:
                    await message.answer("Usage: /auth <code>")
                    return

                code = parts[1].upper().strip()
                if self._auth_code and code == self._auth_code:
                    self._chat_id = message.chat.id
                    await message.answer("[OK] Linked. You will now receive requests.")
                    logger.info("[HUMAN_LINK] Authorized chat_id=%s", self._chat_id)
                else:
                    await message.answer("[ERROR] Invalid code")

            @dp.message(Command("status"))
            async def cmd_status(message: Message) -> None:
                await message.answer(f"Pending requests: {len(self._pending)}")

            @dp.callback_query(F.data)
            async def handle_callback(callback: CallbackQuery) -> None:
                data = callback.data or ""
                if ":" not in data:
                    await callback.answer("Malformed callback")
                    return

                req_id, option_idx_s = data.split(":", 1)
                req = self._pending.get(req_id)
                if not req:
                    await callback.answer("Request expired")
                    return

                try:
                    idx = int(option_idx_s)
                    req.response = req.options[idx]
                    req.status = RequestStatus.APPROVED if idx == 0 else RequestStatus.REJECTED
                except Exception:
                    req.response = None
                    req.status = RequestStatus.REJECTED

                await callback.answer(f"Selected: {req.response}")
                try:
                    await callback.message.edit_text(f"[{req.status.value.upper()}] {req.message}")
                except Exception:
                    pass

            self._polling_task = asyncio.create_task(dp.start_polling(self._bot))
            logger.info("[HUMAN_LINK] Bot started")
        except ImportError:
            logger.error("[HUMAN_LINK] aiogram not installed: pip install aiogram>=3.0")
        except Exception as e:
            logger.error("[HUMAN_LINK] Failed to start: %s", e)

    async def stop(self) -> None:
        """Stop polling and close bot session."""
        if self._polling_task:
            self._polling_task.cancel()
            self._polling_task = None
        if self._bot is not None:
            try:
                await self._bot.session.close()
            except Exception:
                pass

    async def ask_human(
        self,
        message: str,
        options: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Send a request and wait for response.

        Returns selected option, or None on timeout/offline.
        """
        options = options or ["Approve", "Reject"]

        if not self._chat_id or not self._bot:
            logger.warning("[HUMAN_LINK] No authorized chat/bot, returning None")
            return None

        req_id = secrets.token_hex(8)
        req = HumanRequest(
            request_id=req_id,
            message=message,
            options=list(options),
            metadata=metadata or {},
        )
        self._pending[req_id] = req

        try:
            from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup  # type: ignore

            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text=opt, callback_data=f"{req_id}:{i}")]
                    for i, opt in enumerate(options)
                ]
            )

            await self._bot.send_message(
                chat_id=self._chat_id,
                text=f"[REQUEST]\n\n{message}",
                reply_markup=keyboard,
            )

            loop = asyncio.get_running_loop()
            start = loop.time()
            while req.status == RequestStatus.PENDING:
                if loop.time() - start > self.timeout:
                    req.status = RequestStatus.TIMEOUT
                    break
                await asyncio.sleep(0.5)

            return req.response
        except Exception as e:
            logger.error("[HUMAN_LINK] ask_human failed: %s", e)
            return None
        finally:
            self._pending.pop(req_id, None)

    def is_human_online(self) -> bool:
        """Return True if a chat is linked."""
        return self._chat_id is not None


_human_link: Optional[HumanLinkAgent] = None


def get_human_link() -> HumanLinkAgent:
    """Return a singleton HumanLinkAgent instance."""
    global _human_link
    if _human_link is None:
        _human_link = HumanLinkAgent()
    return _human_link

