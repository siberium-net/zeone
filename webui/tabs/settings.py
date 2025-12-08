"""
Settings Tab
============

[WEBUI] Network selection and configuration.
"""

import os
import sys
import logging
from typing import Dict

from nicegui import ui

from config import NETWORKS, ZEONE_NETWORK, get_current_network
from core.utils.env import update_env_variable

logger = logging.getLogger(__name__)


class SettingsTab:
    """UI for runtime settings such as blockchain network selection."""

    def __init__(self):
        self._current_network = get_current_network()

    def create_page(self, parent=None):
        @ui.page("/settings")
        async def settings():
            if parent:
                await parent._create_header()
                await parent._create_sidebar()

            with ui.column().classes("w-full p-4 gap-4"):
                ui.label("Settings").classes("text-2xl font-bold")

                # Blockchain Network Section
                ui.label("Blockchain Network").classes("text-xl font-semibold mt-2")

                network_options = [
                    {"label": v["name"], "value": key} for key, v in NETWORKS.items()
                ]

                selected = ui.select(
                    options=network_options,
                    label="Select Network",
                    value=self._current_network["key"],
                    with_input=False,
                ).classes("w-96")

                rpc_input = ui.input(
                    "Custom RPC (optional)",
                    value=os.getenv("SIBERIUM_RPC_URL", self._current_network["rpc_url"]),
                ).classes("w-96")

                def save_and_restart():
                    try:
                        update_env_variable("ZEONE_NETWORK", selected.value)
                        if rpc_input.value:
                            update_env_variable("SIBERIUM_RPC_URL", rpc_input.value)
                        ui.notify("Rebooting with new network...", type="info")
                        # Restart process
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                    except Exception as e:
                        logger.exception("Failed to apply settings")
                        ui.notify(f"Error: {e}", type="negative")

                ui.button(
                    "Save & Restart",
                    icon="restart_alt",
                    color="primary",
                    on_click=save_and_restart,
                )

                with ui.card().classes("w-full max-w-xl"):
                    ui.label("Active Network").classes("text-lg font-semibold")
                    ui.label(
                        f"{self._current_network['name']} (Chain ID: {self._current_network['chain_id']})"
                    ).classes("text-sm text-gray-400")
                    ui.label(f"RPC: {self._current_network['rpc_url']}").classes(
                        "text-sm text-gray-400"
                    )
                    ui.label(f"Explorer: {self._current_network['explorer_url']}").classes(
                        "text-sm text-gray-400"
                    )


__all__ = ["SettingsTab"]

