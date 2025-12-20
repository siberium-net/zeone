"""
Settings Tab
============

Goals:
- CLI parity: show every `main.py` argparse option with current runtime value.
- Config parity: show key `config.py` values (runtime snapshot).
- State sync: show sources (CLI/ENV/default) and highlight restart-required settings.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nicegui import app, ui

from config import NETWORKS, config, get_current_network
from core.utils.env import update_env_variable
from economy.ledger import DEFAULT_DEBT_LIMIT_BYTES


def _try_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _try_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _mask(s: Any, keep: int = 3) -> str:
    txt = str(s) if s is not None else ""
    if len(txt) <= keep:
        return txt
    return f"{txt[:keep]}***{txt[-keep:]}"


def _cli_parser() -> argparse.ArgumentParser:
    # Mirror `main.py` (keep in sync with argparse definitions there)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--port", "-p", type=int, default=config.network.default_port)
    p.add_argument("--host", "-H", type=str, default="0.0.0.0")
    p.add_argument("--bootstrap", "-b", type=str, default="boot.ze1.org:80")
    p.add_argument("--identity", "-i", type=str, default=config.crypto.identity_file)
    p.add_argument("--db", "-d", type=str, default=config.ledger.database_path)
    p.add_argument("--debt-limit", type=int, default=DEFAULT_DEBT_LIMIT_BYTES)
    p.add_argument("--masking", "-m", action="store_true")
    p.add_argument("--no-shell", action="store_true")
    p.add_argument("--webui", "-w", action="store_true")
    p.add_argument("--webui-port", type=int, default=8080)
    p.add_argument("--mcp", action="store_true")
    p.add_argument("--mcp-host", type=str, default="0.0.0.0")
    p.add_argument("--mcp-port", type=int, default=8090)
    p.add_argument("--auto-update", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--genesis", action="store_true")
    p.add_argument("--genesis-epochs", type=int, default=10)
    p.add_argument("--genesis-population", type=int, default=20)
    p.add_argument("--genesis-data-dir", type=str, default="data")
    p.add_argument("--genesis-niche", type=str, default="")
    p.add_argument("--metrics", action="store_true")
    p.add_argument("--exit-node", action="store_true")
    p.add_argument("--health-port", type=int, default=0)
    p.add_argument("--no-persistence", action="store_true")
    p.add_argument("--no-security", action="store_true")
    return p


def _parse_runtime_args(argv: List[str]) -> argparse.Namespace:
    parser = _cli_parser()
    ns, _unknown = parser.parse_known_args(argv[1:])
    return ns


def _flag_present(argv: List[str], flags: Tuple[str, ...]) -> bool:
    s = set(argv[1:])
    return any(f in s for f in flags)


class SettingsTab:
    def __init__(self) -> None:
        self._argv = list(sys.argv)
        self._args = _parse_runtime_args(self._argv)
        self._current_network = get_current_network()

    def create_page(self, parent=None) -> None:
        @ui.page("/settings")
        async def settings():
            if parent:
                await parent._create_header()
                await parent._create_sidebar()

            with ui.column().classes("w-full p-4 gap-4"):
                ui.label("Settings").classes("text-2xl font-bold")
                ui.label("CLI parity + runtime config snapshot").classes("text-sm text-gray-400")

                with ui.card().classes("w-full"):
                    ui.label("Runtime").classes("text-lg font-semibold")
                    ui.label(f"argv: {' '.join(self._argv)}").classes("text-xs font-mono text-gray-400 break-all")

                self._render_cli_sections()
                self._render_config_snapshot()
                self._render_restart_builder()

                with ui.card().classes("w-full"):
                    ui.label("Restart").classes("text-lg font-semibold")
                    ui.label("Most settings require a restart to take effect.").classes("text-sm text-gray-400")
                    ui.button("Restart process", icon="restart_alt", on_click=self._restart_same_args).props("color=primary")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _render_cli_sections(self) -> None:
        # Network (ENV-driven preset + CLI network basics)
        with ui.expansion("Network", icon="lan").classes("w-full"):
            with ui.card().classes("w-full"):
                ui.label("Blockchain Preset").classes("font-semibold")

                network_options = {k: v["name"] for k, v in NETWORKS.items()}
                current_key = self._current_network["key"]
                sel = ui.select(
                    options=network_options,
                    value=current_key if current_key in network_options else next(iter(network_options.keys())),
                    label="ZEONE_NETWORK (ENV)",
                    with_input=False,
                ).props("outlined").classes("w-96")

                rpc_in = ui.input(
                    "SIBERIUM_RPC_URL (ENV override, optional)",
                    value=os.getenv("SIBERIUM_RPC_URL", str(self._current_network["rpc_url"])),
                ).classes("w-96")

                with ui.row().classes("gap-2 items-center"):
                    ui.badge("Requires Restart", color="orange")
                    ui.button(
                        "Apply & Restart",
                        icon="restart_alt",
                        on_click=lambda: self._apply_network_env_and_restart(sel.value, rpc_in.value),
                    ).props("color=primary")

            with ui.card().classes("w-full"):
                ui.label("P2P Bind / Bootstrap (CLI)").classes("font-semibold")
                self._cli_row(
                    "Identity file",
                    value=str(self._args.identity),
                    source="CLI" if _flag_present(self._argv, ("--identity", "-i")) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Ledger DB",
                    value=str(self._args.db),
                    source="CLI" if _flag_present(self._argv, ("--db", "-d")) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Host",
                    value=str(self._args.host),
                    source="CLI" if _flag_present(self._argv, ("--host", "-H")) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Port",
                    value=str(self._args.port),
                    source="CLI" if _flag_present(self._argv, ("--port", "-p")) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Bootstrap",
                    value=str(self._args.bootstrap),
                    source="CLI" if _flag_present(self._argv, ("--bootstrap", "-b")) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Masking",
                    value=str(bool(self._args.masking)),
                    source="CLI" if _flag_present(self._argv, ("--masking", "-m")) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Interactive shell disabled",
                    value=str(bool(self._args.no_shell)),
                    source="CLI" if _flag_present(self._argv, ("--no-shell",)) else "default",
                    requires_restart=True,
                )

        with ui.expansion("Hardware", icon="memory").classes("w-full"):
            with ui.card().classes("w-full"):
                ui.label("Runtime profile").classes("font-semibold")
                ui.label("Use the Genesis tab for live hardware monitoring and niche evaluation.").classes(
                    "text-sm text-gray-400"
                )

        with ui.expansion("AI", icon="psychology").classes("w-full"):
            with ui.card().classes("w-full"):
                ui.label("Runtime flags").classes("font-semibold")
                self._cli_row(
                    "Verbose logging",
                    value=str(bool(self._args.verbose)),
                    source="CLI" if _flag_present(self._argv, ("--verbose", "-v")) else "default",
                    requires_restart=False,
                )

        with ui.expansion("Evolution", icon="science").classes("w-full"):
            with ui.card().classes("w-full"):
                ui.label("Genesis options (CLI)").classes("font-semibold")
                self._cli_row(
                    "Genesis enabled",
                    value=str(bool(self._args.genesis)),
                    source="CLI" if _flag_present(self._argv, ("--genesis",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Genesis niche override",
                    value=str(self._args.genesis_niche or ""),
                    source="CLI" if _flag_present(self._argv, ("--genesis-niche",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Genesis epochs",
                    value=str(self._args.genesis_epochs),
                    source="CLI" if _flag_present(self._argv, ("--genesis-epochs",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Genesis population",
                    value=str(self._args.genesis_population),
                    source="CLI" if _flag_present(self._argv, ("--genesis-population",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Genesis data dir",
                    value=str(self._args.genesis_data_dir),
                    source="CLI" if _flag_present(self._argv, ("--genesis-data-dir",)) else "default",
                    requires_restart=True,
                )

        with ui.expansion("Security", icon="shield").classes("w-full"):
            with ui.card().classes("w-full"):
                ui.label("Security flags (CLI)").classes("font-semibold")
                self._cli_row(
                    "No security",
                    value=str(bool(self._args.no_security)),
                    source="CLI" if _flag_present(self._argv, ("--no-security",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Debt limit (bytes)",
                    value=str(self._args.debt_limit),
                    source="CLI" if _flag_present(self._argv, ("--debt-limit",)) else "default",
                    requires_restart=True,
                )

        with ui.expansion("Monitoring", icon="monitoring").classes("w-full"):
            with ui.card().classes("w-full"):
                ui.label("WebUI / Health / Metrics (CLI)").classes("font-semibold")
                self._cli_row(
                    "WebUI enabled",
                    value=str(bool(self._args.webui)),
                    source="CLI" if _flag_present(self._argv, ("--webui", "-w")) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "WebUI port",
                    value=str(self._args.webui_port),
                    source="CLI" if _flag_present(self._argv, ("--webui-port",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "MCP enabled",
                    value=str(bool(self._args.mcp)),
                    source="CLI" if _flag_present(self._argv, ("--mcp",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "MCP host",
                    value=str(self._args.mcp_host),
                    source="CLI" if _flag_present(self._argv, ("--mcp-host",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "MCP port",
                    value=str(self._args.mcp_port),
                    source="CLI" if _flag_present(self._argv, ("--mcp-port",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Metrics enabled",
                    value=str(bool(self._args.metrics)),
                    source="CLI" if _flag_present(self._argv, ("--metrics",)) else "default",
                    requires_restart=True,
                )
                self._cli_row(
                    "Health port",
                    value=str(self._args.health_port),
                    source="CLI" if _flag_present(self._argv, ("--health-port",)) else "default",
                    requires_restart=True,
                )

        with ui.expansion("Persistence", icon="save").classes("w-full"):
            with ui.card().classes("w-full"):
                ui.label("Persistence flags (CLI)").classes("font-semibold")
                self._cli_row(
                    "No persistence",
                    value=str(bool(self._args.no_persistence)),
                    source="CLI" if _flag_present(self._argv, ("--no-persistence",)) else "default",
                    requires_restart=True,
                )

        with ui.expansion("Updater", icon="system_update_alt").classes("w-full"):
            with ui.card().classes("w-full"):
                ui.label("Auto-update (CLI)").classes("font-semibold")
                self._cli_row(
                    "Auto-update",
                    value=str(bool(self._args.auto_update)),
                    source="CLI" if _flag_present(self._argv, ("--auto-update",)) else "default",
                    requires_restart=True,
                )

        with ui.expansion("VPN / Exit", icon="vpn_lock").classes("w-full"):
            with ui.card().classes("w-full"):
                ui.label("VPN mode (CLI)").classes("font-semibold")
                self._cli_row(
                    "Exit node mode",
                    value=str(bool(self._args.exit_node)),
                    source="CLI" if _flag_present(self._argv, ("--exit-node",)) else "default",
                    requires_restart=True,
                )

    def _render_config_snapshot(self) -> None:
        with ui.expansion("Config Snapshot (runtime)", icon="tune").classes("w-full"):
            ui.label("Read-only view of `config.py` dataclass values.").classes("text-sm text-gray-400")

            cfg = config
            if is_dataclass(cfg):
                cfg_dict = asdict(cfg)
            else:
                cfg_dict = {"config": str(cfg)}

            for section, data in cfg_dict.items():
                with ui.card().classes("w-full"):
                    ui.label(section).classes("text-lg font-semibold")
                    if isinstance(data, dict):
                        rows = [{"key": k, "value": str(v)} for k, v in data.items()]
                        ui.table(
                            columns=[
                                {"name": "key", "label": "Key", "field": "key", "align": "left"},
                                {"name": "value", "label": "Value", "field": "value", "align": "left"},
                            ],
                            rows=rows,
                            row_key="key",
                            pagination=10,
                        ).classes("w-full")
                    else:
                        ui.label(str(data)).classes("text-sm font-mono text-gray-400")

    def _render_restart_builder(self) -> None:
        """
        Editable form to build a restart command.

        NiceGUI pages cannot mutate argparse/runtime state safely; we apply changes by restarting.
        """
        with ui.expansion("Restart Command Builder", icon="terminal").classes("w-full"):
            ui.label("Edit settings and restart with generated CLI flags.").classes("text-sm text-gray-400")

            store = app.storage.user
            overrides: Dict[str, Any] = dict(store.get("cli_overrides") or {})

            def set_override(key: str, value: Any) -> None:
                overrides[key] = value
                store["cli_overrides"] = overrides

            with ui.card().classes("w-full"):
                ui.label("Network").classes("text-lg font-semibold")
                host = ui.input("Host", value=str(overrides.get("host", self._args.host))).classes("w-96")
                port = ui.number("Port", value=_try_int(overrides.get("port", self._args.port)) or int(self._args.port)).classes("w-96")
                bootstrap = ui.input("Bootstrap (host:port, comma-separated)", value=str(overrides.get("bootstrap", self._args.bootstrap))).classes("w-96")
                masking = ui.switch("Enable masking", value=bool(overrides.get("masking", bool(self._args.masking))))

                host.on("change", lambda e: set_override("host", e.value))
                port.on("change", lambda e: set_override("port", e.value))
                bootstrap.on("change", lambda e: set_override("bootstrap", e.value))
                masking.on("change", lambda e: set_override("masking", e.value))

            with ui.card().classes("w-full"):
                ui.label("Core").classes("text-lg font-semibold")
                identity = ui.input("Identity file", value=str(overrides.get("identity", self._args.identity))).classes("w-96")
                db = ui.input("Ledger DB", value=str(overrides.get("db", self._args.db))).classes("w-96")
                debt = ui.number(
                    "Debt limit (bytes)",
                    value=_try_int(overrides.get("debt_limit", self._args.debt_limit)) or int(self._args.debt_limit),
                ).classes("w-96")
                no_shell = ui.switch("Disable interactive shell", value=bool(overrides.get("no_shell", bool(self._args.no_shell))))

                identity.on("change", lambda e: set_override("identity", e.value))
                db.on("change", lambda e: set_override("db", e.value))
                debt.on("change", lambda e: set_override("debt_limit", e.value))
                no_shell.on("change", lambda e: set_override("no_shell", e.value))

            with ui.card().classes("w-full"):
                ui.label("Evolution").classes("text-lg font-semibold")
                genesis = ui.switch("Run genesis on start", value=bool(overrides.get("genesis", bool(self._args.genesis))))
                niche = ui.input("Genesis niche override", value=str(overrides.get("genesis_niche", self._args.genesis_niche or ""))).classes("w-96")
                epochs = ui.number("Genesis epochs", value=_try_int(overrides.get("genesis_epochs", self._args.genesis_epochs)) or int(self._args.genesis_epochs)).classes("w-96")
                pop = ui.number("Genesis population", value=_try_int(overrides.get("genesis_population", self._args.genesis_population)) or int(self._args.genesis_population)).classes("w-96")
                data_dir = ui.input("Genesis data dir", value=str(overrides.get("genesis_data_dir", self._args.genesis_data_dir))).classes("w-96")

                genesis.on("change", lambda e: set_override("genesis", e.value))
                niche.on("change", lambda e: set_override("genesis_niche", e.value))
                epochs.on("change", lambda e: set_override("genesis_epochs", e.value))
                pop.on("change", lambda e: set_override("genesis_population", e.value))
                data_dir.on("change", lambda e: set_override("genesis_data_dir", e.value))

            with ui.card().classes("w-full"):
                ui.label("Monitoring / Security").classes("text-lg font-semibold")
                webui_enabled = ui.switch("WebUI enabled", value=bool(overrides.get("webui", bool(self._args.webui))))
                webui_port = ui.number("WebUI port", value=_try_int(overrides.get("webui_port", self._args.webui_port)) or int(self._args.webui_port)).classes("w-96")
                mcp_enabled = ui.switch("MCP enabled", value=bool(overrides.get("mcp", bool(self._args.mcp))))
                mcp_host = ui.input("MCP host", value=str(overrides.get("mcp_host", self._args.mcp_host))).classes("w-96")
                mcp_port = ui.number("MCP port", value=_try_int(overrides.get("mcp_port", self._args.mcp_port)) or int(self._args.mcp_port)).classes("w-96")
                metrics = ui.switch("Metrics enabled", value=bool(overrides.get("metrics", bool(self._args.metrics))))
                health_port = ui.number("Health port", value=_try_int(overrides.get("health_port", self._args.health_port)) or int(self._args.health_port)).classes("w-96")
                exit_node = ui.switch("Exit node mode", value=bool(overrides.get("exit_node", bool(self._args.exit_node))))
                no_security = ui.switch("Disable security modules", value=bool(overrides.get("no_security", bool(self._args.no_security))))
                no_persistence = ui.switch("Disable persistence", value=bool(overrides.get("no_persistence", bool(self._args.no_persistence))))
                auto_update = ui.switch("Auto-update", value=bool(overrides.get("auto_update", bool(self._args.auto_update))))
                verbose = ui.switch("Verbose logs", value=bool(overrides.get("verbose", bool(self._args.verbose))))

                webui_enabled.on("change", lambda e: set_override("webui", e.value))
                webui_port.on("change", lambda e: set_override("webui_port", e.value))
                mcp_enabled.on("change", lambda e: set_override("mcp", e.value))
                mcp_host.on("change", lambda e: set_override("mcp_host", e.value))
                mcp_port.on("change", lambda e: set_override("mcp_port", e.value))
                metrics.on("change", lambda e: set_override("metrics", e.value))
                health_port.on("change", lambda e: set_override("health_port", e.value))
                exit_node.on("change", lambda e: set_override("exit_node", e.value))
                no_security.on("change", lambda e: set_override("no_security", e.value))
                no_persistence.on("change", lambda e: set_override("no_persistence", e.value))
                auto_update.on("change", lambda e: set_override("auto_update", e.value))
                verbose.on("change", lambda e: set_override("verbose", e.value))

            preview = ui.label("").classes("text-xs font-mono text-gray-400 break-all")

            def build_cmd() -> List[str]:
                cmd = [sys.executable, str(Path(sys.argv[0]).name)]

                def add_kv(flag: str, val: Any) -> None:
                    if val is None or val == "":
                        return
                    cmd.extend([flag, str(val)])

                def add_flag(flag: str, enabled: Any) -> None:
                    if bool(enabled):
                        cmd.append(flag)

                add_kv("--host", overrides.get("host", self._args.host))
                add_kv("--port", overrides.get("port", self._args.port))
                add_kv("--bootstrap", overrides.get("bootstrap", self._args.bootstrap))
                add_flag("--masking", overrides.get("masking", self._args.masking))
                add_kv("--identity", overrides.get("identity", self._args.identity))
                add_kv("--db", overrides.get("db", self._args.db))
                add_kv("--debt-limit", overrides.get("debt_limit", self._args.debt_limit))
                add_flag("--no-shell", overrides.get("no_shell", self._args.no_shell))
                add_flag("--webui", overrides.get("webui", self._args.webui))
                add_kv("--webui-port", overrides.get("webui_port", self._args.webui_port))
                add_flag("--mcp", overrides.get("mcp", self._args.mcp))
                add_kv("--mcp-host", overrides.get("mcp_host", self._args.mcp_host))
                add_kv("--mcp-port", overrides.get("mcp_port", self._args.mcp_port))
                add_flag("--auto-update", overrides.get("auto_update", self._args.auto_update))
                add_flag("--verbose", overrides.get("verbose", self._args.verbose))
                add_flag("--metrics", overrides.get("metrics", self._args.metrics))
                add_kv("--health-port", overrides.get("health_port", self._args.health_port))
                add_flag("--no-security", overrides.get("no_security", self._args.no_security))
                add_flag("--no-persistence", overrides.get("no_persistence", self._args.no_persistence))
                add_flag("--exit-node", overrides.get("exit_node", self._args.exit_node))
                add_flag("--genesis", overrides.get("genesis", self._args.genesis))
                add_kv("--genesis-niche", overrides.get("genesis_niche", self._args.genesis_niche))
                add_kv("--genesis-epochs", overrides.get("genesis_epochs", self._args.genesis_epochs))
                add_kv("--genesis-population", overrides.get("genesis_population", self._args.genesis_population))
                add_kv("--genesis-data-dir", overrides.get("genesis_data_dir", self._args.genesis_data_dir))
                return cmd

            def refresh_preview() -> None:
                preview.text = " ".join(build_cmd())

            refresh_preview()
            ui.timer(0.5, refresh_preview)

            def _copy_cli() -> None:
                ui.run_javascript(f"navigator.clipboard.writeText({preview.text!r})")

            with ui.row().classes("gap-2"):
                ui.button(
                    "Copy CLI",
                    icon="content_copy",
                    on_click=_copy_cli,
                )
                ui.button("Restart with overrides", icon="restart_alt", on_click=lambda: os.execv(sys.executable, build_cmd())).props(
                    "color=primary"
                )

    def _cli_row(self, label: str, *, value: str, source: str, requires_restart: bool) -> None:
        with ui.row().classes("items-center justify-between w-full gap-4"):
            with ui.row().classes("items-center gap-2"):
                ui.label(label).classes("font-medium")
                if requires_restart:
                    ui.badge("Requires Restart", color="orange")
            ui.label(value).classes("font-mono text-sm text-gray-200")
            ui.badge(source.upper(), color="blue" if source.lower() == "cli" else "grey")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _apply_network_env_and_restart(self, network_key: str, rpc_url: str) -> None:
        try:
            update_env_variable("ZEONE_NETWORK", str(network_key))
            if rpc_url and str(rpc_url).strip():
                update_env_variable("SIBERIUM_RPC_URL", str(rpc_url).strip())
            self._restart_same_args()
        except Exception as e:
            ui.notify(f"Failed to apply: {e}", type="negative")

    def _restart_same_args(self) -> None:
        ui.notify("Restarting...", type="info")
        os.execv(sys.executable, [sys.executable] + sys.argv)


__all__ = ["SettingsTab"]
