"""
Zeone MCP Server
================

Provides MCP resources and tools over SSE transport.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from mcp.server.fastmcp import FastMCP

from config import config, get_current_network
from cortex.genesis import run_genesis
from cortex.evolution.sandbox import AgentSandbox, ZeoneAPI

logger = logging.getLogger(__name__)


_HTML_RE = re.compile(r"<[^>]+>")


def _strip_html(value: str) -> str:
    return _HTML_RE.sub("", value)


def _safe_agent_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip())
    cleaned = re.sub(r"__+", "_", cleaned).strip("_")
    return cleaned or "agent"


def _sandbox_log(message: str) -> None:
    logger.info("[MCP_AGENT] %s", message)


def _sandbox_get_balance() -> float:
    return 0.0


def _sandbox_send_message(_peer_id: str, _message: str) -> None:
    return None


@dataclass
class MCPConfig:
    host: str = "0.0.0.0"
    port: int = 8090
    sse_path: str = "/mcp/sse"
    message_path: str = "/mcp/messages"


class ZeoneMCPServer:
    def __init__(
        self,
        *,
        node: Any,
        ledger: Any,
        agent_manager: Any,
        cortex: Any = None,
        log_buffer: Optional[Deque[str]] = None,
        mcp_config: Optional[MCPConfig] = None,
    ) -> None:
        self.node = node
        self.ledger = ledger
        self.agent_manager = agent_manager
        self.cortex = cortex
        self.log_buffer = log_buffer or deque(maxlen=1000)
        self._chain_manager = None

        cfg = mcp_config or MCPConfig()
        self._mcp = FastMCP(
            "zeone",
            instructions="ZEONE MCP server for node control and diagnostics.",
            host=cfg.host,
            port=cfg.port,
            sse_path=cfg.sse_path,
            message_path=cfg.message_path,
        )

        self._register_resources()
        self._register_tools()

    @property
    def mcp(self) -> FastMCP:
        return self._mcp

    def sse_app(self):
        return self._mcp.sse_app()

    async def run_sse(self, mount_path: Optional[str] = None) -> None:
        await self._mcp.run_sse_async(mount_path)

    def start_sse(self) -> asyncio.Task:
        return asyncio.create_task(self.run_sse())

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def _register_resources(self) -> None:
        @self._mcp.resource("zeone://balance")
        async def balance() -> Dict[str, Any]:
            if not self.ledger:
                return {"error": "ledger_unavailable"}
            stats = await self.ledger.get_stats()
            return {
                "node_id": getattr(self.node, "node_id", ""),
                "stats": stats,
            }

        @self._mcp.resource("zeone://logs/recent")
        def logs_recent() -> Dict[str, Any]:
            lines = list(self.log_buffer)[-50:]
            cleaned = [_strip_html(str(line)) for line in lines]
            return {"lines": cleaned, "count": len(cleaned)}

        @self._mcp.resource("zeone://peers")
        def peers() -> Dict[str, Any]:
            if not self.node:
                return {"peers": [], "count": 0}
            peer_manager = getattr(self.node, "peer_manager", None)
            if not peer_manager:
                return {"peers": [], "count": 0}
            peers = []
            for peer in peer_manager.get_active_peers():
                peers.append(
                    {
                        "node_id": peer.node_id,
                        "host": peer.host,
                        "port": peer.port,
                        "trust_score": peer.trust_score,
                        "last_seen": peer.last_seen,
                        "bytes_sent": peer.bytes_sent,
                        "bytes_received": peer.bytes_received,
                        "blocked": peer.blocked,
                    }
                )
            return {"peers": peers, "count": len(peers)}

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _register_tools(self) -> None:
        @self._mcp.tool()
        async def send_tokens(address: str, amount: float) -> Dict[str, Any]:
            """Transfer SIBR tokens on-chain."""
            try:
                if amount <= 0:
                    return {"ok": False, "error": "amount_must_be_positive"}
                manager = await self._get_chain_manager()
                if manager is None:
                    return {"ok": False, "error": "chain_manager_unavailable"}
                tx_hash = await manager.transfer(address, Decimal(str(amount)))
                return {"ok": True, "tx_hash": tx_hash}
            except Exception as e:
                logger.exception("[MCP] send_tokens failed")
                return {"ok": False, "error": str(e)}

        @self._mcp.tool()
        async def ask_cortex(query: str) -> Dict[str, Any]:
            """Ask local Cortex (LLM/RAG)."""
            if not self.cortex:
                return {"ok": False, "error": "cortex_unavailable"}
            try:
                if hasattr(self.cortex, "search_or_investigate"):
                    result = await self.cortex.search_or_investigate(query)
                    return {"ok": True, "result": result}
                if hasattr(self.cortex, "search"):
                    reports = await self.cortex.search(query)
                    items = [r.to_dict() if hasattr(r, "to_dict") else r for r in reports]
                    return {"ok": True, "reports": items, "count": len(items)}
                return {"ok": False, "error": "cortex_no_query_api"}
            except Exception as e:
                logger.exception("[MCP] ask_cortex failed")
                return {"ok": False, "error": str(e)}

        @self._mcp.tool()
        async def deploy_agent(code: str, name: str) -> Dict[str, Any]:
            """Save code and register a sandboxed agent."""
            if not self.agent_manager:
                return {"ok": False, "error": "agent_manager_unavailable"}

            max_size = int(getattr(config.agent, "max_code_size", 65536))
            if len(code.encode("utf-8")) > max_size:
                return {"ok": False, "error": f"code_too_large (max {max_size} bytes)"}

            safe_name = _safe_agent_name(name)
            target_dir = Path("agents/custom")
            target_dir.mkdir(parents=True, exist_ok=True)
            path = target_dir / f"{safe_name}.py"
            path.write_text(code, encoding="utf-8")

            try:
                from agents.manager import BaseAgent

                sandbox = AgentSandbox(timeout=float(getattr(config.agent, "max_execution_time", 5.0)))

                class DeployedAgent(BaseAgent):
                    def __init__(self, agent_code: str):
                        super().__init__()
                        self._code = agent_code

                    @property
                    def service_name(self) -> str:  # type: ignore[override]
                        return safe_name

                    @property
                    def price_per_unit(self) -> float:  # type: ignore[override]
                        return 0.0

                    async def execute(self, payload: Any):  # type: ignore[override]
                        api = ZeoneAPI(
                            log=_sandbox_log,
                            get_balance=_sandbox_get_balance,
                            send_message=_sandbox_send_message,
                        )
                        result = sandbox.run(self._code, api)
                        return result, 0.0

                self.agent_manager.register_agent(DeployedAgent(code))
            except Exception as e:
                logger.exception("[MCP] deploy_agent failed")
                return {"ok": False, "error": str(e)}

            return {"ok": True, "path": str(path), "service": safe_name}

        @self._mcp.tool()
        async def evolve(niche: str) -> Dict[str, Any]:
            """Run Genesis evolution with optional niche override."""
            try:
                result = await run_genesis(niche=niche or None)
                return {"ok": True, "result": result}
            except Exception as e:
                logger.exception("[MCP] evolve failed")
                return {"ok": False, "error": str(e)}

    async def _get_chain_manager(self):
        if self._chain_manager is not None:
            return self._chain_manager

        try:
            from economy.chain import SiberiumManager
        except Exception as e:
            logger.warning("[MCP] SiberiumManager unavailable: %s", e)
            return None

        env_key = (os.getenv("PRIVATE_KEY") or "").strip()
        if not env_key:
            logger.warning("[MCP] PRIVATE_KEY not set; cannot send tokens")
            return None

        rpc_url = (os.getenv("SIBERIUM_RPC_URL") or config.network.rpc_url).strip()
        settlement = (os.getenv("SIBERIUM_SETTLEMENT_ADDRESS") or config.network.contract_address).strip()

        try:
            self._chain_manager = SiberiumManager(
                rpc_url=rpc_url,
                private_key=env_key,
                settlement_address=settlement or None,
            )
            net = get_current_network()
            logger.info("[MCP] Chain manager ready: %s (%s)", net.get("name"), net.get("chain_id"))
            return self._chain_manager
        except Exception as e:
            logger.warning("[MCP] Failed to init chain manager: %s", e)
            return None
