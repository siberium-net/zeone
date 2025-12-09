import multiprocessing
import signal
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable


class SandboxedExecutionError(Exception):
    """Raised when sandboxed code fails or times out."""


@dataclass
class ZeoneAPI:
    """Minimal safe API surface exposed to agent code."""

    log: Callable[[str], None]
    get_balance: Callable[[], Any]
    send_message: Callable[[str, str], None]


def _run_code(code: str, api: ZeoneAPI, result_queue: multiprocessing.Queue) -> None:
    """Execute user code with restricted builtins."""
    safe_builtins: Dict[str, Any] = {
        "True": True,
        "False": False,
        "None": None,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
    }
    sandbox_globals = {
        "__builtins__": safe_builtins,
        "api": api,
    }
    local_env: Dict[str, Any] = {}
    try:
        exec(code, sandbox_globals, local_env)
        fitness = local_env.get("fitness") or sandbox_globals.get("fitness")
        result_queue.put({"ok": True, "fitness": fitness})
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})


class AgentSandbox:
    """
    Executes untrusted agent code with time limits and restricted builtins.
    """

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout

    def run(self, code: str, api: ZeoneAPI) -> Dict[str, Any]:
        """Run code in a separate process with a hard timeout."""
        ctx = multiprocessing.get_context("spawn")
        result_queue: multiprocessing.Queue = ctx.Queue()
        proc = ctx.Process(target=_run_code, args=(code, api, result_queue))
        proc.start()
        proc.join(self.timeout)
        if proc.is_alive():
            proc.terminate()
            return {"ok": False, "error": "timeout"}
        try:
            return result_queue.get_nowait()
        except Exception:
            return {"ok": False, "error": "no result"}
