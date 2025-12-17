import multiprocessing
from dataclasses import dataclass
from typing import Any, Callable, Dict


class SandboxedExecutionError(Exception):
    """Raised when sandboxed code fails or times out."""


@dataclass
class ZeoneAPI:
    """Minimal safe API surface exposed to agent code."""

    log: Callable[[str], None]
    get_balance: Callable[[], Any]
    send_message: Callable[[str, str], None]


def _run_code(code: str, api: ZeoneAPI, result_conn: Any) -> None:
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
        agent_output = local_env.get("agent_output")
        history = local_env.get("history")
        result_conn.send(
            {"ok": True, "fitness": fitness, "agent_output": agent_output, "history": history}
        )
    except Exception as e:
        try:
            result_conn.send({"ok": False, "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            result_conn.close()
        except Exception:
            pass


class AgentSandbox:
    """
    Executes untrusted agent code with time limits and restricted builtins.
    """

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout

    def run(self, code: str, api: ZeoneAPI) -> Dict[str, Any]:
        """Run code in a separate process with a hard timeout."""
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_run_code, args=(code, api, child_conn))
        proc.start()
        child_conn.close()
        proc.join(self.timeout)
        if proc.is_alive():
            proc.terminate()
            try:
                parent_conn.close()
            except Exception:
                pass
            return {"ok": False, "error": "timeout"}
        try:
            if parent_conn.poll(0.01):
                return parent_conn.recv()
            return {"ok": False, "error": "no result"}
        except Exception:
            return {"ok": False, "error": "no result"}
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass
