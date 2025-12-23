"""
Evolution Tab (GGGP)
====================

WebUI control + monitoring for the genetic evolution engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional

from nicegui import ui

from cortex.ai_interface import IntellectualCore
from cortex.evolution.challenges import PricingChallenge
from cortex.evolution.engine import EvolutionEngine, SimpleAPI
from cortex.evolution.genetics import compile_genome
from cortex.evolution.sandbox import AgentSandbox

logger = logging.getLogger(__name__)


class EvolutionTab:
    def __init__(self, parent=None):
        self.parent = parent
        self.core = IntellectualCore()

        # UI elements
        self._goal_input = None
        self._spec_editor = None
        self._spec_status = None
        self._population_input = None
        self._start_button = None
        self._deploy_button = None
        self._chart = None
        self._log = None
        self._best_code = None
        self._market_chart = None
        self._market_status = None

        # State
        self._species_spec: Dict[str, Any] = {}
        self._engine: Optional[EvolutionEngine] = None
        self._running = False
        self._stats_queue: queue.Queue = queue.Queue()
        self._executor_future: Optional[asyncio.Future] = None
        self._timer = None

        self._gens: List[int] = []
        self._max_fitness: List[float] = []
        self._avg_fitness: List[float] = []
        self._best_code_text: str = ""
        self._market_times: List[str] = []
        self._market_prices: List[float] = []
        self._market_demand: List[float] = []
        self._market_running = False

    # ------------------------------------------------------------------
    # Page
    # ------------------------------------------------------------------

    def create_page(self, parent):
        self.parent = parent

        @ui.page("/evolution")
        async def evolution():
            await parent._create_header()
            await parent._create_sidebar()

            with ui.row().classes("w-full p-4 gap-6"):
                # Control panel (left)
                with ui.column().classes("w-1/3 gap-3"):
                    ui.label("Evolution Control").classes("text-2xl font-bold mb-1")

                    self._goal_input = ui.textarea(
                        "Goal Description",
                        placeholder="Optimize storage usage...",
                    ).classes("w-full h-40")

                    ui.button(
                        "Design Species",
                        icon="architecture",
                        on_click=self._on_design_species,
                    ).classes("w-full")

                    self._spec_status = ui.row().classes("items-center gap-2 text-sm text-gray-400")

                    self._spec_editor = ui.json_editor(
                        {"content": {"json": self._species_spec or {}}},
                        on_change=self._on_spec_change,
                    ).classes("w-full h-80")

                    self._population_input = ui.number(
                        "Population Size",
                        value=50,
                        min=2,
                        max=500,
                    ).classes("w-full")

                    self._start_button = ui.button(
                        "START EVOLUTION",
                        icon="play_arrow",
                        on_click=self._on_start_evolution,
                    ).props("color=primary").classes("w-full")

                    self._deploy_button = ui.button(
                        "Deploy Alpha",
                        icon="rocket_launch",
                        on_click=self._on_deploy_alpha,
                    ).props("color=positive").classes("w-full")

                # Monitoring (right)
                with ui.column().classes("w-2/3 gap-3"):
                    ui.label("Monitoring").classes("text-2xl font-bold mb-1")

                    self._chart = ui.echart(self._chart_options()).classes("w-full h-64")

                    self._log = ui.log(max_lines=1000).classes("w-full h-48")

                    with ui.card().classes("w-full"):
                        ui.label("Market Simulation").classes("text-lg font-semibold")
                        self._market_status = ui.label("Idle").classes("text-sm text-gray-400")
                        ui.button(
                            "Run Market Simulation",
                            icon="timeline",
                            on_click=self._on_run_market_simulation,
                        ).props("color=secondary").classes("w-full")
                        self._market_chart = ui.echart(self._market_chart_options()).classes(
                            "w-full h-64"
                        )

                    with ui.card().classes("w-full"):
                        ui.label("Best Agent Code").classes("text-lg font-semibold mb-2")
                        self._best_code = ui.code("", language="python").classes(
                            "w-full h-96 overflow-auto"
                        )

            # Poll background stats queue
            self._timer = ui.timer(0.5, self._drain_stats_queue)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _on_design_species(self):
        goal = self._goal_input.value if self._goal_input else ""
        if not goal or not str(goal).strip():
            ui.notify("Enter goal description", type="warning")
            return

        if self._spec_status:
            try:
                self._spec_status.clear()
                with self._spec_status:
                    ui.label("Designing...").classes("text-blue-400")
                    ui.spinner()
            except Exception:
                pass

        try:
            spec = await self.core.generate_spec(str(goal))
            self._species_spec = spec if isinstance(spec, dict) else {}
            self._set_editor_spec(self._species_spec)
            ui.notify("Species spec generated", type="positive")
        except Exception as e:
            logger.exception("[WEBUI] Species design failed")
            ui.notify(f"Design failed: {e}", type="negative")
        finally:
            if self._spec_status:
                try:
                    self._spec_status.clear()
                except Exception:
                    pass

    def _on_spec_change(self, e):
        content = getattr(e, "content", None)
        if isinstance(content, dict):
            json_obj = content.get("json")
            if isinstance(json_obj, dict):
                self._species_spec = json_obj
                return
            text_obj = content.get("text")
            if isinstance(text_obj, str):
                try:
                    self._species_spec = json.loads(text_obj)
                except Exception:
                    pass

    async def _on_start_evolution(self):
        if self._running:
            ui.notify("Evolution already running", type="warning")
            return

        spec = self._species_spec or {}
        pop_size = 50
        if self._population_input:
            try:
                pop_size = int(self._population_input.value)
            except Exception:
                pop_size = 50

        try:
            self._engine = EvolutionEngine(population_size=pop_size, species_spec=spec)
            self._engine.initialize_population()
        except Exception as e:
            logger.exception("[WEBUI] Failed to initialize EvolutionEngine")
            ui.notify(f"Engine init failed: {e}", type="negative")
            return

        # Reset chart state
        self._gens.clear()
        self._max_fitness.clear()
        self._avg_fitness.clear()
        self._best_code_text = ""
        self._update_chart()

        self._running = True
        self._push_log("Evolution started...")

        loop = asyncio.get_running_loop()
        self._executor_future = loop.run_in_executor(None, self._run_evolution_loop)

    def _run_evolution_loop(self):
        async def _inner():
            assert self._engine is not None
            while self._running:
                best, avg, best_gene = await self._engine.run_epoch()
                code = compile_genome(best_gene)
                self._stats_queue.put(
                    {
                        "gen": int(self._engine.generation),
                        "best": float(best),
                        "avg": float(avg),
                        "code": code,
                    }
                )
                await asyncio.sleep(0)

        try:
            asyncio.run(_inner())
        except Exception as e:
            self._stats_queue.put({"error": str(e)})

    async def _on_deploy_alpha(self):
        if not self._best_code_text:
            ui.notify("No best agent code yet", type="warning")
            return

        try:
            target_dir = Path("agents/custom")
            target_dir.mkdir(parents=True, exist_ok=True)
            path = target_dir / "alpha_v1.py"
            path.write_text(self._best_code_text)
            ui.notify(f"Saved best agent to {path}", type="positive")
        except Exception as e:
            ui.notify(f"Save failed: {e}", type="negative")
            return

        # Hot register into AgentManager if available
        manager = getattr(self.parent, "agent_manager", None)
        if not manager:
            ui.notify("AgentManager not available for hot deploy", type="warning")
            return

        try:
            from agents.manager import BaseAgent

            class AlphaV1Agent(BaseAgent):
                def __init__(self, code: str):
                    super().__init__()
                    self._code = code
                    self._sandbox = AgentSandbox(timeout=1.0)

                @property
                def service_name(self) -> str:  # type: ignore[override]
                    return "alpha_v1"

                @property
                def price_per_unit(self) -> float:  # type: ignore[override]
                    return 0.0

                async def execute(self, payload: Any):  # type: ignore[override]
                    api = None
                    if isinstance(payload, dict):
                        api = payload.get("api")
                    if api is None:
                        api = SimpleAPI()
                    result = self._sandbox.run(self._code, api)
                    return result, 0.0

            manager.register_agent(AlphaV1Agent(self._best_code_text))
            ui.notify("Alpha agent registered", type="positive")
        except Exception as e:
            logger.exception("[WEBUI] Hot deploy failed")
            ui.notify(f"Hot deploy failed: {e}", type="warning")

    async def _on_run_market_simulation(self):
        if self._market_running:
            ui.notify("Market simulation already running", type="warning")
            return

        code = self._best_code_text
        if not code:
            code = PricingChallenge.baseline_code()
            ui.notify("Using baseline pricing agent (no evolved code yet)", type="warning")

        self._market_running = True
        if self._market_status:
            self._market_status.text = "Running simulation..."

        try:
            challenge = PricingChallenge()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, challenge.simulate_agent_code, code, False)

            self._market_times = [f"{tick.time_of_day:.1f}" for tick in result.ticks]
            self._market_prices = [float(tick.agent_price) for tick in result.ticks]
            self._market_demand = [float(tick.demand) for tick in result.ticks]
            self._update_market_chart()

            if self._market_status:
                self._market_status.text = (
                    f"Fitness {result.fitness:.1f} | Profit {result.total_profit:.1f} "
                    f"| Overload {result.overload_minutes}m"
                )
        except Exception as e:
            logger.exception("[WEBUI] Market simulation failed")
            ui.notify(f"Simulation failed: {e}", type="negative")
            if self._market_status:
                self._market_status.text = "Simulation failed"
        finally:
            self._market_running = False

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _chart_options(self) -> Dict[str, Any]:
        return {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Max Fitness", "Avg Fitness"]},
            "xAxis": {"type": "category", "data": []},
            "yAxis": {"type": "value"},
            "series": [
                {"name": "Max Fitness", "type": "line", "data": []},
                {"name": "Avg Fitness", "type": "line", "data": []},
            ],
            "animation": False,
        }

    def _market_chart_options(self) -> Dict[str, Any]:
        return {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Agent Price", "Market Demand"]},
            "xAxis": {"type": "category", "data": []},
            "yAxis": [
                {"type": "value", "name": "Price"},
                {"type": "value", "name": "Demand", "position": "right"},
            ],
            "series": [
                {"name": "Agent Price", "type": "line", "data": [], "yAxisIndex": 0},
                {"name": "Market Demand", "type": "line", "data": [], "yAxisIndex": 1},
            ],
            "animation": False,
        }

    def _set_editor_spec(self, spec: Dict[str, Any]) -> None:
        if not self._spec_editor:
            return
        try:
            props = self._spec_editor.properties
            props["content"] = {"json": spec}
            self._spec_editor.update()
        except Exception:
            pass

    def _drain_stats_queue(self):
        updated = False
        while True:
            try:
                stats = self._stats_queue.get_nowait()
            except queue.Empty:
                break

            if "error" in stats:
                self._push_log(f"Evolution stopped: {stats['error']}")
                self._running = False
                continue

            gen = stats.get("gen", 0)
            best = stats.get("best", 0.0)
            avg = stats.get("avg", 0.0)
            code = stats.get("code", "")

            self._gens.append(int(gen))
            self._max_fitness.append(float(best))
            self._avg_fitness.append(float(avg))
            self._best_code_text = str(code)

            self._push_log(f"Gen {gen} complete. Max {best:.2f}, Avg {avg:.2f}")
            self._update_chart()
            self._update_best_code()
            updated = True

        if updated and self._chart:
            try:
                self._chart.update()
            except Exception:
                pass

    def _update_chart(self):
        if not self._chart:
            return
        try:
            self._chart.options["xAxis"]["data"] = self._gens
            self._chart.options["series"][0]["data"] = self._max_fitness
            self._chart.options["series"][1]["data"] = self._avg_fitness
        except Exception:
            pass

    def _update_market_chart(self):
        if not self._market_chart:
            return
        try:
            self._market_chart.options["xAxis"]["data"] = self._market_times
            self._market_chart.options["series"][0]["data"] = self._market_prices
            self._market_chart.options["series"][1]["data"] = self._market_demand
            self._market_chart.update()
        except Exception:
            pass

    def _update_best_code(self):
        if not self._best_code:
            return
        try:
            self._best_code.content = self._best_code_text
            self._best_code.update()
        except Exception:
            pass

    def _push_log(self, text: str):
        if not self._log:
            return
        try:
            self._log.push(text)
            self._log.scroll_to_end()
        except Exception:
            pass


__all__ = ["EvolutionTab"]
