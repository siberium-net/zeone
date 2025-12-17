"""
AI Interface for Cortex / Evolution
===================================

`IntellectualCore` abstracts over available LLM sources and provides a
single entry point for generating a structured species spec for the
GGGP evolutionary engine.

Priority order (from smartest to cheapest):
1. External API via OpenAI-compatible cloud agent (if `LLM_API_KEY` is set)
2. Distributed swarm inference (if a distributed client/agent is configured
   and peers can serve a 70B Llama model)
3. Local Ollama inference (fallback)

The module is defensive against "stupid" / non‑compliant model outputs:
it extracts JSON from noisy text and falls back to a heuristic spec.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class IntellectualCore:
    """
    Selects the best available intelligence provider and generates a species spec.

    You may pass pre‑constructed agents/clients; otherwise the core will try to
    instantiate cloud/local agents from the environment.
    """

    def __init__(
        self,
        cloud_agent: Optional[Any] = None,
        distributed_agent: Optional[Any] = None,
        local_agent: Optional[Any] = None,
        distributed_client: Optional[Any] = None,
        distributed_model_candidates: Optional[Tuple[str, ...]] = None,
    ):
        self.cloud_agent = cloud_agent
        self.distributed_agent = distributed_agent
        self.local_agent = local_agent
        self.distributed_client = distributed_client
        self.distributed_model_candidates = distributed_model_candidates or (
            "llama2-70b",
            "llama2-70b-chat",
            "llama-70b",
        )

    async def generate_spec(self, user_goal: str) -> Dict[str, Any]:
        """
        Turn a natural language goal into a validated species spec dict.
        """
        prompts = self._build_prompts(user_goal)
        system_prompt = prompts["system"]
        user_prompt = prompts["user"]

        agent, meta = await self._select_agent()
        if not agent:
            logger.warning("[AI] No agent available, using heuristic spec.")
            return self._heuristic_spec(user_goal)

        response_text = await self._run_agent(
            agent,
            system_prompt,
            user_prompt,
            model=meta.get("model"),
        )

        spec_obj = self._parse_json(response_text)
        if not isinstance(spec_obj, dict):
            logger.warning("[AI] Model returned non‑JSON spec, using heuristic.")
            return self._heuristic_spec(user_goal)

        # Prefer central sanitizer if available.
        try:
            from cortex.evolution.architect import sanitize_species_spec

            return sanitize_species_spec(spec_obj, user_goal=user_goal)
        except Exception as e:
            logger.debug(f"[AI] sanitize_species_spec unavailable: {e}")
            return self._normalize_spec(spec_obj, user_goal)

    # ------------------------------------------------------------------
    # Agent selection
    # ------------------------------------------------------------------

    async def _select_agent(self) -> Tuple[Optional[Any], Dict[str, Any]]:
        # 1) Cloud
        if self._has_cloud_key():
            if self.cloud_agent is None:
                self.cloud_agent = self._try_init_cloud()
            if self.cloud_agent is not None:
                return self.cloud_agent, {"source": "cloud"}

        # 2) Distributed swarm
        dist_agent = self.distributed_agent
        dist_client = (
            self.distributed_client
            or getattr(dist_agent, "distributed_client", None)
        )
        if dist_client is not None:
            model = await self._pick_distributed_model(dist_client)
            if model:
                if dist_agent is None:
                    dist_agent = self._try_init_distributed(dist_client, model)
                    self.distributed_agent = dist_agent
                if dist_agent is not None:
                    return dist_agent, {"source": "distributed", "model": model}

        # 3) Local Ollama
        if self.local_agent is None:
            self.local_agent = self._try_init_local()
        if self.local_agent is not None:
            return self.local_agent, {"source": "local"}

        return None, {}

    def _has_cloud_key(self) -> bool:
        return bool(os.getenv("LLM_API_KEY", "").strip())

    def _try_init_cloud(self) -> Optional[Any]:
        try:
            from agents.ai_assistant import LlmAgent

            return LlmAgent()
        except Exception as e:
            logger.debug(f"[AI] Cloud agent init failed: {e}")
            return None

    def _try_init_local(self) -> Optional[Any]:
        try:
            from agents.local_llm import OllamaAgent

            return OllamaAgent()
        except Exception as e:
            logger.debug(f"[AI] Local agent init failed: {e}")
            return None

    def _try_init_distributed(self, client: Any, model: str) -> Optional[Any]:
        try:
            from agents.distributed_agent import DistributedLlmAgent

            return DistributedLlmAgent(distributed_client=client, default_model=model)
        except Exception as e:
            logger.debug(f"[AI] Distributed agent init failed: {e}")
            return None

    async def _pick_distributed_model(self, client: Any) -> Optional[str]:
        for name in self.distributed_model_candidates:
            try:
                info = await client.check_availability(name)
                if isinstance(info, dict) and info.get("distributed_available"):
                    return name
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # LLM call / parsing
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        agent: Any,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
    ) -> str:
        payload = {
            "prompt": user_prompt,
            "system": system_prompt,
            "system_prompt": system_prompt,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if model:
            payload["model"] = model

        try:
            res = await agent.execute(payload)
        except Exception as e:
            logger.warning(f"[AI] Agent execute failed: {e}")
            return ""

        # Unify possible return shapes
        result_obj: Any = res
        error: Optional[str] = None
        if isinstance(res, tuple):
            if len(res) == 2:
                result_obj, _units = res
            elif len(res) == 3:
                result_obj, _cost, error = res

        if error:
            logger.warning(f"[AI] Agent returned error: {error}")
            return ""

        if isinstance(result_obj, dict):
            if "error" in result_obj:
                logger.warning(f"[AI] Agent error: {result_obj.get('error')}")
            text = result_obj.get("response") or result_obj.get("text") or ""
            return str(text)

        return str(result_obj)

    def _parse_json(self, text: str) -> Optional[Any]:
        if not text:
            return None

        cleaned = text.strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        # Remove markdown fences
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # Extract first JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except Exception:
            try:
                return ast.literal_eval(cleaned)
            except Exception:
                return None

    def _build_prompts(self, user_goal: str) -> Dict[str, str]:
        try:
            from cortex.evolution.prompts import format_species_spec_prompt

            return format_species_spec_prompt(user_goal)
        except Exception:
            # Minimal fallback prompt to keep module standalone.
            system = (
                "You are a metaprogramming architect. "
                "Return ONLY valid JSON for a GGGP species spec."
            )
            user = (
                f"User goal: {user_goal}\n\n"
                "Return a JSON object with keys: species, grammar, fitness."
            )
            return {"system": system, "user": user}

    # ------------------------------------------------------------------
    # Fallbacks / normalization
    # ------------------------------------------------------------------

    def _heuristic_spec(self, user_goal: str) -> Dict[str, Any]:
        goal_l = user_goal.lower()
        domain = "trading" if any(k in goal_l for k in ("trade", "торг", "buy", "sell", "profit", "прибыл")) else "general"
        target_asset = None
        if "sibr" in goal_l or "сибр" in goal_l:
            target_asset = "SIBR"

        return {
            "species": {
                "name": self._slugify(user_goal) or "species",
                "goal": user_goal,
                "domain": domain,
                "target_asset": target_asset,
            },
            "grammar": {
                "actions": [
                    {"name": "buy", "params": {"amount": {"min": 0.1, "max": 10.0}}},
                    {"name": "sell", "params": {"amount": {"min": 0.1, "max": 10.0}}},
                    {"name": "hold", "params": {}},
                ],
                "conditions": [
                    {
                        "left": "balance",
                        "comparators": ["<", "<=", ">", ">=", "=="],
                        "threshold_range": [0, 100],
                    }
                ],
                "rule_count": {"min": 2, "max": 5},
                "max_depth": 2,
            },
            "fitness": {
                "objective": "maximize_balance",
                "notes": "Default fitness uses api.get_balance() deltas.",
            },
        }

    def _normalize_spec(self, spec: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        base = self._heuristic_spec(user_goal)
        merged = base
        merged.update({k: v for k, v in spec.items() if v is not None})
        return merged

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
        return slug[:64]

