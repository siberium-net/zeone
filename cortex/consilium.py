"""
Cortex Consilium - Оркестратор консенсуса через совет аналитиков
===============================================================

[CONSILIUM] Механизм решения сложных задач через консенсус:
1. Публикация задачи в сеть (Broadcast)
2. Выбор исполнителей по Trust Score
3. Параллельное выполнение
4. Self-Reflection через Judge LLM
5. Выплата наград (IOU)

[FLOW]
Topic → Broadcast → Select 3 Analysts → Parallel Analysis →
→ Judge Synthesis → Final Answer → Pay Rewards
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from core.node import Node
    from economy.ledger import Ledger
    from agents.local_llm import OllamaAgent

from .prompts import format_judge_prompt, format_analyst_prompt
from .roles import Analyst, Task, RoleResult, AnalysisResult

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class CouncilStatus(Enum):
    """Статус совета."""
    PENDING = "pending"
    RECRUITING = "recruiting"
    ANALYZING = "analyzing"
    JUDGING = "judging"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalystCandidate:
    """Кандидат-аналитик для совета."""
    node_id: str
    trust_score: float
    is_local: bool = False
    response_time: float = 0.0


@dataclass
class AnalystResponse:
    """Ответ аналитика."""
    analyst_id: str
    analysis: Optional[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    cost: float = 0.0


@dataclass
class CouncilResult:
    """Результат работы совета."""
    topic: str
    status: CouncilStatus
    final_summary: str
    consensus_facts: List[str]
    disputed_facts: List[str]
    filtered_hallucinations: List[str]
    sentiment_consensus: str
    confidence: float
    analyst_agreement: float
    reasoning: str
    participants: List[str]
    total_cost: float
    execution_time: float
    raw_responses: List[AnalystResponse] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в словарь."""
        return {
            "topic": self.topic,
            "status": self.status.value,
            "final_summary": self.final_summary,
            "consensus_facts": self.consensus_facts,
            "disputed_facts": self.disputed_facts,
            "filtered_hallucinations": self.filtered_hallucinations,
            "sentiment_consensus": self.sentiment_consensus,
            "confidence": self.confidence,
            "analyst_agreement": self.analyst_agreement,
            "reasoning": self.reasoning,
            "participants": self.participants,
            "total_cost": self.total_cost,
            "execution_time": self.execution_time,
        }


@dataclass
class CouncilSession:
    """Сессия совета."""
    session_id: str
    topic: str
    budget: float
    status: CouncilStatus
    created_at: float
    analysts: List[AnalystCandidate] = field(default_factory=list)
    responses: List[AnalystResponse] = field(default_factory=list)
    result: Optional[CouncilResult] = None


# =============================================================================
# CONSILIUM ORCHESTRATOR
# =============================================================================

class Consilium:
    """
    Оркестратор консенсуса - "Совет Аналитиков".
    
    [USAGE]
    ```python
    consilium = Consilium(node, ledger, judge_llm)
    result = await consilium.convene_council(
        topic="quantum computing",
        text="...",
        budget=100.0,
    )
    ```
    
    [PROCESS]
    1. convene_council() получает тему и бюджет
    2. Выбираются 3 аналитика с высоким Trust Score
    3. Каждый анализирует текст параллельно
    4. Judge LLM синтезирует финальный ответ
    5. Награды распределяются участникам
    """
    
    # Configuration
    DEFAULT_ANALYST_COUNT = 3
    ANALYST_TIMEOUT = 120.0  # секунды
    MIN_RESPONSES_FOR_CONSENSUS = 2
    
    def __init__(
        self,
        node: Optional["Node"] = None,
        ledger: Optional["Ledger"] = None,
        judge_llm: Optional["OllamaAgent"] = None,
        local_analyst: Optional[Analyst] = None,
        node_id: str = "",
    ):
        """
        Args:
            node: P2P Node для broadcast и выбора пиров
            ledger: Ledger для IOU транзакций
            judge_llm: LLM для синтеза ответов (Judge)
            local_analyst: Локальный Analyst для участия в совете
            node_id: ID текущего узла
        """
        self.node = node
        self.ledger = ledger
        self.judge = judge_llm
        self.local_analyst = local_analyst
        self.node_id = node_id
        
        # Active sessions
        self._sessions: Dict[str, CouncilSession] = {}
        
        # Stats
        self._total_councils = 0
        self._successful_councils = 0
    
    async def convene_council(
        self,
        topic: str,
        text: str,
        budget: float = 50.0,
        analyst_count: int = DEFAULT_ANALYST_COUNT,
        include_local: bool = True,
    ) -> CouncilResult:
        """
        Созвать совет аналитиков для анализа темы.
        
        Args:
            topic: Тема для анализа
            text: Текст для анализа
            budget: Бюджет для оплаты участников
            analyst_count: Количество аналитиков
            include_local: Включить локального аналитика
        
        Returns:
            CouncilResult с синтезированным ответом
        """
        start_time = time.time()
        session_id = f"council_{int(time.time())}_{hash(topic) % 10000}"
        
        logger.info(f"[CONSILIUM] Convening council for '{topic}' budget={budget}")
        
        session = CouncilSession(
            session_id=session_id,
            topic=topic,
            budget=budget,
            status=CouncilStatus.RECRUITING,
            created_at=start_time,
        )
        self._sessions[session_id] = session
        
        try:
            # 1. RECRUITING: Выбираем аналитиков
            session.status = CouncilStatus.RECRUITING
            analysts = await self._select_analysts(analyst_count, include_local)
            session.analysts = analysts
            
            if len(analysts) < self.MIN_RESPONSES_FOR_CONSENSUS:
                logger.warning(f"[CONSILIUM] Not enough analysts: {len(analysts)}")
                return self._create_failed_result(
                    topic, "Not enough analysts available", start_time
                )
            
            logger.info(f"[CONSILIUM] Selected {len(analysts)} analysts")
            
            # 2. ANALYZING: Параллельный анализ
            session.status = CouncilStatus.ANALYZING
            responses = await self._parallel_analysis(text, topic, analysts)
            session.responses = responses
            
            successful_responses = [r for r in responses if r.success]
            
            if len(successful_responses) < self.MIN_RESPONSES_FOR_CONSENSUS:
                logger.warning(
                    f"[CONSILIUM] Not enough successful responses: {len(successful_responses)}"
                )
                return self._create_failed_result(
                    topic, "Not enough analysts responded successfully", start_time
                )
            
            logger.info(
                f"[CONSILIUM] Got {len(successful_responses)}/{len(responses)} successful analyses"
            )
            
            # 3. JUDGING: Синтез через Judge LLM
            session.status = CouncilStatus.JUDGING
            result = await self._judge_synthesis(topic, successful_responses, start_time)
            
            if result.status == CouncilStatus.COMPLETED:
                # 4. PAY: Выплачиваем награды
                await self._pay_rewards(session, successful_responses, budget)
                self._successful_councils += 1
            
            session.status = result.status
            session.result = result
            self._total_councils += 1
            
            logger.info(
                f"[CONSILIUM] Council completed: confidence={result.confidence:.2f} "
                f"agreement={result.analyst_agreement:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[CONSILIUM] Council failed: {e}")
            session.status = CouncilStatus.FAILED
            return self._create_failed_result(topic, str(e), start_time)
    
    # =========================================================================
    # STEP 1: SELECT ANALYSTS
    # =========================================================================
    
    async def _select_analysts(
        self,
        count: int,
        include_local: bool,
    ) -> List[AnalystCandidate]:
        """Выбрать аналитиков по Trust Score."""
        candidates = []
        
        # Добавляем локального аналитика если есть
        if include_local and self.local_analyst:
            candidates.append(AnalystCandidate(
                node_id=self.node_id,
                trust_score=1.0,  # Себе доверяем максимально
                is_local=True,
            ))
        
        # Получаем пиров с высоким Trust Score
        if self.node and self.ledger:
            peers = self._get_trusted_peers(count * 2)  # Берём с запасом
            
            for peer in peers:
                if len(candidates) >= count:
                    break
                
                # Проверяем Trust Score
                trust = await self.ledger.get_trust_score(peer.node_id)
                
                if trust >= 0.3:  # Минимальный порог доверия
                    candidates.append(AnalystCandidate(
                        node_id=peer.node_id,
                        trust_score=trust,
                        is_local=False,
                    ))
        
        # Если не хватает - добавляем локальных копий
        while len(candidates) < count and include_local and self.local_analyst:
            # Симулируем разных локальных аналитиков
            candidates.append(AnalystCandidate(
                node_id=f"{self.node_id}_local_{len(candidates)}",
                trust_score=0.9,
                is_local=True,
            ))
        
        # Сортируем по Trust Score
        candidates.sort(key=lambda c: c.trust_score, reverse=True)
        
        return candidates[:count]
    
    def _get_trusted_peers(self, limit: int) -> List:
        """Получить список пиров."""
        if not self.node:
            return []
        
        peer_manager = getattr(self.node, 'peer_manager', None)
        if peer_manager:
            return peer_manager.get_active_peers()[:limit]
        
        return []
    
    # =========================================================================
    # STEP 2: PARALLEL ANALYSIS
    # =========================================================================
    
    async def _parallel_analysis(
        self,
        text: str,
        topic: str,
        analysts: List[AnalystCandidate],
    ) -> List[AnalystResponse]:
        """Выполнить параллельный анализ."""
        tasks = []
        
        for analyst in analysts:
            task = asyncio.create_task(
                self._analyze_with_timeout(text, topic, analyst)
            )
            tasks.append(task)
        
        # Ждём все задачи с общим таймаутом
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.ANALYST_TIMEOUT * 1.5,
            )
        except asyncio.TimeoutError:
            logger.warning("[CONSILIUM] Parallel analysis timeout")
            responses = []
            for task in tasks:
                if task.done():
                    try:
                        responses.append(task.result())
                    except Exception:
                        pass
                else:
                    task.cancel()
        
        # Обрабатываем результаты
        results = []
        for i, resp in enumerate(responses):
            if isinstance(resp, AnalystResponse):
                results.append(resp)
            elif isinstance(resp, Exception):
                results.append(AnalystResponse(
                    analyst_id=analysts[i].node_id if i < len(analysts) else "unknown",
                    analysis=None,
                    success=False,
                    error=str(resp),
                ))
        
        return results
    
    async def _analyze_with_timeout(
        self,
        text: str,
        topic: str,
        analyst: AnalystCandidate,
    ) -> AnalystResponse:
        """Анализ с таймаутом для одного аналитика."""
        start_time = time.time()
        
        try:
            if analyst.is_local and self.local_analyst:
                # Локальный анализ
                task = Task(
                    task_id=f"analysis_{analyst.node_id}_{int(time.time())}",
                    task_type="analysis",
                    payload={"text": text, "topic": topic},
                )
                
                result = await asyncio.wait_for(
                    self.local_analyst.execute(task),
                    timeout=self.ANALYST_TIMEOUT,
                )
                
                if result.success:
                    return AnalystResponse(
                        analyst_id=analyst.node_id,
                        analysis=result.data,
                        success=True,
                        execution_time=time.time() - start_time,
                        cost=result.cost,
                    )
                else:
                    return AnalystResponse(
                        analyst_id=analyst.node_id,
                        analysis=None,
                        success=False,
                        error=result.error,
                        execution_time=time.time() - start_time,
                    )
            else:
                # Удалённый анализ через P2P
                # TODO: Implement remote analysis via P2P messaging
                # Для MVP используем локальный анализ
                logger.warning(
                    f"[CONSILIUM] Remote analysis not implemented, "
                    f"using local for {analyst.node_id}"
                )
                
                if self.local_analyst:
                    task = Task(
                        task_id=f"analysis_{analyst.node_id}_{int(time.time())}",
                        task_type="analysis",
                        payload={"text": text, "topic": topic},
                    )
                    
                    result = await asyncio.wait_for(
                        self.local_analyst.execute(task),
                        timeout=self.ANALYST_TIMEOUT,
                    )
                    
                    if result.success:
                        return AnalystResponse(
                            analyst_id=analyst.node_id,
                            analysis=result.data,
                            success=True,
                            execution_time=time.time() - start_time,
                            cost=result.cost,
                        )
                
                return AnalystResponse(
                    analyst_id=analyst.node_id,
                    analysis=None,
                    success=False,
                    error="Remote analysis not implemented",
                    execution_time=time.time() - start_time,
                )
                
        except asyncio.TimeoutError:
            return AnalystResponse(
                analyst_id=analyst.node_id,
                analysis=None,
                success=False,
                error="Analysis timeout",
                execution_time=self.ANALYST_TIMEOUT,
            )
        except Exception as e:
            return AnalystResponse(
                analyst_id=analyst.node_id,
                analysis=None,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    # =========================================================================
    # STEP 3: JUDGE SYNTHESIS
    # =========================================================================
    
    async def _judge_synthesis(
        self,
        topic: str,
        responses: List[AnalystResponse],
        start_time: float,
    ) -> CouncilResult:
        """Синтезировать финальный ответ через Judge LLM."""
        
        if not self.judge:
            # Без Judge - просто объединяем ответы
            return self._simple_merge(topic, responses, start_time)
        
        # Подготавливаем отчёты для Judge
        reports = []
        for resp in responses:
            if resp.analysis:
                reports.append(json.dumps(resp.analysis, ensure_ascii=False, indent=2))
        
        if not reports:
            return self._create_failed_result(topic, "No valid analyses to judge", start_time)
        
        # Формируем промпт для Judge
        prompts = format_judge_prompt(topic, reports)
        
        try:
            # Запрос к Judge LLM
            result, cost = await self.judge.execute({
                "prompt": prompts["user"],
                "system": prompts["system"],
            })
            
            if isinstance(result, dict) and "response" in result:
                judgment = self._parse_judgment(result["response"])
                
                if judgment:
                    total_cost = sum(r.cost for r in responses) + (cost * self.judge.price_per_unit)
                    
                    return CouncilResult(
                        topic=topic,
                        status=CouncilStatus.COMPLETED,
                        final_summary=judgment.get("final_summary", ""),
                        consensus_facts=judgment.get("consensus_facts", []),
                        disputed_facts=judgment.get("disputed_facts", []),
                        filtered_hallucinations=judgment.get("filtered_hallucinations", []),
                        sentiment_consensus=judgment.get("sentiment_consensus", "neutral"),
                        confidence=judgment.get("confidence", 0.5),
                        analyst_agreement=judgment.get("analyst_agreement", 0.5),
                        reasoning=judgment.get("reasoning", ""),
                        participants=[r.analyst_id for r in responses],
                        total_cost=total_cost,
                        execution_time=time.time() - start_time,
                        raw_responses=responses,
                    )
            
            # Fallback to simple merge
            logger.warning("[CONSILIUM] Judge failed, using simple merge")
            return self._simple_merge(topic, responses, start_time)
            
        except Exception as e:
            logger.error(f"[CONSILIUM] Judge synthesis error: {e}")
            return self._simple_merge(topic, responses, start_time)
    
    def _parse_judgment(self, response: str) -> Optional[Dict[str, Any]]:
        """Парсить JSON ответ Judge."""
        try:
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"[CONSILIUM] Judge JSON parse error: {e}")
            return None
    
    def _simple_merge(
        self,
        topic: str,
        responses: List[AnalystResponse],
        start_time: float,
    ) -> CouncilResult:
        """Простое объединение ответов без Judge."""
        
        all_facts = []
        all_summaries = []
        sentiments = []
        confidences = []
        
        for resp in responses:
            if resp.analysis:
                all_facts.extend(resp.analysis.get("key_facts", []))
                all_summaries.append(resp.analysis.get("summary", ""))
                sentiments.append(resp.analysis.get("sentiment", "neutral"))
                confidences.append(resp.analysis.get("confidence", 0.5))
        
        # Дедупликация фактов (простая)
        unique_facts = list(set(all_facts))[:10]
        
        # Определяем консенсус по sentiment
        sentiment_counts = {}
        for s in sentiments:
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
        consensus_sentiment = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "neutral"
        
        # Средняя уверенность
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Первый summary как основной
        final_summary = all_summaries[0] if all_summaries else ""
        
        total_cost = sum(r.cost for r in responses)
        
        return CouncilResult(
            topic=topic,
            status=CouncilStatus.COMPLETED,
            final_summary=final_summary,
            consensus_facts=unique_facts,
            disputed_facts=[],
            filtered_hallucinations=[],
            sentiment_consensus=consensus_sentiment,
            confidence=avg_confidence,
            analyst_agreement=1.0 if len(set(sentiments)) == 1 else 0.5,
            reasoning="Simple merge without Judge LLM",
            participants=[r.analyst_id for r in responses],
            total_cost=total_cost,
            execution_time=time.time() - start_time,
            raw_responses=responses,
        )
    
    def _create_failed_result(
        self,
        topic: str,
        error: str,
        start_time: float,
    ) -> CouncilResult:
        """Создать результат с ошибкой."""
        return CouncilResult(
            topic=topic,
            status=CouncilStatus.FAILED,
            final_summary=f"Council failed: {error}",
            consensus_facts=[],
            disputed_facts=[],
            filtered_hallucinations=[],
            sentiment_consensus="neutral",
            confidence=0.0,
            analyst_agreement=0.0,
            reasoning=error,
            participants=[],
            total_cost=0.0,
            execution_time=time.time() - start_time,
        )
    
    # =========================================================================
    # STEP 4: PAY REWARDS
    # =========================================================================
    
    async def _pay_rewards(
        self,
        session: CouncilSession,
        responses: List[AnalystResponse],
        budget: float,
    ) -> None:
        """Выплатить награды участникам."""
        if not self.ledger:
            logger.warning("[CONSILIUM] No ledger for rewards")
            return
        
        if not responses:
            return
        
        # Делим бюджет поровну между успешными участниками
        reward_per_analyst = budget / len(responses)
        
        for resp in responses:
            if resp.analyst_id == self.node_id:
                # Не платим себе
                continue
            
            try:
                # Создаём IOU
                await self.ledger.create_iou(
                    debtor_id=self.node_id,
                    creditor_id=resp.analyst_id,
                    amount=reward_per_analyst,
                    description=f"Council reward: {session.topic}",
                )
                
                # Обновляем Trust Score за успешное участие
                await self.ledger.update_trust_score(
                    resp.analyst_id,
                    "council_participation",
                    0.5,  # magnitude
                )
                
                logger.debug(
                    f"[CONSILIUM] Paid {reward_per_analyst:.2f} to {resp.analyst_id[:16]}..."
                )
                
            except Exception as e:
                logger.warning(f"[CONSILIUM] Failed to pay {resp.analyst_id}: {e}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_session(self, session_id: str) -> Optional[CouncilSession]:
        """Получить сессию по ID."""
        return self._sessions.get(session_id)
    
    def get_active_sessions(self) -> List[CouncilSession]:
        """Получить активные сессии."""
        return [
            s for s in self._sessions.values()
            if s.status not in (CouncilStatus.COMPLETED, CouncilStatus.FAILED)
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            "total_councils": self._total_councils,
            "successful_councils": self._successful_councils,
            "success_rate": (
                self._successful_councils / self._total_councils
                if self._total_councils > 0 else 0.0
            ),
            "active_sessions": len(self.get_active_sessions()),
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def convene_council(
    topic: str,
    text: str,
    budget: float,
    node=None,
    ledger=None,
    judge_llm=None,
    local_analyst=None,
    node_id: str = "",
) -> CouncilResult:
    """
    Удобная функция для созыва совета.
    
    Args:
        topic: Тема анализа
        text: Текст для анализа
        budget: Бюджет
        node: P2P Node
        ledger: Ledger
        judge_llm: LLM для Judge
        local_analyst: Локальный Analyst
        node_id: ID узла
    
    Returns:
        CouncilResult
    """
    consilium = Consilium(
        node=node,
        ledger=ledger,
        judge_llm=judge_llm,
        local_analyst=local_analyst,
        node_id=node_id,
    )
    
    return await consilium.convene_council(topic, text, budget)

