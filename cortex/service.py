"""
Cortex Service - Главный сервис интеграции
==========================================

[SERVICE] CortexService объединяет все компоненты:
- Roles: Scout, Analyst, Librarian
- Library: Семантическая библиотека
- Consilium: Оркестратор консенсуса
- Automata: Автономный цикл

[INTEGRATION] Интегрируется с:
- core.node.Node - P2P сеть
- economy.ledger.Ledger - Транзакции
- agents.* - LLM агенты
- core.dht.* - DHT хранилище
"""

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.node import Node
    from economy.ledger import Ledger
    from agents.web_reader import ReaderAgent
    from agents.local_llm import OllamaAgent
    from agents.distributed_agent import DistributedLlmAgent

from .roles import Scout, Analyst, Librarian, Task, RoleResult
from .library import SemanticLibrary, Report, Bounty
from .consilium import Consilium, CouncilResult, CouncilStatus
from .automata import Automata, Investigation, InvestigationStatus
from .evolution.engine import EvolutionEngine, run_background

logger = logging.getLogger(__name__)


@dataclass
class CortexConfig:
    """Конфигурация Cortex."""
    enable_automata: bool = True
    automata_interval: float = 300.0  # 5 минут
    max_concurrent_investigations: int = 2
    default_bounty_reward: float = 10.0
    default_council_budget: float = 50.0
    evolution_interval: float = 5.0
    evolution_log_every: int = 10


class CortexService:
    """
    Главный сервис Cortex - Автономная система знаний.
    
    [USAGE]
    ```python
    cortex = CortexService(
        node=node,
        ledger=ledger,
        reader_agent=reader,
        llm_agent=ollama,
        kademlia=kademlia,
    )
    await cortex.start()
    
    # Исследовать тему
    result = await cortex.investigate("quantum computing")
    
    # Созвать совет
    council = await cortex.convene_council("AI safety", text, budget=100)
    
    # Поиск в библиотеке
    reports = await cortex.search("machine learning")
    ```
    
    [COMPONENTS]
    - scout: Разведчик (ReaderAgent)
    - analyst: Аналитик (OllamaAgent)
    - librarian: Библиотекарь (DHT)
    - library: Семантическая библиотека
    - consilium: Совет аналитиков
    - automata: Автономный мыслитель
    """
    
    def __init__(
        self,
        node: Optional["Node"] = None,
        ledger: Optional["Ledger"] = None,
        reader_agent: Optional["ReaderAgent"] = None,
        llm_agent: Optional["OllamaAgent"] = None,
        distributed_agent: Optional["DistributedLlmAgent"] = None,
        kademlia=None,
        dht_storage=None,
        config: Optional[CortexConfig] = None,
    ):
        """
        Args:
            node: P2P Node для сетевых операций
            ledger: Ledger для транзакций
            reader_agent: ReaderAgent для Scout
            llm_agent: OllamaAgent для Analyst и Judge
            distributed_agent: DistributedLlmAgent (опционально)
            kademlia: KademliaNode для DHT
            dht_storage: DHTStorage для локального хранения
            config: Конфигурация
        """
        self.node = node
        self.ledger = ledger
        self.config = config or CortexConfig()
        
        # Node ID
        self.node_id = ""
        if node:
            self.node_id = getattr(node, 'node_id', '')
        
        # Initialize roles
        self.scout = Scout(
            reader_agent=reader_agent,
            llm_agent=llm_agent,
        )
        
        self.analyst = Analyst(
            llm_agent=llm_agent,
            distributed_agent=distributed_agent,
        )
        
        self.librarian = Librarian(
            dht_storage=dht_storage,
            kademlia_node=kademlia,
            llm_agent=llm_agent,
        )
        
        # Initialize library
        self.library = SemanticLibrary(
            kademlia_node=kademlia,
            dht_storage=dht_storage,
            ledger=ledger,
            node_id=self.node_id,
        )
        
        # Initialize consilium
        self.consilium = Consilium(
            node=node,
            ledger=ledger,
            judge_llm=llm_agent,
            local_analyst=self.analyst,
            node_id=self.node_id,
        )
        
        # Initialize automata
        self.automata = Automata(
            scout=self.scout,
            analyst=self.analyst,
            librarian=self.librarian,
            library=self.library,
            consilium=self.consilium,
            node_id=self.node_id,
        )
        self.automata.THOUGHT_INTERVAL = self.config.automata_interval
        self.automata.MAX_CONCURRENT_INVESTIGATIONS = self.config.max_concurrent_investigations
        
        # State
        self._running = False
        self._evo_task: Optional[asyncio.Task] = None
        self.evolution_engine: Optional[EvolutionEngine] = None
        
        logger.info("[CORTEX] Service initialized")
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    async def start(self) -> None:
        """Запустить сервис Cortex."""
        if self._running:
            logger.warning("[CORTEX] Already running")
            return
        
        self._running = True
        
        # Запускаем автономный цикл если включён
        if self.config.enable_automata:
            await self.automata.start()
            logger.info("[CORTEX] Automata started")

        # Evolution engine background loop
        try:
            self.evolution_engine = EvolutionEngine()
            self.evolution_engine.initialize_population()
            self._evo_task = asyncio.create_task(
                run_background(
                    self.evolution_engine,
                    interval=self.config.evolution_interval,
                    log_every=self.config.evolution_log_every,
                )
            )
            logger.info("[CORTEX] Evolution engine started")
        except Exception as e:
            logger.warning(f"[CORTEX] Evolution engine failed to start: {e}")
        
        logger.info("[CORTEX] Service started")
    
    async def stop(self) -> None:
        """Остановить сервис Cortex."""
        self._running = False
        
        if self.automata.is_running:
            await self.automata.stop()

        if self._evo_task:
            self._evo_task.cancel()
            try:
                await self._evo_task
            except asyncio.CancelledError:
                pass
        
        logger.info("[CORTEX] Service stopped")
    
    # =========================================================================
    # INVESTIGATION API
    # =========================================================================
    
    async def investigate(
        self,
        topic: str,
        use_council: bool = False,
        budget: float = None,
    ) -> Dict[str, Any]:
        """
        Исследовать тему.
        
        Args:
            topic: Тема для исследования
            use_council: Использовать совет аналитиков
            budget: Бюджет (для совета)
        
        Returns:
            Результат исследования
        """
        logger.info(f"[CORTEX] Investigating: '{topic}'")
        
        if use_council:
            # Сначала собираем информацию
            scout_task = Task(
                task_id=f"scout_{hash(topic)}",
                task_type="scout",
                payload={"topic": topic, "max_sources": 5},
            )
            scout_result = await self.scout.execute(scout_task)
            
            if not scout_result.success:
                return {
                    "success": False,
                    "error": f"Scout failed: {scout_result.error}",
                }
            
            text = "\n\n".join(scout_result.data.get("raw_texts", []))
            
            # Созываем совет
            council_budget = budget or self.config.default_council_budget
            council_result = await self.consilium.convene_council(
                topic=topic,
                text=text,
                budget=council_budget,
            )
            
            # Сохраняем в библиотеку
            if council_result.status == CouncilStatus.COMPLETED:
                cid = await self.library.store(
                    topic=topic,
                    analysis={
                        "summary": council_result.final_summary,
                        "sentiment": council_result.sentiment_consensus,
                        "key_facts": council_result.consensus_facts,
                        "topics": [topic],
                        "confidence": council_result.confidence,
                    },
                    author_id=self.node_id,
                )
                
                return {
                    "success": True,
                    "topic": topic,
                    "method": "council",
                    "summary": council_result.final_summary,
                    "facts": council_result.consensus_facts,
                    "confidence": council_result.confidence,
                    "agreement": council_result.analyst_agreement,
                    "cid": cid,
                    "cost": council_result.total_cost,
                }
            else:
                return {
                    "success": False,
                    "error": council_result.reasoning,
                }
        
        else:
            # Простое расследование через Automata
            investigation = await self.automata.investigate(topic)
            
            if investigation and investigation.status == InvestigationStatus.COMPLETED:
                return {
                    "success": True,
                    "topic": topic,
                    "method": "automata",
                    "summary": investigation.analysis_result.get("summary", "") if investigation.analysis_result else "",
                    "facts": investigation.analysis_result.get("key_facts", []) if investigation.analysis_result else [],
                    "confidence": investigation.analysis_result.get("confidence", 0) if investigation.analysis_result else 0,
                    "cid": investigation.library_cid,
                }
            else:
                return {
                    "success": False,
                    "error": investigation.error if investigation else "Investigation failed",
                }
    
    # =========================================================================
    # COUNCIL API
    # =========================================================================
    
    async def convene_council(
        self,
        topic: str,
        text: str,
        budget: float = None,
    ) -> CouncilResult:
        """
        Созвать совет аналитиков.
        
        Args:
            topic: Тема анализа
            text: Текст для анализа
            budget: Бюджет
        
        Returns:
            CouncilResult
        """
        budget = budget or self.config.default_council_budget
        return await self.consilium.convene_council(topic, text, budget)
    
    # =========================================================================
    # LIBRARY API
    # =========================================================================
    
    async def search(self, topic: str, limit: int = 10) -> List[Report]:
        """
        Поиск в библиотеке.
        
        Args:
            topic: Тема для поиска
            limit: Максимум результатов
        
        Returns:
            Список отчётов
        """
        return await self.library.search(topic, limit)
    
    async def search_or_investigate(
        self,
        topic: str,
        auto_investigate: bool = True,
    ) -> Dict[str, Any]:
        """
        Поиск с автоматическим исследованием если не найдено.
        
        Args:
            topic: Тема
            auto_investigate: Автоматически исследовать если не найдено
        
        Returns:
            Результат поиска или исследования
        """
        reports = await self.search(topic)
        
        if reports:
            return {
                "found": True,
                "reports": [r.to_dict() for r in reports],
                "count": len(reports),
            }
        
        if auto_investigate:
            result = await self.investigate(topic)
            return {
                "found": False,
                "investigated": True,
                "result": result,
            }
        
        return {
            "found": False,
            "investigated": False,
        }
    
    async def get_report(self, cid: str) -> Optional[Report]:
        """Получить отчёт по CID."""
        return await self.library.get_report(cid)
    
    # =========================================================================
    # BOUNTY API
    # =========================================================================
    
    async def create_bounty(
        self,
        topic: str,
        reward: float = None,
    ) -> Optional[Bounty]:
        """
        Создать bounty на исследование темы.
        
        Args:
            topic: Тема
            reward: Награда
        
        Returns:
            Bounty или None
        """
        reward = reward or self.config.default_bounty_reward
        return await self.library.create_bounty(topic, reward)
    
    async def get_open_bounties(self, limit: int = 50) -> List[Bounty]:
        """Получить открытые bounties."""
        return await self.library.get_open_bounties(limit)
    
    async def claim_bounty(
        self,
        bounty_id: str,
        report_cid: str,
    ) -> bool:
        """
        Выполнить bounty.
        
        Args:
            bounty_id: ID bounty
            report_cid: CID отчёта
        
        Returns:
            Успешно ли
        """
        return await self.library.claim_bounty(bounty_id, report_cid, self.node_id)
    
    # =========================================================================
    # STATUS & STATS
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус сервиса."""
        return {
            "running": self._running,
            "automata_running": self.automata.is_running,
            "node_id": self.node_id[:16] + "..." if self.node_id else "",
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            "service": self.get_status(),
            "automata": self.automata.get_stats(),
            "consilium": self.consilium.get_stats(),
            "library": asyncio.get_event_loop().run_until_complete(
                self.library.get_stats()
            ) if asyncio.get_event_loop().is_running() else {},
        }
    
    async def get_stats_async(self) -> Dict[str, Any]:
        """Получить статистику (async версия)."""
        return {
            "service": self.get_status(),
            "automata": self.automata.get_stats(),
            "consilium": self.consilium.get_stats(),
            "library": await self.library.get_stats(),
        }
    
    # =========================================================================
    # RECENT ACTIVITY
    # =========================================================================
    
    def get_recent_investigations(self, limit: int = 10) -> List[Investigation]:
        """Получить последние расследования."""
        return self.automata.get_recent_investigations(limit)
    
    def get_active_investigations(self) -> List[Investigation]:
        """Получить активные расследования."""
        return self.automata.get_active_investigations()
    
    def get_active_councils(self):
        """Получить активные советы."""
        return self.consilium.get_active_sessions()


# =============================================================================
# FACTORY
# =============================================================================

def create_cortex_service(
    node=None,
    ledger=None,
    agent_manager=None,
    kademlia=None,
    dht_storage=None,
    enable_automata: bool = True,
) -> CortexService:
    """
    Factory для создания CortexService.
    
    Args:
        node: P2P Node
        ledger: Ledger
        agent_manager: AgentManager с агентами
        kademlia: KademliaNode
        dht_storage: DHTStorage
        enable_automata: Включить автономный цикл
    
    Returns:
        CortexService
    """
    # Извлекаем агентов из manager
    reader_agent = None
    llm_agent = None
    distributed_agent = None
    
    if agent_manager:
        agents = getattr(agent_manager, '_agents', {})
        
        # ReaderAgent
        reader_agent = agents.get('web_read')
        
        # OllamaAgent
        llm_agent = agents.get('llm_local')
        
        # DistributedLlmAgent
        distributed_agent = agents.get('llm_distributed')
    
    config = CortexConfig(enable_automata=enable_automata)
    
    return CortexService(
        node=node,
        ledger=ledger,
        reader_agent=reader_agent,
        llm_agent=llm_agent,
        distributed_agent=distributed_agent,
        kademlia=kademlia,
        dht_storage=dht_storage,
        config=config,
    )
