"""
P2P Node Web UI - Главное приложение
====================================

[ARCHITECTURE]
- NiceGUI для UI
- Async интеграция с Node
- WebSocket для real-time updates
- Modular pages
- 3D Neural Visualization (Three.js)
"""

import asyncio
import logging
import uuid
from collections import deque
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from logging import Handler
from cortex.archivist import AsyncFileScanner, DocumentProcessor, VectorStore
from core.events import event_bus
from webui.components.downloader import DownloadManager
from webui.tabs.library import LibraryTab
from webui.tabs.ingest import IngestTab
from webui.tabs.activity import ActivityTab
from webui.components.gallery import Gallery

logger = logging.getLogger(__name__)

# Попытка импорта NiceGUI
try:
    from nicegui import ui, app, Client
    NICEGUI_AVAILABLE = True
except ImportError:
    NICEGUI_AVAILABLE = False
    logger.warning("[WEBUI] NiceGUI not installed. Run: pip install nicegui")

# Path to static files
STATIC_DIR = Path(__file__).parent / "static"


class GUILogHandler(Handler):
    """Logging handler that stores formatted records in a deque for UI display."""
    def __init__(self, buffer: deque):
        super().__init__()
        self.buffer = buffer
    
    def emit(self, record) -> None:
        try:
            msg = self.format(record)
            self.buffer.append(msg)
        except Exception:
            self.handleError(record)


@dataclass
class NodeState:
    """Состояние узла для отображения в UI."""
    node_id: str = ""
    host: str = "0.0.0.0"
    port: int = 8468
    is_running: bool = False
    peers_count: int = 0
    services: list = field(default_factory=list)
    dht_available: bool = False
    
    # Статистика
    total_sent: int = 0
    total_received: int = 0
    uptime_seconds: float = 0
    
    # Последнее обновление
    last_update: float = 0


class P2PWebUI:
    """
    Web UI для P2P узла.
    
    [USAGE]
    ```python
    webui = P2PWebUI(node, ledger, agent_manager)
    await webui.start(port=8080)
    ```
    """
    
    def __init__(
        self,
        node=None,
        ledger=None,
        agent_manager=None,
        kademlia=None,
        cortex=None,
        visualizer=None,
        idle_worker=None,
        title: str = "P2P Node",
        dark_mode: bool = True,
    ):
        """
        Args:
            node: P2P Node instance
            ledger: Ledger instance
            agent_manager: AgentManager instance
            kademlia: KademliaNode instance
            cortex: CortexService instance
            visualizer: CortexVisualizer instance
            title: Заголовок страницы
            dark_mode: Тёмная тема
        """
        self.node = node
        self.ledger = ledger
        self.agent_manager = agent_manager
        self.kademlia = kademlia
        self.cortex = cortex
        self.visualizer = visualizer
        self.idle_worker = idle_worker
        self.title = title
        self.dark_mode = dark_mode
        
        self._state = NodeState()
        self._update_interval = 2.0  # секунды
        self._running = False
        self._log_buffer = deque(maxlen=500)
        self._log_handler: Optional[logging.Handler] = None
        self._dream_mode_enabled = False
        self._version_text = "unknown"
        self.downloader = DownloadManager()
        self.library_tab = LibraryTab(Path("ledger.db"), self)
        self.gallery = Gallery()
        self.ingest_tab = IngestTab(self.gallery)
        self.activity_tab = ActivityTab(ui_log_buffer if 'ui_log_buffer' in globals() else deque(maxlen=1000))
        
        # Callbacks для обновления UI
        self._update_callbacks: list = []
        
        # Cortex UI state
        self._use_council = False
        
        # Hook logs into UI
        self._attach_log_handler()
    
    def _attach_log_handler(self) -> None:
        """Attach log handler to feed UI log panel."""
        try:
            handler = GUILogHandler(self._log_buffer)
            handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(handler)
            self._log_handler = handler
            logger.info("[WEBUI] GUI log handler attached")
        except Exception as e:
            logger.warning(f"[WEBUI] Failed to attach GUI log handler: {e}")
        
        # Subscribe to download_progress events
        async def on_progress(payload):
            self.downloader.update(payload)
        try:
            import asyncio
            asyncio.create_task(event_bus.subscribe("download_progress", on_progress))
        except Exception:
            pass
    
    def _update_state(self) -> None:
        """Обновить состояние из node."""
        if not self.node:
            return
        
        import time
        
        self._state.node_id = getattr(self.node, 'node_id', '')[:32]
        self._state.host = getattr(self.node, 'host', '0.0.0.0')
        self._state.port = getattr(self.node, 'port', 8468)
        self._state.is_running = getattr(self.node, '_running', False)
        
        # Peers
        peer_manager = getattr(self.node, 'peer_manager', None)
        if peer_manager:
            peers = peer_manager.get_active_peers()
            self._state.peers_count = len(peers) if peers else 0
        
        # Services
        if self.agent_manager:
            self._state.services = list(self.agent_manager._agents.keys())
        
        # DHT
        self._state.dht_available = self.kademlia is not None
        
        self._state.last_update = time.time()
    
    async def _create_header(self) -> None:
        """Создать шапку страницы."""
        # Mount downloader UI lazily inside a page context to avoid global-scope UI
        self.downloader.mount()
        with ui.header().classes('items-center justify-between'):
            ui.label(self.title).classes('text-xl font-bold')
            
            with ui.row().classes('items-center gap-4'):
                # Статус
                self._status_badge = ui.badge('Offline', color='red')
                
                # Переключатель темы
                dark = ui.dark_mode(value=self.dark_mode)
                ui.switch('Dark', on_change=lambda e: dark.set_value(e.value))
    
    async def _create_sidebar(self) -> None:
        """Создать боковое меню."""
        with ui.left_drawer(value=True).classes('bg-gray-100 dark:bg-gray-800'):
            ui.label('Navigation').classes('text-lg font-bold p-4')
            
            with ui.column().classes('w-full'):
                ui.button('Dashboard', icon='dashboard', on_click=lambda: ui.navigate.to('/')).classes('w-full justify-start')
                ui.button('Peers', icon='people', on_click=lambda: ui.navigate.to('/peers')).classes('w-full justify-start')
                ui.button('Services', icon='build', on_click=lambda: ui.navigate.to('/services')).classes('w-full justify-start')
                ui.button('AI / LLM', icon='psychology', on_click=lambda: ui.navigate.to('/ai')).classes('w-full justify-start')
                ui.button('Cortex', icon='hub', on_click=lambda: ui.navigate.to('/cortex')).classes('w-full justify-start')
                ui.button('DHT', icon='storage', on_click=lambda: ui.navigate.to('/dht')).classes('w-full justify-start')
                ui.button('Economy', icon='account_balance', on_click=lambda: ui.navigate.to('/economy')).classes('w-full justify-start')
                ui.button('Storage', icon='folder', on_click=lambda: ui.navigate.to('/storage')).classes('w-full justify-start')
                ui.button('Compute', icon='memory', on_click=lambda: ui.navigate.to('/compute')).classes('w-full justify-start')
                ui.button('Ingest', icon='cloud_upload', on_click=lambda: ui.navigate.to('/ingest')).classes('w-full justify-start')
                
                ui.separator()
                
                ui.button('Settings', icon='settings', on_click=lambda: ui.navigate.to('/settings')).classes('w-full justify-start')
                ui.button('Logs', icon='article', on_click=lambda: ui.navigate.to('/logs')).classes('w-full justify-start')
    
    def _create_dashboard_page(self) -> None:
        """Dashboard страница."""
        
        @ui.page('/')
        async def dashboard():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Dashboard').classes('text-2xl font-bold mb-4')
                
                # Карточки статистики
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    # Node Info
                    with ui.card().classes('w-64'):
                        ui.label('Node').classes('text-lg font-bold')
                        self._node_id_label = ui.label(f'ID: {self._state.node_id}...').classes('text-sm font-mono')
                        self._node_addr_label = ui.label(f'{self._state.host}:{self._state.port}')
                        self._node_status = ui.badge('Checking...', color='yellow')
                    
                    # Peers
                    with ui.card().classes('w-64'):
                        ui.label('Peers').classes('text-lg font-bold')
                        self._peers_count_label = ui.label('0').classes('text-4xl font-bold text-blue-500')
                        ui.label('connected').classes('text-sm text-gray-500')
                    
                    # Services
                    with ui.card().classes('w-64'):
                        ui.label('Services').classes('text-lg font-bold')
                        self._services_count_label = ui.label('0').classes('text-4xl font-bold text-green-500')
                        ui.label('available').classes('text-sm text-gray-500')
                    
                    # DHT
                    with ui.card().classes('w-64'):
                        ui.label('DHT').classes('text-lg font-bold')
                        self._dht_status = ui.badge('Checking...', color='yellow')
                        self._dht_keys_label = ui.label('Keys: -')
                
                # Графики (placeholder)
                with ui.row().classes('w-full gap-4 mt-4'):
                    with ui.card().classes('flex-1'):
                        ui.label('Network Activity').classes('text-lg font-bold')
                        ui.label('Real-time charts coming soon...').classes('text-gray-500')
                
                # Быстрые действия
                ui.label('Quick Actions').classes('text-xl font-bold mt-6 mb-2')
                with ui.row().classes('gap-2'):
                    ui.button('Ping All Peers', icon='wifi', on_click=self._ping_all_peers)
                    ui.button('Refresh DHT', icon='refresh', on_click=self._refresh_dht)
                    ui.button('View Logs', icon='article', on_click=lambda: ui.navigate.to('/logs'))
                
                # Auto-refresh
                ui.timer(self._update_interval, self._refresh_dashboard)
    
    async def _refresh_dashboard(self) -> None:
        """Обновить данные на dashboard."""
        self._update_state()
        
        if hasattr(self, '_node_id_label'):
            self._node_id_label.text = f'ID: {self._state.node_id}...'
        
        if hasattr(self, '_node_addr_label'):
            self._node_addr_label.text = f'{self._state.host}:{self._state.port}'
        
        if hasattr(self, '_node_status'):
            if self._state.is_running:
                self._node_status.set_text('Online')
                self._node_status._props['color'] = 'green'
            else:
                self._node_status.set_text('Offline')
                self._node_status._props['color'] = 'red'
            self._node_status.update()
        
        if hasattr(self, '_peers_count_label'):
            self._peers_count_label.text = str(self._state.peers_count)
        
        if hasattr(self, '_services_count_label'):
            self._services_count_label.text = str(len(self._state.services))
        
        if hasattr(self, '_dht_status'):
            if self._state.dht_available:
                self._dht_status.set_text('Active')
                self._dht_status._props['color'] = 'green'
            else:
                self._dht_status.set_text('Inactive')
                self._dht_status._props['color'] = 'gray'
            self._dht_status.update()
    
    async def _ping_all_peers(self) -> None:
        """Отправить ping всем пирам."""
        ui.notify('Pinging all peers...', type='info')
        # TODO: Implement
    
    async def _refresh_dht(self) -> None:
        """Обновить DHT."""
        if not self.kademlia:
            ui.notify('DHT not available', type='warning')
            return
        try:
            stats = await self.kademlia.get_full_stats()
            routing = stats.get('routing_table', {})
            stored = stats.get('storage', {})
            if hasattr(self, '_dht_node_id'):
                self._dht_node_id.text = stats.get('local_id', '')[:32]
            if hasattr(self, '_dht_routing'):
                self._dht_routing.text = f"Buckets: {routing.get('bucket_count', '-')}, Peers: {routing.get('total_nodes', '-')}"
            if hasattr(self, '_dht_stored_keys'):
                self._dht_stored_keys.text = str(stored.get('total_entries', stored.get('active_entries', '-')))
            ui.notify('DHT refreshed', type='positive')
        except Exception as e:
            logger.error(f"[WEBUI] DHT refresh failed: {e}")
            ui.notify(f'DHT refresh error: {e}', type='negative')
    
    def _create_peers_page(self) -> None:
        """Страница Peers."""
        
        @ui.page('/peers')
        async def peers():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Peers').classes('text-2xl font-bold mb-4')
                
                # Таблица пиров
                columns = [
                    {'name': 'node_id', 'label': 'Node ID', 'field': 'node_id', 'align': 'left'},
                    {'name': 'address', 'label': 'Address', 'field': 'address'},
                    {'name': 'direction', 'label': 'Direction', 'field': 'direction'},
                    {'name': 'balance', 'label': 'Balance', 'field': 'balance'},
                    {'name': 'status', 'label': 'Status', 'field': 'status'},
                    {'name': 'actions', 'label': 'Actions', 'field': 'actions'},
                ]
                
                self._peers_table = ui.table(
                    columns=columns,
                    rows=[],
                    row_key='node_id',
                ).classes('w-full')
                
                with ui.row().classes('mt-4 gap-2'):
                    ui.button('Refresh', icon='refresh', on_click=self._refresh_peers)
                    
                    with ui.input('Bootstrap node').classes('w-64') as bootstrap_input:
                        pass
                    ui.button('Connect', icon='add', on_click=lambda: self._connect_to_peer(bootstrap_input.value))
                
                ui.timer(self._update_interval, self._refresh_peers)
    
    async def _refresh_peers(self) -> None:
        """Обновить список пиров."""
        if not self.node:
            return
        
        rows = []
        peer_manager = getattr(self.node, 'peer_manager', None)
        if peer_manager:
            for peer in peer_manager.get_active_peers():
                balance = 0
                if self.ledger:
                    balance = await self.ledger.get_balance(peer.node_id)
                
                rows.append({
                    'node_id': peer.node_id[:16] + '...',
                    'address': f'{peer.host}:{peer.port}',
                    'direction': 'OUT' if peer.is_outbound else 'IN',
                    'balance': f'{balance:+.0f}',
                    'status': 'BLOCKED' if peer.blocked else 'Active',
                })
        
        if hasattr(self, '_peers_table'):
            self._peers_table.rows = rows
            self._peers_table.update()
    
    async def _connect_to_peer(self, address: str) -> None:
        """Подключиться к пиру."""
        if not address:
            ui.notify('Enter address', type='warning')
            return
        
        ui.notify(f'Connecting to {address}...', type='info')
        try:
            host, port_str = address.rsplit(':', 1)
            port = int(port_str)
        except Exception:
            ui.notify('Address must be host:port', type='warning')
            return
        
        if not self.node:
            ui.notify('Node not available', type='warning')
            return
        
        peer = await self.node.connect_to_peer(host, port)
        if peer:
            ui.notify(f'Connected to {host}:{port}', type='positive')
            await self._refresh_peers()
        else:
            ui.notify(f'Failed to connect {host}:{port}', type='negative')
    
    def _create_services_page(self) -> None:
        """Страница Services."""
        
        @ui.page('/services')
        async def services():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Services').classes('text-2xl font-bold mb-4')
                
                if not self.agent_manager:
                    ui.label('Agent Manager not available').classes('text-red-500')
                    return
                
                # Список услуг
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    for name, agent in self.agent_manager._agents.items():
                        with ui.card().classes('w-72'):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('build').classes('text-2xl')
                                ui.label(name).classes('text-lg font-bold')
                            
                            ui.label(getattr(agent, 'description', 'No description')).classes('text-sm text-gray-500')
                            ui.label(f'Price: {agent.price_per_unit} per unit').classes('text-sm')
                            
                            with ui.row().classes('mt-2'):
                                ui.button('Test', icon='play_arrow', on_click=lambda n=name: self._test_service(n)).props('size=sm')
                
                # Запрос услуги
                ui.label('Request Service').classes('text-xl font-bold mt-6 mb-2')
                with ui.card().classes('w-full max-w-2xl'):
                    with ui.row().classes('w-full gap-4'):
                        service_select = ui.select(
                            list(self.agent_manager._agents.keys()),
                            label='Service',
                            value='echo'
                        ).classes('w-48')
                        
                        peer_input = ui.input('Peer ID (optional)').classes('w-64')
                    
                    payload_input = ui.textarea('Payload (JSON)').classes('w-full')
                    payload_input.value = '{"message": "Hello"}'
                    
                    budget_input = ui.number('Budget', value=10)
                    
                    with ui.row().classes('mt-2'):
                        ui.button('Send Request', icon='send', on_click=lambda: self._send_service_request(
                            service_select.value,
                            peer_input.value,
                            payload_input.value,
                            budget_input.value
                        ))
                    
                    self._service_result = ui.label('').classes('mt-2 font-mono text-sm')
    
    async def _test_service(self, service_name: str) -> None:
        """Тестировать услугу локально."""
        if not self.agent_manager:
            return
        
        agent = self.agent_manager._agents.get(service_name)
        if not agent:
            ui.notify(f'Service {service_name} not found', type='error')
            return
        
        ui.notify(f'Testing {service_name}...', type='info')
        
        # Тестовый payload
        test_payloads = {
            'echo': {'message': 'Test from WebUI'},
            'storage': {'action': 'list', 'owner_id': 'webui'},
            'compute': {'task': 'eval', 'expression': '2+2'},
            'web_read': {'url': 'https://example.com'},
            'llm_local': {'prompt': 'Say hello'},
        }
        
        payload = test_payloads.get(service_name, {})
        
        try:
            result, cost = await agent.execute(payload)
            ui.notify(f'Result: {result}, Cost: {cost}', type='positive')
        except Exception as e:
            ui.notify(f'Error: {e}', type='negative')
    
    async def _send_service_request(self, service: str, peer_id: str, payload_json: str, budget: float) -> None:
        """Отправить запрос услуги."""
        import json
        
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as e:
            ui.notify(f'Invalid JSON: {e}', type='error')
            return
        
        if peer_id:
            # Remote request
            ui.notify(f'Sending {service} request to {peer_id}...', type='info')
            # TODO: Implement remote request
        else:
            # Local request
            if self.agent_manager:
                agent = self.agent_manager._agents.get(service)
                if agent:
                    try:
                        result, cost = await agent.execute(payload)
                        if hasattr(self, '_service_result'):
                            self._service_result.text = f'Result: {result}\nCost: {cost}'
                        ui.notify('Success!', type='positive')
                    except Exception as e:
                        ui.notify(f'Error: {e}', type='negative')
    
    def _create_ai_page(self) -> None:
        """Страница AI / LLM."""
        
        @ui.page('/ai')
        async def ai():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('AI / LLM').classes('text-2xl font-bold mb-4')
                
                # Chat interface
                with ui.card().classes('w-full max-w-4xl'):
                    ui.label('Chat with AI').classes('text-lg font-bold')
                    
                    # Chat history
                    self._chat_container = ui.column().classes('w-full h-96 overflow-auto bg-gray-50 dark:bg-gray-900 p-4 rounded')
                    
                    # Input
                    with ui.row().classes('w-full mt-4 gap-2'):
                        self._chat_input = ui.textarea('Your message...').classes('flex-grow')
                        
                        with ui.column().classes('gap-1'):
                            ui.button('Send', icon='send', on_click=self._send_chat_message).classes('w-full')
                            ui.button('Clear', icon='delete', on_click=self._clear_chat).classes('w-full')
                    
                    # Options
                    with ui.row().classes('mt-2 gap-4'):
                        self._model_select = ui.select(
                            ['llm_local (Ollama)', 'llm_prompt (Cloud)', 'llm_distributed'],
                            label='Model',
                            value='llm_local (Ollama)'
                        )
                        self._temp_slider = ui.slider(min=0, max=2, value=0.7, step=0.1).props('label')
                        ui.label('Temperature')
                
                # Distributed Inference Status
                ui.label('Distributed Inference').classes('text-xl font-bold mt-6 mb-2')
                with ui.card().classes('w-full max-w-4xl'):
                    with ui.row().classes('gap-4'):
                        with ui.column():
                            ui.label('Status').classes('font-bold')
                            self._dist_status = ui.badge('Checking...', color='yellow')
                        
                        with ui.column():
                            ui.label('Loaded Shards').classes('font-bold')
                            self._dist_shards = ui.label('-')
                        
                        with ui.column():
                            ui.label('Available Models').classes('font-bold')
                            self._dist_models = ui.label('-')
                    
                    with ui.row().classes('mt-4 gap-2'):
                        ui.button('Load Shard', icon='download', on_click=self._show_load_shard_dialog)
                        ui.button('Check Models', icon='search', on_click=self._check_dist_models)
    
    async def _send_chat_message(self) -> None:
        """Отправить сообщение в чат."""
        if not hasattr(self, '_chat_input'):
            return
        
        message = self._chat_input.value
        if not message.strip():
            return
        
        # Добавляем сообщение пользователя
        with self._chat_container:
            with ui.row().classes('w-full justify-end'):
                ui.label(message).classes('bg-blue-500 text-white p-2 rounded-lg max-w-md')
        
        self._chat_input.value = ''
        
        # Получаем ответ
        model_type = 'llm_local'
        if 'Cloud' in self._model_select.value:
            model_type = 'llm_prompt'
        elif 'distributed' in self._model_select.value:
            model_type = 'llm_distributed'
        
        if self.agent_manager:
            agent = self.agent_manager._agents.get(model_type)
            if agent:
                with self._chat_container:
                    loading = ui.label('Thinking...').classes('text-gray-500 italic')
                
                try:
                    result, cost = await agent.execute({
                        'prompt': message,
                        'temperature': self._temp_slider.value,
                    })
                    
                    loading.delete()
                    
                    response = result.get('response', str(result))
                    with self._chat_container:
                        with ui.row().classes('w-full'):
                            ui.label(response).classes('bg-gray-200 dark:bg-gray-700 p-2 rounded-lg max-w-md')
                    
                except Exception as e:
                    loading.text = f'Error: {e}'
    
    async def _clear_chat(self) -> None:
        """Очистить чат."""
        if hasattr(self, '_chat_container'):
            self._chat_container.clear()
    
    async def _show_load_shard_dialog(self) -> None:
        """Диалог загрузки shard."""
        with ui.dialog() as dialog, ui.card():
            ui.label('Load Model Shard').classes('text-lg font-bold')
            
            model_input = ui.select(
                ['qwen2.5-32b', 'llama2-70b', 'mixtral-8x7b'],
                label='Model',
                value='qwen2.5-32b'
            )
            layer_start = ui.number('Layer Start', value=0)
            layer_end = ui.number('Layer End', value=16)
            use_mock = ui.checkbox('Use Mock (no GPU)')
            
            with ui.row().classes('mt-4 gap-2'):
                ui.button('Load', on_click=lambda: self._load_shard(
                    dialog, model_input.value, int(layer_start.value), int(layer_end.value), use_mock.value
                ))
                ui.button('Cancel', on_click=dialog.close)
        
        dialog.open()
    
    async def _load_shard(self, dialog, model: str, start: int, end: int, use_mock: bool) -> None:
        """Загрузить shard."""
        ui.notify(f'Loading {model} layers {start}-{end}...', type='info')
        dialog.close()
        # TODO: Implement
    
    async def _check_dist_models(self) -> None:
        """Проверить доступные distributed модели."""
        try:
            from agents.distributed.registry import KNOWN_MODELS
            models = list(KNOWN_MODELS.keys())
            if hasattr(self, '_dist_models'):
                self._dist_models.text = ', '.join(models)
            ui.notify(f'Found {len(models)} models', type='info')
        except ImportError:
            ui.notify('Distributed module not available', type='warning')
    
    def _create_dht_page(self) -> None:
        """Страница DHT."""
        
        @ui.page('/dht')
        async def dht():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Distributed Hash Table').classes('text-2xl font-bold mb-4')
                
                if not self.kademlia:
                    ui.label('DHT not available').classes('text-red-500')
                    return
                
                # DHT Info
                with ui.card().classes('w-full max-w-2xl'):
                    ui.label('DHT Status').classes('text-lg font-bold')
                    with ui.row().classes('gap-8'):
                        with ui.column():
                            ui.label('Node ID').classes('font-bold')
                            self._dht_node_id = ui.label('-').classes('font-mono text-sm')
                        with ui.column():
                            ui.label('Stored Keys').classes('font-bold')
                            self._dht_stored_keys = ui.label('-')
                        with ui.column():
                            ui.label('Routing Table').classes('font-bold')
                            self._dht_routing = ui.label('-')
                
                # Put/Get
                ui.label('Operations').classes('text-xl font-bold mt-6 mb-2')
                
                with ui.tabs().classes('w-full') as tabs:
                    put_tab = ui.tab('Put')
                    get_tab = ui.tab('Get')
                    delete_tab = ui.tab('Delete')
                
                with ui.tab_panels(tabs, value=put_tab).classes('w-full'):
                    with ui.tab_panel(put_tab):
                        with ui.card().classes('w-full'):
                            key_put = ui.input('Key').classes('w-full')
                            value_put = ui.textarea('Value').classes('w-full')
                            ui.button('Store', icon='save', on_click=lambda: self._dht_put(key_put.value, value_put.value))
                    
                    with ui.tab_panel(get_tab):
                        with ui.card().classes('w-full'):
                            key_get = ui.input('Key').classes('w-full')
                            ui.button('Retrieve', icon='search', on_click=lambda: self._dht_get(key_get.value))
                            self._dht_get_result = ui.label('').classes('mt-2 font-mono')
                    
                    with ui.tab_panel(delete_tab):
                        with ui.card().classes('w-full'):
                            key_del = ui.input('Key').classes('w-full')
                            ui.button('Delete', icon='delete', color='red', on_click=lambda: self._dht_delete(key_del.value))
    
    async def _dht_put(self, key: str, value: str) -> None:
        """Сохранить в DHT."""
        if not key or not value:
            ui.notify('Key and value required', type='warning')
            return
        
        if self.kademlia:
            try:
                await self.kademlia.dht_put(key, value.encode())
                ui.notify(f'Stored: {key}', type='positive')
            except Exception as e:
                ui.notify(f'Error: {e}', type='negative')
    
    async def _dht_get(self, key: str) -> None:
        """Получить из DHT."""
        if not key:
            ui.notify('Key required', type='warning')
            return
        
        if self.kademlia:
            try:
                value = await self.kademlia.dht_get(key)
                if value:
                    if hasattr(self, '_dht_get_result'):
                        self._dht_get_result.text = f'Value: {value.decode()}'
                    ui.notify('Found!', type='positive')
                else:
                    if hasattr(self, '_dht_get_result'):
                        self._dht_get_result.text = 'Not found'
                    ui.notify('Key not found', type='warning')
            except Exception as e:
                ui.notify(f'Error: {e}', type='negative')
    
    async def _dht_delete(self, key: str) -> None:
        """Удалить из DHT."""
        if not key:
            ui.notify('Key required', type='warning')
            return
        
        if self.kademlia:
            try:
                deleted = await self.kademlia.dht_delete(key)
                if deleted:
                    ui.notify(f'Deleted: {key}', type='positive')
                else:
                    ui.notify('Key not found', type='warning')
            except Exception as e:
                ui.notify(f'Error: {e}', type='negative')
    
    def _create_economy_page(self) -> None:
        """Страница Economy."""
        
        @ui.page('/economy')
        async def economy():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Economy').classes('text-2xl font-bold mb-4')
                
                if not self.ledger:
                    ui.label('Ledger not available').classes('text-red-500')
                    return
                
                # Summary
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    with ui.card().classes('w-64'):
                        ui.label('Total Owed to Us').classes('font-bold')
                        self._total_claims = ui.label('0').classes('text-3xl font-bold text-green-500')
                    
                    with ui.card().classes('w-64'):
                        ui.label('Total We Owe').classes('font-bold')
                        self._total_debts = ui.label('0').classes('text-3xl font-bold text-red-500')
                    
                    with ui.card().classes('w-64'):
                        ui.label('Net Balance').classes('font-bold')
                        self._net_balance = ui.label('0').classes('text-3xl font-bold')
                
                # Balances table
                ui.label('Balances by Peer').classes('text-xl font-bold mt-6 mb-2')
                
                columns = [
                    {'name': 'peer_id', 'label': 'Peer ID', 'field': 'peer_id', 'align': 'left'},
                    {'name': 'balance', 'label': 'Balance', 'field': 'balance'},
                    {'name': 'total_sent', 'label': 'Sent', 'field': 'total_sent'},
                    {'name': 'total_received', 'label': 'Received', 'field': 'total_received'},
                    {'name': 'status', 'label': 'Status', 'field': 'status'},
                ]
                
                self._balances_table = ui.table(columns=columns, rows=[]).classes('w-full')
                
                ui.button('Refresh', icon='refresh', on_click=self._refresh_economy)
                ui.timer(5, self._refresh_economy)
    
    async def _refresh_economy(self) -> None:
        """Обновить данные экономики."""
        if not self.ledger:
            return
        
        try:
            balances_list = await self.ledger.get_all_balances()
            
            total_claims = 0.0
            total_debts = 0.0
            rows = []
            
            for entry in balances_list:
                peer_id = entry.get("peer_id", "") if isinstance(entry, dict) else entry[0]
                balance = entry.get("balance", 0.0) if isinstance(entry, dict) else entry[1]
                sent = entry.get("total_sent", 0.0) if isinstance(entry, dict) else entry[2]
                received = entry.get("total_received", 0.0) if isinstance(entry, dict) else entry[3]
                
                if balance >= 0:
                    total_claims += balance
                else:
                    total_debts += abs(balance)
                
                rows.append({
                    'peer_id': peer_id[:16] + '...',
                    'balance': f"{balance:+.0f}",
                    'total_sent': f"{sent:.0f}",
                    'total_received': f"{received:.0f}",
                    'status': 'Blocked' if balance > getattr(self.ledger, 'debt_limit', 0) else 'OK',
                })
            
            net = total_claims - total_debts
            
            if hasattr(self, '_total_claims'):
                self._total_claims.text = f"{total_claims:.0f}"
            
            if hasattr(self, '_total_debts'):
                self._total_debts.text = f"{total_debts:.0f}"
            
            if hasattr(self, '_net_balance'):
                self._net_balance.text = f"{net:+.0f}"
                if net >= 0:
                    self._net_balance.classes(replace='text-3xl font-bold text-green-500')
                else:
                    self._net_balance.classes(replace='text-3xl font-bold text-red-500')
            
            # Balances table
            if hasattr(self, '_balances_table'):
                self._balances_table.rows = rows
                self._balances_table.update()
                
        except Exception as e:
            logger.error(f"[WEBUI] Economy refresh error: {e}")
    
    def _create_storage_page(self) -> None:
        """Страница Storage."""
        
        @ui.page('/storage')
        async def storage():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Storage').classes('text-2xl font-bold mb-4')
                
                # Upload
                with ui.card().classes('w-full max-w-2xl'):
                    ui.label('Upload File').classes('text-lg font-bold')
                    
                    self._upload_content = ui.textarea('Content (text)').classes('w-full')
                    
                    with ui.row().classes('gap-4'):
                        ttl_input = ui.number('TTL (hours)', value=24)
                        ui.button('Store', icon='upload', on_click=lambda: self._store_data(
                            self._upload_content.value, int(ttl_input.value)
                        ))
                    
                    self._upload_result = ui.label('').classes('mt-2 font-mono text-sm')
                
                # Retrieve
                ui.label('Retrieve').classes('text-xl font-bold mt-6 mb-2')
                with ui.card().classes('w-full max-w-2xl'):
                    with ui.row().classes('gap-2'):
                        storage_id_input = ui.input('Storage ID').classes('flex-grow')
                        ui.button('Get', icon='download', on_click=lambda: self._get_data(storage_id_input.value))
                    
                    self._retrieve_result = ui.textarea('Result').classes('w-full mt-2').props('readonly')
                
                # List
                ui.label('My Files').classes('text-xl font-bold mt-6 mb-2')
                self._storage_list = ui.column().classes('w-full')
                ui.button('Refresh', icon='refresh', on_click=self._refresh_storage)
    
    async def _store_data(self, content: str, ttl: int) -> None:
        """Сохранить данные."""
        if not content:
            ui.notify('Content required', type='warning')
            return
        
        if self.agent_manager:
            agent = self.agent_manager._agents.get('storage')
            if agent:
                try:
                    result, cost = await agent.execute({
                        'action': 'store',
                        'data': content,
                        'ttl_hours': ttl,
                        'owner_id': self._state.node_id[:16],
                    })
                    
                    storage_id = result.get('storage_id', '')
                    if hasattr(self, '_upload_result'):
                        self._upload_result.text = f'Stored: {storage_id}'
                    ui.notify(f'Stored! ID: {storage_id}', type='positive')
                except Exception as e:
                    ui.notify(f'Error: {e}', type='negative')
    
    async def _get_data(self, storage_id: str) -> None:
        """Получить данные."""
        if not storage_id:
            ui.notify('Storage ID required', type='warning')
            return
        
        if self.agent_manager:
            agent = self.agent_manager._agents.get('storage')
            if agent:
                try:
                    result, cost = await agent.execute({
                        'action': 'get',
                        'storage_id': storage_id,
                    })
                    
                    import base64
                    data = result.get('data', '')
                    try:
                        decoded = base64.b64decode(data).decode('utf-8')
                    except:
                        decoded = data
                    
                    if hasattr(self, '_retrieve_result'):
                        self._retrieve_result.value = decoded
                    ui.notify('Retrieved!', type='positive')
                except Exception as e:
                    ui.notify(f'Error: {e}', type='negative')
    
    async def _refresh_storage(self) -> None:
        """Обновить список файлов."""
        if not hasattr(self, '_storage_list'):
            return
        
        self._storage_list.clear()
        
        if self.agent_manager:
            agent = self.agent_manager._agents.get('storage')
            if agent:
                try:
                    result, cost = await agent.execute({
                        'action': 'list',
                        'owner_id': self._state.node_id[:16],
                    })
                    
                    objects = result.get('objects', [])
                    
                    with self._storage_list:
                        if not objects:
                            ui.label('No files stored').classes('text-gray-500')
                        else:
                            for obj in objects:
                                with ui.card().classes('w-full'):
                                    with ui.row().classes('justify-between items-center'):
                                        ui.label(obj['storage_id'][:16] + '...').classes('font-mono')
                                        ui.label(f"{obj['size_bytes']} bytes")
                                        ui.button('Delete', icon='delete', color='red', on_click=lambda sid=obj['storage_id']: self._delete_data(sid)).props('size=sm')
                except Exception as e:
                    with self._storage_list:
                        ui.label(f'Error: {e}').classes('text-red-500')
    
    async def _delete_data(self, storage_id: str) -> None:
        """Удалить данные."""
        if self.agent_manager:
            agent = self.agent_manager._agents.get('storage')
            if agent:
                try:
                    await agent.execute({
                        'action': 'delete',
                        'storage_id': storage_id,
                    })
                    ui.notify('Deleted!', type='positive')
                    await self._refresh_storage()
                except Exception as e:
                    ui.notify(f'Error: {e}', type='negative')
    
    def _create_compute_page(self) -> None:
        """Страница Compute."""
        
        @ui.page('/compute')
        async def compute():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Compute').classes('text-2xl font-bold mb-4')
                
                # Eval
                with ui.card().classes('w-full max-w-2xl'):
                    ui.label('Evaluate Expression').classes('text-lg font-bold')
                    
                    expr_input = ui.input('Expression (e.g., 2+2*sin(pi/4))').classes('w-full')
                    
                    with ui.row().classes('gap-2'):
                        ui.button('Evaluate', icon='calculate', on_click=lambda: self._compute_eval(expr_input.value))
                    
                    self._eval_result = ui.label('').classes('mt-2 font-mono text-lg')
                
                # Math operations
                ui.label('Math Operations').classes('text-xl font-bold mt-6 mb-2')
                with ui.card().classes('w-full max-w-2xl'):
                    with ui.row().classes('gap-4'):
                        op_select = ui.select(
                            ['factorial', 'sqrt', 'power', 'prime_check'],
                            label='Operation',
                            value='factorial'
                        )
                        n_input = ui.number('n', value=10)
                        base_input = ui.number('base', value=2)
                        exp_input = ui.number('exp', value=10)
                    
                    ui.button('Calculate', icon='calculate', on_click=lambda: self._compute_math(
                        op_select.value, int(n_input.value), int(base_input.value), int(exp_input.value)
                    ))
                    
                    self._math_result = ui.label('').classes('mt-2 font-mono text-lg')
                
                # Hash
                ui.label('Hash').classes('text-xl font-bold mt-6 mb-2')
                with ui.card().classes('w-full max-w-2xl'):
                    hash_input = ui.textarea('Data to hash').classes('w-full')
                    algo_select = ui.select(['sha256', 'sha512', 'md5', 'blake2b'], value='sha256')
                    
                    ui.button('Hash', icon='fingerprint', on_click=lambda: self._compute_hash(
                        hash_input.value, algo_select.value
                    ))
                    
                    self._hash_result = ui.label('').classes('mt-2 font-mono text-sm break-all')
    
    async def _compute_eval(self, expression: str) -> None:
        """Вычислить выражение."""
        if not expression:
            return
        
        if self.agent_manager:
            agent = self.agent_manager._agents.get('compute')
            if agent:
                try:
                    result, cost = await agent.execute({
                        'task': 'eval',
                        'expression': expression,
                    })
                    
                    value = result.get('result', result)
                    if hasattr(self, '_eval_result'):
                        self._eval_result.text = f'= {value}'
                except Exception as e:
                    if hasattr(self, '_eval_result'):
                        self._eval_result.text = f'Error: {e}'
    
    async def _compute_math(self, operation: str, n: int, base: int, exp: int) -> None:
        """Математическая операция."""
        if self.agent_manager:
            agent = self.agent_manager._agents.get('compute')
            if agent:
                try:
                    payload = {'task': 'math', 'operation': operation, 'n': n}
                    if operation == 'power':
                        payload['base'] = base
                        payload['exp'] = exp
                    
                    result, cost = await agent.execute(payload)
                    
                    value = result.get('result', result)
                    if hasattr(self, '_math_result'):
                        self._math_result.text = f'= {value}'
                except Exception as e:
                    if hasattr(self, '_math_result'):
                        self._math_result.text = f'Error: {e}'
    
    async def _compute_hash(self, data: str, algorithm: str) -> None:
        """Вычислить хэш."""
        if not data:
            return
        
        if self.agent_manager:
            agent = self.agent_manager._agents.get('compute')
            if agent:
                try:
                    result, cost = await agent.execute({
                        'task': 'hash',
                        'data': data,
                        'algorithm': algorithm,
                    })
                    
                    value = result.get('result', result)
                    if hasattr(self, '_hash_result'):
                        self._hash_result.text = f'{algorithm}: {value}'
                except Exception as e:
                    if hasattr(self, '_hash_result'):
                        self._hash_result.text = f'Error: {e}'
    
    def _create_settings_page(self) -> None:
        """Страница Settings."""
        
        @ui.page('/settings')
        async def settings():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Settings').classes('text-2xl font-bold mb-4')
                
                with ui.card().classes('w-full max-w-2xl'):
                    ui.label('Node Settings').classes('text-lg font-bold')
                    ui.label(f"Version: {self._version_text}").classes('text-sm text-gray-500')
                    ui.button('Update & Restart', icon='system_update_alt', on_click=self._trigger_update)
                    
                    ui.input('Node ID', value=self._state.node_id).props('readonly')
                    ui.input('Host', value=self._state.host)
                    ui.number('Port', value=self._state.port)
                    
                    ui.separator()
                    
                    ui.label('Network').classes('font-bold mt-4')
                    ui.number('Max Peers', value=50)
                    ui.number('Debt Limit (bytes)', value=100_000_000)
                    
                    ui.button('Test P2P Download', icon='cloud_download', on_click=self._test_p2p_download)
                    
                    ui.separator()
                    
                    ui.label('AI').classes('font-bold mt-4')
                    ui.input('Ollama Host', value='localhost')
                    ui.number('Ollama Port', value=11434)
                    ui.input('Default Model', value='qwen3:32b')
                    
                    ui.button('Save', icon='save', on_click=lambda: ui.notify('Settings saved', type='positive'))
    
    def _create_logs_page(self) -> None:
        """Страница Logs."""
        
        @ui.page('/logs')
        async def logs():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Logs').classes('text-2xl font-bold mb-4')
                
                with ui.row().classes('gap-2 mb-4'):
                    ui.button('Clear', icon='delete', on_click=self._clear_logs)
                    level_select = ui.select(['ALL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], value='ALL')
                
                self._logs_container = ui.column().classes(
                    'w-full h-96 overflow-auto bg-gray-900 text-green-400 p-4 font-mono text-sm rounded'
                )
                
                ui.timer(1.0, self._flush_logs)
    
    async def _clear_logs(self) -> None:
        """Очистить логи."""
        if hasattr(self, '_logs_container'):
            self._logs_container.clear()
        self._log_buffer.clear()
    
    async def _flush_logs(self) -> None:
        """Вывести накопленные логи в UI."""
        if not hasattr(self, '_logs_container'):
            return
        while self._log_buffer:
            record = self._log_buffer.popleft()
            with self._logs_container:
                ui.label(record)

    async def _trigger_update(self) -> None:
        """Trigger backend update and restart (placeholder)."""
        ui.notify("Update requested; node will restart if auto-update is enabled.", type="info")
    
    def _create_cortex_page(self) -> None:
        """Страница Cortex - Автономная система знаний."""
        
        @ui.page('/cortex')
        async def cortex():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Cortex - Автономная Система Знаний').classes('text-2xl font-bold mb-4')
                
                # Status cards
                with ui.row().classes('w-full gap-4 flex-wrap mb-6'):
                    # Cortex Status
                    with ui.card().classes('w-64'):
                        ui.label('Cortex Status').classes('text-lg font-bold')
                        self._cortex_status = ui.badge('Checking...', color='yellow')
                        self._cortex_automata = ui.label('Automata: -')
                    
                    # Investigations
                    with ui.card().classes('w-64'):
                        ui.label('Investigations').classes('text-lg font-bold')
                        self._cortex_inv_count = ui.label('0').classes('text-4xl font-bold text-blue-500')
                        ui.label('completed').classes('text-sm text-gray-500')
                    
                    # Library
                    with ui.card().classes('w-64'):
                        ui.label('Library').classes('text-lg font-bold')
                        self._cortex_lib_topics = ui.label('0').classes('text-4xl font-bold text-green-500')
                        ui.label('topics indexed').classes('text-sm text-gray-500')
                    
                    # Open Bounties
                    with ui.card().classes('w-64'):
                        ui.label('Open Bounties').classes('text-lg font-bold')
                        self._cortex_bounties = ui.label('0').classes('text-4xl font-bold text-orange-500')
                        ui.label('available').classes('text-sm text-gray-500')
                
                # Tabs
                with ui.tabs().classes('w-full') as tabs:
                    investigate_tab = ui.tab('Investigate')
                    library_tab = ui.tab('Library')
                    council_tab = ui.tab('Council')
                    bounties_tab = ui.tab('Bounties')
                    activity_tab = ui.tab('Activity')
                    neural_tab = ui.tab('Neural 3D')
                
                with ui.tab_panels(tabs, value=investigate_tab).classes('w-full'):
                    # INVESTIGATE TAB
                    with ui.tab_panel(investigate_tab):
                        with ui.card().classes('w-full max-w-2xl'):
                            ui.label('Investigate Topic').classes('text-lg font-bold')
                            ui.label('Scout → Analyst → Librarian → DHT').classes('text-sm text-gray-500 mb-4')
                            
                            self._cortex_topic_input = ui.input('Topic to investigate').classes('w-full')
                            self._cortex_topic_input.value = 'quantum computing'
                            
                            with ui.row().classes('gap-4 mt-4'):
                                self._council_checkbox = ui.checkbox('Use Council (3 analysts)', value=self._use_council)
                                self._council_checkbox.on_value_change(lambda e: setattr(self, '_use_council', e.value))
                                
                                budget_input = ui.number('Budget', value=50)
                            
                            with ui.row().classes('gap-2 mt-4'):
                                ui.button(
                                    'Start Investigation',
                                    icon='search',
                                    on_click=lambda: self._start_investigation(
                                        self._cortex_topic_input.value,
                                        self._use_council,
                                        budget_input.value,
                                    )
                                ).classes('bg-blue-500')
                                
                                ui.button(
                                    'Quick Search',
                                    icon='bolt',
                                    on_click=lambda: self._quick_search(self._cortex_topic_input.value)
                                )
                            
                            # Status
                            self._cortex_inv_status = ui.label('').classes('mt-4 text-sm')
                            
                            # Result
                            self._cortex_inv_result = ui.card().classes('w-full mt-4 hidden')
                    
                    # LIBRARY TAB
                    with ui.tab_panel(library_tab):
                        with ui.card().classes('w-full'):
                            ui.label('Semantic Library').classes('text-lg font-bold')
                            
                            with ui.row().classes('gap-2 mb-4'):
                                self._lib_search_input = ui.input('Search topic').classes('flex-grow')
                                ui.button('Search', icon='search', on_click=lambda: self._search_library(self._lib_search_input.value))
                                ui.button('Refresh Library', icon='refresh', on_click=self._refresh_library)
                            
                            self._lib_results = ui.column().classes('w-full')
                    
                    # COUNCIL TAB  
                    with ui.tab_panel(council_tab):
                        with ui.card().classes('w-full max-w-4xl'):
                            ui.label('Convene Council').classes('text-lg font-bold')
                            ui.label('Parallel analysis by multiple analysts + Judge synthesis').classes('text-sm text-gray-500 mb-4')
                            
                            council_topic = ui.input('Topic').classes('w-full')
                            council_text = ui.textarea('Text to analyze').classes('w-full h-48')
                            council_budget = ui.number('Budget', value=100)
                            
                            ui.button(
                                'Convene Council',
                                icon='groups',
                                on_click=lambda: self._convene_council(
                                    council_topic.value,
                                    council_text.value,
                                    council_budget.value,
                                )
                            ).classes('mt-4 bg-purple-500')
                            
                            # Council status
                            self._council_status = ui.column().classes('w-full mt-4')
                    
                    # BOUNTIES TAB
                    with ui.tab_panel(bounties_tab):
                        with ui.card().classes('w-full'):
                            ui.label('Open Bounties').classes('text-lg font-bold mb-4')
                            
                            with ui.row().classes('gap-2 mb-4'):
                                bounty_topic = ui.input('New bounty topic').classes('flex-grow')
                                bounty_reward = ui.number('Reward', value=10)
                                ui.button(
                                    'Create Bounty',
                                    icon='add',
                                    on_click=lambda: self._create_bounty(bounty_topic.value, bounty_reward.value)
                                )
                            
                            ui.button('Refresh', icon='refresh', on_click=self._refresh_bounties)
                            
                            self._bounties_list = ui.column().classes('w-full mt-4')
                    
                    # ACTIVITY TAB
                    with ui.tab_panel(activity_tab):
                        with ui.card().classes('w-full'):
                            ui.label('Recent Activity').classes('text-lg font-bold mb-4')
                            
                            ui.button('Refresh', icon='refresh', on_click=self._refresh_activity)
                            ui.button('Force Scan', icon='bolt', on_click=self._force_scan)
                            
                            self._activity_list = ui.column().classes('w-full mt-4')
                    
                    # NEURAL 3D TAB
                    with ui.tab_panel(neural_tab):
                        with ui.column().classes('w-full'):
                            ui.label('Neural Network Visualization').classes('text-lg font-bold mb-2')
                            ui.label('3D interactive view of Cortex agents and connections').classes('text-sm text-gray-500 mb-4')
                            
                            with ui.row().classes('gap-2 mb-2'):
                                ui.button(
                                    'Open Fullscreen',
                                    icon='fullscreen',
                                    on_click=lambda: ui.run_javascript('window.open("/static/cortex_vis.html", "_blank")')
                                ).classes('bg-blue-500')
                                ui.button(
                                    'Refresh Graph',
                                    icon='refresh',
                                    on_click=self._refresh_neural_graph
                                )
                            
                            # Embedded iframe
                            ui.html('''
                                <iframe 
                                    src="/static/cortex_vis.html" 
                                    style="width: 100%; height: 600px; border: 1px solid #333; border-radius: 8px;"
                                    allow="fullscreen"
                                ></iframe>
                            ''', sanitize=False).classes('w-full')
                            
                            # Legend
                            with ui.card().classes('w-full mt-4'):
                                ui.label('Node Types').classes('font-bold')
                                with ui.row().classes('gap-6 mt-2'):
                                    with ui.row().classes('items-center gap-2'):
                                        ui.element('div').style('width:12px;height:12px;border-radius:50%;background:#ffffff;')
                                        ui.label('My Node')
                                    with ui.row().classes('items-center gap-2'):
                                        ui.element('div').style('width:12px;height:12px;border-radius:50%;background:#00ffff;')
                                        ui.label('Scout')
                                    with ui.row().classes('items-center gap-2'):
                                        ui.element('div').style('width:12px;height:12px;border-radius:50%;background:#bf40ff;')
                                        ui.label('Analyst')
                                    with ui.row().classes('items-center gap-2'):
                                        ui.element('div').style('width:12px;height:12px;border-radius:50%;background:#ffd700;')
                                        ui.label('Librarian')
                
                # Auto-refresh
                ui.timer(5.0, self._refresh_cortex_stats)
    
    async def _refresh_cortex_stats(self) -> None:
        """Обновить статистику Cortex."""
        if not self.cortex:
            if hasattr(self, '_cortex_status'):
                self._cortex_status.set_text('Not Available')
                self._cortex_status._props['color'] = 'red'
                self._cortex_status.update()
            return
        
        try:
            stats = await self.cortex.get_stats_async()
            
            if hasattr(self, '_cortex_status'):
                if stats['service']['running']:
                    self._cortex_status.set_text('Running')
                    self._cortex_status._props['color'] = 'green'
                else:
                    self._cortex_status.set_text('Stopped')
                    self._cortex_status._props['color'] = 'gray'
                self._cortex_status.update()
            
            if hasattr(self, '_cortex_automata'):
                automata = stats.get('automata', {})
                self._cortex_automata.text = f"Automata: {'Running' if automata.get('running') else 'Stopped'}"
            
            if hasattr(self, '_cortex_inv_count'):
                self._cortex_inv_count.text = str(stats.get('automata', {}).get('successful_investigations', 0))
            
            if hasattr(self, '_cortex_lib_topics'):
                self._cortex_lib_topics.text = str(stats.get('library', {}).get('cached_topics', 0))
            
            if hasattr(self, '_cortex_bounties'):
                self._cortex_bounties.text = str(stats.get('library', {}).get('open_bounties', 0))
                
        except Exception as e:
            logger.warning(f"[WEBUI] Cortex stats error: {e}")
    
    async def _start_investigation(self, topic: str, use_council: bool, budget: float) -> None:
        """Начать исследование темы."""
        if not topic:
            ui.notify('Enter a topic', type='warning')
            return
        
        if not self.cortex:
            ui.notify('Cortex not available', type='error')
            return
        
        if hasattr(self, '_cortex_inv_status'):
            self._cortex_inv_status.text = f"Investigating '{topic}'..."
        
        ui.notify(f"Starting investigation: {topic}", type='info')
        
        try:
            result = await self.cortex.investigate(topic, use_council=use_council, budget=budget)
            
            if result.get('success'):
                if hasattr(self, '_cortex_inv_status'):
                    self._cortex_inv_status.text = f"[OK] Investigation complete!"
                
                # Show result
                if hasattr(self, '_cortex_inv_result'):
                    self._cortex_inv_result.clear()
                    self._cortex_inv_result.classes(remove='hidden')
                    
                    with self._cortex_inv_result:
                        ui.label('Result').classes('text-lg font-bold')
                        ui.label(f"Method: {result.get('method', 'unknown')}").classes('text-sm')
                        ui.label(f"CID: {result.get('cid', 'N/A')[:32]}...").classes('text-sm font-mono')
                        ui.label(f"Confidence: {result.get('confidence', 0):.2f}").classes('text-sm')
                        
                        ui.separator()
                        
                        ui.label('Summary').classes('font-bold mt-2')
                        ui.label(result.get('summary', 'No summary')).classes('text-sm')
                        
                        if result.get('facts'):
                            ui.label('Key Facts').classes('font-bold mt-2')
                            for fact in result.get('facts', [])[:5]:
                                ui.label(f"• {fact}").classes('text-sm')
                
                ui.notify('Investigation complete!', type='positive')
            else:
                if hasattr(self, '_cortex_inv_status'):
                    self._cortex_inv_status.text = f"[FAIL] {result.get('error', 'Unknown error')}"
                ui.notify(f"Investigation failed: {result.get('error')}", type='negative')
                
        except Exception as e:
            if hasattr(self, '_cortex_inv_status'):
                self._cortex_inv_status.text = f"[ERROR] {str(e)}"
            ui.notify(f"Error: {e}", type='negative')
    
    async def _quick_search(self, topic: str) -> None:
        """Быстрый поиск в библиотеке."""
        if not topic:
            ui.notify('Enter a topic', type='warning')
            return
        
        if not self.cortex:
            ui.notify('Cortex not available', type='error')
            return
        
        try:
            reports = await self.cortex.search(topic)
            
            if reports:
                ui.notify(f'Found {len(reports)} reports', type='positive')
                # Show first result
                if hasattr(self, '_cortex_inv_result'):
                    self._cortex_inv_result.clear()
                    self._cortex_inv_result.classes(remove='hidden')
                    
                    with self._cortex_inv_result:
                        report = reports[0]
                        ui.label(f"Found: {report.topic}").classes('text-lg font-bold')
                        ui.label(f"Quality: {report.quality_score:.2f}").classes('text-sm')
                        ui.label(report.summary).classes('mt-2')
            else:
                ui.notify('No reports found - try investigating', type='warning')
                
        except Exception as e:
            ui.notify(f"Search error: {e}", type='negative')
    
    async def _search_library(self, query: str) -> None:
        """Поиск в библиотеке."""
        if not query:
            return
        
        if not self.cortex:
            ui.notify('Cortex not available', type='error')
            return
        
        if hasattr(self, '_lib_results'):
            self._lib_results.clear()
            
            with self._lib_results:
                ui.label(f"Searching for '{query}'...").classes('text-gray-500')
        
        try:
            reports = await self.cortex.search(query)
            
            if hasattr(self, '_lib_results'):
                self._lib_results.clear()
                
                with self._lib_results:
                    if not reports:
                        ui.label('No results found').classes('text-gray-500')
                        ui.button(
                            f'Investigate "{query}"',
                            icon='search',
                            on_click=lambda: self._start_investigation(query, False, 50)
                        )
                    else:
                        for report in reports:
                            with ui.card().classes('w-full mb-2'):
                                with ui.row().classes('justify-between items-start'):
                                    with ui.column():
                                        ui.label(report.topic).classes('font-bold')
                                        ui.label(report.summary[:200] + '...' if len(report.summary) > 200 else report.summary).classes('text-sm')
                                        status = getattr(report, 'compliance_status', None) or getattr(report, 'status', None)
                                        if status:
                                            color = {'SAFE': 'green', 'WARNING': 'orange', 'BLOCKED': 'red'}.get(status, 'gray')
                                            ui.badge(status, color=color)
                                    
                                    with ui.column().classes('text-right'):
                                        ui.badge(f'{report.quality_score:.2f}', color='green' if report.quality_score > 0.7 else 'yellow')
                                        ui.label(report.sentiment).classes('text-xs')
                                
                                if report.key_facts:
                                    ui.label('Facts:').classes('text-sm font-bold mt-2')
                                    for fact in report.key_facts[:3]:
                                        ui.label(f"• {fact}").classes('text-xs')
                                        
        except Exception as e:
            ui.notify(f"Search error: {e}", type='negative')
    
    async def _refresh_library(self) -> None:
        """Показать последние темы библиотеки."""
        if not self.cortex:
            ui.notify('Cortex not available', type='error')
            return
        
        if hasattr(self, '_lib_results'):
            self._lib_results.clear()
            with self._lib_results:
                ui.label('Loading recent topics...').classes('text-gray-500')
        
        try:
            investigations = self.cortex.get_recent_investigations(10)
            topics = [inv.topic for inv in investigations]
            
            if hasattr(self, '_lib_results'):
                self._lib_results.clear()
                with self._lib_results:
                    if not topics:
                        ui.label('No topics yet. Try Force Scan or Investigate.').classes('text-gray-500')
                    else:
                        for topic in topics:
                            reports = await self.cortex.search(topic)
                            if not reports:
                                with ui.card().classes('w-full mb-2'):
                                    ui.label(topic).classes('font-bold')
                                    ui.label('No reports yet').classes('text-sm text-gray-500')
                                continue
                            
                            for report in reports[:1]:
                                with ui.card().classes('w-full mb-2'):
                                    with ui.row().classes('justify-between items-start'):
                                        with ui.column():
                                            ui.label(report.topic).classes('font-bold')
                                            ui.label(report.summary).classes('text-sm')
                                        with ui.column().classes('text-right'):
                                            ui.label(f"CID: {report.cid[:12]}...").classes('text-xs font-mono')
                                            ts = datetime.fromtimestamp(report.created_at).strftime('%Y-%m-%d %H:%M:%S')
                                            ui.label(ts).classes('text-xs text-gray-500')
                                            ui.badge(f'{report.quality_score:.2f}', color='green' if report.quality_score > 0.7 else 'yellow')
        except Exception as e:
            ui.notify(f"Library refresh error: {e}", type='negative')
    
    async def _convene_council(self, topic: str, text: str, budget: float) -> None:
        """Созвать совет аналитиков."""
        if not topic or not text:
            ui.notify('Enter topic and text', type='warning')
            return
        
        if not self.cortex:
            ui.notify('Cortex not available', type='error')
            return
        
        if hasattr(self, '_council_status'):
            self._council_status.clear()
            with self._council_status:
                ui.label('Council convening...').classes('text-blue-500')
                ui.spinner()
        
        ui.notify('Convening council...', type='info')
        
        try:
            result = await self.cortex.convene_council(topic, text, budget)
            
            if hasattr(self, '_council_status'):
                self._council_status.clear()
                
                with self._council_status:
                    status_color = 'green' if result.status.value == 'completed' else 'red'
                    ui.badge(result.status.value.upper(), color=status_color)
                    
                    ui.label(f"Confidence: {result.confidence:.2f}").classes('mt-2')
                    ui.label(f"Agreement: {result.analyst_agreement:.2f}")
                    ui.label(f"Participants: {len(result.participants)}")
                    ui.label(f"Cost: {result.total_cost:.2f}")
                    
                    ui.separator()
                    
                    ui.label('Final Summary').classes('font-bold mt-2')
                    ui.label(result.final_summary)
                    
                    if result.consensus_facts:
                        ui.label('Consensus Facts').classes('font-bold mt-2')
                        for fact in result.consensus_facts[:5]:
                            ui.label(f"[OK] {fact}").classes('text-sm text-green-600')
                    
                    if result.disputed_facts:
                        ui.label('Disputed Facts').classes('font-bold mt-2')
                        for fact in result.disputed_facts[:3]:
                            ui.label(f"[?] {fact}").classes('text-sm text-yellow-600')
                    
                    if result.filtered_hallucinations:
                        ui.label('Filtered (possible hallucinations)').classes('font-bold mt-2')
                        for h in result.filtered_hallucinations[:3]:
                            ui.label(f"[X] {h}").classes('text-sm text-red-600')
            
            ui.notify('Council complete!', type='positive')
            
        except Exception as e:
            if hasattr(self, '_council_status'):
                self._council_status.clear()
                with self._council_status:
                    ui.label(f"Error: {e}").classes('text-red-500')
            ui.notify(f"Council error: {e}", type='negative')
    
    async def _create_bounty(self, topic: str, reward: float) -> None:
        """Создать bounty."""
        if not topic:
            ui.notify('Enter topic', type='warning')
            return
        
        if not self.cortex:
            ui.notify('Cortex not available', type='error')
            return
        
        try:
            bounty = await self.cortex.create_bounty(topic, reward)
            if bounty:
                ui.notify(f'Bounty created: {bounty.bounty_id[:16]}...', type='positive')
                await self._refresh_bounties()
            else:
                ui.notify('Failed to create bounty', type='negative')
        except Exception as e:
            ui.notify(f"Error: {e}", type='negative')
    
    async def _refresh_bounties(self) -> None:
        """Обновить список bounties."""
        if not self.cortex:
            return
        
        if hasattr(self, '_bounties_list'):
            self._bounties_list.clear()
            
            try:
                bounties = await self.cortex.get_open_bounties()
                
                with self._bounties_list:
                    if not bounties:
                        ui.label('No open bounties').classes('text-gray-500')
                    else:
                        for bounty in bounties:
                            with ui.card().classes('w-full mb-2'):
                                with ui.row().classes('justify-between items-center'):
                                    with ui.column():
                                        ui.label(bounty.topic).classes('font-bold')
                                        ui.label(f"ID: {bounty.bounty_id[:16]}...").classes('text-xs font-mono')
                                    
                                    with ui.column().classes('text-right'):
                                        ui.badge(f'{bounty.reward:.0f}', color='orange')
                                        ui.label(bounty.status.value).classes('text-xs')
                                        
            except Exception as e:
                with self._bounties_list:
                    ui.label(f"Error: {e}").classes('text-red-500')
    
    async def _refresh_activity(self) -> None:
        """Обновить список активности."""
        if not self.cortex:
            return
        
        if hasattr(self, '_activity_list'):
            self._activity_list.clear()
            
            try:
                investigations = self.cortex.get_recent_investigations(10)
                
                with self._activity_list:
                    if not investigations:
                        ui.label('No recent activity').classes('text-gray-500')
                    else:
                        for inv in investigations:
                            with ui.card().classes('w-full mb-2'):
                                with ui.row().classes('justify-between items-center'):
                                    with ui.column():
                                        ui.label(inv.topic).classes('font-bold')
                                        ui.label(f"ID: {inv.investigation_id}").classes('text-xs font-mono')
                                        if hasattr(inv, 'updated_at') and inv.updated_at:
                                            ts = datetime.fromtimestamp(inv.updated_at).strftime('%Y-%m-%d %H:%M:%S')
                                            ui.label(f"Updated: {ts}").classes('text-xs text-gray-500')
                                    
                                    status_color = {
                                        'completed': 'green',
                                        'failed': 'red',
                                        'pending': 'yellow',
                                        'scouting': 'blue',
                                        'analyzing': 'blue',
                                        'storing': 'blue',
                                    }.get(inv.status.value, 'gray')
                                    
                                    ui.badge(inv.status.value, color=status_color)
                    
                    # Append latest logs snapshot
                    ui.label('Recent Logs').classes('text-lg font-bold mt-4')
                    for record in list(self._log_buffer)[-10:]:
                        ui.label(record).classes('text-xs font-mono text-gray-400')
            
            except Exception as e:
                with self._activity_list:
                    ui.label(f"Error: {e}").classes('text-red-500')

    async def _refresh_neural_graph(self) -> None:
        """Обновить 3D граф через WebSocket."""
        if self.visualizer:
            await self.visualizer._build_graph()
            ui.notify('Graph refreshed', type='positive')

    async def _force_scan(self) -> None:
        """Принудительно обработать тестовую тему через автомату."""
        if not self.cortex or not getattr(self.cortex, 'automata', None):
            ui.notify('Cortex automata not available', type='warning')
            return
        
        topic = f"Test Topic {uuid.uuid4().hex[:6]}"
        ui.notify(f'Processing {topic}...', type='info')
        
        try:
            result = await self.cortex.automata.process_topic(topic)
            ui.notify(f'Completed {topic}: {result.status.value}', type='positive')
            await self._refresh_library()
            await self._refresh_activity()
        except Exception as e:
            ui.notify(f'Force scan error: {e}', type='negative')

    async def _test_p2p_download(self) -> None:
        """Trigger small test download via P2P loader."""
        ui.notify('Starting P2P test download...', type='info')
        try:
            loader = P2PLoader(kademlia=getattr(self, 'kademlia', None), node=self, base_dir=Path("data/models"))
            await loader.ensure_model("prajjwal1/bert-tiny", fallback_http=True)
            ui.notify('P2P test download finished', type='positive')
        except Exception as e:
            ui.notify(f'P2P test download failed: {e}', type='negative')

    # =========================================================================
    # Ingest
    # =========================================================================
    def _create_ingest_page(self) -> None:
        """Страница для загрузки локальных файлов."""
        
        @ui.page('/ingest')
        async def ingest():
            await self._create_header()
            await self._create_sidebar()
            
            with ui.column().classes('w-full p-4'):
                ui.label('Ingest Documents').classes('text-2xl font-bold mb-4')
                ui.label('Point to a directory to scan, summarize, and index for RAG.').classes('text-sm text-gray-500')
                
                self._ingest_path = ui.input('Directory path', value=str(Path.cwd())).classes('w-full')
                self._dream_toggle = ui.checkbox('Enable Background Media Analysis (Dream Mode)', value=self._dream_mode_enabled)
                self._ingest_progress = ui.label('Idle').classes('text-sm text-gray-500')
                self._ingest_button = ui.button('Start Digestion', icon='cloud_upload', on_click=self._start_ingest)
                
                self._ingest_logs = ui.column().classes('w-full mt-4')
    
    async def _start_ingest(self) -> None:
        """Запустить процесс инжеста документов."""
        path = getattr(self, "_ingest_path", None)
        if path:
            path = path.value
        if not path:
            ui.notify('Provide directory path', type='warning')
            return
        self._dream_mode_enabled = getattr(self, "_dream_toggle", None).value if hasattr(self, "_dream_toggle") else False
        
        ui.notify(f'Scanning {path}...', type='info')
        if hasattr(self, "_ingest_progress"):
            self._ingest_progress.text = "Scanning..."
        
        if self._dream_mode_enabled and self.idle_worker:
            self.idle_worker.enqueue([path])
            ui.notify('Queued for background processing', type='positive')
            if hasattr(self, "_ingest_progress"):
                self._ingest_progress.text = "Queued in background..."
            return
        
        scanner = AsyncFileScanner()
        processor = DocumentProcessor()
        vector_store = VectorStore()
        
        processed = 0
        added = 0
        
        async for doc in scanner.scan_directory(path):
            processed += 1
            if hasattr(self, "_ingest_progress"):
                self._ingest_progress.text = f"Processed {processed}"
            try:
                result = await processor.process_document(doc.text, doc.metadata)
                cid = result.get("summary", "")[:32]
                summary = result.get("summary", "")[:512]
                tags = ""
                if isinstance(result.get("summary"), str):
                    tags = ""
                if self.ledger:
                    await self.ledger.add_knowledge_entry(
                        cid=cid,
                        path=str(doc.path),
                        summary=summary,
                        tags=tags,
                        size=int(doc.metadata.get("size", 0)),
                        metadata=doc.metadata,
                    )
                # Vector store
                vector_store.embed_and_store([doc.text], metadata=doc.metadata)
                added += 1
                if hasattr(self, "_ingest_logs"):
                    with self._ingest_logs:
                        ui.label(f"[INGEST] {doc.path.name} ingested").classes('text-xs')
            except Exception as e:
                logger.warning(f"[INGEST] Failed on {doc.path}: {e}")
                if hasattr(self, "_ingest_logs"):
                    with self._ingest_logs:
                        ui.label(f"[ERR] {doc.path.name}: {e}").classes('text-xs text-red-500')
        
        if hasattr(self, "_ingest_progress"):
            self._ingest_progress.text = f"Completed. Added {added} / {processed}"
        ui.notify(f'Ingestion complete: {added} files', type='positive')
    
    def _setup_websocket_endpoint(self) -> None:
        """Настроить WebSocket endpoint для визуализации."""
        if not NICEGUI_AVAILABLE:
            return
        
        visualizer = self.visualizer
        
        # NiceGUI использует FastAPI/Starlette под капотом
        # Добавляем WebSocket endpoint через декоратор app
        @app.on_startup
        async def start_visualizer():
            """Start visualizer when app starts."""
            if visualizer:
                try:
                    await visualizer.start()
                    logger.info("[WEBUI] Visualizer started")
                except Exception as e:
                    logger.warning(f"[WEBUI] Visualizer start error: {e}")
        
        # Add API endpoint for graph data using FastAPI route
        from fastapi import Response
        import json
        
        @app.get('/api/vis/graph')
        async def api_vis_graph():
            """API endpoint to get graph data as JSON."""
            if visualizer:
                data = visualizer.get_graph_data()
            else:
                from .vis_endpoint import generate_demo_graph
                data = generate_demo_graph()
            return Response(
                content=json.dumps(data),
                media_type="application/json"
            )
        
        logger.info("[WEBUI] Visualization API endpoint /api/vis/graph configured")
    
    def setup_pages(self) -> None:
        """Настроить все страницы."""
        # Mount static files for 3D visualization
        if NICEGUI_AVAILABLE and STATIC_DIR.exists():
            app.add_static_files('/static', str(STATIC_DIR))
            logger.info(f"[WEBUI] Static files mounted from {STATIC_DIR}")
        
        self._create_dashboard_page()
        self._create_peers_page()
        self._create_services_page()
        self._create_ai_page()
        self._create_cortex_page()
        self._create_dht_page()
        self._create_economy_page()
        self._create_storage_page()
        self._create_compute_page()
        self._create_settings_page()
        self._create_logs_page()
        self._create_ingest_page()
        self.library_tab.create_page()
        self.ingest_tab.create_page(self)
        self.activity_tab.create_page(self)
        self._setup_websocket_endpoint()
    
    def run_sync(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        Запустить Web UI синхронно (блокирующий вызов).
        
        Используется как основная точка входа.
        
        Args:
            host: Адрес для прослушивания
            port: Порт
        """
        if not NICEGUI_AVAILABLE:
            logger.error("[WEBUI] NiceGUI not installed. Run: pip install nicegui")
            return
        
        self._running = True
        self._update_state()
        
        # Setup pages
        self.setup_pages()
        
        logger.info(f"[WEBUI] Starting on http://{host}:{port}")
        
        # Run NiceGUI (блокирующий вызов)
        ui.run(
            host=host,
            port=port,
            title=self.title,
            dark=self.dark_mode,
            reload=False,
            show=False,
            storage_secret="p2p_node_secret_key_change_in_production",
        )
    
    async def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        Запустить Web UI в отдельном потоке (для использования из async контекста).
        
        Args:
            host: Адрес для прослушивания
            port: Порт
        """
        if not NICEGUI_AVAILABLE:
            logger.error("[WEBUI] NiceGUI not installed. Run: pip install nicegui")
            return
        
        import threading
        
        self._running = True
        self._update_state()
        
        # Setup pages
        self.setup_pages()
        
        logger.info(f"[WEBUI] Starting on http://{host}:{port}")
        
        # Запускаем NiceGUI в отдельном потоке
        def run_ui():
            ui.run(
                host=host,
                port=port,
                title=self.title,
                dark=self.dark_mode,
                reload=False,
                show=False,
                storage_secret="p2p_node_secret_key_change_in_production",
            )
        
        ui_thread = threading.Thread(target=run_ui, daemon=True)
        ui_thread.start()
        
        # Ждём пока не остановят
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass


def create_webui(
    node=None,
    ledger=None,
    agent_manager=None,
    kademlia=None,
    cortex=None,
    visualizer=None,
    idle_worker=None,
    **kwargs,
) -> P2PWebUI:
    """
    Factory function для создания WebUI.
    
    Args:
        node: P2P Node
        ledger: Ledger
        agent_manager: AgentManager
        kademlia: KademliaNode
        cortex: CortexService
        visualizer: CortexVisualizer
    
    Returns:
        P2PWebUI instance
    """
    # Auto-create visualizer if not provided but cortex is available
    if visualizer is None and cortex is not None:
        try:
            from .vis_endpoint import CortexVisualizer
            visualizer = CortexVisualizer(cortex=cortex, node=node, ledger=ledger)
            logger.info("[WEBUI] Auto-created CortexVisualizer")
        except Exception as e:
            logger.warning(f"[WEBUI] Could not create visualizer: {e}")
    
    return P2PWebUI(
        node=node,
        ledger=ledger,
        agent_manager=agent_manager,
        kademlia=kademlia,
        cortex=cortex,
        visualizer=visualizer,
        idle_worker=idle_worker,
        **kwargs,
    )
