import asyncio
import time
from typing import Any, Dict, List, Optional

from nicegui import ui

from core.events import event_bus
from cortex.media.service import MediaService


class CinemaTab:
    def __init__(self, media: Optional[MediaService] = None):
        self.media = media or MediaService.get_instance()
        self._results_grid = None
        self._search_input = None
        self._tabs = None
        self._player_tab = None
        self._player_html = None
        self._player_input = None
        self._context_timeline = None
        self._context_desc = None
        self._progress_bar = None
        self._progress_label = None
        self._earned_label = None
        self._selected_hash = ""
        self._selected_metadata: Optional[Dict[str, Any]] = None
        self._last_refresh = 0.0
        self._register_listeners()

    def _register_listeners(self) -> None:
        async def on_progress(payload):
            self._update_progress(payload)

        async def on_storyboard(payload):
            self._maybe_refresh(payload)

        try:
            asyncio.create_task(event_bus.subscribe("media_index_progress", on_progress))
            asyncio.create_task(event_bus.subscribe("media_storyboard", on_storyboard))
            asyncio.create_task(event_bus.subscribe("media_index_complete", on_storyboard))
        except Exception:
            pass

    def create_page(self, parent) -> None:
        @ui.page('/cinema')
        async def cinema():
            await parent._create_header()
            await parent._create_sidebar()

            try:
                self.media.attach_loop(asyncio.get_running_loop())
            except Exception:
                pass

            with ui.column().classes('w-full p-4'):
                ui.label('Cinema').classes('text-2xl font-bold mb-4')

                with ui.tabs().classes('w-full') as tabs:
                    search_tab = ui.tab('Search / Feed')
                    self._player_tab = ui.tab('Player View')
                    monitor_tab = ui.tab('Indexing Monitor')
                self._tabs = tabs

                with ui.tab_panels(tabs, value=search_tab).classes('w-full'):
                    with ui.tab_panel(search_tab):
                        self._render_search_panel()
                    with ui.tab_panel(self._player_tab):
                        self._render_player_panel()
                    with ui.tab_panel(monitor_tab):
                        self._render_monitor_panel()

            await self._refresh_results()

    def _render_search_panel(self) -> None:
        with ui.row().classes('gap-2 mb-4'):
            self._search_input = ui.input('Search').classes('w-72')
            ui.button('Search', icon='search', on_click=self._on_search)
            ui.button('Refresh', icon='refresh', on_click=self._refresh_results)

        self._results_grid = ui.grid(columns=3).classes('gap-4')

    def _render_player_panel(self) -> None:
        with ui.row().classes('w-full gap-4'):
            with ui.column().classes('w-2/3'):
                ui.label('Now Playing').classes('text-lg font-bold')
                with ui.row().classes('gap-2 mb-2'):
                    self._player_input = ui.input('Info hash').classes('w-96')
                    ui.button('Stream', icon='play_arrow', on_click=self._stream_selected)
                self._player_html = ui.html(self._build_player_html(''))
            with ui.column().classes('w-1/3'):
                ui.label('AI Context').classes('text-lg font-bold')
                ui.label('Scene Timeline').classes('text-sm text-gray-500')
                self._context_timeline = ui.column().classes('gap-2')
                ui.separator()
                ui.label('Current Scene').classes('text-sm text-gray-500')
                self._context_desc = ui.label('No scene selected').classes('text-sm')

    def _render_monitor_panel(self) -> None:
        ui.label('Mining Meaning').classes('text-lg font-bold mb-2')
        self._progress_label = ui.label('Waiting for indexing...').classes('text-sm text-gray-500')
        self._progress_bar = ui.linear_progress(value=0).classes('w-full')
        self._earned_label = ui.label('Potential earned: 0.0 SIBR').classes('text-sm text-green-400')

        ui.separator().classes('my-4')
        ui.label('Add Magnet').classes('text-sm font-bold')
        magnet_input = ui.input('Magnet link').classes('w-full')
        ui.button('Index', icon='cloud_download', on_click=lambda: self._add_magnet(magnet_input.value))

    async def _refresh_results(self) -> None:
        query = self._search_input.value if self._search_input else ''
        results = self.media.search(query)
        self._render_results(results)

    def _render_results(self, results: List[Dict[str, Any]]) -> None:
        if not self._results_grid:
            return
        self._results_grid.clear()
        with self._results_grid:
            for item in results:
                self._render_card(item)

    def _render_card(self, item: Dict[str, Any]) -> None:
        storyboard = item.get('storyboard', [])
        frames = [
            entry.get('frame_path')
            for entry in storyboard
            if entry.get('frame_path')
        ]
        title = item.get('name') or item.get('info_hash', '')
        rights_status = item.get('rights_status', 'UNKNOWN')
        indexed = bool(storyboard)

        with ui.card().classes('w-full'):
            image = ui.image(frames[0] if frames else '').classes('w-full h-40 object-cover bg-black')
            if len(frames) > 1:
                self._attach_slideshow(image, frames)
            ui.label(title[:80]).classes('text-sm font-bold')

            with ui.row().classes('gap-2 my-1'):
                if rights_status == 'LICENSED':
                    ui.badge('Verified', color='green')
                if indexed:
                    ui.badge('Indexed', color='blue')

            ui.button('Play', icon='play_circle', on_click=lambda item=item: self._select_media(item)).classes('w-full')

    def _attach_slideshow(self, image, frames: List[str]) -> None:
        state = {'idx': 0}

        def _advance():
            state['idx'] = (state['idx'] + 1) % len(frames)
            try:
                image.set_source(frames[state['idx']])
            except Exception:
                pass

        ui.timer(2.5, _advance)

    def _select_media(self, item: Dict[str, Any]) -> None:
        self._selected_hash = item.get('info_hash', '')
        self._selected_metadata = item
        if self._player_input:
            self._player_input.value = self._selected_hash
        self._update_context_panel(item)
        if self._tabs and self._player_tab:
            try:
                self._tabs.set_value(self._player_tab)
            except Exception:
                pass

    def _update_context_panel(self, item: Dict[str, Any]) -> None:
        storyboard = item.get('storyboard', [])
        if self._context_timeline:
            try:
                self._context_timeline.clear()
                with self._context_timeline:
                    for entry in storyboard:
                        time_label = entry.get('time', '--:--')
                        tags = entry.get('tags', [])
                        with ui.row().classes('gap-2 items-center'):
                            ui.label(time_label).classes('text-xs text-gray-400 w-12')
                            for tag in tags:
                                ui.chip(str(tag))
            except Exception:
                pass
        if self._context_desc:
            desc = storyboard[-1].get('desc') if storyboard else 'No scene selected'
            self._context_desc.text = desc or 'No scene selected'

    def _build_player_html(self, url: str) -> str:
        if not url:
            return '<div class="text-sm text-gray-500">No stream selected</div>'
        return (
            '<video controls preload="metadata" '
            'style="width:100%; max-height:480px; background:black;">'
            f'<source src="{url}" type="video/mp4">'
            'Your browser does not support the video tag.'</n            '</video>'
        )

    def _stream_selected(self) -> None:
        info_hash = self._player_input.value if self._player_input else ''
        if not info_hash:
            ui.notify('Info hash required', type='warning')
            return
        self.media.stream(info_hash)
        url = f"http://localhost:8080/stream/{info_hash}"
        if self._player_html:
            self._player_html.content = self._build_player_html(url)
            self._player_html.update()

    def _on_search(self) -> None:
        asyncio.create_task(self._refresh_results())

    def _add_magnet(self, link: str) -> None:
        if not link:
            ui.notify('Magnet link required', type='warning')
            return
        info_hash = self.media.add_magnet(link)
        if info_hash:
            ui.notify(f'Indexing started: {info_hash}', type='positive')

    def _update_progress(self, payload: Dict[str, Any]) -> None:
        processed = payload.get('processed', 0)
        total = payload.get('total', 1)
        earned = payload.get('earned', 0.0)
        if self._progress_label:
            self._progress_label.text = (
                f"Processed {processed}/{total} keyframes"
            )
        if self._progress_bar:
            self._progress_bar.value = processed / max(total, 1)
        if self._earned_label:
            self._earned_label.text = f"Potential earned: {earned:.2f} SIBR"
        if self._selected_hash and payload.get('info_hash') == self._selected_hash:
            meta = payload.get('metadata')
            if isinstance(meta, dict):
                self._update_context_panel(meta)

    def _maybe_refresh(self, payload: Dict[str, Any]) -> None:
        now = time.time()
        if now - self._last_refresh < 1.0:
            return
        self._last_refresh = now
        asyncio.create_task(self._refresh_results())


__all__ = ["CinemaTab"]
