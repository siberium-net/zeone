"""
Wallet Management Tab
=====================

[WEBUI] NiceGUI interface for ZEONE wallet and staking management.

Features:
- Native SIBR balance display
- Staking deposit/withdraw
- Settlement contract info
- Real-time blockchain status

[DESIGN] Modern dark theme with Siberium branding colors.
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from decimal import Decimal
from typing import Optional, Dict, Any
from dataclasses import dataclass

from nicegui import ui

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEPLOYMENT_FILE = DATA_DIR / "siberium_deployment.json"


# ============================================================================
# Siberium Manager Wrapper
# ============================================================================

@dataclass
class WalletState:
    """Current wallet state."""
    connected: bool = False
    address: str = ""
    balance: Decimal = Decimal("0")
    staked: Decimal = Decimal("0")
    pending_unstake: Decimal = Decimal("0")
    unlock_time: int = 0
    block_number: int = 0
    chain_id: int = 0
    network_name: str = "Unknown"
    contract_address: str = ""
    error: str = ""


class WalletManager:
    """
    Manages wallet connection and operations.
    
    Wraps SiberiumManager with error handling and state management.
    """
    
    def __init__(self):
        self.state = WalletState()
        self._manager = None
        self._deployment: Dict[str, Any] = {}
        self._refresh_task: Optional[asyncio.Task] = None
    
    def _load_deployment(self) -> bool:
        """Load deployment configuration."""
        if not DEPLOYMENT_FILE.exists():
            self.state.error = "Deployment file not found. Run deploy_siberium.py first."
            return False
        
        try:
            with open(DEPLOYMENT_FILE) as f:
                self._deployment = json.load(f)
            self.state.contract_address = self._deployment.get("address", "")
            return True
        except Exception as e:
            self.state.error = f"Failed to load deployment: {e}"
            return False
    
    async def connect(self) -> bool:
        """Connect to Siberium blockchain."""
        try:
            # Load deployment first
            if not self._load_deployment():
                return False
            
            # Import here to avoid circular imports
            from economy.chain import SiberiumManager
            
            rpc_url = os.getenv("SIBERIUM_RPC_URL", "https://rpc.test.siberium.net")
            private_key = os.getenv("PRIVATE_KEY", "")
            
            if not private_key:
                self.state.error = "PRIVATE_KEY not set in environment"
                return False
            
            # Initialize manager
            self._manager = SiberiumManager(
                rpc_url=rpc_url,
                private_key=private_key,
                settlement_address=self.state.contract_address,
            )
            
            # Get network info
            self.state.chain_id = self._manager.w3.eth.chain_id
            self.state.network_name = {
                111111: "Siberium Mainnet",
                111000: "Siberium Testnet",
            }.get(self.state.chain_id, f"Chain {self.state.chain_id}")
            
            self.state.address = self._manager.address
            self.state.connected = True
            self.state.error = ""
            
            # Initial refresh
            await self.refresh()
            
            return True
            
        except Exception as e:
            self.state.error = f"Connection failed: {e}"
            self.state.connected = False
            logger.exception("Wallet connection failed")
            return False
    
    async def refresh(self) -> None:
        """Refresh wallet state from blockchain."""
        if not self._manager:
            return
        
        try:
            # Get block number
            self.state.block_number = self._manager.w3.eth.block_number
            
            # Get native balance
            self.state.balance = self._manager.get_balance()
            
            # Get staking info
            if self._manager.settlement:
                info = await self._manager.get_account_info()
                self.state.staked = info["staked"]
                self.state.pending_unstake = info["pending_unstake"]
                self.state.unlock_time = info["unlock_time"]
            
            self.state.error = ""
            
        except Exception as e:
            self.state.error = f"Refresh failed: {e}"
            logger.warning(f"Wallet refresh failed: {e}")
    
    async def deposit(self, amount: Decimal) -> str:
        """Deposit SIBR as stake."""
        if not self._manager:
            raise RuntimeError("Not connected")
        
        return await self._manager.deposit_stake(amount)
    
    async def withdraw(self) -> str:
        """Withdraw pending unstake."""
        if not self._manager:
            raise RuntimeError("Not connected")
        
        return await self._manager.withdraw()
    
    async def request_unstake(self, amount: Decimal) -> str:
        """Request unstake (starts timelock)."""
        if not self._manager:
            raise RuntimeError("Not connected")
        
        return await self._manager.request_unstake(amount)
    
    async def withdraw_balance(self, amount: Decimal) -> str:
        """Withdraw available staked balance."""
        if not self._manager:
            raise RuntimeError("Not connected")
        
        return await self._manager.withdraw_balance(amount)
    
    def get_explorer_url(self, address_or_tx: str, is_tx: bool = False) -> str:
        """Get block explorer URL."""
        if self.state.chain_id == 111000:
            base = "https://explorer.test.siberium.net"
        elif self.state.chain_id == 111111:
            base = "https://explorer.siberium.net"
        else:
            return ""
        
        path = "tx" if is_tx else "address"
        return f"{base}/{path}/{address_or_tx}"


# ============================================================================
# Global Wallet Instance
# ============================================================================

wallet = WalletManager()


# ============================================================================
# UI Components
# ============================================================================

def create_status_indicator(connected: bool, block: int = 0) -> ui.element:
    """Create connection status indicator."""
    with ui.row().classes("items-center gap-2"):
        if connected:
            ui.icon("radio_button_checked", color="green").classes("text-lg")
            ui.label(f"Online (Block #{block:,})").classes("text-green-400 text-sm")
        else:
            ui.icon("radio_button_unchecked", color="red").classes("text-lg")
            ui.label("Offline").classes("text-red-400 text-sm")


def format_sibr(amount: Decimal, decimals: int = 4) -> str:
    """Format SIBR amount for display."""
    return f"{amount:,.{decimals}f} SIBR"


# ============================================================================
# Wallet Tab
# ============================================================================

class WalletTab:
    """
    Wallet management tab for NiceGUI.
    
    [USAGE]
        with ui.tab_panel("wallet"):
            WalletTab().render()
    """
    
    def __init__(self):
        self.balance_label: Optional[ui.label] = None
        self.staked_label: Optional[ui.label] = None
        self.status_container: Optional[ui.element] = None
        self.error_label: Optional[ui.label] = None
        self.deposit_input: Optional[ui.input] = None
        self.withdraw_input: Optional[ui.input] = None
        self._refresh_timer: Optional[ui.timer] = None
    
    def render(self) -> None:
        """Render the wallet tab content."""
        # Custom CSS
        ui.add_head_html("""
        <style>
            .wallet-card {
                background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 16px;
            }
            .wallet-header {
                background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 24px;
            }
            .sibr-balance {
                font-family: 'JetBrains Mono', monospace;
                font-size: 2.5rem;
                font-weight: bold;
                color: #ffffff;
            }
            .stat-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.25rem;
                color: #58a6ff;
            }
            .action-btn {
                background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                color: white;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .action-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(35, 134, 54, 0.4);
            }
            .action-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            .warning-btn {
                background: linear-gradient(135deg, #da3633 0%, #f85149 100%);
            }
            .contract-link {
                color: #58a6ff;
                text-decoration: none;
            }
            .contract-link:hover {
                text-decoration: underline;
            }
        </style>
        """)
        
        with ui.column().classes("w-full max-w-4xl mx-auto p-4"):
            # Header
            self._render_header()
            
            # Error message
            self.error_label = ui.label().classes("text-red-400 text-center mb-4")
            self.error_label.set_visibility(False)
            
            # Main content grid
            with ui.row().classes("w-full gap-4"):
                # Left column: Balance & Staking
                with ui.column().classes("flex-1"):
                    self._render_balance_card()
                    self._render_staking_card()
                
                # Right column: Contract & Actions
                with ui.column().classes("flex-1"):
                    self._render_contract_card()
                    self._render_actions_card()
        
        # Start auto-refresh
        self._start_refresh_timer()
        
        # Initial connection
        ui.timer(0.5, self._initial_connect, once=True)
    
    def _render_header(self) -> None:
        """Render wallet header with balance."""
        with ui.element("div").classes("wallet-header w-full"):
            with ui.row().classes("justify-between items-center"):
                with ui.column():
                    ui.label("ZEONE Wallet").classes("text-white text-lg font-bold mb-1")
                    self.balance_label = ui.label("0.0000 SIBR").classes("sibr-balance")
                
                with ui.column().classes("items-end"):
                    self.status_container = ui.element("div")
                    with self.status_container:
                        create_status_indicator(False)
    
    def _render_balance_card(self) -> None:
        """Render balance overview card."""
        with ui.element("div").classes("wallet-card"):
            ui.label("Account Overview").classes("text-white text-lg font-bold mb-4")
            
            with ui.column().classes("gap-3"):
                # Address
                with ui.row().classes("justify-between items-center"):
                    ui.label("Address").classes("text-gray-400")
                    self.address_label = ui.label("—").classes("text-sm text-gray-300 font-mono")
                
                # Network
                with ui.row().classes("justify-between items-center"):
                    ui.label("Network").classes("text-gray-400")
                    self.network_label = ui.label("—").classes("text-sm text-gray-300")
                
                # Available balance
                with ui.row().classes("justify-between items-center"):
                    ui.label("Available").classes("text-gray-400")
                    self.available_label = ui.label("0.0000 SIBR").classes("stat-value")
    
    def _render_staking_card(self) -> None:
        """Render staking management card."""
        with ui.element("div").classes("wallet-card"):
            ui.label("Staking").classes("text-white text-lg font-bold mb-4")
            
            with ui.column().classes("gap-3"):
                # Staked balance
                with ui.row().classes("justify-between items-center"):
                    ui.label("Staked").classes("text-gray-400")
                    self.staked_label = ui.label("0.0000 SIBR").classes("stat-value")
                
                # Pending unstake
                with ui.row().classes("justify-between items-center"):
                    ui.label("Pending Unstake").classes("text-gray-400")
                    self.pending_label = ui.label("0.0000 SIBR").classes("text-yellow-400")
                
                # Unlock time
                with ui.row().classes("justify-between items-center"):
                    ui.label("Unlock Time").classes("text-gray-400")
                    self.unlock_label = ui.label("—").classes("text-gray-300 text-sm")
                
                ui.separator().classes("my-2")
                
                # Deposit input
                with ui.row().classes("w-full gap-2 items-end"):
                    self.deposit_input = ui.input(
                        "Amount",
                        placeholder="100.0"
                    ).classes("flex-1").props("dense outlined dark")
                    
                    ui.button(
                        "Deposit",
                        on_click=self._on_deposit
                    ).classes("action-btn")
                
                # Withdraw input (for available staked balance)
                with ui.row().classes("w-full gap-2 items-end"):
                    self.withdraw_input = ui.input(
                        "Amount",
                        placeholder="50.0"
                    ).classes("flex-1").props("dense outlined dark")
                    
                    ui.button(
                        "Withdraw",
                        on_click=self._on_withdraw_balance
                    ).classes("action-btn warning-btn")
    
    def _render_contract_card(self) -> None:
        """Render contract info card."""
        with ui.element("div").classes("wallet-card"):
            ui.label("Settlement Contract").classes("text-white text-lg font-bold mb-4")
            
            with ui.column().classes("gap-3"):
                # Contract address
                with ui.row().classes("justify-between items-center"):
                    ui.label("Address").classes("text-gray-400")
                    self.contract_link = ui.link(
                        "—",
                        target="_blank"
                    ).classes("contract-link text-sm font-mono")
                
                # Total staked
                with ui.row().classes("justify-between items-center"):
                    ui.label("Total Staked").classes("text-gray-400")
                    self.total_staked_label = ui.label("—").classes("text-gray-300")
                
                # Fee rate
                with ui.row().classes("justify-between items-center"):
                    ui.label("Fee Rate").classes("text-gray-400")
                    self.fee_rate_label = ui.label("0.5%").classes("text-gray-300")
                
                # Unstake delay
                with ui.row().classes("justify-between items-center"):
                    ui.label("Unstake Delay").classes("text-gray-400")
                    self.unstake_delay_label = ui.label("7 days").classes("text-gray-300")
    
    def _render_actions_card(self) -> None:
        """Render quick actions card."""
        with ui.element("div").classes("wallet-card"):
            ui.label("Quick Actions").classes("text-white text-lg font-bold mb-4")
            
            with ui.column().classes("gap-3 w-full"):
                # Request unstake
                ui.button(
                    "Request Full Unstake",
                    on_click=self._on_request_unstake,
                    icon="schedule"
                ).classes("w-full").props("outline dark")
                
                # Complete withdrawal
                ui.button(
                    "Complete Withdrawal",
                    on_click=self._on_complete_withdraw,
                    icon="download"
                ).classes("w-full").props("outline dark")
                
                # Refresh
                ui.button(
                    "Refresh",
                    on_click=self._on_refresh,
                    icon="refresh"
                ).classes("w-full").props("outline dark")
    
    # ========================================================================
    # Event Handlers
    # ========================================================================
    
    async def _initial_connect(self) -> None:
        """Initial connection on page load."""
        await self._connect_wallet()
    
    async def _connect_wallet(self) -> None:
        """Connect to wallet."""
        self._show_notification("Connecting to Siberium...", "info")
        
        success = await wallet.connect()
        
        if success:
            self._show_notification("Connected successfully", "positive")
            await self._update_ui()
        else:
            self._show_error(wallet.state.error)
    
    async def _on_deposit(self) -> None:
        """Handle deposit button click."""
        try:
            amount_str = self.deposit_input.value
            if not amount_str:
                self._show_notification("Please enter an amount", "warning")
                return
            
            amount = Decimal(amount_str)
            if amount <= 0:
                self._show_notification("Amount must be positive", "warning")
                return
            
            if amount > wallet.state.balance:
                self._show_notification("Insufficient balance", "warning")
                return
            
            self._show_notification(f"Depositing {amount} SIBR...", "info")
            
            tx_hash = await wallet.deposit(amount)
            
            self._show_notification(
                f"Deposit successful! TX: {tx_hash[:16]}...",
                "positive"
            )
            
            self.deposit_input.value = ""
            await self._refresh_data()
            
        except Exception as e:
            self._show_error(f"Deposit failed: {e}")
    
    async def _on_withdraw_balance(self) -> None:
        """Handle withdraw balance button click."""
        try:
            amount_str = self.withdraw_input.value
            if not amount_str:
                self._show_notification("Please enter an amount", "warning")
                return
            
            amount = Decimal(amount_str)
            if amount <= 0:
                self._show_notification("Amount must be positive", "warning")
                return
            
            if amount > wallet.state.staked:
                self._show_notification("Insufficient staked balance", "warning")
                return
            
            self._show_notification(f"Withdrawing {amount} SIBR...", "info")
            
            tx_hash = await wallet.withdraw_balance(amount)
            
            self._show_notification(
                f"Withdrawal successful! TX: {tx_hash[:16]}...",
                "positive"
            )
            
            self.withdraw_input.value = ""
            await self._refresh_data()
            
        except Exception as e:
            self._show_error(f"Withdrawal failed: {e}")
    
    async def _on_request_unstake(self) -> None:
        """Request full unstake."""
        try:
            if wallet.state.staked <= 0:
                self._show_notification("No staked balance", "warning")
                return
            
            self._show_notification("Requesting unstake...", "info")
            
            tx_hash = await wallet.request_unstake(wallet.state.staked)
            
            self._show_notification(
                f"Unstake requested! 7-day lock started. TX: {tx_hash[:16]}...",
                "positive"
            )
            
            await self._refresh_data()
            
        except Exception as e:
            self._show_error(f"Unstake request failed: {e}")
    
    async def _on_complete_withdraw(self) -> None:
        """Complete pending withdrawal."""
        try:
            if wallet.state.pending_unstake <= 0:
                self._show_notification("No pending unstake", "warning")
                return
            
            import time
            if wallet.state.unlock_time > time.time():
                remaining = int(wallet.state.unlock_time - time.time())
                hours = remaining // 3600
                self._show_notification(
                    f"Still locked. {hours}h remaining.",
                    "warning"
                )
                return
            
            self._show_notification("Completing withdrawal...", "info")
            
            tx_hash = await wallet.withdraw()
            
            self._show_notification(
                f"Withdrawal complete! TX: {tx_hash[:16]}...",
                "positive"
            )
            
            await self._refresh_data()
            
        except Exception as e:
            self._show_error(f"Withdrawal failed: {e}")
    
    async def _on_refresh(self) -> None:
        """Manual refresh."""
        await self._refresh_data()
        self._show_notification("Refreshed", "info")
    
    # ========================================================================
    # UI Update Methods
    # ========================================================================
    
    async def _refresh_data(self) -> None:
        """Refresh data from blockchain."""
        await wallet.refresh()
        await self._update_ui()
    
    async def _update_ui(self) -> None:
        """Update all UI elements with current state."""
        state = wallet.state
        
        # Update header
        self.balance_label.text = format_sibr(state.balance)
        
        # Update status indicator
        self.status_container.clear()
        with self.status_container:
            create_status_indicator(state.connected, state.block_number)
        
        # Update account info
        if state.address:
            short_addr = f"{state.address[:8]}...{state.address[-6:]}"
            self.address_label.text = short_addr
        self.network_label.text = state.network_name
        self.available_label.text = format_sibr(state.balance)
        
        # Update staking info
        self.staked_label.text = format_sibr(state.staked)
        self.pending_label.text = format_sibr(state.pending_unstake)
        
        if state.unlock_time > 0:
            import time
            if state.unlock_time > time.time():
                remaining = int(state.unlock_time - time.time())
                days = remaining // 86400
                hours = (remaining % 86400) // 3600
                self.unlock_label.text = f"{days}d {hours}h remaining"
            else:
                self.unlock_label.text = "Ready to withdraw"
        else:
            self.unlock_label.text = "—"
        
        # Update contract info
        if state.contract_address:
            short_contract = f"{state.contract_address[:10]}...{state.contract_address[-8:]}"
            explorer_url = wallet.get_explorer_url(state.contract_address)
            self.contract_link.text = short_contract
            if explorer_url:
                self.contract_link._props["href"] = explorer_url
        
        # Update error display
        if state.error:
            self._show_error(state.error)
        else:
            self.error_label.set_visibility(False)
    
    def _show_notification(self, message: str, type: str = "info") -> None:
        """Show notification toast."""
        ui.notify(message, type=type, position="top-right")
    
    def _show_error(self, message: str) -> None:
        """Show error message."""
        self.error_label.text = message
        self.error_label.set_visibility(True)
        ui.notify(message, type="negative", position="top-right")
    
    def _start_refresh_timer(self) -> None:
        """Start auto-refresh timer."""
        async def auto_refresh():
            if wallet.state.connected:
                try:
                    await wallet.refresh()
                    await self._update_ui()
                except Exception as e:
                    logger.warning(f"Auto-refresh failed: {e}")
        
        # Refresh every 15 seconds
        self._refresh_timer = ui.timer(15.0, auto_refresh)


# ============================================================================
# Standalone Demo
# ============================================================================

def demo():
    """Run standalone wallet demo."""
    from dotenv import load_dotenv
    load_dotenv()
    
    @ui.page("/")
    def main_page():
        ui.dark_mode().enable()
        
        with ui.header().classes("bg-gray-900 border-b border-gray-700"):
            ui.label("ZEONE Network").classes("text-xl font-bold text-white")
        
        with ui.tabs().classes("w-full bg-gray-800") as tabs:
            ui.tab("wallet", label="Wallet", icon="account_balance_wallet")
            ui.tab("network", label="Network", icon="hub")
        
        with ui.tab_panels(tabs).classes("w-full"):
            with ui.tab_panel("wallet"):
                WalletTab().render()
            with ui.tab_panel("network"):
                ui.label("Network tab coming soon...").classes("text-gray-400")
    
    ui.run(
        title="ZEONE Wallet",
        port=8080,
        dark=True,
        reload=False,
    )


if __name__ == "__main__":
    demo()

