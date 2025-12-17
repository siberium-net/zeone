"""
Chainsmith / Chain Weaver Agent
===============================
Utilities for compiling and deploying simple smart contracts.

[SECURITY]
Deployment can require explicit human approval via HumanLinkAgent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


CONTRACT_TEMPLATES: Dict[str, str] = {
    "event_logger": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EventLogger {
    event LogEvent(address indexed sender, string message, uint256 timestamp);

    function log(string calldata message) external {
        emit LogEvent(msg.sender, message, block.timestamp);
    }
}
""",
    "simple_escrow": """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleEscrow {
    address public payer;
    address public payee;
    uint256 public amount;
    bool public released;

    constructor(address _payee) payable {
        payer = msg.sender;
        payee = _payee;
        amount = msg.value;
    }

    function release() external {
        require(msg.sender == payer, "Only payer");
        require(!released, "Already released");
        released = true;
        payable(payee).transfer(amount);
    }

    function refund() external {
        require(msg.sender == payee, "Only payee");
        require(!released, "Already released");
        released = true;
        payable(payer).transfer(amount);
    }
}
""",
}


class ChainWeaverAgent:
    """
    Blockchain automation helper.

    `chain_manager` is expected to be compatible with `economy.chain.SiberiumManager`
    (i.e., exposes `w3`, `private_key`, and `address`) or implement `deploy_contract`
    / `call_contract`.
    """

    def __init__(self, chain_manager: Any = None, human_link: Any = None):
        self.chain_manager = chain_manager
        self.human_link = human_link
        self._compiled_cache: Dict[str, Dict[str, Any]] = {}

    async def compile_contract(
        self,
        template_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Compile a Solidity contract template.

        Returns:
            {"bytecode": "0x...", "abi": [...], "source": "..."} or None.
        """
        if template_name in self._compiled_cache:
            return self._compiled_cache[template_name]

        source = CONTRACT_TEMPLATES.get(template_name)
        if not source:
            logger.error("[CHAINSMITH] Unknown template: %s", template_name)
            return None

        if params:
            for key, value in params.items():
                source = source.replace(f"${{{key}}}", str(value))

        try:
            import solcx  # type: ignore
        except ImportError:
            logger.error("[CHAINSMITH] solcx not installed: pip install py-solc-x>=2.0")
            return None

        try:
            installed = solcx.get_installed_solc_versions()
            if not installed:
                logger.error(
                    "[CHAINSMITH] No solc installed. Install one with: "
                    "python -c \"import solcx; solcx.install_solc('0.8.19')\""
                )
                return None

            # Pick latest installed version for determinism.
            solcx.set_solc_version(max(installed))

            compiled = solcx.compile_source(source, output_values=["abi", "bin"])
            contract_id = next(iter(compiled.keys()))
            contract = compiled[contract_id]

            result = {
                "bytecode": "0x" + (contract.get("bin") or ""),
                "abi": contract.get("abi") or [],
                "source": source,
                "contract_id": contract_id,
            }
            self._compiled_cache[template_name] = result
            return result
        except Exception as e:
            logger.error("[CHAINSMITH] Compilation failed: %s", e)
            return None

    async def deploy_contract(
        self,
        bytecode: str,
        abi: list,
        constructor_args: Optional[list] = None,
        require_human_approval: bool = True,
        gas: int = 5_000_000,
        value_wei: int = 0,
    ) -> Optional[str]:
        """
        Deploy a contract.

        Returns deployed address or None.
        """
        if not self.chain_manager:
            logger.error("[CHAINSMITH] No chain_manager configured")
            return None

        if require_human_approval and self.human_link:
            response = await self.human_link.ask_human(
                "Deploy contract?\n"
                f"Estimated gas: {gas}\n"
                f"Constructor args: {constructor_args or []}\n"
                f"Value (wei): {value_wei}",
                options=["Deploy", "Cancel"],
            )
            if response != "Deploy":
                logger.info("[CHAINSMITH] Deployment cancelled by user")
                return None

        # If the chain_manager exposes a deploy method, prefer it.
        deploy_fn = getattr(self.chain_manager, "deploy_contract", None)
        if callable(deploy_fn):
            try:
                return await deploy_fn(
                    bytecode=bytecode,
                    abi=abi,
                    constructor_args=constructor_args or [],
                    gas=gas,
                    value_wei=value_wei,
                )
            except TypeError:
                # Backward/alternate signatures.
                return await deploy_fn(bytecode=bytecode, abi=abi, constructor_args=constructor_args or [])
            except Exception as e:
                logger.error("[CHAINSMITH] chain_manager.deploy_contract failed: %s", e)
                return None

        # Generic Web3 deployment (SiberiumManager-compatible).
        try:
            from web3 import Web3  # type: ignore
        except Exception as e:
            logger.error("[CHAINSMITH] web3 not available: %s", e)
            return None

        w3 = getattr(self.chain_manager, "w3", None)
        private_key = getattr(self.chain_manager, "private_key", "") or ""
        sender = getattr(self.chain_manager, "address", None)
        if not w3 or not private_key or not sender:
            logger.error("[CHAINSMITH] chain_manager missing w3/private_key/address")
            return None

        try:
            sender = Web3.to_checksum_address(sender)
            contract = w3.eth.contract(abi=abi, bytecode=bytecode)
            construct = contract.constructor(*(constructor_args or []))
            tx = construct.build_transaction(
                {
                    "from": sender,
                    "nonce": w3.eth.get_transaction_count(sender),
                    "gas": int(gas),
                    "gasPrice": w3.eth.gas_price,
                    "value": int(value_wei),
                }
            )
            signed = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            if getattr(receipt, "status", 0) != 1:
                logger.error("[CHAINSMITH] Deployment failed: %s", receipt)
                return None
            address = receipt.contractAddress
            logger.info("[CHAINSMITH] Contract deployed: %s", address)
            return address
        except Exception as e:
            logger.error("[CHAINSMITH] Deployment failed: %s", e)
            return None

    async def call_contract(
        self,
        address: str,
        abi: list,
        method: str,
        args: Optional[list] = None,
        value_wei: int = 0,
        gas: int = 500_000,
    ) -> Any:
        """Call or transact against a deployed contract method."""
        if not self.chain_manager:
            logger.error("[CHAINSMITH] No chain_manager configured")
            return None

        call_fn = getattr(self.chain_manager, "call_contract", None)
        if callable(call_fn):
            try:
                return await call_fn(
                    address=address,
                    abi=abi,
                    method=method,
                    args=args or [],
                    value=value_wei,
                    gas=gas,
                )
            except TypeError:
                return await call_fn(address=address, abi=abi, method=method, args=args or [], value=value_wei)
            except Exception as e:
                logger.error("[CHAINSMITH] chain_manager.call_contract failed: %s", e)
                return None

        try:
            from web3 import Web3  # type: ignore
        except Exception as e:
            logger.error("[CHAINSMITH] web3 not available: %s", e)
            return None

        w3 = getattr(self.chain_manager, "w3", None)
        private_key = getattr(self.chain_manager, "private_key", "") or ""
        sender = getattr(self.chain_manager, "address", None)
        if not w3 or not sender:
            logger.error("[CHAINSMITH] chain_manager missing w3/address")
            return None

        sender = Web3.to_checksum_address(sender)
        contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)
        fn = getattr(contract.functions, method)(*(args or []))

        mutability = _get_function_mutability(abi, method) or ""
        if mutability in {"view", "pure"}:
            try:
                return fn.call({"from": sender})
            except Exception as e:
                logger.error("[CHAINSMITH] Call failed: %s", e)
                return None

        if not private_key:
            logger.error("[CHAINSMITH] private_key required for transactions")
            return None

        try:
            tx = fn.build_transaction(
                {
                    "from": sender,
                    "nonce": w3.eth.get_transaction_count(sender),
                    "gas": int(gas),
                    "gasPrice": w3.eth.gas_price,
                    "value": int(value_wei),
                }
            )
            signed = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            return {"tx_hash": tx_hash.hex(), "status": getattr(receipt, "status", 0), "receipt": receipt}
        except Exception as e:
            logger.error("[CHAINSMITH] Transaction failed: %s", e)
            return None

    def get_available_templates(self) -> List[str]:
        return list(CONTRACT_TEMPLATES.keys())


def _get_function_mutability(abi: list, method: str) -> Optional[str]:
    for entry in abi or []:
        if isinstance(entry, dict) and entry.get("type") == "function" and entry.get("name") == method:
            return entry.get("stateMutability") or entry.get("mutability")
    return None


_chain_weaver: Optional[ChainWeaverAgent] = None


def get_chain_weaver(chain_manager: Any = None, human_link: Any = None) -> ChainWeaverAgent:
    global _chain_weaver
    if _chain_weaver is None:
        _chain_weaver = ChainWeaverAgent(chain_manager=chain_manager, human_link=human_link)
    return _chain_weaver

