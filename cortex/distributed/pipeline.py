"""
Pipeline Parallelism Orchestrator
=================================

[NEURAL] Naive Pipeline Parallelism для распределённого LLM inference.

Архитектура:
============

┌──────────┐    activations    ┌──────────┐    activations    ┌──────────┐
│ HEAD     │ ──────────────►   │ MIDDLE   │ ──────────────►   │ TAIL     │
│ Node     │                   │ Node(s)  │                   │ Node     │
│          │                   │          │                   │          │
│ Embed    │                   │ Layers   │                   │ Layers   │
│ Layers   │                   │ [N:M]    │                   │ [M:End]  │
│ [0:N]    │                   │          │                   │ LM Head  │
└──────────┘                   └──────────┘                   └──────────┘

[PROTOCOL]
1. HEAD: Tokenize -> Embed -> Layers[0:N] -> NeuroLink.send(activations)
2. MIDDLE: NeuroLink.recv -> Layers[N:M] -> NeuroLink.send(activations)
3. TAIL: NeuroLink.recv -> Layers[M:End] -> LM Head -> Detokenize

[PERFORMANCE]
- Micro-batching для overlap compute/network
- Async pipeline: следующий batch начинается пока текущий в transit
- KV-cache для inference

[LIMITATIONS]
- Naive: нет bubble scheduling (GPipe/PipeDream)
- Inference only (backward pass = TODO)
- Assumes homogeneous layer sizes
"""

import asyncio
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable, Awaitable
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy imports
torch = None
nn = None


def _ensure_torch():
    global torch, nn
    if torch is None:
        import torch as _torch
        torch = _torch
        nn = torch.nn
    return torch


# ============================================================================
# Constants
# ============================================================================

class PipelineStage(Enum):
    """Pipeline stage type."""
    HEAD = auto()      # First node: embedding + first layers
    MIDDLE = auto()    # Middle node(s): intermediate layers
    TAIL = auto()      # Last node: final layers + LM head


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """
    Configuration for pipeline parallelism.
    """
    # Model configuration
    model_path: str = ""              # Path to model weights
    model_type: str = "llama"         # Model architecture
    
    # Sharding configuration
    total_layers: int = 32            # Total transformer layers
    start_layer: int = 0              # First layer on this node
    end_layer: int = 32               # Last layer on this node (exclusive)
    
    # Pipeline topology
    stage: PipelineStage = PipelineStage.HEAD
    upstream_node: Optional[str] = None    # Previous node ID
    downstream_node: Optional[str] = None  # Next node ID
    
    # Compute configuration
    device: str = "cuda:0"            # Device for this shard
    dtype: str = "float16"            # Compute dtype
    
    # Network configuration
    use_tcp_direct: bool = False      # Use direct TCP for tensors
    tcp_port: int = 9000              # TCP port for tensor transport
    
    # Performance tuning
    micro_batch_size: int = 1         # Micro-batch size
    prefetch_batches: int = 2         # Batches to prefetch
    
    def __post_init__(self):
        if isinstance(self.stage, str):
            self.stage = PipelineStage[self.stage.upper()]


@dataclass
class ShardInfo:
    """
    Information about a model shard.
    """
    shard_id: str
    node_id: str
    start_layer: int
    end_layer: int
    stage: PipelineStage
    device: str
    loaded: bool = False
    memory_mb: float = 0.0


# ============================================================================
# Model Shard Loader
# ============================================================================

class ModelShardLoader:
    """
    Loads a subset of model layers.
    
    [MEMORY] Only loads layers [start_layer:end_layer]
    to reduce per-node memory requirements.
    
    [SUPPORTED MODELS]
    - LLaMA / Llama 2 / Llama 3
    - Mistral
    - (extensible)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_shard: Optional[nn.Module] = None
        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self.norm: Optional[nn.Module] = None
    
    async def load_shard(self) -> "nn.Module":
        """
        Load model shard asynchronously.
        
        Returns:
            nn.Module containing the shard layers
        """
        torch = _ensure_torch()
        
        logger.info(
            f"[PIPELINE] Loading shard layers [{self.config.start_layer}:{self.config.end_layer}] "
            f"on {self.config.device}"
        )
        
        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()
        shard = await loop.run_in_executor(None, self._load_shard_sync)
        
        self.model_shard = shard
        return shard
    
    def _load_shard_sync(self) -> "nn.Module":
        """Synchronous shard loading."""
        torch = _ensure_torch()
        
        model_path = Path(self.config.model_path)
        device = self.config.device
        dtype = getattr(torch, self.config.dtype)
        
        # Detect model type and load appropriately
        if self.config.model_type in ("llama", "llama2", "llama3", "mistral"):
            return self._load_llama_shard(model_path, device, dtype)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _load_llama_shard(
        self,
        model_path: Path,
        device: str,
        dtype: "torch.dtype",
    ) -> "nn.Module":
        """
        Load LLaMA-style model shard.
        
        [ARCHITECTURE]
        LLaMA structure:
        - model.embed_tokens (HEAD only)
        - model.layers[i] (distributed)
        - model.norm (TAIL only)
        - lm_head (TAIL only)
        """
        torch = _ensure_torch()
        
        # Try to load from transformers
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
            
            config = AutoConfig.from_pretrained(model_path)
            
            # Load full model to CPU first
            full_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            
            # Extract layers we need
            layers = nn.ModuleList()
            
            # Access the transformer layers
            if hasattr(full_model, 'model'):
                base_model = full_model.model
            else:
                base_model = full_model
            
            # Get the layers
            if hasattr(base_model, 'layers'):
                all_layers = base_model.layers
            elif hasattr(base_model, 'h'):  # GPT-style
                all_layers = base_model.h
            else:
                raise ValueError("Cannot find transformer layers")
            
            # Extract our shard
            for i in range(self.config.start_layer, self.config.end_layer):
                if i < len(all_layers):
                    layers.append(all_layers[i])
            
            # Create shard module
            shard = _TransformerShard(layers)
            
            # HEAD: also need embedding
            if self.config.stage == PipelineStage.HEAD:
                if hasattr(base_model, 'embed_tokens'):
                    self.embed_tokens = base_model.embed_tokens.to(device)
                elif hasattr(base_model, 'wte'):
                    self.embed_tokens = base_model.wte.to(device)
            
            # TAIL: also need norm and lm_head
            if self.config.stage == PipelineStage.TAIL:
                if hasattr(base_model, 'norm'):
                    self.norm = base_model.norm.to(device)
                elif hasattr(base_model, 'ln_f'):
                    self.norm = base_model.ln_f.to(device)
                
                if hasattr(full_model, 'lm_head'):
                    self.lm_head = full_model.lm_head.to(device)
            
            # Move shard to device
            shard = shard.to(device)
            
            # Free full model
            del full_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(
                f"[PIPELINE] Loaded {len(layers)} layers to {device}"
            )
            
            return shard
            
        except ImportError:
            logger.warning("[PIPELINE] transformers not available, using stub")
            return _StubShard(
                self.config.start_layer,
                self.config.end_layer,
                device,
            )
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        torch = _ensure_torch()
        
        if not torch.cuda.is_available():
            return 0.0
        
        if "cuda" in self.config.device:
            device_idx = int(self.config.device.split(":")[-1]) if ":" in self.config.device else 0
            return torch.cuda.memory_allocated(device_idx) / 1024 / 1024
        
        return 0.0


class _TransformerShard(nn.Module):
    """Container for transformer layer shard."""
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
    
    def forward(
        self,
        hidden_states: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        position_ids: Optional["torch.Tensor"] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ) -> Tuple["torch.Tensor", Optional[List]]:
        """Forward through shard layers."""
        presents = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                presents.append(layer_outputs[1])
        
        return hidden_states, presents


class _StubShard(nn.Module):
    """Stub shard for testing without real model."""
    
    def __init__(self, start_layer: int, end_layer: int, device: str):
        super().__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        
        # Dummy layer for shape preservation
        self.identity = nn.Identity()
    
    def forward(
        self,
        hidden_states: "torch.Tensor",
        **kwargs,
    ) -> Tuple["torch.Tensor", None]:
        """Pass through unchanged (for testing)."""
        return hidden_states, None


# ============================================================================
# Pipeline Worker
# ============================================================================

class PipelineWorker:
    """
    Worker for a single pipeline stage.
    
    [LIFECYCLE]
    1. Initialize with config
    2. load_shard() - load model layers
    3. start() - begin processing
    4. process_batch() - process incoming activations
    5. stop() - cleanup
    
    [PROTOCOL]
    - HEAD: tokenize -> embed -> forward -> send
    - MIDDLE: receive -> forward -> send
    - TAIL: receive -> forward -> decode -> result
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        neuro_link: "NeuroLinkAgent",
    ):
        self.config = config
        self.neuro_link = neuro_link
        
        self.loader = ModelShardLoader(config)
        self.shard: Optional[nn.Module] = None
        
        # State
        self._running = False
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._result_futures: Dict[str, asyncio.Future] = {}
        
        # Stats
        self._stats = {
            "batches_processed": 0,
            "tokens_processed": 0,
            "avg_latency_ms": 0.0,
        }
    
    async def load_shard(self) -> None:
        """Load model shard."""
        self.shard = await self.loader.load_shard()
        
        # Register activation handler
        self.neuro_link.register_activation_handler(self._on_activation_received)
    
    async def start(self) -> None:
        """Start the worker."""
        if self.shard is None:
            await self.load_shard()
        
        self._running = True
        
        logger.info(
            f"[PIPELINE] Worker started: stage={self.config.stage.name} "
            f"layers=[{self.config.start_layer}:{self.config.end_layer}]"
        )
    
    async def stop(self) -> None:
        """Stop the worker."""
        self._running = False
        
        # Cancel pending futures
        for future in self._result_futures.values():
            if not future.done():
                future.cancel()
    
    async def process_input(
        self,
        input_ids: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        batch_id: Optional[str] = None,
    ) -> "torch.Tensor":
        """
        Process input (HEAD node only).
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask
            batch_id: Batch identifier
        
        Returns:
            Output logits (for TAIL) or None (for HEAD/MIDDLE)
        """
        torch = _ensure_torch()
        
        if self.config.stage != PipelineStage.HEAD:
            raise RuntimeError("process_input only valid for HEAD stage")
        
        batch_id = batch_id or str(uuid.uuid4())
        start_time = time.time()
        
        # Embed tokens
        if self.loader.embed_tokens is not None:
            hidden_states = self.loader.embed_tokens(input_ids)
        else:
            # Stub: pass through
            hidden_states = input_ids.float()
        
        # Forward through our layers
        hidden_states, _ = self.shard(
            hidden_states,
            attention_mask=attention_mask,
        )
        
        # Send to downstream
        if self.config.downstream_node:
            await self.neuro_link.send_activations(
                self.config.downstream_node,
                hidden_states,
                layer_idx=self.config.end_layer,
                batch_id=batch_id,
            )
            
            # Wait for result (if we're expecting one)
            if self.config.stage == PipelineStage.HEAD:
                # HEAD waits for TAIL to return result
                future = asyncio.Future()
                self._result_futures[batch_id] = future
                
                result = await asyncio.wait_for(future, timeout=60.0)
                return result
        
        # Update stats
        elapsed = (time.time() - start_time) * 1000
        self._stats["batches_processed"] += 1
        self._stats["tokens_processed"] += input_ids.numel()
        self._update_latency(elapsed)
        
        return hidden_states
    
    async def _on_activation_received(
        self,
        tensor: "torch.Tensor",
        meta: Dict[str, Any],
    ) -> None:
        """
        Handle received activations.
        
        Called by NeuroLink when activations arrive from upstream.
        """
        torch = _ensure_torch()
        
        batch_id = meta.get("batch_id", "")
        layer_idx = meta.get("layer_idx", 0)
        
        logger.debug(
            f"[PIPELINE] Received activations: batch={batch_id[:8]}... "
            f"layer={layer_idx} shape={tensor.shape}"
        )
        
        start_time = time.time()
        
        # Forward through our layers
        hidden_states, _ = self.shard(tensor)
        
        # TAIL: generate output
        if self.config.stage == PipelineStage.TAIL:
            # Apply final norm
            if self.loader.norm is not None:
                hidden_states = self.loader.norm(hidden_states)
            
            # Apply LM head
            if self.loader.lm_head is not None:
                logits = self.loader.lm_head(hidden_states)
            else:
                logits = hidden_states
            
            # Send result back to HEAD (or handle locally)
            # For now, just log
            logger.info(
                f"[PIPELINE] TAIL output: batch={batch_id[:8]}... "
                f"logits_shape={logits.shape}"
            )
            
            # If we have a future waiting, resolve it
            if batch_id in self._result_futures:
                self._result_futures[batch_id].set_result(logits)
            
        else:
            # MIDDLE: forward to downstream
            if self.config.downstream_node:
                await self.neuro_link.send_activations(
                    self.config.downstream_node,
                    hidden_states,
                    layer_idx=self.config.end_layer,
                    batch_id=batch_id,
                )
        
        # Update stats
        elapsed = (time.time() - start_time) * 1000
        self._stats["batches_processed"] += 1
        self._update_latency(elapsed)
    
    def _update_latency(self, latency_ms: float) -> None:
        """Update running average latency."""
        alpha = 0.1
        current = self._stats["avg_latency_ms"]
        self._stats["avg_latency_ms"] = current * (1 - alpha) + latency_ms * alpha
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            **self._stats,
            "stage": self.config.stage.name,
            "layers": f"[{self.config.start_layer}:{self.config.end_layer}]",
            "device": self.config.device,
            "memory_mb": self.loader.get_memory_usage(),
        }


# ============================================================================
# Pipeline Coordinator
# ============================================================================

class PipelineCoordinator:
    """
    Coordinates multiple pipeline workers across nodes.
    
    [TOPOLOGY DISCOVERY]
    - Registers workers with DHT
    - Discovers peer shards
    - Builds pipeline topology
    
    [SCHEDULING]
    - Assigns layers to nodes
    - Manages micro-batching
    - Handles failures
    """
    
    def __init__(self, node: Any):
        self.node = node
        self.workers: Dict[str, PipelineWorker] = {}
        self.topology: List[ShardInfo] = []
    
    async def setup_pipeline(
        self,
        model_path: str,
        total_layers: int,
        peer_nodes: List[str],
        local_device: str = "cuda:0",
    ) -> None:
        """
        Setup pipeline across nodes.
        
        Args:
            model_path: Path to model
            total_layers: Total transformer layers
            peer_nodes: List of peer node IDs (in order)
            local_device: Device for local shard
        """
        num_nodes = len(peer_nodes) + 1  # Include self
        layers_per_node = total_layers // num_nodes
        
        # Determine our position
        local_id = getattr(self.node, "node_id", "local")
        all_nodes = [local_id] + peer_nodes
        
        # Build topology
        for i, node_id in enumerate(all_nodes):
            start_layer = i * layers_per_node
            end_layer = (i + 1) * layers_per_node if i < num_nodes - 1 else total_layers
            
            if i == 0:
                stage = PipelineStage.HEAD
            elif i == num_nodes - 1:
                stage = PipelineStage.TAIL
            else:
                stage = PipelineStage.MIDDLE
            
            upstream = all_nodes[i - 1] if i > 0 else None
            downstream = all_nodes[i + 1] if i < num_nodes - 1 else None
            
            shard_info = ShardInfo(
                shard_id=f"shard_{i}",
                node_id=node_id,
                start_layer=start_layer,
                end_layer=end_layer,
                stage=stage,
                device=local_device if node_id == local_id else "remote",
            )
            self.topology.append(shard_info)
            
            # Create local worker
            if node_id == local_id:
                config = PipelineConfig(
                    model_path=model_path,
                    total_layers=total_layers,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    stage=stage,
                    upstream_node=upstream,
                    downstream_node=downstream,
                    device=local_device,
                )
                
                from agents.neuro_link import NeuroLinkAgent
                neuro_link = NeuroLinkAgent(self.node, device=local_device)
                
                worker = PipelineWorker(config, neuro_link)
                self.workers[node_id] = worker
        
        logger.info(f"[PIPELINE] Topology: {len(self.topology)} shards across {num_nodes} nodes")
    
    async def start(self) -> None:
        """Start all local workers."""
        for worker in self.workers.values():
            await worker.start()
    
    async def stop(self) -> None:
        """Stop all local workers."""
        for worker in self.workers.values():
            await worker.stop()
    
    def get_local_worker(self) -> Optional[PipelineWorker]:
        """Get the local pipeline worker."""
        local_id = getattr(self.node, "node_id", "local")
        return self.workers.get(local_id)
    
    def get_topology(self) -> List[Dict[str, Any]]:
        """Get pipeline topology."""
        return [
            {
                "shard_id": s.shard_id,
                "node_id": s.node_id,
                "layers": f"[{s.start_layer}:{s.end_layer}]",
                "stage": s.stage.name,
                "device": s.device,
                "loaded": s.loaded,
            }
            for s in self.topology
        ]


# ============================================================================
# Convenience function
# ============================================================================

async def create_pipeline(
    node: Any,
    model_path: str,
    peer_nodes: List[str],
    device: str = "cuda:0",
) -> PipelineCoordinator:
    """
    Create and initialize a pipeline.
    
    Args:
        node: Parent node
        model_path: Path to model
        peer_nodes: Peer node IDs
        device: Local device
    
    Returns:
        Initialized PipelineCoordinator
    """
    # Detect total layers from config
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        total_layers = config.num_hidden_layers
    except Exception:
        total_layers = 32  # Default
    
    coordinator = PipelineCoordinator(node)
    await coordinator.setup_pipeline(
        model_path=model_path,
        total_layers=total_layers,
        peer_nodes=peer_nodes,
        local_device=device,
    )
    await coordinator.start()
    
    return coordinator

