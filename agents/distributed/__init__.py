"""
Distributed AI Inference Module
===============================

Распределённый инференс больших моделей через P2P сеть.

[ARCHITECTURE]
- Pipeline Parallelism: модель разбита на shards по слоям
- Каждый узел обслуживает свой shard
- Активации передаются между узлами в quantized int8

[COMPONENTS]
- ModelRegistry: DHT реестр доступных model shards
- ModelShard: Загрузка и инференс части модели
- InferenceWorker: Сервис на GPU узле
- PipelineCoordinator: Координация pipeline
- DistributedClient: Клиент для запросов

[FLOW]
1. Client tokenizes input
2. Client looks up model shards in DHT
3. Client sends embeddings to first shard
4. Each shard processes and forwards to next
5. Last shard returns logits to client
6. Client detokenizes output

[FAULT TOLERANCE]
- Multiple nodes can serve same shard
- Dynamic switching if node fails
- Health checks via DHT
"""

from .registry import (
    ModelRegistry,
    ModelShardInfo,
    ModelInfo,
)

from .protocol import (
    DistributedMessageType,
    ForwardRequest,
    ForwardResponse,
    ShardHealthCheck,
)

from .shard import (
    ModelShard,
    ActivationBuffer,
    quantize_activations,
    dequantize_activations,
)

from .worker import (
    InferenceWorker,
    WorkerState,
)

from .pipeline import (
    PipelineCoordinator,
    PipelineState,
)

from .client import (
    DistributedInferenceClient,
    InferenceResult,
)

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelShardInfo",
    "ModelInfo",
    # Protocol
    "DistributedMessageType",
    "ForwardRequest",
    "ForwardResponse",
    "ShardHealthCheck",
    # Shard
    "ModelShard",
    "ActivationBuffer",
    "quantize_activations",
    "dequantize_activations",
    # Worker
    "InferenceWorker",
    "WorkerState",
    # Pipeline
    "PipelineCoordinator",
    "PipelineState",
    # Client
    "DistributedInferenceClient",
    "InferenceResult",
]

