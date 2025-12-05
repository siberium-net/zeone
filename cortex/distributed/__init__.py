"""
Distributed Neural Computation
==============================

[NEURAL] Модули для распределённого inference LLM через Pipeline Parallelism.

Components:
- codec: Tensor serialization/deserialization
- pipeline: Pipeline parallelism orchestration

[USAGE]
    from cortex.distributed.codec import encode_tensor, decode_tensor
    from cortex.distributed.pipeline import PipelineCoordinator, create_pipeline
"""

from .codec import (
    TensorEncoder,
    TensorDecoder,
    TensorChunker,
    TensorAssembler,
    TensorFrame,
    encode_tensor,
    decode_tensor,
    TENSOR_MAGIC,
    MAX_TENSOR_SIZE,
)

from .pipeline import (
    PipelineConfig,
    PipelineStage,
    PipelineWorker,
    PipelineCoordinator,
    ModelShardLoader,
    ShardInfo,
    create_pipeline,
)

__all__ = [
    # Codec
    "TensorEncoder",
    "TensorDecoder",
    "TensorChunker",
    "TensorAssembler",
    "TensorFrame",
    "encode_tensor",
    "decode_tensor",
    "TENSOR_MAGIC",
    "MAX_TENSOR_SIZE",
    # Pipeline
    "PipelineConfig",
    "PipelineStage",
    "PipelineWorker",
    "PipelineCoordinator",
    "ModelShardLoader",
    "ShardInfo",
    "create_pipeline",
]

