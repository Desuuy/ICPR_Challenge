"""Models module containing network architectures."""
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR

from src.models.components import (
    AttentionFusion,
    CNNBackbone,
    ResNetFeatureExtractor,
    PositionalEncoding,
)

__all__ = [
    "MultiFrameCRNN",
    "ResTranOCR",
    "AttentionFusion",
    "CNNBackbone",
    "ResNetFeatureExtractor",
    "PositionalEncoding",
]

# NOTE: MultiFrameSVTRv2 không được import ở đây để tránh circular import.
# Nó nằm ở src.mf_svtrv2 và được import trực tiếp trong train.py:
#   from src.mf_svtrv2 import MultiFrameSVTRv2
