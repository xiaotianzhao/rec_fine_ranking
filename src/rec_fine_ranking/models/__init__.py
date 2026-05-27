"""Model package: shared backbone + seven ranking architectures.

`MODEL_REGISTRY` maps the CLI/config model name to its class so the trainer can
instantiate any model via `MODEL_REGISTRY[name]()`.
"""
from .base import BaseRanker, FeatureEncoder
from .wide_deep import WideDeep
from .dcn import DCN
from .deepfm import DeepFM
from .deepfm_field import DeepFMField
from .onetrans import OneTrans
from .rankmixer import RankMixer
from .unimixer import UniMixer
from .hyformer import HyFormer

MODEL_REGISTRY = {
    "wide_deep": WideDeep,
    "dcn": DCN,
    "deepfm": DeepFM,
    "deepfm_field": DeepFMField,
    "onetrans": OneTrans,
    "rankmixer": RankMixer,
    "unimixer": UniMixer,
    "hyformer": HyFormer,
}

__all__ = [
    "BaseRanker", "FeatureEncoder", "MODEL_REGISTRY",
    "WideDeep", "DCN", "DeepFM", "DeepFMField", "OneTrans",
    "RankMixer", "UniMixer", "HyFormer",
]
