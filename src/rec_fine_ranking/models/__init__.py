"""Model package: shared backbone + seven ranking architectures.

`MODEL_REGISTRY` maps the CLI/config model name to its class so the trainer can
instantiate any model via `MODEL_REGISTRY[name]()`.
"""
from .base import BaseRanker, FeatureEncoder
from .wide_deep import WideDeep
from .dcn import DCN
from .dcn_v2 import DCNv2
from .deepfm import DeepFM
from .din import DIN
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
    "din": DIN,
    "dcn_v2": DCNv2,
    "onetrans": OneTrans,
    "rankmixer": RankMixer,
    "unimixer": UniMixer,
    "hyformer": HyFormer,
}

__all__ = [
    "BaseRanker", "FeatureEncoder", "MODEL_REGISTRY",
    "WideDeep", "DCN", "DCNv2", "DeepFM", "DeepFMField", "DIN", "OneTrans",
    "RankMixer", "UniMixer", "HyFormer",
]
