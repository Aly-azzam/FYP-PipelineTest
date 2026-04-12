from dataclasses import dataclass
from typing import Optional

@dataclass
class VJepaConfig:
    hub_repo: str = "facebookresearch/vjepa2"
    model_name: str = "vjepa2_1_vit_base_384"
    frames_per_clip: int = 30
    device: str = "cuda"
    checkpoint_path: Optional[str] = None
    checkpoint_url: Optional[str] = None
    num_segments: int = 8
    debug_export_raw_embeddings: bool = False