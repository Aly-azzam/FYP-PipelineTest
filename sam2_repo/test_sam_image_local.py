import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "../checkpoints/sam2_hiera_tiny.pt"
model_cfg = "configs/sam2/sam2_hiera_t.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

predictor = SAM2ImagePredictor(
    build_sam2(model_cfg, checkpoint, device=device)
)

print("SAM image predictor chargé avec succès")