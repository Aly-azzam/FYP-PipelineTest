import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "configs/sam2/sam2_hiera_t.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = SAM2ImagePredictor(
    build_sam2(model_cfg, checkpoint, device=device)
)

def segment_image_from_point(image_bgr, x, y):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    best_mask = masks[np.argmax(scores)]
    return best_mask