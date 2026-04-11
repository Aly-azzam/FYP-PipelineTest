import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "../checkpoints/sam2_hiera_tiny.pt"
model_cfg = "configs/sam2/sam2_hiera_t.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

image = np.ones((400, 400, 3), dtype=np.uint8) * 255
cv2.rectangle(image, (120, 120), (280, 280), (0, 0, 255), -1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor = SAM2ImagePredictor(
    build_sam2(model_cfg, checkpoint, device=device)
)

predictor.set_image(image_rgb)

input_point = np.array([[200, 200]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

print("Nombre de masques :", len(masks))
print("Scores :", scores)
print("Meilleur score :", float(scores[np.argmax(scores)]))
best_mask = masks[np.argmax(scores)]
print("Taille du meilleur masque :", best_mask.shape)
print("Nombre de pixels du masque :", int(best_mask.sum()))
mask_uint8 = (best_mask * 255).astype(np.uint8)
cv2.imwrite("best_mask.png", mask_uint8)
overlay = image.copy()
overlay[best_mask.astype(bool)] = [0, 255, 0]
cv2.imwrite("overlay.png", overlay)