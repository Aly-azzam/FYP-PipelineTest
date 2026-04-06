"""
import gradio as gr
from compare_utils import compare_two_videos
from sam_image_utils import segment_image_from_point
import cv2
import os
import numpy as np

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        return None

    return frame

def save_first_frame(video_path, output_path):
    frame = extract_first_frame(video_path)
    if frame is None:
        return None

    cv2.imwrite(output_path, frame)
    return output_path

def prepare_preview_frames(expert_video, learner_video):
    os.makedirs("outputs", exist_ok=True)
    expert_preview_path = "outputs/expert_first_frame.jpg"
    learner_preview_path = "outputs/learner_first_frame.jpg"
    save_first_frame(expert_video, expert_preview_path)
    save_first_frame(learner_video, learner_preview_path)
    return expert_preview_path, learner_preview_path

def run_prepare_previews(expert_video, learner_video):
    return prepare_preview_frames(expert_video, learner_video)

def get_select_coords(evt: gr.SelectData):
    return evt.index

def run_sam_with_points(expert_video, learner_video, expert_point, learner_point):
    expert_frame = extract_first_frame(expert_video)
    learner_frame = extract_first_frame(learner_video)
    if expert_frame is None or learner_frame is None:
        return None, None, "Erreur lors de la lecture des vidéos"    
    if expert_point is None or learner_point is None:
        return None, None, "Clique sur les deux images d'abord"
    expert_x, expert_y = expert_point
    learner_x, learner_y = learner_point
    expert_mask = segment_image_from_point(expert_frame, expert_x, expert_y)
    learner_mask = segment_image_from_point(learner_frame, learner_x, learner_y)
    expert_overlay = mask_to_overlay(expert_frame, expert_mask)
    learner_overlay = mask_to_overlay(learner_frame, learner_mask)
    return expert_overlay, learner_overlay, "SAM appliqué avec les points choisis"




def mask_to_overlay(image_bgr, mask):
    overlay = image_bgr.copy()
    overlay[mask.astype(bool)] = [0, 255, 0]
    return overlay

def run_sam_on_first_frames(expert_video, learner_video):
    expert_frame = extract_first_frame(expert_video)
    learner_frame = extract_first_frame(learner_video)
    if expert_frame is None or learner_frame is None:
        return None, None, "Erreur lors de la lecture des vidéos"
    expert_mask = segment_image_from_point(expert_frame, expert_frame.shape[1] // 2, expert_frame.shape[0] // 2)
    learner_mask = segment_image_from_point(learner_frame, learner_frame.shape[1] // 2, learner_frame.shape[0] // 2)
    expert_overlay = mask_to_overlay(expert_frame, expert_mask)
    learner_overlay = mask_to_overlay(learner_frame, learner_mask)
    return expert_overlay, learner_overlay, "SAM appliqué sur la première frame des 2 vidéos"


def run_demo(expert_video, learner_video):
    compare_two_videos(expert_video, learner_video)
    return compare_two_videos(expert_video, learner_video)


demo = gr.Interface(
    fn=run_sam_with_points,
    inputs=[
        gr.Video(label="Vidéo expert"),
        gr.Video(label="Vidéo learner"),
        gr.State(),
        gr.State(),
    ],
    outputs=[
        gr.Image(label="Overlay expert"),
        gr.Image(label="Overlay learner"),
        gr.Textbox(label="Résultat"),
    ],
    title="Comparaison simple de 2 vidéos avec sélection de point",
)

demo.launch()
"""

import os
import cv2
import gradio as gr
from sam_image_utils import segment_image_from_point


def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        return None

    return frame


def save_first_frame(video_path, output_path):
    frame = extract_first_frame(video_path)
    if frame is None:
        return None

    cv2.imwrite(output_path, frame)
    return output_path


def mask_to_overlay(image_bgr, mask):
    overlay = image_bgr.copy()
    overlay[mask.astype(bool)] = [0, 255, 0]
    return overlay

def mask_area(mask):
    return int(mask.astype(bool).sum())

def compare_mask_areas(expert_mask, learner_mask):
    expert_area = mask_area(expert_mask)
    learner_area = mask_area(learner_mask)
    difference = abs(expert_area - learner_area)
    return f"Surface expert = {expert_area} | Surface learner = {learner_area} | Différence = {difference}"

def mask_centroid(mask):
    ys, xs = mask.astype(bool).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.mean()), int(ys.mean())



def compare_mask_centroids(expert_mask, learner_mask):
    expert_center = mask_centroid(expert_mask)
    learner_center = mask_centroid(learner_mask)
    if expert_center is None or learner_center is None:
        return "Centre introuvable"

    expert_x, expert_y = expert_center
    learner_x, learner_y = learner_center

    dx = learner_x - expert_x
    dy = learner_y - expert_y

    return f"Centre expert = ({expert_x}, {expert_y}) | Centre learner = ({learner_x}, {learner_y}) | Décalage = ({dx}, {dy})"

def mask_bbox(mask):
    ys, xs = mask.astype(bool).nonzero()

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())

    return x_min, y_min, x_max, y_max


def compare_mask_bboxes(expert_mask, learner_mask):
    expert_bbox = mask_bbox(expert_mask)
    learner_bbox = mask_bbox(learner_mask)

    if expert_bbox is None or learner_bbox is None:
        return "Boîte introuvable"

    return f"BBox expert = {expert_bbox} | BBox learner = {learner_bbox}"

def prepare_preview_frames(expert_video, learner_video):
    os.makedirs("outputs", exist_ok=True)

    expert_preview_path = "outputs/expert_first_frame.jpg"
    learner_preview_path = "outputs/learner_first_frame.jpg"

    save_first_frame(expert_video, expert_preview_path)
    save_first_frame(learner_video, learner_preview_path)

    return expert_preview_path, learner_preview_path


def get_select_coords(evt: gr.SelectData):
    return evt.index


def run_sam_with_points(expert_video, learner_video, expert_point, learner_point):
    expert_frame = extract_first_frame(expert_video)
    learner_frame = extract_first_frame(learner_video)

    if expert_frame is None or learner_frame is None:
        return None, None, "Erreur lors de la lecture des vidéos"

    if expert_point is None or learner_point is None:
        return None, None, "Clique sur les deux images d'abord"

    expert_x, expert_y = expert_point
    learner_x, learner_y = learner_point

    expert_mask = segment_image_from_point(expert_frame, expert_x, expert_y)
    learner_mask = segment_image_from_point(learner_frame, learner_x, learner_y)

    expert_overlay = mask_to_overlay(expert_frame, expert_mask)
    learner_overlay = mask_to_overlay(learner_frame, learner_mask)

    area_text = compare_mask_areas(expert_mask, learner_mask)
    centroid_text = compare_mask_centroids(expert_mask, learner_mask)
    bbox_text = compare_mask_bboxes(expert_mask, learner_mask)
    comparison_text = area_text + " || " + centroid_text + " || " + bbox_text
    return expert_overlay, learner_overlay, comparison_text


with gr.Blocks() as demo:
    gr.Markdown("## Comparaison simple de 2 vidéos avec sélection de point")

    expert_video = gr.Video(label="Vidéo expert")
    learner_video = gr.Video(label="Vidéo learner")

    load_button = gr.Button("Charger les premières frames")

    with gr.Row():
        expert_preview = gr.Image(label="Première frame expert", interactive=True)
        learner_preview = gr.Image(label="Première frame learner", interactive=True)

    expert_point_state = gr.State()
    learner_point_state = gr.State()

    expert_point_text = gr.Textbox(label="Point expert", interactive=False)
    learner_point_text = gr.Textbox(label="Point learner", interactive=False)

    run_button = gr.Button("Lancer SAM avec les points choisis")

    with gr.Row():
        expert_output = gr.Image(label="Overlay expert")
        learner_output = gr.Image(label="Overlay learner")

    result_text = gr.Textbox(label="Résultat")

    load_button.click(
        fn=prepare_preview_frames,
        inputs=[expert_video, learner_video],
        outputs=[expert_preview, learner_preview],
    )

    expert_preview.select(
        fn=get_select_coords,
        inputs=None,
        outputs=[expert_point_state],
    ).then(
        fn=lambda p: f"x={p[0]}, y={p[1]}" if p else "",
        inputs=[expert_point_state],
        outputs=[expert_point_text],
    )

    learner_preview.select(
        fn=get_select_coords,
        inputs=None,
        outputs=[learner_point_state],
    ).then(
        fn=lambda p: f"x={p[0]}, y={p[1]}" if p else "",
        inputs=[learner_point_state],
        outputs=[learner_point_text],
    )

    run_button.click(
        fn=run_sam_with_points,
        inputs=[expert_video, learner_video, expert_point_state, learner_point_state],
        outputs=[expert_output, learner_output, result_text],
    )

demo.launch()