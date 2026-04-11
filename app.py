import gradio as gr
from sam_image_utils import segment_image_from_point
from sam_runner import track_two_videos_from_selected_points
from sam_metrics import load_json_file, export_derived_metrics_json
from video_utils import get_video_info, read_frame_at_time


def mask_to_overlay(image_bgr, mask):
    overlay = image_bgr.copy()
    overlay[mask.astype(bool)] = [0, 255, 0]
    return overlay


def sanitize_time_seconds(video_path, time_seconds):
    info = get_video_info(video_path)

    if "error" in info:
        return None, None, info["error"]

    frame_count = info["frame_count"]
    fps = info["fps"]
    duration_seconds = info["duration_seconds"]

    if frame_count <= 0:
        return None, None, "Vidéo vide"

    if time_seconds is None:
        time_seconds = 0.0

    time_seconds = float(time_seconds)

    if time_seconds < 0:
        time_seconds = 0.0

    if duration_seconds > 0 and time_seconds > duration_seconds:
        time_seconds = duration_seconds

    frame_index = 0
    if fps > 0:
        frame_index = int(time_seconds * fps)

    if frame_index >= frame_count:
        frame_index = frame_count - 1

    return time_seconds, frame_index, None


def build_video_summary(expert_video, learner_video, expert_time_seconds, learner_time_seconds):
    expert_info = get_video_info(expert_video)
    learner_info = get_video_info(learner_video)

    if "error" in expert_info or "error" in learner_info:
        return "Erreur lors de la lecture des informations vidéo"

    _, expert_frame_index, _ = sanitize_time_seconds(expert_video, expert_time_seconds)
    _, learner_frame_index, _ = sanitize_time_seconds(learner_video, learner_time_seconds)

    return (
        f"Expert -> frames={expert_info['frame_count']}, fps={expert_info['fps']:.2f}, "
        f"durée={expert_info['duration_seconds']:.2f}s, temps choisi={expert_time_seconds:.2f}s, frame={expert_frame_index} || "
        f"Learner -> frames={learner_info['frame_count']}, fps={learner_info['fps']:.2f}, "
        f"durée={learner_info['duration_seconds']:.2f}s, temps choisi={learner_time_seconds:.2f}s, frame={learner_frame_index}"
    )


def load_selected_times(expert_video, learner_video, expert_time_seconds, learner_time_seconds):
    if not expert_video or not learner_video:
        return None, None, None, None, "Charge les deux vidéos d'abord"

    expert_time_seconds, expert_frame_index, expert_error = sanitize_time_seconds(expert_video, expert_time_seconds)
    learner_time_seconds, learner_frame_index, learner_error = sanitize_time_seconds(learner_video, learner_time_seconds)

    if expert_error is not None:
        return None, None, None, None, f"Erreur expert: {expert_error}"

    if learner_error is not None:
        return None, None, None, None, f"Erreur learner: {learner_error}"

    expert_frame = read_frame_at_time(expert_video, expert_time_seconds)
    learner_frame = read_frame_at_time(learner_video, learner_time_seconds)

    if expert_frame is None or learner_frame is None:
        return None, None, None, None, "Erreur lors de la lecture des temps choisis"

    summary_text = build_video_summary(
        expert_video,
        learner_video,
        expert_time_seconds,
        learner_time_seconds,
    )

    return expert_frame, learner_frame, expert_time_seconds, learner_time_seconds, summary_text


def get_select_coords(evt: gr.SelectData):
    return evt.index


def preview_sam_on_selected_times(
    expert_video,
    learner_video,
    expert_time_seconds,
    learner_time_seconds,
    expert_point,
    learner_point,
):
    if not expert_video or not learner_video:
        return None, None, "Charge les deux vidéos d'abord"

    if expert_point is None or learner_point is None:
        return None, None, "Clique sur les deux images d'abord"

    expert_time_seconds, _, expert_error = sanitize_time_seconds(expert_video, expert_time_seconds)
    learner_time_seconds, _, learner_error = sanitize_time_seconds(learner_video, learner_time_seconds)

    if expert_error is not None:
        return None, None, f"Erreur expert: {expert_error}"

    if learner_error is not None:
        return None, None, f"Erreur learner: {learner_error}"

    expert_frame = read_frame_at_time(expert_video, expert_time_seconds)
    learner_frame = read_frame_at_time(learner_video, learner_time_seconds)

    if expert_frame is None or learner_frame is None:
        return None, None, "Erreur lors de la lecture des temps choisis"

    expert_x, expert_y = expert_point
    learner_x, learner_y = learner_point

    expert_mask = segment_image_from_point(expert_frame, expert_x, expert_y)
    learner_mask = segment_image_from_point(learner_frame, learner_x, learner_y)

    expert_overlay = mask_to_overlay(expert_frame, expert_mask)
    learner_overlay = mask_to_overlay(learner_frame, learner_mask)

    return expert_overlay, learner_overlay, "Prévisualisation SAM prête"


def run_video_tracking(
    expert_video,
    learner_video,
    expert_time_seconds,
    learner_time_seconds,
    expert_point,
    learner_point,
    analysis_mode,
    max_seconds,
    frame_stride,
):
    if not expert_video or not learner_video:
        return None, None, {"error": "Charge les deux vidéos d'abord"}, {"error": "RAW JSON manquant"}, {"error": "METRICS JSON manquant"}

    if expert_point is None or learner_point is None:
        return None, None, {"error": "Clique sur les deux images d'abord"}, {"error": "RAW JSON manquant"}, {"error": "METRICS JSON manquant"}

    if frame_stride is None:
        frame_stride = 1

    frame_stride = int(frame_stride)

    result = track_two_videos_from_selected_points(
        expert_video_path=expert_video,
        learner_video_path=learner_video,
        expert_time_seconds=expert_time_seconds,
        learner_time_seconds=learner_time_seconds,
        expert_point_xy=expert_point,
        learner_point_xy=learner_point,
        analysis_mode=analysis_mode,
        max_seconds=max_seconds,
        frame_stride=frame_stride,
    )

    if "error" in result:
        return None, None, result, {"error": "RAW JSON non généré"}, {"error": "METRICS JSON non généré"}

    expert_annotated = result["expert_video"]["annotated_video_path"]
    learner_annotated = result["learner_video"]["annotated_video_path"]

    expert_raw_json_path = result["expert_video"]["raw_json_path"]
    learner_raw_json_path = result["learner_video"]["raw_json_path"]

    expert_metrics_json_path = export_derived_metrics_json(expert_raw_json_path)
    learner_metrics_json_path = export_derived_metrics_json(learner_raw_json_path)

    raw_json_bundle = {
        "expert_raw_json_path": expert_raw_json_path,
        "learner_raw_json_path": learner_raw_json_path,
        "expert_raw_json": load_json_file(expert_raw_json_path),
        "learner_raw_json": load_json_file(learner_raw_json_path),
    }

    metrics_json_bundle = {
        "expert_metrics_json_path": expert_metrics_json_path,
        "learner_metrics_json_path": learner_metrics_json_path,
        "expert_metrics_json": load_json_file(expert_metrics_json_path),
        "learner_metrics_json": load_json_file(learner_metrics_json_path),
    }

    return expert_annotated, learner_annotated, result, raw_json_bundle, metrics_json_bundle


with gr.Blocks() as demo:
    gr.Markdown("## SAM 2 Video Tracking Demo")

    with gr.Row():
        expert_video = gr.Video(label="Vidéo expert")
        learner_video = gr.Video(label="Vidéo learner")

    with gr.Row():
        expert_time_input = gr.Number(label="Temps expert (secondes)", value=1.00, precision=2)
        learner_time_input = gr.Number(label="Temps learner (secondes)", value=1.00, precision=2)

    load_button = gr.Button("Charger les temps choisis")

    with gr.Row():
        expert_preview = gr.Image(label="Image expert au temps choisi", interactive=True)
        learner_preview = gr.Image(label="Image learner au temps choisi", interactive=True)

    expert_selected_time_state = gr.State()
    learner_selected_time_state = gr.State()

    expert_point_state = gr.State()
    learner_point_state = gr.State()

    info_text = gr.Textbox(label="Infos vidéo / temps", interactive=False)
    expert_point_text = gr.Textbox(label="Point expert", interactive=False)
    learner_point_text = gr.Textbox(label="Point learner", interactive=False)

    preview_button = gr.Button("Prévisualiser SAM sur l'image choisie")

    with gr.Row():
        expert_preview_output = gr.Image(label="Overlay expert")
        learner_preview_output = gr.Image(label="Overlay learner")

    preview_text = gr.Textbox(label="Résultat preview", interactive=False)

    gr.Markdown("## Paramètres analyse vidéo")

    with gr.Row():
        analysis_mode = gr.Dropdown(
            choices=["first_n_seconds", "full"],
            value="first_n_seconds",
            label="Mode analyse"
        )
        max_seconds = gr.Number(label="Nombre de secondes si mode first_n_seconds", value=3, precision=0)
        frame_stride = gr.Number(label="Frame stride", value=3, precision=0)

    run_button = gr.Button("Lancer SAM sur la vidéo")

    with gr.Row():
        expert_video_output = gr.Video(label="Vidéo expert annotée")
        learner_video_output = gr.Video(label="Vidéo learner annotée")

    result_json = gr.JSON(label="JSON résultat")
    raw_json_output = gr.JSON(label="RAW JSON SAM 2")
    metrics_json_output = gr.JSON(label="Derived Metrics JSON")

    load_button.click(
        fn=load_selected_times,
        inputs=[
            expert_video,
            learner_video,
            expert_time_input,
            learner_time_input,
        ],
        outputs=[
            expert_preview,
            learner_preview,
            expert_selected_time_state,
            learner_selected_time_state,
            info_text,
        ],
    )

    expert_preview.select(
        fn=get_select_coords,
        inputs=None,
        outputs=[expert_point_state],
    ).then(
        fn=lambda p, t: f"temps={t:.2f}s, x={p[0]}, y={p[1]}" if p is not None and t is not None else "",
        inputs=[expert_point_state, expert_selected_time_state],
        outputs=[expert_point_text],
    )

    learner_preview.select(
        fn=get_select_coords,
        inputs=None,
        outputs=[learner_point_state],
    ).then(
        fn=lambda p, t: f"temps={t:.2f}s, x={p[0]}, y={p[1]}" if p is not None and t is not None else "",
        inputs=[learner_point_state, learner_selected_time_state],
        outputs=[learner_point_text],
    )

    preview_button.click(
        fn=preview_sam_on_selected_times,
        inputs=[
            expert_video,
            learner_video,
            expert_selected_time_state,
            learner_selected_time_state,
            expert_point_state,
            learner_point_state,
        ],
        outputs=[
            expert_preview_output,
            learner_preview_output,
            preview_text,
        ],
    )

    run_button.click(
        fn=run_video_tracking,
        inputs=[
            expert_video,
            learner_video,
            expert_selected_time_state,
            learner_selected_time_state,
            expert_point_state,
            learner_point_state,
            analysis_mode,
            max_seconds,
            frame_stride,
        ],
        outputs=[
            expert_video_output,
            learner_video_output,
            result_json,
            raw_json_output,
            metrics_json_output,
        ],
    )

demo.launch()