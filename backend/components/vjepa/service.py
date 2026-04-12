from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import numpy as np

from .config import VJepaConfig
from .sampler import sample_uniform_frames

logger = logging.getLogger(__name__)

# Public checkpoints (same files as commented-out upstream default).
_REMOTE_CHECKPOINT_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
# Hub V-JEPA 2.1 models use this key in published .pt files (see hub entry points).
_MODEL_FILE_MAP = {
    "vjepa2_1_vit_base_384": "vjepa2_1_vitb_dist_vitG_384",
    "vjepa2_1_vit_large_384": "vjepa2_1_vitl_dist_vitG_384",
    "vjepa2_1_vit_giant_384": "vjepa2_1_vitg_384",
    "vjepa2_1_vit_gigantic_384": "vjepa2_1_vitG_384",
}
# Primary state_dict key inside each checkpoint (must match facebookresearch/vjepa2 hub backbones).
_MODEL_ENCODER_KEYS = {
    "vjepa2_1_vit_base_384": "ema_encoder",
    "vjepa2_1_vit_large_384": "ema_encoder",
    "vjepa2_1_vit_giant_384": "target_encoder",
    "vjepa2_1_vit_gigantic_384": "target_encoder",
}
_ENCODER_KEY_FALLBACKS = ("ema_encoder", "target_encoder", "encoder")

# Why pretrained=True on torch.hub breaks:
# In the upstream repo, src/hub/backbones.py sets (as of main branch):
#   VJEPA_BASE_URL = "http://localhost:8300"  # labeled "for testing"
# with the real CDN URL commented out. torch.hub then downloads weights from
# localhost and fails. We always use pretrained=False and load weights ourselves
# from dl.fbaipublicfiles.com or from checkpoint_path / checkpoint_url.


class VJepa21Service:
    def __init__(self, config: Optional[VJepaConfig] = None):
        self.config = config or VJepaConfig()
        self.processor = None
        self.model = None
        self.device = "cpu"
        self._loaded = False
        self._real_backend_failed = False
        self.last_error: str | None = None
        self.used_chunk_fallback = False
        self.last_segment_debug: list[dict[str, Any]] = []
        self.last_video_debug: dict[str, Any] = {}

    def _torch_load_checkpoint(self, torch: Any, source: Any) -> dict:
        try:
            return torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(source, map_location="cpu")

    def _hub_load_state_dict_from_url(self, torch: Any, url: str) -> dict:
        try:
            return torch.hub.load_state_dict_from_url(
                url, map_location="cpu", weights_only=False
            )
        except TypeError:
            return torch.hub.load_state_dict_from_url(url, map_location="cpu")

    def load(self) -> None:
        import torch

        load_started = perf_counter()
        logger.info("V-JEPA load() start: model=%s", self.config.model_name)

        self.device = (
            self.config.device
            if self.config.device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )

        crop_size = 384 if self.config.model_name.endswith("_384") else 256
        self.processor = torch.hub.load(
            self.config.hub_repo,
            "vjepa2_preprocessor",
            trust_repo=True,
            pretrained=False,
            crop_size=crop_size,
        )
        logger.info(
            "V-JEPA preprocessor loaded in %.2fs",
            perf_counter() - load_started,
        )

        model, _predictor = torch.hub.load(
            self.config.hub_repo,
            self.config.model_name,
            trust_repo=True,
            pretrained=False,
        )
        self.model = model.to(self.device).eval()
        logger.info(
            "V-JEPA encoder architecture built in %.2fs",
            perf_counter() - load_started,
        )

        logger.info("V-JEPA loading checkpoint weights")
        checkpoint = self._load_checkpoint(torch)
        encoder_state = self._encoder_state_from_checkpoint(checkpoint)
        self.model.load_state_dict(encoder_state, strict=False)
        logger.info(
            "V-JEPA checkpoint weights loaded in %.2fs",
            perf_counter() - load_started,
        )
        self._loaded = True
        self._real_backend_failed = False

    def _load_checkpoint(self, torch: Any) -> dict:
        if self.config.checkpoint_path:
            checkpoint_path = Path(self.config.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Configured V-JEPA checkpoint not found: {checkpoint_path}"
                )
            ckpt = self._torch_load_checkpoint(torch, str(checkpoint_path))
        else:
            model_file = _MODEL_FILE_MAP.get(self.config.model_name)
            if model_file is None:
                raise ValueError(f"Unsupported V-JEPA model name: {self.config.model_name}")

            checkpoint_url = (
                self.config.checkpoint_url
                or f"{_REMOTE_CHECKPOINT_BASE_URL}/{model_file}.pt"
            )
            ckpt = self._hub_load_state_dict_from_url(torch, checkpoint_url)

        if not isinstance(ckpt, dict):
            raise ValueError(f"Expected checkpoint dict, got {type(ckpt).__name__}")
        return ckpt

    def _encoder_state_from_checkpoint(self, checkpoint: dict) -> dict[str, Any]:
        preferred = _MODEL_ENCODER_KEYS.get(self.config.model_name)
        ordered_keys = []
        if preferred:
            ordered_keys.append(preferred)
        for k in _ENCODER_KEY_FALLBACKS:
            if k not in ordered_keys:
                ordered_keys.append(k)

        raw_encoder: dict[str, Any] | None = None
        used_key: str | None = None
        for key in ordered_keys:
            block = checkpoint.get(key)
            if isinstance(block, dict) and block:
                raw_encoder = block
                used_key = key
                break

        if raw_encoder is None:
            top = list(checkpoint.keys())[:12]
            raise KeyError(
                f"No encoder weights found (tried {ordered_keys}); "
                f"checkpoint keys (first 12): {top}"
            )

        logger.debug("V-JEPA checkpoint encoder block: %s", used_key)
        return self._clean_backbone_key(raw_encoder)

    @staticmethod
    def _clean_backbone_key(state_dict: dict[str, Any]) -> dict[str, Any]:
        cleaned = {}
        for key, val in state_dict.items():
            new_key = key.replace("module.", "").replace("backbone.", "")
            cleaned[new_key] = val
        return cleaned

    @staticmethod
    def _fallback_embedding(frames: np.ndarray) -> np.ndarray:
        chunk_count = 8
        chunks = np.array_split(frames, chunk_count, axis=0)
        features: list[np.ndarray] = []

        for chunk in chunks:
            if chunk.size == 0:
                features.append(np.zeros(6, dtype=np.float32))
                continue

            mean_rgb = chunk.mean(axis=(0, 1, 2))
            std_rgb = chunk.std(axis=(0, 1, 2))
            features.append(
                np.concatenate([mean_rgb, std_rgb], axis=0).astype(np.float32)
            )

        return np.concatenate(features, axis=0).astype(np.float32)

    @staticmethod
    def _resample_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
        if frames.ndim != 4 or frames.shape[0] == 0:
            raise ValueError("Expected non-empty frame tensor with shape [T, H, W, C]")

        if frames.shape[0] == num_frames:
            return frames

        indices = np.linspace(0, frames.shape[0] - 1, num_frames).round().astype(int)
        return frames[indices]

    def _ensure_hub_repo_on_path(self, torch: Any) -> None:
        repo_dir = Path(torch.hub.get_dir()) / "facebookresearch_vjepa2_main"
        if repo_dir.exists():
            repo_dir_str = str(repo_dir)
            if repo_dir_str not in sys.path:
                sys.path.insert(0, repo_dir_str)

    @staticmethod
    def _get_video_debug(video_path: str) -> dict[str, Any]:
        import cv2

        cap = cv2.VideoCapture(str(Path(video_path)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()

        return {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps if fps > 0 else None,
            "duration_sec": (total_frames / fps) if fps > 0 and total_frames > 0 else None,
        }

    def _encode_frames(self, frames: np.ndarray) -> np.ndarray:
        import torch

        if self._real_backend_failed:
            self.used_chunk_fallback = True
            logger.warning(
                "V-JEPA fallback triggered before inference because real backend is marked unavailable: %s",
                self.last_error or "unknown previous error",
            )
            return self._fallback_embedding(frames)

        if not self._loaded:
            try:
                self.load()
            except Exception as exc:
                self.last_error = str(exc)
                self._real_backend_failed = True
                self._loaded = False
                logger.warning(
                    "V-JEPA fallback triggered during load; real backend unavailable: %s",
                    exc,
                )
                self.used_chunk_fallback = True
                return self._fallback_embedding(frames)

        try:
            self._ensure_hub_repo_on_path(torch)

            logger.info("V-JEPA preprocessing frames / converting to tensor")
            processed = self.processor(frames)
            if isinstance(processed, list):
                processed = processed[0]
            if isinstance(processed, tuple):
                processed = processed[0]
            if not isinstance(processed, torch.Tensor):
                raise ValueError(
                    f"Unexpected preprocessor output type: {type(processed).__name__}"
                )

            if processed.ndim == 4:
                processed = processed.unsqueeze(0)
            processed = processed.to(self.device)

            logger.info(
                "V-JEPA forward pass starting: input_shape=%s device=%s",
                tuple(processed.shape),
                self.device,
            )
            forward_started = perf_counter()
            with torch.no_grad():
                tokens = self.model(processed)
            logger.info(
                "V-JEPA forward pass finished in %.2fs",
                perf_counter() - forward_started,
            )

            if not isinstance(tokens, torch.Tensor):
                raise ValueError(
                    f"Unexpected encoder output type: {type(tokens).__name__}"
                )

            if tokens.ndim == 3:
                embedding = tokens.mean(dim=1)
            elif tokens.ndim == 2:
                embedding = tokens
            else:
                raise ValueError(
                    f"Unexpected encoder output shape: {tuple(tokens.shape)}"
                )

            self.last_error = None
            logger.info(
                "V-JEPA returning embedding: shape=%s dtype=%s",
                tuple(embedding.shape),
                embedding.dtype,
            )
            return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            self.last_error = str(exc)
            self.used_chunk_fallback = True
            logger.warning(
                "V-JEPA fallback triggered during inference; using chunk fallback: %s",
                exc,
            )
            return self._fallback_embedding(frames)

    def extract_embedding(self, video_path: str) -> np.ndarray:
        self.used_chunk_fallback = False
        logger.info("V-JEPA sampling frames from %s", video_path)
        sampling_started = perf_counter()
        frames = sample_uniform_frames(
            video_path,
            num_frames=self.config.frames_per_clip,
        )
        logger.info(
            "V-JEPA frame sampling finished in %.2fs: shape=%s",
            perf_counter() - sampling_started,
            tuple(frames.shape),
        )
        return self._encode_frames(frames)

    def extract_segment_embeddings(
        self,
        video_path: str,
        num_segments: int = 4,
    ) -> list[np.ndarray]:
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")

        self.used_chunk_fallback = False
        self.last_segment_debug = []
        logger.info(
            "V-JEPA sampling frames for segment extraction from %s", video_path
        )
        video_debug = self._get_video_debug(video_path)
        self.last_video_debug = video_debug
        total_frames = int(video_debug["total_frames"])
        if total_frames <= 0:
            raise ValueError(f"Could not read video: {video_path}")

        segment_edges = np.linspace(0, total_frames, num_segments + 1).astype(int)
        sampling_started = perf_counter()
        segment_embeddings: list[np.ndarray] = []
        for idx in range(num_segments):
            start_frame = segment_edges[idx]
            end_frame = max(start_frame, segment_edges[idx + 1] - 1)
            segment_frames, sampled_indices = sample_uniform_frames(
                video_path,
                num_frames=self.config.frames_per_clip,
                start_frame=start_frame,
                end_frame=end_frame,
                return_indices=True,
            )
            logger.info(
                "V-JEPA encoding segment %s/%s for %s",
                idx + 1,
                num_segments,
                video_path,
            )
            embedding = self._encode_frames(segment_frames)
            segment_embeddings.append(embedding)
            fps = video_debug.get("fps")
            self.last_segment_debug.append(
                {
                    "segment_index": idx,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time_sec": (start_frame / fps) if fps else None,
                    "end_time_sec": (end_frame / fps) if fps else None,
                    "frames_used": int(len(sampled_indices)),
                    "sampled_frame_indices": sampled_indices.astype(int).tolist(),
                    "embedding_norm": float(np.linalg.norm(embedding)),
                }
            )

        logger.info(
            "V-JEPA segment sampling finished in %.2fs: segments=%s frames_per_segment=%s",
            perf_counter() - sampling_started,
            num_segments,
            self.config.frames_per_clip,
        )
        return segment_embeddings
