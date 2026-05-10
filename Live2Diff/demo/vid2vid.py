import os
import sys
import time
from collections import deque
from typing import Dict, Literal, Optional, Tuple

import numpy as np

_demo_dir = os.path.dirname(os.path.abspath(__file__))
if _demo_dir not in sys.path:
    sys.path.insert(0, _demo_dir)
sys.path.append(os.path.join(_demo_dir, ".."))

import torch
from config import Args
from PIL import Image
from pydantic import BaseModel, Field, create_model

from live2diff.utils.config import load_config
from live2diff.utils.wrapper import StreamAnimateDiffusionDepthWrapper

from temporal_stability import (
    compute_flow_warp_errors,
    compute_optical_flow,
    normalize_flow_backend,
    stability_score,
)


DELTA_WINDOW_S = 30.0


def _warped_noise_blend_from_cfg(noise_control_cfg: dict) -> float:
    if "warped_noise_blend" in noise_control_cfg:
        return min(max(float(noise_control_cfg["warped_noise_blend"]), 0.0), 1.0)
    ru = float(noise_control_cfg.get("warped_noise_reuse", 0.85))
    rs = float(noise_control_cfg.get("warped_noise_residual", 0.15))
    s = ru + rs
    if s <= 1e-6:
        return 0.85
    return min(max(ru / s, 0.0), 1.0)


default_prompt = "masterpiece, best quality, felted, 1man with glasses, glasses, play with his pen"

page_content = """<h1 class="text-3xl font-bold">Live2Diff: </h1>
<h2 class="text-xl font-bold">Live Stream Translation via Uni-directional Attention in Video Diffusion Models</h2>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/open-mmlab/Live2Diff"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Live2Diff
</a>
pipeline using
    <a
    href="https://huggingface.co/latent-consistency/lcm-lora-sdv1-5"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">LCM-LoRA</a
    > with a MJPEG stream server.
</p>
"""


WARMUP_FRAMES = 8
WINDOW_SIZE = 16


class Pipeline:
    class Info(BaseModel):
        name: str = "Live2Diff"
        input_mode: str = "image"
        page_content: str = page_content

    def build_input_params(
        self,
        width=512,
        height=512,
        use_softedge=False,
        softedge_scale=0.35,
        depth_scale=1.0,
        softedge_mode="classical",
        softedge_debug=False,
        subject_mask_debug=True,
        subject_mask_backend="auto",
        subject_mask_keyframe_interval=8,
        subject_mask_ema=0.65,
        enable_stylize_preprocess=False,
        depth_blur_sigma=1.2,
        depth_power=0.2,
        depth_smoothstep_min=0.5,
        depth_smoothstep_max=1.0,
        saturation_scale=0.2,
        image_blur_sigma=2.0,
        fog_white_mix=0.5,
        outline_blur_sigma=0.8,
        outline_power=0.1,
        outline_smoothstep_min=0.57,
        outline_smoothstep_max=1.0,
        fog_depth_mask_blend=0.5,
        subject_mask_fog_blur_sigma=1.0,
        warped_noise_blend=0.85,
        key_step_sync_enabled=False,
        key_step_sync_strength=0.0,
        key_step_sync_index=0,
        key_step_sync_flow_backend="farneback",
    ):
        key_step_sync_flow_backend = normalize_flow_backend(key_step_sync_flow_backend)
        return create_model(
            "InputParams",
            width=(
                int,
                Field(
                    width,
                    title="Width",
                    json_schema_extra={
                        "id": "width",
                        "minimum": 64,
                        "maximum": 2048,
                        "disabled": True,
                        "hide": True,
                    },
                ),
            ),
            height=(
                int,
                Field(
                    height,
                    title="Height",
                    json_schema_extra={
                        "id": "height",
                        "minimum": 64,
                        "maximum": 2048,
                        "disabled": True,
                        "hide": True,
                    },
                ),
            ),
            use_softedge=(
                bool,
                Field(use_softedge, title="Use SoftEdge", json_schema_extra={"id": "use_softedge"}),
            ),
            softedge_scale=(
                float,
                Field(
                    softedge_scale,
                    title="SoftEdge Scale",
                    json_schema_extra={
                        "id": "softedge_scale",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                ),
            ),
            depth_scale=(
                float,
                Field(
                    depth_scale,
                    title="Depth Scale",
                    json_schema_extra={
                        "id": "depth_scale",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                ),
            ),
            softedge_mode=(
                str,
                Field(softedge_mode, title="SoftEdge Mode", json_schema_extra={"id": "softedge_mode"}),
            ),
            softedge_debug=(
                bool,
                Field(
                    softedge_debug,
                    title="Show SoftEdge Debug",
                    json_schema_extra={"id": "softedge_debug"},
                ),
            ),
            subject_mask_debug=(
                bool,
                Field(
                    subject_mask_debug,
                    title="Show Subject Mask Debug",
                    json_schema_extra={"id": "subject_mask_debug"},
                ),
            ),
            subject_mask_backend=(
                str,
                Field(
                    subject_mask_backend,
                    title="Subject Mask Backend",
                    json_schema_extra={"id": "subject_mask_backend"},
                ),
            ),
            subject_mask_keyframe_interval=(
                int,
                Field(
                    subject_mask_keyframe_interval,
                    title="Subject Mask Keyframe Interval",
                    json_schema_extra={
                        "id": "subject_mask_keyframe_interval",
                        "minimum": 1,
                        "maximum": 60,
                    },
                ),
            ),
            subject_mask_ema=(
                float,
                Field(
                    subject_mask_ema,
                    ge=0.0,
                    le=0.98,
                    title="Subject Mask EMA",
                    json_schema_extra={
                        "id": "subject_mask_ema",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 0.98,
                    },
                ),
            ),
            enable_stylize_preprocess=(
                bool,
                Field(
                    enable_stylize_preprocess,
                    title="Enable Stylize Preprocess",
                    json_schema_extra={"id": "enable_stylize_preprocess"},
                ),
            ),
            depth_blur_sigma=(
                float,
                Field(
                    depth_blur_sigma,
                    ge=0.0,
                    title="Depth Blur Sigma",
                    json_schema_extra={
                        "id": "depth_blur_sigma",
                        "step": 0.05,
                        "minimum": 0.0,
                        "maximum": 6.0,
                    },
                ),
            ),
            depth_power=(
                float,
                Field(
                    depth_power,
                    ge=0.0,
                    title="Depth Power",
                    json_schema_extra={
                        "id": "depth_power",
                        "step": 0.02,
                        "minimum": 0.05,
                        "maximum": 3.0,
                    },
                ),
            ),
            depth_smoothstep_min=(
                float,
                Field(
                    depth_smoothstep_min,
                    title="Depth SmoothStep Min",
                    json_schema_extra={
                        "id": "depth_smoothstep_min",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                ),
            ),
            depth_smoothstep_max=(
                float,
                Field(
                    depth_smoothstep_max,
                    title="Depth SmoothStep Max",
                    json_schema_extra={
                        "id": "depth_smoothstep_max",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                ),
            ),
            saturation_scale=(
                float,
                Field(
                    saturation_scale,
                    ge=0.0,
                    title="Saturation Scale",
                    json_schema_extra={
                        "id": "saturation_scale",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                ),
            ),
            image_blur_sigma=(
                float,
                Field(
                    image_blur_sigma,
                    ge=0.0,
                    title="Image Blur Sigma",
                    json_schema_extra={
                        "id": "image_blur_sigma",
                        "step": 0.05,
                        "minimum": 0.0,
                        "maximum": 6.0,
                    },
                ),
            ),
            fog_white_mix=(
                float,
                Field(
                    fog_white_mix,
                    ge=0.0,
                    title="Fog White Mix",
                    json_schema_extra={
                        "id": "fog_white_mix",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                ),
            ),
            outline_blur_sigma=(
                float,
                Field(
                    outline_blur_sigma,
                    ge=0.0,
                    title="Outline Blur Sigma",
                    json_schema_extra={
                        "id": "outline_blur_sigma",
                        "step": 0.05,
                        "minimum": 0.0,
                        "maximum": 5.0,
                    },
                ),
            ),
            outline_power=(
                float,
                Field(
                    outline_power,
                    ge=0.0,
                    title="Outline Power",
                    json_schema_extra={
                        "id": "outline_power",
                        "step": 0.02,
                        "minimum": 0.05,
                        "maximum": 3.0,
                    },
                ),
            ),
            outline_smoothstep_min=(
                float,
                Field(
                    outline_smoothstep_min,
                    title="Outline SmoothStep Min",
                    json_schema_extra={
                        "id": "outline_smoothstep_min",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                ),
            ),
            outline_smoothstep_max=(
                float,
                Field(
                    outline_smoothstep_max,
                    title="Outline SmoothStep Max",
                    json_schema_extra={
                        "id": "outline_smoothstep_max",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                ),
            ),
            fog_depth_mask_blend=(
                float,
                Field(
                    fog_depth_mask_blend,
                    ge=0.0,
                    le=1.0,
                    title="Fog: depth vs mask blend",
                    description="0=仅 depth，1=仅 mask；中间为线性插值。",
                    json_schema_extra={
                        "id": "fog_depth_mask_blend",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                ),
            ),
            subject_mask_fog_blur_sigma=(
                float,
                Field(
                    subject_mask_fog_blur_sigma,
                    ge=0.0,
                    title="Mask fog blur sigma",
                    json_schema_extra={
                        "id": "subject_mask_fog_blur_sigma",
                        "step": 0.05,
                        "minimum": 0.0,
                        "maximum": 8.0,
                    },
                ),
            ),
            warped_noise_blend=(
                float,
                Field(
                    warped_noise_blend,
                    ge=0.0,
                    le=1.0,
                    title="Warped noise reuse",
                    description="与新鲜噪声权重互补：residual=1−此值；仅影响光流 warp 混合系数。",
                    json_schema_extra={
                        "id": "warped_noise_blend",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                ),
            ),
            key_step_sync_enabled=(
                bool,
                Field(
                    key_step_sync_enabled,
                    title="Key-step sync",
                    json_schema_extra={"id": "key_step_sync_enabled", "hide": True},
                ),
            ),
            key_step_sync_strength=(
                float,
                Field(
                    key_step_sync_strength,
                    ge=0.0,
                    le=1.0,
                    title="Key-step sync strength",
                    json_schema_extra={
                        "id": "key_step_sync_strength",
                        "step": 0.01,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "hide": True,
                    },
                ),
            ),
            key_step_sync_index=(
                int,
                Field(
                    key_step_sync_index,
                    title="Key-step sync index",
                    json_schema_extra={
                        "id": "key_step_sync_index",
                        "minimum": 0,
                        "maximum": 8,
                        "hide": True,
                    },
                ),
            ),
            key_step_sync_flow_backend=(
                Literal["farneback", "raft", "sea-raft"],
                Field(
                    key_step_sync_flow_backend,
                    title="Flow backend",
                    json_schema_extra={"id": "key_step_sync_flow_backend", "hide": True},
                ),
            ),
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        config_path = args.config

        cfg = load_config(config_path)
        prompt = cfg.get("prompt", None) or args.prompt or default_prompt
        width = int(cfg.get("width", 512))
        height = int(cfg.get("height", 512))
        conditioning_kwargs = cfg.get("conditioning_kwargs", {})
        stylization_kwargs = cfg.get("stylization_kwargs", {})
        noise_control_cfg = cfg.get("noise_control", {})
        subject_mask_cfg = cfg.get("subject_mask", {})
        key_step_sync_cfg = cfg.get("key_step_sync", {})

        self.InputParams = self.build_input_params(
            width,
            height,
            use_softedge=conditioning_kwargs.get("use_softedge", False),
            softedge_scale=conditioning_kwargs.get("softedge_scale", 0.35),
            depth_scale=conditioning_kwargs.get("depth_scale", 1.0),
            softedge_mode=conditioning_kwargs.get("softedge_mode", "classical"),
            softedge_debug=conditioning_kwargs.get("softedge_debug", False),
            subject_mask_debug=subject_mask_cfg.get("debug", True),
            subject_mask_backend=subject_mask_cfg.get("backend", "auto"),
            subject_mask_keyframe_interval=subject_mask_cfg.get("keyframe_interval", 8),
            subject_mask_ema=subject_mask_cfg.get("ema", 0.65),
            enable_stylize_preprocess=stylization_kwargs.get("enable_stylize_preprocess", False),
            depth_blur_sigma=stylization_kwargs.get("depth_blur_sigma", 1.2),
            depth_power=stylization_kwargs.get("depth_power", 0.2),
            depth_smoothstep_min=stylization_kwargs.get("depth_smoothstep_min", 0.5),
            depth_smoothstep_max=stylization_kwargs.get("depth_smoothstep_max", 1.0),
            saturation_scale=stylization_kwargs.get("saturation_scale", 0.2),
            image_blur_sigma=stylization_kwargs.get("image_blur_sigma", 2.0),
            fog_white_mix=stylization_kwargs.get("fog_white_mix", 0.5),
            outline_blur_sigma=stylization_kwargs.get("outline_blur_sigma", 0.8),
            outline_power=stylization_kwargs.get("outline_power", 0.1),
            outline_smoothstep_min=stylization_kwargs.get("outline_smoothstep_min", 0.57),
            outline_smoothstep_max=stylization_kwargs.get("outline_smoothstep_max", 1.0),
            fog_depth_mask_blend=subject_mask_cfg.get("fog_depth_mask_blend", 0.5),
            subject_mask_fog_blur_sigma=subject_mask_cfg.get("fog_blur_sigma", 1.0),
            warped_noise_blend=_warped_noise_blend_from_cfg(noise_control_cfg),
            key_step_sync_enabled=key_step_sync_cfg.get("enabled", False),
            key_step_sync_strength=key_step_sync_cfg.get("strength", 0.0),
            key_step_sync_index=key_step_sync_cfg.get("key_step_index", 0),
            key_step_sync_flow_backend=key_step_sync_cfg.get("flow_backend", "farneback"),
        )
        params = self.InputParams()
        self.prompt = prompt
        self.width = params.width
        self.height = params.height

        num_inference_steps = args.num_inference_steps or cfg.get("num_inference_steps", None)
        strength = args.strength or cfg.get("strength", None)
        t_index_list = args.t_index_list or cfg.get("t_index_list", None)

        self.stream = StreamAnimateDiffusionDepthWrapper(
            few_step_model_type="lcm",
            config_path=config_path,
            cfg_type="none",
            strength=strength,
            num_inference_steps=num_inference_steps,
            t_index_list=t_index_list,
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            acceleration=args.acceleration,
            do_add_noise=True,
            output_type="pil",
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
            use_denoising_batch=True,
            use_tiny_vae=True,
            seed=args.seed,
            engine_dir=args.engine_dir,
        )
        self.stream.stream.pipe.sam2_model_cfg = subject_mask_cfg.get("sam2_model_cfg")
        self.stream.stream.pipe.sam2_checkpoint = subject_mask_cfg.get("sam2_checkpoint")
        warped_blend = _warped_noise_blend_from_cfg(noise_control_cfg)
        self.noise_control_cfg = {
            "enabled": bool(noise_control_cfg.get("enabled", True)),
            "motion_low_threshold": float(noise_control_cfg.get("low_threshold", 0.015)),
            "motion_high_threshold": float(noise_control_cfg.get("high_threshold", 0.06)),
            "min_noise_rate": float(noise_control_cfg.get("min_noise_rate", 0.05)),
            "max_noise_rate": float(noise_control_cfg.get("max_noise_rate", 1.0)),
            "enable_warped_noise": bool(noise_control_cfg.get("enable_warped_noise", True)),
            "warped_noise_blend": warped_blend,
            "flow_max_side": int(noise_control_cfg.get("flow_max_side", 256)),
        }
        flow_backend = normalize_flow_backend(key_step_sync_cfg.get("flow_backend", "farneback"))
        self.key_step_sync_cfg = {
            "enabled": bool(key_step_sync_cfg.get("enabled", False)),
            "key_step_index": int(key_step_sync_cfg.get("key_step_index", 0)),
            "strength": float(key_step_sync_cfg.get("strength", 0.0)),
            "low_threshold": float(key_step_sync_cfg.get("low_threshold", 0.015)),
            "high_threshold": float(key_step_sync_cfg.get("high_threshold", 0.06)),
            "flow_backend": flow_backend,
        }
        self.stream.set_key_step_sync(**self.key_step_sync_cfg)
        self.stream.set_noise_control(
            enabled=self.noise_control_cfg["enabled"],
            motion_low_threshold=self.noise_control_cfg["motion_low_threshold"],
            motion_high_threshold=self.noise_control_cfg["motion_high_threshold"],
            min_noise_rate=self.noise_control_cfg["min_noise_rate"],
            max_noise_rate=self.noise_control_cfg["max_noise_rate"],
            enable_warped_noise=self.noise_control_cfg["enable_warped_noise"],
            warped_noise_reuse=warped_blend,
            warped_noise_residual=1.0 - warped_blend,
        )

        self.warmup_frame_list = []
        self.has_prepared = False
        self._prev_input_rgb: Optional[np.ndarray] = None
        self._prev_output_rgb: Optional[np.ndarray] = None
        self._temporal_delta_deque: deque = deque()
        self._temporal_debug: Dict[str, Optional[float]] = {}
        self._reset_temporal_debug_only()

    def _reset_temporal_debug_only(self) -> None:
        self._temporal_debug = {
            "temporal_stability_src": None,
            "temporal_stability_out": None,
            "temporal_stability_delta": None,
            "temporal_stability_delta_avg_30s": None,
            "temporal_stability_e_src": None,
            "temporal_stability_e_out": None,
        }

    def _reset_temporal_state(self) -> None:
        self._prev_input_rgb = None
        self._prev_output_rgb = None
        self._temporal_delta_deque.clear()
        self._reset_temporal_debug_only()
        self.stream.reset_noise_state()

    def reset_runtime_state(self) -> None:
        self.warmup_frame_list = []
        self.has_prepared = False
        self._reset_temporal_state()

    def get_debug_info(self, phase: str) -> Dict:
        debug_info = {
            "phase": phase,
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
        }
        debug_info.update(self.stream.get_debug_info())
        debug_info.update(self._temporal_debug)
        return debug_info

    def predict(
        self, params: "Pipeline.InputParams"
    ) -> Tuple[
        Image.Image,
        Dict,
        Optional[Image.Image],
        Optional[Image.Image],
        Optional[Image.Image],
        Optional[Image.Image],
    ]:
        self.stream.set_conditioning(
            use_softedge=getattr(params, "use_softedge", False),
            softedge_scale=getattr(params, "softedge_scale", 0.35),
            depth_scale=getattr(params, "depth_scale", 1.0),
            softedge_mode=getattr(params, "softedge_mode", "classical"),
            softedge_debug=getattr(params, "softedge_debug", False),
            subject_mask_debug=getattr(params, "subject_mask_debug", True),
            subject_mask_backend=getattr(params, "subject_mask_backend", "auto"),
            subject_mask_keyframe_interval=getattr(params, "subject_mask_keyframe_interval", 8),
            subject_mask_ema=getattr(params, "subject_mask_ema", 0.65),
        )
        self.stream.set_stylization(
            enable_stylize_preprocess=getattr(params, "enable_stylize_preprocess", False),
            depth_blur_sigma=getattr(params, "depth_blur_sigma", 1.2),
            depth_power=getattr(params, "depth_power", 0.2),
            depth_smoothstep_min=getattr(params, "depth_smoothstep_min", 0.5),
            depth_smoothstep_max=getattr(params, "depth_smoothstep_max", 1.0),
            saturation_scale=getattr(params, "saturation_scale", 0.2),
            image_blur_sigma=getattr(params, "image_blur_sigma", 2.0),
            fog_white_mix=getattr(params, "fog_white_mix", 0.5),
            outline_blur_sigma=getattr(params, "outline_blur_sigma", 0.8),
            outline_power=getattr(params, "outline_power", 0.1),
            outline_smoothstep_min=getattr(params, "outline_smoothstep_min", 0.57),
            outline_smoothstep_max=getattr(params, "outline_smoothstep_max", 1.0),
            fog_depth_mask_blend=getattr(params, "fog_depth_mask_blend", 0.5),
            subject_mask_fog_blur_sigma=getattr(params, "subject_mask_fog_blur_sigma", 1.0),
        )
        flow_backend = normalize_flow_backend(
            getattr(params, "key_step_sync_flow_backend", self.key_step_sync_cfg["flow_backend"])
        )
        self.stream.set_key_step_sync(
            enabled=getattr(params, "key_step_sync_enabled", self.key_step_sync_cfg["enabled"]),
            key_step_index=getattr(params, "key_step_sync_index", self.key_step_sync_cfg["key_step_index"]),
            strength=getattr(params, "key_step_sync_strength", self.key_step_sync_cfg["strength"]),
            low_threshold=self.key_step_sync_cfg["low_threshold"],
            high_threshold=self.key_step_sync_cfg["high_threshold"],
            flow_backend=flow_backend,
        )

        if not self.has_prepared:
            self._reset_temporal_state()

        if len(self.warmup_frame_list) < WARMUP_FRAMES:
            self.warmup_frame_list.append(self.stream.preprocess_image(params.image))

        elif len(self.warmup_frame_list) == WARMUP_FRAMES and not self.has_prepared:
            warmup_frames = torch.stack(self.warmup_frame_list)
            self.stream.prepare(
                warmup_frames=warmup_frames,
                prompt=self.prompt,
                guidance_scale=1,
            )
            self.has_prepared = True

        if self.has_prepared:
            src_pil = params.image.convert("RGB") if params.image.mode != "RGB" else params.image
            curr_in = np.asarray(
                src_pil.resize((self.width, self.height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
            flow_stats = None
            if self._prev_input_rgb is not None:
                flow_stats = compute_optical_flow(
                    self._prev_input_rgb,
                    curr_in,
                    flow_max_side=self.noise_control_cfg["flow_max_side"],
                    backend=flow_backend,
                )
                if flow_stats is not None:
                    self.stream.set_key_step_sync(flow_backend=flow_stats.backend)
            blend = float(
                getattr(params, "warped_noise_blend", self.noise_control_cfg["warped_noise_blend"])
            )
            blend = min(max(blend, 0.0), 1.0)
            self.stream.set_noise_control(
                enabled=self.noise_control_cfg["enabled"],
                motion_score=None if flow_stats is None else flow_stats.e_src,
                mean_flow_mag=None if flow_stats is None else flow_stats.mean_flow_mag,
                flow=None if flow_stats is None else flow_stats.flow,
                motion_low_threshold=self.noise_control_cfg["motion_low_threshold"],
                motion_high_threshold=self.noise_control_cfg["motion_high_threshold"],
                min_noise_rate=self.noise_control_cfg["min_noise_rate"],
                max_noise_rate=self.noise_control_cfg["max_noise_rate"],
                enable_warped_noise=self.noise_control_cfg["enable_warped_noise"],
                warped_noise_reuse=blend,
                warped_noise_residual=1.0 - blend,
            )
            image_tensor = self.stream.preprocess_image(params.image)
            output_image = self.stream(image=image_tensor)
            out_pil = output_image.convert("RGB") if output_image.mode != "RGB" else output_image
            curr_out = np.asarray(
                out_pil.resize((self.width, self.height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
            if self._prev_input_rgb is not None and self._prev_output_rgb is not None:
                pair = compute_flow_warp_errors(
                    self._prev_input_rgb,
                    curr_in,
                    self._prev_output_rgb,
                    curr_out,
                    flow_max_side=self.noise_control_cfg["flow_max_side"],
                    flow_stats=flow_stats,
                    flow_backend=flow_backend,
                )
                if pair is not None:
                    e_src, e_out = pair
                    s_src = stability_score(e_src)
                    s_out = stability_score(e_out)
                    delta = s_out - s_src
                    now = time.monotonic()
                    self._temporal_delta_deque.append((now, delta))
                    while self._temporal_delta_deque and self._temporal_delta_deque[0][0] < now - DELTA_WINDOW_S:
                        self._temporal_delta_deque.popleft()
                    deltas = [d for _, d in self._temporal_delta_deque]
                    avg_d = sum(deltas) / len(deltas) if deltas else 0.0
                    self._temporal_debug = {
                        "temporal_stability_src": round(s_src, 4),
                        "temporal_stability_out": round(s_out, 4),
                        "temporal_stability_delta": round(delta, 4),
                        "temporal_stability_delta_avg_30s": round(avg_d, 4),
                        "temporal_stability_e_src": round(e_src, 6),
                        "temporal_stability_e_out": round(e_out, 6),
                    }
                else:
                    self._reset_temporal_debug_only()
            self._prev_input_rgb = curr_in.copy()
            self._prev_output_rgb = curr_out.copy()
            return (
                output_image,
                self.get_debug_info("running"),
                self.stream.get_last_depth_image(),
                self.stream.get_last_softedge_image(),
                self.stream.get_last_subject_mask_image(),
                self.stream.get_last_stylized_image(),
            )
        return Image.new("RGB", (params.width, params.height)), self.get_debug_info("warming_up"), None, None, None, None
