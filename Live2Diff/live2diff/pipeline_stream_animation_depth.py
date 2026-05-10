import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import LCMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from einops import rearrange

# from live2diff.image_filter import SimilarImageFilter

from .animatediff.pipeline import AnimationDepthPipeline


logger = logging.getLogger(__name__)

WARMUP_FRAMES = 8
WINDOW_SIZE = 16


class StreamAnimateDiffusionDepth:
    def __init__(
        self,
        pipe: AnimationDepthPipeline,
        num_inference_steps: int,
        t_index_list: Optional[List[int]] = None,
        strength: Optional[float] = None,
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        clip_skip: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "none",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.pipe = pipe

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.clip_skip = clip_skip

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        if strength is not None:
            t_index_list, timesteps = self.get_timesteps(num_inference_steps, strength, self.device)
            print(
                f"Generate t_index_list: {t_index_list} via "
                f"num_inference_steps: {num_inference_steps}, strength: {strength}"
            )
            self.timesteps = timesteps
        else:
            print(
                f"t_index_list is passed: {t_index_list}. "
                f"Number Inference Steps: {num_inference_steps}, "
                f"equivalents to strength {1 - t_index_list[0] / num_inference_steps}."
            )
            self.timesteps = self.scheduler.timesteps.to(self.device)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)
        self.strength = strength

        assert cfg_type == "none", f'cfg_type must be "none" for now, but got {cfg_type}.'
        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (self.denoising_steps_num + 1) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = 2 * self.denoising_steps_num * self.frame_bff_size
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        # self.similar_filter = SimilarImageFilter()
        # self.prev_image_result = None
        self.last_depth_map = None
        self.last_softedge_map = None
        self.last_subject_mask_map = None
        self.last_stylized_map = None
        self.last_processed_depth_map = None
        self.last_processed_outline_map = None
        self.softedge_backend = "disabled"
        self.subject_mask_backend = "disabled"
        self.subject_mask_requested_backend = "auto"
        self.subject_mask_debug = False
        self.subject_mask_keyframe_interval = 8
        self.subject_mask_ema = 0.65
        self.subject_mask_frame_idx = 0
        self.prev_subject_mask_map = None
        self._sam2_mask_generator = None
        self._sam2_unavailable = False
        self._sam2_load_error: Optional[str] = None
        self.use_softedge = False
        self.softedge_scale = 0.0
        self.depth_scale = 1.0
        self.softedge_mode = "classical"
        self.softedge_debug = False
        self.enable_stylize_preprocess = False
        self.depth_blur_sigma = 1.2
        self.depth_power = 0.2
        self.depth_smoothstep_min = 0.5
        self.depth_smoothstep_max = 1.0
        self.saturation_scale = 0.2
        self.image_blur_sigma = 2.0
        self.fog_white_mix = 0.5
        self.outline_blur_sigma = 0.8
        self.outline_power = 0.1
        self.outline_smoothstep_min = 0.57
        self.outline_smoothstep_max = 1.0
        self.fog_depth_mask_blend = 0.5
        self.subject_mask_fog_blur_sigma = 1.0
        self.last_processed_fog_lerp_map = None
        self._pidinet_detector = None
        self._pidinet_unavailable = False
        self._pidinet_load_error: Optional[str] = None
        self._last_softedge_path_key: Optional[Tuple[str, str]] = None

        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.depth_detector = pipe.depth_model
        self.inference_time_ema = 0
        self.depth_time_ema = 0
        self.softedge_time_ema = 0.0
        self.subject_mask_time_ema = 0.0
        self.inference_time_list = []
        self.depth_time_list = []
        self.softedge_time_list = []
        self.subject_mask_time_list = []
        self.mask_shift = 1
        self.noise_control_enabled = False
        self.motion_score = None
        self.mean_flow_mag = None
        self.current_flow = None
        self.motion_low_threshold = 0.015
        self.motion_high_threshold = 0.06
        self.min_noise_rate = 0.0
        self.max_noise_rate = 1.0
        self.enable_warped_noise = False
        self.warped_noise_reuse = 0.85
        self.warped_noise_residual = 0.15
        self.current_noise_rate = 1.0 if do_add_noise else 0.0
        self.last_noise_rate = self.current_noise_rate
        self.prev_noise_latent = None
        self.prev_buffer_noise = None
        self.key_step_sync_enabled = False
        self.key_step_sync_index = 0
        self.key_step_sync_strength = 0.0
        self.key_step_sync_low_threshold = 0.015
        self.key_step_sync_high_threshold = 0.06
        self.key_step_sync_weight = 0.0
        self.key_step_sync_memory_valid = False
        self.key_step_sync_flow_backend = "farneback"
        self.prev_sync_x0 = None

        self.is_tensorrt = False

    def prepare_cache(self, height, width, denoising_steps_num):
        kv_cache_list = self.pipe.prepare_cache(
            height=height,
            width=width,
            denoising_steps_num=denoising_steps_num,
        )
        self.pipe.prepare_warmup_unet(height=height, width=width, unet=self.unet_warmup)
        self.kv_cache_list = kv_cache_list

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(device)
        t_index = list(range(len(timesteps)))

        return t_index, timesteps

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict,
            adapter_name,
            **kwargs,
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(
        self,
        threshold: float = 0.98,
        max_skip_frame: float = 10,
    ) -> None:
        # self.similar_image_filter = True
        # self.similar_filter.set_threshold(threshold)
        # self.similar_filter.set_max_skip_frame(max_skip_frame)
        pass

    def disable_similar_image_filter(self) -> None:
        # self.similar_image_filter = False
        pass

    def set_conditioning(
        self,
        use_softedge: Optional[bool] = None,
        softedge_scale: Optional[float] = None,
        depth_scale: Optional[float] = None,
        softedge_mode: Optional[str] = None,
        softedge_debug: Optional[bool] = None,
        subject_mask_debug: Optional[bool] = None,
        subject_mask_backend: Optional[str] = None,
        subject_mask_keyframe_interval: Optional[int] = None,
        subject_mask_ema: Optional[float] = None,
    ) -> None:
        if use_softedge is not None:
            self.use_softedge = bool(use_softedge)
        if softedge_scale is not None:
            self.softedge_scale = max(float(softedge_scale), 0.0)
        if depth_scale is not None:
            self.depth_scale = max(float(depth_scale), 0.0)
        if softedge_mode is not None:
            softedge_mode = str(softedge_mode).lower()
            self.softedge_mode = softedge_mode if softedge_mode in {"classical", "pidinet"} else "classical"
        if softedge_debug is not None:
            self.softedge_debug = bool(softedge_debug)
        if subject_mask_debug is not None:
            self.subject_mask_debug = bool(subject_mask_debug)
        if subject_mask_backend is not None:
            backend = str(subject_mask_backend).lower()
            self.subject_mask_requested_backend = backend if backend in {"auto", "sam2", "heuristic"} else "auto"
        if subject_mask_keyframe_interval is not None:
            self.subject_mask_keyframe_interval = max(int(subject_mask_keyframe_interval), 1)
        if subject_mask_ema is not None:
            self.subject_mask_ema = min(max(float(subject_mask_ema), 0.0), 0.98)

    def set_stylization(
        self,
        enable_stylize_preprocess: Optional[bool] = None,
        depth_blur_sigma: Optional[float] = None,
        depth_power: Optional[float] = None,
        depth_smoothstep_min: Optional[float] = None,
        depth_smoothstep_max: Optional[float] = None,
        saturation_scale: Optional[float] = None,
        image_blur_sigma: Optional[float] = None,
        fog_white_mix: Optional[float] = None,
        outline_blur_sigma: Optional[float] = None,
        outline_power: Optional[float] = None,
        outline_smoothstep_min: Optional[float] = None,
        outline_smoothstep_max: Optional[float] = None,
        fog_depth_mask_blend: Optional[float] = None,
        subject_mask_fog_blur_sigma: Optional[float] = None,
    ) -> None:
        if enable_stylize_preprocess is not None:
            self.enable_stylize_preprocess = bool(enable_stylize_preprocess)
        if depth_blur_sigma is not None:
            self.depth_blur_sigma = max(float(depth_blur_sigma), 0.0)
        if depth_power is not None:
            self.depth_power = max(float(depth_power), 1e-4)
        if depth_smoothstep_min is not None:
            self.depth_smoothstep_min = float(depth_smoothstep_min)
        if depth_smoothstep_max is not None:
            self.depth_smoothstep_max = float(depth_smoothstep_max)
        if self.depth_smoothstep_min > self.depth_smoothstep_max:
            self.depth_smoothstep_min, self.depth_smoothstep_max = (
                self.depth_smoothstep_max,
                self.depth_smoothstep_min,
            )
        if saturation_scale is not None:
            self.saturation_scale = min(max(float(saturation_scale), 0.0), 1.0)
        if image_blur_sigma is not None:
            self.image_blur_sigma = max(float(image_blur_sigma), 0.0)
        if fog_white_mix is not None:
            self.fog_white_mix = min(max(float(fog_white_mix), 0.0), 1.0)
        if outline_blur_sigma is not None:
            self.outline_blur_sigma = max(float(outline_blur_sigma), 0.0)
        if outline_power is not None:
            self.outline_power = max(float(outline_power), 1e-4)
        if outline_smoothstep_min is not None:
            self.outline_smoothstep_min = float(outline_smoothstep_min)
        if outline_smoothstep_max is not None:
            self.outline_smoothstep_max = float(outline_smoothstep_max)
        if self.outline_smoothstep_min > self.outline_smoothstep_max:
            self.outline_smoothstep_min, self.outline_smoothstep_max = (
                self.outline_smoothstep_max,
                self.outline_smoothstep_min,
            )
        if fog_depth_mask_blend is not None:
            self.fog_depth_mask_blend = min(max(float(fog_depth_mask_blend), 0.0), 1.0)
        if subject_mask_fog_blur_sigma is not None:
            self.subject_mask_fog_blur_sigma = max(float(subject_mask_fog_blur_sigma), 0.0)

    def reset_noise_state(self) -> None:
        self.motion_score = None
        self.mean_flow_mag = None
        self.current_flow = None
        self.prev_noise_latent = None
        self.prev_buffer_noise = None
        self.prev_sync_x0 = None
        self.key_step_sync_weight = 0.0
        self.key_step_sync_memory_valid = False
        self.prev_subject_mask_map = None
        self.last_subject_mask_map = None
        self.subject_mask_frame_idx = 0
        self.current_noise_rate = 1.0 if self.do_add_noise else 0.0
        self.last_noise_rate = self.current_noise_rate

    def set_key_step_sync(
        self,
        enabled: Optional[bool] = None,
        key_step_index: Optional[int] = None,
        strength: Optional[float] = None,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        flow_backend: Optional[str] = None,
    ) -> None:
        if enabled is not None:
            self.key_step_sync_enabled = bool(enabled)
        if key_step_index is not None:
            self.key_step_sync_index = max(int(key_step_index), 0)
        if strength is not None:
            self.key_step_sync_strength = min(max(float(strength), 0.0), 1.0)
        if low_threshold is not None:
            self.key_step_sync_low_threshold = max(float(low_threshold), 0.0)
        if high_threshold is not None:
            self.key_step_sync_high_threshold = max(
                float(high_threshold), self.key_step_sync_low_threshold + 1e-6
            )
        if flow_backend is not None:
            backend = str(flow_backend).lower().replace("_", "-")
            self.key_step_sync_flow_backend = backend if backend in {"farneback", "raft", "sea-raft"} else "farneback"

    def set_noise_control(
        self,
        enabled: Optional[bool] = None,
        motion_score: Optional[float] = None,
        mean_flow_mag: Optional[float] = None,
        flow: Optional[np.ndarray] = None,
        motion_low_threshold: Optional[float] = None,
        motion_high_threshold: Optional[float] = None,
        min_noise_rate: Optional[float] = None,
        max_noise_rate: Optional[float] = None,
        enable_warped_noise: Optional[bool] = None,
        warped_noise_reuse: Optional[float] = None,
        warped_noise_residual: Optional[float] = None,
    ) -> None:
        if enabled is not None:
            self.noise_control_enabled = bool(enabled)
        if motion_low_threshold is not None:
            self.motion_low_threshold = max(float(motion_low_threshold), 0.0)
        if motion_high_threshold is not None:
            self.motion_high_threshold = max(float(motion_high_threshold), self.motion_low_threshold + 1e-6)
        if min_noise_rate is not None:
            self.min_noise_rate = min(max(float(min_noise_rate), 0.0), 1.0)
        if max_noise_rate is not None:
            self.max_noise_rate = min(max(float(max_noise_rate), 0.0), 1.0)
        if self.min_noise_rate > self.max_noise_rate:
            self.min_noise_rate, self.max_noise_rate = self.max_noise_rate, self.min_noise_rate
        if enable_warped_noise is not None:
            self.enable_warped_noise = bool(enable_warped_noise)
        if warped_noise_reuse is not None:
            self.warped_noise_reuse = min(max(float(warped_noise_reuse), 0.0), 1.0)
        if warped_noise_residual is not None:
            self.warped_noise_residual = min(max(float(warped_noise_residual), 0.0), 1.0)
        self.motion_score = None if motion_score is None else max(float(motion_score), 0.0)
        self.mean_flow_mag = None if mean_flow_mag is None else max(float(mean_flow_mag), 0.0)
        self.current_flow = None if flow is None else np.asarray(flow, dtype=np.float32).copy()
        self.current_noise_rate = self._resolve_noise_rate()
        self.last_noise_rate = self.current_noise_rate

    def _resolve_noise_rate(self) -> float:
        if not self.do_add_noise:
            return 0.0
        if not self.noise_control_enabled or self.motion_score is None:
            return 1.0
        denom = max(self.motion_high_threshold - self.motion_low_threshold, 1e-6)
        ratio = (self.motion_score - self.motion_low_threshold) / denom
        ratio = min(max(ratio, 0.0), 1.0)
        noise_rate = self.min_noise_rate + ratio * (self.max_noise_rate - self.min_noise_rate)
        return min(max(noise_rate, 0.0), 1.0)

    def _fresh_noise_like(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            ref.shape,
            device=ref.device,
            dtype=ref.dtype,
            generator=self.generator,
        )

    def _warp_noise_like(self, prev_noise: torch.Tensor, target_shape: torch.Size) -> Optional[torch.Tensor]:
        if self.current_flow is None:
            return None
        target_h, target_w = target_shape[-2], target_shape[-1]
        flow_h, flow_w = self.current_flow.shape[:2]
        if target_h <= 0 or target_w <= 0 or flow_h <= 1 or flow_w <= 1:
            return None
        flow = torch.from_numpy(self.current_flow).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        flow = F.interpolate(flow, size=(target_h, target_w), mode="bilinear", align_corners=False)
        flow[:, 0] *= float(target_w) / float(flow_w)
        flow[:, 1] *= float(target_h) / float(flow_h)
        ys = torch.linspace(-1.0, 1.0, target_h, device=self.device, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, target_w, device=self.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        flow_x = flow[:, 0] * (2.0 / max(target_w - 1, 1))
        flow_y = flow[:, 1] * (2.0 / max(target_h - 1, 1))
        grid = grid + torch.stack((flow_x, flow_y), dim=-1)
        if prev_noise.ndim == 4:
            warped_in = prev_noise.to(dtype=self.dtype)
            restore = lambda x: x
        elif prev_noise.ndim == 5:
            batch, channels, frames = prev_noise.shape[:3]
            warped_in = prev_noise.permute(0, 2, 1, 3, 4).reshape(-1, channels, target_h, target_w).to(dtype=self.dtype)
            restore = lambda x: x.reshape(batch, frames, channels, target_h, target_w).permute(0, 2, 1, 3, 4)
        else:
            return None
        warped = F.grid_sample(
            warped_in,
            grid.repeat(warped_in.shape[0], 1, 1, 1).to(dtype=warped_in.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return restore(warped)

    def _sample_noise_like(self, ref: torch.Tensor, prev_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        fresh_noise = self._fresh_noise_like(ref)
        if (
            not self.enable_warped_noise
            or prev_noise is None
            or self.current_flow is None
        ):
            return fresh_noise * self.current_noise_rate
        warped_noise = self._warp_noise_like(prev_noise.to(device=self.device, dtype=ref.dtype), ref.shape)
        if warped_noise is None:
            return fresh_noise * self.current_noise_rate
        fresh_ratio = max(self.current_noise_rate, self.warped_noise_residual)
        reuse_ratio = max(0.0, 1.0 - fresh_ratio) * self.warped_noise_reuse
        mix_sum = fresh_ratio + reuse_ratio
        if mix_sum <= 1e-6:
            return fresh_noise * 0.0
        return (fresh_ratio * fresh_noise + reuse_ratio * warped_noise) / mix_sum

    def initialize_softedge_branch(self) -> None:
        if hasattr(self.unet, "init_softedge_from_depth"):
            self.unet.init_softedge_from_depth()
        if hasattr(self, "unet_warmup") and hasattr(self.unet_warmup, "init_softedge_from_depth"):
            self.unet_warmup.init_softedge_from_depth()

    def _gaussian_blur(
        self,
        x: torch.Tensor,
        sigma: float = 1.0,
        kernel_size: Optional[int] = None,
        repeats: int = 1,
    ) -> torch.Tensor:
        sigma = float(sigma)
        if sigma <= 0:
            return x
        if kernel_size is None:
            kernel_size = max(3, int(round(sigma * 6)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        radius = kernel_size // 2
        coords = torch.arange(kernel_size, device=x.device, dtype=torch.float32) - radius
        kernel_1d = torch.exp(-(coords.square()) / (2 * sigma * sigma))
        kernel_1d = (kernel_1d / kernel_1d.sum()).to(dtype=x.dtype)
        kernel_x = kernel_1d.view(1, 1, 1, kernel_size).repeat(x.shape[1], 1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, kernel_size, 1).repeat(x.shape[1], 1, 1, 1)
        for _ in range(max(int(repeats), 1)):
            x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
            x = F.conv2d(x, kernel_x, groups=x.shape[1])
            x = F.pad(x, (0, 0, radius, radius), mode="reflect")
            x = F.conv2d(x, kernel_y, groups=x.shape[1])
        return x

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, t: Union[torch.Tensor, float]) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=a.device, dtype=a.dtype)
        return a + (b - a) * t

    def _smoothstep(self, x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
        t = ((x - edge0) / max(edge1 - edge0, 1e-6)).clamp(0, 1)
        return t * t * (3.0 - 2.0 * t)

    def _to_unit_range(self, x: torch.Tensor) -> torch.Tensor:
        return ((x + 1.0) * 0.5).clamp(0, 1)

    def _to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 2.0 - 1.0).clamp(-1, 1)

    def _ensure_three_channels(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            return x
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] > 3:
            return x[:, :3]
        return x[:, :1].repeat(1, 3, 1, 1)

    def _desaturate(self, image: torch.Tensor, saturation_scale: float) -> torch.Tensor:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        return self._lerp(gray.repeat(1, 3, 1, 1), image, saturation_scale)

    def _needs_softedge_map(self) -> bool:
        return self.enable_stylize_preprocess or self.use_softedge or self.softedge_debug

    def _needs_subject_mask_map(self) -> bool:
        if self.subject_mask_debug:
            return True
        return self.enable_stylize_preprocess

    def _clear_stylization_debug(self) -> None:
        self.last_stylized_map = None
        self.last_processed_depth_map = None
        self.last_processed_fog_lerp_map = None
        self.last_processed_outline_map = None

    def _normalize_condition_map(self, cond_map: torch.Tensor) -> torch.Tensor:
        cond_min = cond_map.amin(dim=(-2, -1), keepdim=True)
        cond_max = cond_map.amax(dim=(-2, -1), keepdim=True)
        cond_map = (cond_map - cond_min) / (cond_max - cond_min).clamp_min(1e-6)
        return cond_map

    def _encode_condition_map(self, cond_map: torch.Tensor) -> torch.Tensor:
        cond_for_vae = (cond_map + 1.0) * 0.5
        cond_latent = retrieve_latents(self.vae.encode(cond_for_vae.to(dtype=self.vae.dtype)), self.generator)
        cond_latent = cond_latent * self.vae.config.scaling_factor
        return cond_latent

    def _build_classical_softedge_map(self, image_tensors: torch.Tensor) -> torch.Tensor:
        h, w = image_tensors.shape[2], image_tensors.shape[3]
        rgb = (image_tensors.to(dtype=self.dtype) + 1.0) * 0.5
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        gray = F.interpolate(gray, (384, 384), mode="bilinear", align_corners=False)
        gray = self._gaussian_blur(gray)

        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        edge = self._gaussian_blur(edge)
        edge = self._normalize_condition_map(edge).pow(0.75)
        edge = F.interpolate(edge, (h, w), mode="bilinear", align_corners=False)
        return edge.repeat(1, 3, 1, 1) * 2 - 1

    def _get_pidinet_detector(self):
        if self._pidinet_unavailable:
            return None
        if self._pidinet_detector is not None:
            return self._pidinet_detector

        try:
            from controlnet_aux import PidiNetDetector

            self._pidinet_detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            self._pidinet_load_error = None
            logger.info("PiDiNet: PidiNetDetector loaded (lllyasviel/Annotators)")
        except Exception as e:
            self._pidinet_unavailable = True
            self._pidinet_detector = None
            self._pidinet_load_error = f"{type(e).__name__}: {e}"
            el = str(e).lower()
            net_hint = ""
            if any(
                k in el
                for k in (
                    "timeout",
                    "connection",
                    "network",
                    "huggingface",
                    "locate the file on the hub",
                    "max retries",
                )
            ):
                net_hint = (
                    " Network/model cache: try HF_ENDPOINT=https://hf-mirror.com (or proxy), "
                    "or download lllyasviel/Annotators table5_pidinet.pth into HF cache, then restart."
                )
            logger.warning(
                "PiDiNet unavailable (pip install 'controlnet-aux' or pip install -e '.[pidinet]').%s %s",
                net_hint,
                self._pidinet_load_error,
            )
        return self._pidinet_detector

    def _build_pidinet_softedge_map(self, image_tensors: torch.Tensor) -> Optional[torch.Tensor]:
        detector = self._get_pidinet_detector()
        if detector is None:
            return None

        h, w = image_tensors.shape[2], image_tensors.shape[3]
        softedge_list = []
        for image_tensor in image_tensors:
            image_np = ((image_tensor.detach().float().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255)
            edge_image = detector(image_np.astype(np.uint8), safe=True)
            edge_np = np.array(edge_image, dtype=np.float32)
            if edge_np.ndim == 2:
                edge_np = edge_np[..., None]
            edge_tensor = torch.from_numpy(edge_np).to(device=self.device, dtype=self.dtype) / 255.0
            edge_tensor = edge_tensor.permute(2, 0, 1)
            softedge_list.append(edge_tensor)

        softedge = torch.stack(softedge_list, dim=0)
        softedge = self._ensure_three_channels(softedge)
        softedge = F.interpolate(softedge, (h, w), mode="bilinear", align_corners=False)
        softedge = self._normalize_condition_map(softedge)
        return softedge * 2 - 1

    def build_softedge_map(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(device=self.device, dtype=self.dtype)
        softedge_map = None
        backend = "classical"
        if self.softedge_mode == "pidinet":
            softedge_map = self._build_pidinet_softedge_map(image_tensors)
            backend = "pidinet" if softedge_map is not None else "classical_fallback"
        if softedge_map is None:
            softedge_map = self._build_classical_softedge_map(image_tensors)
        softedge_map = self._ensure_three_channels(softedge_map)
        self.last_softedge_map = softedge_map.detach().float().cpu()
        self.softedge_backend = backend
        path_key = (self.softedge_mode, backend)
        if path_key != self._last_softedge_path_key:
            self._last_softedge_path_key = path_key
            print(f"[softedge] active: mode={self.softedge_mode} backend={backend}", flush=True)
            if backend == "classical_fallback" and self._pidinet_load_error:
                print(f"[softedge] pidinet error: {self._pidinet_load_error}", flush=True)
                el = self._pidinet_load_error.lower()
                if "modulenotfound" in el.replace(" ", ""):
                    print("[softedge] hint: pip install 'controlnet-aux' 或 pip install -e '.[pidinet]'", flush=True)
                elif any(
                    k in el
                    for k in (
                        "timeout",
                        "connection",
                        "huggingface",
                        "locate the file",
                        "max retries",
                    )
                ):
                    print(
                        "[softedge] hint: export HF_ENDPOINT=https://hf-mirror.com 或配置代理后重启；"
                        "也可手动缓存 lllyasviel/Annotators 的 table5_pidinet.pth",
                        flush=True,
                    )
        return softedge_map

    def encode_softedge_map(self, softedge_map: torch.Tensor) -> torch.Tensor:
        return self._encode_condition_map(softedge_map)

    def encode_softedge(self, image_tensors: torch.Tensor) -> torch.Tensor:
        return self.encode_softedge_map(self.build_softedge_map(image_tensors))

    def _get_sam2_mask_generator(self):
        if self._sam2_unavailable:
            return None
        if self._sam2_mask_generator is not None:
            return self._sam2_mask_generator
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2

            model_cfg = getattr(self.pipe, "sam2_model_cfg", None) or os.getenv("SAM2_MODEL_CFG")
            checkpoint = getattr(self.pipe, "sam2_checkpoint", None) or os.getenv("SAM2_CHECKPOINT")
            if not model_cfg or not checkpoint:
                raise RuntimeError("missing sam2_model_cfg or sam2_checkpoint")
            sam2 = build_sam2(model_cfg, checkpoint, device=self.device)
            self._sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2)
            self._sam2_load_error = None
        except Exception as e:
            self._sam2_unavailable = True
            self._sam2_mask_generator = None
            self._sam2_load_error = f"{type(e).__name__}: {e}"
            logger.warning("SAM2 subject mask unavailable; falling back to heuristic. %s", self._sam2_load_error)
        return self._sam2_mask_generator

    def _center_prior(self, batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        prior = torch.exp(-(grid_x.square() / 0.65 + grid_y.square() / 0.9))
        return prior.view(1, 1, height, width).repeat(batch, 1, 1, 1)

    def _morph_close_open(self, mask: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
        pad = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
        closed = -F.max_pool2d(-dilated, kernel_size, stride=1, padding=pad)
        eroded = -F.max_pool2d(-closed, kernel_size, stride=1, padding=pad)
        return F.max_pool2d(eroded, kernel_size, stride=1, padding=pad)

    def _warp_subject_mask(self, target_shape: torch.Size) -> Optional[torch.Tensor]:
        if self.prev_subject_mask_map is None:
            return None
        prev = self.prev_subject_mask_map.to(device=self.device, dtype=self.dtype)
        return self._warp_noise_like(prev, target_shape)

    def _build_sam2_subject_mask_map(self, image_tensors: torch.Tensor) -> Optional[torch.Tensor]:
        generator = self._get_sam2_mask_generator()
        if generator is None:
            return None
        masks = []
        h, w = image_tensors.shape[2], image_tensors.shape[3]
        center = self._center_prior(1, h, w, image_tensors.device, torch.float32)[0, 0].cpu().numpy()
        for image_tensor in image_tensors:
            image_np = ((image_tensor.detach().float().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255)
            candidates = generator.generate(image_np.astype(np.uint8))
            if not candidates:
                return None
            best_mask = None
            best_score = -1.0
            for candidate in candidates:
                seg = np.asarray(candidate.get("segmentation"), dtype=np.float32)
                area = float(seg.mean())
                if area < 0.02 or area > 0.85:
                    continue
                score = float((seg * center).mean() / max(area, 1e-6))
                score += 0.15 * float(candidate.get("predicted_iou", 0.0))
                score += 0.10 * float(candidate.get("stability_score", 0.0))
                if score > best_score:
                    best_score = score
                    best_mask = seg
            if best_mask is None:
                return None
            masks.append(torch.from_numpy(best_mask).to(device=self.device, dtype=self.dtype).view(1, h, w))
        return torch.stack(masks, dim=0)

    def _build_heuristic_subject_mask_map(
        self,
        image_tensors: torch.Tensor,
        depth_map: torch.Tensor,
        softedge_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, _, height, width = image_tensors.shape
        depth = self._to_unit_range(depth_map.to(device=self.device, dtype=self.dtype))[:, :1]
        center = self._center_prior(batch, height, width, self.device, self.dtype)
        border = torch.cat(
            [
                depth[:, :, : max(height // 8, 1), :].flatten(2),
                depth[:, :, -max(height // 8, 1) :, :].flatten(2),
                depth[:, :, :, : max(width // 8, 1)].flatten(2),
                depth[:, :, :, -max(width // 8, 1) :].flatten(2),
            ],
            dim=2,
        ).mean(dim=2, keepdim=True).view(batch, 1, 1, 1)
        center_depth = (depth * center).flatten(2).sum(dim=2, keepdim=True).view(batch, 1, 1, 1)
        center_depth = center_depth / center.flatten(2).sum(dim=2, keepdim=True).view(batch, 1, 1, 1).clamp_min(1e-6)
        depth_fg = torch.where(center_depth >= border, depth, 1.0 - depth)
        depth_fg = self._smoothstep(depth_fg, 0.42, 0.78)

        motion = torch.zeros_like(depth_fg)
        if self.current_flow is not None:
            flow = torch.from_numpy(self.current_flow).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)
            flow = F.interpolate(flow, (height, width), mode="bilinear", align_corners=False)
            mag = torch.sqrt(flow[:, 0:1].square() + flow[:, 1:2].square() + 1e-6)
            motion = self._normalize_condition_map(mag)

        edge_prior = torch.zeros_like(depth_fg)
        if softedge_map is not None:
            edge_prior = self._gaussian_blur(self._to_unit_range(softedge_map.to(dtype=self.dtype))[:, :1], sigma=2.0)

        mask = 0.58 * depth_fg + 0.24 * center + 0.12 * motion + 0.06 * edge_prior
        mask = self._smoothstep(mask, 0.48, 0.76)
        mask = self._morph_close_open(mask, kernel_size=7)
        mask = self._gaussian_blur(mask, sigma=1.4)
        return mask.clamp(0, 1)

    def build_subject_mask_map(
        self,
        image_tensors: torch.Tensor,
        depth_map: torch.Tensor,
        softedge_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_tensors = image_tensors.to(device=self.device, dtype=self.dtype)
        use_keyframe = self.subject_mask_frame_idx % self.subject_mask_keyframe_interval == 0
        mask = None
        backend = "heuristic"
        if self.subject_mask_requested_backend in {"auto", "sam2"} and use_keyframe:
            mask = self._build_sam2_subject_mask_map(image_tensors)
            if mask is not None:
                backend = "sam2"
        if mask is None:
            mask = self._build_heuristic_subject_mask_map(image_tensors, depth_map, softedge_map)
        mask = mask[:, :1].clamp(0, 1)
        warped = self._warp_subject_mask(mask.shape)
        if warped is not None and warped.shape == mask.shape:
            motion_score = 0.0 if self.motion_score is None else min(max(float(self.motion_score), 0.0), 1.0)
            ema = self.subject_mask_ema * (1.0 - min(motion_score * 4.0, 0.45))
            mask = (ema * warped.to(dtype=mask.dtype) + (1.0 - ema) * mask).clamp(0, 1)
        mask = self._gaussian_blur(mask, sigma=0.8)
        mask_rgb = mask.repeat(1, 3, 1, 1) * 2 - 1
        self.last_subject_mask_map = mask_rgb.detach().float().cpu()
        self.prev_subject_mask_map = mask[-1:].detach().float().cpu()
        self.subject_mask_backend = backend
        self.subject_mask_frame_idx += int(mask.shape[0])
        return mask[:, :1].detach()

    @torch.no_grad()
    def prepare(
        self,
        warmup_frames: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = None,
        seed: int = 2,
    ) -> None:
        """
        Forward warm-up frames and fill the buffer
        images: [warmup_size, 3, h, w] in [0, 1]
        """

        if generator is None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(seed)
        else:
            self.generator = generator
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    1,  # for video
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )

            self.depth_latent_buffer = torch.zeros_like(self.x_t_latent_buffer)
            self.softedge_latent_buffer = torch.zeros_like(self.x_t_latent_buffer)
        else:
            self.x_t_latent_buffer = None
            self.depth_latent_buffer = None
            self.softedge_latent_buffer = None

        self.attn_bias, self.pe_idx, self.update_idx = self.initialize_attn_bias_pe_and_update_idx()

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        encoder_output = self.pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            clip_skip=self.clip_skip,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize" or self.cfg_type == "full"):
            self.prompt_embeds = torch.cat([uncond_prompt_embeds, self.prompt_embeds], dim=0)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, WARMUP_FRAMES, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list).view(len(self.t_list), 1, 1, 1, 1).to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list).view(len(self.t_list), 1, 1, 1, 1).to(dtype=self.dtype, device=self.device)
        )
        # print(self.c_skip)

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        # do warmup
        # 1. encode images
        warmup_x_list = []
        for f in warmup_frames:
            x = self.image_processor.preprocess(f, self.height, self.width)
            warmup_x_list.append(x.to(device=self.device, dtype=self.dtype))
        warmup_x = torch.cat(warmup_x_list, dim=0)  # [warmup_size, c, h, w]
        depth_map = self.build_depth_map(warmup_x)
        depth_latent = self.encode_depth_map(depth_map)
        softedge_map = self.build_softedge_map(warmup_x) if self._needs_softedge_map() else None
        if softedge_map is None:
            self.last_softedge_map = None
            self.softedge_backend = "disabled"
        subject_mask_01 = None
        if self._needs_subject_mask_map():
            subject_mask_01 = self.build_subject_mask_map(warmup_x, depth_map, softedge_map)
        else:
            self.last_subject_mask_map = None
            self.prev_subject_mask_map = None
            self.subject_mask_backend = "disabled"
        if self.enable_stylize_preprocess and softedge_map is not None:
            warmup_x = self.build_stylized_image(
                warmup_x,
                depth_map,
                softedge_map,
                subject_mask_01=subject_mask_01,
            )
        else:
            self._clear_stylization_debug()
        warmup_x_t = self.encode_image(warmup_x)
        x_t_latent = rearrange(warmup_x_t, "f c h w -> c f h w")[None, ...]
        depth_latent = rearrange(depth_latent, "f c h w -> c f h w")[None, ...]
        if self.use_softedge or self.softedge_debug:
            if softedge_map is None:
                softedge_map = self.build_softedge_map(warmup_x)
            softedge_latent = self.encode_softedge_map(softedge_map)
            softedge_latent = rearrange(softedge_latent, "f c h w -> c f h w")[None, ...]
        else:
            softedge_latent = torch.zeros_like(depth_latent)

        # 2. run warmup denoising
        self.unet_warmup = self.unet_warmup.to(device="cuda", dtype=self.dtype)
        warmup_prompt = self.prompt_embeds[0:1]
        for idx, t in enumerate(self.sub_timesteps_tensor):
            t = t.view(1).repeat(x_t_latent.shape[0])

            output_t = self.unet_warmup(
                x_t_latent,
                t,
                temporal_attention_mask=None,
                depth_sample=depth_latent * self.depth_scale,
                softedge_sample=softedge_latent * self.softedge_scale,
                encoder_hidden_states=warmup_prompt,
                kv_cache=[cache[idx] for cache in self.kv_cache_list],
                return_dict=True,
            )
            model_pred = output_t["sample"]
            x_0_pred = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if idx < len(self.sub_timesteps_tensor) - 1:
                # x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred

                x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred + self.beta_prod_t_sqrt[
                    idx + 1
                ] * torch.randn_like(x_0_pred, device=self.device, dtype=self.dtype)

        self.unet_warmup = self.unet_warmup.to(device="cpu")
        x_0_pred = rearrange(x_0_pred, "b c f h w -> b f c h w")[0]  # [f, c, h, w]
        denoisied_frame = self.decode_image(x_0_pred)

        self.warmup_engine()
        self.reset_noise_state()

        return denoisied_frame

    def warmup_engine(self):
        """Warmup tensorrt engine."""

        if not self.is_tensorrt:
            return

        print("Warmup TensorRT engine.")
        pseudo_latent = self.init_noise[:, :, 0:1, ...]
        for _ in range(self.batch_size):
            self.unet(
                pseudo_latent,
                self.sub_timesteps_tensor,
                depth_sample=pseudo_latent,
                softedge_sample=pseudo_latent,
                encoder_hidden_states=self.prompt_embeds,
                temporal_attention_mask=self.attn_bias,
                kv_cache=self.kv_cache_list,
                pe_idx=self.pe_idx,
                update_idx=self.update_idx,
                return_dict=True,
            )
        print("Warmup TensorRT engine finished.")

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = self.alpha_prod_t_sqrt[t_index] * original_samples + self.beta_prod_t_sqrt[t_index] * noise
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch

        return denoised_batch

    def initialize_attn_bias_pe_and_update_idx(self):
        attn_mask = torch.zeros((self.denoising_steps_num, WINDOW_SIZE), dtype=torch.bool, device=self.device)
        attn_mask[:, :WARMUP_FRAMES] = True
        attn_mask[0, WARMUP_FRAMES] = True
        attn_bias = torch.zeros_like(attn_mask, dtype=self.dtype)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

        pe_idx = torch.arange(WINDOW_SIZE).unsqueeze(0).repeat(self.denoising_steps_num, 1).cuda()
        update_idx = torch.ones(self.denoising_steps_num, dtype=torch.int64, device=self.device) * WARMUP_FRAMES
        if self.denoising_steps_num > 1:
            update_idx[1] = WARMUP_FRAMES + 1

        return attn_bias, pe_idx, update_idx

    def update_attn_bias(self, attn_bias, pe_idx, update_idx):
        """
        attn_bias: (timesteps, prev_len), init value: [[0, 0, 0, inf], [0, 0, inf, inf]]
        pe_idx: (timesteps, prev_len), init value: [[0, 1, 2, 3], [0, 1, 2, 3]]
        update_idx: (timesteps, ), init value: [2, 1]
        """

        for idx in range(self.denoising_steps_num):
            # update pe_idx and update_idx based on attn_bias from last iteration
            if torch.isinf(attn_bias[idx]).any():
                # some position not filled, do not change pe
                # some position not filled, fill the last position
                update_idx[idx] = (attn_bias[idx] == 0).sum()
            else:
                # all position are filled, roll pe
                pe_idx[idx, WARMUP_FRAMES:] = pe_idx[idx, WARMUP_FRAMES:].roll(shifts=1, dims=0)
                # all position are filled, fill the position with largest PE
                update_idx[idx] = pe_idx[idx].argmax()

            num_unmask = (attn_bias[idx] == 0).sum()
            attn_bias[idx, : min(num_unmask + 1, WINDOW_SIZE)] = 0

        return attn_bias, pe_idx, update_idx

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        depth_latent: torch.Tensor,
        softedge_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        output = self.unet(
            x_t_latent_plus_uc,
            t_list,
            depth_sample=depth_latent * self.depth_scale,
            softedge_sample=softedge_latent * self.softedge_scale,
            encoder_hidden_states=self.prompt_embeds,
            temporal_attention_mask=self.attn_bias,
            kv_cache=self.kv_cache_list,
            pe_idx=self.pe_idx,
            update_idx=self.update_idx,
            return_dict=True,
        )
        model_pred = output["sample"]
        kv_cache_list = output["kv_cache"]
        self.kv_cache_list = kv_cache_list

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (self.cfg_type == "self" or self.cfg_type == "initialize"):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat([self.init_noise[1:], self.init_noise[0:1]], dim=0)
                self.stock_noise = init_noise + delta_x

        else:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        image_tensors: [f, c, h, w]
        """
        # num_frames = image_tensors.shape[2]
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        noise = self._sample_noise_like(img_latent, self.prev_noise_latent)
        self.prev_noise_latent = noise.detach().clone()
        x_t_latent = self.add_noise(img_latent, noise, 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        """
        x_0_pred: [f, c, h, w]
        """
        output_latent = self.vae.decode(x_0_pred_out / self.vae.config.scaling_factor, return_dict=False)[0]
        return output_latent.clip(-1, 1)

    def build_depth_map(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        image_tensor: [f, c, h, w], [-1, 1]
        """
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.depth_detector.dtype,
        )
        # depth_map = self.depth_detector(image_tensors)
        # depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_map_norm = depth_map_norm[:, None].repeat(1, 3, 1, 1) * 2 - 1
        # depth_latent = retrieve_latents(self.vae.encode(depth_map_norm.to(dtype=self.vae.dtype)), self.generator)
        # depth_latent = depth_latent * self.vae.config.scaling_factor
        # return depth_latent

        # preprocess
        h, w = image_tensors.shape[2], image_tensors.shape[3]
        images_input = F.interpolate(image_tensors, (384, 384), mode="bilinear", align_corners=False)
        # forward
        depth_map = self.depth_detector(images_input)
        # postprocess
        depth_min = depth_map.amin(dim=(-2, -1), keepdim=True)
        depth_max = depth_map.amax(dim=(-2, -1), keepdim=True)
        depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min).clamp_min(1e-6)
        depth_map_norm = depth_map_norm[:, None].repeat(1, 3, 1, 1) * 2 - 1
        depth_map_norm = F.interpolate(depth_map_norm, (h, w), mode="bilinear", align_corners=False)
        self.last_depth_map = depth_map_norm.detach().float().cpu()
        return depth_map_norm

    def encode_depth_map(self, depth_map: torch.Tensor) -> torch.Tensor:
        return self._encode_condition_map(depth_map)

    def encode_depth(self, image_tensors: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.encode_depth_map(self.build_depth_map(image_tensors))

    def build_stylized_image(
        self,
        image_tensors: torch.Tensor,
        depth_map: torch.Tensor,
        softedge_map: torch.Tensor,
        subject_mask_01: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image = self._desaturate(self._to_unit_range(image_tensors.to(dtype=self.dtype)), self.saturation_scale)
        depth = self._to_unit_range(depth_map.to(dtype=self.dtype))
        outline = self._to_unit_range(self._ensure_three_channels(softedge_map.to(dtype=self.dtype)))

        pro_depth = self._gaussian_blur(depth, sigma=self.depth_blur_sigma).pow(self.depth_power)
        pro_depth = self._smoothstep(pro_depth, self.depth_smoothstep_min, self.depth_smoothstep_max)

        blurred_image = self._gaussian_blur(
            image,
            sigma=self.image_blur_sigma,
        )
        fog_base = self._lerp(blurred_image, torch.ones_like(blurred_image), self.fog_white_mix)
        fog_lerp = pro_depth
        if subject_mask_01 is not None:
            pro_mask = subject_mask_01.to(device=image.device, dtype=self.dtype)
            if pro_mask.shape[1] != 1:
                pro_mask = pro_mask[:, :1]
            pro_mask = 1.0 - pro_mask
            pro_mask = self._gaussian_blur(pro_mask, sigma=self.subject_mask_fog_blur_sigma)
            t = self.fog_depth_mask_blend
            fog_lerp = (1.0 - t) * pro_depth + t * pro_mask
        fog_image = self._lerp(fog_base, image, fog_lerp)

        pro_outline = self._gaussian_blur(outline, sigma=self.outline_blur_sigma).pow(self.outline_power)
        pro_outline = self._smoothstep(
            pro_outline,
            self.outline_smoothstep_min,
            self.outline_smoothstep_max,
        )

        stylized = self._lerp(fog_image, torch.zeros_like(fog_image), pro_outline).clamp(0, 1)
        self.last_processed_depth_map = self._to_model_range(pro_depth).detach().float().cpu()
        self.last_processed_fog_lerp_map = self._to_model_range(fog_lerp).detach().float().cpu()
        self.last_processed_outline_map = self._to_model_range(pro_outline).detach().float().cpu()
        self.last_stylized_map = self._to_model_range(stylized).detach().float().cpu()
        return self._to_model_range(stylized)

    def _resolve_key_step_sync_weight(self) -> float:
        if (
            not self.key_step_sync_enabled
            or self.key_step_sync_strength <= 0
            or self.prev_sync_x0 is None
            or self.current_flow is None
            or self.motion_score is None
        ):
            return 0.0
        denom = max(self.key_step_sync_high_threshold - self.key_step_sync_low_threshold, 1e-6)
        ratio = (self.motion_score - self.key_step_sync_low_threshold) / denom
        ratio = min(max(ratio, 0.0), 1.0)
        return self.key_step_sync_strength * (1.0 - ratio)

    def _apply_key_step_sync(self, x_0_pred_batch: torch.Tensor) -> torch.Tensor:
        self.key_step_sync_memory_valid = self.prev_sync_x0 is not None
        sync_idx = min(self.key_step_sync_index, max(x_0_pred_batch.shape[0] - 1, 0))
        weight = self._resolve_key_step_sync_weight()
        self.key_step_sync_weight = weight
        if weight <= 0:
            self.prev_sync_x0 = x_0_pred_batch[sync_idx : sync_idx + 1].detach().clone()
            return x_0_pred_batch

        target = x_0_pred_batch[sync_idx : sync_idx + 1]
        warped = self._warp_noise_like(
            self.prev_sync_x0.to(device=self.device, dtype=x_0_pred_batch.dtype),
            target.shape,
        )
        if warped is None:
            self.prev_sync_x0 = x_0_pred_batch[sync_idx : sync_idx + 1].detach().clone()
            self.key_step_sync_weight = 0.0
            return x_0_pred_batch

        synced = x_0_pred_batch.clone()
        synced[sync_idx : sync_idx + 1] = (1.0 - weight) * target + weight * warped
        self.prev_sync_x0 = synced[sync_idx : sync_idx + 1].detach().clone()
        return synced

    def predict_x0_batch(
        self,
        x_t_latent: torch.Tensor,
        depth_latent: torch.Tensor,
        softedge_latent: torch.Tensor,
    ) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer
        prev_depth_latent_batch = self.depth_latent_buffer
        prev_softedge_latent_batch = self.softedge_latent_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                depth_latent = torch.cat((depth_latent, prev_depth_latent_batch), dim=0)
                softedge_latent = torch.cat((softedge_latent, prev_softedge_latent_batch), dim=0)

                self.stock_noise = torch.cat((self.init_noise[0:1], self.stock_noise[:-1]), dim=0)
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, depth_latent, softedge_latent, t_list)
            x_0_pred_batch = self._apply_key_step_sync(x_0_pred_batch)
            self.attn_bias, self.pe_idx, self.update_idx = self.update_attn_bias(
                self.attn_bias, self.pe_idx, self.update_idx
            )

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                noise = self._sample_noise_like(x_0_pred_batch[:-1], self.prev_buffer_noise)
                self.prev_buffer_noise = noise.detach().clone()
                self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1] + self.beta_prod_t_sqrt[
                    1:
                ] * noise
                self.depth_latent_buffer = depth_latent[:-1]
                self.softedge_latent_buffer = softedge_latent[:-1]
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
                self.depth_latent_buffer = None
                self.softedge_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                x_0_pred, model_pred = self.unet_step(x_t_latent, depth_latent, softedge_latent, t, idx)
                if idx == self.key_step_sync_index:
                    x_0_pred = self._apply_key_step_sync(x_0_pred)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    noise = self._sample_noise_like(x_0_pred, self.prev_buffer_noise)
                    self.prev_buffer_noise = noise.detach().clone()
                    x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred + self.beta_prod_t_sqrt[idx + 1] * noise
            x_0_pred_out = x_0_pred

        return x_0_pred_out

    @torch.no_grad()
    def __call__(self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = self.image_processor.preprocess(x, self.height, self.width).to(device=self.device, dtype=self.dtype)
        # if self.similar_image_filter:
        #     x = self.similar_filter(x)
        #     if x is None:
        #         time.sleep(self.inference_time_ema)
        #         return self.prev_image_result
        start_depth = torch.cuda.Event(enable_timing=True)
        end_depth = torch.cuda.Event(enable_timing=True)
        start_depth.record()
        depth_map = self.build_depth_map(x)
        depth_latent = self.encode_depth_map(depth_map)
        end_depth.record()
        softedge_time = 0.0
        softedge_map = None
        if self._needs_softedge_map():
            start_se = torch.cuda.Event(enable_timing=True)
            end_se = torch.cuda.Event(enable_timing=True)
            start_se.record()
            softedge_map = self.build_softedge_map(x)
            end_se.record()
        else:
            self.last_softedge_map = None
            self.softedge_backend = "disabled"
        subject_mask_time = 0.0
        subject_mask_01 = None
        if self._needs_subject_mask_map():
            start_mask = torch.cuda.Event(enable_timing=True)
            end_mask = torch.cuda.Event(enable_timing=True)
            start_mask.record()
            subject_mask_01 = self.build_subject_mask_map(x, depth_map, softedge_map)
            end_mask.record()
        else:
            self.last_subject_mask_map = None
            self.prev_subject_mask_map = None
            self.subject_mask_backend = "disabled"
        if self.enable_stylize_preprocess and softedge_map is not None:
            x = self.build_stylized_image(
                x,
                depth_map,
                softedge_map,
                subject_mask_01=subject_mask_01,
            )
        else:
            self._clear_stylization_debug()
        x_t_latent = self.encode_image(x)
        if self.use_softedge or self.softedge_debug:
            if softedge_map is None:
                softedge_map = self.build_softedge_map(x)
            softedge_latent = self.encode_softedge_map(softedge_map)
        else:
            softedge_latent = torch.zeros_like(depth_latent)
        torch.cuda.synchronize()
        depth_time = start_depth.elapsed_time(end_depth) / 1000
        if self._needs_softedge_map():
            softedge_time = start_se.elapsed_time(end_se) / 1000
        if self._needs_subject_mask_map():
            subject_mask_time = start_mask.elapsed_time(end_mask) / 1000

        x_t_latent = x_t_latent.unsqueeze(2)
        depth_latent = depth_latent.unsqueeze(2)
        softedge_latent = softedge_latent.unsqueeze(2)
        if self.denoising_steps_num <= 1:
            self.prev_buffer_noise = None
        x_0_pred_out = self.predict_x0_batch(x_t_latent, depth_latent, softedge_latent)  # [1, c, 1, h, w]
        x_0_pred_out = rearrange(x_0_pred_out, "b c f h w -> (b f) c h w")
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        # self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        self.depth_time_ema = 0.9 * self.depth_time_ema + 0.1 * depth_time
        self.inference_time_list.append(inference_time)
        self.depth_time_list.append(depth_time)
        if self._needs_softedge_map():
            self.softedge_time_ema = 0.9 * self.softedge_time_ema + 0.1 * softedge_time
            self.softedge_time_list.append(softedge_time)
        else:
            self.softedge_time_ema = 0.0
        if self._needs_subject_mask_map():
            self.subject_mask_time_ema = 0.9 * self.subject_mask_time_ema + 0.1 * subject_mask_time
            self.subject_mask_time_list.append(subject_mask_time)
        else:
            self.subject_mask_time_ema = 0.0
        return x_output

    def load_warmup_unet(self, config):
        unet_warmup = self.pipe.build_warmup_unet(config)
        self.unet_warmup = unet_warmup
        self.pipe.unet_warmup = unet_warmup
        print("Load Warmup UNet.")
