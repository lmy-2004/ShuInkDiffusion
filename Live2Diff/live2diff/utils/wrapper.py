import gc
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from diffusers import AutoencoderTiny
from PIL import Image

from live2diff import StreamAnimateDiffusionDepth
from live2diff.image_utils import postprocess_image
from live2diff.pipeline_stream_animation_depth import WARMUP_FRAMES


class StreamAnimateDiffusionDepthWrapper:
    @staticmethod
    def _get_local_tiny_vae_path(vae_id: Optional[str]) -> str:
        vae_path = Path(vae_id).expanduser() if vae_id is not None else Path(__file__).resolve().parents[2] / "models" / "taesd"
        config_path = vae_path / "config.json"
        weights_path = vae_path / "diffusion_pytorch_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Local TinyVAE files are missing in {vae_path}. "
                "Expected config.json and diffusion_pytorch_model.safetensors."
            )
        return str(vae_path)

    def __init__(
        self,
        config_path: str,
        few_step_model_type: str,
        num_inference_steps: int,
        t_index_list: Optional[List[int]] = None,
        strength: Optional[float] = None,
        dreambooth_path: Optional[str] = None,
        lora_dict: Optional[Dict[str, float]] = None,
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 42,
        engine_dir: Optional[Union[str, Path]] = "engines",
        opt_unet: bool = False,
    ):
        """
        Initializes the StreamAnimateDiffusionWrapper.

        Parameters
        ----------
        config_path : str
            The model id or path to load.
        few_step_model_type : str
            The few step model type to use.
        num_inference_steps : int
            The number of inference steps to perform. If `t_index_list`
            is passed, `num_infernce_steps` will parsed as the number
            of denoising steps before apply few-step lora. Otherwise,
            `num_inference_steps` will be parsed as the number of
            steps after applying few-step lora.
        t_index_list : List[int]
            The t_index_list to use for inference.
        strength : Optional[float]
            The strength to use for inference.
        dreambooth_path : Optional[str]
            The dreambooth path to use for inference. If not passed,
            will use dreambooth from config.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        vae_id : Optional[str], optional
            The local TinyVAE directory to load. If None, use
            `models/taesd` in this project.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 42.
        engine_dir : Optional[Union[str, Path]], optional
            The directory to save TensorRT engines, by default "engines".
        opt_unet : bool, optional
            Whether to optimize UNet or not, by default False.
        """
        self.sd_turbo = False

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size

        self.use_denoising_batch = use_denoising_batch

        self.stream: StreamAnimateDiffusionDepth = self._load_model(
            config_path=config_path,
            lora_dict=lora_dict,
            dreambooth_path=dreambooth_path,
            few_step_model_type=few_step_model_type,
            vae_id=vae_id,
            num_inference_steps=num_inference_steps,
            t_index_list=t_index_list,
            strength=strength,
            height=height,
            width=width,
            acceleration=acceleration,
            do_add_noise=do_add_noise,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
            opt_unet=opt_unet,
        )
        self.batch_size = len(self.stream.t_list) * frame_buffer_size if use_denoising_batch else frame_buffer_size

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(self.stream.unet, device_ids=device_ids)

        # if enable_similar_image_filter:
        #     self.stream.enable_similar_image_filter(
        #         similar_image_filter_threshold, similar_image_filter_max_skip_frame
        #     )

    def prepare(
        self,
        warmup_frames: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> torch.Tensor:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.

        Returns
        ----------
        warmup_frames : torch.Tensor
            generated warmup-frames.

        """
        warmup_frames = self.stream.prepare(
            warmup_frames=warmup_frames,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            delta=delta,
        )

        warmup_frames = warmup_frames.permute(0, 2, 3, 1)
        warmup_frames = (warmup_frames.clip(-1, 1) + 1) / 2
        return warmup_frames

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        return self.img2img(image, prompt)

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        return image

    def get_last_depth_image(self) -> Optional[Image.Image]:
        depth_map = getattr(self.stream, "last_depth_map", None)
        if depth_map is None:
            return None
        return postprocess_image(depth_map, output_type="pil")[0]

    def get_last_softedge_image(self) -> Optional[Image.Image]:
        softedge_map = getattr(self.stream, "last_softedge_map", None)
        if softedge_map is None:
            return None
        return postprocess_image(softedge_map, output_type="pil")[0]

    def get_last_subject_mask_image(self) -> Optional[Image.Image]:
        subject_mask_map = getattr(self.stream, "last_subject_mask_map", None)
        if subject_mask_map is None:
            return None
        return postprocess_image(subject_mask_map, output_type="pil")[0]

    def get_last_stylized_image(self) -> Optional[Image.Image]:
        stylized_map = getattr(self.stream, "last_stylized_map", None)
        if stylized_map is None:
            return None
        return postprocess_image(stylized_map, output_type="pil")[0]

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
        self.stream.set_conditioning(
            use_softedge=use_softedge,
            softedge_scale=softedge_scale,
            depth_scale=depth_scale,
            softedge_mode=softedge_mode,
            softedge_debug=softedge_debug,
            subject_mask_debug=subject_mask_debug,
            subject_mask_backend=subject_mask_backend,
            subject_mask_keyframe_interval=subject_mask_keyframe_interval,
            subject_mask_ema=subject_mask_ema,
        )

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
        self.stream.set_stylization(
            enable_stylize_preprocess=enable_stylize_preprocess,
            depth_blur_sigma=depth_blur_sigma,
            depth_power=depth_power,
            depth_smoothstep_min=depth_smoothstep_min,
            depth_smoothstep_max=depth_smoothstep_max,
            saturation_scale=saturation_scale,
            image_blur_sigma=image_blur_sigma,
            fog_white_mix=fog_white_mix,
            outline_blur_sigma=outline_blur_sigma,
            outline_power=outline_power,
            outline_smoothstep_min=outline_smoothstep_min,
            outline_smoothstep_max=outline_smoothstep_max,
            fog_depth_mask_blend=fog_depth_mask_blend,
            subject_mask_fog_blur_sigma=subject_mask_fog_blur_sigma,
        )

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
        self.stream.set_noise_control(
            enabled=enabled,
            motion_score=motion_score,
            mean_flow_mag=mean_flow_mag,
            flow=flow,
            motion_low_threshold=motion_low_threshold,
            motion_high_threshold=motion_high_threshold,
            min_noise_rate=min_noise_rate,
            max_noise_rate=max_noise_rate,
            enable_warped_noise=enable_warped_noise,
            warped_noise_reuse=warped_noise_reuse,
            warped_noise_residual=warped_noise_residual,
        )

    def set_key_step_sync(
        self,
        enabled: Optional[bool] = None,
        key_step_index: Optional[int] = None,
        strength: Optional[float] = None,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        flow_backend: Optional[str] = None,
    ) -> None:
        self.stream.set_key_step_sync(
            enabled=enabled,
            key_step_index=key_step_index,
            strength=strength,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            flow_backend=flow_backend,
        )

    def reset_noise_state(self) -> None:
        self.stream.reset_noise_state()

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "inference_time": round(float(getattr(self.stream, "inference_time_ema", 0.0)), 4),
            "depth_time": round(float(getattr(self.stream, "depth_time_ema", 0.0)), 4),
            "softedge_time": round(float(getattr(self.stream, "softedge_time_ema", 0.0)), 4),
            "subject_mask_time": round(float(getattr(self.stream, "subject_mask_time_ema", 0.0)), 4),
            "inference_steps": len(getattr(self.stream, "t_list", []) or []),
            "has_depth": getattr(self.stream, "last_depth_map", None) is not None,
            "has_softedge": getattr(self.stream, "last_softedge_map", None) is not None,
            "has_subject_mask": getattr(self.stream, "last_subject_mask_map", None) is not None,
            "use_softedge": bool(getattr(self.stream, "use_softedge", False)),
            "softedge_scale": round(float(getattr(self.stream, "softedge_scale", 0.0)), 4),
            "depth_scale": round(float(getattr(self.stream, "depth_scale", 1.0)), 4),
            "softedge_mode": getattr(self.stream, "softedge_mode", "classical"),
            "softedge_backend": getattr(self.stream, "softedge_backend", "disabled"),
            "softedge_pidinet_error": getattr(self.stream, "_pidinet_load_error", None),
            "softedge_pidinet_error": getattr(self.stream, "_pidinet_load_error", None),
            "subject_mask_debug": bool(getattr(self.stream, "subject_mask_debug", False)),
            "subject_mask_backend": getattr(self.stream, "subject_mask_backend", "disabled"),
            "subject_mask_requested_backend": getattr(self.stream, "subject_mask_requested_backend", "auto"),
            "subject_mask_sam2_error": getattr(self.stream, "_sam2_load_error", None),
            "has_stylized": getattr(self.stream, "last_stylized_map", None) is not None,
            "enable_stylize_preprocess": bool(getattr(self.stream, "enable_stylize_preprocess", False)),
            "depth_blur_sigma": round(float(getattr(self.stream, "depth_blur_sigma", 1.2)), 4),
            "depth_power": round(float(getattr(self.stream, "depth_power", 0.2)), 4),
            "depth_smoothstep_min": round(float(getattr(self.stream, "depth_smoothstep_min", 0.5)), 4),
            "depth_smoothstep_max": round(float(getattr(self.stream, "depth_smoothstep_max", 1.0)), 4),
            "saturation_scale": round(float(getattr(self.stream, "saturation_scale", 0.2)), 4),
            "image_blur_sigma": round(float(getattr(self.stream, "image_blur_sigma", 2.0)), 4),
            "fog_white_mix": round(float(getattr(self.stream, "fog_white_mix", 0.5)), 4),
            "outline_blur_sigma": round(float(getattr(self.stream, "outline_blur_sigma", 0.8)), 4),
            "outline_power": round(float(getattr(self.stream, "outline_power", 0.1)), 4),
            "outline_smoothstep_min": round(float(getattr(self.stream, "outline_smoothstep_min", 0.57)), 4),
            "outline_smoothstep_max": round(float(getattr(self.stream, "outline_smoothstep_max", 1.0)), 4),
            "fog_depth_mask_blend": round(float(getattr(self.stream, "fog_depth_mask_blend", 0.5)), 4),
            "subject_mask_fog_blur_sigma": round(float(getattr(self.stream, "subject_mask_fog_blur_sigma", 1.0)), 4),
            "noise_control_enabled": bool(getattr(self.stream, "noise_control_enabled", False)),
            "motion_score": None
            if getattr(self.stream, "motion_score", None) is None
            else round(float(getattr(self.stream, "motion_score", 0.0)), 6),
            "mean_flow_mag": None
            if getattr(self.stream, "mean_flow_mag", None) is None
            else round(float(getattr(self.stream, "mean_flow_mag", 0.0)), 6),
            "noise_rate": round(float(getattr(self.stream, "last_noise_rate", 1.0)), 6),
            "warped_noise_enabled": bool(getattr(self.stream, "enable_warped_noise", False)),
            "warped_noise_reuse": round(float(getattr(self.stream, "warped_noise_reuse", 0.85)), 6),
            "warped_noise_residual": round(float(getattr(self.stream, "warped_noise_residual", 0.15)), 6),
            "sync_enabled": bool(getattr(self.stream, "key_step_sync_enabled", False)),
            "sync_key_step_index": int(getattr(self.stream, "key_step_sync_index", 0)),
            "sync_strength": round(float(getattr(self.stream, "key_step_sync_strength", 0.0)), 6),
            "sync_weight": round(float(getattr(self.stream, "key_step_sync_weight", 0.0)), 6),
            "sync_memory_valid": bool(getattr(self.stream, "key_step_sync_memory_valid", False)),
            "flow_backend": getattr(self.stream, "key_step_sync_flow_backend", "farneback"),
        }

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(image, self.height, self.width).to(
            device=self.device, dtype=self.dtype
        )

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            output = postprocess_image(image_tensor, output_type=output_type)
        else:
            output = postprocess_image(image_tensor, output_type=output_type)[0]

        if output_type not in ["pil", "np"]:
            return output.cpu()
        else:
            return output

    @staticmethod
    def get_model_prefix(
        config_path: str,
        few_step_model_type: str,
        use_tiny_vae: bool,
        num_denoising_steps: int,
        height: int,
        width: int,
        dreambooth: Optional[str] = None,
        lora_dict: Optional[dict] = None,
    ) -> str:
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        third_party = config.third_party_dict
        dreambooth_path = dreambooth or third_party.dreambooth
        if dreambooth_path is None:
            dreambooth_name = "sd15"
        else:
            dreambooth_name = Path(dreambooth_path).stem

        base_lora_list = third_party.get("lora_list", [])
        lora_dict = lora_dict or {}
        for lora_alpha in base_lora_list:
            lora_name = lora_alpha["lora"]
            alpha = lora_alpha["lora_alpha"]
            if lora_name not in lora_dict:
                lora_dict[lora_name] = alpha

        prefix = f"{dreambooth_name}--{few_step_model_type}--step{num_denoising_steps}--"
        for k, v in lora_dict.items():
            prefix += f"{Path(k).stem}-{v}--"
        prefix += f"tiny_vae-{use_tiny_vae}--h-{height}--w-{width}--cond-v2"
        return prefix

    def _load_model(
        self,
        config_path: str,
        num_inference_steps: int,
        height: int,
        width: int,
        t_index_list: Optional[List[int]] = None,
        strength: Optional[float] = None,
        dreambooth_path: Optional[str] = None,
        lora_dict: Optional[Dict[str, float]] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        few_step_model_type: Optional[str] = None,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
        opt_unet: bool = False,
    ) -> StreamAnimateDiffusionDepth:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        6. Load the safety checker if needed.

        Parameters
        ----------
        config_path : str
            The path to config, all needed checkpoints are list in config file.
        t_index_list : List[int]
            The t_index_list to use for inference.
        dreambooth_path : Optional[str]
            The dreambooth path to use for inference. If not passed,
            will use dreambooth from config.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        vae_id : Optional[str], optional
            The local TinyVAE directory to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        opt_unet : bool, optional
            Whether to optimize UNet or not, by default False.

        Returns
        -------
        AnimatePipeline
            The loaded pipeline.
        """
        supported_few_step_model = ["LCM"]
        assert (
            few_step_model_type.upper() in supported_few_step_model
        ), f"Only support few_step_model: {supported_few_step_model}, but receive {few_step_model_type}."

        # NOTE: build animatediff pipeline
        from live2diff.animatediff.pipeline import AnimationDepthPipeline

        try:
            pipe = AnimationDepthPipeline.build_pipeline(
                config_path,
            ).to(device=self.device, dtype=self.dtype)
        except Exception:  # No model found
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

        if few_step_model_type.upper() == "LCM":
            local_lcm_lora = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models",
                "lora",
                "lcm-lora-sdv1-5",
            )
            has_local_lora = (
                os.path.isfile(os.path.join(local_lcm_lora, "pytorch_lora_weights.safetensors"))
                or os.path.isfile(os.path.join(local_lcm_lora, "pytorch_lora_weights.bin"))
            )
            if has_local_lora:
                few_step_lora = local_lcm_lora
            else:
                few_step_lora = "latent-consistency/lcm-lora-sdv1-5"
            stream_pipeline_cls = StreamAnimateDiffusionDepth

        print(f"Pipeline class: {stream_pipeline_cls}")
        print(f"Few-step LoRA: {few_step_lora}")

        # parse clip skip from config
        from .config import load_config

        cfg = load_config(config_path)
        third_party_dict = cfg.third_party_dict
        clip_skip = third_party_dict.get("clip_skip", 1)
        conditioning_kwargs = cfg.get("conditioning_kwargs", {})
        stylization_kwargs = cfg.get("stylization_kwargs", {})
        subject_mask_cfg = cfg.get("subject_mask", {})
        key_step_sync_cfg = cfg.get("key_step_sync", {})

        stream = stream_pipeline_cls(
            pipe=pipe,
            num_inference_steps=num_inference_steps,
            t_index_list=t_index_list,
            strength=strength,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
            clip_skip=clip_skip,
        )

        stream.load_warmup_unet(config_path)
        stream.set_conditioning(
            use_softedge=conditioning_kwargs.get("use_softedge", False),
            softedge_scale=conditioning_kwargs.get("softedge_scale", 0.35),
            depth_scale=conditioning_kwargs.get("depth_scale", 1.0),
            softedge_mode=conditioning_kwargs.get("softedge_mode", "classical"),
            softedge_debug=conditioning_kwargs.get("softedge_debug", False),
            subject_mask_debug=subject_mask_cfg.get("debug", False),
            subject_mask_backend=subject_mask_cfg.get("backend", "auto"),
            subject_mask_keyframe_interval=subject_mask_cfg.get("keyframe_interval", 8),
            subject_mask_ema=subject_mask_cfg.get("ema", 0.65),
        )
        stream.set_stylization(
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
        )
        stream.set_key_step_sync(
            enabled=key_step_sync_cfg.get("enabled", False),
            key_step_index=key_step_sync_cfg.get("key_step_index", 0),
            strength=key_step_sync_cfg.get("strength", 0.0),
            low_threshold=key_step_sync_cfg.get("low_threshold", 0.015),
            high_threshold=key_step_sync_cfg.get("high_threshold", 0.06),
            flow_backend=key_step_sync_cfg.get("flow_backend", "farneback"),
        )
        stream.load_lora(few_step_lora)
        stream.fuse_lora()

        denoising_steps_num = len(stream.t_list)
        stream.prepare_cache(
            height=height,
            width=width,
            denoising_steps_num=denoising_steps_num,
        )
        kv_cache_list = stream.kv_cache_list
        kv_cache_shapes = [tuple(cache.shape) for cache in kv_cache_list]

        if lora_dict is not None:
            for lora_name, lora_scale in lora_dict.items():
                stream.load_lora(lora_name)
                stream.fuse_lora(lora_scale=lora_scale)
                print(f"Use LoRA: {lora_name} in weights {lora_scale}")

        stream.initialize_softedge_branch()

        if use_tiny_vae:
            vae_path = self._get_local_tiny_vae_path(vae_id)
            print(f"TinyVAE path: {vae_path}")
            stream.vae = AutoencoderTiny.from_pretrained(vae_path, local_files_only=True).to(
                device=pipe.device, dtype=pipe.dtype
            )

        try:
            if acceleration == "none":
                stream.pipe.unet = torch.compile(stream.pipe.unet, options={"triton.cudagraphs": True}, fullgraph=True)
                stream.vae = torch.compile(stream.vae, options={"triton.cudagraphs": True}, fullgraph=True)
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                from polygraphy import cuda

                from live2diff.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_engine,
                )
                from live2diff.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    MidasEngine,
                    UNet2DConditionModelDepthEngine,
                )
                from live2diff.acceleration.tensorrt.models import (
                    VAE,
                    InflatedUNetDepth,
                    Midas,
                    VAEEncoder,
                )

                stream.kv_cache_list = None
                del kv_cache_list
                gc.collect()
                torch.cuda.empty_cache()

                prefix = self.get_model_prefix(
                    config_path=config_path,
                    few_step_model_type=few_step_model_type,
                    use_tiny_vae=use_tiny_vae,
                    num_denoising_steps=denoising_steps_num,
                    height=height,
                    width=width,
                    dreambooth=dreambooth_path,
                    lora_dict=lora_dict,
                )

                engine_dir = os.path.join(Path(engine_dir), prefix)
                unet_path = os.path.join(engine_dir, "unet", "unet.engine")
                unet_opt_path = os.path.join(engine_dir, "unet-opt", "unet.engine.opt")
                midas_path = os.path.join(engine_dir, "depth", "midas.engine")
                vae_encoder_path = os.path.join(engine_dir, "vae", "vae_encoder.engine")
                vae_decoder_path = os.path.join(engine_dir, "vae", "vae_decoder.engine")

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    os.makedirs(os.path.dirname(unet_opt_path), exist_ok=True)
                    unet_model = InflatedUNetDepth(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                        kv_cache_list=kv_cache_shapes,
                    )
                    compile_engine(
                        torch_model=stream.unet,
                        model_data=unet_model,
                        onnx_path=unet_path + ".onnx",
                        onnx_opt_path=unet_opt_path,  # use specific folder for external data
                        engine_path=unet_path,
                        opt_image_height=height,
                        opt_image_width=width,
                        opt_batch_size=stream.trt_unet_batch_size,
                        engine_build_options={
                            "ignore_onnx_optimize": not opt_unet,
                            "build_all_tactics": True,
                        },
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    max_bz = WARMUP_FRAMES
                    opt_bz = min_bz = 1
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=max_bz,
                        min_batch_size=min_bz,
                    )
                    compile_engine(
                        torch_model=stream.vae,
                        model_data=vae_decoder_model,
                        onnx_path=vae_decoder_path + ".onnx",
                        onnx_opt_path=vae_decoder_path + ".opt.onnx",
                        engine_path=vae_decoder_path,
                        opt_image_height=height,
                        opt_image_width=width,
                        opt_batch_size=opt_bz,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(midas_path):
                    os.makedirs(os.path.dirname(midas_path), exist_ok=True)
                    max_bz = WARMUP_FRAMES
                    opt_bz = min_bz = 1
                    midas = Midas(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=max_bz,
                        min_batch_size=min_bz,
                    )
                    compile_engine(
                        torch_model=stream.depth_detector.half(),
                        model_data=midas,
                        onnx_path=midas_path + ".onnx",
                        onnx_opt_path=midas_path + ".opt.onnx",
                        engine_path=midas_path,
                        opt_batch_size=opt_bz,
                        opt_image_height=384,
                        opt_image_width=384,
                        engine_build_options={
                            "auto_cast": False,
                            "handle_batch_norm": True,
                            "ignore_onnx_optimize": True,
                        },
                    )

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    max_bz = WARMUP_FRAMES
                    opt_bz = min_bz = 1
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=max_bz,
                        min_batch_size=min_bz,
                    )
                    compile_engine(
                        torch_model=vae_encoder,
                        model_data=vae_encoder_model,
                        onnx_path=vae_encoder_path + ".onnx",
                        onnx_opt_path=vae_encoder_path + ".opt.onnx",
                        engine_path=vae_encoder_path,
                        opt_batch_size=opt_bz,
                        opt_image_height=height,
                        opt_image_width=width,
                    )
                cuda_stream = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype
                midas_dtype = stream.depth_detector.dtype

                stream.unet = UNet2DConditionModelDepthEngine(unet_path, cuda_stream, use_cuda_graph=False)
                stream.depth_detector = MidasEngine(midas_path, cuda_stream, use_cuda_graph=False)
                setattr(stream.depth_detector, "dtype", midas_dtype)
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                stream.prepare_cache(
                    height=height,
                    width=width,
                    denoising_steps_num=denoising_steps_num,
                )
                stream.is_tensorrt = True

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")

        except Exception:
            traceback.print_exc()
            if acceleration == "tensorrt" and getattr(stream, "kv_cache_list", None) is None:
                try:
                    stream.prepare_cache(
                        height=height,
                        width=width,
                        denoising_steps_num=denoising_steps_num,
                    )
                except Exception:
                    pass
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        return stream
