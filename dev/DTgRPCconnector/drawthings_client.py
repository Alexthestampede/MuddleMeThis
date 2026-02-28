"""
Draw Things gRPC Client Library

This module provides a modular Python client for the Draw Things image generation server.
It handles gRPC communication, FlatBuffer configuration building, and streaming response processing.

Example usage:
    from drawthings_client import DrawThingsClient, ImageGenerationConfig

    # Create client
    client = DrawThingsClient("192.168.2.150:7859")

    # Configure generation parameters
    config = ImageGenerationConfig(
        model="realDream",
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays"
    )

    # Generate image
    image_data = client.generate_image(
        prompt="A beautiful sunset",
        config=config
    )
"""

import grpc
import flatbuffers
import random
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass, field

from PIL import Image

# Import generated protobuf files
import imageService_pb2
import imageService_pb2_grpc

# Import FlatBuffer schemas
import GenerationConfiguration
import SamplerType
import LoRA as LoRAFB
import Control as ControlFB


# Scheduler name to SamplerType enum mapping
SCHEDULER_MAP = {
    "DPMPP2M Karras": SamplerType.SamplerType.DPMPP2MKarras,
    "Euler A": SamplerType.SamplerType.EulerA,
    "DDIM": SamplerType.SamplerType.DDIM,
    "PLMS": SamplerType.SamplerType.PLMS,
    "DPMPP SDE Karras": SamplerType.SamplerType.DPMPPSDEKarras,
    "UniPC": SamplerType.SamplerType.UniPC,
    "LCM": SamplerType.SamplerType.LCM,
    "Euler A Substep": SamplerType.SamplerType.EulerASubstep,
    "DPMPP SDE Substep": SamplerType.SamplerType.DPMPPSDESubstep,
    "TCD": SamplerType.SamplerType.TCD,
    "Euler A Trailing": SamplerType.SamplerType.EulerATrailing,
    "DPMPP SDE Trailing": SamplerType.SamplerType.DPMPPSDETrailing,
    "DPMPP2M AYS": SamplerType.SamplerType.DPMPP2MAYS,
    "Euler A AYS": SamplerType.SamplerType.EulerAAYS,
    "DPMPP SDE AYS": SamplerType.SamplerType.DPMPPSDEAYS,
    "DPMPP2M Trailing": SamplerType.SamplerType.DPMPP2MTrailing,
    "DDIM Trailing": SamplerType.SamplerType.DDIMTrailing,
    "UniPC Trailing": SamplerType.SamplerType.UniPCTrailing,
    "UniPC AYS": SamplerType.SamplerType.UniPCAYS,
    # Common aliases
    "UniPC ays": SamplerType.SamplerType.UniPCAYS,
    "unipc ays": SamplerType.SamplerType.UniPCAYS,
}


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter.

    Attributes:
        file: LoRA filename
        weight: LoRA weight (default 0.6)
        mode: 0=All, 1=Base, 2=Refiner
    """
    file: str
    weight: float = 0.6
    mode: int = 0  # LoRAMode: All=0, Base=1, Refiner=2


@dataclass
class ControlNetConfig:
    """Configuration for a ControlNet.

    Attributes:
        file: ControlNet model filename (empty string for built-in like inpaint)
        weight: Control weight (default 1.0)
        guidance_start: When to start applying control (0.0-1.0)
        guidance_end: When to stop applying control (0.0-1.0)
        no_prompt: Disable prompt for this control
        global_average_pooling: Use global average pooling (default True)
        down_sampling_rate: Down sampling rate (default 1.0)
        control_mode: 0=Balanced, 1=Prompt, 2=Control
        target_blocks: Target blocks for IP-Adapter etc.
        input_override: Override input type (ControlInputType enum value)
    """
    file: str = ""
    weight: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    no_prompt: bool = False
    global_average_pooling: bool = True
    down_sampling_rate: float = 1.0
    control_mode: int = 0  # ControlMode.Balanced
    target_blocks: Optional[List[str]] = None
    input_override: int = 0  # ControlInputType.Unspecified


@dataclass
class ImageGenerationConfig:
    """Configuration for image generation.

    Attributes:
        model: Model filename (e.g., "realdream_15sd15_q6p_q8p.ckpt")
        steps: Number of generation steps
        width: Image width in pixels
        height: Image height in pixels
        cfg_scale: Classifier-free guidance scale
        scheduler: Scheduler/sampler name (e.g., "UniPC ays")
        seed: Random seed (auto-generated if None)
        strength: Image-to-image strength (0.0-1.0)
        batch_count: Number of batches
        batch_size: Number of images per batch
        seed_mode: Seed mode (0=Legacy, 1=TorchCpuCompatible, 2=ScaleAlike, 3=NvidiaGpuCompatible)
        clip_skip: CLIP skip layers
        clip_weight: CLIP weight (default 1.0)
        shift: Shift parameter for certain models
        upscaler: Upscaler model name
        upscaler_scale_factor: Upscaler scale factor (0=auto)
        face_restoration: Face restoration model
        refiner_model: Refiner model name
        refiner_start: When to switch to refiner (0.0-1.0, default 0.7)
        hires_fix: Enable hires fix
        hires_fix_start_width: Hires fix starting width in scale units
        hires_fix_start_height: Hires fix starting height in scale units
        hires_fix_strength: Hires fix strength (0.0-1.0)
        tiled_decoding: Enable tiled decoding
        decoding_tile_width: Decoding tile width (default 10)
        decoding_tile_height: Decoding tile height (default 10)
        decoding_tile_overlap: Decoding tile overlap (default 2)
        tiled_diffusion: Enable tiled diffusion
        diffusion_tile_width: Diffusion tile width (default 16)
        diffusion_tile_height: Diffusion tile height (default 16)
        diffusion_tile_overlap: Diffusion tile overlap (default 2)
        mask_blur: Mask blur amount
        mask_blur_outset: Mask blur outset
        sharpness: Sharpness amount
        preserve_original_after_inpaint: Preserve original after inpainting
        image_guidance_scale: Image guidance scale for edit models (default 1.5)
        original_image_width: Original image width in pixels (for SDXL/edit models)
        original_image_height: Original image height in pixels (for SDXL/edit models)
        target_image_width: Target image width in pixels (for SDXL/edit models)
        target_image_height: Target image height in pixels (for SDXL/edit models)
        crop_top: Crop top offset (for SDXL)
        crop_left: Crop left offset (for SDXL)
        aesthetic_score: Aesthetic score for SDXL (default 6.0)
        negative_aesthetic_score: Negative aesthetic score for SDXL (default 2.5)
        zero_negative_prompt: Zero out negative prompt
        negative_prompt_for_image_prior: Use negative prompt for image prior (default True)
        image_prior_steps: Image prior steps (default 5)
        cfg_zero_star: Enable CFG zero star
        cfg_zero_init_steps: CFG zero initialization steps
        stochastic_sampling_gamma: Stochastic sampling gamma for TCD (default 0.3)
        guidance_embed: Guidance embed for FLUX (default 3.5)
        speed_up_with_guidance_embed: Speed up with guidance embed (default True)
        resolution_dependent_shift: Resolution dependent shift (default True)
        t5_text_encoder: Enable T5 text encoder (default True)
        separate_clip_l: Use separate CLIP-L prompt
        clip_l_text: Separate CLIP-L prompt text
        separate_open_clip_g: Use separate OpenCLIP-G prompt
        open_clip_g_text: Separate OpenCLIP-G prompt text
        separate_t5: Use separate T5 prompt
        t5_text: Separate T5 prompt text
        tea_cache: Enable TeaCache acceleration
        tea_cache_start: TeaCache start step (default 5)
        tea_cache_end: TeaCache end step (default -1)
        tea_cache_threshold: TeaCache threshold (default 0.06)
        tea_cache_max_skip_steps: TeaCache max skip steps (default 3)
        causal_inference_enabled: Enable causal inference
        causal_inference: Causal inference value (default 3)
        causal_inference_pad: Causal inference padding
        stage_2_steps: Stage 2 steps for Stable Cascade etc. (default 10)
        stage_2_cfg: Stage 2 CFG (default 1.0)
        stage_2_shift: Stage 2 shift (default 1.0)
        fps_id: Video FPS ID (default 5)
        motion_bucket_id: Video motion bucket ID (default 127)
        cond_aug: Video conditioning augmentation (default 0.02)
        start_frame_cfg: Video start frame CFG (default 1.0)
        num_frames: Video number of frames (default 14)
        loras: List of LoRA configurations
        controls: List of ControlNet configurations
    """
    model: str
    steps: int
    width: int
    height: int
    cfg_scale: float
    scheduler: str
    seed: Optional[int] = None
    strength: float = 1.0
    batch_count: int = 1
    batch_size: int = 1
    seed_mode: int = 2  # ScaleAlike
    clip_skip: int = 1
    clip_weight: float = 1.0
    shift: float = 1.0
    # Upscaler / face restoration / refiner
    upscaler: str = ""
    upscaler_scale_factor: int = 0
    face_restoration: str = ""
    refiner_model: str = ""
    refiner_start: float = 0.7
    # Hires fix
    hires_fix: bool = False
    hires_fix_start_width: int = 0
    hires_fix_start_height: int = 0
    hires_fix_strength: float = 0.7
    # Tiled decoding
    tiled_decoding: bool = False
    decoding_tile_width: int = 10
    decoding_tile_height: int = 10
    decoding_tile_overlap: int = 2
    # Tiled diffusion
    tiled_diffusion: bool = False
    diffusion_tile_width: int = 16
    diffusion_tile_height: int = 16
    diffusion_tile_overlap: int = 2
    # Inpainting / mask
    mask_blur: float = 2.5
    mask_blur_outset: int = 0
    sharpness: float = 0.0
    preserve_original_after_inpaint: bool = True
    # Edit model parameters
    image_guidance_scale: float = 1.5
    original_image_width: Optional[int] = None
    original_image_height: Optional[int] = None
    target_image_width: Optional[int] = None
    target_image_height: Optional[int] = None
    crop_top: int = 0
    crop_left: int = 0
    # SDXL parameters
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    zero_negative_prompt: bool = False
    negative_prompt_for_image_prior: bool = True
    image_prior_steps: int = 5
    # CFG
    cfg_zero_star: bool = False
    cfg_zero_init_steps: int = 0
    stochastic_sampling_gamma: float = 0.3
    # FLUX parameters
    guidance_embed: float = 3.5
    speed_up_with_guidance_embed: bool = True
    resolution_dependent_shift: bool = True
    # Text encoder options
    t5_text_encoder: bool = True
    separate_clip_l: bool = False
    clip_l_text: str = ""
    separate_open_clip_g: bool = False
    open_clip_g_text: str = ""
    separate_t5: bool = False
    t5_text: str = ""
    # TeaCache
    tea_cache: bool = False
    tea_cache_start: int = 5
    tea_cache_end: int = -1
    tea_cache_threshold: float = 0.06
    tea_cache_max_skip_steps: int = 3
    # Causal inference
    causal_inference_enabled: bool = False
    causal_inference: int = 3
    causal_inference_pad: int = 0
    # Stage 2 (Stable Cascade etc.)
    stage_2_steps: int = 10
    stage_2_cfg: float = 1.0
    stage_2_shift: float = 1.0
    # Video generation (SVD)
    fps_id: int = 5
    motion_bucket_id: int = 127
    cond_aug: float = 0.02
    start_frame_cfg: float = 1.0
    num_frames: int = 14
    # LoRA and ControlNet
    loras: List[LoRAConfig] = field(default_factory=list)
    controls: List[ControlNetConfig] = field(default_factory=list)

    def __post_init__(self):
        """Generate random seed if not provided."""
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)

    def to_flatbuffer(self) -> bytes:
        """Convert configuration to FlatBuffer bytes."""
        builder = flatbuffers.Builder(4096)

        # Create string offsets first (must be done before StartObject)
        model_offset = builder.CreateString(self.model)
        upscaler_offset = builder.CreateString(self.upscaler) if self.upscaler else None
        face_restoration_offset = builder.CreateString(self.face_restoration) if self.face_restoration else None
        refiner_model_offset = builder.CreateString(self.refiner_model) if self.refiner_model else None
        clip_l_text_offset = builder.CreateString(self.clip_l_text) if self.separate_clip_l and self.clip_l_text else None
        open_clip_g_text_offset = builder.CreateString(self.open_clip_g_text) if self.separate_open_clip_g and self.open_clip_g_text else None
        t5_text_offset = builder.CreateString(self.t5_text) if self.separate_t5 and self.t5_text else None

        # Build LoRA vector
        lora_offsets = []
        for lora in self.loras:
            lora_file_offset = builder.CreateString(lora.file)
            LoRAFB.Start(builder)
            LoRAFB.AddFile(builder, lora_file_offset)
            LoRAFB.AddWeight(builder, lora.weight)
            LoRAFB.AddMode(builder, lora.mode)
            lora_offsets.append(LoRAFB.End(builder))

        GenerationConfiguration.StartLorasVector(builder, len(lora_offsets))
        for offset in reversed(lora_offsets):
            builder.PrependUOffsetTRelative(offset)
        loras_offset = builder.EndVector()

        # Build Controls vector
        control_offsets = []
        for ctrl in self.controls:
            ctrl_file_offset = builder.CreateString(ctrl.file)
            # Build target_blocks vector if provided
            target_blocks_offset = None
            if ctrl.target_blocks:
                block_offsets = [builder.CreateString(b) for b in ctrl.target_blocks]
                ControlFB.StartTargetBlocksVector(builder, len(block_offsets))
                for bo in reversed(block_offsets):
                    builder.PrependUOffsetTRelative(bo)
                target_blocks_offset = builder.EndVector()

            ControlFB.Start(builder)
            ControlFB.AddFile(builder, ctrl_file_offset)
            ControlFB.AddWeight(builder, ctrl.weight)
            ControlFB.AddGuidanceStart(builder, ctrl.guidance_start)
            ControlFB.AddGuidanceEnd(builder, ctrl.guidance_end)
            ControlFB.AddNoPrompt(builder, ctrl.no_prompt)
            ControlFB.AddGlobalAveragePooling(builder, ctrl.global_average_pooling)
            ControlFB.AddDownSamplingRate(builder, ctrl.down_sampling_rate)
            ControlFB.AddControlMode(builder, ctrl.control_mode)
            if target_blocks_offset is not None:
                ControlFB.AddTargetBlocks(builder, target_blocks_offset)
            ControlFB.AddInputOverride(builder, ctrl.input_override)
            control_offsets.append(ControlFB.End(builder))

        GenerationConfiguration.StartControlsVector(builder, len(control_offsets))
        for offset in reversed(control_offsets):
            builder.PrependUOffsetTRelative(offset)
        controls_offset = builder.EndVector()

        # Get sampler type
        sampler_type = SCHEDULER_MAP.get(
            self.scheduler,
            SamplerType.SamplerType.UniPC
        )

        # Convert pixels to scale units (64px per unit)
        scale_width = self.width // 64
        scale_height = self.height // 64

        # Build GenerationConfiguration
        GenerationConfiguration.Start(builder)
        GenerationConfiguration.AddId(builder, 0)
        GenerationConfiguration.AddStartWidth(builder, scale_width)
        GenerationConfiguration.AddStartHeight(builder, scale_height)
        GenerationConfiguration.AddSeed(builder, self.seed)
        GenerationConfiguration.AddSteps(builder, self.steps)
        GenerationConfiguration.AddGuidanceScale(builder, self.cfg_scale)
        GenerationConfiguration.AddStrength(builder, self.strength)
        GenerationConfiguration.AddModel(builder, model_offset)
        GenerationConfiguration.AddSampler(builder, sampler_type)
        GenerationConfiguration.AddBatchCount(builder, self.batch_count)
        GenerationConfiguration.AddBatchSize(builder, self.batch_size)
        GenerationConfiguration.AddHiresFix(builder, self.hires_fix)
        if self.hires_fix and self.hires_fix_start_width > 0:
            GenerationConfiguration.AddHiresFixStartWidth(builder, self.hires_fix_start_width)
        if self.hires_fix and self.hires_fix_start_height > 0:
            GenerationConfiguration.AddHiresFixStartHeight(builder, self.hires_fix_start_height)
        if self.hires_fix:
            GenerationConfiguration.AddHiresFixStrength(builder, self.hires_fix_strength)
        if upscaler_offset:
            GenerationConfiguration.AddUpscaler(builder, upscaler_offset)
        GenerationConfiguration.AddImageGuidanceScale(builder, self.image_guidance_scale)
        GenerationConfiguration.AddSeedMode(builder, self.seed_mode)
        GenerationConfiguration.AddClipSkip(builder, self.clip_skip)
        GenerationConfiguration.AddControls(builder, controls_offset)
        GenerationConfiguration.AddLoras(builder, loras_offset)
        GenerationConfiguration.AddMaskBlur(builder, self.mask_blur)
        if face_restoration_offset:
            GenerationConfiguration.AddFaceRestoration(builder, face_restoration_offset)
        GenerationConfiguration.AddClipWeight(builder, self.clip_weight)
        GenerationConfiguration.AddNegativePromptForImagePrior(builder, self.negative_prompt_for_image_prior)
        GenerationConfiguration.AddImagePriorSteps(builder, self.image_prior_steps)
        if refiner_model_offset:
            GenerationConfiguration.AddRefinerModel(builder, refiner_model_offset)
        # FIXED: height=slot29, width=slot30 (was swapped before)
        if self.original_image_height is not None:
            GenerationConfiguration.AddOriginalImageHeight(builder, self.original_image_height)
        if self.original_image_width is not None:
            GenerationConfiguration.AddOriginalImageWidth(builder, self.original_image_width)
        GenerationConfiguration.AddCropTop(builder, self.crop_top)
        GenerationConfiguration.AddCropLeft(builder, self.crop_left)
        # FIXED: height=slot33, width=slot34 (was swapped before)
        if self.target_image_height is not None:
            GenerationConfiguration.AddTargetImageHeight(builder, self.target_image_height)
        if self.target_image_width is not None:
            GenerationConfiguration.AddTargetImageWidth(builder, self.target_image_width)
        GenerationConfiguration.AddAestheticScore(builder, self.aesthetic_score)
        GenerationConfiguration.AddNegativeAestheticScore(builder, self.negative_aesthetic_score)
        GenerationConfiguration.AddZeroNegativePrompt(builder, self.zero_negative_prompt)
        GenerationConfiguration.AddRefinerStart(builder, self.refiner_start)
        # Video generation
        GenerationConfiguration.AddFpsId(builder, self.fps_id)
        GenerationConfiguration.AddMotionBucketId(builder, self.motion_bucket_id)
        GenerationConfiguration.AddCondAug(builder, self.cond_aug)
        GenerationConfiguration.AddStartFrameCfg(builder, self.start_frame_cfg)
        GenerationConfiguration.AddNumFrames(builder, self.num_frames)
        # Mask / sharpness / shift
        GenerationConfiguration.AddMaskBlurOutset(builder, self.mask_blur_outset)
        GenerationConfiguration.AddSharpness(builder, self.sharpness)
        GenerationConfiguration.AddShift(builder, self.shift)
        # Stage 2
        GenerationConfiguration.AddStage2Steps(builder, self.stage_2_steps)
        GenerationConfiguration.AddStage2Cfg(builder, self.stage_2_cfg)
        GenerationConfiguration.AddStage2Shift(builder, self.stage_2_shift)
        # Tiled decoding
        GenerationConfiguration.AddTiledDecoding(builder, self.tiled_decoding)
        GenerationConfiguration.AddDecodingTileWidth(builder, self.decoding_tile_width)
        GenerationConfiguration.AddDecodingTileHeight(builder, self.decoding_tile_height)
        GenerationConfiguration.AddDecodingTileOverlap(builder, self.decoding_tile_overlap)
        GenerationConfiguration.AddStochasticSamplingGamma(builder, self.stochastic_sampling_gamma)
        GenerationConfiguration.AddPreserveOriginalAfterInpaint(builder, self.preserve_original_after_inpaint)
        # Tiled diffusion
        GenerationConfiguration.AddTiledDiffusion(builder, self.tiled_diffusion)
        GenerationConfiguration.AddDiffusionTileWidth(builder, self.diffusion_tile_width)
        GenerationConfiguration.AddDiffusionTileHeight(builder, self.diffusion_tile_height)
        GenerationConfiguration.AddDiffusionTileOverlap(builder, self.diffusion_tile_overlap)
        # Upscaler scale factor
        GenerationConfiguration.AddUpscalerScaleFactor(builder, self.upscaler_scale_factor)
        # Text encoders
        GenerationConfiguration.AddT5TextEncoder(builder, self.t5_text_encoder)
        GenerationConfiguration.AddSeparateClipL(builder, self.separate_clip_l)
        if clip_l_text_offset:
            GenerationConfiguration.AddClipLText(builder, clip_l_text_offset)
        GenerationConfiguration.AddSeparateOpenClipG(builder, self.separate_open_clip_g)
        if open_clip_g_text_offset:
            GenerationConfiguration.AddOpenClipGText(builder, open_clip_g_text_offset)
        # FLUX guidance
        GenerationConfiguration.AddSpeedUpWithGuidanceEmbed(builder, self.speed_up_with_guidance_embed)
        GenerationConfiguration.AddGuidanceEmbed(builder, self.guidance_embed)
        GenerationConfiguration.AddResolutionDependentShift(builder, self.resolution_dependent_shift)
        # TeaCache
        GenerationConfiguration.AddTeaCacheStart(builder, self.tea_cache_start)
        GenerationConfiguration.AddTeaCacheEnd(builder, self.tea_cache_end)
        GenerationConfiguration.AddTeaCacheThreshold(builder, self.tea_cache_threshold)
        GenerationConfiguration.AddTeaCache(builder, self.tea_cache)
        GenerationConfiguration.AddSeparateT5(builder, self.separate_t5)
        if t5_text_offset:
            GenerationConfiguration.AddT5Text(builder, t5_text_offset)
        GenerationConfiguration.AddTeaCacheMaxSkipSteps(builder, self.tea_cache_max_skip_steps)
        # Causal inference
        GenerationConfiguration.AddCausalInferenceEnabled(builder, self.causal_inference_enabled)
        GenerationConfiguration.AddCausalInference(builder, self.causal_inference)
        GenerationConfiguration.AddCausalInferencePad(builder, self.causal_inference_pad)
        # CFG zero
        GenerationConfiguration.AddCfgZeroStar(builder, self.cfg_zero_star)
        GenerationConfiguration.AddCfgZeroInitSteps(builder, self.cfg_zero_init_steps)

        config = GenerationConfiguration.End(builder)
        builder.Finish(config)
        return bytes(builder.Output())


class DrawThingsClient:
    """Client for Draw Things image generation server."""

    def __init__(self, server_address: str, insecure: bool = True,
                 verify_ssl: bool = False, enable_compression: bool = False,
                 ssl_cert_path: Optional[str] = None):
        """Initialize Draw Things client.

        Args:
            server_address: Server address (e.g., "192.168.2.150:7859")
            insecure: Use insecure channel (no TLS). If False, uses TLS.
            verify_ssl: Verify SSL certificates (only used when insecure=False).
            enable_compression: Enable gzip compression for requests/responses.
            ssl_cert_path: Path to SSL certificate file (for self-signed certs).
        """
        self.server_address = server_address

        options = [
            ('grpc.max_send_message_length', 256 * 1024 * 1024),
            ('grpc.max_receive_message_length', 256 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 5000),
        ]

        if enable_compression:
            options.extend([
                ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
                ('grpc.default_compression_level', 2),
            ])

        if insecure:
            self.channel = grpc.insecure_channel(server_address, options=options)
        else:
            if verify_ssl:
                credentials = grpc.ssl_channel_credentials()
            else:
                root_certs = None
                if ssl_cert_path:
                    try:
                        with open(ssl_cert_path, 'rb') as f:
                            root_certs = f.read()
                    except FileNotFoundError:
                        print(f"Warning: Certificate file not found: {ssl_cert_path}")
                else:
                    try:
                        with open('server_cert.pem', 'rb') as f:
                            root_certs = f.read()
                    except FileNotFoundError:
                        pass

                credentials = grpc.ssl_channel_credentials(
                    root_certificates=root_certs,
                    private_key=None,
                    certificate_chain=None
                )
                options.extend([
                    ('grpc.ssl_target_name_override', 'localhost'),
                ])

            self.channel = grpc.secure_channel(server_address, credentials, options=options)

        self.stub = imageService_pb2_grpc.ImageGenerationServiceStub(self.channel)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.channel:
            self.channel.close()

    def echo(self, name: str = "test") -> imageService_pb2.EchoReply:
        """Test connection with echo request."""
        request = imageService_pb2.EchoRequest(name=name)
        return self.stub.Echo(request)

    def _encode_image(self, image_input, target_width: int, target_height: int) -> bytes:
        """Encode an image to Draw Things tensor format.

        Args:
            image_input: PIL Image, file path string, or raw bytes
            target_width: Target width to resize to
            target_height: Target height to resize to

        Returns:
            Tensor-encoded bytes
        """
        from tensor_encoder import encode_image_to_tensor
        from io import BytesIO

        if isinstance(image_input, str):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            pil_img = Image.open(BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        target_size = (target_width, target_height)
        if pil_img.size != target_size:
            pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            pil_img.save(tmp_path)

        try:
            # Uncompressed float16 matches the Swift reference client exactly
            # and is required for proper identity preservation in edit/kontext models
            return encode_image_to_tensor(tmp_path, compress=False)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _encode_mask(self, mask_input, target_width: int, target_height: int) -> bytes:
        """Encode a mask image to Draw Things tensor format.

        The mask should be a grayscale image where white (255) = inpaint area
        and black (0) = preserve area. It will be encoded as a single-channel tensor.

        Args:
            mask_input: PIL Image, file path string, or raw bytes
            target_width: Target width to resize to
            target_height: Target height to resize to

        Returns:
            Tensor-encoded bytes (single channel)
        """
        import numpy as np
        import struct
        import fpzip
        from io import BytesIO

        if isinstance(mask_input, str):
            pil_img = Image.open(mask_input)
        elif isinstance(mask_input, bytes):
            pil_img = Image.open(BytesIO(mask_input))
        elif isinstance(mask_input, Image.Image):
            pil_img = mask_input
        else:
            raise TypeError(f"Unsupported mask type: {type(mask_input)}")

        # Convert to grayscale
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')

        target_size = (target_width, target_height)
        if pil_img.size != target_size:
            pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

        img_array = np.array(pil_img, dtype=np.uint8)
        height, width = img_array.shape
        channels = 1

        # Convert to float32 [-1, 1]
        float_array = (img_array.astype(np.float32) / 127.5) - 1.0
        float_array = float_array.reshape(1, height, width, channels)

        pixel_data = fpzip.compress(float_array, order='C')

        # CCV tensor header (68 bytes)
        header = bytearray(68)
        struct.pack_into('<I', header, 0, 1012247)   # [0] fpzip magic
        struct.pack_into('<I', header, 4, 0x1)       # [1] CCV_TENSOR_CPU_MEMORY
        struct.pack_into('<I', header, 8, 0x02)      # [2] CCV_TENSOR_FORMAT_NHWC
        struct.pack_into('<I', header, 12, 0x10000)  # [3] CCV_32F (float32)
        struct.pack_into('<I', header, 20, 1)        # [5] batch = 1
        struct.pack_into('<I', header, 24, height)   # [6] height
        struct.pack_into('<I', header, 28, width)    # [7] width
        struct.pack_into('<I', header, 32, channels) # [8] channels

        return bytes(header) + pixel_data

    def generate_image(
        self,
        prompt: str,
        config: ImageGenerationConfig,
        negative_prompt: str = "",
        scale_factor: int = 1,
        input_image=None,
        mask_image=None,
        hints: Optional[List] = None,
        metadata_override=None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        preview_callback: Optional[Callable[[bytes], None]] = None
    ) -> List[bytes]:
        """Generate image(s) using the specified configuration.

        Args:
            prompt: Text prompt for image generation
            config: Image generation configuration
            negative_prompt: Negative prompt
            scale_factor: Image scale factor
            input_image: Input image for img2img/edit (PIL Image, path, or bytes)
            mask_image: Mask image for inpainting (PIL Image, path, or bytes).
                       White=inpaint area, black=preserve area.
            hints: List of HintProto objects for ControlNet hints
            metadata_override: MetadataOverride protobuf object (for LoRA metadata)
            progress_callback: Callback for progress (stage_name, step_number)
            preview_callback: Callback for preview images

        Returns:
            List of generated image data as bytes
        """
        config_bytes = config.to_flatbuffer()

        # Content-addressable storage
        contents = []

        # Process input image
        image_hash = None
        if input_image is not None:
            image_tensor = self._encode_image(input_image, config.width, config.height)
            image_hash = hashlib.sha256(image_tensor).digest()
            contents.append(image_tensor)

        # Process mask image
        mask_hash = None
        if mask_image is not None:
            mask_tensor = self._encode_mask(mask_image, config.width, config.height)
            mask_hash = hashlib.sha256(mask_tensor).digest()
            contents.append(mask_tensor)

        # Build gRPC request
        request_kwargs = {
            'prompt': prompt,
            'negativePrompt': negative_prompt,
            'configuration': config_bytes,
            'scaleFactor': scale_factor,
            'user': "DrawThingsPythonClient",
            'device': imageService_pb2.LAPTOP,
            'chunked': True,
        }

        if image_hash is not None:
            request_kwargs['image'] = image_hash
        if mask_hash is not None:
            request_kwargs['mask'] = mask_hash
        if contents:
            request_kwargs['contents'] = contents
        if hints:
            request_kwargs['hints'] = hints
        if metadata_override is not None:
            request_kwargs['override'] = metadata_override

        request = imageService_pb2.ImageGenerationRequest(**request_kwargs)

        # Stream response
        generated_images = []
        image_chunks = []

        try:
            for response in self.stub.GenerateImage(request):
                # Handle progress signposts
                if response.HasField('currentSignpost'):
                    signpost = response.currentSignpost

                    if signpost.HasField('sampling'):
                        if progress_callback:
                            progress_callback("Sampling", signpost.sampling.step)
                    elif signpost.HasField('textEncoded'):
                        if progress_callback:
                            progress_callback("Text Encoded", 0)
                    elif signpost.HasField('imageEncoded'):
                        if progress_callback:
                            progress_callback("Image Encoded", 0)
                    elif signpost.HasField('imageDecoded'):
                        if progress_callback:
                            progress_callback("Image Decoded", 0)
                    elif signpost.HasField('secondPassSampling'):
                        if progress_callback:
                            progress_callback("Second Pass Sampling", signpost.secondPassSampling.step)
                    elif signpost.HasField('secondPassImageEncoded'):
                        if progress_callback:
                            progress_callback("Second Pass Image Encoded", 0)
                    elif signpost.HasField('secondPassImageDecoded'):
                        if progress_callback:
                            progress_callback("Second Pass Image Decoded", 0)
                    elif signpost.HasField('faceRestored'):
                        if progress_callback:
                            progress_callback("Face Restored", 0)
                    elif signpost.HasField('imageUpscaled'):
                        if progress_callback:
                            progress_callback("Image Upscaled", 0)

                # Handle preview images
                if response.HasField('previewImage') and preview_callback:
                    preview_callback(response.previewImage)

                # Handle chunked responses
                if response.generatedImages:
                    for img_data in response.generatedImages:
                        image_chunks.append(img_data)

                    if response.chunkState == imageService_pb2.LAST_CHUNK:
                        if len(image_chunks) > 1:
                            combined = b''.join(image_chunks)
                            generated_images.append(combined)
                        elif len(image_chunks) == 1:
                            generated_images.append(image_chunks[0])
                        image_chunks = []

        except grpc.RpcError as e:
            raise Exception(f"gRPC error: {e.code()}: {e.details()}")

        return generated_images

    def files_exist(self, files: List[str]) -> dict:
        """Check if files exist on the server.

        Args:
            files: List of filenames to check

        Returns:
            Dict mapping filename to existence bool
        """
        request = imageService_pb2.FileListRequest(files=files)
        response = self.stub.FilesExist(request)
        result = {}
        for fname, exists in zip(response.files, response.existences):
            result[fname] = exists
        return result

    def save_images(
        self,
        images: List[bytes],
        output_dir: str = ".",
        prefix: str = "generated"
    ) -> List[str]:
        """Save generated images to disk.

        Args:
            images: List of image data as bytes
            output_dir: Directory to save images
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for i, image_data in enumerate(images):
            filename = f"{prefix}_{i+1}.png"
            filepath = output_path / filename

            with open(filepath, 'wb') as f:
                f.write(image_data)

            saved_files.append(str(filepath))

        return saved_files


class StreamingProgressHandler:
    """Helper class to handle streaming progress updates with display."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.current_stage = ""

    def on_progress(self, stage: str, step: int):
        self.current_stage = stage
        self.current_step = step

        if stage == "Sampling":
            percent = (step / self.total_steps) * 100
            print(f"\r{stage}: {step}/{self.total_steps} ({percent:.1f}%)", end="", flush=True)
        elif stage == "Second Pass Sampling":
            print(f"\r{stage}: step {step}", end="", flush=True)
        else:
            print(f"\n{stage}", flush=True)

    def on_complete(self):
        print("\nGeneration complete!")


def quick_generate(
    server_address: str,
    prompt: str,
    model: str = "realDream",
    steps: int = 16,
    width: int = 512,
    height: int = 512,
    cfg_scale: float = 5.0,
    scheduler: str = "UniPC ays",
    output_path: str = "output.png",
    show_progress: bool = True
) -> str:
    """Quick convenience function to generate a single image."""
    config = ImageGenerationConfig(
        model=model,
        steps=steps,
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        scheduler=scheduler
    )

    progress_handler = StreamingProgressHandler(steps) if show_progress else None

    with DrawThingsClient(server_address) as client:
        images = client.generate_image(
            prompt=prompt,
            config=config,
            progress_callback=progress_handler.on_progress if progress_handler else None
        )

        if progress_handler:
            progress_handler.on_complete()

        if images:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                f.write(images[0])

            return output_path
        else:
            raise Exception("No images generated")


if __name__ == "__main__":
    print("Draw Things Client Library")
    print("Import this module to use the client")
    print("\nExample:")
    print("  from drawthings_client import DrawThingsClient, ImageGenerationConfig")
    print("  client = DrawThingsClient('192.168.2.150:7859')")
    print("  config = ImageGenerationConfig(model='realDream', steps=16, width=512, height=512, cfg_scale=5.0, scheduler='UniPC ays')")
    print("  images = client.generate_image(prompt='A beautiful sunset', config=config)")
