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

# Import FlatBuffer schema
import GenerationConfiguration
import SamplerType


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
    # Add common aliases
    "UniPC ays": SamplerType.SamplerType.UniPCAYS,
    "unipc ays": SamplerType.SamplerType.UniPCAYS,
}


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
        shift: Shift parameter for certain models
        upscaler: Upscaler model name (optional)
        face_restoration: Face restoration model (optional)
        refiner_model: Refiner model name (optional)
        hires_fix: Enable hires fix
        hires_fix_start_width: Hires fix starting width in scale units (NOT pixels!)
        hires_fix_start_height: Hires fix starting height in scale units (NOT pixels!)
        hires_fix_strength: Hires fix strength (0.0-1.0, typically 0.7)
        tiled_decoding: Enable tiled decoding
        tiled_diffusion: Enable tiled diffusion
        mask_blur: Mask blur amount
        mask_blur_outset: Mask blur outset
        sharpness: Sharpness amount
        preserve_original_after_inpaint: Preserve original after inpainting
        cfg_zero_star: Enable CFG zero star
        cfg_zero_init_steps: CFG zero initialization steps
        causal_inference_pad: Causal inference padding
        tea_cache: Enable TeaCache acceleration (timestep embedding aware cache)
        original_image_width: Original image width in pixels (for edit models, optional)
        original_image_height: Original image height in pixels (for edit models, optional)
        target_image_width: Target image width in pixels (for edit models, optional)
        target_image_height: Target image height in pixels (for edit models, optional)
        image_guidance_scale: Image guidance scale for edit models (typically 1.5, optional)
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
    shift: float = 1.0
    upscaler: str = ""
    face_restoration: str = ""
    refiner_model: str = ""
    hires_fix: bool = False
    hires_fix_start_width: int = 0
    hires_fix_start_height: int = 0
    hires_fix_strength: float = 0.7
    tiled_decoding: bool = False
    tiled_diffusion: bool = False
    mask_blur: float = 2.5
    mask_blur_outset: int = 0
    sharpness: float = 0.0
    preserve_original_after_inpaint: bool = True
    cfg_zero_star: bool = False
    cfg_zero_init_steps: int = 0
    causal_inference_pad: int = 0
    tea_cache: bool = False
    original_image_width: Optional[int] = None
    original_image_height: Optional[int] = None
    target_image_width: Optional[int] = None
    target_image_height: Optional[int] = None
    image_guidance_scale: Optional[float] = None

    def __post_init__(self):
        """Generate random seed if not provided."""
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)

    def to_flatbuffer(self) -> bytes:
        """Convert configuration to FlatBuffer bytes.

        Returns:
            Serialized FlatBuffer configuration as bytes
        """
        builder = flatbuffers.Builder(2048)  # Increased size for more fields

        # Create string offsets (only for non-empty strings)
        model_offset = builder.CreateString(self.model)
        upscaler_offset = builder.CreateString(self.upscaler) if self.upscaler and len(self.upscaler) > 0 else None
        face_restoration_offset = builder.CreateString(self.face_restoration) if self.face_restoration and len(self.face_restoration) > 0 else None
        refiner_model_offset = builder.CreateString(self.refiner_model) if self.refiner_model and len(self.refiner_model) > 0 else None

        # Create empty arrays for controls and loras (required even if empty)
        GenerationConfiguration.StartControlsVector(builder, 0)
        controls_offset = builder.EndVector()

        GenerationConfiguration.StartLorasVector(builder, 0)
        loras_offset = builder.EndVector()

        # Get sampler type
        sampler_type = SCHEDULER_MAP.get(
            self.scheduler,
            SamplerType.SamplerType.UniPC
        )

        # Convert pixels to scale units (server always uses 64 as divisor)
        scale_width = self.width // 64
        scale_height = self.height // 64

        # Build GenerationConfiguration
        GenerationConfiguration.Start(builder)
        GenerationConfiguration.AddId(builder, 0)
        GenerationConfiguration.AddStartWidth(builder, scale_width)
        GenerationConfiguration.AddStartHeight(builder, scale_height)

        # Add edit-specific image dimensions if provided (for edit models like Qwen Edit)
        # IMPORTANT: These must be added early, before Seed
        if self.original_image_width is not None:
            GenerationConfiguration.AddOriginalImageWidth(builder, self.original_image_width)
        if self.original_image_height is not None:
            GenerationConfiguration.AddOriginalImageHeight(builder, self.original_image_height)
        if self.target_image_width is not None:
            GenerationConfiguration.AddTargetImageWidth(builder, self.target_image_width)
        if self.target_image_height is not None:
            GenerationConfiguration.AddTargetImageHeight(builder, self.target_image_height)

        GenerationConfiguration.AddSeed(builder, self.seed)
        GenerationConfiguration.AddSteps(builder, self.steps)
        GenerationConfiguration.AddGuidanceScale(builder, self.cfg_scale)

        # Add ImageGuidanceScale if provided (for edit models)
        if self.image_guidance_scale is not None:
            GenerationConfiguration.AddImageGuidanceScale(builder, self.image_guidance_scale)

        GenerationConfiguration.AddStrength(builder, self.strength)
        GenerationConfiguration.AddModel(builder, model_offset)
        GenerationConfiguration.AddSampler(builder, sampler_type)
        GenerationConfiguration.AddBatchCount(builder, self.batch_count)
        GenerationConfiguration.AddBatchSize(builder, self.batch_size)

        # Add new fields
        GenerationConfiguration.AddSeedMode(builder, self.seed_mode)
        GenerationConfiguration.AddClipSkip(builder, self.clip_skip)
        GenerationConfiguration.AddControls(builder, controls_offset)
        GenerationConfiguration.AddLoras(builder, loras_offset)
        GenerationConfiguration.AddShift(builder, self.shift)
        if upscaler_offset:
            GenerationConfiguration.AddUpscaler(builder, upscaler_offset)
        if face_restoration_offset:
            GenerationConfiguration.AddFaceRestoration(builder, face_restoration_offset)
        if refiner_model_offset:
            GenerationConfiguration.AddRefinerModel(builder, refiner_model_offset)
        GenerationConfiguration.AddHiresFix(builder, self.hires_fix)
        if self.hires_fix and self.hires_fix_start_width > 0:
            GenerationConfiguration.AddHiresFixStartWidth(builder, self.hires_fix_start_width)
        if self.hires_fix and self.hires_fix_start_height > 0:
            GenerationConfiguration.AddHiresFixStartHeight(builder, self.hires_fix_start_height)
        if self.hires_fix:
            GenerationConfiguration.AddHiresFixStrength(builder, self.hires_fix_strength)
        GenerationConfiguration.AddMaskBlur(builder, self.mask_blur)
        GenerationConfiguration.AddMaskBlurOutset(builder, self.mask_blur_outset)
        GenerationConfiguration.AddSharpness(builder, self.sharpness)
        GenerationConfiguration.AddTiledDecoding(builder, self.tiled_decoding)
        GenerationConfiguration.AddTiledDiffusion(builder, self.tiled_diffusion)
        GenerationConfiguration.AddPreserveOriginalAfterInpaint(builder, self.preserve_original_after_inpaint)
        GenerationConfiguration.AddCfgZeroStar(builder, self.cfg_zero_star)
        GenerationConfiguration.AddCfgZeroInitSteps(builder, self.cfg_zero_init_steps)
        GenerationConfiguration.AddCausalInferencePad(builder, self.causal_inference_pad)
        GenerationConfiguration.AddTeaCache(builder, self.tea_cache)

        config = GenerationConfiguration.End(builder)

        builder.Finish(config)
        return bytes(builder.Output())


class DrawThingsClient:
    """Client for Draw Things image generation server.

    This class provides a high-level interface for connecting to and interacting
    with a Draw Things server via gRPC.

    Attributes:
        server_address: Server address in format "host:port"
        channel: gRPC channel
        stub: gRPC service stub
    """

    def __init__(self, server_address: str, insecure: bool = True,
                 verify_ssl: bool = False, enable_compression: bool = False,
                 ssl_cert_path: Optional[str] = None):
        """Initialize Draw Things client.

        Args:
            server_address: Server address (e.g., "192.168.2.150:7859")
            insecure: Use insecure channel (no TLS). If False, uses TLS.
            verify_ssl: Verify SSL certificates (only used when insecure=False).
                       Set to False to accept self-signed certificates.
            enable_compression: Enable gzip compression for requests/responses.
            ssl_cert_path: Path to SSL certificate file (for self-signed certs).
                          If not provided, will look for 'server_cert.pem' in current directory.
        """
        self.server_address = server_address

        # Channel options for compression and keep-alive
        options = [
            # Message size limits (increased for large images)
            ('grpc.max_send_message_length', 32 * 1024 * 1024),  # 32MB max send
            ('grpc.max_receive_message_length', 32 * 1024 * 1024),  # 32MB max receive
            # Keep-alive settings to prevent connection drops during long operations
            ('grpc.keepalive_time_ms', 30000),  # Send keepalive ping every 30 seconds
            ('grpc.keepalive_timeout_ms', 10000),  # Wait 10 seconds for keepalive response
            ('grpc.keepalive_permit_without_calls', 1),  # Allow keepalive pings when no calls
            ('grpc.http2.max_pings_without_data', 0),  # No limit on pings without data
            ('grpc.http2.min_time_between_pings_ms', 10000),  # Min 10s between pings
            ('grpc.http2.min_ping_interval_without_data_ms', 5000),  # Min 5s between pings without data
        ]

        if enable_compression:
            options.extend([
                ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
                ('grpc.default_compression_level', 2),  # 0=none, 1=low, 2=medium, 3=high
            ])

        if insecure:
            self.channel = grpc.insecure_channel(server_address, options=options)
        else:
            # Create SSL credentials
            if verify_ssl:
                # Standard SSL with certificate verification
                credentials = grpc.ssl_channel_credentials()
            else:
                # SSL with self-signed certificate support
                # Try to load certificate from file
                root_certs = None
                if ssl_cert_path:
                    try:
                        with open(ssl_cert_path, 'rb') as f:
                            root_certs = f.read()
                    except FileNotFoundError:
                        print(f"Warning: Certificate file not found: {ssl_cert_path}")
                else:
                    # Try default location
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
                # Add option to accept certificate with CN=localhost
                options.extend([
                    ('grpc.ssl_target_name_override', 'localhost'),
                ])

            self.channel = grpc.secure_channel(server_address, credentials, options=options)

        self.stub = imageService_pb2_grpc.ImageGenerationServiceStub(self.channel)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close channel."""
        self.close()

    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()

    def echo(self, name: str = "test") -> imageService_pb2.EchoReply:
        """Test connection with echo request.

        Args:
            name: Name to send in echo request

        Returns:
            Echo reply from server
        """
        request = imageService_pb2.EchoRequest(name=name)
        return self.stub.Echo(request)

    def generate_image(
        self,
        prompt: str,
        config: ImageGenerationConfig,
        negative_prompt: str = "",
        scale_factor: int = 1,
        input_image: Optional[bytes] = None,
        metadata_override: Optional[any] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        preview_callback: Optional[Callable[[bytes], None]] = None
    ) -> List[bytes]:
        """Generate image(s) using the specified configuration.

        Args:
            prompt: Text prompt for image generation
            config: Image generation configuration
            negative_prompt: Negative prompt (optional)
            scale_factor: Image scale factor
            input_image: Optional input image bytes (PIL Image format) for img2img/edit.
                        Will be automatically resized to match config dimensions.
            metadata_override: Optional MetadataOverride protobuf object for LoRA metadata.
                              Used when LoRAs need to be specified with version info.
            progress_callback: Optional callback for progress updates.
                               Called with (stage_name, step_number)
            preview_callback: Optional callback for preview images.
                             Called with preview image bytes

        Returns:
            List of generated image data as bytes

        Example:
            def on_progress(stage, step):
                print(f"Stage: {stage}, Step: {step}")

            images = client.generate_image(
                prompt="A beautiful sunset",
                config=config,
                progress_callback=on_progress
            )
        """
        # Build FlatBuffer configuration
        config_bytes = config.to_flatbuffer()

        # Process input image if provided
        image_hash = None
        image_tensor = None
        if input_image is not None:
            # Import tensor encoder
            from tensor_encoder import encode_image_to_tensor

            # Load image from bytes
            from io import BytesIO
            pil_img = Image.open(BytesIO(input_image))

            # Convert to RGB if needed
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # Resize to match config dimensions
            target_size = (config.width, config.height)
            if pil_img.size != target_size:
                pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

            # Save to temporary file for encoding
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                pil_img.save(tmp_path)

            try:
                # Encode to tensor format
                image_tensor = encode_image_to_tensor(tmp_path, compress=True)

                # Calculate SHA256 hash
                image_hash = hashlib.sha256(image_tensor).digest()
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

        # Create gRPC request
        request_kwargs = {
            'prompt': prompt,
            'negativePrompt': negative_prompt,
            'configuration': config_bytes,
            'scaleFactor': scale_factor,
            'user': "DrawThingsPythonClient",
            'device': imageService_pb2.LAPTOP,
            'chunked': True  # Accept chunked responses
        }

        # Add image data if provided
        if image_hash is not None and image_tensor is not None:
            request_kwargs['image'] = image_hash
            request_kwargs['contents'] = [image_tensor]

        # Add metadata override if provided (for LoRA metadata)
        if metadata_override is not None:
            request_kwargs['override'] = metadata_override

        request = imageService_pb2.ImageGenerationRequest(**request_kwargs)

        # Stream response
        generated_images = []
        image_chunks = []  # Buffer for chunked image data

        try:
            for response in self.stub.GenerateImage(request):
                # Handle progress signposts
                if response.HasField('currentSignpost'):
                    signpost = response.currentSignpost

                    if signpost.HasField('sampling'):
                        step = signpost.sampling.step
                        if progress_callback:
                            progress_callback("Sampling", step)
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
                        step = signpost.secondPassSampling.step
                        if progress_callback:
                            progress_callback("Second Pass Sampling", step)

                # Handle preview images
                if response.HasField('previewImage') and preview_callback:
                    preview_callback(response.previewImage)

                # Handle chunked responses
                if response.generatedImages:
                    for img_data in response.generatedImages:
                        image_chunks.append(img_data)

                    # Check if this is the last chunk
                    if response.chunkState == imageService_pb2.LAST_CHUNK:
                        # Combine chunks if needed
                        if len(image_chunks) > 1:
                            combined = b''.join(image_chunks)
                            generated_images.append(combined)
                        elif len(image_chunks) == 1:
                            generated_images.append(image_chunks[0])
                        image_chunks = []  # Reset for next image

        except grpc.RpcError as e:
            raise Exception(f"gRPC error: {e.code()}: {e.details()}")

        return generated_images

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
        import os
        from pathlib import Path

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
        """Initialize progress handler.

        Args:
            total_steps: Total number of generation steps
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.current_stage = ""

    def on_progress(self, stage: str, step: int):
        """Progress callback handler.

        Args:
            stage: Current generation stage
            step: Current step number
        """
        self.current_stage = stage
        self.current_step = step

        if stage == "Sampling":
            percent = (step / self.total_steps) * 100
            print(f"\r{stage}: {step}/{self.total_steps} ({percent:.1f}%)", end="", flush=True)
        else:
            print(f"\n{stage}", flush=True)

    def on_complete(self):
        """Called when generation is complete."""
        print("\nGeneration complete!")


# Convenience function for quick image generation
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
    """Quick convenience function to generate a single image.

    Args:
        server_address: Server address (e.g., "192.168.2.150:7859")
        prompt: Text prompt for generation
        model: Model name
        steps: Number of steps
        width: Image width
        height: Image height
        cfg_scale: CFG scale
        scheduler: Scheduler name
        output_path: Where to save the image
        show_progress: Show progress updates

    Returns:
        Path to saved image file

    Example:
        filepath = quick_generate(
            "192.168.2.150:7859",
            "A beautiful sunset over mountains",
            output_path="sunset.png"
        )
    """
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
            import os
            from pathlib import Path

            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                f.write(images[0])

            return output_path
        else:
            raise Exception("No images generated")


if __name__ == "__main__":
    # Quick test if run directly
    print("Draw Things Client Library")
    print("Import this module to use the client")
    print("\nExample:")
    print("  from drawthings_client import DrawThingsClient, ImageGenerationConfig")
    print("  client = DrawThingsClient('192.168.2.150:7859')")
    print("  config = ImageGenerationConfig(model='realDream', steps=16, width=512, height=512, cfg_scale=5.0, scheduler='UniPC ays')")
    print("  images = client.generate_image(prompt='A beautiful sunset', config=config)")
