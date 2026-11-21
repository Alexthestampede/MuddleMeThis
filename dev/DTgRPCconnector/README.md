# Draw Things gRPC Python Client

A comprehensive Python client library for the [Draw Things](https://drawthings.ai/) gRPC server, enabling programmatic access to state-of-the-art image generation models including FLUX, Flux Kontext (edit/reference), Stable Diffusion XL, Qwen Image, and more.

## Features

- **Text-to-Image Generation**: Generate images from text prompts across all supported architectures
- **Edit/Kontext Models**: Full support for Flux Klein and similar reference-image models with correct float16 tensor encoding
- **Reference Images / Moodboard**: Multiple reference images for edit models (FLUX.2 Klein) and IP-Adapter style conditioning
- **Video Generation with Audio**: LTX 2.3 support via `generate_media()` — captures synchronized frames + stereo audio, muxed with `save_video()`
- **Inpainting**: Mask-based inpainting with `mask_image` support (white = inpaint, black = preserve)
- **ControlNet**: Complete ControlNet support via `ControlNetConfig` with 18 input types
- **LoRA in FlatBuffer**: Pass LoRA adapters directly in the generation config via `LoRAConfig`
- **Smart Model Detection**: Automatically fetches latent size and metadata from server
- **Wide Model Support**: SD 1.5, SD 2.x, SDXL, FLUX (incl. FLUX.2/FLUX.3 Klein), Z-Image, Qwen, MOSS Image, Cosmos 2.5, MiniMax H3 Image, Stable Video Diffusion, LTX 2.3, and more
- **Tensor Encoding/Decoding**: Proper handling of both fpzip-compressed and uncompressed float16 tensors
- **Model Discovery**: List and browse available models and LoRAs
- **Streaming Progress**: Real-time progress updates during generation
- **Advanced Config**: FLUX guidance embed, separate text encoders, TeaCache, tiled diffusion/decoding, video generation, stage 2, causal inference, compression artifacts, and more

## Requirements

```bash
pip install grpcio grpcio-tools flatbuffers fpzip Pillow numpy

# Optional, for video assembly (LTX 2.3)
pip install imageio imageio-ffmpeg
```

## TLS Certificate — Required for Secure Connections

**TLS will fail without the Draw Things root CA certificate.** Download `root_ca.crt` from:
https://github.com/drawthingsai/draw-things-community/tree/main/Libraries/BinaryResources/Resources

Place it as `root_ca.crt` in the working directory, or pass the path via `ssl_cert_path` when creating the client.

## Quick Start

### List Available Models

Before generating images, discover what models are available on your server:

```bash
# List all models
python examples/list_models.py

# Also show LoRAs
python examples/list_models.py --loras

# Connect to a different server
python examples/list_models.py --server your-server:7859
```

### Basic Text-to-Image

```bash
# Simple generation
python examples/generate_image.py "a cute puppy"

# With options
python examples/generate_image.py "sunset over mountains" \
  --size 768 \
  --steps 30 \
  --output sunset.png \
  --negative "blurry, bad quality"
```

### Edit / Kontext Models (e.g., Flux Klein)

Kontext models use a reference image as visual context. The model VAE-encodes it into reference latents and feeds these as visual tokens to the text encoder. **Strength must be 1.0** and images must be encoded as uncompressed float16.

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="flux_klein_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    strength=1.0,          # REQUIRED for kontext/edit models
    guidance_embed=3.5,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="the same scene but in winter",
        config=config,
        input_image="reference.png",  # encoded as uncompressed float16 automatically
    )
```

### Inpainting

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=25,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
    strength=1.0,
    preserve_original_after_inpaint=True,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a red apple",
        config=config,
        input_image="photo.png",
        mask_image="mask.png",   # white = repaint, black = keep
    )
```

### ControlNet

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, ControlNetConfig
from Control import ControlInputType, ControlMode

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=25,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
    controls=[
        ControlNetConfig(
            file="control_v11p_sd15_canny.ckpt",
            weight=1.0,
            guidance_start=0.0,
            guidance_end=1.0,
            input_override=ControlInputType.Canny,
            control_mode=ControlMode.Balanced,
        )
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a building made of glass",
        config=config,
        input_image="edge_map.png",
    )
```

### LoRA via FlatBuffer Config

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, LoRAConfig

config = ImageGenerationConfig(
    model="flux_1_dev_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    loras=[
        LoRAConfig(file="my_style_lora_f16.ckpt", weight=0.8),
        LoRAConfig(file="another_lora_f16.ckpt",  weight=0.5),
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(prompt="a portrait in my style", config=config)
```

### Multiple Reference Images (Moodboard)

Edit models (FLUX.2 Klein, etc.) can see and combine the content of multiple reference images:

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, ReferenceImage

config = ImageGenerationConfig(
    model="flux_2_klein_4b_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    strength=1.0,
    guidance_embed=3.5,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="the person from reference 3, wearing the clothes from reference 2, on the beach from reference 1",
        config=config,
        reference_images=[
            ReferenceImage(image="beach.jpg", weight=0.33, hint_type="shuffle"),
            ReferenceImage(image="clothes.jpg", weight=0.33, hint_type="shuffle"),
            ReferenceImage(image="person.jpg", weight=0.34, hint_type="shuffle"),
        ],
    )
```

### Video with Audio (LTX 2.3)

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="ltx_2_3_540p_q8p.ckpt",
    steps=30,
    width=960,
    height=540,
    cfg_scale=1.0,
    scheduler="Euler A",
    num_frames=121,
    fps_id=24,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    result = client.generate_media(
        prompt="a cinematic drone shot flying over a tropical beach at sunset",
        config=config,
    )
    client.save_video(
        result.images,
        output_path="output/video.mp4",
        fps=24,
        audio=result.audio if result.audio else None,
    )
```

---

## Python API Reference

### DrawThingsClient

```python
from drawthings_client import DrawThingsClient

client = DrawThingsClient(
    server_address="server.example.com:7859",
    insecure=False,          # True for plain TCP, False for TLS
    verify_ssl=False,        # True to verify CA chain, False to accept any cert
    ssl_cert_path=None,      # Path to custom CA cert PEM (for self-signed certs)
    enable_compression=False,
)
```

#### generate_image()

```python
images: List[bytes] = client.generate_image(
    prompt="...",
    config=config,                  # ImageGenerationConfig
    negative_prompt="",
    scale_factor=1,
    input_image=None,               # PIL Image, file path str, or bytes
    mask_image=None,                # PIL Image, file path str, or bytes
    hints=None,                     # List of HintProto for ControlNet hints
    reference_images=None,          # List of ReferenceImage for moodboard
    metadata_override=None,
    progress_callback=None,         # Callable[[str, int], None]
    preview_callback=None,          # Callable[[bytes], None]
)
```

The returned list contains CCV tensor bytes; `save_images()` decodes them to PNG. 4-channel tensors (ARGB on the wire, normalized to RGBA — e.g. Qwen Image 2.1) are saved with their alpha channel intact. Save them with:

```python
client.save_images(images, output_dir="output", prefix="generated")
```

> **Note:** `generate_image()` discards any generated audio. For video models like LTX 2.3, use `generate_media()` instead.

#### generate_media()

```python
from drawthings_client import GenerationResult

result: GenerationResult = client.generate_media(
    prompt="...",
    config=config,
    # ... same arguments as generate_image()
)

result.images   # List[bytes] — image / video frame data
result.audio    # List[bytes] — audio chunks (empty for image models)
```

For LTX 2.3, assemble frames + audio into a playable video:

```python
client.save_video(
    result.images,
    output_path="output/video.mp4",
    fps=24,
    audio=result.audio if result.audio else None,
)
```

`save_video()` requires `imageio` and `imageio-ffmpeg` (`pip install imageio imageio-ffmpeg`).

**Audio is a tensor too.** Each entry in `result.audio` is a fpzip-compressed CCV tensor of shape `(channels, samples)` (stereo: `(2, N)`), float values in `[-1.0, 1.0]`, chunked exactly like image tensors. `save_video()` decodes them to interleaved float32 PCM and muxes via FFmpeg. Sample rates: **48 kHz for LTX 2.3**, 24 kHz for most other models (matching upstream `ModelZoo.audioSampleRateForModel`).

---

### ImageGenerationConfig

All parameters with their defaults:

```python
from drawthings_client import ImageGenerationConfig

config = ImageGenerationConfig(
    # --- Required ---
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=16,
    width=512,
    height=512,
    cfg_scale=5.0,
    scheduler="UniPC ays",

    # --- Core generation ---
    seed=None,              # auto-randomised if None
    strength=1.0,           # img2img/edit strength; must be 1.0 for kontext/edit models
    batch_count=1,
    batch_size=1,
    seed_mode=2,            # 0=Legacy, 1=TorchCpuCompatible, 2=ScaleAlike, 3=NvidiaGpuCompatible

    # --- CLIP / text encoders ---
    clip_skip=1,
    clip_weight=1.0,
    t5_text_encoder=True,           # Enable T5 encoder (FLUX models)
    separate_clip_l=False,          # Use a separate CLIP-L prompt
    clip_l_text="",
    separate_open_clip_g=False,     # Use a separate OpenCLIP-G prompt (SDXL)
    open_clip_g_text="",
    separate_t5=False,              # Use a separate T5 prompt
    t5_text="",

    # --- FLUX-specific ---
    guidance_embed=3.5,             # FLUX guidance scale (distinct from cfg_scale)
    speed_up_with_guidance_embed=True,
    resolution_dependent_shift=True,

    # --- SDXL-specific ---
    original_image_width=None,
    original_image_height=None,
    target_image_width=None,
    target_image_height=None,
    crop_top=0,
    crop_left=0,
    aesthetic_score=6.0,
    negative_aesthetic_score=2.5,

    # --- Upscaler / face restoration / refiner ---
    upscaler="",
    upscaler_scale_factor=0,        # 0 = auto
    face_restoration="",
    refiner_model="",
    refiner_start=0.85,

    # --- Hires fix ---
    hires_fix=False,
    hires_fix_start_width=0,        # scale units (pixels // 64)
    hires_fix_start_height=0,
    hires_fix_strength=0.7,

    # --- Tiled decoding (large images) ---
    tiled_decoding=False,
    decoding_tile_width=10,         # scale units
    decoding_tile_height=10,
    decoding_tile_overlap=2,

    # --- Tiled diffusion (large images) ---
    tiled_diffusion=False,
    diffusion_tile_width=16,        # scale units
    diffusion_tile_height=16,
    diffusion_tile_overlap=2,

    # --- Inpainting ---
    mask_blur=2.5,
    mask_blur_outset=0,
    sharpness=0.0,
    preserve_original_after_inpaint=True,

    # --- Edit model parameters ---
    image_guidance_scale=1.5,       # InstructPix2Pix / edit model image guidance

    # --- CFG zero star ---
    cfg_zero_star=False,
    cfg_zero_init_steps=0,
    stochastic_sampling_gamma=0.3,  # TCD sampler gamma

    # --- TeaCache (inference acceleration) ---
    tea_cache=False,
    tea_cache_start=5,
    tea_cache_end=-1,               # -1 means end of steps
    tea_cache_threshold=0.2,
    tea_cache_max_skip_steps=3,

    # --- Causal inference ---
    causal_inference_enabled=False,
    causal_inference=3,
    causal_inference_pad=0,

    # --- Stage 2 (Stable Cascade, etc.) ---
    stage_2_steps=10,
    stage_2_cfg=1.0,
    stage_2_shift=1.0,

    # --- Shift ---
    shift=1.0,

    # --- Video generation (Stable Video Diffusion / LTX) ---
    fps_id=5,
    motion_bucket_id=127,
    cond_aug=0.02,
    start_frame_cfg=1.0,
    num_frames=14,
    compression_artifacts=0,        # CompressionMethod: Disabled=0, H264=1, H265=2, Jpeg=3
    compression_artifacts_quality=43.1,
    color_calibration=0,            # ColorCalibration: Disabled=0, Lab=1
    expand_prompt_to_json=False,
    shift_for_audio=3.0,            # LTX audio shift
    uses_sol_attention=False,       # Sol attention (FLUX.2 Klein etc.)
    sol_attention_start=2,
    sol_attention_tau=0.5,

    # --- Misc ---
    zero_negative_prompt=False,
    negative_prompt_for_image_prior=True,
    image_prior_steps=5,

    # --- LoRA adapters (passed in FlatBuffer config) ---
    loras=[],                       # List[LoRAConfig]

    # --- ControlNet ---
    controls=[],                    # List[ControlNetConfig]
)
```

---

### LoRAConfig

```python
from drawthings_client import LoRAConfig

LoRAConfig(
    file="my_lora_f16.ckpt",
    weight=0.6,    # 0.0–1.0
    mode=0,        # 0=All, 1=Base, 2=Refiner
)
```

Multiple LoRAs can be stacked:

```python
loras=[
    LoRAConfig(file="style_lora.ckpt", weight=0.8),
    LoRAConfig(file="detail_lora.ckpt", weight=0.5),
]
```

---

### ControlNetConfig

```python
from drawthings_client import ControlNetConfig
from Control import ControlInputType, ControlMode

ControlNetConfig(
    file="control_v11p_sd15_canny.ckpt",  # empty string for built-in (e.g., inpaint)
    weight=1.0,
    guidance_start=0.0,     # fraction of steps where control begins
    guidance_end=1.0,       # fraction of steps where control ends
    no_prompt=False,
    global_average_pooling=True,
    down_sampling_rate=1.0,
    control_mode=ControlMode.Balanced,   # Balanced=0, Prompt=1, Control=2
    target_blocks=None,                  # List[str] for IP-Adapter etc.
    input_override=ControlInputType.Unspecified,
)
```

#### ControlInputType values

| Value | Constant | Description |
|-------|----------|-------------|
| 0 | `Unspecified` | Use model default |
| 1 | `Custom` | Custom input |
| 2 | `Depth` | Depth map |
| 3 | `Canny` | Canny edges |
| 4 | `Scribble` | Scribble sketch |
| 5 | `Pose` | Human pose (OpenPose) |
| 6 | `Normalbae` | Normal map |
| 7 | `Color` | Color guidance |
| 8 | `Lineart` | Line art |
| 9 | `Softedge` | Soft edges (HED) |
| 10 | `Seg` | Segmentation map |
| 11 | `Inpaint` | Inpaint (built-in, no model file) |
| 12 | `Ip2p` | InstructPix2Pix |
| 13 | `Shuffle` | Shuffle / reference |
| 14 | `Mlsd` | M-LSD straight lines |
| 15 | `Tile` | Tile / super-resolution |
| 16 | `Blur` | Blur |
| 17 | `Lowquality` | Low-quality input |
| 18 | `Gray` | Grayscale |

---

### ReferenceImage

```python
from drawthings_client import ReferenceImage

ReferenceImage(
    image="reference.jpg",  # PIL Image, file path str, or bytes
    weight=1.0,             # 0.0–1.0 influence weight
    hint_type="shuffle",    # "shuffle" (edit models), "ipadapterplus", "ipadapterfull"
)
```

Pass a list to `generate_image(..., reference_images=[...])`. For edit/kontext models use `hint_type="shuffle"` with `strength=1.0`.

---

## Technical Details

### Understanding Scale Factors

The `start_width` and `start_height` parameters in the FlatBuffer config are **scale factors**, not pixel dimensions. `ImageGenerationConfig` handles this conversion automatically when you specify `width` and `height` in pixels.

```
scale_factor = desired_pixels / 64

SD 1.5 / FLUX / Z-Image / Qwen (latent_size = 64):
  512px  / 64 = 8
  768px  / 64 = 12
  1024px / 64 = 16

SDXL (latent_size = 128):
  1024px / 128 = 8
  1536px / 128 = 12
```

Using pixel dimensions directly instead of scale factors causes server crashes or extremely slow generation.

### Tensor Format for Input Images

Draw Things uses the CCV NNC tensor format with a 68-byte header followed by pixel data.

**Header layout (17 x uint32, little-endian):**

| Offset | Field | Uncompressed (float16) | fpzip (float32) |
|--------|-------|------------------------|-----------------|
| 0 | type/magic | `0` | `1012247` |
| 4 | memory type | `0x1` (CPU) | `0x1` (CPU) |
| 8 | format | `0x02` (NHWC) | `0x02` (NHWC) |
| 12 | datatype | `0x20000` (CCV_16F) | `0x10000` (CCV_32F) |
| 16 | reserved | `0` | `0` |
| 20 | batch | `1` | `1` |
| 24 | height | H | H |
| 28 | width | W | W |
| 32 | channels | 3 (RGB) | 3 (RGB) |
| 36-67 | reserved | `0` | `0` |

**Pixel data (after byte 68):**
- Pixel values are normalized from uint8 `[0, 255]` to float `[-1.0, 1.0]`
- Formula: `float_value = uint8_value / 127.5 - 1.0`
- Uncompressed format: raw float16 bytes, NHWC row-major order
- Compressed format: fpzip-compressed float32 array

> **⚠️ Important: the server likely outputs fpzip-compressed tensors for everything.**
> Unless the server is started with `--no-response-compression`, all response data —
> generated images, video frames, masks, hints — arrives as fpzip-compressed float32
> tensors (header magic `1012247`). `tensor_decoder.decode_tensor()` auto-detects the
> format from the header magic and handles both cases transparently, but **fpzip must
> be installed** or decoding will raise:
> `"Response is compressed but fpzip is not available"`.
> Input tensors you *send* should remain uncompressed float16 for edit/kontext models
> (see below); the compression distinction only matters on the response side.

**Why uncompressed float16 matters for edit/kontext models:**

The Swift reference client (`euphoriacyberware-ai/DT-gRPC-Swift-Client`) encodes input images as uncompressed float16. This matches what the Draw Things server expects for identity preservation in kontext and reference-image workflows. The fpzip-compressed float32 path works for standard img2img but can degrade quality in edit models because the quantization path differs from the native encoding path.

`DrawThingsClient._encode_image()` always uses uncompressed float16 (`compress=False`). If you call `tensor_encoder.encode_image_to_tensor()` directly and need edit/kontext quality, pass `compress=False`.

```python
from tensor_encoder import encode_image_to_tensor

# Correct for edit/kontext models
tensor_bytes = encode_image_to_tensor("input.png", compress=False)

# For standard img2img (fpzip float32 also works)
tensor_bytes = encode_image_to_tensor("input.png", compress=True)
```

### Edit / Kontext Models Explained

Kontext models such as Flux Klein work differently from traditional img2img:

- The input image is **VAE-encoded into reference latents** (not blended into the noise).
- These reference latents are injected as **visual tokens into the text encoder**, giving the model full contextual understanding of the reference.
- This is also how reference/style images work in general (distinct from inpainting, which blends at the latent noise level).
- Because the image is encoded into the latent space at full strength, **`strength` must be `1.0`**. Setting it lower will not provide "less editing" — it will corrupt the reference encoding.
- The input image must be the **same resolution** as the generation target (`width` x `height`).

Correct configuration for a kontext edit:

```python
config = ImageGenerationConfig(
    model="flux_klein_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    strength=1.0,       # always 1.0
    guidance_embed=3.5, # FLUX guidance
    t5_text_encoder=True,
)
```

### Video Generation / LTX Notes

- **Audio**: The gRPC server returns `generatedAudio` alongside frames for video models (LTX 2.x). `generate_media()` collects it; `generate_image()` silently discards it.
- **Audio is a tensor**: fpzip-compressed CCV tensor, shape `(channels, samples)` (stereo `(2, N)`), values in `[-1.0, 1.0]`, chunked like images. `save_video()` decodes to interleaved float32 PCM and muxes via FFmpeg.
- **Sample rate**: 48 kHz for LTX 2.3; 24 kHz default for other models (upstream `ModelZoo.audioSampleRateForModel`).
- **LTX scale factors**: LTX models use different VAE ratios than image models — the server internally uses `startScaleFactor = 32` and doubles the scale. Pass pixel dimensions that are multiples of 64 (540p, 720p, etc.) to avoid reshape crashes.
- **Frames are not a video**: The response contains individual decoded frames, not a multiplexed file. Use `save_video()` (or mux externally with FFmpeg).
- **batch_count/batch_size are ignored for video models**: The server forces batch size to 1 for video models regardless of config.

### Model Architecture Detection

The client automatically fetches latent size from server metadata:

```python
from model_metadata import ModelMetadata

metadata = ModelMetadata("server.example.com:7859")
model_info = metadata.get_latent_info("flux_1_schnell_q8p.ckpt")
latent_size = model_info["latent_size"]   # 64
version     = model_info["version"]       # "flux1"
```

Known automatic mappings:

| Model pattern | latent_size | Architecture |
|---------------|-------------|--------------|
| `*sd15*`, `*sd1.5*` | 64 | Stable Diffusion 1.5 |
| `*xl*`, `*sdxl*` | 128 | Stable Diffusion XL |
| `*flux*`, `*klein*` | 64 | FLUX.1 / FLUX.2 / FLUX.3 Klein |
| `*z_image*` | 64 | Z-Image (FLUX VAE) |
| `*qwen*` | 64 | Qwen Image / Qwen Image 2.1 (RGBA) |
| `*moss_image*` | 64 | MOSS Image (FLUX-2 VAE) |
| `*minimax_h3_image*` | 64 | MiniMax H3 Image (edit, 4096px) |
| `*cosmos*` | 64 | Cosmos 2.5 (image model) |
| `*ltx*` | 64 | LTX 2.x video |

Qwen Image 2.1 can emit **4-channel** tensors. Draw Things orders those channels as **ARGB** (alpha first); `decode_tensor()` normalizes them to RGBA, `tensor_to_pil()` returns a PIL `RGBA` image, and `save_images()` / `save_tensor_image()` write PNGs that preserve alpha.

If metadata is unavailable the client falls back to filename heuristics. Override with `--latent-size` if needed.

---

## Server Setup

### Using Docker

```bash
# Start server without response compression (recommended for speed)
docker run -d \
  -v /path/to/models:/grpc-models \
  -p 7859:7859 \
  --gpus all \
  drawthingsai/draw-things-grpc-server-cli:latest \
  gRPCServerCLI /grpc-models --model-browser --cpu-offload --supervised --no-response-compression

# With compression (smaller responses, slightly slower)
docker run -d \
  -v /path/to/models:/grpc-models \
  -p 7859:7859 \
  --gpus all \
  drawthingsai/draw-things-grpc-server-cli:latest \
  gRPCServerCLI /grpc-models --model-browser --cpu-offload --supervised
```

---

## Project Structure

```
gRPC/
├── README.md
├── EXAMPLES.md
├── AGENTS.md                    # Agent/contributor guidance
├── requirements.txt
├── __init__.py
├── drawthings_client.py         # Main client: DrawThingsClient, ImageGenerationConfig,
│                                #   LoRAConfig, ControlNetConfig, ReferenceImage, GenerationResult
├── model_metadata.py            # Model discovery / latent size detection
├── tensor_decoder.py            # Decode CCV tensors to PNG (auto-detects fpzip) + audio tensors
├── tensor_encoder.py            # Encode images to CCV tensor format
├── GenerationConfiguration.py   # FlatBuffer schema (86 fields)
├── LoRA.py                      # FlatBuffer LoRA table
├── Control.py                   # FlatBuffer Control table + enums
├── SamplerType.py               # Sampler enum (incl. TCDTrailing)
├── imageService_pb2.py          # gRPC protobuf (generated)
├── imageService_pb2_grpc.py     # gRPC stubs (generated)
├── imageService.proto           # Proto source (synced from draw-things-community)
├── examples/
│   ├── generate_image.py        # CLI: text-to-image
│   ├── list_models.py           # CLI: list server models/LoRAs
│   ├── generate_with_references.py  # CLI: moodboard/reference images
│   └── add_moodboard.py         # Moodboard preparation helpers
└── tests/
    └── test_*.py
```

---

## Troubleshooting

### Server crashes with "Bad pointer dereference"

**Cause**: Passing pixel dimensions as scale factors.

**Fix**: `ImageGenerationConfig` handles this automatically. If building the FlatBuffer manually:

```python
scale = pixel_size // 64   # e.g. 512 // 64 = 8
config.start_width  = scale
config.start_height = scale
```

### Very slow generation (minutes instead of seconds)

**Cause**: Wrong scale factor produces 262,144 tokens instead of 4,096.

**Check server logs for**:
```
[cnnp_reshape_build] dim: (2, 4096, 320)    <- correct
[cnnp_reshape_build] dim: (2, 262144, 320)  <- wrong, scale factor too large
```

### "Response is compressed but fpzip is not available"

```bash
pip install fpzip
```

### Edit/kontext output does not preserve reference identity

- Confirm `strength=1.0` in your config.
- Confirm the input image dimensions match `width` and `height`.
- Confirm you are using uncompressed float16 encoding (this is the default in `DrawThingsClient._encode_image()`).
- Check that the model is actually a kontext/edit model and not a standard diffusion model.

### Image is the wrong size

**Cause**: Wrong architecture detection.

**Fix**: Pass `width` and `height` explicitly. The client calculates scale as `pixels // 64` regardless of architecture.

---

## Performance

| Model | Resolution | Steps | Approx. time | Notes |
|-------|-----------|-------|--------------|-------|
| SD 1.5 | 512x512 | 16 | ~7s | Standard |
| SD 1.5 | 768x768 | 20 | ~15s | Higher resolution |
| SDXL | 1024x1024 | 30 | ~45s | Best quality |
| FLUX Schnell | 1024x1024 | 4 | ~12s | Distilled, fast |
| Z-Image Turbo | 1024x1024 | 8 | ~32s | Alibaba model |

Times measured on an NVIDIA GPU with `--cpu-offload` enabled.

---

## Credits

- **Draw Things** by [Liu Liu](https://drawthings.ai/) - the underlying image generation app and gRPC server
- **Draw Things Community** - [drawthingsai/draw-things-community](https://github.com/drawthingsai/draw-things-community) - protocol definitions and community resources
- **Swift gRPC Reference Client** - [euphoriacyberware-ai/DT-gRPC-Swift-Client](https://github.com/euphoriacyberware-ai/DT-gRPC-Swift-Client) - instrumental in understanding the correct tensor encoding for edit/kontext models, particularly the uncompressed float16 `ImageHelpers.imageToDTTensor()` implementation
- **TypeScript Client** - [kcjerrell/dt-grpc-ts](https://github.com/kcjerrell/dt-grpc-ts) - helpful reference for tensor decoding

## License

This client implementation is provided as-is for educational purposes. Please respect Draw Things' terms of service.
