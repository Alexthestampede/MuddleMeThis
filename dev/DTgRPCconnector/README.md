# Draw Things gRPC Python Client

A comprehensive Python client library for the [Draw Things](https://drawthings.ai/) gRPC server, enabling programmatic access to state-of-the-art image generation models including FLUX, Stable Diffusion XL, Qwen Image, and more.

## 🎉 Features

- ✅ **Text-to-Image Generation**: Generate images from text prompts
- ✅ **Image Editing**: Edit existing images with Qwen Edit models
- ✅ **LoRA Support**: Apply Low-Rank Adaptation models for specialized outputs
- ✅ **Smart Model Detection**: Automatically fetches latent size and metadata from server
- ✅ **Wide Model Support**: SD 1.5, SD 2.x, SDXL, FLUX, Z-Image, Qwen, and more
- ✅ **Tensor Encoding/Decoding**: Proper handling of fpzip-compressed tensors
- ✅ **Model Discovery**: List and browse available models and LoRAs
- ✅ **Streaming Progress**: Real-time progress updates during generation
- ✅ **Production Ready**: Handles errors, edge cases, and various model architectures

**Note**: This client implements text-to-image generation, image editing, and LoRA support. Additional Draw Things features (ControlNets, upscaling, inpainting, etc.) are not yet implemented.

## 📦 Requirements

```bash
pip install grpcio grpcio-tools flatbuffers fpzip Pillow numpy
```

## 🚀 Quick Start

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

### Using LoRA Models

```bash
# Generate with Qwen Edit 2511 + 4-step Lightning LoRA
python examples/generate_image.py "a serene zen garden with cherry blossoms" \
  --model qwen_image_edit_2511_q6p.ckpt \
  --lora qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt \
  --lora-weight 1.0 \
  --steps 4 \
  --cfg 1.0
```

### Image Editing

```bash
# Edit an existing image
python examples/edit_image.py input.png "add warm sunset lighting" \
  --output edited.png \
  --strength 0.7 \
  --steps 10

# Subtle edit (lower strength)
python examples/edit_image.py photo.jpg "make it snowy" \
  --strength 0.5 \
  --steps 10
```

### Turbo Models (Fast!)

```bash
# Z-Image Turbo - automatically detected as latent_size=64
python generate_image.py "Japanese garden with cherry blossoms" \
  --model z_image_turbo_1.0_q6p.ckpt \
  --size 1024 \
  --steps 8 \
  --cfg 1.0
```

**Note**: Z-Image Turbo is not actually Stable Diffusion (it's from Alibaba), but the client automatically detects it uses latent_size=64 from server metadata!

### SDXL Models

```bash
python generate_image.py "futuristic cityscape" \
  --model juggernaut_xl_v9_q6p_q8p.ckpt \
  --size 1024 \
  --steps 30
```

## 📖 Complete Documentation

### Command-Line Options

```
usage: generate_image.py [-h] [--server SERVER] [--output OUTPUT]
                         [--negative NEGATIVE] [--model MODEL]
                         [--size SIZE] [--steps STEPS] [--cfg CFG]
                         [--seed SEED] [--latent-size {64,128}] prompt

positional arguments:
  prompt                Text description of image to generate

optional arguments:
  --server SERVER       Server address (default: 192.168.2.150:7859)
  --output OUTPUT, -o   Output filename (default: output.png)
  --negative NEGATIVE   Negative prompt
  --model MODEL         Model checkpoint name
  --size SIZE           Image size in pixels (512, 768, 1024)
  --steps STEPS         Number of diffusion steps (default: 16)
  --cfg CFG             CFG guidance scale (default: 5.0)
  --seed SEED           Random seed (None for random)
  --latent-size {64,128}  Latent grid size override (64=SD1.5, 128=SDXL)
                          Auto-detects if not specified, but may be wrong!
```

### Python API

```python
from generate_image import generate_image

# Generate an image
result = generate_image(
    prompt="a majestic lion",
    size=512,
    steps=20,
    cfg_scale=7.0,
    output="lion.png"
)
```

### Direct Tensor Decoding

```python
from tensor_decoder import save_tensor_image

# If you have raw tensor data from the gRPC response
with open('response_data.bin', 'rb') as f:
    tensor_data = f.read()

# Decode and save as PNG
save_tensor_image(tensor_data, 'output.png')
```

## ⚙️ Server Setup

### Using Docker

```bash
# Start server with compression (smaller responses, slightly slower)
docker run -d \
  -v /path/to/models:/grpc-models \
  -p 7859:7859 \
  --gpus all \
  drawthingsai/draw-things-grpc-server-cli:latest \
  gRPCServerCLI /grpc-models --model-browser --cpu-offload --supervised

# Start server without compression (faster, larger responses)
docker run -d \
  -v /path/to/models:/grpc-models \
  -p 7859:7859 \
  --gpus all \
  drawthingsai/draw-things-grpc-server-cli:latest \
  gRPCServerCLI /grpc-models --model-browser --cpu-offload --supervised --no-response-compression
```

## 🔧 Technical Details

### Understanding Scale Factors

**This is the critical discovery!** The `start_width` and `start_height` parameters are **scale factors**, not pixel dimensions.

```python
# WRONG - causes crashes or extreme slowness
config.start_width = 512   # ❌
config.start_height = 512  # ❌

# CORRECT - use scale factors
config.start_width = 8     # ✅ for 512px with SD 1.5
config.start_height = 8    # ✅
```

### Scale Factor Calculation

```
scale_factor = desired_pixels ÷ latent_size

SD 1.5 Architecture (latent_size = 64):
  512px  ÷ 64 = 8
  768px  ÷ 64 = 12
  1024px ÷ 64 = 16

SDXL Architecture (latent_size = 128):
  1024px ÷ 128 = 8
  1536px ÷ 128 = 12
```

### Model Architecture Detection

✅ **NEW**: The client now automatically fetches latent size from server metadata!

The client queries the Draw Things server for model metadata, which includes:
- Model version (v1, v2, sdxl, flux1, z_image, qwen_image, etc.)
- Default scale factor
- Latent space dimensions
- VAE configuration

**How it works:**
```python
# 1. Query server metadata
metadata = ModelMetadata(server)
model_info = metadata.get_latent_info(model_filename)

# 2. Get correct latent size
latent_size = model_info['latent_size']  # 64 for SD1.5/FLUX/Z-Image, 128 for SDXL
```

**Examples of automatic detection:**
- `realdream_15sd15_q6p_q8p.ckpt` → latent_size=64 (SD 1.5)
- `juggernaut_xl_v9_q6p_q8p.ckpt` → latent_size=128 (SDXL)
- `z_image_turbo_1.0_q8p.ckpt` → latent_size=64 (Z-Image, uses FLUX VAE)
- `flux_1_schnell_q5p.ckpt` → latent_size=64 (FLUX.1)
- `qwen_image_edit_2511_q6p.ckpt` → latent_size=64 (Qwen Image)

**Fallback behavior:**
If metadata is unavailable, falls back to naive detection (checking for 'xl' in filename).
You can always override with `--latent-size` parameter.

**Best Practice**: The automatic detection should work for all models with metadata. Only use `--latent-size` override if you encounter issues or are using custom models not in the server's metadata database.

### Tensor Format

Draw Things uses a custom tensor format:

**Header (68 bytes)**:
- Bytes 0-3: Magic number (1012247 = compressed, other = uncompressed)
- Bytes 24-27: Height (uint32, little-endian)
- Bytes 28-31: Width (uint32, little-endian)  
- Bytes 32-35: Channels (uint32, little-endian)

**Pixel Data (after byte 68)**:
- Format: float16 (2 bytes per value)
- Range: -1.0 to 1.0
- Order: RGB(A), row-major
- Compression: fpzip (if magic = 1012247)

**Conversion to PNG**:
```python
# Convert float16 [-1, 1] to uint8 [0, 255]
uint8_value = (float16_value + 1.0) * 127.0
```

## 📊 Performance

| Model | Resolution | Steps | Time | Notes |
|-------|-----------|-------|------|-------|
| SD 1.5 | 512×512 | 16 | ~7s | Standard quality |
| SD 1.5 | 768×768 | 20 | ~15s | Higher resolution |
| SD 1.5 Turbo | 1024×1024 | 8 | ~32s | Fast, fewer steps |
| SDXL | 1024×1024 | 30 | ~45s | Best quality |

*Tested on NVIDIA GPU with `--cpu-offload` enabled*

## 🐛 Troubleshooting

### Server crashes with "Bad pointer dereference"

**Cause**: Using pixel dimensions instead of scale factors

**Solution**: Make sure you're using scale factors:
```python
# For 512×512 with SD 1.5
scale = 512 // 64  # = 8
config.start_width = scale
config.start_height = scale
```

### "Response is compressed but fpzip is not available"

**Solution**: Install fpzip
```bash
pip install fpzip
```

### Very slow generation (minutes instead of seconds)

**Cause**: Wrong scale factor causing 262,144 tokens instead of 4,096

**Solution**: Check the debug output. Should show:
```
[cnnp_reshape_build] dim: (2, 4096, 320)  ✅ Correct
[cnnp_reshape_build] dim: (2, 262144, 320)  ❌ Wrong!
```

### Image is wrong size (e.g., 512×512 instead of 1024×1024)

**Cause**: Wrong architecture detection (using SDXL scale for SD 1.5 model)

**Solution**: Override architecture detection:
```python
# Force SD 1.5
latent_size = 64
scale = 1024 // 64  # = 16

# Force SDXL
latent_size = 128
scale = 1024 // 128  # = 8
```

## ⚠️ Known Issues

### Image Editing Output Quality

**Status**: Under investigation

The image editing feature (`examples/edit_image.py`) successfully sends images to the server and receives edited outputs, but the quality and structural preservation differ from the official Draw Things app.

**What works**:
- ✅ Tensor encoding/decoding with fpzip compression
- ✅ Server receives and encodes the input image correctly
- ✅ Generation completes without errors
- ✅ Output images are saved successfully

**Current limitation**:
- ❌ Edited images may not preserve the original composition/structure as expected
- ❌ Output can be significantly different from input (e.g., input: Japanese garden → output: sunset silhouette)

**Technical details**:
- The server correctly reports "Input image encoded" signpost
- Qwen Edit models use `.qwenimageEditPlus` modifier which passes images to the vision-language encoder
- This differs from traditional img2img latent blending (`.editing` modifier)
- The issue may be model-specific or related to additional parameters not yet discovered

**Workaround**:
- Use lower strength values (0.3-0.5) for more subtle edits
- Experiment with different edit models (2509 vs 2511)
- The official Draw Things app produces much better results with the same models

See [GitHub Issues](#) for updates on this investigation.

## 📁 Project Structure

```
gRPC/
├── README.md
├── EXAMPLES.md
├── requirements.txt
├── .gitignore
├── __init__.py
├── drawthings_client.py       # Main gRPC client
├── model_metadata.py           # Model discovery
├── tensor_decoder.py           # Tensor decoding
├── tensor_encoder.py           # Tensor encoding
├── GenerationConfiguration.py  # FlatBuffer config
├── LoRA.py                     # FlatBuffer LoRA
├── SamplerType.py             # Sampler enum
├── imageService_pb2.py        # gRPC protobuf (generated)
├── imageService_pb2_grpc.py   # gRPC stubs (generated)
├── examples/
│   ├── generate_image.py
│   ├── edit_image.py
│   ├── list_models.py
│   └── example_usage.py
└── tests/
    └── test_*.py
```

## 🙏 Credits

- **Draw Things**: Amazing AI image generation app by [Liu Liu](https://drawthings.ai/)
- **TypeScript Client**: [kcjerrell/dt-grpc-ts](https://github.com/kcjerrell/dt-grpc-ts) - Extremely helpful reference for tensor decoding
- **fpzip**: Fast compression for floating-point data

## 📝 License

This client implementation is provided as-is for educational purposes. Please respect Draw Things' terms of service.

## 🔗 Links

- [Draw Things App](https://drawthings.ai/)
- [Official GitHub](https://github.com/drawthingsai/draw-things-community)
- [TypeScript Client](https://github.com/kcjerrell/dt-grpc-ts)

## 🎯 Examples

### High-Quality Portrait

```bash
python generate_image.py \
  "professional portrait photo, woman with flowing red hair, soft lighting, bokeh background, 85mm lens" \
  --steps 40 \
  --cfg 7.0 \
  --size 768 \
  --negative "cartoon, anime, painting, distorted, blurry" \
  --output portrait.png
```

### Landscape with SDXL

```bash
python generate_image.py \
  "epic mountain landscape at sunset, dramatic clouds, golden hour lighting, photorealistic" \
  --model juggernaut_xl_v9_q6p_q8p.ckpt \
  --size 1024 \
  --steps 35 \
  --cfg 6.0 \
  --output landscape.png
```

### Fast Iteration with Turbo

```bash
python generate_image.py \
  "concept art of a futuristic vehicle, sleek design, neon accents" \
  --model z_image_turbo_1.0_q6p.ckpt \
  --size 1024 \
  --steps 8 \
  --cfg 1.0 \
  --output concept.png
```

### Specific Seed for Reproducibility

```bash
python generate_image.py \
  "cute cartoon cat wearing a wizard hat" \
  --seed 42 \
  --steps 25 \
  --output cat_wizard.png
```

---

**Made with ❤️ for the Draw Things community**
