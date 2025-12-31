# Model Conversion and Quantization Tools

## Overview

Draw Things includes powerful command-line tools for converting models from various formats (SafeTensors, PyTorch checkpoints) to the optimized `.ckpt` format used by the gRPC server. These tools also automatically generate JSON specifications needed for model discovery.

## Available Tools

1. **ModelConverter** - Convert full diffusion models (UNet, VAE, text encoders)
2. **LoRAConverter** - Convert LoRA weights
3. **EmbeddingConverter** - Convert textual inversion embeddings
4. **ModelQuantizer** - Quantize existing models to reduce file size

All tools are built using Bazel and output both converted model files and JSON specifications.

---

## Table of Contents

- [Building the Tools](#building-the-tools)
- [ModelConverter](#modelconverter)
- [LoRAConverter](#loraconverter)
- [EmbeddingConverter](#embeddingconverter)
- [ModelQuantizer](#modelquantizer)
- [JSON Specifications for gRPC Server](#json-specifications-for-grpc-server)
- [Complete Workflow Examples](#complete-workflow-examples)

---

## Building the Tools

### Build All Tools

```bash
# Build all conversion tools
bazel build Apps:ModelConverter
bazel build Apps:LoRAConverter
bazel build Apps:EmbeddingConverter
bazel build Apps:ModelQuantizer
```

### Build Location

Built binaries are located in:
```bash
bazel-bin/Apps/ModelConverter
bazel-bin/Apps/LoRAConverter
bazel-bin/Apps/EmbeddingConverter
bazel-bin/Apps/ModelQuantizer
```

### macOS Release Build

For optimized release binaries on macOS:

```bash
bazel build Apps:ModelConverter --config=release --macos_cpus=arm64,x86_64
bazel build Apps:LoRAConverter --config=release --macos_cpus=arm64,x86_64
bazel build Apps:EmbeddingConverter --config=release --macos_cpus=arm64,x86_64
bazel build Apps:ModelQuantizer --config=release --macos_cpus=arm64,x86_64
```

### Linux Build

```bash
bazel build Apps:ModelConverter --keep_going --spawn_strategy=local --compilation_mode=opt
bazel build Apps:LoRAConverter --keep_going --spawn_strategy=local --compilation_mode=opt
bazel build Apps:EmbeddingConverter --keep_going --spawn_strategy=local --compilation_mode=opt
bazel build Apps:ModelQuantizer --keep_going --spawn_strategy=local --compilation_mode=opt
```

---

## ModelConverter

### Purpose

Converts full diffusion models from SafeTensors or PyTorch checkpoint format to Draw Things `.ckpt` format. Automatically detects model type and generates appropriate JSON specification.

### Supported Model Types

- Stable Diffusion v1.x / v2.x
- SDXL Base / Refiner
- SSD-1B
- SD3 / SD3 Large
- PixArt
- SVD (Stable Video Diffusion)
- FLUX.1
- Hunyuan Video
- Wan v2.1 / v2.2

### Usage

```bash
ModelConverter \
  --file <input_file> \
  --name <model_name> \
  --output-directory <output_dir> \
  [--autoencoder-file <vae_file>] \
  [--text-encoders]
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--file`, `-f` | Yes | Path to SafeTensors or PyTorch checkpoint file |
| `--name`, `-n` | Yes | Name for the converted model |
| `--output-directory`, `-o` | Yes | Directory to write output files |
| `--autoencoder-file` | No | Custom VAE file (optional) |
| `--text-encoders` | No | Include text encoders in conversion |

### Output

The tool creates:
1. **Converted model files** (UNet, VAE, text encoders)
2. **JSON specification** printed to stdout

### Examples

#### Convert SDXL Base Model

```bash
# Basic conversion
bazel-bin/Apps/ModelConverter \
  --file ~/models/sd_xl_base_1.0.safetensors \
  --name "Stable Diffusion XL Base" \
  --output-directory ~/draw-things-models/

# Output files:
# - stable_diffusion_xl_base_f16.ckpt (UNet)
# - stable_diffusion_xl_base_vae_f16.ckpt (VAE, if not present uses default)
# - stable_diffusion_xl_base_clip_vit_l14_f16.ckpt (CLIP-L, if --text-encoders)
# - stable_diffusion_xl_base_open_clip_vit_bigg14_f16.ckpt (OpenCLIP-G, if --text-encoders)

# JSON output:
{
  "name": "Stable Diffusion XL Base",
  "file": "stable_diffusion_xl_base_f16.ckpt",
  "version": "sdxlBase",
  "modifier": "none",
  "text_encoder": "open_clip_vit_bigg14_f16.ckpt",
  "autoencoder": "sdxl_vae_v1.0_f16.ckpt",
  "clip_encoder": "clip_vit_l14_f16.ckpt"
}
```

#### Convert SD v1.5 Model

```bash
bazel-bin/Apps/ModelConverter \
  --file ~/models/v1-5-pruned-emaonly.safetensors \
  --name "Stable Diffusion v1.5" \
  --output-directory ~/draw-things-models/

# Output:
{
  "name": "Stable Diffusion v1.5",
  "file": "stable_diffusion_v1_5_f16.ckpt",
  "version": "v1",
  "modifier": "none",
  "text_encoder": "clip_vit_l14_f16.ckpt",
  "autoencoder": "vae_ft_mse_840000_f16.ckpt"
}
```

#### Convert FLUX.1 Model

```bash
bazel-bin/Apps/ModelConverter \
  --file ~/models/flux1-dev.safetensors \
  --name "FLUX.1 Dev" \
  --output-directory ~/draw-things-models/

# Output:
{
  "name": "FLUX.1 Dev",
  "file": "flux_1_dev_f16.ckpt",
  "version": "flux1",
  "modifier": "none",
  "text_encoder": "t5_xxl_encoder_q6p.ckpt",
  "autoencoder": "flux_1_vae_f16.ckpt",
  "clip_encoder": "clip_vit_l14_f16.ckpt"
}
```

#### Convert with Custom VAE

```bash
bazel-bin/Apps/ModelConverter \
  --file ~/models/model.safetensors \
  --name "Custom Model" \
  --output-directory ~/draw-things-models/ \
  --autoencoder-file ~/models/custom_vae.safetensors

# Uses custom VAE instead of default
```

#### Convert Inpainting Model

```bash
bazel-bin/Apps/ModelConverter \
  --file ~/models/sd-v1-5-inpainting.ckpt \
  --name "SD 1.5 Inpainting" \
  --output-directory ~/draw-things-models/

# Automatically detects inpainting variant
# Output includes:
{
  "name": "SD 1.5 Inpainting",
  "file": "sd_1_5_inpainting_f16.ckpt",
  "version": "v1",
  "modifier": "inpainting",  # ← Detected automatically
  ...
}
```

### Model Detection

The converter automatically detects:
- **Model version** (v1, v2, SDXL, FLUX, etc.)
- **Model modifier** (none, inpainting, depth, etc.)
- **Required encoders** (CLIP, T5, etc.)
- **Default VAE** for the model family
- **Guidance embed** support

### Text Encoder Conversion

By default, the tool **does not convert text encoders** (to save time/space). Use `--text-encoders` to include them:

```bash
bazel-bin/Apps/ModelConverter \
  --file model.safetensors \
  --name "My Model" \
  --output-directory ~/models/ \
  --text-encoders  # ← Include text encoder conversion
```

**Note**: Most models can share text encoders. For example, all SDXL models use the same CLIP and OpenCLIP encoders, so you only need to convert them once.

### Supported Model Formats

- **SafeTensors** (`.safetensors`) - Recommended
- **PyTorch Checkpoint** (`.ckpt`, `.pt`, `.pth`)
- **Diffusers format** (folder with separate files)

---

## LoRAConverter

### Purpose

Converts LoRA (Low-Rank Adaptation) weights from SafeTensors or PyTorch format to Draw Things format.

### Usage

```bash
LoRAConverter \
  --file <lora_file> \
  --name <lora_name> \
  --output-directory <output_dir> \
  [--version <model_version>] \
  [--scale-factor <factor>]
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--file`, `-f` | Yes | Path to LoRA SafeTensors or checkpoint |
| `--name`, `-n` | Yes | Name for the LoRA |
| `--output-directory`, `-o` | Yes | Directory to write output |
| `--version` | No | Force specific model version (auto-detected if omitted) |
| `--scale-factor` | No | Network scale factor (default: 1.0) |

### Model Version Values

When specifying `--version`, use one of these values:

```
v1, v2, kandinsky21, sdxlBase, sdxlRefiner, ssd1b, svdI2v,
wurstchenStageC, wurstchenStageB, sd3, sd3Large, pixart,
auraflow, flux1, flux2, hunyuanVideo, wan21_1_3b, wan21_14b,
wan22_5b, hiDreamI1, qwenImage, zImage
```

### Output

The tool creates:
1. **Converted LoRA file** (`*_lora_f16.ckpt`)
2. **JSON specification** printed to stdout

### Examples

#### Basic LoRA Conversion

```bash
bazel-bin/Apps/LoRAConverter \
  --file ~/loras/detail_tweaker.safetensors \
  --name "Detail Tweaker" \
  --output-directory ~/draw-things-models/

# Output:
{
  "name": "Detail Tweaker",
  "file": "detail_tweaker_lora_f16.ckpt",
  "version": "sdxlBase",  # Auto-detected
  "ti_embedding": false,
  "text_embedding_length": 0,
  "is_lo_ha": false
}
```

#### Force Specific Version

```bash
# If auto-detection fails, specify version
bazel-bin/Apps/LoRAConverter \
  --file ~/loras/custom_lora.safetensors \
  --name "Custom LoRA" \
  --output-directory ~/models/ \
  --version flux1  # Force FLUX.1 compatibility
```

#### Convert with Scale Factor

```bash
# Adjust network scale during conversion
bazel-bin/Apps/LoRAConverter \
  --file ~/loras/style_lora.safetensors \
  --name "Style LoRA" \
  --output-directory ~/models/ \
  --scale-factor 0.8  # Scale down by 20%
```

#### Convert LyCORIS/LoHa

```bash
# LoHa (LoRA with Hadamard Product) is auto-detected
bazel-bin/Apps/LoRAConverter \
  --file ~/loras/loha_model.safetensors \
  --name "LoHa Style" \
  --output-directory ~/models/

# Output:
{
  "name": "LoHa Style",
  "file": "loha_style_lora_f16.ckpt",
  "version": "sdxlBase",
  "ti_embedding": false,
  "text_embedding_length": 0,
  "is_lo_ha": true  # ← Detected as LoHa
}
```

#### LoRA with Embedded Textual Inversion

```bash
# Some LoRAs include TI embeddings
bazel-bin/Apps/LoRAConverter \
  --file ~/loras/character_lora.safetensors \
  --name "Character LoRA" \
  --output-directory ~/models/

# Output:
{
  "name": "Character LoRA",
  "file": "character_lora_lora_f16.ckpt",
  "version": "v1",
  "ti_embedding": true,  # ← Has embedded TI
  "text_embedding_length": 4,  # Token length
  "is_lo_ha": false
}
```

### Auto-Detection Features

The converter automatically detects:
- **Model version compatibility** (SD v1.5, SDXL, FLUX, etc.)
- **LoHa format** (vs standard LoRA)
- **Embedded textual inversions**
- **Text embedding length** (for TI)

---

## EmbeddingConverter

### Purpose

Converts Textual Inversion embeddings from SafeTensors or PyTorch format to Draw Things format.

### Usage

```bash
EmbeddingConverter \
  --file <embedding_file> \
  --name <embedding_name> \
  --output-directory <output_dir>
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--file`, `-f` | Yes | Path to embedding file |
| `--name`, `-n` | Yes | Name for the embedding |
| `--output-directory`, `-o` | Yes | Directory to write output |

### Output

The tool creates:
1. **Converted embedding file** (`*_ti_f16.ckpt`)
2. **JSON specification** printed to stdout

### Examples

#### Convert Textual Inversion

```bash
bazel-bin/Apps/EmbeddingConverter \
  --file ~/embeddings/style_embedding.pt \
  --name "Watercolor Style" \
  --output-directory ~/draw-things-models/

# Output:
{
  "name": "Watercolor Style",
  "file": "watercolor_style_ti_f16.ckpt",
  "version": "v1",  # Auto-detected
  "length": 3  # Number of tokens
}
```

#### Convert Multi-Token Embedding

```bash
bazel-bin/Apps/EmbeddingConverter \
  --file ~/embeddings/character.safetensors \
  --name "Character Concept" \
  --output-directory ~/models/

# Output:
{
  "name": "Character Concept",
  "file": "character_concept_ti_f16.ckpt",
  "version": "sdxlBase",
  "length": 8  # 8 tokens for this embedding
}
```

### Usage in Prompts

After conversion, use the embedding keyword (derived from filename) in prompts:

```python
# If converted file is "watercolor_style_ti_f16.ckpt"
# Keyword is "watercolor_style"

prompt = "A landscape painting in watercolor_style"
```

The keyword triggers the embedding, replacing tokens with learned vectors.

---

## ModelQuantizer

### Purpose

Quantizes existing Draw Things `.ckpt` models to reduce file size while maintaining quality. Supports multiple quantization levels (q4p, q5p, q6p, q8p) with model-specific strategies.

### Usage

```bash
ModelQuantizer \
  --input-file <input.ckpt> \
  --model-version <version> \
  --output-file <output.ckpt>
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--input-file`, `-i` | Yes | Path to input `.ckpt` file (FP16) |
| `--model-version`, `-m` | Yes | Model version (see list below) |
| `--output-file`, `-o` | Yes | Path for quantized output file |

### Model Version Values

```
v1, v2, kandinsky21, sdxl_base_v0.9, sdxl_refiner_v0.9, ssd_1b,
svd_i2v, wurstchen_v3.0_stage_c, wurstchen_v3.0_stage_b, sd3,
sd3_large, pixart, auraflow, flux1, flux2, hunyuan_video,
wan_v2.1_1.3b, wan_v2.1_14b, wan_v2.2_5b, hidream_i1, qwen_image,
z_image
```

### Quantization Levels

| Level | Bits | Size Reduction | Quality | Use Case |
|-------|------|----------------|---------|----------|
| **q8p** | 8-bit | ~50% | Excellent | Recommended for most models |
| **q6p** | 6-bit | ~62.5% | Very Good | Large models (FLUX, SD3 Large) |
| **q5p** | 5-bit | ~68.75% | Good | Specific layers in large models |
| **q4p** | 4-bit | ~75% | Acceptable | Experimental (not used by default) |

### Quantization Strategy by Model

Different models use different quantization strategies to balance size and quality:

#### SD v1.5 / v2.x / SDXL Strategy

```
- 2D layers (linear): q6p
- 4D layers (conv): q8p
- Other layers: FP16 (no quantization)
```

#### FLUX.1 / FLUX.2 Strategy

```
- Embedders, pos_embed, linear: FP16 (critical, no quantization)
- ada_ln layers: q8p
- Convolutions (4D): q8p
- Other weights: q8p
```

#### SD3 / SD3 Large Strategy

```
- Embedders, pos_embed, ada_ln: FP16 (no quantization)
- Normalization layers: FP16 with compression
- Other multi-dimensional: q8p
```

#### Wan / HiDream / Qwen Strategy

```
- Embedders, pos_embed, linear: FP16 (no quantization)
- ada_ln layers: q8p
- Convolutions (4D): q8p
- 2D weights: q6p or q8p (depends on model size)
- MoE layers (3D): q5p (HiDream only)
```

### Examples

#### Quantize SDXL Model

```bash
# Input: 6.9 GB FP16 model
# Output: ~3.5 GB quantized model

bazel-bin/Apps/ModelQuantizer \
  --input-file ~/models/sd_xl_base_1.0_f16.ckpt \
  --model-version sdxl_base_v0.9 \
  --output-file ~/models/sd_xl_base_1.0_q6p.ckpt

# Strategy:
# - 2D layers → q6p
# - 4D layers → q8p
# - Result: ~50% size reduction
```

#### Quantize FLUX.1 Model

```bash
# Input: 23.8 GB FP16 model
# Output: ~12 GB quantized model

bazel-bin/Apps/ModelQuantizer \
  --input-file ~/models/flux_1_dev_f16.ckpt \
  --model-version flux1 \
  --output-file ~/models/flux_1_dev_q8p.ckpt

# Strategy:
# - Critical layers (embedders) → FP16
# - ada_ln, convolutions → q8p
# - Result: ~50% size reduction, minimal quality loss
```

#### Quantize SD3 Large

```bash
# Input: 11.9 GB FP16 model
# Output: ~6 GB quantized model

bazel-bin/Apps/ModelQuantizer \
  --input-file ~/models/sd3_large_f16.ckpt \
  --model-version sd3_large \
  --output-file ~/models/sd3_large_q8p.ckpt
```

#### Quantize Wan 14B Model

```bash
# Input: ~28 GB FP16 model
# Output: ~11 GB quantized model

bazel-bin/Apps/ModelQuantizer \
  --input-file ~/models/wan_v2.1_14b_f16.ckpt \
  --model-version wan_v2.1_14b \
  --output-file ~/models/wan_v2.1_14b_q6p.ckpt

# Uses q6p for 2D layers (more aggressive for large model)
```

### Quality Guidelines

**Recommended quantization by model size:**

| Model Size | Recommended | Notes |
|------------|-------------|-------|
| < 3 GB | q8p | Minimal quality loss |
| 3-8 GB | q8p | Still excellent quality |
| 8-15 GB | q6p/q8p mix | Good balance |
| > 15 GB | q6p | Necessary for practicality |

**Quality impact:**
- **q8p**: Virtually indistinguishable from FP16
- **q6p**: Very minor quality loss, usually imperceptible
- **q5p**: Noticeable in fine details, acceptable for large models
- **q4p**: Visible quality loss, avoid for final models

### Compression Format

All quantized models also use **ezm7 compression** which provides additional lossless compression on top of quantization. Total size reduction combines both:

```
FP16 → q6p + ezm7 = ~65% size reduction
FP16 → q8p + ezm7 = ~55% size reduction
```

---

## JSON Specifications for gRPC Server

### Overview

The gRPC server discovers models through JSON specification files. These can be:
1. **Auto-generated** by conversion tools (printed to stdout)
2. **Manually created** for custom configurations
3. **Loaded from files** in the models directory

### Specification Formats

#### Model Specification

```json
{
  "name": "Stable Diffusion XL Base",
  "file": "sd_xl_base_1.0_f16.ckpt",
  "prefix": "sd_xl_",
  "version": "sdxlBase",
  "upcast_attention": false,
  "default_scale": 16,
  "text_encoder": "open_clip_vit_bigg14_f16.ckpt",
  "autoencoder": "sdxl_vae_v1.0_f16.ckpt",
  "clip_encoder": "clip_vit_l14_f16.ckpt",
  "modifier": "none",
  "deprecated": false
}
```

**Required fields:**
- `name`: Display name
- `file`: UNet filename
- `version`: Model version enum
- `default_scale`: Default resolution (8 = 512px, 16 = 1024px)

**Optional fields:**
- `prefix`: Filename prefix for related files
- `upcast_attention`: Use FP32 for attention (quality vs speed)
- `text_encoder`: CLIP/T5 encoder filename
- `autoencoder`: VAE filename
- `clip_encoder`: Additional CLIP encoder
- `t5_encoder`: T5 encoder filename (SD3, FLUX)
- `modifier`: Sampling modifier (none, inpainting, etc.)
- `deprecated`: Hide from UI

#### LoRA Specification

```json
{
  "name": "Detail Tweaker",
  "file": "detail_tweaker_lora_f16.ckpt",
  "prefix": "detail_tweaker_",
  "version": "sdxlBase",
  "is_lo_ha": false,
  "modifier": "none",
  "weight": {
    "value": 1.0,
    "lower_bound": -1.5,
    "upper_bound": 2.5
  }
}
```

**Weight field** (optional):
- `value`: Default weight
- `lower_bound`: Minimum allowed weight
- `upper_bound`: Maximum allowed weight

#### Textual Inversion Specification

```json
{
  "name": "Watercolor Style",
  "file": "watercolor_style_ti_f16.ckpt",
  "prefix": "watercolor_style_",
  "version": "v1",
  "length": 3
}
```

**Length field**: Number of tokens this embedding uses

### Creating JSON Files for Server

#### Option 1: Redirect Tool Output

```bash
# Generate JSON and save to file
bazel-bin/Apps/ModelConverter \
  --file model.safetensors \
  --name "My Model" \
  --output-directory ~/models/ \
  > ~/models/specifications/my_model.json

# The gRPC server can load this JSON
```

#### Option 2: Manual JSON Creation

Create a file `models.json`:

```json
[
  {
    "name": "Custom Model",
    "file": "custom_model_f16.ckpt",
    "version": "sdxlBase",
    "default_scale": 16,
    "text_encoder": "open_clip_vit_bigg14_f16.ckpt",
    "autoencoder": "sdxl_vae_v1.0_f16.ckpt",
    "clip_encoder": "clip_vit_l14_f16.ckpt"
  }
]
```

#### Option 3: Inline JSON in Echo Response

The server can return specifications via the `Echo` RPC:

```python
# Server-side: Load custom specifications
import json

custom_models = [
    {
        "name": "My FLUX Model",
        "file": "my_flux_f16.ckpt",
        "version": "flux1",
        "default_scale": 16,
        # ... other fields
    }
]

# Encode as JSON for MetadataOverride
override = imageService_pb2.MetadataOverride(
    models=json.dumps(custom_models).encode('utf-8')
)
```

### Advanced Specification Fields

#### latentsScalingFactor

Override default latent scaling:

```json
{
  "name": "Custom VAE Model",
  "file": "model.ckpt",
  "version": "sdxlBase",
  "latents_scaling_factor": 0.13025,
  "latents_mean": null,
  "latents_std": null
}
```

#### MMDiT Configuration (SD3, FLUX)

For transformer-based models:

```json
{
  "name": "SD3 Large",
  "file": "sd3_large_f16.ckpt",
  "version": "sd3Large",
  "mmdit": {
    "qk_norm": true,
    "dual_attention_layers": [0, 1, 2, 3, 4, 5],
    "distilled_guidance_layers": 24
  }
}
```

#### Sampler Modifier

```json
{
  "name": "LCM Model",
  "file": "lcm_model_f16.ckpt",
  "version": "v1",
  "modifier": "lcm",  # Changes default sampler
  "is_consistency_model": true
}
```

Modifier values:
- `none`: Standard model
- `inpainting`: Inpainting model
- `depth`: Depth-conditioned
- `lcm`: LCM (Latent Consistency Model)
- `tcd`: TCD (Trajectory Consistency Distillation)

---

## Complete Workflow Examples

### Workflow 1: Convert and Deploy SDXL Model

```bash
#!/bin/bash

# Step 1: Convert the model
echo "Converting SDXL model..."
bazel-bin/Apps/ModelConverter \
  --file ~/downloads/sd_xl_base_1.0.safetensors \
  --name "Stable Diffusion XL Base" \
  --output-directory ~/draw-things-models/ \
  --text-encoders \
  > ~/draw-things-models/sdxl_base.json

# Step 2: Quantize to save space
echo "Quantizing model..."
bazel-bin/Apps/ModelQuantizer \
  --input-file ~/draw-things-models/stable_diffusion_xl_base_f16.ckpt \
  --model-version sdxl_base_v0.9 \
  --output-file ~/draw-things-models/stable_diffusion_xl_base_q6p.ckpt

# Step 3: Update JSON to point to quantized version
sed 's/f16\.ckpt/q6p.ckpt/g' ~/draw-things-models/sdxl_base.json \
  > ~/draw-things-models/sdxl_base_q6p.json

# Step 4: Start gRPC server
bazel-bin/Apps/gRPCServerCLI ~/draw-things-models/

echo "Server ready with SDXL model!"
```

### Workflow 2: Batch Convert LoRAs

```bash
#!/bin/bash

LORA_DIR=~/loras
OUTPUT_DIR=~/draw-things-models/loras
SPECS_DIR=~/draw-things-models/specifications

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SPECS_DIR"

# Convert all LoRAs
for lora in "$LORA_DIR"/*.safetensors; do
    filename=$(basename "$lora" .safetensors)
    echo "Converting $filename..."

    bazel-bin/Apps/LoRAConverter \
      --file "$lora" \
      --name "$filename" \
      --output-directory "$OUTPUT_DIR" \
      > "$SPECS_DIR/${filename}.json"

    echo "✓ Converted $filename"
done

# Combine all JSON specs into one array
echo "[" > "$SPECS_DIR/loras.json"
first=true
for json in "$SPECS_DIR"/*.json; do
    if [ "$first" = true ]; then
        first=false
    else
        echo "," >> "$SPECS_DIR/loras.json"
    fi
    cat "$json" >> "$SPECS_DIR/loras.json"
done
echo "]" >> "$SPECS_DIR/loras.json"

echo "All LoRAs converted! Specification: $SPECS_DIR/loras.json"
```

### Workflow 3: Full Model Family Setup

```bash
#!/bin/bash

# Setup complete SD v1.5 ecosystem

MODELS_DIR=~/draw-things-models

echo "Step 1: Converting base model..."
bazel-bin/Apps/ModelConverter \
  --file ~/downloads/v1-5-pruned-emaonly.safetensors \
  --name "Stable Diffusion v1.5" \
  --output-directory "$MODELS_DIR" \
  --text-encoders

echo "Step 2: Converting inpainting variant..."
bazel-bin/Apps/ModelConverter \
  --file ~/downloads/sd-v1-5-inpainting.ckpt \
  --name "SD v1.5 Inpainting" \
  --output-directory "$MODELS_DIR"

echo "Step 3: Converting LoRAs..."
bazel-bin/Apps/LoRAConverter \
  --file ~/downloads/detail_tweaker.safetensors \
  --name "Detail Tweaker" \
  --output-directory "$MODELS_DIR" \
  --version v1

bazel-bin/Apps/LoRAConverter \
  --file ~/downloads/lcm_lora.safetensors \
  --name "LCM LoRA" \
  --output-directory "$MODELS_DIR" \
  --version v1

echo "Step 4: Converting embeddings..."
bazel-bin/Apps/EmbeddingConverter \
  --file ~/downloads/negative_embedding.pt \
  --name "EasyNegative" \
  --output-directory "$MODELS_DIR"

echo "Step 5: Quantizing base model..."
bazel-bin/Apps/ModelQuantizer \
  --input-file "$MODELS_DIR/stable_diffusion_v1_5_f16.ckpt" \
  --model-version v1 \
  --output-file "$MODELS_DIR/stable_diffusion_v1_5_q8p.ckpt"

echo "Complete! Models ready in $MODELS_DIR"
```

### Workflow 4: FLUX.1 with Quantization

```bash
#!/bin/bash

# FLUX.1 is large - quantization highly recommended

echo "Converting FLUX.1 Dev..."
bazel-bin/Apps/ModelConverter \
  --file ~/downloads/flux-dev.safetensors \
  --name "FLUX.1 Dev" \
  --output-directory ~/models/

echo "Quantizing FLUX.1 (this will take several minutes)..."
bazel-bin/Apps/ModelQuantizer \
  --input-file ~/models/flux_1_dev_f16.ckpt \
  --model-version flux1 \
  --output-file ~/models/flux_1_dev_q8p.ckpt

# Size comparison
echo "Size comparison:"
ls -lh ~/models/flux_1_dev_f16.ckpt ~/models/flux_1_dev_q8p.ckpt

# FP16: ~23.8 GB
# q8p:  ~12 GB (50% reduction)

echo "FLUX.1 ready for deployment!"
```

---

## Troubleshooting

### Conversion Fails with "Unknown model version"

**Problem**: Converter can't detect model type

**Solution**:
```bash
# For LoRA: manually specify version
bazel-bin/Apps/LoRAConverter \
  --file lora.safetensors \
  --name "LoRA" \
  --output-directory ~/models/ \
  --version sdxlBase  # ← Force version
```

### Quantization produces artifacts

**Problem**: Too aggressive quantization

**Solution**:
- Use q8p instead of q6p for critical models
- Some models are pre-quantized, don't re-quantize
- Check if model has special requirements

### JSON specification rejected by server

**Problem**: Invalid JSON format

**Solution**:
```bash
# Validate JSON
python3 -m json.tool specification.json

# Fix common issues:
# - Use snake_case for keys (text_encoder not textEncoder)
# - Version must be valid enum value
# - All file paths must be relative to models directory
```

### Server can't find converted model

**Problem**: File paths don't match

**Solution**:
```bash
# Ensure files are in the same directory
ls ~/draw-things-models/
# Should show:
# - model_f16.ckpt
# - model_vae_f16.ckpt
# - model_clip_vit_l14_f16.ckpt

# JSON file field must match exactly:
{
  "file": "model_f16.ckpt",  # Must match actual filename
  "autoencoder": "model_vae_f16.ckpt"
}
```

### Quantization takes too long

**Problem**: Processing large model

**Solution**:
- This is normal for large models (FLUX, Wan 14B)
- FLUX.1: ~10-15 minutes
- Wan 14B: ~20-30 minutes
- SD3 Large: ~5-10 minutes
- Use faster machine or be patient

---

## Best Practices

### File Organization

```
models/
├── specifications/
│   ├── models.json        # All model specs
│   ├── loras.json         # All LoRA specs
│   └── embeddings.json    # All TI specs
├── base_models/
│   ├── sd_xl_base_q6p.ckpt
│   ├── flux_1_dev_q8p.ckpt
│   └── sd3_large_q8p.ckpt
├── loras/
│   ├── detail_tweaker_lora_f16.ckpt
│   └── style_lora_f16.ckpt
├── embeddings/
│   └── easynegative_ti_f16.ckpt
└── shared_encoders/
    ├── clip_vit_l14_f16.ckpt
    ├── t5_xxl_encoder_q6p.ckpt
    └── sdxl_vae_v1.0_f16.ckpt
```

### Naming Conventions

- **Models**: `{name}_{precision}.ckpt`
  - Example: `sd_xl_base_q6p.ckpt`
- **LoRAs**: `{name}_lora_{precision}.ckpt`
  - Example: `detail_tweaker_lora_f16.ckpt`
- **Embeddings**: `{keyword}_ti_{precision}.ckpt`
  - Example: `easynegative_ti_f16.ckpt`

### Quantization Strategy

1. **Always keep FP16 master**: Don't delete original after quantizing
2. **Test quality**: Generate comparison images before deploying
3. **Quantize by size**:
   - < 3 GB: Don't quantize (already small)
   - 3-10 GB: q8p
   - 10-20 GB: q6p
   - > 20 GB: Must quantize (q6p minimum)

### Version Control

Keep a manifest of conversions:

```bash
# manifest.txt
sd_xl_base_1.0.safetensors → sd_xl_base_q6p.ckpt (2024-01-15)
flux-dev.safetensors → flux_1_dev_q8p.ckpt (2024-01-16)
detail_tweaker.safetensors → detail_tweaker_lora_f16.ckpt (2024-01-15)
```

---

## Additional Resources

- **Source Code**:
  - ModelConverter: `Apps/ModelConverter/Converter.swift`
  - LoRAConverter: `Apps/LoRAConverter/Converter.swift`
  - EmbeddingConverter: `Apps/EmbeddingConverter/Converter.swift`
  - ModelQuantizer: `Apps/ModelQuantizer/Quantizer.swift`

- **Related Documentation**:
  - `TechnicalQuirks-ThirdPartyDevelopers.md`: Latent scaling, resolution constraints
  - `ThirdPartyIntegration-gRPCServer.md`: gRPC API and model discovery
  - `CLAUDE.md`: Build system and deployment

- **Model Import Libraries**:
  - Model import: `Libraries/ModelOp/Sources/ModelImporter.swift`
  - LoRA import: `Libraries/ModelOp/Sources/LoRAImporter.swift`
  - Embedding import: `Libraries/ModelOp/Sources/EmbeddingImporter.swift`
