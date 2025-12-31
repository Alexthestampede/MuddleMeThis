# Technical Quirks and Critical Details for Third-Party Developers

## Overview

This document covers critical technical details that commonly trip up developers building third-party applications for Draw Things gRPC server. Understanding these quirks is essential for correct image generation and tensor handling.

## Table of Contents

1. [Latent Scaling Factors (Tensor Scale)](#latent-scaling-factors-tensor-scale)
2. [Automatic Shift Calculation](#automatic-shift-calculation)
3. [High Resolution Fix](#high-resolution-fix)
4. [Resolution Constraints](#resolution-constraints)
5. [Model-Specific Defaults](#model-specific-defaults)

---

## Latent Scaling Factors (Tensor Scale)

### What is Latent Scaling?

The VAE (Variational Autoencoder) operates in latent space, not pixel space. When encoding images to latents or decoding latents to images, a scaling factor must be applied. **Different model families use different scaling factors**, and using the wrong one will produce completely incorrect images.

### Critical Issue

A common mistake is assuming all models use the same scaling factor (e.g., 0.18215 from SD v1.5). This will break image generation for newer models like FLUX, SD3, SDXL, etc.

### Model-Specific Scaling Factors

Here are the scaling factors for different model families:

| Model Family | Scaling Factor | Shift Factor | Mean/Std |
|--------------|----------------|--------------|----------|
| **SD v1.x / v2.x** | 0.18215 | None | None |
| **SDXL / SSD-1B / PixArt / AuraFlow** | 0.13025 | None | None |
| **Kandinsky 2.1** | 1.0 | None | None |
| **Würstchen (Stable Cascade)** | 2.32558139535 | None | None |
| **SD3 / SD3 Large** | 1.5305 | 0.0609 | None |
| **FLUX.1 / HiDream I1 / Z Image** | 0.3611 | 0.1159 | None |
| **FLUX.2** | (see special case below) | None | Mean & Std arrays |
| **Hunyuan Video** | 0.476986 | None | None |
| **Wan 2.1 / Wan 2.2 / Qwen Image** | 1.0 | None | 16-channel Mean & Std |

### Encoding and Decoding Formulas

#### Standard Models (SD v1/v2, SDXL)

**Encoding** (image → latent):
```
latent = (image_tensor) * scaling_factor
```

**Decoding** (latent → image):
```
image_tensor = latent / scaling_factor
```

Example for SD v1.5:
```python
# Encoding
latent = image_tensor * 0.18215

# Decoding
image_tensor = latent / 0.18215
```

#### Models with Shift Factor (SD3, FLUX.1)

**Encoding**:
```
latent = (image_tensor - shift_factor) * scaling_factor
```

**Decoding**:
```
image_tensor = (latent / scaling_factor) + shift_factor
```

Example for FLUX.1:
```python
scaling_factor = 0.3611
shift_factor = 0.1159

# Encoding
latent = (image_tensor - 0.1159) * 0.3611

# Decoding
image_tensor = (latent / 0.3611) + 0.1159
```

#### Models with Mean & Std Normalization (Wan, Qwen Image)

**Encoding**:
```
for each channel i:
    latent[i] = (image_tensor[i] - mean[i]) / (std[i] / scaling_factor)
```

**Decoding**:
```
for each channel i:
    image_tensor[i] = latent[i] * (std[i] / scaling_factor) + mean[i]
```

Example for Wan 2.1 / Qwen Image (16 channels):
```python
mean = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
]
std = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
]
scaling_factor = 1.0

# Decoding
for i in range(16):
    image_tensor[i] = latent[i] * (std[i] / scaling_factor) + mean[i]
```

#### FLUX.2 Special Case

FLUX.2 has a complex reshaping operation during decode:

```python
mean = [/* 16-channel mean array */]
std = [/* 16-channel std array */]
scaling_factor = 1.0

# Decoding (simplified - actual implementation has reshape/permute ops)
# 1. Apply std * latent + mean
# 2. Reshape from [B, H, W, C] to [B, H/2, 2, W/2, 2, C]
# 3. Permute dimensions
# 4. Reshape back with doubled spatial resolution
```

This is significantly more complex - refer to `FirstStage.swift:73-83` for the exact implementation.

### How to Get Scaling Factors for Any Model

#### Method 1: Query via Echo (Recommended)

When you call the `Echo` RPC, parse the model specifications from `MetadataOverride`:

```python
import json

response = stub.Echo(EchoRequest(name="client"))

if response.override.models:
    models = json.loads(response.override.models)

    for model in models:
        print(f"Model: {model['name']}")
        print(f"  Version: {model['version']}")

        # Scaling factor (if present)
        if 'latentsScalingFactor' in model:
            print(f"  Scaling Factor: {model['latentsScalingFactor']}")

        # Mean and std (if present)
        if 'latentsMean' in model and 'latentsStd' in model:
            print(f"  Mean: {model['latentsMean']}")
            print(f"  Std: {model['latentsStd']}")

        # Note: shift factor is NOT in the specification
        # It must be calculated for FLUX models (see next section)
```

#### Method 2: Hardcode Based on Version

```python
def get_latent_scaling(model_version):
    """Get latent scaling parameters for a model version"""

    scaling_map = {
        'v1': (0.18215, None, None, None),
        'v2': (0.18215, None, None, None),
        'sdxlBase': (0.13025, None, None, None),
        'sdxlRefiner': (0.13025, None, None, None),
        'ssd1b': (0.13025, None, None, None),
        'pixart': (0.13025, None, None, None),
        'auraflow': (0.13025, None, None, None),
        'kandinsky21': (1.0, None, None, None),
        'wurstchenStageC': (2.32558139535, None, None, None),
        'wurstchenStageB': (2.32558139535, None, None, None),
        'sd3': (1.5305, 0.0609, None, None),
        'sd3Large': (1.5305, 0.0609, None, None),
        'flux1': (0.3611, 0.1159, None, None),
        'flux2': (1.0, None, FLUX2_MEAN, FLUX2_STD),
        'hiDreamI1': (0.3611, 0.1159, None, None),
        'zImage': (0.3611, 0.1159, None, None),
        'hunyuanVideo': (0.476986, None, None, None),
        'svdI2v': (0.18215, None, None, None),
        'wan21_1_3b': (1.0, None, WAN_MEAN, WAN_STD),
        'wan21_14b': (1.0, None, WAN_MEAN, WAN_STD),
        'wan22_5b': (1.0, None, WAN_MEAN, WAN_STD),
        'qwenImage': (1.0, None, WAN_MEAN, WAN_STD),
    }

    return scaling_map.get(model_version, (0.18215, None, None, None))

# Constants
WAN_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
]

WAN_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
]
```

### Common Mistakes

❌ **Wrong**: Using SD v1.5 scaling for all models
```python
# This breaks FLUX, SD3, SDXL!
latent = image / 0.18215
```

✅ **Correct**: Use model-specific scaling
```python
scaling_factor, shift_factor, mean, std = get_latent_scaling(model_version)

if mean is not None and std is not None:
    # Wan/Qwen path
    for i in range(len(mean)):
        latent[i] = (latent[i] * (std[i] / scaling_factor)) + mean[i]
elif shift_factor is not None:
    # FLUX/SD3 path
    latent = (latent / scaling_factor) + shift_factor
else:
    # SD/SDXL path
    latent = latent / scaling_factor
```

---

## Automatic Shift Calculation

### What is Shift?

The "shift" parameter controls the noise schedule for diffusion models. For FLUX.1 and other modern models, the optimal shift value **depends on the image resolution**. Draw Things automatically calculates this unless you explicitly disable it.

### The Problem

If you manually set `shift: 1.0` for all resolutions, you'll get suboptimal results at non-standard resolutions. The quality degradation can be significant.

### Resolution-Dependent Shift Formula

When `resolution_dependent_shift: true` (default), the shift is calculated as:

```python
import math

def calculate_shift(width, height):
    """
    Calculate shift value based on resolution.

    Args:
        width: Image width in pixels (not latent width!)
        height: Image height in pixels (not latent height!)

    Returns:
        Shift value (typically 0.5 to 1.15)
    """
    # Convert to latent resolution (divide by 8 for most models)
    latent_width = width // 8
    latent_height = height // 8

    # Total latent pixels multiplied by 16
    resolution_factor = (latent_height * latent_width) * 16

    # Formula from ModelZoo.swift:2358-2360
    shift = math.exp(
        ((resolution_factor - 256) * (1.15 - 0.5) / (4096 - 256)) + 0.5
    )

    return shift

# Examples
print(calculate_shift(1024, 1024))  # ~1.0 (standard)
print(calculate_shift(512, 512))    # ~0.64 (lower res)
print(calculate_shift(2048, 2048))  # ~1.15 (higher res)
print(calculate_shift(1536, 640))   # ~0.82 (widescreen)
```

### Breakdown of the Formula

The formula maps resolution to a shift range of 0.5 to 1.15:

```
resolution_factor = (latent_h * latent_w) * 16

shift = exp(
    ((resolution_factor - 256) * (1.15 - 0.5) / (4096 - 256)) + 0.5
)
```

Where:
- `256` = minimum resolution factor (16×16 latents = 128×128 pixels)
- `4096` = maximum resolution factor (64×64 latents = 512×512 pixels at 8x downscale, but formula works beyond this)
- `0.5` = minimum shift value
- `1.15` = maximum shift value
- `exp()` = exponential function to create non-linear mapping

### Resolution Examples

| Resolution | Latent Size | Shift Value | Notes |
|------------|-------------|-------------|-------|
| 512×512 | 64×64 | ~0.64 | Lower shift for small images |
| 768×768 | 96×96 | ~0.82 | |
| 1024×1024 | 128×128 | ~1.0 | Standard FLUX resolution |
| 1536×640 | 192×80 | ~0.82 | Widescreen |
| 1920×1080 | 240×135 | ~1.03 | HD |
| 2048×2048 | 256×256 | ~1.15 | High resolution (max shift) |

### Configuration in Requests

```python
# FlatBuffer configuration
config = {
    'resolution_dependent_shift': True,  # Enable automatic calculation
    'shift': 1.0,  # Fallback value if disabled
    # ... other params
}
```

### When to Disable Automatic Shift

You might want to set `resolution_dependent_shift: false` if:

1. **Testing**: You want consistent shift across resolutions for A/B testing
2. **Fine-tuned models**: Model was trained with specific shift value
3. **Artistic control**: You want manual control over the noise schedule

### Common Mistakes

❌ **Wrong**: Always using shift=1.0
```python
config = {
    'shift': 1.0,
    'resolution_dependent_shift': False  # Bad for non-standard resolutions
}
```

✅ **Correct**: Enable automatic calculation
```python
config = {
    'shift': 1.0,  # Fallback only
    'resolution_dependent_shift': True  # Let server calculate optimal shift
}
```

### Which Models Use Shift?

**FLUX.1 / FLUX.2 only** (as of current version). Other models ignore this parameter.

Check the model's `objective` field:
- If `objective` includes RF (Rectified Flow), shift is used
- SD3 uses shift factor in latent scaling (different concept)
- SD/SDXL models don't use shift

---

## High Resolution Fix

### What is Hires Fix?

High Resolution Fix (Hires Fix) is a two-pass generation technique that produces higher quality images at large resolutions. It works by:

1. **First Pass**: Generate at lower resolution
2. **Upscale**: Scale the latent to target resolution
3. **Second Pass**: Refine at full resolution with img2img

This produces better results than generating directly at high resolution because:
- Models are typically trained at 512×512 or 1024×1024
- Direct high-res generation can produce repetition/artifacts
- Two-pass allows composition at low-res, details at high-res

### How It Works

```
User requests: 2048×2048 image with hires fix

Step 1: Generate at 1024×1024 (hiresFixStart: 16×16 latent)
  ↓
Step 2: Upscale latent 2× to 2048×2048 (32×32 latent)
  ↓
Step 3: img2img refinement at 2048×2048 with strength 0.7
  ↓
Final: High quality 2048×2048 image
```

### Configuration Parameters

```python
config = {
    # Enable hires fix
    'hiresFix': True,

    # Starting resolution (in "scale" units, not pixels!)
    'hiresFixStartWidth': 16,   # 16 * 64 = 1024 pixels
    'hiresFixStartHeight': 16,  # 16 * 64 = 1024 pixels

    # Target resolution
    'startWidth': 32,   # 32 * 64 = 2048 pixels
    'startHeight': 32,  # 32 * 64 = 2048 pixels

    # Refinement strength (0.0 = no change, 1.0 = full regeneration)
    'hiresFixStrength': 0.7,

    # Other parameters
    'steps': 28,
    'guidanceScale': 7.5
}
```

### The Scale Factor Mystery

**Critical**: `startWidth` and `startHeight` are NOT in pixels! They're in "scale units" where:

```
pixels = scale_value * 64
```

For example:
- `startWidth: 16` = 16 × 64 = **1024 pixels**
- `startWidth: 8` = 8 × 64 = **512 pixels**
- `startWidth: 24` = 24 × 64 = **1536 pixels**

This also applies to `hiresFixStartWidth` and `hiresFixStartHeight`.

### Resolution Calculation

The actual pixel dimensions are:

```python
def scale_to_pixels(scale_value):
    """Convert scale units to pixels"""
    return scale_value * 64

def pixels_to_scale(pixels):
    """Convert pixels to scale units"""
    return pixels // 64

# Examples
print(scale_to_pixels(16))  # 1024 pixels
print(pixels_to_scale(2048))  # 32 scale units
```

### Hires Fix Workflow

```python
def generate_with_hires_fix(stub, prompt):
    """Generate 2048×2048 image using hires fix"""

    config = {
        'model': 'sd_xl_base_1.0_f16.ckpt',

        # First pass: 1024×1024
        'hiresFixStartWidth': 16,    # 1024px
        'hiresFixStartHeight': 16,   # 1024px

        # Final resolution: 2048×2048
        'startWidth': 32,            # 2048px
        'startHeight': 32,           # 2048px

        # Enable hires fix
        'hiresFix': True,
        'hiresFixStrength': 0.7,

        # Generation params
        'steps': 30,
        'guidanceScale': 7.5,
        'seed': 42
    }

    # Generate
    # Server will automatically:
    # 1. Generate at 1024×1024
    # 2. Upscale latent to 2048×2048
    # 3. Refine with img2img at strength 0.7

    return generate(stub, prompt, config)
```

### When Hires Fix is Applied

Hires fix is only activated when **ALL** conditions are met:

```python
def should_use_hires_fix(config):
    """Check if hires fix will be used"""

    # Must be explicitly enabled
    if not config.get('hiresFix', False):
        return False

    # Starting resolution must be set
    if config.get('hiresFixStartWidth', 0) <= 0:
        return False
    if config.get('hiresFixStartHeight', 0) <= 0:
        return False

    # Start resolution must be smaller than target
    if config['hiresFixStartWidth'] >= config['startWidth']:
        return False
    if config['hiresFixStartHeight'] >= config['startHeight']:
        return False

    # Strength must allow refinement steps
    sampler = get_sampler(config)
    init_timestep = sampler.timestep_for_strength(config['hiresFixStrength'])
    if init_timestep.start_step <= 0:
        return False

    return True
```

### Strength Parameter

The `hiresFixStrength` controls how much the second pass modifies the upscaled image:

| Strength | Effect | Use Case |
|----------|--------|----------|
| 0.0 - 0.3 | Minimal changes | Subtle detail enhancement |
| 0.4 - 0.6 | Moderate refinement | Balanced quality improvement |
| **0.7** | Significant refinement | **Default/recommended** |
| 0.8 - 0.9 | Heavy modification | When first pass is rough |
| 1.0 | Complete regeneration | Effectively just high-res generation |

### Typical Hires Fix Scales

Common configurations:

```python
# SD v1.5: 512 → 1024
{
    'hiresFixStartWidth': 8,   # 512px
    'hiresFixStartHeight': 8,  # 512px
    'startWidth': 16,          # 1024px
    'startHeight': 16,         # 1024px
}

# SDXL: 1024 → 2048
{
    'hiresFixStartWidth': 16,  # 1024px
    'hiresFixStartHeight': 16, # 1024px
    'startWidth': 32,          # 2048px
    'startHeight': 32,         # 2048px
}

# FLUX: 1024 → 1536
{
    'hiresFixStartWidth': 16,  # 1024px
    'hiresFixStartHeight': 16, # 1024px
    'startWidth': 24,          # 1536px
    'startHeight': 24,         # 1536px
}
```

### Model-Specific Hires Fix Scale

Some models have a default hires fix scale in their specification:

```python
model_spec = {
    'name': 'Some Model',
    'hiresFixScale': 16,  # Default starting resolution
    # ...
}

# Use this as starting point
config['hiresFixStartWidth'] = model_spec.get('hiresFixScale', 8)
config['hiresFixStartHeight'] = model_spec.get('hiresFixScale', 8)
```

### Common Mistakes

❌ **Wrong**: Using pixel values directly
```python
config = {
    'hiresFixStartWidth': 1024,  # This is 65,536 pixels!
    'startWidth': 2048,          # This is 131,072 pixels!
}
```

✅ **Correct**: Use scale units
```python
config = {
    'hiresFixStartWidth': 16,  # 1024 pixels
    'startWidth': 32,          # 2048 pixels
}
```

---

## Resolution Constraints

### The Scale System

Draw Things uses a "scale" system where dimensions are multiples of 64 pixels:

```
actual_pixels = scale_units * 64
```

This is because:
1. VAE downsamples by 8× (most models)
2. Latent space operates on 8×8 blocks
3. 64 = 8 × 8, ensuring clean divisions

### Valid Resolutions

**Only resolutions that are multiples of 64 pixels are valid:**

✅ Valid:
- 512×512 (8×8 scale)
- 768×768 (12×12 scale)
- 1024×1024 (16×16 scale)
- 1536×640 (24×10 scale)
- 2048×896 (32×14 scale)

❌ Invalid:
- 1000×1000 (not divisible by 64)
- 1920×1080 (1920 is divisible, but 1080 is not)
- 800×600 (800 is not divisible by 64)

### How to Validate Resolution

```python
def is_valid_resolution(width, height):
    """Check if resolution is valid for Draw Things"""
    return (width % 64 == 0) and (height % 64 == 0)

def nearest_valid_resolution(width, height):
    """Round to nearest valid resolution"""
    valid_width = round(width / 64) * 64
    valid_height = round(height / 64) * 64
    return valid_width, valid_height

# Examples
print(is_valid_resolution(1024, 1024))  # True
print(is_valid_resolution(1920, 1080))  # False

print(nearest_valid_resolution(1920, 1080))  # (1920, 1088)
print(nearest_valid_resolution(1000, 1000))  # (1024, 1024)
```

### Device-Dependent Default Scale

The default resolution depends on device memory:

```python
def get_default_scale():
    """Get default scale based on device performance"""

    physical_memory = get_physical_memory_bytes()

    if physical_memory < 3.5 * 1024**3:  # < 3.5 GB
        # Low performance (3GB devices like iPad Air 2)
        return (6, 6)  # 384×384 pixels

    elif physical_memory < 5 * 1024**3:  # < 5 GB
        # Medium performance (4GB devices)
        return (6, 6)  # 384×384 pixels

    else:  # >= 5 GB
        # Good performance (6GB+ devices, desktops)
        return (8, 8)  # 512×512 pixels

# Usage
default_width, default_height = get_default_scale()
print(f"Default: {default_width * 64}×{default_height * 64} pixels")
```

On desktop/server (no memory constraints):
- Default is typically `(8, 8)` = **512×512 pixels**
- But you can use any valid resolution

### Aspect Ratio Preservation

When scaling to fit a specific aspect ratio while maintaining valid resolutions:

```python
def fit_to_aspect_ratio(target_area, aspect_ratio):
    """
    Calculate dimensions that fit target area and aspect ratio.

    Args:
        target_area: Total pixels (e.g., 8*8 for 512×512)
        aspect_ratio: Width/height ratio (e.g., 16/9)

    Returns:
        (width_scale, height_scale) in scale units
    """
    import math

    # Calculate dimensions
    height_scale = math.sqrt(target_area / aspect_ratio)
    width_scale = height_scale * aspect_ratio

    # Round to nearest integer
    width_scale = round(width_scale)
    height_scale = round(height_scale)

    # Ensure we don't exceed target area
    while width_scale * height_scale > target_area:
        if width_scale > height_scale:
            width_scale -= 1
        else:
            height_scale -= 1

    return width_scale, height_scale

# Examples
print(fit_to_aspect_ratio(64, 16/9))   # (10, 6) = 640×384
print(fit_to_aspect_ratio(64, 1))      # (8, 8) = 512×512
print(fit_to_aspect_ratio(256, 16/9))  # (20, 13) = 1280×832
```

### Model-Specific Constraints

Some models have additional resolution constraints:

#### FLUX Models

FLUX prefers resolutions where both dimensions are multiples of 16 pixels (not just 64):

```python
def is_valid_flux_resolution(width, height):
    """FLUX works better with 16-pixel alignment"""
    return (width % 16 == 0) and (height % 16 == 0)

# Still use 64-pixel multiples for compatibility
# But FLUX is more flexible
```

#### SD3 Models

SD3 works with standard 64-pixel multiples.

#### Video Models (SVD, Hunyuan Video)

Video models may have stricter requirements:
- Both dimensions must be multiples of 64
- Frame count must be specific values (e.g., 14 for SVD)
- Aspect ratios may be constrained

### Common Resolution Presets

Standard resolutions that work well:

```python
STANDARD_RESOLUTIONS = {
    # Square
    'SD_512': (8, 8),      # 512×512
    'SD_768': (12, 12),    # 768×768
    'SDXL_1024': (16, 16), # 1024×1024
    'FLUX_1024': (16, 16), # 1024×1024

    # Landscape 16:9
    'HD_READY': (20, 11),  # 1280×704
    'FULL_HD': (30, 17),   # 1920×1088

    # Portrait 9:16
    'MOBILE': (11, 20),    # 704×1280

    # Widescreen 21:9
    'ULTRAWIDE': (24, 10), # 1536×640

    # Cinema 2.35:1
    'CINEMA': (24, 10),    # 1536×640
}

def get_resolution_pixels(preset):
    """Get pixel dimensions from preset"""
    w, h = STANDARD_RESOLUTIONS[preset]
    return w * 64, h * 64
```

### Configuration in Requests

```python
# Correct way to specify resolution
config = {
    'startWidth': 16,   # 1024 pixels
    'startHeight': 16,  # 1024 pixels
    # NOT: 'width': 1024, 'height': 1024
}

# For img2img, also set original dimensions
config_img2img = {
    'startWidth': 16,
    'startHeight': 16,
    'originalImageWidth': 1024,   # In actual pixels
    'originalImageHeight': 1024,  # In actual pixels
}
```

---

## Model-Specific Defaults

### Default Scale Values

Each model has a default scale in its specification:

```python
MODEL_DEFAULT_SCALES = {
    'v1': 8,           # 512×512
    'v2': 8,           # 512×512
    'sdxlBase': 16,    # 1024×1024
    'sdxlRefiner': 16, # 1024×1024
    'flux1': 16,       # 1024×1024
    'flux2': 16,       # 1024×1024
    'sd3': 16,         # 1024×1024
    'sd3Large': 16,    # 1024×1024
    'qwenImage': 16,   # 1024×1024
    # etc.
}

def get_default_scale_for_model(model_version):
    """Get recommended default scale for a model"""
    return MODEL_DEFAULT_SCALES.get(model_version, 8)
```

### Retrieving from Server

The Echo response includes default scale:

```python
response = stub.Echo(EchoRequest(name="client"))
models = json.loads(response.override.models)

for model in models:
    default_scale = model.get('defaultScale', 8)
    print(f"{model['name']}: {default_scale} ({default_scale * 64}px)")
```

### Other Model-Specific Parameters

#### Guidance Scale Defaults

Different model families have different optimal guidance scales:

| Model Family | Default Guidance | Range |
|--------------|------------------|-------|
| SD v1.x | 7.5 | 5.0 - 15.0 |
| SD v2.x | 7.5 | 5.0 - 15.0 |
| SDXL | 7.5 | 5.0 - 12.0 |
| FLUX.1 Dev | 3.5 | 2.0 - 5.0 |
| FLUX.1 Schnell | 0.0 | 0.0 (distilled) |
| SD3 | 4.5 | 3.0 - 7.0 |

#### Step Count Defaults

| Model Type | Recommended Steps |
|------------|------------------|
| Standard (DDPM) | 20-50 |
| SDXL | 30-40 |
| FLUX.1 Dev | 20-28 |
| FLUX.1 Schnell | 4 |
| LCM (with LoRA) | 4-8 |
| TCD (with LoRA) | 4-8 |

#### Sampler Compatibility

Not all samplers work with all models:

```python
def is_compatible_sampler(model_objective, sampler):
    """Check if sampler works with model objective"""

    # Rectified Flow models (FLUX)
    if model_objective == 'rf':
        # Only trailing/AYS samplers
        compatible = [
            'DPMPP2MAYS', 'DPMPPSDEAYS', 'dDIMTrailing',
            'dPMPP2MTrailing', 'dPMPPSDETrailing',
            'eulerAAYS', 'eulerATrailing',
            'uniPCAYS', 'uniPCTrailing'
        ]
        return sampler in compatible

    # EDM/v/epsilon models (SD, SDXL, SD3)
    else:
        return True  # All samplers work
```

---

## Complete Example: Robust Generation

Here's a complete example that handles all the quirks correctly:

```python
import grpc
import json
import math
from generated import imageService_pb2, imageService_pb2_grpc

def robust_generate(stub, model_name, prompt, width_px, height_px):
    """
    Generate image with all quirks handled correctly.

    Args:
        model_name: Model filename
        prompt: Text prompt
        width_px: Desired width in pixels
        height_px: Desired height in pixels
    """

    # 1. Get model information
    echo_resp = stub.Echo(imageService_pb2.EchoRequest(name="client"))
    models = json.loads(echo_resp.override.models)

    model_spec = None
    for m in models:
        if m['file'] == model_name:
            model_spec = m
            break

    if not model_spec:
        raise ValueError(f"Model {model_name} not found")

    # 2. Validate and adjust resolution
    width_px = round(width_px / 64) * 64
    height_px = round(height_px / 64) * 64

    width_scale = width_px // 64
    height_scale = height_px // 64

    print(f"Adjusted resolution: {width_px}×{height_px}")

    # 3. Get model-specific parameters
    model_version = model_spec['version']
    default_scale = model_spec.get('defaultScale', 8)

    # Latent scaling
    scaling_factor = model_spec.get('latentsScalingFactor')
    if not scaling_factor:
        scaling_factor = get_default_scaling_for_version(model_version)

    print(f"Model: {model_spec['name']}")
    print(f"Version: {model_version}")
    print(f"Scaling factor: {scaling_factor}")

    # 4. Build configuration
    config = {
        'model': model_name,
        'startWidth': width_scale,
        'startHeight': height_scale,
        'steps': get_default_steps(model_version),
        'guidanceScale': get_default_guidance(model_version),
        'seed': 42,
        'sampler': get_compatible_sampler(model_version),
        'batchCount': 1,

        # Resolution-dependent shift (for FLUX)
        'resolutionDependentShift': True,
        'shift': 1.0,  # Fallback

        # Hires fix for large images
        'hiresFix': should_use_hires_fix(width_scale, height_scale, default_scale),
        'hiresFixStartWidth': default_scale,
        'hiresFixStartHeight': default_scale,
        'hiresFixStrength': 0.7,
    }

    # 5. Serialize and generate
    config_bytes = serialize_configuration(config)

    request = imageService_pb2.ImageGenerationRequest(
        prompt=prompt,
        negativePrompt="blurry, low quality",
        configuration=config_bytes,
        scaleFactor=1,
        override=imageService_pb2.MetadataOverride(),
        user="RobustClient",
        device=imageService_pb2.LAPTOP,
        chunked=True
    )

    # 6. Generate and handle scaling correctly
    images = []
    for response in stub.GenerateImage(request):
        if response.generatedImages and \
           response.chunkState == imageService_pb2.LAST_CHUNK:
            for img_data in response.generatedImages:
                # Decompress with correct scaling
                tensor = decompress_tensor(img_data)
                decoded = decode_latent(
                    tensor,
                    scaling_factor=scaling_factor,
                    model_version=model_version
                )
                images.append(tensor_to_image(decoded))

    return images

def get_default_scaling_for_version(version):
    """Fallback scaling factors"""
    mapping = {
        'v1': 0.18215, 'v2': 0.18215,
        'sdxlBase': 0.13025, 'sdxlRefiner': 0.13025,
        'flux1': 0.3611, 'flux2': 1.0,
        'sd3': 1.5305, 'sd3Large': 1.5305,
    }
    return mapping.get(version, 0.18215)

def get_default_steps(version):
    """Model-appropriate step counts"""
    if 'flux1' in version.lower() and 'schnell' in version.lower():
        return 4
    elif 'flux' in version.lower():
        return 28
    elif 'sdxl' in version.lower():
        return 30
    else:
        return 20

def get_default_guidance(version):
    """Model-appropriate guidance scale"""
    if 'schnell' in version.lower():
        return 0.0
    elif 'flux' in version.lower():
        return 3.5
    elif 'sd3' in version.lower():
        return 4.5
    else:
        return 7.5

def get_compatible_sampler(version):
    """Get compatible sampler for model"""
    if version in ['flux1', 'flux2']:
        return 1  # EULER_A
    else:
        return 0  # DPMPP_2M_KARRAS

def should_use_hires_fix(width, height, default_scale):
    """Decide if hires fix would help"""
    total_area = width * height
    default_area = default_scale * default_scale

    # Use hires fix if target is >2x default area
    return total_area > (default_area * 2)

def decode_latent(latent, scaling_factor, model_version):
    """Decode latent with model-specific scaling"""

    # This is a simplified version
    # Real implementation needs mean/std handling for Wan/Qwen

    if model_version in ['flux1', 'hiDreamI1', 'zImage']:
        shift_factor = 0.1159
        return (latent / scaling_factor) + shift_factor
    elif model_version in ['sd3', 'sd3Large']:
        shift_factor = 0.0609
        return (latent / scaling_factor) + shift_factor
    else:
        return latent / scaling_factor
```

---

## Quick Reference

### Checklist for Third-Party Developers

✅ **Latent Scaling**
- [ ] Query model specification for `latentsScalingFactor`
- [ ] Check for shift factor (SD3, FLUX)
- [ ] Check for mean/std arrays (Wan, Qwen)
- [ ] Implement correct decode formula for model family

✅ **Shift Calculation**
- [ ] Enable `resolutionDependentShift: true` for FLUX models
- [ ] Implement shift calculation formula if needed
- [ ] Use appropriate fallback shift value

✅ **High Resolution Fix**
- [ ] Convert pixels to scale units (÷64)
- [ ] Set `hiresFixStartWidth/Height` < `startWidth/Height`
- [ ] Use appropriate strength (0.7 recommended)
- [ ] Understand it's a two-pass process

✅ **Resolution Constraints**
- [ ] Ensure width and height are multiples of 64
- [ ] Use scale units in configuration, not pixels
- [ ] Round user input to nearest valid resolution
- [ ] Validate before sending request

✅ **Model Defaults**
- [ ] Query default scale from Echo response
- [ ] Use model-appropriate guidance scale
- [ ] Use model-appropriate step count
- [ ] Check sampler compatibility

---

## Debugging Common Issues

### "Image is completely wrong/noisy"

**Likely cause**: Wrong latent scaling factor

**Solution**:
```python
# Check you're using the right scaling for the model
print(f"Using scaling: {scaling_factor}")
print(f"Model version: {model_version}")

# Compare with correct values from this guide
```

### "Server returns resolution error"

**Likely cause**: Resolution not multiple of 64

**Solution**:
```python
# Always validate
width = round(width / 64) * 64
height = round(height / 64) * 64
```

### "Hires fix not working"

**Likely cause**: Using pixel values instead of scale units

**Solution**:
```python
# Wrong
config['hiresFixStartWidth'] = 1024

# Correct
config['hiresFixStartWidth'] = 16  # 1024/64
```

### "FLUX images poor quality at custom resolutions"

**Likely cause**: Disabled resolution-dependent shift

**Solution**:
```python
config['resolutionDependentShift'] = True
```

### "Can't load generated image"

**Likely cause**: Wrong tensor decompression or scaling

**Solution**:
```python
# Ensure you decompress first, then apply latent scaling
tensor = decompress_tensor(data)  # Get latent
image = decode_latent(tensor, scaling_factor, shift_factor)  # Scale correctly
pil_image = tensor_to_image(image)  # Convert to image
```

---

## Additional Resources

- **Source Code References**:
  - Latent scaling: `Libraries/SwiftDiffusion/Sources/FirstStage.swift:20-91`
  - Shift calculation: `Libraries/ModelZoo/Sources/ModelZoo.swift:2357-2361`
  - Hires fix: `Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift:3609-3724`
  - Resolution validation: `Libraries/DataModels/Sources/DeviceCapability.swift:138-166`

- **Related Documentation**:
  - `BridgeMode.md`: Authentication and cloud infrastructure
  - `ThirdPartyIntegration-gRPCServer.md`: Complete gRPC API guide
  - `ThirdPartyIntegration-BridgeMode.md`: JavaScript API reference

- **Model Zoo**:
  - All model specifications: `Libraries/ModelZoo/Sources/ModelZoo.swift`
  - LoRA specifications: `Libraries/ModelZoo/Sources/LoRAZoo.swift`
  - ControlNet specifications: `Libraries/ModelZoo/Sources/ControlNetZoo.swift`
