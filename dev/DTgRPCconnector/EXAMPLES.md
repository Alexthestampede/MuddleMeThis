# Draw Things gRPC Client - Examples

Practical code examples for all major features of the Draw Things gRPC Python client.

## Table of Contents

- [Text-to-Image Generation](#text-to-image-generation)
- [Edit and Kontext Models](#edit-and-kontext-models)
- [Reference Images / Moodboard](#reference-images--moodboard)
- [Inpainting](#inpainting)
- [ControlNet](#controlnet)
- [LoRA Models](#lora-models)
- [Advanced Configuration](#advanced-configuration)
  - [FLUX Parameters](#flux-parameters)
  - [SDXL Parameters](#sdxl-parameters)
  - [Separate Text Encoders](#separate-text-encoders)
  - [TeaCache Acceleration](#teacache-acceleration)
  - [Tiled Diffusion and Decoding](#tiled-diffusion-and-decoding)
  - [Video Generation (SVD / LTX 2.3)](#video-generation-svd--ltx-23)
  - [Stage 2 Models](#stage-2-models)
  - [Causal Inference](#causal-inference)
- [Model Discovery](#model-discovery)
- [Tensor Encoding and Decoding](#tensor-encoding-and-decoding)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

---

## Text-to-Image Generation

### Basic Generation (CLI)

```bash
python examples/generate_image.py "a cute puppy playing in a garden"
```

### Basic Generation (Python API)

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=20,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a cute puppy playing in a garden",
        config=config,
        negative_prompt="blurry, bad quality, distorted",
    )
    client.save_images(images, output_dir="output", prefix="puppy")
```

### FLUX Schnell (Ultra-Fast)

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="flux_1_schnell_q8p.ckpt",
    steps=4,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    guidance_embed=3.5,
    t5_text_encoder=True,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a majestic dragon flying over a medieval castle",
        config=config,
    )
    client.save_images(images, output_dir="output", prefix="dragon")
```

### SDXL

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="juggernaut_xl_v9_q6p_q8p.ckpt",
    steps=30,
    width=1024,
    height=1024,
    cfg_scale=6.0,
    scheduler="DPMPP2M Karras",
    original_image_width=1024,
    original_image_height=1024,
    target_image_width=1024,
    target_image_height=1024,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="epic mountain landscape at sunset, dramatic clouds, golden hour lighting, photorealistic",
        config=config,
        negative_prompt="cartoon, painting, illustration, blurry",
    )
    client.save_images(images, output_dir="output", prefix="landscape")
```

### Reproducible Results with Seed

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=25,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
    seed=42,     # fixed seed for reproducibility
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="cute cartoon cat wearing a wizard hat",
        config=config,
    )
    client.save_images(images, output_dir="output", prefix="cat_wizard")
```

---

## Reference Images / Moodboard

Edit models (FLUX.2 Klein, etc.) can use **multiple reference images** for composition. Each reference is VAE-encoded as visual tokens and fed to the model. This lets you prompt combinations like "the person from image 3, wearing the clothes from image 2, on the beach from image 1".

### Multiple References for Edit Models

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, ReferenceImage

config = ImageGenerationConfig(
    model="flux_2_klein_4b_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    strength=1.0,          # required for edit/kontext models
    guidance_embed=3.5,
    t5_text_encoder=True,
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
    client.save_images(images, output_dir="output", prefix="moodboard_edit")
```

### IP-Adapter Style References

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, ReferenceImage

config = ImageGenerationConfig(
    model="realvisionxl_v40inpaint_q6p_q8p.ckpt",
    steps=25,
    width=1024,
    height=1024,
    cfg_scale=7.0,
    scheduler="DPM++ 2M Karras",
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a warrior in fantasy armor, dramatic lighting",
        config=config,
        reference_images=[
            ReferenceImage(image="style_ref.jpg", weight=0.8, hint_type="ipadapterplus"),
        ],
    )
    client.save_images(images, output_dir="output", prefix="ipadapter_style")
```

---

## Edit and Kontext Models

Kontext models (such as Flux Klein) use an input image as a **reference**. The model VAE-encodes the image into reference latents and feeds them as visual tokens to the text encoder. This gives the model full semantic understanding of the reference image.

**Key requirements:**
- `strength` must be `1.0` (the reference encoding is full-strength by design)
- The input image must match the generation dimensions (`width` x `height`)
- Images must be encoded as uncompressed float16 - `DrawThingsClient` does this automatically

### Flux Klein / Kontext Edit

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="flux_klein_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    strength=1.0,          # required for kontext/edit models
    guidance_embed=3.5,    # FLUX guidance embed
    t5_text_encoder=True,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="the same scene but in winter, heavy snowfall",
        config=config,
        input_image="reference.png",   # resized to 1024x1024 automatically
    )
    client.save_images(images, output_dir="output", prefix="winter_edit")
```

### Style Transfer with Reference Image

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="flux_klein_q6p.ckpt",
    steps=25,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    strength=1.0,
    guidance_embed=3.5,
    t5_text_encoder=True,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="convert to oil painting style, impressionist brushstrokes",
        config=config,
        input_image="portrait.jpg",
    )
    client.save_images(images, output_dir="output", prefix="oil_painting")
```

### Qwen Edit Models

Qwen edit models work similarly but use a vision-language encoder. A Lightning LoRA reduces steps from ~30 to 4.

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, LoRAConfig

config = ImageGenerationConfig(
    model="qwen_image_edit_2511_q6p.ckpt",
    steps=4,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    strength=1.0,
    loras=[
        LoRAConfig(
            file="qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt",
            weight=1.0,
        )
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="add warm autumn colors to the trees",
        config=config,
        input_image="landscape.jpg",
    )
    client.save_images(images, output_dir="output", prefix="autumn")
```

---

## Inpainting

Inpainting uses a mask image to define which areas to regenerate. White pixels in the mask indicate areas to repaint; black pixels are preserved.

### Basic Inpainting

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
    mask_blur=2.5,
    preserve_original_after_inpaint=True,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a red apple on a wooden table",
        config=config,
        input_image="scene.png",
        mask_image="mask.png",    # white = repaint, black = keep
    )
    client.save_images(images, output_dir="output", prefix="inpainted")
```

### Creating a Mask Programmatically

```python
from PIL import Image, ImageDraw
import numpy as np

# Load the source image to get dimensions
source = Image.open("scene.png")
width, height = source.size

# Create a black mask (preserve everything)
mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)

# Paint the region to repaint as white
# Here we repaint a 200x200 area in the center
cx, cy = width // 2, height // 2
draw.rectangle(
    [cx - 100, cy - 100, cx + 100, cy + 100],
    fill=255
)

mask.save("mask.png")
```

### Using ControlNet Inpaint (Built-In)

For inpainting with the built-in ControlNet inpaint mode, use an empty `file` and `input_override=ControlInputType.Inpaint`:

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, ControlNetConfig
from Control import ControlInputType

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=25,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
    strength=1.0,
    controls=[
        ControlNetConfig(
            file="",                              # empty = built-in
            input_override=ControlInputType.Inpaint,
            weight=1.0,
        )
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a golden retriever",
        config=config,
        input_image="scene.png",
        mask_image="mask.png",
    )
    client.save_images(images, output_dir="output", prefix="controlnet_inpaint")
```

---

## ControlNet

ControlNet conditions image generation on structural guides such as edge maps, depth maps, or pose skeletons.

### Canny Edge Control

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
        prompt="a glass building reflecting the sky",
        config=config,
        input_image="edge_map.png",  # pre-computed canny edges
    )
    client.save_images(images, output_dir="output", prefix="canny_controlled")
```

### Depth Map Control

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, ControlNetConfig
from Control import ControlInputType

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=25,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
    controls=[
        ControlNetConfig(
            file="control_v11f1p_sd15_depth.ckpt",
            weight=0.8,
            input_override=ControlInputType.Depth,
        )
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a futuristic living room, warm lighting",
        config=config,
        input_image="depth_map.png",
    )
    client.save_images(images, output_dir="output", prefix="depth_controlled")
```

### Pose Control (OpenPose)

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, ControlNetConfig
from Control import ControlInputType

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=25,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
    controls=[
        ControlNetConfig(
            file="control_v11p_sd15_openpose.ckpt",
            weight=1.0,
            input_override=ControlInputType.Pose,
        )
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a dancer on stage, dramatic lighting",
        config=config,
        input_image="pose_skeleton.png",
    )
    client.save_images(images, output_dir="output", prefix="pose_controlled")
```

### Multiple ControlNets

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, ControlNetConfig
from Control import ControlInputType

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=30,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
    controls=[
        ControlNetConfig(
            file="control_v11p_sd15_canny.ckpt",
            weight=0.7,
            guidance_end=0.6,          # apply only for first 60% of steps
            input_override=ControlInputType.Canny,
        ),
        ControlNetConfig(
            file="control_v11f1p_sd15_depth.ckpt",
            weight=0.5,
            guidance_start=0.2,        # begin at 20% of steps
            guidance_end=1.0,
            input_override=ControlInputType.Depth,
        ),
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a medieval castle at dusk",
        config=config,
        input_image="reference.png",
    )
    client.save_images(images, output_dir="output", prefix="multi_control")
```

### Tile ControlNet (Super-Resolution / Upscaling Detail)

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, ControlNetConfig
from Control import ControlInputType

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=25,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
    controls=[
        ControlNetConfig(
            file="control_v11f1e_sd15_tile.ckpt",
            weight=1.0,
            input_override=ControlInputType.Tile,
        )
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="highly detailed, sharp, 8k quality",
        config=config,
        input_image="low_res.png",
    )
    client.save_images(images, output_dir="output", prefix="tile_upscale")
```

---

## LoRA Models

LoRA adapters are passed directly inside the FlatBuffer generation config via `LoRAConfig`. Multiple LoRAs stack additively.

### Single LoRA

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, LoRAConfig

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=20,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
    loras=[
        LoRAConfig(file="anime_style_lora_f16.ckpt", weight=0.8),
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a warrior in anime style",
        config=config,
    )
    client.save_images(images, output_dir="output", prefix="anime_warrior")
```

### Multiple LoRAs

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, LoRAConfig

config = ImageGenerationConfig(
    model="flux_1_dev_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    guidance_embed=3.5,
    loras=[
        LoRAConfig(file="detail_enhancer_lora_f16.ckpt", weight=0.6),
        LoRAConfig(file="lighting_style_lora_f16.ckpt",  weight=0.5),
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a portrait with dramatic cinematic lighting",
        config=config,
    )
    client.save_images(images, output_dir="output", prefix="portrait")
```

### Lightning LoRA (4-Step Fast Generation)

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig, LoRAConfig

config = ImageGenerationConfig(
    model="qwen_image_edit_2511_q6p.ckpt",
    steps=4,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    strength=1.0,
    loras=[
        LoRAConfig(
            file="qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt",
            weight=1.0,
        )
    ],
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a mystical forest with glowing mushrooms",
        config=config,
    )
    client.save_images(images, output_dir="output", prefix="mystical_forest")
```

### LoRA Mode (Base vs. Refiner)

For two-stage models (SDXL with refiner), apply a LoRA only to the base or refiner:

```python
from drawthings_client import LoRAConfig

# Apply only during base pass
LoRAConfig(file="style_lora.ckpt", weight=0.7, mode=1)   # mode=1 = Base

# Apply only during refiner pass
LoRAConfig(file="detail_lora.ckpt", weight=0.5, mode=2)  # mode=2 = Refiner

# Apply during both (default)
LoRAConfig(file="general_lora.ckpt", weight=0.6, mode=0) # mode=0 = All
```

---

## Advanced Configuration

### FLUX Parameters

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="flux_1_dev_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",

    # FLUX-specific
    guidance_embed=3.5,                 # guidance scale for FLUX guidance embed
    speed_up_with_guidance_embed=True,  # enables guidance embed acceleration
    resolution_dependent_shift=True,    # shift parameter scales with resolution

    # T5 text encoder
    t5_text_encoder=True,               # enable T5 for FLUX
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a futuristic city at night, neon reflections on wet pavement",
        config=config,
    )
    client.save_images(images, output_dir="output", prefix="flux_city")
```

### SDXL Parameters

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="juggernaut_xl_v9_q6p_q8p.ckpt",
    steps=30,
    width=1024,
    height=1024,
    cfg_scale=6.0,
    scheduler="DPMPP2M Karras",

    # SDXL conditioning
    original_image_width=1024,     # original image resolution hint
    original_image_height=1024,
    target_image_width=1024,       # target resolution hint
    target_image_height=1024,
    crop_top=0,
    crop_left=0,
    aesthetic_score=6.0,           # positive aesthetic conditioning
    negative_aesthetic_score=2.5,  # negative aesthetic conditioning

    # SDXL refiner
    refiner_model="sdxl_refiner_1.0_q6p_q8p.ckpt",
    refiner_start=0.7,             # switch to refiner at 70% of steps

    # Separate CLIP-L and OpenCLIP-G prompts (advanced)
    separate_clip_l=True,
    clip_l_text="sharp focus, 8k resolution",
    separate_open_clip_g=True,
    open_clip_g_text="professional photography, masterpiece",
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="professional portrait of a woman with flowing red hair, bokeh background",
        config=config,
        negative_prompt="cartoon, anime, painting, blurry, distorted",
    )
    client.save_images(images, output_dir="output", prefix="sdxl_portrait")
```

### Separate Text Encoders

For models with multiple text encoders, you can provide different text to each:

```python
from drawthings_client import ImageGenerationConfig

# SDXL has CLIP-L and OpenCLIP-G
config = ImageGenerationConfig(
    model="juggernaut_xl_v9_q6p_q8p.ckpt",
    steps=30,
    width=1024,
    height=1024,
    cfg_scale=6.0,
    scheduler="DPMPP2M Karras",
    # CLIP-L tends to handle short style tags
    separate_clip_l=True,
    clip_l_text="cinematic photography",
    # OpenCLIP-G handles longer descriptive prompts better
    separate_open_clip_g=True,
    open_clip_g_text="a majestic snow leopard sitting on a rocky mountain peak at golden hour",
)

# FLUX has CLIP-L and T5
config_flux = ImageGenerationConfig(
    model="flux_1_dev_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    guidance_embed=3.5,
    # T5 handles long, detailed descriptions
    separate_t5=True,
    t5_text="A highly detailed macro photograph of a dewdrop on a spider web at sunrise, golden bokeh background, 8k resolution",
    # CLIP-L for short style hint
    separate_clip_l=True,
    clip_l_text="macro photography, 8k",
)
```

### TeaCache Acceleration

TeaCache reduces computation by caching and reusing intermediate activations for similar timesteps. It trades a small quality reduction for significant speed gains.

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="flux_1_dev_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    guidance_embed=3.5,

    # TeaCache
    tea_cache=True,
    tea_cache_start=5,           # start caching after step 5
    tea_cache_end=-1,            # -1 means run until end
    tea_cache_threshold=0.06,    # similarity threshold; higher = more aggressive
    tea_cache_max_skip_steps=3,  # max consecutive skipped steps
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="an astronaut floating in space",
        config=config,
    )
    client.save_images(images, output_dir="output", prefix="teacache_output")
```

### Tiled Diffusion and Decoding

For generating or decoding large images, tiling prevents out-of-memory errors:

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=30,
    width=2048,
    height=2048,
    cfg_scale=7.0,
    scheduler="UniPC ays",

    # Tiled diffusion - for high-res generation
    tiled_diffusion=True,
    diffusion_tile_width=16,     # tile width in scale units (16 * 64 = 1024px)
    diffusion_tile_height=16,
    diffusion_tile_overlap=2,    # overlap in scale units (2 * 64 = 128px)

    # Tiled decoding - for high-res VAE decode
    tiled_decoding=True,
    decoding_tile_width=10,      # tile width in scale units (10 * 64 = 640px)
    decoding_tile_height=10,
    decoding_tile_overlap=2,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a vast panoramic landscape, photorealistic",
        config=config,
    )
    client.save_images(images, output_dir="output", prefix="large_image")
```

### Video Generation (SVD)

Stable Video Diffusion generates a short video from a single image:

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="svd_xt_q6p.ckpt",
    steps=25,
    width=1024,
    height=576,
    cfg_scale=1.0,
    scheduler="Euler A",
    strength=1.0,

    # Video parameters
    fps_id=6,              # frames per second ID
    motion_bucket_id=127,  # motion intensity (0=still, 255=very dynamic)
    cond_aug=0.02,         # conditioning augmentation strength
    start_frame_cfg=1.0,   # CFG scale for the first frame
    num_frames=25,         # number of frames to generate
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    # Pass the first frame as input
    images = client.generate_image(
        prompt="",   # SVD does not use text prompt
        config=config,
        input_image="first_frame.png",
    )
    # Returns individual frame tensors; assemble into video externally
    client.save_images(images, output_dir="output/frames", prefix="frame")
```

### Video Generation (LTX 2.3 with Audio)

LTX 2.3 generates video from text prompts and returns synchronized stereo audio. Use `generate_media()` to capture both frames and audio, then `save_video()` to mux them into a playable file.

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="ltx_2_3_540p_q8p.ckpt",  # or 720p variant
    steps=30,
    width=960,              # 540p (540 * 16 / 9)
    height=540,
    cfg_scale=1.0,
    scheduler="Euler A",

    # Video parameters
    num_frames=121,         # LTX 2.3 supports up to 257 frames
    fps_id=24,
    motion_bucket_id=127,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    result = client.generate_media(
        prompt="a cinematic drone shot flying over a tropical beach at sunset",
        config=config,
    )

    # Save individual frames
    client.save_images(result.images, output_dir="output/ltx_frames", prefix="frame")

    # Save muxed video with audio (requires imageio + imageio-ffmpeg)
    if result.images:
        client.save_video(
            result.images,
            output_path="output/ltx_video.mp4",
            fps=24,
            audio=b"".join(result.audio) if result.audio else None,
        )
```

**Notes:**
- LTX 2.3 VAE uses different latent ratios than image models; the server internally computes `startScaleFactor = 32` and doubles the scale. Pass pixel dimensions that are multiples of 64 (e.g., 540p, 720p) to avoid reshape crashes.
- `generate_image()` will also return LTX frames but silently discards the audio track. Use `generate_media()` for full audio-video capture.
- The audio bytes are raw stereo float32 PCM at 44.1 kHz. `save_video()` handles the FFmpeg muxing for you.

### Stage 2 Models

Stable Cascade and similar two-stage architectures use `stage_2_*` parameters for the second pass:

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

config = ImageGenerationConfig(
    model="stable_cascade_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=4.0,
    scheduler="DPMPP2M Karras",

    # Stage 2 decoder pass
    stage_2_steps=10,
    stage_2_cfg=1.0,
    stage_2_shift=1.0,
)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    images = client.generate_image(
        prompt="a field of sunflowers under a blue sky",
        config=config,
    )
    client.save_images(images, output_dir="output", prefix="cascade")
```

### Causal Inference

Causal inference is used with certain streaming/flow models:

```python
from drawthings_client import ImageGenerationConfig

config = ImageGenerationConfig(
    model="some_causal_model_q6p.ckpt",
    steps=20,
    width=1024,
    height=1024,
    cfg_scale=1.0,
    scheduler="Euler A",
    causal_inference_enabled=True,
    causal_inference=3,      # inference window
    causal_inference_pad=0,  # padding frames
)
```

---

## Model Discovery

### List All Models (CLI)

```bash
python examples/list_models.py
```

### List Models and LoRAs (CLI)

```bash
python examples/list_models.py --loras
```

### Connect to a Different Server

```bash
python examples/list_models.py --server your-server.local:7859
```

### Model Discovery (Python API)

```python
from model_metadata import ModelMetadata

metadata = ModelMetadata("server.example.com:7859")

# Get latent size for a specific model
info = metadata.get_latent_info("flux_1_dev_q6p.ckpt")
print(f"Latent size: {info['latent_size']}")   # 64
print(f"Version: {info['version']}")           # flux1

# Check if files exist on server
from drawthings_client import DrawThingsClient
with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    result = client.files_exist(["my_lora.ckpt", "missing_file.ckpt"])
    # {"my_lora.ckpt": True, "missing_file.ckpt": False}
```

---

## Tensor Encoding and Decoding

### Decoding a Response Tensor to PNG

```python
from tensor_decoder import save_tensor_image

# If you have raw tensor bytes from a gRPC response
with open("response.bin", "rb") as f:
    tensor_data = f.read()

save_tensor_image(tensor_data, "output.png")
```

### Encoding an Image for Input

```python
from tensor_encoder import encode_image_to_tensor

# Uncompressed float16 - required for edit/kontext models
tensor_bytes = encode_image_to_tensor("input.jpg", compress=False)

# fpzip-compressed float32 - works for standard img2img
tensor_bytes = encode_image_to_tensor("input.jpg", compress=True)
```

### Understanding the CCV Tensor Format

The Draw Things server uses the CCV NNC tensor format:

```
[68-byte header][pixel data]

Header (17 x uint32, little-endian):
  [0]  type/magic    0 = uncompressed float16
                     1012247 = fpzip compressed float32
  [1]  memory type   0x1 = CCV_TENSOR_CPU_MEMORY
  [2]  format        0x02 = CCV_TENSOR_FORMAT_NHWC
  [3]  datatype      0x20000 = CCV_16F (float16)
                     0x10000 = CCV_32F (float32)
  [4]  reserved      0
  [5]  batch         1
  [6]  height        H
  [7]  width         W
  [8]  channels      3 (RGB)
  [9-16] reserved    0

Pixel values:
  float_value = uint8_value / 127.5 - 1.0
  Range: -1.0 to +1.0
  Order: NHWC row-major (batch, height, width, channel)
```

This format matches the Swift reference client's `ImageHelpers.imageToDTTensor()` implementation from [euphoriacyberware-ai/DT-gRPC-Swift-Client](https://github.com/euphoriacyberware-ai/DT-gRPC-Swift-Client).

### Batch Generation

```python
import os
from drawthings_client import DrawThingsClient, ImageGenerationConfig

prompts = [
    "a red apple on a wooden table",
    "a blue butterfly on a flower",
    "a golden sunset over the ocean",
    "a snowy mountain peak",
]

config = ImageGenerationConfig(
    model="realdream_15sd15_q6p_q8p.ckpt",
    steps=20,
    width=512,
    height=512,
    cfg_scale=7.0,
    scheduler="UniPC ays",
)

os.makedirs("batch_output", exist_ok=True)

with DrawThingsClient("server.example.com:7859", insecure=False, verify_ssl=False) as client:
    for i, prompt in enumerate(prompts):
        config.seed = i   # deterministic seed per prompt
        images = client.generate_image(prompt=prompt, config=config)
        client.save_images(images, output_dir="batch_output", prefix=f"image_{i:03d}")
        print(f"Generated {i+1}/{len(prompts)}: {prompt}")
```

---

## Performance Tips

1. **Use distilled/turbo models for drafts:**
   - FLUX Schnell: 4 steps, cfg_scale=1.0
   - SDXL Turbo: 1 step, cfg_scale=0.0
   - Z-Image Turbo: 8 steps, cfg_scale=1.0
   - Lightning LoRAs: 4-8 steps

2. **Step count vs. quality:**
   - Draft/iteration: 4-8 steps
   - Good quality: 16-25 steps
   - Best quality: 30-50 steps

3. **CFG scale guidance:**
   - Distilled/turbo models: 1.0 or lower
   - Standard models: 5.0-8.0
   - FLUX: use `guidance_embed` (default 3.5), not `cfg_scale`
   - Higher CFG = closer prompt following; lower CFG = more creative

4. **Resolution guidelines:**
   - SD 1.5: 512x512 native
   - SDXL: 1024x1024 native
   - FLUX: 1024x1024 or larger
   - Non-native resolutions work but may reduce quality

5. **Enable TeaCache for FLUX:** ~20-30% speed improvement with minimal quality loss at `tea_cache_threshold=0.06`.

6. **Use tiled diffusion for images larger than 1024x1024** to avoid memory errors.

---

## Troubleshooting

### "Socket closed" errors

Some model/configuration combinations cause the server to crash. Try:
- A different model
- Removing LoRAs
- Reducing resolution or steps
- Checking server logs for specific errors

### Slow generation

- Check server logs for dimension errors
- Ensure GPU is in use (not CPU fallback)
- Start with smaller resolution to confirm correctness
- Use distilled/turbo models for iteration

### Wrong colors or corrupted output

- Confirm `fpzip` is installed: `pip install fpzip`
- Verify tensor decoding is working with a known-good output

### Image does not follow the prompt

- Increase `cfg_scale` (e.g., 7.0-9.0)
- Increase number of steps
- Add specific negative prompts
- For FLUX, increase `guidance_embed` (e.g., 4.0-5.0)

### Edit/kontext output ignores reference image

- Confirm `strength=1.0`
- Confirm input image dimensions match `width` and `height`
- Confirm the model is a kontext/edit model (e.g., `flux_klein`)
- Review server logs for "Input image encoded" signpost

### ControlNet has no effect

- Verify the ControlNet model filename is correct and the file exists on server
- Confirm `input_override` matches the type of guide image you are providing
- Try `weight=1.0` and `guidance_start=0.0, guidance_end=1.0` first, then tune
