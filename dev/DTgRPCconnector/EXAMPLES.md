# Draw Things gRPC Client - Examples

Comprehensive examples for using the Draw Things gRPC Python client.

## Table of Contents

- [Text-to-Image Generation](#text-to-image-generation)
- [Image Editing](#image-editing)
- [LoRA Models](#lora-models)
- [Model Discovery](#model-discovery)
- [Advanced Usage](#advanced-usage)

## Text-to-Image Generation

### Basic Generation

```bash
python examples/generate_image.py "a cute puppy playing in a garden"
```

### High-Quality Portrait

```bash
python examples/generate_image.py \
  "professional portrait photo, woman with flowing red hair, soft lighting, bokeh background, 85mm lens" \
  --steps 40 \
  --cfg 7.0 \
  --size 768 \
  --negative "cartoon, anime, painting, distorted, blurry" \
  --output portrait.png
```

### FLUX Schnell (Ultra-Fast)

```bash
python examples/generate_image.py \
  "a majestic dragon flying over a medieval castle" \
  --model flux_1_schnell_q8p.ckpt \
  --steps 4 \
  --cfg 1.0 \
  --size 1024 \
  --output dragon.png
```

### SDXL Generation

```bash
python examples/generate_image.py \
  "epic mountain landscape at sunset, dramatic clouds, golden hour lighting, photorealistic" \
  --model juggernaut_xl_v9_q6p_q8p.ckpt \
  --size 1024 \
  --steps 35 \
  --cfg 6.0 \
  --output landscape.png
```

### Z-Image Turbo (Fast Iteration)

```bash
python examples/generate_image.py \
  "concept art of a futuristic vehicle, sleek design, neon accents" \
  --model z_image_turbo_1.0_q6p.ckpt \
  --size 1024 \
  --steps 8 \
  --cfg 1.0 \
  --output concept.png
```

### Reproducible Results with Seed

```bash
python examples/generate_image.py \
  "cute cartoon cat wearing a wizard hat" \
  --seed 42 \
  --steps 25 \
  --output cat_wizard.png
```

## Image Editing

### Basic Edit

```bash
python examples/edit_image.py input.png "add warm sunset lighting" \
  --output edited.png \
  --steps 10
```

### Subtle Changes (Lower Strength)

```bash
# strength=0.5 for more subtle edits that preserve more of the original
python examples/edit_image.py photo.jpg "make it snowy" \
  --strength 0.5 \
  --steps 10 \
  --output snowy.png
```

### Dramatic Changes (Higher Strength)

```bash
# strength=1.0 for dramatic transformations
python examples/edit_image.py landscape.png "make it nighttime with stars" \
  --strength 1.0 \
  --steps 10 \
  --output night.png
```

### Using Different Edit Models

```bash
# Qwen Edit 2509 (stable)
python examples/edit_image.py input.png "add autumn colors" \
  --model qwen_image_edit_2509_q6p.ckpt \
  --no-lora \
  --steps 30 \
  --output autumn.png

# Qwen Edit 2511 (latest)
python examples/edit_image.py input.png "make it rainy" \
  --model qwen_image_edit_2511_q6p.ckpt \
  --steps 4 \
  --output rainy.png
```

## LoRA Models

### Text-to-Image with LoRA

```bash
# Qwen Edit with Lightning LoRA
python examples/generate_image.py "a mystical forest with glowing mushrooms" \
  --model qwen_image_edit_2511_q6p.ckpt \
  --lora qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt \
  --lora-weight 1.0 \
  --steps 4 \
  --cfg 1.0 \
  --output mystical_forest.png
```

### Adjusting LoRA Weight

```bash
# Lower weight (0.6) for subtle effect
python examples/generate_image.py "portrait of a warrior" \
  --model qwen_image_edit_2511_q6p.ckpt \
  --lora qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt \
  --lora-weight 0.6 \
  --steps 6 \
  --output warrior.png

# Full weight (1.0) for maximum effect
python examples/generate_image.py "cyberpunk cityscape" \
  --model qwen_image_edit_2511_q6p.ckpt \
  --lora qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt \
  --lora-weight 1.0 \
  --steps 4 \
  --output cyberpunk.png
```

## Model Discovery

### List All Models

```bash
python examples/list_models.py
```

### List Models and LoRAs

```bash
python examples/list_models.py --loras
```

### Connect to Different Server

```bash
python examples/list_models.py --server your-server.local:7859
```

## Advanced Usage

### Python API

```python
from examples.generate_image import generate_image

# Generate an image programmatically
result = generate_image(
    prompt="a majestic lion in the savanna",
    size=1024,
    steps=20,
    cfg_scale=7.0,
    negative_prompt="blurry, low quality",
    model="flux_1_dev_q8p.ckpt",
    output="lion.png",
    seed=12345,
    server="192.168.2.150:7859"
)

if result:
    print(f"Image saved to: {result}")
```

### Custom Configuration with Client

```python
from drawthings_client import DrawThingsClient
from model_metadata import get_model_metadata
import flatbuffers
import GenerationConfiguration

# Connect
client = DrawThingsClient("192.168.2.150:7859", insecure=False, verify_ssl=False)

# Get model metadata
model = "flux_1_schnell_q8p.ckpt"
metadata = get_model_metadata(server_address, model)
latent_size = metadata.get('latent_size', 64)

# Calculate scale
size = 1024
scale = size // latent_size

# Build configuration
builder = flatbuffers.Builder(2048)
model_offset = builder.CreateString(model)

GenerationConfiguration.Start(builder)
GenerationConfiguration.AddId(builder, 0)
GenerationConfiguration.AddStartWidth(builder, scale)
GenerationConfiguration.AddStartHeight(builder, scale)
GenerationConfiguration.AddSeed(builder, 42)
GenerationConfiguration.AddSteps(builder, 4)
GenerationConfiguration.AddGuidanceScale(builder, 1.0)
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddSampler(builder, 18)  # UniPC Trailing
GenerationConfiguration.AddBatchCount(builder, 1)
GenerationConfiguration.AddBatchSize(builder, 1)

config = GenerationConfiguration.End(builder)
builder.Finish(config)

# Create request
request = imageService_pb2.ImageGenerationRequest(
    prompt="your prompt here",
    configuration=bytes(builder.Output()),
    scaleFactor=1,
    user='PythonClient',
    device=imageService_pb2.LAPTOP,
    chunked=False
)

# Generate
for response in client.stub.GenerateImage(request):
    # Handle response...
    pass
```

### Batch Generation

```python
import os
from examples.generate_image import generate_image

prompts = [
    "a red apple on a wooden table",
    "a blue butterfly on a flower",
    "a golden sunset over the ocean",
    "a snowy mountain peak",
]

os.makedirs("batch_output", exist_ok=True)

for i, prompt in enumerate(prompts):
    output_path = f"batch_output/image_{i:03d}.png"
    print(f"Generating {i+1}/{len(prompts)}: {prompt}")
    
    generate_image(
        prompt=prompt,
        steps=20,
        size=768,
        output=output_path,
        seed=i  # Different seed for each
    )
```

### Tensor Decoding

```python
from tensor_decoder import save_tensor_image

# If you have raw tensor data
with open('response.bin', 'rb') as f:
    tensor_data = f.read()

# Decode and save
save_tensor_image(tensor_data, 'decoded.png')
```

### Tensor Encoding (for Image Editing)

```python
from tensor_encoder import encode_image_to_tensor

# Encode image to tensor format
tensor_bytes = encode_image_to_tensor('input.jpg', compress=True)

# Now you can use this in an ImageGenerationRequest
# (see examples/edit_image.py for full implementation)
```

## Performance Tips

1. **Use Turbo/Lightning models** for faster iteration:
   - FLUX Schnell: 4 steps
   - SDXL Turbo: 1 step  
   - Z-Image Turbo: 8 steps
   - Lightning LoRAs: 4-8 steps

2. **Adjust steps** based on quality needs:
   - Draft/iteration: 4-8 steps
   - Good quality: 16-25 steps
   - Best quality: 30-50 steps

3. **Use appropriate CFG scale**:
   - Turbo models: 1.0
   - Standard models: 5.0-8.0
   - For following prompt closely: Higher CFG (7.0-10.0)
   - For creativity: Lower CFG (3.0-5.0)

4. **Resolution guidelines**:
   - SD 1.5: 512x512 optimal
   - SDXL: 1024x1024 optimal
   - FLUX: 1024x1024 optimal
   - Higher resolution = longer generation time

## Troubleshooting

### "Socket closed" Errors

Some model combinations may cause the server to crash. Try:
- Using a different model
- Removing LoRA (`--no-lora`)
- Reducing resolution or steps

### Slow Generation

- Check server logs for errors
- Ensure GPU is being used (not CPU fallback)
- Try smaller resolution first
- Use turbo/lightning models

### Wrong Colors in Output

- Make sure fpzip is installed: `pip install fpzip`
- Check that tensor decoding is working correctly

### Image Doesn't Match Prompt

- Increase CFG scale (e.g., 7.0-8.0)
- Increase number of steps
- Add negative prompts to avoid unwanted elements
- Try a different model

## Contributing

Found a useful pattern? Submit a PR to add it to these examples!
