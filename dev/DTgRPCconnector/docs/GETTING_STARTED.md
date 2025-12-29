# Getting Started with Draw Things Python gRPC Client

A quick-start guide to get you up and running in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Access to Draw Things server at `192.168.2.150:7859`

## Step 1: Install Dependencies (30 seconds)

```bash
cd /home/alexthestampede/Aish/gRPC
pip install -r requirements.txt
```

This installs:
- `grpcio` - gRPC library
- `grpcio-tools` - Protocol buffer tools
- `flatbuffers` - FlatBuffer library

## Step 2: Verify Server Connection (1 minute)

Create a simple test file or use Python interactive mode:

```python
from drawthings_client import DrawThingsClient

# Test connection
with DrawThingsClient("192.168.2.150:7859") as client:
    response = client.echo("Hello")
    print(f"Server: {response.message}")
    print(f"Server ID: {response.serverIdentifier}")
```

**Expected output:**
```
Server: Echo reply
Server ID: [some number]
```

If this works, you're connected! If not, check:
- Server is running
- Network connection
- Firewall settings

## Step 3: Generate Your First Image (2 minutes)

### Quick Method (One Line)

```python
from drawthings_client import quick_generate

# Generate and save image
quick_generate(
    server_address="192.168.2.150:7859",
    prompt="A beautiful sunset over mountains",
    output_path="my_first_image.png"
)

print("Image saved to: my_first_image.png")
```

### Standard Method (Full Control)

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

# Create configuration
config = ImageGenerationConfig(
    model="realDream",
    steps=16,
    width=512,
    height=512,
    cfg_scale=5.0,
    scheduler="UniPC ays"
)

# Generate image
with DrawThingsClient("192.168.2.150:7859") as client:
    images = client.generate_image(
        prompt="A serene Japanese garden with cherry blossoms",
        negative_prompt="blurry, low quality",
        config=config
    )

    # Save images
    saved_files = client.save_images(images, output_dir="output")
    print(f"Saved {len(saved_files)} images:")
    for filepath in saved_files:
        print(f"  - {filepath}")
```

## Step 4: Run the Test Suite (1 minute)

Run the comprehensive test suite to verify everything works:

```bash
python test_client.py
```

This will:
1. Test server connectivity
2. Generate test images
3. Save outputs to `./output/` directory

## Step 5: Explore Examples (Optional)

Run the example script to see different usage patterns:

```bash
python example_usage.py
```

This demonstrates:
- Simple generation
- Custom configuration
- Progress tracking
- Batch processing
- Different schedulers

## Common Use Cases

### Use Case 1: Simple Script

```python
#!/usr/bin/env python3
from drawthings_client import quick_generate

# Generate image
quick_generate(
    "192.168.2.150:7859",
    "A cyberpunk city at night",
    output_path="city.png"
)
```

### Use Case 2: With Progress Bar

```python
from drawthings_client import (
    DrawThingsClient,
    ImageGenerationConfig,
    StreamingProgressHandler
)

config = ImageGenerationConfig(
    model="realDream",
    steps=16,
    width=512,
    height=512,
    cfg_scale=5.0,
    scheduler="UniPC ays"
)

progress = StreamingProgressHandler(config.steps)

with DrawThingsClient("192.168.2.150:7859") as client:
    images = client.generate_image(
        prompt="A fantasy castle",
        config=config,
        progress_callback=progress.on_progress
    )
    progress.on_complete()

    client.save_images(images, output_dir="output")
```

### Use Case 3: Batch Processing

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

prompts = [
    "A dragon in flight",
    "A spaceship in orbit",
    "A magical forest"
]

config = ImageGenerationConfig(
    model="realDream",
    steps=16,
    width=512,
    height=512,
    cfg_scale=5.0,
    scheduler="UniPC ays"
)

with DrawThingsClient("192.168.2.150:7859") as client:
    for i, prompt in enumerate(prompts):
        print(f"Generating {i+1}/{len(prompts)}: {prompt}")

        images = client.generate_image(prompt, config)
        saved = client.save_images(images, prefix=f"batch_{i+1}")

        print(f"  Saved: {saved[0]}")
```

### Use Case 4: Custom Parameters

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

# High-quality, large image
config = ImageGenerationConfig(
    model="realDream",
    steps=30,          # More steps = better quality
    width=768,         # Larger size
    height=768,
    cfg_scale=7.0,     # Higher guidance
    scheduler="UniPC ays",
    seed=42            # Fixed seed for reproducibility
)

with DrawThingsClient("192.168.2.150:7859") as client:
    images = client.generate_image(
        prompt="A photorealistic portrait, highly detailed",
        negative_prompt="cartoon, anime, low quality, blurry",
        config=config
    )

    client.save_images(images, prefix="high_quality")
```

## Configuration Options

### Required Parameters

```python
ImageGenerationConfig(
    model="realDream",      # Model name
    steps=16,               # Generation steps (1-100)
    width=512,              # Image width (multiples of 64)
    height=512,             # Image height (multiples of 64)
    cfg_scale=5.0,          # Guidance scale (1.0-20.0)
    scheduler="UniPC ays"   # Scheduler type
)
```

### Optional Parameters

```python
ImageGenerationConfig(
    # ... required params ...
    seed=12345,             # Fixed seed (default: random)
    strength=0.75,          # Img2img strength (default: 0.75)
    batch_count=1,          # Number of batches (default: 1)
    batch_size=1            # Images per batch (default: 1)
)
```

### Available Schedulers

- `"UniPC ays"` ⭐ (Recommended - fast and high quality)
- `"Euler A"`
- `"DPMPP2M Karras"`
- `"DDIM"`
- `"PLMS"`
- `"LCM"` (for fast generation)
- And many more...

## Tips and Best Practices

### 1. Start Simple

Begin with the default parameters, then adjust as needed:

```python
config = ImageGenerationConfig(
    model="realDream",
    steps=16,
    width=512,
    height=512,
    cfg_scale=5.0,
    scheduler="UniPC ays"
)
```

### 2. Use Context Managers

Always use `with` statements to ensure proper cleanup:

```python
with DrawThingsClient("192.168.2.150:7859") as client:
    # Your code here
    pass
# Connection automatically closed
```

### 3. Handle Errors

Wrap generation in try-except blocks:

```python
try:
    images = client.generate_image(prompt, config)
except Exception as e:
    print(f"Error: {e}")
```

### 4. Save Incrementally

For batch processing, save images as you go:

```python
for i, prompt in enumerate(prompts):
    images = client.generate_image(prompt, config)
    client.save_images(images, prefix=f"img_{i}")
    # Each image saved immediately
```

### 5. Monitor Progress

Use progress callbacks for long-running operations:

```python
def on_progress(stage, step):
    print(f"{stage}: {step}")

images = client.generate_image(
    prompt,
    config,
    progress_callback=on_progress
)
```

## Troubleshooting

### Problem: Import Error

```
ModuleNotFoundError: No module named 'grpc'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: Connection Error

```
grpc._channel._InactiveRpcError: <_InactiveRpcError ...>
```

**Solutions:**
1. Verify server is running: `telnet 192.168.2.150 7859`
2. Check network connectivity
3. Verify server address and port
4. Check firewall settings

### Problem: Model Not Found

```
Model 'realDream' not found
```

**Solutions:**
1. Check available models:
   ```python
   response = client.echo("test")
   print(response.files)  # List of available files/models
   ```
2. Use exact model name (case-sensitive)
3. Ensure model is installed on server

### Problem: Generation Takes Too Long

**Solutions:**
1. Reduce steps: `steps=8` instead of `steps=16`
2. Use smaller resolution: `width=512, height=512`
3. Try faster scheduler: `scheduler="LCM"`

### Problem: Low Quality Images

**Solutions:**
1. Increase steps: `steps=30` or higher
2. Adjust CFG scale: `cfg_scale=7.0` or `8.0`
3. Use better prompt engineering
4. Try different schedulers

## Next Steps

Now that you're up and running:

1. **Read the documentation**
   - `CLIENT_README.md` - Complete API reference
   - `INTEGRATION_GUIDE.md` - Integration examples
   - `PROJECT_SUMMARY.md` - Project overview

2. **Explore examples**
   - `test_client.py` - Comprehensive test suite
   - `example_usage.py` - Various usage patterns

3. **Integrate into your project**
   - See `INTEGRATION_GUIDE.md` for Flask, FastAPI, Discord bot examples
   - Copy files to your project
   - Import and use the client

4. **Experiment**
   - Try different prompts
   - Adjust parameters
   - Test different schedulers
   - Build your own tools

## Quick Reference

### Import

```python
from drawthings_client import (
    DrawThingsClient,
    ImageGenerationConfig,
    StreamingProgressHandler,
    quick_generate
)
```

### Basic Generation

```python
config = ImageGenerationConfig(
    model="realDream",
    steps=16,
    width=512,
    height=512,
    cfg_scale=5.0,
    scheduler="UniPC ays"
)

with DrawThingsClient("192.168.2.150:7859") as client:
    images = client.generate_image("prompt", config)
    client.save_images(images)
```

### One-Liner

```python
quick_generate(
    "192.168.2.150:7859",
    "prompt",
    output_path="output.png"
)
```

## Help and Support

- **Documentation**: See `CLIENT_README.md`
- **Examples**: Run `python example_usage.py`
- **Integration**: See `INTEGRATION_GUIDE.md`
- **Project Info**: See `PROJECT_SUMMARY.md`

## Files Location

All files are in: `/home/alexthestampede/Aish/gRPC/`

Core files:
- `drawthings_client.py` - Main library
- `test_client.py` - Test suite
- `example_usage.py` - Examples
- `requirements.txt` - Dependencies

Documentation:
- `GETTING_STARTED.md` - This file
- `CLIENT_README.md` - Complete docs
- `INTEGRATION_GUIDE.md` - Integration examples
- `PROJECT_SUMMARY.md` - Project overview

---

**You're all set! Start generating amazing images with Draw Things!** 🎨
