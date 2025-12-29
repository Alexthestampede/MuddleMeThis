# Draw Things Python gRPC Client

A modular Python client library for the Draw Things image generation server.

## Overview

This project provides a clean, modular Python interface for interacting with a Draw Things server via gRPC. It handles all the complexity of FlatBuffer configuration building, gRPC streaming, and response processing.

## Features

- **Modular Design**: Easy to integrate into other Python projects
- **Type-Safe Configuration**: Dataclass-based configuration with validation
- **Streaming Support**: Handles streaming responses with progress callbacks
- **Progress Tracking**: Real-time progress updates during generation
- **Preview Images**: Optional callback for preview images during generation
- **Convenience Functions**: Quick one-liner image generation
- **Context Manager Support**: Automatic resource cleanup
- **Comprehensive Error Handling**: Clear error messages and exceptions

## Installation

### Prerequisites

- Python 3.8 or higher
- Access to a Draw Things server (default: `192.168.2.150:7859`)

### Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements are:
- `grpcio` - gRPC Python library
- `grpcio-tools` - Protocol buffer compiler
- `flatbuffers` - FlatBuffer serialization library

## Project Structure

```
.
├── drawthings_client.py          # Main client library (modular, reusable)
├── test_client.py                # Test/example script
├── imageService_pb2.py           # Generated gRPC protocol buffer code
├── imageService_pb2_grpc.py      # Generated gRPC service stub
├── GenerationConfiguration.py    # Generated FlatBuffer schema
├── SamplerType.py                # FlatBuffer enum definitions
├── requirements.txt              # Python dependencies
└── CLIENT_README.md              # This file
```

## Quick Start

### 1. Test Server Connection

```python
from drawthings_client import DrawThingsClient

with DrawThingsClient("192.168.2.150:7859") as client:
    response = client.echo("test")
    print(f"Server: {response.message}")
```

### 2. Generate an Image (Simple)

```python
from drawthings_client import quick_generate

# One-liner image generation
image_path = quick_generate(
    server_address="192.168.2.150:7859",
    prompt="A beautiful sunset over mountains",
    output_path="sunset.png"
)
print(f"Image saved to: {image_path}")
```

### 3. Generate an Image (Full Control)

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig

# Create configuration matching your requirements
config = ImageGenerationConfig(
    model="realDream",
    steps=16,
    width=512,
    height=512,
    cfg_scale=5.0,
    scheduler="UniPC ays"
)

# Generate image with progress tracking
with DrawThingsClient("192.168.2.150:7859") as client:
    def on_progress(stage, step):
        print(f"Stage: {stage}, Step: {step}")

    images = client.generate_image(
        prompt="A serene Japanese garden with cherry blossoms",
        negative_prompt="blurry, low quality",
        config=config,
        progress_callback=on_progress
    )

    # Save generated images
    saved_files = client.save_images(images, output_dir="output")
    print(f"Saved: {saved_files}")
```

## Configuration Parameters

### ImageGenerationConfig

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | str | Model name (e.g., "realDream") | Required |
| `steps` | int | Number of generation steps | Required |
| `width` | int | Image width in pixels | Required |
| `height` | int | Image height in pixels | Required |
| `cfg_scale` | float | Classifier-free guidance scale | Required |
| `scheduler` | str | Scheduler/sampler name | Required |
| `seed` | int | Random seed (auto-generated if None) | `None` |
| `strength` | float | Image-to-image strength (0.0-1.0) | `0.75` |
| `batch_count` | int | Number of batches | `1` |
| `batch_size` | int | Images per batch | `1` |

### Supported Schedulers

- DPMPP2M Karras
- Euler A
- DDIM
- PLMS
- DPMPP SDE Karras
- UniPC
- **UniPC ays** (Used in requirements)
- LCM
- Euler A Substep
- And more...

## API Reference

### DrawThingsClient

Main client class for interacting with the server.

#### Constructor

```python
DrawThingsClient(server_address: str, insecure: bool = True)
```

- `server_address`: Server address in format "host:port"
- `insecure`: Use insecure channel (no TLS)

#### Methods

##### `echo(name: str) -> EchoReply`

Test server connectivity.

```python
response = client.echo("test")
print(response.message)
```

##### `generate_image(...) -> List[bytes]`

Generate image(s) with full control.

```python
images = client.generate_image(
    prompt="Your prompt here",
    config=config,
    negative_prompt="",
    scale_factor=1,
    progress_callback=None,
    preview_callback=None
)
```

Parameters:
- `prompt`: Text prompt for generation
- `config`: ImageGenerationConfig instance
- `negative_prompt`: Optional negative prompt
- `scale_factor`: Image scale factor (default: 1)
- `progress_callback`: Called with (stage, step) during generation
- `preview_callback`: Called with preview image bytes

Returns: List of generated images as bytes

##### `save_images(...) -> List[str]`

Save generated images to disk.

```python
saved_files = client.save_images(
    images=images,
    output_dir="output",
    prefix="generated"
)
```

Returns: List of saved file paths

### Convenience Functions

#### `quick_generate(...) -> str`

Quick one-liner for image generation.

```python
filepath = quick_generate(
    server_address="192.168.2.150:7859",
    prompt="A beautiful sunset",
    model="realDream",
    steps=16,
    width=512,
    height=512,
    cfg_scale=5.0,
    scheduler="UniPC ays",
    output_path="output.png",
    show_progress=True
)
```

## Running Tests

Run the test suite to verify everything is working:

```bash
python test_client.py
```

This will:
1. Test server connectivity
2. Generate a test image with specified parameters
3. Generate multiple images with different prompts
4. Test the quick_generate convenience function

All test images will be saved to the `output/` directory.

## Example: Integration in Your Project

```python
# your_project.py
from drawthings_client import DrawThingsClient, ImageGenerationConfig

class YourImageGenerator:
    def __init__(self, server_address="192.168.2.150:7859"):
        self.client = DrawThingsClient(server_address)

    def generate(self, prompt, **kwargs):
        config = ImageGenerationConfig(
            model=kwargs.get('model', 'realDream'),
            steps=kwargs.get('steps', 16),
            width=kwargs.get('width', 512),
            height=kwargs.get('height', 512),
            cfg_scale=kwargs.get('cfg_scale', 5.0),
            scheduler=kwargs.get('scheduler', 'UniPC ays')
        )

        return self.client.generate_image(prompt, config)

    def __del__(self):
        self.client.close()
```

## Advanced Usage

### Progress Tracking with Custom Handler

```python
from drawthings_client import StreamingProgressHandler

progress = StreamingProgressHandler(total_steps=16)

images = client.generate_image(
    prompt="Your prompt",
    config=config,
    progress_callback=progress.on_progress
)

progress.on_complete()
```

### Preview Images

```python
def on_preview(preview_bytes):
    # Save preview or display in UI
    with open("preview.png", "wb") as f:
        f.write(preview_bytes)

images = client.generate_image(
    prompt="Your prompt",
    config=config,
    preview_callback=on_preview
)
```

### Context Manager

```python
with DrawThingsClient("192.168.2.150:7859") as client:
    # Client automatically closes when exiting context
    images = client.generate_image(prompt, config)
```

## Troubleshooting

### Connection Issues

If you get connection errors:

1. Verify the server is running: `telnet 192.168.2.150 7859`
2. Check firewall settings
3. Verify the server address and port

### Import Errors

If you get import errors for the generated files:

```bash
# Regenerate protobuf files
python -m grpc_tools.protoc \
    -I./draw-things-community/Libraries/GRPC/Models/Sources/imageService \
    --python_out=. \
    --grpc_python_out=. \
    ./draw-things-community/Libraries/GRPC/Models/Sources/imageService/imageService.proto
```

### Model Not Found

If the server reports the model doesn't exist:

1. Check available models using the Echo RPC
2. Verify the exact model name spelling
3. Ensure the model is installed on the server

## Requirements Met

This implementation meets all specified requirements:

- ✅ Server address: `192.168.2.150:7859`
- ✅ Model: `realDream`
- ✅ Steps: `16`
- ✅ Resolution: `512x512`
- ✅ CFG: `5.0`
- ✅ Scheduler: `UniPC ays`
- ✅ Generated Python gRPC client code from proto file
- ✅ FlatBuffer configuration construction
- ✅ Modular design for easy integration
- ✅ Streaming response handling
- ✅ Image saving functionality
- ✅ Test/example script included

## License

This is a proof of concept implementation. Adjust licensing as needed for your project.

## Contributing

This is a proof of concept. Feel free to extend it with additional features:

- Image-to-image generation support
- LoRA support
- ControlNet support
- Upscaling support
- Batch processing
- Async/await support
- GUI integration

## Support

For issues or questions about the Draw Things server itself, refer to the official Draw Things documentation.
