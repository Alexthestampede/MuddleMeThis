# Integration Guide: Using Draw Things Client in Your Project

This guide shows you how to integrate the Draw Things Python client library into your own projects.

## Installation

### Option 1: Copy Files Directly

Copy these files to your project:

```
your_project/
├── drawthings/
│   ├── __init__.py
│   ├── drawthings_client.py
│   ├── imageService_pb2.py
│   ├── imageService_pb2_grpc.py
│   ├── GenerationConfiguration.py
│   └── SamplerType.py
```

Create `__init__.py`:

```python
# drawthings/__init__.py
from .drawthings_client import (
    DrawThingsClient,
    ImageGenerationConfig,
    StreamingProgressHandler,
    quick_generate
)

__all__ = [
    'DrawThingsClient',
    'ImageGenerationConfig',
    'StreamingProgressHandler',
    'quick_generate'
]
```

### Option 2: Add to Python Path

Add the directory containing the client files to your Python path:

```python
import sys
sys.path.insert(0, '/path/to/drawthings/client')

from drawthings_client import DrawThingsClient, ImageGenerationConfig
```

## Basic Integration Examples

### 1. Simple Web Service Integration

```python
# app.py - Flask web service
from flask import Flask, request, jsonify, send_file
from drawthings_client import DrawThingsClient, ImageGenerationConfig
import io

app = Flask(__name__)
client = DrawThingsClient("192.168.2.150:7859")

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json

    config = ImageGenerationConfig(
        model=data.get('model', 'realDream'),
        steps=data.get('steps', 16),
        width=data.get('width', 512),
        height=data.get('height', 512),
        cfg_scale=data.get('cfg_scale', 5.0),
        scheduler=data.get('scheduler', 'UniPC ays')
    )

    images = client.generate_image(
        prompt=data['prompt'],
        negative_prompt=data.get('negative_prompt', ''),
        config=config
    )

    if images:
        return send_file(io.BytesIO(images[0]), mimetype='image/png')
    else:
        return jsonify({'error': 'Generation failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. Background Job Queue (Celery)

```python
# tasks.py
from celery import Celery
from drawthings_client import DrawThingsClient, ImageGenerationConfig

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def generate_image_task(prompt, config_dict, output_path):
    """Generate image as a background task."""
    config = ImageGenerationConfig(**config_dict)

    with DrawThingsClient("192.168.2.150:7859") as client:
        images = client.generate_image(prompt, config)

        if images:
            with open(output_path, 'wb') as f:
                f.write(images[0])
            return output_path
        else:
            raise Exception("No images generated")
```

### 3. Command-Line Tool

```python
#!/usr/bin/env python3
# generate.py - Simple CLI tool
import argparse
from drawthings_client import quick_generate

def main():
    parser = argparse.ArgumentParser(description='Generate images with Draw Things')
    parser.add_argument('prompt', help='Image prompt')
    parser.add_argument('--output', '-o', default='output.png', help='Output file')
    parser.add_argument('--steps', type=int, default=16, help='Generation steps')
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--cfg', type=float, default=5.0, help='CFG scale')
    parser.add_argument('--model', default='realDream', help='Model name')

    args = parser.parse_args()

    print(f"Generating: {args.prompt}")

    result = quick_generate(
        server_address="192.168.2.150:7859",
        prompt=args.prompt,
        model=args.model,
        steps=args.steps,
        width=args.width,
        height=args.height,
        cfg_scale=args.cfg,
        output_path=args.output
    )

    print(f"Saved to: {result}")

if __name__ == '__main__':
    main()
```

Usage:
```bash
./generate.py "A beautiful sunset" --output sunset.png --steps 20
```

### 4. Discord Bot Integration

```python
# bot.py - Discord bot
import discord
from discord.ext import commands
from drawthings_client import DrawThingsClient, ImageGenerationConfig
import io

bot = commands.Bot(command_prefix='!')
client = DrawThingsClient("192.168.2.150:7859")

@bot.command()
async def generate(ctx, *, prompt: str):
    """Generate an image from a prompt."""
    await ctx.send(f"Generating: {prompt}...")

    config = ImageGenerationConfig(
        model="realDream",
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays"
    )

    try:
        images = client.generate_image(prompt=prompt, config=config)

        if images:
            file = discord.File(io.BytesIO(images[0]), filename="generated.png")
            await ctx.send(file=file)
        else:
            await ctx.send("Failed to generate image")

    except Exception as e:
        await ctx.send(f"Error: {e}")

bot.run('YOUR_BOT_TOKEN')
```

### 5. Streamlit Web UI

```python
# app.py - Streamlit UI
import streamlit as st
from drawthings_client import DrawThingsClient, ImageGenerationConfig
from PIL import Image
import io

st.title("Draw Things Image Generator")

# Sidebar configuration
st.sidebar.header("Configuration")
model = st.sidebar.selectbox("Model", ["realDream", "other_model"])
steps = st.sidebar.slider("Steps", 1, 50, 16)
width = st.sidebar.selectbox("Width", [512, 768, 1024], index=0)
height = st.sidebar.selectbox("Height", [512, 768, 1024], index=0)
cfg_scale = st.sidebar.slider("CFG Scale", 1.0, 20.0, 5.0)
scheduler = st.sidebar.selectbox("Scheduler", ["UniPC ays", "Euler A", "DDIM"])

# Main interface
prompt = st.text_area("Prompt", placeholder="Describe the image you want to generate...")
negative_prompt = st.text_area("Negative Prompt", placeholder="What to avoid...")

if st.button("Generate"):
    if not prompt:
        st.error("Please enter a prompt")
    else:
        with st.spinner("Generating image..."):
            config = ImageGenerationConfig(
                model=model,
                steps=steps,
                width=width,
                height=height,
                cfg_scale=cfg_scale,
                scheduler=scheduler
            )

            client = DrawThingsClient("192.168.2.150:7859")

            try:
                images = client.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    config=config
                )

                if images:
                    image = Image.open(io.BytesIO(images[0]))
                    st.image(image, caption="Generated Image")

                    # Download button
                    st.download_button(
                        "Download Image",
                        data=images[0],
                        file_name="generated.png",
                        mime="image/png"
                    )
                else:
                    st.error("No image generated")

            except Exception as e:
                st.error(f"Error: {e}")

            finally:
                client.close()
```

Run with:
```bash
streamlit run app.py
```

### 6. FastAPI Async Service

```python
# main.py - FastAPI service
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
from drawthings_client import DrawThingsClient, ImageGenerationConfig
import base64

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model: str = "realDream"
    steps: int = 16
    width: int = 512
    height: int = 512
    cfg_scale: float = 5.0
    scheduler: str = "UniPC ays"

def generate_sync(request: GenerationRequest):
    """Synchronous generation function to run in thread pool."""
    config = ImageGenerationConfig(
        model=request.model,
        steps=request.steps,
        width=request.width,
        height=request.height,
        cfg_scale=request.cfg_scale,
        scheduler=request.scheduler
    )

    with DrawThingsClient("192.168.2.150:7859") as client:
        images = client.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            config=config
        )
        return images[0] if images else None

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """Generate image endpoint."""
    loop = asyncio.get_event_loop()

    try:
        image_data = await loop.run_in_executor(executor, generate_sync, request)

        if image_data:
            # Return base64 encoded image
            return {
                "success": True,
                "image": base64.b64encode(image_data).decode('utf-8')
            }
        else:
            raise HTTPException(status_code=500, detail="Generation failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        client = DrawThingsClient("192.168.2.150:7859")
        response = client.echo("health")
        client.close()
        return {"status": "healthy", "server_id": response.serverIdentifier}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Advanced Integration Patterns

### Connection Pool

For high-traffic applications, reuse connections:

```python
# connection_pool.py
from queue import Queue
from drawthings_client import DrawThingsClient

class ClientPool:
    def __init__(self, server_address, pool_size=5):
        self.pool = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            self.pool.put(DrawThingsClient(server_address))

    def get_client(self):
        return self.pool.get()

    def return_client(self, client):
        self.pool.put(client)

# Usage
pool = ClientPool("192.168.2.150:7859", pool_size=10)

def generate(prompt, config):
    client = pool.get_client()
    try:
        return client.generate_image(prompt, config)
    finally:
        pool.return_client(client)
```

### Async Wrapper

For async frameworks:

```python
# async_client.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from drawthings_client import DrawThingsClient, ImageGenerationConfig

class AsyncDrawThingsClient:
    def __init__(self, server_address, max_workers=4):
        self.server_address = server_address
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def generate_image(self, prompt, config, **kwargs):
        loop = asyncio.get_event_loop()

        def _generate():
            with DrawThingsClient(self.server_address) as client:
                return client.generate_image(prompt, config, **kwargs)

        return await loop.run_in_executor(self.executor, _generate)

# Usage
async def main():
    client = AsyncDrawThingsClient("192.168.2.150:7859")
    config = ImageGenerationConfig(
        model="realDream",
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays"
    )

    images = await client.generate_image("A beautiful sunset", config)
```

## Error Handling

Always handle errors appropriately:

```python
from drawthings_client import DrawThingsClient, ImageGenerationConfig
import grpc

try:
    with DrawThingsClient("192.168.2.150:7859") as client:
        config = ImageGenerationConfig(
            model="realDream",
            steps=16,
            width=512,
            height=512,
            cfg_scale=5.0,
            scheduler="UniPC ays"
        )

        images = client.generate_image("test", config)

except grpc.RpcError as e:
    print(f"gRPC Error: {e.code()}")
    print(f"Details: {e.details()}")
    # Handle specific error codes
    if e.code() == grpc.StatusCode.UNAVAILABLE:
        print("Server is unavailable")
    elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        print("Request timed out")

except Exception as e:
    print(f"General error: {e}")
```

## Configuration Management

Use environment variables or config files:

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    server_address: str = os.getenv("DRAWTHINGS_SERVER", "192.168.2.150:7859")
    default_model: str = os.getenv("DRAWTHINGS_MODEL", "realDream")
    default_steps: int = int(os.getenv("DRAWTHINGS_STEPS", "16"))
    default_width: int = int(os.getenv("DRAWTHINGS_WIDTH", "512"))
    default_height: int = int(os.getenv("DRAWTHINGS_HEIGHT", "512"))

config = AppConfig()
```

## Testing

Example unit tests:

```python
# test_integration.py
import unittest
from drawthings_client import DrawThingsClient, ImageGenerationConfig

class TestDrawThingsClient(unittest.TestCase):
    def setUp(self):
        self.client = DrawThingsClient("192.168.2.150:7859")

    def tearDown(self):
        self.client.close()

    def test_echo(self):
        response = self.client.echo("test")
        self.assertIsNotNone(response.message)

    def test_image_generation(self):
        config = ImageGenerationConfig(
            model="realDream",
            steps=1,  # Minimal steps for testing
            width=64,
            height=64,
            cfg_scale=5.0,
            scheduler="UniPC ays"
        )

        images = self.client.generate_image("test", config)
        self.assertGreater(len(images), 0)

if __name__ == '__main__':
    unittest.test()
```

## Best Practices

1. **Use context managers** - Always use `with` statements to ensure proper cleanup
2. **Handle errors gracefully** - Network issues, timeouts, and server errors can occur
3. **Reuse connections** - Create connection pools for high-traffic applications
4. **Monitor progress** - Use callbacks for long-running operations
5. **Validate inputs** - Check prompts and configuration before sending requests
6. **Log appropriately** - Log errors and important events for debugging
7. **Test thoroughly** - Test connection handling, timeouts, and error cases

## Support

For more examples and documentation, see:
- `CLIENT_README.md` - Main documentation
- `test_client.py` - Comprehensive test suite
- `example_usage.py` - Usage examples
