# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MuddleMeThis is a generative AI application that connects vision-enabled LLMs with image generation via Draw Things gRPC. It provides an interface for prompt manipulation (expansion, extraction, refinement) and uses the resulting prompts to generate images.

**GitHub**: https://github.com/Alexthestampede/MuddleMeThis

## Development Environment

### Test Configuration
- **LLM Server**: LM Studio at `192.168.2.20:1234` with model `qwen3-vl-8b-instruct-abliterated-v2.0`
- **gRPC Server**: Draw Things at `192.168.2.150:7859`
- **Virtual Environment**: Use `venv` for dependency isolation

### Setup Commands
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install DTgRPCconnector dependencies
pip install -r dev/DTgRPCconnector/requirements.txt

# Install ModuLLe
cd dev/ModuLLe
pip install -e .           # Basic install
pip install -e ".[all]"    # Install with all cloud providers
cd ../..
```

## Project Structure

```
MuddleMeThis/
├── dev/                           # Development code and libraries
│   ├── DTgRPCconnector/          # Draw Things gRPC Python client
│   ├── ModuLLe/                  # LLM provider abstraction layer
│   └── plans.txt                 # Project planning document
├── venv/                          # Virtual environment (gitignored)
└── [main application TBD]         # Future UI and core logic
```

### dev/DTgRPCconnector - Draw Things gRPC Client

A Python client library for Draw Things gRPC server enabling programmatic image generation.

**Key Files**:
- `drawthings_client.py` - Main gRPC client implementation
- `model_metadata.py` - Automatic model discovery and latent size detection
- `tensor_encoder.py` / `tensor_decoder.py` - Image tensor encoding/decoding with fpzip compression
- `GenerationConfiguration.py` - FlatBuffer configuration builder
- `LoRA.py` - FlatBuffer LoRA support

**Critical Concepts**:
- **Scale Factors**: `start_width` and `start_height` are NOT pixel dimensions, they are scale factors
  - Formula: `scale_factor = desired_pixels ÷ latent_size`
  - SD 1.5 (latent_size=64): 512px → scale=8, 768px → scale=12, 1024px → scale=16
  - SDXL (latent_size=128): 1024px → scale=8, 1536px → scale=12
- **Automatic Detection**: Client queries server metadata to determine correct latent_size for each model
- **Clip Skip**: Critical parameter - some models require clip_skip=1, others need clip_skip=2

**Example Usage**:
```bash
# List available models
python dev/DTgRPCconnector/examples/list_models.py --server 192.168.2.150:7859

# Generate image
python dev/DTgRPCconnector/examples/generate_image.py "a cute puppy" \
  --server 192.168.2.150:7859 \
  --size 512 \
  --steps 16 \
  --output output.png

# Generate with LoRA
python dev/DTgRPCconnector/examples/generate_image.py "zen garden" \
  --model qwen_image_edit_2511_q6p.ckpt \
  --lora qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt \
  --lora-weight 1.0 \
  --steps 4
```

### dev/ModuLLe - LLM Provider Abstraction

A modular abstraction layer providing unified interface across multiple AI providers (Ollama, LM Studio, OpenAI, Gemini, Claude).

**Design Philosophy**:
- Generic abstraction - no domain-specific logic
- Three core methods: `generate()`, `chat()`, `analyze_image()`
- Application logic built through prompt engineering

**Key Files**:
- `base.py` - Abstract base classes (BaseAIClient, BaseTextProcessor, BaseVisionProcessor)
- `factory.py` - `create_ai_client()` factory function
- `config.py` - Provider configuration management
- `providers/` - Individual provider implementations (5 providers)

**Setup Commands**:
```bash
# Interactive configuration wizard
modulle-config

# Inspect Ollama models
modulle-inspect-ollama http://192.168.2.20:1234
```

**Example Usage**:
```python
from modulle import create_ai_client

# Create client for LM Studio
_, text_processor, vision_processor = create_ai_client(
    provider='lm_studio',
    base_url='http://192.168.2.20:1234',
    text_model='qwen3-vl-8b-instruct-abliterated-v2.0'
)

# Generate text
result = text_processor.generate(
    prompt="Expand this prompt: a peaceful garden",
    system_prompt="You are a prompt expansion assistant",
    temperature=0.7
)

# Analyze image
description = vision_processor.analyze_image(
    image_data=base64_image,
    prompt="Write a detailed prompt that would generate this image"
)
```

## Planned Features

### Core Functionality
1. **Prompt Expansion**: User inputs brief prompt, LLM expands it with details
2. **Prompt Extraction**: User uploads image, LLM generates matching prompt
3. **Prompt Refinement**: User requests modifications ("change hair to red"), LLM adjusts prompt
4. **Direct Mode**: User writes complete prompt, uses as-is

### UI Components (TBD)
- Prompt input text area
- Image file selector with preview
- Task selection (expansion/extraction/refinement/direct)
- LLM output display area
- Generated image display area
- Settings panel for LLM and gRPC configuration
- Generate button to trigger image creation

### Settings Management
- LLM server address (default: localhost for simplicity)
- gRPC server address (default: localhost)
- Available LLM model list
- Available gRPC model + LoRA lists
- gRPC model presets (different models need different clip_skip, sampler settings, etc.)
- System prompts stored as editable text files

## Development Guidelines

### File Organization
- Keep ALL development/temporary files in `dev/` subdirectory
- Organize into logical subfolders as needed
- Main application code stays in root
- This keeps `.gitignore` simple and lean

### Working with gRPC
When implementing image generation:
1. Always query model metadata first to get correct latent_size
2. Calculate scale factors properly: `scale = pixels // latent_size`
3. Use appropriate clip_skip for the model (check model presets)
4. Handle fpzip tensor compression for efficiency
5. Verify server metadata response before generation

### Working with ModuLLe
When implementing LLM features:
1. Use generic `generate()` method with carefully crafted prompts
2. Don't add domain-specific methods to ModuLLe - keep it generic
3. Build application logic in main application code
4. Store system prompts as external text files for easy editing
5. Consider using `chat()` for multi-turn interactions (refinement mode)

### Testing
Always test against the configured servers:
- LM Studio: `192.168.2.20:1234` with `qwen3-vl-8b-instruct-abliterated-v2.0`
- Draw Things gRPC: `192.168.2.150:7859`

## Common Tasks

### Test DTgRPCconnector
```bash
cd dev/DTgRPCconnector
python examples/list_models.py --server 192.168.2.150:7859
python examples/generate_image.py "test prompt" --server 192.168.2.150:7859
```

### Test ModuLLe
```bash
cd dev/ModuLLe
python test_basic.py
python examples/basic_usage.py
```

### Add New gRPC Model Preset
Create/edit text file with model-specific settings:
- Model name
- Recommended clip_skip value
- Recommended sampler
- Optimal step count
- CFG scale range

### Add New System Prompt
Create text file in `settings/` folder with:
- Clear description of prompt purpose
- Example usage context
- The app automatically loads from expand.txt, extract.txt, refine.txt

### Add New Model Preset
Create JSON file in `settings/presets/` folder:
```json
{
  "name": "Model Name",
  "description": "Description",
  "base_resolution": 1024,
  "clip_skip": 2,
  "recommended_steps": 20,
  "recommended_cfg": 7.0,
  "sampler": "DPM++ 2M Karras",
  "notes": "Additional notes"
}
```

## Settings System

User configuration is stored in `settings/config.json` (gitignored, auto-created on first run):
- LLM and gRPC server addresses
- Last used models and LoRAs
- Default generation parameters
- User preferences

The app loads:
- System prompts from `settings/*.txt`
- Aspect ratios from `settings/aspectratio.txt`
- Model presets from `settings/presets/*.json`
- User config from `settings/config.json`
