# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MuddleMeThis is a production-ready Gradio web application that connects vision-enabled LLMs (LM Studio, Ollama) with Draw Things gRPC server for image generation. It provides a comprehensive interface for AI-powered prompt manipulation (expansion, extraction, refinement) and high-quality image generation with advanced features like dual LoRA support, resolution-dependent shift calculation, and PWA installation.

**Version**: 1.0.0
**GitHub**: https://github.com/AlexTheStampede/MuddleMeThis

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
├── app.py                         # Main Gradio application (52KB, ~1150 lines)
├── settings_manager.py            # Settings persistence and management
├── convert_presets.py             # Utility to convert Draw Things presets
├── launch.sh / launch.bat         # Cross-platform launch scripts
├── setup.sh                       # Automated installation script
├── manifest.json                  # PWA configuration
├── MuddleMeThis.desktop.template  # Linux desktop integration template
├── requirements.txt               # Python dependencies
├── settings/
│   ├── config.json               # User settings (auto-created, gitignored)
│   ├── config.example.json       # Example configuration
│   ├── expand.txt                # System prompt for expansion
│   ├── extract.txt               # System prompt for extraction
│   ├── refine.txt                # System prompt for refinement
│   ├── stylecopy.txt             # System prompt for style analysis
│   ├── aspectratio.txt           # Aspect ratio definitions
│   ├── presets/                  # Model presets (JSON)
│   │   ├── flux_official.json
│   │   ├── schnell_official.json
│   │   ├── ponysdxl_official.json
│   │   ├── sd15_official.json
│   │   ├── qwenimage_official.json
│   │   ├── chroma_official.json
│   │   └── straysignal_chroma.json
│   └── negative_prompts/         # Negative prompt presets (JSON)
│       ├── realistic_quality.json
│       ├── anime_pony.json
│       ├── simple.json
│       ├── none.json
│       └── chroma.json
├── dev/
│   ├── DTgRPCconnector/          # Draw Things gRPC Python client
│   ├── ModuLLe/                  # LLM provider abstraction layer
│   └── *.txt                     # Official preset source files
├── outputs/                       # Generated images (auto-created, gitignored)
└── venv/                          # Virtual environment (gitignored)
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
  - **IMPORTANT**: Server always multiplies by 64, regardless of model latent_size!
  - Universal formula: `scale_factor = desired_pixels ÷ 64`
  - All models: 512px → scale=8, 1024px → scale=16, 1536px → scale=24
  - SDXL workaround: Use `÷ 64` even though latent_size is 128
- **Resolution-Dependent Shift**: Client-side calculation using official exponential formula
  - Formula: `shift = exp(((resolution_factor - 256) * (1.15 - 0.5) / (4096 - 256)) + 0.5)`
  - Where: `resolution_factor = (pixel_width * pixel_height) / 256`
  - This is a universal calculation independent of model latent size
  - Maps resolution to shift range 0.5-1.15 exactly as in official Draw Things app
  - When enabled, the calculated shift replaces the manual shift value (not multiplied)
  - Source: ModelZoo.swift:2358-2360 from official Draw Things app
- **High-Res Fix**: Two-pass generation for better quality at high resolutions
  - First pass: Generate at lower resolution (e.g., 512×512 for SD 1.5)
  - Upscale: Scale latent to target resolution
  - Second pass: Refine at target resolution with img2img (strength typically 0.7)
  - Parameters: `hiresFixStartWidth`, `hiresFixStartHeight` (in scale units), `hiresFixStrength`
  - UI shows pixels; internally converted to scale units (pixels ÷ 64)
- **Automatic Detection**: Client queries server metadata to determine correct latent_size for each model
- **CLIP Skip**: Critical parameter - SD 1.5 needs clip_skip=1, Pony/Illustrious need clip_skip=2
- **Dual LoRA Support**: Up to 2 LoRAs simultaneously with independent weight control (0.0-2.0)

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

## Implemented Features

### Core Functionality (✅ Complete)
1. **Prompt Expansion**: User inputs brief prompt, LLM expands it with rich details
2. **Prompt Extraction**: User uploads image, vision LLM generates matching prompt
3. **Bofonchio MC's Restyler**: User uploads image, vision LLM analyzes and describes the visual style in detail
4. **Prompt Refinement**: User requests modifications, LLM adjusts prompt intelligently
5. **Direct Mode**: User writes complete prompt, uses as-is for generation
6. **Edit Image**: Upload image and provide text instructions to edit it (works with Qwen Edit, Flux Kontext, etc.)

### UI Implementation (Gradio)
- Tabbed interface with Settings, Expand, Extract, Bofonchio MC's Restyler, Refine, Direct, Edit Image, and Generate sections
- Text areas with copy buttons and character counts
- Image upload with preview and drag-drop support
- Real-time progress tracking during generation
- Generated image display with metadata
- Searchable/filterable dropdowns for models and LoRAs
- Comprehensive settings panel with connection status indicators

### Advanced Features
- **Multi-Provider LLM**: Supports both LM Studio and Ollama with automatic model discovery
- **Separate Vision Models**: Independent text and vision model selection (supports Qwen3-VL, LLaVA, Llama Vision, etc.)
- **Dual LoRA Support**: Use up to 2 LoRAs simultaneously with adjustable weights
- **High-Res Fix**: Two-pass generation (512→1024 for SD 1.5) with configurable start resolution and refinement strength
- **Resolution Scale Multiplier**: Scale any aspect ratio by 0.5x-4x for flexible output sizes
- **Resolution-Dependent Shift**: Official exponential formula from Draw Things (ModelZoo.swift)
- **Negative Prompt Presets**: Quick-select common negative prompts (Realistic, Anime/Pony, Simple, None, Straysignal Chroma)
- **Official Presets**: 7 official/community presets (FLUX, Schnell, Pony, SDXL, SD1.5, Qwen, Chroma, Straysignal Chroma)
- **Auto-Save**: Images saved with descriptive filenames and comprehensive PNG metadata
- **Generation Timing**: Track how long each image takes to generate
- **PWA Support**: Install as standalone progressive web app
- **Cross-Platform**: Launch scripts for Linux, Mac, and Windows

### Settings Management
- Persistent JSON configuration (auto-created, gitignored)
- LLM and gRPC server addresses with connection testing
- Model and LoRA selection from server listings
- Custom aspect ratios and resolution presets
- Model-specific presets (steps, CFG, sampler, CLIP skip, shift)
- Editable system prompts in text files
- Settings survive restarts and sync across tabs

## Development Guidelines

### File Organization
- Keep ALL development/temporary files in `dev/` subdirectory
- Organize into logical subfolders as needed
- Main application code stays in root
- This keeps `.gitignore` simple and lean

### Working with gRPC
When implementing image generation:
1. Always query model metadata first to get correct latent_size
2. Calculate scale factors properly: `scale = pixels // 64` (universal formula)
3. Use appropriate clip_skip for the model (check model presets)
4. Handle fpzip tensor compression for efficiency
5. Verify server metadata response before generation
6. For edit models: Set `ImageGuidanceScale` (typically 1.5) and `OriginalImageWidth/Height` + `TargetImageWidth/Height` fields in the configuration

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

## Running the Application

### Quick Start
```bash
# Easy launch (recommended)
./launch.sh              # Linux/Mac
launch.bat               # Windows

# Or manually
python app.py
```

Access at: **http://localhost:7860**

### First-Time Setup
1. **Start LLM Server**:
   - LM Studio: Load a model and start server (port 1234)
   - Ollama: `ollama run qwen3-vl:4b-instruct`

2. **Start Draw Things**:
   - Enable gRPC server in settings (port 7859)
   - Load desired models

3. **Configure MuddleMeThis**:
   - Go to Settings tab
   - Select LLM provider and connect
   - Select text and vision models
   - Connect to gRPC server
   - Select generation model

## Updating the Application

MuddleMeThis includes built-in git-based auto-update functionality. This requires that you installed via `git clone` (the recommended installation method).

### Using the Built-In Updater

1. **Open the Settings Tab**:
   - Navigate to the Settings tab in the web interface
   - Click the "Updates" accordion to expand it

2. **Check for Updates**:
   - Click the "Check for Updates" button
   - The system will check GitHub for new commits
   - Status will show either:
     - "✅ Already up to date!" - You have the latest version
     - "✅ Updates available! New commits (X)" - Updates are ready to install

3. **Install Updates**:
   - If updates are available, click the "Update Now" button
   - The system will run `git pull` to fetch and apply updates
   - User settings (`settings/config.json`) are preserved (gitignored)
   - On success, you'll see: "✅ Update complete!"

4. **Restart the Application**:
   - Close the browser window
   - Stop the server (Ctrl+C in terminal)
   - Restart using `./launch.sh` or `python app.py`

### Manual Updates (Alternative)

If you prefer to update manually via command line:

```bash
# Pull latest changes
git pull origin main

# Update dependencies if needed
pip install -r requirements.txt

# Restart the application
./launch.sh
```

### Troubleshooting

**"Not installed via git clone" Error:**
- You downloaded the code as a ZIP instead of cloning it
- To enable auto-updates, reinstall using: `git clone https://github.com/AlexTheStampede/MuddleMeThis.git`

**"Local changes detected" Error:**
- You have uncommitted modifications to files
- Either:
  - Backup your changes and resolve conflicts
  - Or run manually: `git stash && git pull && git stash pop`

**Network/Connection Errors:**
- Check your internet connection
- Verify you can reach GitHub: `ping github.com`
- Try updating manually: `git pull origin main`

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
python examples/basic_usage.py
```

### Run MuddleMeThis
```bash
# Activate venv if not already active
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Launch application
python app.py
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
