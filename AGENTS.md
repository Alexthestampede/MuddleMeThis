# AGENTS.md - MuddleMeThis

Gradio app connecting vision LLMs (LM Studio/Ollama) to Draw Things gRPC for AI prompt engineering and image generation.

## Setup (Non-obvious)

```bash
# 1. Main deps
pip install -r requirements.txt

# 2. DTgRPCconnector (gRPC client for Draw Things) - MUST install separately
pip install -r dev/DTgRPCconnector/requirements.txt
# Python 3.13+ aarch64: pip install "flatbuffers>=24.3.0"  # override for piwheels

# 3. ModuLLe (LLM abstraction) - MUST install as editable
cd dev/ModuLLe && pip install -e . && cd ../..
```

**Why**: `dev/` contains git submodules not auto-installed. Both must be present or app fails at runtime.

## Running

```bash
./launch.sh        # Preferred - auto-detects and activates venv
python app.py      # Direct (requires manual venv activation)
```

Access: http://localhost:7860

## Entry Points & Structure

```
app.py                 # Main Gradio app (~2500 lines)
settings_manager.py     # Config persistence (JSON)
dev/
├── DTgRPCconnector/  # gRPC client for Draw Things image generation
│   ├── drawthings_client.py
│   └── requirements.txt
└── ModuLLe/            # LLM provider abstraction (Ollama/LM Studio)
    └── pyproject.toml
settings/
├── config.json         # User settings (gitignored, auto-created)
├── config.example.json # Reference/copy to create config.json
├── presets/            # Model presets (JSON)
└── prompts/            # System prompts (*.txt)
outputs/                # Generated images (auto-created)
```

## Critical Domain Knowledge

### gRPC Scale Factors (NOT pixels)
Draw Things gRPC uses scale factors, not pixel dimensions.
- Formula: `scale = desired_pixels ÷ 64`
- Examples: 512px → scale=8, 1024px → scale=16
- File: `dev/DTgRPCconnector/drawthings_client.py`

### Resolution-Dependent Shift
Uses official Draw Things exponential formula for quality scaling:
```python
shift = exp(((resolution_factor - 256) * 0.65 / 3840) + 0.5)
# where resolution_factor = (width * height) / 256
```

### Model-Specific Settings
- SD 1.5: clip_skip=1, base_res=512
- Pony/SDXL: clip_skip=2, base_res=1024
- FLUX: clip_skip=1, shift=1.0

## Testing

```bash
# DTgRPCconnector (requires real Draw Things server)
cd dev/DTgRPCconnector
python examples/list_models.py --server 192.168.2.150:7859
python examples/generate_image.py "test" --server 192.168.2.150:7859

# ModuLLe
cd dev/ModuLLe
python tests/test_import.py
python examples/basic_usage.py
```

## Config & Prompts

- **Config**: Edit `settings/config.json` (auto-saved by app). Reference: `settings/config.example.json`
- **System Prompts**: Edit `settings/prompts/*.txt` (expand.txt, extract.txt, refine.txt)

## Pre-commit Check

```bash
python3 -m py_compile app.py settings_manager.py
find . -name "*.py" -exec python3 -m py_compile {} \;
```

## Resources

- README.md: Full documentation
- CLAUDE.md: Detailed architecture notes
- dev/DTgRPCconnector/CONTRIBUTING.md: gRPC client patterns
