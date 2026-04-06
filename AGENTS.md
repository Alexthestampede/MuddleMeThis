# AGENTS.md - Development Guide for MuddleMeThis

## Project Overview
MuddleMeThis is a Gradio web application connecting vision-enabled LLMs (LM Studio, Ollama) with Draw Things gRPC server for AI-powered prompt manipulation and image generation.

**Version**: 1.0.0 | **GitHub**: https://github.com/AlexTheStampede/MuddleMeThis

## Build & Setup Commands

### Initial Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install main dependencies
pip install -r requirements.txt

# Install DTgRPCconnector (Draw Things gRPC client)
pip install -r dev/DTgRPCconnector/requirements.txt

# Install ModuLLe (LLM provider abstraction)
cd dev/ModuLLe && pip install -e . && cd ../..
# For all cloud providers: pip install -e ".[all]"
```

### Running the Application
```bash
# Quick launch (recommended)
./launch.sh              # Linux/Mac
launch.bat               # Windows

# Manual launch
python app.py
```

### Testing
```bash
# Test DTgRPCconnector
cd dev/DTgRPCconnector
python examples/list_models.py --server 192.168.2.150:7859
python examples/generate_image.py "test prompt" --server 192.168.2.150:7859

# Run specific tests
python tests/test_simple.py
python tests/test_client.py
python tests/test_scalefactor.py

# Test ModuLLe
cd dev/ModuLLe
python tests/test_import.py
python examples/basic_usage.py
```

### Syntax Validation
```bash
# Check Python syntax
python3 -m py_compile app.py
python3 -m py_compile settings_manager.py

# Validate all Python files
find . -name "*.py" -exec python3 -m py_compile {} \;
```

### Updating the Application
```bash
# Built-in git updater (preserves settings)
git pull origin main

# Or use the Settings tab → Updates section in the UI
```

## Code Style Guidelines

### Imports (as seen in app.py, settings_manager.py)
```python
# Order: Standard library → Third-party → Local imports
#!/usr/bin/env python3
"""Module docstring describing purpose"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import gradio as gr
from PIL import Image
import grpc
import flatbuffers

from drawthings_client import DrawThingsClient
from modulle import create_ai_client
```

### Type Hints
- Use `typing` module for complex types: `Optional[T]`, `Tuple[A, B]`, `List[T]`, `Dict[K, V]`
- Always annotate function parameters and return types
- Example: `def init_llm(...) -> Tuple[str, gr.Dropdown, gr.Dropdown]:`

### Naming Conventions
- **files**: `snake_case.py` (e.g., `settings_manager.py`, `drawthings_client.py`)
- **functions**: `snake_case` (e.g., `load_config()`, `generate_image()`)
- **classes**: `PascalCase` (e.g., `AppState`, `SettingsManager`, `DrawThingsClient`)
- **constants**: `UPPER_CASE` (e.g., `APP_VERSION`, `GRPC_AVAILABLE`)
- **variables**: `snake_case` descriptive names

### Docstrings
```python
def function_name(param1: type, param2: type) -> ReturnType:
    """One-line description

    Optional extended description with details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Example:
        result = function_name(value1, value2)
    """
```

### Error Handling
```python
# Use try/except with specific exceptions
try:
    import modulle
    MODULLE_AVAILABLE = True
except ImportError:
    MODULLE_AVAILABLE = False

# Validate inputs and return user-friendly error messages
if not server_url:
    return "❌ Server URL is required", gr.update(choices=[])

# Fail gracefully with informative messages
def safe_operation():
    try:
        # risky operation
        pass
    except SpecificError as e:
        logger.error(f"Operation failed: {e}")
        return None
```

### Data Classes & Configuration
```python
from dataclasses import dataclass, field

@dataclass
class ImageGenerationConfig:
    """Configuration for image generation"""
    model: str
    steps: int = 16
    width: int = 512
    height: int = 512
    cfg_scale: float = 7.0
    scheduler: str = "UniPC ays"
    lora_configs: List[LoRAConfig] = field(default_factory=list)
```

### Comments
- Use `#` for inline comments explaining WHY, not WHAT
- Use `# ====` section dividers in large files
- Comment complex logic, especially mathematical formulas

### Line Length & Formatting
- Maximum 100 characters per line (per pyproject.toml)
- Use 4 spaces for indentation
- Use Black-compatible formatting
- Group related imports together

### File Organization
```
MuddleMeThis/
├── app.py              # Main application (keep < 2500 lines)
├── settings_manager.py # Settings persistence
├── dev/                # ALL development/submodule code
│   ├── DTgRPCconnector/
│   └── ModuLLe/
├── settings/           # User config & prompts (auto-created)
└── outputs/            # Generated images (auto-created)
```

## Key Architectural Patterns

### 1. Client Initialization Pattern
```python
# Always check availability before use
if not MODULLE_AVAILABLE:
    return "❌ ModuLLe not installed", gr.update()

# Initialize clients with error handling
try:
    client, text_proc, vision_proc = create_ai_client(...)
except Exception as e:
    return f"❌ Connection failed: {e}", gr.update()
```

### 2. Settings Persistence
- Use `SettingsManager` class for all config operations
- Settings auto-save to `settings/config.json` (gitignored)
- Load system prompts from `settings/*.txt`

### 3. Gradio UI Updates
```python
# Use gr.update() for dynamic UI changes
return "Success", gr.update(value="new_text"), gr.update(visible=True)
# Use gr.Dropdown for component references
return "Error", gr.Dropdown(choices=[]), gr.update()
```

### 4. Progress Tracking
```python
# Yield progress tuples for long operations
for i, item in enumerate(items):
    progress = (i + 1) / len(items)
    yield intermediate_result, f"Progress: {progress:.0%}"
```

## Critical Domain Knowledge

### gRPC Scale Factors
- `start_width`/`start_height` are scale factors, NOT pixels
- Formula: `scale_factor = desired_pixels ÷ 64` (universal)
- Examples: 512px → scale=8, 1024px → scale=16

### Resolution-Dependent Shift
- Uses official exponential formula from Draw Things
- `shift = exp(((resolution_factor - 256) * 0.65 / 3840) + 0.5)`
- Where `resolution_factor = (width * height) / 256`

### Model-Specific Settings
- SD 1.5: clip_skip=1, base_res=512
- Pony/SDXL: clip_skip=2, base_res=1024
- Always check model preset for correct values

## Testing Guidelines

1. **Test with real servers** when possible:
   - LM Studio: `192.168.2.20:1234`
   - Draw Things: `192.168.2.150:7859`

2. **Unit tests** should be isolated and fast

3. **Integration tests** should verify end-to-end flows

4. **Always test** both success and error paths

## Before Committing

```bash
# 1. Validate syntax
python3 -m py_compile app.py settings_manager.py

# 2. Run relevant tests
cd dev/DTgRPCconnector && python tests/test_simple.py
cd dev/ModuLLe && python tests/test_import.py

# 3. Check for trailing whitespace
grep -r '[[:space:]]$' --include="*.py" .

# 4. Verify no secrets in code
git diff --check
```

## Helpful Resources
- CLAUDE.md: Detailed project documentation
- dev/DTgRPCconnector/CONTRIBUTING.md: gRPC client guidelines
- dev/ModuLLe/README.md: ModuLLe usage guide
- settings/config.example.json: Configuration reference
