# 🎨 MuddleMeThis

AI-powered prompt engineering and image generation workbench. Connect vision-enabled LLMs (LM Studio, Ollama) with Draw Things gRPC server for intelligent prompt manipulation and high-quality image generation.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## ✨ Features

### 🤖 AI-Powered Prompt Engineering
- **Expand**: Transform brief ideas into detailed prompts
- **Extract**: Analyze images and generate matching prompts using vision models
- **Refine**: Modify prompts with natural language instructions
- **Direct Mode**: Write prompts manually with full control

### 🖼️ Advanced Image Generation
- **Draw Things Integration**: Connect to Draw Things gRPC server
- **Multiple Models**: FLUX, SDXL, SD 1.5, Pony, Z-Image, Qwen, and more
- **Dual LoRA Support**: Use up to 2 LoRAs simultaneously with adjustable weights (0.0-2.0)
- **High-Res Fix**: Two-pass generation for superior quality (e.g., 512→1024 for SD 1.5)
- **Resolution Scale**: Multiply any aspect ratio by 0.5x-4x for flexible output sizes
- **Smart Shift**: Official exponential formula from Draw Things for optimal quality
- **Negative Prompt Presets**: Quick-select common negative prompts (Realistic, Anime/Pony, Simple, None)
- **Official Presets**: 6 pre-configured presets from Draw Things (FLUX, Schnell, Pony, SDXL, SD1.5, Qwen, Chroma)
- **Real-time Progress**: Live step tracking during generation
- **Auto-save**: Descriptive filenames with full PNG metadata embedding

### 🔧 Professional Features
- **LLM Flexibility**: Choose between LM Studio and Ollama
- **Vision Models**: Separate text and vision model selection (supports Qwen3-VL, LLaVA, Llama 3.2 Vision, etc.)
- **Searchable UI**: Filter models and LoRAs instantly with built-in search
- **PWA Support**: Install as a standalone progressive web app
- **Cross-platform**: Easy launch scripts for Linux, Mac, and Windows
- **Generation Time Tracking**: See how long each image took to generate
- **Rich Metadata**: All generation parameters embedded in saved PNG files

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Draw Things** with gRPC enabled ([Download](https://drawthings.ai/))
- **One of the following:**
  - [LM Studio](https://lmstudio.ai/) - Easy local LLM server with GUI
  - [Ollama](https://ollama.com/) - Lightweight LLM runtime for terminal

### Installation

```bash
# Clone the repository
git clone https://github.com/Alexthestampede/MuddleMeThis.git
cd MuddleMeThis

# Run setup script (recommended)
chmod +x setup.sh
./setup.sh
```

**Or manual installation:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r dev/DTgRPCconnector/requirements.txt
cd dev/ModuLLe && pip install -e . && cd ../..
```

### Launch

**Linux/Mac:**
```bash
./launch.sh
```

**Windows:**
```cmd
launch.bat
```

**Traditional:**
```bash
python app.py
```

**Access the app:** Open browser to **http://localhost:7860**

## 📖 Setup Guide

### Step 1: Configure LLM (for Prompt Engineering)

#### Option A: LM Studio
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download models:
   - **Text Model**: Any model (Mistral, Llama, etc.)
   - **Vision Model**: [Qwen3-VL-4B](https://lmstudio.ai/models/qwen/qwen3-vl-4b) (recommended)
3. Load the vision model in LM Studio
4. Start server (default port: 1234)

#### Option B: Ollama (Recommended for Vision)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install models
ollama pull mistral                    # Text model (for expand/refine)
ollama pull qwen3-vl:4b-instruct      # Vision model (for extract)
```

**Alternative vision models:**
- `qwen3-vl:2b-instruct` - Faster, lighter (2B params)
- `llava` - Original LLaVA vision model
- `llama3.2-vision:11b` - Latest Meta vision model

#### In MuddleMeThis:
1. Go to **Settings** tab
2. Select provider (LM Studio or Ollama)
3. Enter server URL:
   - LM Studio: `http://localhost:1234`
   - Ollama: `http://localhost:11434`
4. Click **"Connect to LLM Server"**
5. Select **Text Model** from dropdown
6. Select **Vision Model** from dropdown (or leave empty to use text model)

### Step 2: Configure Draw Things (for Image Generation)

1. Download [Draw Things](https://drawthings.ai/) (Mac/iOS)
2. In Draw Things Settings:
   - Enable gRPC server
   - Note the server address (usually `localhost:7859`)
3. Download models in Draw Things (FLUX, SDXL, Pony, etc.)

#### In MuddleMeThis:
1. Go to **Settings** → **gRPC Settings**
2. Enter server address (e.g., `localhost:7859`)
3. Click **"Connect to gRPC Server"**
4. Select your model from the dropdown

### Step 3: Start Creating!

You're ready! Try:
- 📝 **Expand Prompt**: "a peaceful garden" → detailed description
- 🖼️ **Extract from Image**: Upload an image → get its prompt
- ✏️ **Refine Prompt**: "add sunset lighting" → modified prompt
- 🎨 **Generate**: Create images with your perfected prompts

## 🎯 Usage Examples

### Expand a Prompt
```
Input: "a peaceful garden"

Output: "A serene Japanese zen garden at golden hour, with carefully
raked gravel patterns in concentric circles, ancient moss-covered
stones placed with intention, a small koi pond reflecting vibrant
cherry blossoms, soft diffused sunlight filtering through bamboo
groves, creating dappled shadows on the pristine white gravel..."
```

### Extract Prompt from Image
1. Go to **"Extract from Image"** tab
2. Upload a reference image
3. Click **"Extract Prompt"**
4. AI analyzes the image and generates a detailed prompt
5. Click **"Send to Refine"** to modify, or use directly for generation

### Refine a Prompt
```
Current Prompt: "a cat sitting on a chair"
Refinement: "make it a cyberpunk setting with neon lights"

Output: "A sleek cat with glowing cyan eyes sitting on a chrome
chair in a dark cyberpunk alley, surrounded by vibrant neon signs
in Japanese characters, holographic advertisements flickering in
the rain, purple and pink light reflecting off wet pavement..."
```

### Generate Images
1. **Select Model**: Choose from available models (FLUX, SDXL, Pony, etc.)
2. **Choose Preset**: Official presets or "Custom"
3. **Set Parameters**:
   - Aspect ratio
   - Steps, CFG scale
   - LoRAs (up to 2)
   - CLIP Skip
   - Advanced settings (shift, seed mode)
4. Click **"Generate Image"**
5. Images auto-save to `outputs/` with descriptive filenames and metadata

**Example filename:**
```
20250129_154230_FLUX.1_[dev]_peaceful_garden_sunset_s1234567890.png
```

## 📁 Project Structure

```
MuddleMeThis/
├── app.py                      # Main Gradio application
├── settings_manager.py         # Configuration management
├── convert_presets.py          # Preset conversion utility
│
├── launch.sh                   # Linux/Mac launcher
├── launch.bat                  # Windows launcher
├── MuddleMeThis.desktop.template  # Linux desktop file template
│
├── requirements.txt            # Python dependencies
├── setup.sh                    # Automated setup
│
├── settings/
│   ├── aspect_ratios/          # Resolution presets (512, 1024, 2048)
│   ├── presets/                # Model presets (JSON)
│   │   ├── flux_official.json
│   │   ├── schnell_official.json
│   │   ├── ponysdxl_official.json
│   │   └── ...
│   └── prompts/                # System prompts for LLM
│       ├── expand.txt
│       ├── extract.txt
│       └── refine.txt
│
├── dev/
│   ├── ModuLLe/                # LLM integration library
│   └── DTgRPCconnector/        # Draw Things gRPC client
│
└── outputs/                    # Generated images (auto-created)
```

## ⚙️ Configuration

### System Prompts
Customize LLM behavior by editing files in `settings/prompts/`:
- **expand.txt** - Instructions for prompt expansion
- **extract.txt** - Instructions for image analysis
- **refine.txt** - Instructions for prompt refinement

### Model Presets
Official presets from Draw Things included:
- **flux_official.json** - FLUX Dev (28 steps, CFG 4.5)
- **schnell_official.json** - FLUX Schnell (4 steps, fast)
- **ponysdxl_official.json** - Pony Diffusion (CLIP Skip 2)
- **sd15_official.json** - Stable Diffusion 1.5
- **qwenimage_official.json** - Qwen Image
- **chroma_official.json** - Chroma

Add custom presets in `settings/presets/` (JSON format).

### Aspect Ratios
Customize available aspect ratios in `settings/aspect_ratios/`:
- `1024.json` - For FLUX, SDXL, modern models
- `512.json` - For SD 1.5
- `2048.json` - For high-resolution generation

## 🖼️ Image Metadata

Every generated PNG includes comprehensive metadata:
```
- prompt                    (full positive prompt)
- negative_prompt           (negative prompt)
- model                     (display name)
- model_file                (actual filename)
- lora1, lora2              (with weights, if used)
- steps, cfg_scale, sampler
- resolution                (e.g., "1024x768")
- seed                      (actual seed used)
- shift                     (adjusted shift value)
- clip_skip
- generation_time           (in seconds)
- created_with             ("MuddleMeThis")
```

**View metadata:**
```bash
exiftool outputs/20250129_154230_*.png
```

## 🔧 Advanced Features

### Resolution-Dependent Shift
Automatically adjusts shift parameter based on resolution for optimal quality across different sizes.

**Formula:** `adjusted_shift = base_shift × (1.2 + 2.0 × pixel_ratio)`

Where `pixel_ratio = (target_width × target_height) / (base_resolution²)`

This ensures consistent quality when generating at non-standard resolutions.

### Dual LoRA Support
Use up to 2 LoRAs simultaneously:
- Independent weight control (0.0 - 2.0)
- Select "None" to disable
- Weights >1.0 for stronger effects

### PWA Installation
Install MuddleMeThis as a standalone app:
1. Open http://localhost:7860 in browser
2. Click install icon or menu → "Install MuddleMeThis"
3. Launch like a native app (desktop shortcut)
4. Works offline after first load

### Desktop Integration (Linux)
```bash
# Customize the template
cp MuddleMeThis.desktop.template MuddleMeThis.desktop
nano MuddleMeThis.desktop  # Edit paths

# Install
cp MuddleMeThis.desktop ~/.local/share/applications/
chmod +x ~/.local/share/applications/MuddleMeThis.desktop
```

## 🛠️ Troubleshooting

### LLM Connection Issues
- **LM Studio**: Ensure server is started and model is loaded
- **Ollama**: Check `ollama list` to see installed models
- Verify server URL and port are correct

### Vision Model Not Found
- **Ollama**: Install with `ollama pull qwen3-vl:4b-instruct`
- **LM Studio**: Download vision-capable model from built-in search
- Select correct vision model in Vision Model dropdown

### gRPC Connection Failed
- Ensure Draw Things is running
- Check gRPC is enabled in Draw Things settings
- Try `localhost:7859` or your device's IP address
- Verify firewall isn't blocking port 7859

### Image Generation Errors
- Check model is loaded in Draw Things
- Verify resolution matches model's capabilities
- Try lower resolution if out of memory
- Check advanced settings (shift, CLIP skip)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional LLM providers (OpenAI, Anthropic, Google, Cohere)
- More image generation backends (ComfyUI, Automatic1111)
- Additional preset collections
- UI/UX enhancements
- Documentation improvements
- Internationalization

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Credits

- **Draw Things** - Powerful local image generation
- **ModuLLe** - Flexible LLM integration framework
- **Gradio** - Beautiful web interface framework
- **Qwen Team** - Excellent vision models
- **Ollama** - Easy local LLM runtime

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/Alexthestampede/MuddleMeThis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Alexthestampede/MuddleMeThis/discussions)
- **Wiki**: [Documentation](https://github.com/Alexthestampede/MuddleMeThis/wiki)

---

**Made with ❤️ for the AI art community**

*Star ⭐ this repo if you find it useful!*
