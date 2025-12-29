#!/usr/bin/env python3
"""
MuddleMeThis - AI-Powered Prompt Engineering & Image Generation

A Gradio-based application that connects vision-enabled LLMs with Draw Things gRPC
for prompt manipulation (expansion, extraction, refinement) and image generation.
"""

import gradio as gr
import base64
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import io
import sys
import os
import time
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

# Suppress gRPC SSL handshake warnings (these are harmless when using self-signed certs)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# Add dev modules to path
dev_path = Path(__file__).parent / "dev"
sys.path.insert(0, str(dev_path / "ModuLLe"))
sys.path.insert(0, str(dev_path / "DTgRPCconnector"))

# Import settings manager
from settings_manager import settings

# Import our libraries
try:
    from modulle import create_ai_client
    MODULLE_AVAILABLE = True
except ImportError:
    print("Warning: ModuLLe not installed. Install with: pip install -e dev/ModuLLe")
    MODULLE_AVAILABLE = False

try:
    import flatbuffers
    import random as random_module
    import GenerationConfiguration
    import imageService_pb2
    import LoRA
    from drawthings_client import DrawThingsClient
    from model_metadata import ModelMetadata
    from tensor_decoder import tensor_to_pil
    GRPC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DTgRPCconnector not installed. Install requirements from dev/DTgRPCconnector/requirements.txt")
    print(f"Error: {e}")
    GRPC_AVAILABLE = False


# ============================================================================
# Configuration & State
# ============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.llm_client = None
        self.text_processor = None
        self.vision_processor = None
        self.grpc_client = None
        self.grpc_metadata = None
        self.current_prompt = ""
        self.available_models = []
        self.available_loras = []
        self.current_model_base_resolution = 1024  # Default
        self.model_name_to_file = {}  # Maps display names → filenames
        self.lora_name_to_file = {}   # Maps display names → filenames

state = AppState()


# ============================================================================
# LLM Processing Functions
# ============================================================================

def init_llm(server_url: str, model_name: str, vision_model_name: str, provider: str = 'lm_studio') -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """Initialize LLM connection and fetch available models"""
    try:
        if not MODULLE_AVAILABLE:
            return "❌ ModuLLe not installed", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

        # Map provider names to ModuLLe provider strings
        provider_map = {
            'LM Studio': 'lm_studio',
            'Ollama': 'ollama'
        }
        provider_key = provider_map.get(provider, 'lm_studio')

        # First, create a temporary client to list models
        temp_client, _, _ = create_ai_client(
            provider=provider_key,
            base_url=server_url,
            text_model=model_name or 'placeholder',
            vision_model=vision_model_name or None
        )

        # Try to get available models
        available_models = []
        try:
            available_models = temp_client.list_models()
        except:
            pass  # If listing fails, continue anyway

        # Now create the actual processors with the selected models
        if model_name:
            settings.update_config(
                llm_server=server_url,
                llm_model=model_name,
                llm_vision_model=vision_model_name,
                llm_provider=provider
            )
            state.llm_client, state.text_processor, state.vision_processor = create_ai_client(
                provider=provider_key,
                base_url=server_url,
                text_model=model_name,
                vision_model=vision_model_name or model_name  # Use text model if no vision model specified
            )
            vision_info = f" (Vision: {vision_model_name or model_name})" if vision_model_name else ""
            status = f"✅ Connected to {provider}: {server_url}\nText Model: {model_name}{vision_info}"
        else:
            state.llm_client = temp_client
            status = f"✅ Connected to {provider}: {server_url}\n\nAvailable models: {len(available_models)}\nPlease select models below."

        if available_models:
            status += f"\n\nFound {len(available_models)} model(s)"

        models_dropdown = gr.Dropdown(choices=available_models, value=model_name if model_name else None)
        vision_dropdown = gr.Dropdown(choices=available_models, value=vision_model_name if vision_model_name else None)
        return status, models_dropdown, vision_dropdown
    except Exception as e:
        return f"❌ LLM connection failed: {str(e)}", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])


def expand_prompt(user_prompt: str) -> str:
    """Expand a brief prompt into detailed description"""
    if not state.text_processor:
        return "❌ LLM not initialized. Configure in Settings tab first."

    if not user_prompt.strip():
        return "❌ Please enter a prompt to expand"

    try:
        system_prompt = settings.load_system_prompt('expand')

        result = state.text_processor.generate(
            prompt=user_prompt,
            system_prompt=system_prompt if system_prompt else None
        )

        if result:
            state.current_prompt = result
            return result
        return "❌ Failed to generate expansion"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def extract_prompt(image) -> str:
    """Extract prompt from uploaded image"""
    if not state.vision_processor:
        return "❌ Vision processor not initialized. Configure in Settings tab first."

    if image is None:
        return "❌ Please upload an image first"

    try:
        # Convert image to base64
        buffered = io.BytesIO()
        Image.fromarray(image).save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Load the user's detailed system prompt from extract.txt
        # This contains all the instructions for how to analyze and generate prompts
        extraction_instructions = settings.load_system_prompt('extract')

        # Use the full instructions as the prompt for vision analysis
        # Vision models work best with instructions integrated into the prompt
        result = state.vision_processor.analyze_image(
            image_data=img_base64,
            prompt=extraction_instructions if extraction_instructions else "Analyze this image and write a detailed prompt that could generate a similar image."
        )

        if result:
            state.current_prompt = result
            return result
        return "❌ Failed to extract prompt"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def refine_prompt(current_prompt: str, refinement_instruction: str) -> str:
    """Refine existing prompt based on user instruction"""
    if not state.text_processor:
        return "❌ LLM not initialized. Configure in Settings tab first."

    if not current_prompt.strip():
        return "❌ No prompt to refine. Generate or enter a prompt first."

    if not refinement_instruction.strip():
        return "❌ Please provide refinement instructions"

    try:
        system_prompt = settings.load_system_prompt('refine')

        result = state.text_processor.generate(
            prompt=f"Current prompt: {current_prompt}\n\nModification requested: {refinement_instruction}\n\nProvide the refined prompt:",
            system_prompt=system_prompt if system_prompt else None
        )

        if result:
            state.current_prompt = result
            return result
        return "❌ Failed to refine prompt"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# gRPC Functions
# ============================================================================

def init_grpc(server_url: str) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """Initialize gRPC connection and fetch models/LoRAs"""
    try:
        if not GRPC_AVAILABLE:
            return "❌ DTgRPCconnector not installed", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

        # Update settings
        settings.update_config(grpc_server=server_url)

        # Use Draw Things root CA certificate for SSL validation
        # This allows connection to self-hosted servers with self-signed certificates
        root_ca_path = Path(__file__).parent / "dev" / "DTgRPCconnector" / "root_ca.crt"

        if root_ca_path.exists():
            # Use SSL with the root CA certificate
            state.grpc_client = DrawThingsClient(
                server_address=server_url,
                insecure=False,
                verify_ssl=False,
                ssl_cert_path=str(root_ca_path)
            )
        else:
            # Fallback to insecure connection
            print(f"Warning: Root CA certificate not found at {root_ca_path}")
            print("Attempting insecure connection...")
            state.grpc_client = DrawThingsClient(server_address=server_url, insecure=True)

        # Use Echo request to get structured metadata
        echo_request = imageService_pb2.EchoRequest(name='list_files')
        response = state.grpc_client.stub.Echo(echo_request)

        models = []
        loras = []

        # Use server's structured metadata (properly categorized)
        if response.HasField('override'):
            import json
            try:
                models_data = json.loads(response.override.models) if response.override.models else []
                loras_data = json.loads(response.override.loras) if response.override.loras else []

                # Extract both display names and filenames, create mappings
                # Models: use 'name' field for display, 'file' for actual loading
                model_display_names = []
                state.model_name_to_file = {}
                for m in models_data:
                    if m.get('file'):
                        # Use 'name' field if available, otherwise use filename
                        display_name = m.get('name', m['file'])
                        file_name = m['file']
                        model_display_names.append(display_name)
                        state.model_name_to_file[display_name] = file_name

                # LoRAs: same approach
                lora_display_names = []
                state.lora_name_to_file = {}
                for l in loras_data:
                    if l.get('file'):
                        display_name = l.get('name', l['file'])
                        file_name = l['file']
                        lora_display_names.append(display_name)
                        state.lora_name_to_file[display_name] = file_name

                # Sort alphabetically by display name for better UX
                models = sorted(model_display_names)
                loras = sorted(lora_display_names)

                # Create ModelMetadata and pre-populate its cache with the metadata we already fetched
                # This avoids SSL errors when fetching metadata later during generation
                state.grpc_metadata = ModelMetadata(server_url)
                state.grpc_metadata._models_cache = models_data
                state.grpc_metadata._loras_cache = loras_data
            except json.JSONDecodeError:
                print("Warning: Failed to parse model metadata, falling back to file list")
                models = []
                loras = []
                state.model_name_to_file = {}
                state.lora_name_to_file = {}
                state.grpc_metadata = ModelMetadata(server_url)

        # Fallback: use simple file list if metadata not available
        if not models and response.files:
            for filename in response.files:
                lower = filename.lower()
                if ('.ckpt' in lower or '.safetensors' in lower) and 'lora' not in lower:
                    models.append(filename)
                    # No name mapping for fallback mode
                    state.model_name_to_file[filename] = filename

        state.available_models = models
        state.available_loras = loras

        model_list = "\n".join([f"  • {m}" for m in models[:10]])  # Show first 10
        lora_info = f"\nAvailable LoRAs: {len(loras)}" if loras else ""

        status = f"✅ Connected to gRPC: {server_url}\n\nAvailable models: {len(models)}\n{model_list}{lora_info}"

        # Add "None" option to LoRAs for optional selection
        lora_choices = ["None"] + loras

        # Return updated dropdowns - they'll be used to update both settings and generation sections
        return status, gr.Dropdown(choices=models, value=models[0] if models else None), gr.Dropdown(choices=lora_choices, value="None")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ gRPC connection failed: {str(e)}\n\nDetails:\n{error_details}", gr.Dropdown(choices=[]), gr.Dropdown(choices=[])


def on_model_selected(model_name: str) -> Tuple[gr.Dropdown, str, gr.Dropdown]:
    """When a model is selected, get base resolution and update aspect ratios"""
    if not model_name or not state.grpc_metadata:
        return gr.Dropdown(choices=[]), "", gr.Dropdown(choices=[])

    # Translate display name to actual filename
    model_file = state.model_name_to_file.get(model_name, model_name)

    # Update settings (save the display name for UX)
    settings.update_config(last_used_model=model_name)

    # Get model metadata to determine base resolution (use actual filename)
    try:
        model_info = state.grpc_metadata.get_latent_info(model_file)
        latent_size = model_info.get('latent_size', 128)
        version = model_info.get('version', 'sdxl')

        # Determine base resolution from version (not latent_size!)
        # FLUX/Z-Image/Qwen/SD3 use 64-latent but 1024px output
        # SDXL uses 128-latent with 1024px output
        # SD 1.5/2.x use 64-latent with 512px output
        if version in ['flux1', 'z_image', 'qwen_image', 'sd3', 'sd3_large', 'sdxl', 'sdxl_base_v0.9']:
            base_resolution = 1024
        elif version in ['v1', 'v2']:
            base_resolution = 512
        else:
            # Fallback based on latent size
            base_resolution = 512 if latent_size == 64 else 1024

        state.current_model_base_resolution = base_resolution
    except:
        base_resolution = 1024
        state.current_model_base_resolution = 1024

    # Load aspect ratios for this resolution
    aspect_ratios = settings.load_aspect_ratios(base_resolution)
    aspect_choices = [label for label, _, _ in aspect_ratios]

    # Get available presets for dropdown
    all_presets = settings.load_model_presets()
    preset_choices = ["Custom (no preset)"] + list(all_presets.keys())

    info = f"Model loaded: {model_name}\nBase resolution: {base_resolution}px"

    return (
        gr.Dropdown(choices=preset_choices, value="Custom (no preset)"),
        info,
        gr.Dropdown(choices=aspect_choices, value=aspect_choices[4] if len(aspect_choices) > 4 else aspect_choices[0])
    )


# Sampler mapping from SamplerType.py
SAMPLERS = {
    "DPM++ 2M Karras": 0,
    "Euler A": 1,
    "DDIM": 2,
    "PLMS": 3,
    "DPM++ SDE Karras": 4,
    "UniPC": 5,
    "LCM": 6,
    "Euler A Substep": 7,
    "DPM++ SDE Substep": 8,
    "TCD": 9,
    "Euler A Trailing": 10,
    "DPM++ SDE Trailing": 11,
    "DPM++ 2M AYS": 12,
    "Euler A AYS": 13,
    "DPM++ SDE AYS": 14,
    "DPM++ 2M Trailing": 15,
    "DDIM Trailing": 16,
    "UniPC Trailing": 17,
    "UniPC AYS": 18,
}

SAMPLER_NAMES = list(SAMPLERS.keys())
SAMPLER_DEFAULT = "DPM++ 2M Karras"  # Index 0, universally supported


def on_preset_selected(preset_name: str) -> Tuple[int, float, str, str, float, bool, int, bool, bool, int]:
    """When a preset is selected, apply its settings"""
    if preset_name == "Custom (no preset)" or not preset_name:
        return (gr.update(), gr.update(), "Using custom settings",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    preset = settings.get_model_preset(preset_name)
    if preset:
        steps = preset.get('steps', preset.get('recommended_steps', 16))
        cfg = preset.get('guidanceScale', preset.get('recommended_cfg', 7.0))

        # Get sampler (handle both ID and name formats)
        sampler = preset.get('sampler', 0)
        if isinstance(sampler, str):
            # Preset uses sampler name directly
            sampler_name = sampler if sampler in SAMPLERS else SAMPLER_DEFAULT
        else:
            # Preset uses sampler ID - look up the name
            sampler_name = next((name for name, id in SAMPLERS.items() if id == sampler), SAMPLER_DEFAULT)

        # Advanced settings (shift is Float32 with default 1.0)
        shift = float(preset.get('shift', 1.0))
        res_shift = preset.get('resolutionDependentShift', False)
        seed_mode = preset.get('seedMode', 2)
        cfg_zero = preset.get('cfgZeroStar', False)
        hires_fix = preset.get('hiresFix', False)
        clip_skip = preset.get('clip_skip', 1)  # Pony needs 2, most others need 1

        notes = preset.get('notes', '')
        info = f"✅ Preset applied: {preset.get('name', 'Unknown')}\n{notes}"

        return steps, cfg, info, sampler_name, shift, res_shift, seed_mode, cfg_zero, hires_fix, clip_skip
    else:
        return (gr.update(), gr.update(), f"⚠️ Preset not found: {preset_name}",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())


def generate_image(prompt: str, model: str, lora1: str, lora1_weight: float, lora2: str, lora2_weight: float,
                  steps: int, cfg_scale: float, sampler_name: str, aspect_ratio: str,
                  seed: int, negative_prompt: str,
                  shift: float, res_dependent_shift: bool, seed_mode: int,
                  cfg_zero_star: bool, hires_fix: bool, clip_skip: int, progress=gr.Progress()):
    """Generate image using Draw Things gRPC with progress tracking"""
    if not state.grpc_client:
        return None, "❌ gRPC not initialized. Configure in Settings tab first."

    if not prompt:
        return None, "❌ No prompt provided"

    if not model:
        return None, "❌ No model selected"

    # Translate display names to actual filenames for gRPC
    model_file = state.model_name_to_file.get(model, model)
    lora1_file = state.lora_name_to_file.get(lora1, lora1) if lora1 and lora1.strip() and lora1 != "None" else None
    lora2_file = state.lora_name_to_file.get(lora2, lora2) if lora2 and lora2.strip() and lora2 != "None" else None

    # Start timing
    start_time = time.time()

    try:
        # Get model metadata FIRST to determine latent size and base resolution (use filename)
        try:
            model_info = state.grpc_metadata.get_latent_info(model_file)
            latent_size = model_info.get('latent_size', 128)
            version = model_info.get('version', 'sdxl')

            print(f"\n🔍 Model Metadata for {model} → {model_file}:")
            print(f"   Version: {version}")
            print(f"   Latent Size: {latent_size}")
            print(f"   Default Scale: {model_info.get('default_scale', 'N/A')}")

            # Determine base resolution from version (not latent_size!)
            # FLUX/Z-Image/Qwen/SD3 use 64-latent but 1024px output
            # SDXL uses 128-latent with 1024px output
            # SD 1.5/2.x use 64-latent with 512px output
            if version in ['flux1', 'z_image', 'qwen_image', 'sd3', 'sd3_large', 'sdxl', 'sdxl_base_v0.9']:
                base_resolution = 1024
            elif version in ['v1', 'v2']:
                base_resolution = 512
            else:
                # Fallback based on latent size
                base_resolution = 512 if latent_size == 64 else 1024

            print(f"   → Base Resolution: {base_resolution}")
        except Exception as e:
            # Fallback: assume SDXL
            print(f"\n⚠️  Failed to get model metadata: {e}")
            latent_size = 128
            base_resolution = 1024

        # Parse aspect ratio to get width and height using correct base resolution
        # Format: "ratio widthxheight"
        aspect_ratios = settings.load_aspect_ratios(base_resolution)
        width, height = base_resolution, base_resolution  # Default to square

        for label, w, h in aspect_ratios:
            if label == aspect_ratio:
                width, height = w, h
                break

        print(f"\n📐 Aspect Ratio: {aspect_ratio}")
        print(f"   → Pixel Dimensions: {width}x{height}")
        print(f"   → Latent Size: {latent_size}")

        # Calculate resolution-dependent shift (client-side calculation)
        # The server doesn't have AddResolutionDependentShift - we calculate it ourselves
        final_shift = float(shift)
        if res_dependent_shift:
            # Formula derived from Draw Things official app behavior
            # Formula: shift * (1.2 + 2.0 * pixel_ratio)
            # Verified against official values with average error of 0.12
            base_pixels = base_resolution * base_resolution
            target_pixels = width * height
            pixel_ratio = target_pixels / base_pixels
            final_shift = shift * (1.2 + 2.0 * pixel_ratio)
            print(f"\n⚙️  Resolution-Dependent Shift Calculation:")
            print(f"   Base: {base_resolution}x{base_resolution} = {base_pixels:,} pixels")
            print(f"   Target: {width}x{height} = {target_pixels:,} pixels")
            print(f"   Ratio: {pixel_ratio:.3f}")
            print(f"   Original Shift: {shift}")
            print(f"   → Adjusted Shift: {final_shift:.2f}")
        else:
            print(f"\n⚙️  Shift: {final_shift} (no resolution adjustment)")

        # Calculate scale factors
        # Testing: Server seems to multiply by 64 regardless of model latent_size
        # So for SDXL (latent=128) to get 1024px, we need to send scale=16 (not 8)
        # Because server does: 16 * 64 = 1024
        if latent_size == 128:
            # SDXL: double the scale to compensate for server using 64x multiplier
            scale_width = width // 64   # Use 64 instead of 128
            scale_height = height // 64
            print(f"   → Scale Factors: {scale_width}x{scale_height} (SDXL workaround: {width}÷64 = {scale_width})")
        else:
            # SD 1.5, FLUX, etc: normal calculation
            scale_width = width // latent_size
            scale_height = height // latent_size
            print(f"   → Scale Factors: {scale_width}x{scale_height} ({width}÷{latent_size} = {scale_width})")

        # Handle seed: -1 or None means random
        if seed is None or seed == -1:
            actual_seed = random_module.randint(0, 2**32 - 1)
        else:
            actual_seed = seed

        seed_display = 'random' if (seed is None or seed == -1) else str(seed)

        status = f"🎨 Generating image...\n\nPrompt: {prompt[:80]}...\nModel: {model}\nSize: {width}x{height}\nSteps: {steps}, CFG: {cfg_scale}\nSeed: {seed_display}\n"

        # Build FlatBuffer configuration
        builder = flatbuffers.Builder(2048)  # Increased size for more fields

        # Create strings (use actual filenames for gRPC)
        model_offset = builder.CreateString(model_file)

        # Handle LoRAs if specified
        lora_offsets = []

        if lora1_file:
            lora1_file_offset = builder.CreateString(lora1_file)
            LoRA.Start(builder)
            LoRA.AddFile(builder, lora1_file_offset)
            LoRA.AddWeight(builder, lora1_weight)
            lora_offsets.append(LoRA.End(builder))
            status += f"LoRA 1: {lora1} (weight: {lora1_weight})\n"

        if lora2_file:
            lora2_file_offset = builder.CreateString(lora2_file)
            LoRA.Start(builder)
            LoRA.AddFile(builder, lora2_file_offset)
            LoRA.AddWeight(builder, lora2_weight)
            lora_offsets.append(LoRA.End(builder))
            status += f"LoRA 2: {lora2} (weight: {lora2_weight})\n"

        # Create empty controls vector (required even if empty)
        GenerationConfiguration.StartControlsVector(builder, 0)
        controls_vector = builder.EndVector()

        # Build loras vector (supports 0, 1, or 2 LoRAs)
        GenerationConfiguration.StartLorasVector(builder, len(lora_offsets))
        for lora_offset in reversed(lora_offsets):  # Reverse for FlatBuffer prepend order
            builder.PrependUOffsetTRelative(lora_offset)
        loras_vector = builder.EndVector()

        # Build main configuration
        GenerationConfiguration.Start(builder)
        GenerationConfiguration.AddId(builder, 0)
        GenerationConfiguration.AddStartWidth(builder, scale_width)   # SCALE FACTOR not pixels!
        GenerationConfiguration.AddStartHeight(builder, scale_height) # SCALE FACTOR not pixels!

        # SDXL conditioning (required for SDXL models with latent_size=128)
        # These tell SDXL what resolution it should target
        if latent_size == 128:  # SDXL models
            GenerationConfiguration.AddOriginalImageWidth(builder, width)
            GenerationConfiguration.AddOriginalImageHeight(builder, height)
            GenerationConfiguration.AddTargetImageWidth(builder, width)
            GenerationConfiguration.AddTargetImageHeight(builder, height)
            print(f"   → SDXL conditioning: Original={width}x{height}, Target={width}x{height}")

        GenerationConfiguration.AddSeed(builder, actual_seed)
        GenerationConfiguration.AddSteps(builder, steps)
        GenerationConfiguration.AddGuidanceScale(builder, cfg_scale)
        GenerationConfiguration.AddStrength(builder, 1.0)
        GenerationConfiguration.AddModel(builder, model_offset)

        # Get sampler ID from name
        sampler_id = SAMPLERS.get(sampler_name, 0)
        GenerationConfiguration.AddSampler(builder, sampler_id)

        GenerationConfiguration.AddBatchCount(builder, 1)
        GenerationConfiguration.AddBatchSize(builder, 1)

        # Always add these core fields (required by server)
        GenerationConfiguration.AddSeedMode(builder, seed_mode)
        GenerationConfiguration.AddClipSkip(builder, clip_skip)  # From preset (Pony needs 2!)
        GenerationConfiguration.AddShift(builder, final_shift)  # Uses calculated shift if res_dependent_shift enabled
        GenerationConfiguration.AddControls(builder, controls_vector)
        GenerationConfiguration.AddLoras(builder, loras_vector)

        # Note: ResolutionDependentShift is calculated CLIENT-SIDE above and applied to final_shift
        # Other settings like CfgZeroStar, HiresFix are not in FlatBuffer schema

        config = GenerationConfiguration.End(builder)
        builder.Finish(config)
        config_bytes = bytes(builder.Output())

        # Create gRPC request
        request = imageService_pb2.ImageGenerationRequest(
            prompt=prompt,
            negativePrompt=negative_prompt if negative_prompt else '',
            configuration=config_bytes,
            scaleFactor=1,
            user='MuddleMeThis',
            device=imageService_pb2.LAPTOP,
            chunked=False
        )

        status += "\n📡 Sending request to server...\n"

        # Initialize progress
        progress(0, desc="Starting generation...")

        # Generate!
        generated_images = []
        current_step = 0

        for response in state.grpc_client.stub.GenerateImage(request):
            # Handle progress updates
            if response.HasField('currentSignpost'):
                signpost = response.currentSignpost
                if signpost.HasField('sampling'):
                    current_step = signpost.sampling.step
                    progress_pct = current_step / steps
                    progress(progress_pct, desc=f"Sampling: step {current_step}/{steps}")
                elif signpost.HasField('textEncoded'):
                    progress(0.05, desc="Text encoded")
                elif signpost.HasField('imageEncoded'):
                    progress(0.95, desc="Image encoded")
                elif signpost.HasField('imageDecoded'):
                    progress(0.98, desc="Image decoded")

            # Collect generated images
            if response.generatedImages:
                generated_images.extend(response.generatedImages)

        if generated_images:
            # Decode tensor to PIL Image
            pil_image = tensor_to_pil(generated_images[0])

            # Calculate generation time
            generation_time = time.time() - start_time

            # Add metadata to image
            metadata = PngInfo()
            metadata.add_text("prompt", prompt)
            metadata.add_text("negative_prompt", negative_prompt if negative_prompt else "")
            metadata.add_text("model", model)
            metadata.add_text("model_file", model_file)
            if lora1 and lora1 != "None":
                metadata.add_text("lora1", f"{lora1} ({lora1_weight})")
            if lora2 and lora2 != "None":
                metadata.add_text("lora2", f"{lora2} ({lora2_weight})")
            metadata.add_text("steps", str(steps))
            metadata.add_text("cfg_scale", str(cfg_scale))
            metadata.add_text("sampler", sampler_name)
            metadata.add_text("resolution", f"{width}x{height}")
            metadata.add_text("seed", str(actual_seed))
            metadata.add_text("shift", str(final_shift))
            metadata.add_text("clip_skip", str(clip_skip))
            metadata.add_text("generation_time", f"{generation_time:.2f}s")
            metadata.add_text("created_with", "MuddleMeThis")

            # Save with metadata and better filename
            # Create output directory if it doesn't exist
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            # Generate descriptive filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Clean model name for filename
            model_clean = model.replace(" ", "_").replace("/", "-")[:30]
            # First few words of prompt for filename
            prompt_short = "_".join(prompt.split()[:3]).replace("/", "-")[:30]
            filename = f"{timestamp}_{model_clean}_{prompt_short}_s{actual_seed}.png"
            filepath = output_dir / filename

            # Save as PNG with metadata
            pil_image.save(filepath, "PNG", pnginfo=metadata)

            final_status = status + f"✅ Generation complete!\n\n"
            final_status += f"⏱️  Generation time: {generation_time:.2f}s\n"
            final_status += f"Image size: {width}x{height}\n"
            final_status += f"Actual seed: {actual_seed}\n"
            final_status += f"💾 Saved: {filename}"

            return pil_image, final_status
        else:
            return None, status + "❌ No images generated"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, f"❌ Error during generation:\n{str(e)}\n\nDetails:\n{error_details}"


# ============================================================================
# Gradio Interface with PWA Support
# ============================================================================

def create_ui():
    """Create the Gradio interface with PWA support and custom styling"""

    # Custom CSS for Calibri font and styling
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Calibri:wght@400;700&display=swap');

    * {
        font-family: Calibri, 'Segoe UI', Tahoma, sans-serif !important;
    }

    .gradio-container {
        font-family: Calibri, 'Segoe UI', Tahoma, sans-serif !important;
    }
    """

    # Gradio 6.0: theme, css, head moved to launch()
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Calibri"), "Arial", "sans-serif"]
    )

    with gr.Blocks(title="MuddleMeThis") as app:
        gr.Markdown("# 🎨 MuddleMeThis")
        gr.Markdown("AI-powered prompt engineering and image generation")

        with gr.Tabs() as tabs:
            # ==================================================================
            # TAB 1: Prompt Expansion
            # ==================================================================
            with gr.Tab("📝 Expand Prompt"):
                gr.Markdown("### Expand a brief prompt into a detailed description")

                with gr.Row():
                    with gr.Column(scale=2):
                        expand_input = gr.Textbox(
                            label="Brief Prompt",
                            placeholder="Enter a short prompt (e.g., 'a peaceful garden')",
                            lines=5
                        )
                        expand_btn = gr.Button("🚀 Expand Prompt", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        expand_output = gr.Textbox(
                            label="Expanded Prompt",
                            lines=12,
                            interactive=True
                        )
                        expand_send_to_refine = gr.Button("➡️ Send to Refine", size="sm")

                expand_btn.click(
                    fn=expand_prompt,
                    inputs=[expand_input],
                    outputs=expand_output
                )

            # ==================================================================
            # TAB 2: Prompt Extraction
            # ==================================================================
            with gr.Tab("🖼️ Extract from Image"):
                gr.Markdown("### Analyze an image and generate a matching prompt")

                with gr.Row():
                    with gr.Column(scale=2):
                        extract_image = gr.Image(label="Upload Image", type="numpy")
                        extract_btn = gr.Button("🔍 Extract Prompt", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        extract_output = gr.Textbox(
                            label="Extracted Prompt",
                            lines=12,
                            interactive=True
                        )
                        extract_send_to_refine = gr.Button("➡️ Send to Refine", size="sm")

                extract_btn.click(
                    fn=extract_prompt,
                    inputs=[extract_image],
                    outputs=extract_output
                )

            # ==================================================================
            # TAB 3: Prompt Refinement
            # ==================================================================
            with gr.Tab("✏️ Refine Prompt"):
                gr.Markdown("### Modify an existing prompt with specific instructions")

                with gr.Row():
                    with gr.Column(scale=2):
                        refine_current = gr.Textbox(
                            label="Current Prompt",
                            placeholder="Paste your current prompt here...",
                            lines=6
                        )
                        refine_instruction = gr.Textbox(
                            label="Refinement Instruction",
                            placeholder="e.g., 'change the hair to red' or 'add sunset lighting'",
                            lines=3
                        )
                        refine_btn = gr.Button("🔧 Refine Prompt", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        refine_output = gr.Textbox(
                            label="Refined Prompt",
                            lines=12,
                            interactive=True
                        )

                refine_btn.click(
                    fn=refine_prompt,
                    inputs=[refine_current, refine_instruction],
                    outputs=refine_output
                )

            # ==================================================================
            # TAB 4: Direct Mode
            # ==================================================================
            with gr.Tab("✍️ Direct Mode"):
                gr.Markdown("### Write your prompt directly and generate")

                direct_prompt = gr.Textbox(
                    label="Your Prompt",
                    placeholder="Enter your complete prompt...",
                    lines=10
                )
                gr.Markdown("*Use the Image Generation section below to create the image*")

            # ==================================================================
            # TAB 5: Settings
            # ==================================================================
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### Configuration")

                with gr.Accordion("LLM Settings", open=True):
                    gr.Markdown("""
                    **💡 Recommended Vision Models:**
                    - **Ollama**: `qwen3-vl:4b-instruct` or better (install: `ollama pull qwen3-vl:4b-instruct`)
                    - **LM Studio**: https://lmstudio.ai/models/qwen/qwen3-vl-4b
                    - **Lightweight**: `qwen3-vl-2b-instruct` (fast, good quality)
                    """)

                    llm_provider = gr.Radio(
                        choices=["LM Studio", "Ollama"],
                        value=settings.get('llm_provider', 'LM Studio'),
                        label="LLM Provider",
                        info="Choose your LLM backend"
                    )
                    llm_server = gr.Textbox(
                        label="LLM Server URL",
                        value=settings.get('llm_server', 'http://localhost:1234'),
                        placeholder="LM Studio: http://localhost:1234 | Ollama: http://localhost:11434"
                    )
                    llm_model_dropdown = gr.Dropdown(
                        label="Text Model",
                        choices=[],
                        value=settings.get('llm_model', ''),
                        allow_custom_value=True,
                        interactive=True,
                        info="For prompt expansion and refinement"
                    )
                    llm_vision_model_dropdown = gr.Dropdown(
                        label="Vision Model (for image analysis)",
                        choices=[],
                        value=settings.get('llm_vision_model', ''),
                        allow_custom_value=True,
                        interactive=True,
                        info="Required for 'Extract from Image' tab. Leave empty to use text model if it supports vision."
                    )
                    llm_connect_btn = gr.Button("Connect to LLM Server")
                    llm_status = gr.Textbox(label="Status", interactive=False, lines=5)

                    llm_connect_btn.click(
                        fn=init_llm,
                        inputs=[llm_server, llm_model_dropdown, llm_vision_model_dropdown, llm_provider],
                        outputs=[llm_status, llm_model_dropdown, llm_vision_model_dropdown]
                    )

                    # When user selects a model, initialize it
                    def on_llm_model_change(server_url: str, model: str, vision_model: str, provider: str) -> str:
                        if not model:
                            return "Please select a model"
                        try:
                            provider_map = {'LM Studio': 'lm_studio', 'Ollama': 'ollama'}
                            provider_key = provider_map.get(provider, 'lm_studio')

                            settings.update_config(llm_model=model, llm_vision_model=vision_model, llm_provider=provider)
                            state.llm_client, state.text_processor, state.vision_processor = create_ai_client(
                                provider=provider_key,
                                base_url=server_url,
                                text_model=model,
                                vision_model=vision_model or model  # Use text model if no vision model
                            )
                            vision_info = f" / Vision: {vision_model}" if vision_model else ""
                            return f"✅ Models loaded: {model}{vision_info} ({provider})"
                        except Exception as e:
                            return f"❌ Failed to load model: {str(e)}"

                    llm_model_dropdown.change(
                        fn=on_llm_model_change,
                        inputs=[llm_server, llm_model_dropdown, llm_vision_model_dropdown, llm_provider],
                        outputs=llm_status
                    )

                    llm_vision_model_dropdown.change(
                        fn=on_llm_model_change,
                        inputs=[llm_server, llm_model_dropdown, llm_vision_model_dropdown, llm_provider],
                        outputs=llm_status
                    )

                with gr.Accordion("gRPC Settings", open=True):
                    grpc_server = gr.Textbox(
                        label="gRPC Server Address",
                        value=settings.get('grpc_server', 'localhost:7859'),
                        placeholder="localhost:7859"
                    )
                    grpc_connect_btn = gr.Button("Connect to gRPC Server")
                    grpc_status = gr.Textbox(label="Status", interactive=False, lines=10)
                    grpc_model_dropdown = gr.Dropdown(
                        label="Available Models",
                        choices=[],
                        allow_custom_value=True,
                        filterable=True
                    )
                    grpc_lora_dropdown = gr.Dropdown(
                        label="Available LoRAs",
                        choices=[],
                        allow_custom_value=True,
                        filterable=True
                    )

                    # Note: We'll wire up the connection button outputs below after creating gen_model and gen_lora

                with gr.Accordion("System Prompts", open=False):
                    gr.Markdown(f"""
                    System prompts are loaded from the `settings/` folder:
                    - **expand.txt**: Prompt expansion instructions
                    - **extract.txt**: Image analysis instructions
                    - **refine.txt**: Prompt refinement instructions

                    Edit these files to customize LLM behavior.
                    """)

        # ======================================================================
        # Image Generation Section (Always Visible) - Outside Tabs
        # ======================================================================
        gr.Markdown("---")
        gr.Markdown("## 🎨 Generate Image")

        with gr.Row():
            with gr.Column(scale=2):
                gen_prompt = gr.Textbox(
                    label="Final Prompt",
                    placeholder="Your prompt will appear here from the tabs above, or type directly",
                    lines=8
                )

                gen_model = gr.Dropdown(
                    label="Model",
                    choices=[],
                    value=settings.get('last_used_model', ''),
                    allow_custom_value=True,
                    filterable=True
                )

                gen_preset = gr.Dropdown(
                    label="Preset",
                    choices=["Custom (no preset)"],
                    value="Custom (no preset)",
                    interactive=True
                )

                with gr.Row():
                    gen_lora1 = gr.Dropdown(
                        label="LoRA 1 (optional)",
                        choices=["None"],
                        value="None",
                        allow_custom_value=True,
                        filterable=True,
                        scale=3
                    )
                    gen_lora1_weight = gr.Slider(
                        0.0, 2.0, 1.0,
                        label="Weight",
                        step=0.05,
                        scale=1
                    )

                with gr.Row():
                    gen_lora2 = gr.Dropdown(
                        label="LoRA 2 (optional)",
                        choices=["None"],
                        value="None",
                        allow_custom_value=True,
                        filterable=True,
                        scale=3
                    )
                    gen_lora2_weight = gr.Slider(
                        0.0, 2.0, 1.0,
                        label="Weight",
                        step=0.05,
                        scale=1
                    )

                with gr.Row():
                    gen_steps = gr.Slider(
                        1, 150, settings.get('default_steps', 16),
                        label="Steps",
                        step=1
                    )
                    gen_cfg = gr.Slider(
                        1.0, 20.0, settings.get('default_cfg', 7.0),
                        label="CFG Scale",
                        step=0.5
                    )

                gen_sampler = gr.Dropdown(
                    choices=SAMPLER_NAMES,
                    value=SAMPLER_DEFAULT,
                    label="Sampler"
                )

                gen_clip_skip = gr.Slider(
                    1, 12, 1,
                    label="CLIP Skip",
                    step=1,
                    info="CLIP layers to skip (1=default, Pony/Illustrious need 2)"
                )

                # Pre-populate with 1024 base resolution defaults
                default_aspects = [label for label, _, _ in settings.load_aspect_ratios(1024)]
                gen_aspect = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=default_aspects,  # Pre-populated, will update when model selected
                    value=settings.get('default_aspect_ratio', default_aspects[4] if len(default_aspects) > 4 else default_aspects[0])
                )

                gen_seed = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                    precision=0
                )

                gen_negative = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What to avoid...",
                    lines=3
                )

                gen_preset_info = gr.Textbox(
                    label="Model Preset Info",
                    interactive=False,
                    lines=2
                )

                # Advanced Settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    gr.Markdown("*Optional advanced generation parameters*")

                    gen_shift = gr.Slider(
                        0.0, 10.0, 1.0,
                        label="Shift",
                        step=0.1,
                        info="Timestep shift for generation (1.0 is default)"
                    )

                    gen_res_shift = gr.Checkbox(
                        value=False,
                        label="Resolution Dependent Shift",
                        info="Automatically adjust shift based on resolution (calculated client-side)"
                    )

                    gen_seed_mode = gr.Slider(
                        0, 5, 2,
                        label="Seed Mode",
                        step=1,
                        info="Random seed generation mode (2 is default)"
                    )

                    # Hidden placeholders for removed settings (kept for preset compatibility)
                    gen_cfg_zero = gr.Checkbox(value=False, visible=False)
                    gen_hires = gr.Checkbox(value=False, visible=False)

                gen_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")

            with gr.Column(scale=3):
                gen_image = gr.Image(label="Generated Image")
                gen_status = gr.Textbox(label="Generation Status", lines=8)

        # Model selection updates aspect ratios and shows available presets
        gen_model.change(
            fn=on_model_selected,
            inputs=[gen_model],
            outputs=[gen_preset, gen_preset_info, gen_aspect]
        )

        # Preset selection updates all settings
        gen_preset.change(
            fn=on_preset_selected,
            inputs=[gen_preset],
            outputs=[gen_steps, gen_cfg, gen_preset_info, gen_sampler,
                    gen_shift, gen_res_shift, gen_seed_mode, gen_cfg_zero, gen_hires, gen_clip_skip]
        )

        # Connect outputs to generation
        gen_btn.click(
            fn=generate_image,
            inputs=[gen_prompt, gen_model, gen_lora1, gen_lora1_weight, gen_lora2, gen_lora2_weight,
                   gen_steps, gen_cfg, gen_sampler, gen_aspect, gen_seed, gen_negative,
                   gen_shift, gen_res_shift, gen_seed_mode, gen_cfg_zero, gen_hires, gen_clip_skip],
            outputs=[gen_image, gen_status]
        )

        # Link outputs from tabs to the generation prompt field
        expand_output.change(lambda x: x, inputs=expand_output, outputs=gen_prompt)
        extract_output.change(lambda x: x, inputs=extract_output, outputs=gen_prompt)
        refine_output.change(lambda x: x, inputs=refine_output, outputs=gen_prompt)
        direct_prompt.change(lambda x: x, inputs=direct_prompt, outputs=gen_prompt)

        # Send to Refine buttons
        expand_send_to_refine.click(lambda x: x, inputs=expand_output, outputs=refine_current)
        extract_send_to_refine.click(lambda x: x, inputs=extract_output, outputs=refine_current)

        # Wire up gRPC connection to update BOTH settings and generation dropdowns
        # This is done here because gen_model and gen_lora1/2 are created after grpc_connect_btn
        def init_grpc_all(server_url):
            """Initialize gRPC and return updates for all dropdowns"""
            status, models_dropdown, loras_dropdown = init_grpc(server_url)
            # Return: status, settings_models, settings_loras, gen_models, gen_lora1, gen_lora2
            return status, models_dropdown, loras_dropdown, models_dropdown, loras_dropdown, loras_dropdown

        grpc_connect_btn.click(
            fn=init_grpc_all,
            inputs=[grpc_server],
            outputs=[grpc_status, grpc_model_dropdown, grpc_lora_dropdown, gen_model, gen_lora1, gen_lora2]
        )

    return app


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("🎨 MuddleMeThis - Starting...")
    print(f"📁 Settings directory: {settings.settings_dir}")
    print(f"⚙️  Config file: {settings.config_file}")

    app = create_ui()

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
        pwa=True
    )
