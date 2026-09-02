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
import subprocess
import time
import hashlib
import json
from math import gcd
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

import numpy as np

# Application version
APP_VERSION = "1.0.0"

# Suppress gRPC SSL handshake warnings (these are harmless when using self-signed certs)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

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
    from drawthings_client import (
        DrawThingsClient,
        ImageGenerationConfig,
        LoRAConfig,
        ReferenceImage,
    )
    from model_metadata import ModelMetadata
    from tensor_decoder import tensor_to_pil

    GRPC_AVAILABLE = True
except ImportError as e:
    print(
        f"Warning: DTgRPCconnector not installed. Install requirements from dev/DTgRPCconnector/requirements.txt"
    )
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
        self.lora_name_to_file = {}  # Maps display names → filenames


state = AppState()


# ============================================================================
# LLM Processing Functions
# ============================================================================


def init_llm(
    server_url: str,
    model_name: str,
    vision_model_name: str,
    provider: str = "lm_studio",
) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """Initialize LLM connection and fetch available models"""
    try:
        if not MODULLE_AVAILABLE:
            return (
                "❌ ModuLLe not installed",
                gr.update(choices=[]),
                gr.update(choices=[]),
            )

        # Map provider names to ModuLLe provider strings
        provider_map = {"LM Studio": "lm_studio", "Ollama": "ollama"}
        provider_key = provider_map.get(provider, "lm_studio")

        # First, create a temporary client to list models
        temp_client, _, _ = create_ai_client(
            provider=provider_key,
            base_url=server_url,
            text_model=model_name or "placeholder",
            vision_model=vision_model_name or None,
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
                llm_provider=provider,
            )
            state.llm_client, state.text_processor, state.vision_processor = (
                create_ai_client(
                    provider=provider_key,
                    base_url=server_url,
                    text_model=model_name,
                    vision_model=vision_model_name
                    or model_name,  # Use text model if no vision model specified
                )
            )
            vision_info = (
                f" (Vision: {vision_model_name or model_name})"
                if vision_model_name
                else ""
            )
            status = f"✅ Connected to {provider}: {server_url}\nText Model: {model_name}{vision_info}"
        else:
            state.llm_client = temp_client
            status = f"✅ Connected to {provider}: {server_url}\n\nAvailable models: {len(available_models)}\nPlease select models below."

        if available_models:
            status += f"\n\nFound {len(available_models)} model(s)"

        models_dropdown = gr.update(
            choices=available_models, value=model_name if model_name else None
        )
        vision_dropdown = gr.update(
            choices=available_models,
            value=vision_model_name if vision_model_name else None,
        )
        return status, models_dropdown, vision_dropdown
    except Exception as e:
        return (
            f"❌ LLM connection failed: {str(e)}",
            gr.update(choices=[]),
            gr.update(choices=[]),
        )


def expand_prompt(user_prompt: str) -> str:
    """Expand a brief prompt into detailed description"""
    if not state.text_processor:
        return "❌ LLM not initialized. Configure in Settings tab first."

    if not user_prompt.strip():
        return "❌ Please enter a prompt to expand"

    try:
        system_prompt = settings.load_system_prompt("expand")

        result = state.text_processor.generate(
            prompt=user_prompt, system_prompt=system_prompt if system_prompt else None
        )

        if result:
            state.current_prompt = result
            return result
        return "❌ Failed to generate expansion"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def expand_prompt_advanced(user_prompt: str, expander_name: str, aspect_ratio_label: str) -> str:
    """Expand a brief prompt using a specific expander with optional aspect ratio
    
    Args:
        user_prompt: The brief prompt to expand
        expander_name: Name of the expander to use (e.g., 'default', 'ernie_en', 'ernie_cn')
        aspect_ratio_label: Aspect ratio label like "1:1 1024x1024" or "1:1 512x512", can be empty
        
    Returns:
        Expanded prompt text, extracted from code blocks if present
    """
    if not state.text_processor:
        return "❌ LLM not initialized. Configure in Settings tab first."

    if not user_prompt.strip():
        return "❌ Please enter a prompt to expand"
    
    if not expander_name:
        expander_name = "default"
    
    try:
        # Load the expander system prompt
        system_prompt = settings.load_expander_prompt(expander_name)
        
        if not system_prompt:
            # Fallback to default if expander not found
            system_prompt = settings.load_expander_prompt("default")
            if not system_prompt:
                # Last resort: simple expansion
                result = state.text_processor.generate(
                    prompt=user_prompt, system_prompt=None
                )
                if result:
                    state.current_prompt = result
                    return result
                return "❌ Failed to generate expansion"
        
        # Prepare the prompt based on whether aspect ratio is provided
        if aspect_ratio_label and "ernie" in expander_name.lower():
            # Parse aspect ratio label to get width and height
            # Format: "1:1 1024x1024" - extract the dimensions
            try:
                # Split by space and take the last part (should be WxH)
                parts = aspect_ratio_label.split()
                if len(parts) >= 2:
                    dims = parts[-1]  # e.g., "1024x1024"
                    width, height = dims.split('x')
                    width = int(width)
                    height = int(height)
                    
                    # Format as JSON for Ernie-style expanders
                    prompt_input = f'{{"prompt": "{user_prompt}", "width": {width}, "height": {height}}}'
                else:
                    # Fallback to simple prompt
                    prompt_input = user_prompt
            except (ValueError, IndexError):
                # If parsing fails, use simple prompt
                prompt_input = user_prompt
        else:
            # For default expander or when no aspect ratio selected
            prompt_input = user_prompt
        
        # Generate the expanded prompt
        result = state.text_processor.generate(
            prompt=prompt_input, system_prompt=system_prompt
        )
        
        if result:
            # Parse response: extract content from markdown code blocks if present
            # Look for ``` or ```text blocks
            lines = result.split('\n')
            in_code_block = False
            code_content = []
            
            for line in lines:
                stripped = line.strip()
                # Check for code block start/end
                if stripped.startswith('```'):
                    if not in_code_block:
                        # Starting code block
                        in_code_block = True
                        continue  # Skip the ``` line
                    else:
                        # Ending code block
                        in_code_block = False
                        break  # Stop at end of code block
                elif in_code_block:
                    code_content.append(line)
            
            # If we found code block content, use it; otherwise use the whole result
            if code_content:
                expanded = '\n'.join(code_content).strip()
            else:
                expanded = result.strip()
            
            state.current_prompt = expanded
            return expanded
        
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
        # Convert image to base64 with resize to avoid Ollama "request body too large" error
        img = Image.fromarray(image)
        # Resize if image is too large (Ollama default limit is ~4MB)
        # Target max 1024px on longest side to keep base64 under limit
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Use JPEG with quality setting for smaller file size
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Load the user's detailed system prompt from extract.txt
        # This contains all the instructions for how to analyze and generate prompts
        extraction_instructions = settings.load_system_prompt("extract")

        # Use the full instructions as the prompt for vision analysis
        # Vision models work best with instructions integrated into the prompt
        result = state.vision_processor.analyze_image(
            image_data=img_base64,
            prompt=extraction_instructions
            if extraction_instructions
            else "Analyze this image and write a detailed prompt that could generate a similar image.",
        )

        if result:
            state.current_prompt = result
            return result
        return "❌ Failed to extract prompt"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def copy_style(image) -> str:
    """Analyze image style and generate detailed style description"""
    if not state.vision_processor:
        return "❌ Vision processor not initialized. Configure in Settings tab first."

    if image is None:
        return "❌ Please upload an image first"

    try:
        # Convert image to base64 with resize to avoid Ollama "request body too large" error
        img = Image.fromarray(image)
        # Resize if image is too large (Ollama default limit is ~4MB)
        # Target max 1024px on longest side to keep base64 under limit
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Use JPEG with quality setting for smaller file size
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Load the style copy prompt from stylecopy.txt
        style_prompt = settings.load_system_prompt("stylecopy")

        # Use vision model to analyze style
        result = state.vision_processor.analyze_image(
            image_data=img_base64,
            prompt=style_prompt
            if style_prompt
            else "Describe the visual style of this image in great detail as if trying to reproduce it just from the description.",
        )

        if result:
            return result
        return "❌ Failed to analyze style"
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
        system_prompt = settings.load_system_prompt("refine")

        result = state.text_processor.generate(
            prompt=f"Current prompt: {current_prompt}\n\nModification requested: {refinement_instruction}\n\nProvide the refined prompt:",
            system_prompt=system_prompt if system_prompt else None,
        )

        if result:
            state.current_prompt = result
            return result
        return "❌ Failed to refine prompt"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def parse_aspect_ratio_label(aspect_ratio_label: str) -> str:
    """Convert an aspect ratio label like '1:1 1024x1024' to 'W:H' format.

    Args:
        aspect_ratio_label: Label from the aspect ratio dropdown

    Returns:
        String in "W:H" form, defaults to "1:1"
    """
    if not aspect_ratio_label or aspect_ratio_label == "(none)":
        return "1:1"

    parts = aspect_ratio_label.split()
    if not parts:
        return "1:1"

    # First part is usually the ratio, e.g. "1:1"
    if ":" in parts[0]:
        return parts[0]

    # Fallback: derive from dimensions like "1024x1024"
    for part in parts:
        if "x" in part:
            try:
                width, height = part.split("x")
                width = int(width)
                height = int(height)
                divisor = gcd(width, height)
                return f"{width // divisor}:{height // divisor}"
            except (ValueError, ZeroDivisionError):
                continue

    return "1:1"


def _extract_and_repair_json(text: str) -> str:
    """Extract a valid JSON object from LLM output, with common-truncation repairs.

    Args:
        text: Raw LLM output

    Returns:
        Valid JSON string

    Raises:
        ValueError: If no valid JSON can be extracted or repaired
    """
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 1. Direct parse attempt
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 2. Extract between first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # 3. Repair known Ideogram truncation: missing opening brace / key prefix
    stripped = text.lstrip()
    if stripped.startswith("_ratio"):
        # Missing "{\"aspect" prefix
        repaired = '{"aspect' + stripped
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass
    elif stripped.startswith('"aspect_ratio"'):
        # Missing opening '{'
        repaired = "{" + stripped
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

    # 4. Balance braces by appending missing '}'
    open_count = text.count("{")
    close_count = text.count("}")
    if open_count > close_count:
        repaired = text + "}" * (open_count - close_count)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from: {text[:300]}")


def _build_fallback_ideogram_json(prompt: str, aspect_ratio: str) -> str:
    """Build a minimal valid Ideogram JSON when the LLM fails.

    Keeps the user's original wording in high_level_description and a single element.
    """
    safe_prompt = prompt.strip().replace('"', "'")
    fallback = {
        "aspect_ratio": aspect_ratio,
        "high_level_description": safe_prompt[:200],
        "compositional_deconstruction": {
            "background": "",
            "elements": [
                {
                    "type": "obj",
                    "desc": safe_prompt[:500],
                }
            ],
        },
    }
    return json.dumps(fallback, separators=(",", ":"), ensure_ascii=False)


def _xml_escape(text: str) -> str:
    """Escape text for safe inclusion in XML/XMP chunks."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_draw_things_xmp(
    prompt: str,
    negative_prompt: str,
    model: str,
    width: int,
    height: int,
    steps: int,
    sampler_name: str,
    cfg_scale: float,
    seed: int,
    shift: float,
    lora1: Optional[str] = None,
    lora1_weight: float = 1.0,
    lora2: Optional[str] = None,
    lora2_weight: float = 1.0,
    strength: float = 1.0,
    seed_mode: int = 2,
    clip_skip: int = 1,
) -> str:
    """Build an XMP metadata block matching Draw Things' PNG structure.

    This lets images saved by MuddleMeThis carry compatible metadata with
    Draw Things and other Stable Diffusion tools.
    """
    size = f"{width}x{height}"

    # Human-readable parameter line (matches Draw Things dc:description format)
    params = [
        f"Steps: {steps}",
        f"Sampler: {sampler_name}",
        f"Guidance Scale: {cfg_scale}",
        f"Seed: {seed}",
        f"Size: {size}",
        f"Model: {model}",
        f"Strength: {strength}",
        f"Seed Mode: {seed_mode}",
        f"Shift: {shift}",
    ]
    if lora1 and lora1 != "None":
        params.append(f"LoRA Model: {lora1}")
        params.append(f"LoRA Weight: {lora1_weight}")
    if lora2 and lora2 != "None":
        params.append(f"LoRA 2 Model: {lora2}")
        params.append(f"LoRA 2 Weight: {lora2_weight}")

    description = prompt + "\n" + ", ".join(params)
    description_escaped = _xml_escape(description)

    # JSON UserComment payload (simplified Draw Things config)
    user_comment = {
        "c": prompt,
        "model": model,
        "sampler": sampler_name,
        "steps": steps,
        "scale": cfg_scale,
        "seed": seed,
        "seed_mode": seed_mode,
        "shift": shift,
        "size": size,
        "strength": strength,
        "width": width,
        "height": height,
        "uc": negative_prompt if negative_prompt else "",
        "clip_skip": clip_skip,
    }

    loras = []
    if lora1 and lora1 != "None":
        loras.append({"file": lora1, "weight": lora1_weight})
    if lora2 and lora2 != "None":
        loras.append({"file": lora2, "weight": lora2_weight})
    if loras:
        user_comment["loras"] = loras

    user_comment_json = json.dumps(user_comment, ensure_ascii=False, separators=(",", ":"))
    user_comment_escaped = _xml_escape(user_comment_json)

    xmp = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 6.0.0">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about=""
            xmlns:dc="http://purl.org/dc/elements/1.1/"
            xmlns:xmp="http://ns.adobe.com/xap/1.0/"
            xmlns:exif="http://ns.adobe.com/exif/1.0/">
         <dc:description>
            <rdf:Alt>
               <rdf:li xml:lang="x-default">{description_escaped}</rdf:li>
            </rdf:Alt>
         </dc:description>
         <xmp:CreatorTool>MuddleMeThis</xmp:CreatorTool>
         <exif:UserComment>
            <rdf:Alt>
               <rdf:li xml:lang="x-default">{user_comment_escaped}</rdf:li>
            </rdf:Alt>
         </exif:UserComment>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>"""
    return xmp


def jsonify_prompt_for_ideogram(prompt: str, aspect_ratio_label: str) -> str:
    """Convert a natural language prompt into Ideogram 4's structured JSON format.

    Args:
        prompt: The image generation prompt to convert
        aspect_ratio_label: Aspect ratio label from the Generate tab dropdown

    Returns:
        Minified JSON string ready for Ideogram 4, or an error message
    """
    if not state.text_processor:
        return "❌ LLM not initialized. Configure in Settings tab first."

    if not prompt.strip():
        return "❌ Please enter a prompt first"

    aspect_ratio = parse_aspect_ratio_label(aspect_ratio_label)
    system_prompt = settings.load_system_prompt("ideogram_json")

    if not system_prompt:
        return "❌ Ideogram JSON system prompt not found: settings/ideogram_json.txt"

    try:
        user_input = f"Prompt: {prompt}\nTarget aspect ratio: {aspect_ratio}"

        result = state.text_processor.generate(
            prompt=user_input, system_prompt=system_prompt
        )

        if result and result.strip():
            try:
                valid_json = _extract_and_repair_json(result)
                parsed = json.loads(valid_json)
                minified = json.dumps(
                    parsed, separators=(",", ":"), ensure_ascii=False
                )
                state.current_prompt = minified
                return minified
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Warning: LLM JSON parse failed ({e}), using fallback.")
                # Fall through to deterministic fallback
        else:
            print("Warning: LLM returned empty JSON-ify response, using fallback.")

        # Deterministic fallback: guaranteed valid Ideogram JSON
        fallback_json = _build_fallback_ideogram_json(prompt, aspect_ratio)
        state.current_prompt = fallback_json
        return fallback_json

    except Exception as e:
        print(f"Warning: LLM JSON-ify raised exception ({e}), using fallback.")
        fallback_json = _build_fallback_ideogram_json(prompt, aspect_ratio)
        state.current_prompt = fallback_json
        return fallback_json


# ============================================================================
# gRPC Functions
# ============================================================================


def init_grpc(server_url: str) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """Initialize gRPC connection and fetch models/LoRAs"""
    try:
        if not GRPC_AVAILABLE:
            return (
                "❌ DTgRPCconnector not installed",
                gr.update(choices=[]),
                gr.update(choices=[]),
            )

        # Update settings
        settings.update_config(grpc_server=server_url)

        # Use Draw Things root CA certificate for SSL validation.
        # The DTgRPCconnector patch removed the hardcoded localhost override,
        # restoring TLS connections for non-localhost addresses.
        root_ca_path = Path(__file__).parent / "dev" / "DTgRPCconnector" / "root_ca.crt"

        if root_ca_path.exists():
            state.grpc_client = DrawThingsClient(
                server_address=server_url,
                insecure=False,
                verify_ssl=False,
                ssl_cert_path=str(root_ca_path),
            )
        else:
            print(f"Warning: Root CA certificate not found at {root_ca_path}")
            print("Attempting insecure connection...")
            state.grpc_client = DrawThingsClient(
                server_address=server_url, insecure=True
            )

        # Use Echo request to get structured metadata
        echo_request = imageService_pb2.EchoRequest(name="list_files")
        response = state.grpc_client.stub.Echo(echo_request)

        models = []
        loras = []

        # Use server's structured metadata (properly categorized)
        if response.HasField("override"):
            import json

            try:
                models_data = (
                    json.loads(response.override.models)
                    if response.override.models
                    else []
                )
                loras_data = (
                    json.loads(response.override.loras)
                    if response.override.loras
                    else []
                )

                # Extract both display names and filenames, create mappings
                # Models: use 'name' field for display, 'file' for actual loading
                model_display_names = []
                state.model_name_to_file = {}
                for m in models_data:
                    if m.get("file"):
                        # Use 'name' field if available, otherwise use filename
                        display_name = m.get("name", m["file"])
                        file_name = m["file"]
                        model_display_names.append(display_name)
                        state.model_name_to_file[display_name] = file_name

                # LoRAs: same approach
                lora_display_names = []
                state.lora_name_to_file = {}
                for l in loras_data:
                    if l.get("file"):
                        display_name = l.get("name", l["file"])
                        file_name = l["file"]
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
                print(
                    "Warning: Failed to parse model metadata, falling back to file list"
                )
                models = []
                loras = []
                state.model_name_to_file = {}
                state.lora_name_to_file = {}
                state.grpc_metadata = ModelMetadata(server_url)

        # Fallback: use simple file list if metadata not available
        if not models and response.files:
            for filename in response.files:
                lower = filename.lower()
                if (
                    ".ckpt" in lower or ".safetensors" in lower
                ) and "lora" not in lower:
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
        return (
            status,
            gr.update(choices=models, value=models[0] if models else None),
            gr.update(choices=lora_choices, value="None"),
        )
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return (
            f"❌ gRPC connection failed: {str(e)}\n\nDetails:\n{error_details}",
            gr.update(choices=[]),
            gr.update(choices=[]),
        )


def on_model_selected(model_name: str) -> Tuple[gr.Dropdown, str, gr.Dropdown]:
    """When a model is selected, get base resolution and update aspect ratios"""
    if not model_name or not state.grpc_metadata:
        return gr.update(choices=[]), "", gr.update(choices=[])

    # Translate display name to actual filename
    model_file = state.model_name_to_file.get(model_name, model_name)

    # Update settings (save the display name for UX)
    settings.update_config(last_used_model=model_name)

    # Get model metadata to determine base resolution (use actual filename)
    try:
        model_info = state.grpc_metadata.get_latent_info(model_file)
        latent_size = model_info.get("latent_size", 128)
        version = model_info.get("version", "sdxl")

        # Determine base resolution from version (not latent_size!)
        # FLUX/Z-Image/Qwen/SD3 use 64-latent but 1024px output
        # SDXL uses 128-latent with 1024px output
        # SD 1.5/2.x use 64-latent with 512px output
        if version in [
            "flux1",
            "z_image",
            "qwen_image",
            "sd3",
            "sd3_large",
            "sdxl",
            "sdxl_base_v0.9",
        ]:
            base_resolution = 1024
        elif version in ["v1", "v2"]:
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

    # Use gr.update() to properly update dropdown choices
    return (
        gr.update(choices=preset_choices, value="Custom (no preset)"),
        info,
        gr.update(
            choices=aspect_choices,
            value=aspect_choices[4] if len(aspect_choices) > 4 else aspect_choices[0],
        ),
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
    "TCD Trailing": 19,
}

SAMPLER_NAMES = list(SAMPLERS.keys())
SAMPLER_DEFAULT = "DPM++ 2M Karras"  # Index 0, universally supported


def on_preset_selected(
    preset_name: str,
) -> Tuple[
    int, float, str, str, float, bool, int, bool, bool, int, int, float, int, bool, any
]:
    """When a preset is selected, apply its settings"""
    if preset_name == "Custom (no preset)" or not preset_name:
        return (
            gr.update(),
            gr.update(),
            "Using custom settings",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    preset = settings.get_model_preset(preset_name)
    if preset:
        steps = preset.get("steps", preset.get("recommended_steps", 16))
        cfg = preset.get("guidanceScale", preset.get("recommended_cfg", 7.0))

        # Get sampler (handle both ID and name formats)
        sampler = preset.get("sampler", 0)
        if isinstance(sampler, str):
            # Preset uses sampler name directly
            sampler_name = sampler if sampler in SAMPLERS else SAMPLER_DEFAULT
        else:
            # Preset uses sampler ID - look up the name
            sampler_name = next(
                (name for name, id in SAMPLERS.items() if id == sampler),
                SAMPLER_DEFAULT,
            )

        # Advanced settings (shift is Float32 with default 1.0)
        shift = float(preset.get("shift", 1.0))
        res_shift = preset.get("resolutionDependentShift", False)
        seed_mode = preset.get("seedMode", 2)
        cfg_zero = preset.get("cfgZeroStar", False)
        hires_fix = preset.get("hiresFix", False)
        # Convert scale units to pixels for UI (scale_units * 64 = pixels)
        hires_fix_start_width = preset.get("hiresFixStartWidth", 0) * 64
        hires_fix_start_height = preset.get("hiresFixStartHeight", 0) * 64
        hires_fix_strength = preset.get("hiresFixStrength", 0.7)
        clip_skip = preset.get("clip_skip", 1)  # Pony needs 2, most others need 1
        tea_cache = preset.get("teaCache", False)

        # Update aspect ratios based on preset's base_resolution
        base_resolution = preset.get("base_resolution", 1024)
        state.current_model_base_resolution = base_resolution
        aspect_ratios = settings.load_aspect_ratios(base_resolution)
        aspect_choices = [label for label, _, _ in aspect_ratios]
        # Default to square aspect ratio (usually 4th or 5th in list)
        default_aspect = (
            aspect_choices[4] if len(aspect_choices) > 4 else aspect_choices[0]
        )

        notes = preset.get("notes", "")
        info = f"✅ Preset applied: {preset.get('name', 'Unknown')}\nBase resolution: {base_resolution}px\n{notes}"

        return (
            steps,
            cfg,
            info,
            sampler_name,
            shift,
            res_shift,
            seed_mode,
            cfg_zero,
            hires_fix,
            hires_fix_start_width,
            hires_fix_start_height,
            hires_fix_strength,
            clip_skip,
            tea_cache,
            gr.update(choices=aspect_choices, value=default_aspect),
        )
    else:
        return (
            gr.update(),
            gr.update(),
            f"⚠️ Preset not found: {preset_name}",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def load_video_presets() -> List[Tuple[str, str]]:
    """Load presets that support video generation (have numFrames field).

    Returns:
        List of (display_name, filename) tuples
    """
    video_presets = [("Custom (no preset)", "")]
    presets_dir = settings.presets_dir

    if not presets_dir.exists():
        return video_presets

    for preset_file in presets_dir.glob("*.json"):
        try:
            with open(preset_file, "r") as f:
                preset_data = json.load(f)

            # A video-capable preset has numFrames or notes mention LTX/video
            if (
                "numFrames" in preset_data
                or "num_frames" in preset_data
                or "video" in preset_data.get("notes", "").lower()
                or "ltx" in preset_data.get("name", "").lower()
                or "ltx" in preset_file.stem.lower()
            ):
                display_name = preset_data.get("name", preset_file.stem)
                video_presets.append((display_name, preset_file.stem))
        except Exception as e:
            print(f"Warning: Failed to load preset {preset_file}: {e}")

    return video_presets


def on_video_preset_selected(preset_name: str):
    """Apply a video preset to the Video tab controls"""
    if preset_name == "Custom (no preset)" or not preset_name:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Using custom settings",
        )

    preset = settings.get_model_preset(preset_name)
    if not preset:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            f"⚠️ Preset not found: {preset_name}",
        )

    # Extract video-relevant settings
    width = preset.get("width", 512)
    height = preset.get("height", 512)
    steps = preset.get("steps", preset.get("recommended_steps", 16))
    cfg = preset.get("guidanceScale", preset.get("recommended_cfg", 1.0))
    shift = float(preset.get("shift", 1.0))
    num_frames = preset.get("numFrames", 14)
    hires_fix = preset.get("hiresFix", False)
    hires_fix_width = preset.get("hiresFixWidth", 640)
    hires_fix_height = preset.get("hiresFixHeight", 384)
    hires_fix_strength = preset.get("hiresFixStrength", 0.7)

    # Resolve sampler name from ID or string
    sampler = preset.get("sampler", 0)
    if isinstance(sampler, str):
        sampler_name = sampler if sampler in SAMPLERS else SAMPLER_DEFAULT
    else:
        sampler_name = next(
            (name for name, sid in SAMPLERS.items() if sid == sampler),
            SAMPLER_DEFAULT,
        )

    model_name = preset.get("model", "")
    notes = preset.get("notes", "")
    info = f"✅ Video preset applied: {preset.get('name', preset_name)}\n{width}x{height}, {num_frames} frames\n{notes}"
    if hires_fix:
        info += f"\n🎬 Hires fix: {hires_fix_width}x{hires_fix_height} → {width}x{height}"

    return (
        gr.update(value=width),
        gr.update(value=height),
        gr.update(value=steps),
        gr.update(value=cfg),
        gr.update(value=sampler_name),
        gr.update(value=shift),
        gr.update(value=num_frames),
        gr.update(value=hires_fix),
        gr.update(value=hires_fix_width),
        gr.update(value=hires_fix_height),
        gr.update(value=hires_fix_strength),
        info,
    )


def on_negative_prompt_preset_selected(preset_name: str) -> str:
    """When a negative prompt preset is selected, return its text"""
    if not preset_name:
        return gr.update()

    negative_prompt_presets = settings.load_negative_prompts()
    if preset_name in negative_prompt_presets:
        preset = negative_prompt_presets[preset_name]
        return preset.get("negative_prompt", "")
    else:
        return gr.update()


def decode_preview(data: bytes) -> Optional[Image.Image]:
    """Decode a preview image from gRPC response.

    The server sends previews as CCV tensors. These may be:
    - Latent space (e.g. 128x128x16): visualize first 3 channels, upscaled
    - Pixel space (e.g. WxHx3/4): decode like a normal tensor
    Returns a PIL Image or None on failure.
    """
    import struct as _struct

    try:
        if len(data) < 68:
            return None
        hdr = _struct.unpack_from("<9I", data, 0)
        channels = hdr[8]
        height = hdr[6]
        width = hdr[7]

        if channels in (3, 4):
            # Standard pixel-space tensor
            return tensor_to_pil(data)

        # Latent-space preview (16 channels, etc.)
        # Decompress the tensor and visualize first 3 channels
        import numpy as np
        import fpzip

        compressed_data = data[68:]
        float_data = fpzip.decompress(compressed_data, order="C")
        if float_data.ndim == 4:
            float_data = float_data[0]  # Remove batch dim

        # Take first 3 channels as pseudo-RGB
        rgb = float_data[:, :, :3]

        # Normalize per-channel to [0, 255] for visibility
        for c in range(3):
            ch = rgb[:, :, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 1e-6:
                rgb[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                rgb[:, :, c] = 0.5

        img_array = (rgb * 255).clip(0, 255).astype(np.uint8)
        preview = Image.fromarray(img_array, "RGB")

        # Upscale small latent previews so they're visible in the UI
        if width < 256 or height < 256:
            scale = max(1, 512 // max(width, height))
            preview = preview.resize(
                (width * scale, height * scale), Image.Resampling.NEAREST
            )

        return preview
    except Exception as e:
        print(f"   ⚠️  Preview decode failed: {e}")
        return None


def generate_image(
    prompt: str,
    model: str,
    lora1: str,
    lora1_weight: float,
    lora2: str,
    lora2_weight: float,
    steps: int,
    cfg_scale: float,
    sampler_name: str,
    aspect_ratio: str,
    resolution_scale: str,
    seed: int,
    negative_prompt: str,
    shift: float,
    res_dependent_shift: bool,
    seed_mode: int,
    cfg_zero_star: bool,
    hires_fix: bool,
    hires_fix_start_width: int,
    hires_fix_start_height: int,
    hires_fix_strength: float,
    clip_skip: int,
    tea_cache: bool,
    tcd_gamma: float,
    live_preview: bool = False,
    progress=gr.Progress(),
):
    """Generate image using Draw Things gRPC with progress tracking (generator for live preview)"""
    if not state.grpc_client:
        yield None, "❌ gRPC not initialized. Configure in Settings tab first."
        return

    if not prompt:
        yield None, "❌ No prompt provided"
        return

    if not model:
        yield None, "❌ No model selected"
        return

    # Translate display names to actual filenames for gRPC
    model_file = state.model_name_to_file.get(model, model)
    lora1_file = (
        state.lora_name_to_file.get(lora1, lora1)
        if lora1 and lora1.strip() and lora1 != "None"
        else None
    )
    lora2_file = (
        state.lora_name_to_file.get(lora2, lora2)
        if lora2 and lora2.strip() and lora2 != "None"
        else None
    )

    # Start timing
    start_time = time.time()

    try:
        # Get model metadata FIRST to determine latent size and base resolution (use filename)
        try:
            model_info = state.grpc_metadata.get_latent_info(model_file)
            latent_size = (
                model_info.get("latent_size") or 128
            )  # Handle None/null values
            version = model_info.get("version") or "sdxl"

            print(f"\n🔍 Model Metadata for {model} → {model_file}:")
            print(f"   Version: {version}")
            print(f"   Latent Size: {latent_size}")
            print(f"   Default Scale: {model_info.get('default_scale', 'N/A')}")
        except Exception as e:
            # Fallback: assume SDXL
            print(f"\n⚠️  Failed to get model metadata: {e}")
            latent_size = 128
            version = "sdxl"

        # Use the base resolution that was set by model/preset selection
        # This ensures aspect ratios match what's in the dropdown
        base_resolution = state.current_model_base_resolution
        print(f"   → Base Resolution (from state): {base_resolution}")

        # Parse aspect ratio to get width and height using correct base resolution
        # Format: "ratio widthxheight"
        aspect_ratios = settings.load_aspect_ratios(base_resolution)
        width, height = base_resolution, base_resolution  # Default to square

        for label, w, h in aspect_ratios:
            if label == aspect_ratio:
                width, height = w, h
                break

        # Apply resolution scale multiplier
        scale_multiplier = float(resolution_scale.replace("x", ""))
        width = int(width * scale_multiplier)
        height = int(height * scale_multiplier)

        # Round to nearest 64 pixels (required for VAE)
        width = (width + 32) // 64 * 64
        height = (height + 32) // 64 * 64

        print(f"\n📐 Aspect Ratio: {aspect_ratio} × {resolution_scale}")
        print(f"   → Pixel Dimensions: {width}x{height}")
        print(f"   → Latent Size: {latent_size}")

        # Calculate resolution-dependent shift (client-side calculation)
        # Official formula from Draw Things ModelZoo.swift:2358-2360
        final_shift = float(shift)
        if res_dependent_shift:
            # Resolution factor: pixel area divided by 256
            # This is the universal formula used by Draw Things regardless of model latent size
            resolution_factor = (width * height) / 256

            # Official exponential formula: maps resolution to shift range 0.5-1.15
            import math

            calculated_shift = math.exp(
                ((resolution_factor - 256) * (1.15 - 0.5) / (4096 - 256)) + 0.5
            )

            # When resolution-dependent shift is enabled, the calculated value replaces the manual shift
            final_shift = calculated_shift

            print(f"\n⚙️  Resolution-Dependent Shift Calculation (Official Formula):")
            print(f"   Pixels: {width}x{height}")
            print(f"   Resolution Factor: {resolution_factor:.1f}")
            print(f"   → Calculated Shift: {final_shift:.2f}")
        else:
            print(f"\n⚙️  Shift: {final_shift} (no resolution adjustment)")

        # High-res fix settings - convert pixels to scale units
        hires_fix_start_width_scale = hires_fix_start_width // 64
        hires_fix_start_height_scale = hires_fix_start_height // 64

        # Validate hires fix settings
        hires_fix_valid = False
        if hires_fix:
            target_width_scale = width // 64
            target_height_scale = height // 64

            if hires_fix_start_width_scale <= 0 or hires_fix_start_height_scale <= 0:
                print(f"\n⚠️  High-Res Fix: DISABLED - Start resolution must be > 0")
                print(f"   Hint: Set start resolution to at least 64×64 pixels")
                hires_fix = False
            elif (
                hires_fix_start_width_scale >= target_width_scale
                or hires_fix_start_height_scale >= target_height_scale
            ):
                print(
                    f"\n⚠️  High-Res Fix: DISABLED - Start resolution must be SMALLER than target"
                )
                print(
                    f"   Start: {hires_fix_start_width}×{hires_fix_start_height}px ({hires_fix_start_width_scale}×{hires_fix_start_height_scale} scale)"
                )
                print(
                    f"   Target: {width}×{height}px ({target_width_scale}×{target_height_scale} scale)"
                )
                print(
                    f"   Hint: Either increase target resolution OR decrease start resolution"
                )
                hires_fix = False
            else:
                hires_fix_valid = True
                print(f"\n🔧 High-Res Fix Enabled:")
                print(
                    f"   Start Resolution: {hires_fix_start_width}×{hires_fix_start_height}px ({hires_fix_start_width_scale}×{hires_fix_start_height_scale} scale)"
                )
                print(
                    f"   Target Resolution: {width}×{height}px ({target_width_scale}×{target_height_scale} scale)"
                )
                print(f"   Refinement Strength: {hires_fix_strength}")
                upscale_factor = width / hires_fix_start_width
                print(f"   Upscale Factor: {upscale_factor:.2f}x")
        else:
            print(f"\n🔧 High-Res Fix: Disabled")

        # Calculate scale factors
        # Testing: Server seems to multiply by 64 regardless of model latent_size
        # So for SDXL (latent=128) to get 1024px, we need to send scale=16 (not 8)
        # Because server does: 16 * 64 = 1024
        if latent_size == 128:
            # SDXL: double the scale to compensate for server using 64x multiplier
            scale_width = width // 64  # Use 64 instead of 128
            scale_height = height // 64
            print(
                f"   → Scale Factors: {scale_width}x{scale_height} (SDXL workaround: {width}÷64 = {scale_width})"
            )
        else:
            # SD 1.5, FLUX, etc: normal calculation
            scale_width = width // latent_size
            scale_height = height // latent_size
            print(
                f"   → Scale Factors: {scale_width}x{scale_height} ({width}÷{latent_size} = {scale_width})"
            )

        # Handle seed: -1 or None means random
        if seed is None or seed == -1:
            actual_seed = random_module.randint(0, 2**32 - 1)
        else:
            actual_seed = seed

        seed_display = "random" if (seed is None or seed == -1) else str(seed)

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
        for lora_offset in reversed(
            lora_offsets
        ):  # Reverse for FlatBuffer prepend order
            builder.PrependUOffsetTRelative(lora_offset)
        loras_vector = builder.EndVector()

        # Build main configuration
        GenerationConfiguration.Start(builder)
        GenerationConfiguration.AddId(builder, 0)
        GenerationConfiguration.AddStartWidth(
            builder, scale_width
        )  # SCALE FACTOR not pixels!
        GenerationConfiguration.AddStartHeight(
            builder, scale_height
        )  # SCALE FACTOR not pixels!

        # SDXL conditioning (required for SDXL models with latent_size=128)
        # These tell SDXL what resolution it should target
        if latent_size == 128:  # SDXL models
            GenerationConfiguration.AddOriginalImageWidth(builder, width)
            GenerationConfiguration.AddOriginalImageHeight(builder, height)
            GenerationConfiguration.AddTargetImageWidth(builder, width)
            GenerationConfiguration.AddTargetImageHeight(builder, height)
            print(
                f"   → SDXL conditioning: Original={width}x{height}, Target={width}x{height}"
            )

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
        GenerationConfiguration.AddClipSkip(
            builder, clip_skip
        )  # From preset (Pony needs 2!)
        GenerationConfiguration.AddShift(
            builder, final_shift
        )  # Uses calculated shift if res_dependent_shift enabled
        GenerationConfiguration.AddControls(builder, controls_vector)
        GenerationConfiguration.AddLoras(builder, loras_vector)

        # Note: ResolutionDependentShift is calculated CLIENT-SIDE above and applied to final_shift
        # HiresFix parameters (when enabled) - use scale units for FlatBuffer
        GenerationConfiguration.AddHiresFix(builder, hires_fix)
        if hires_fix and hires_fix_start_width_scale > 0:
            GenerationConfiguration.AddHiresFixStartWidth(
                builder, hires_fix_start_width_scale
            )
        if hires_fix and hires_fix_start_height_scale > 0:
            GenerationConfiguration.AddHiresFixStartHeight(
                builder, hires_fix_start_height_scale
            )
        if hires_fix:
            GenerationConfiguration.AddHiresFixStrength(builder, hires_fix_strength)

        # Performance optimizations
        GenerationConfiguration.AddTeaCache(builder, tea_cache)

        # TCD sampler parameter
        GenerationConfiguration.AddStochasticSamplingGamma(builder, tcd_gamma)

        config = GenerationConfiguration.End(builder)
        builder.Finish(config)
        config_bytes = bytes(builder.Output())

        # Create gRPC request
        request = imageService_pb2.ImageGenerationRequest(
            prompt=prompt,
            negativePrompt=negative_prompt if negative_prompt else "",
            configuration=config_bytes,
            scaleFactor=1,
            user="MuddleMeThis",
            device=imageService_pb2.LAPTOP,
            chunked=False,
        )

        status += "\n📡 Sending request to server...\n"

        # Initialize progress
        progress(0, desc="Starting generation...")

        # Generate!
        generated_images = []
        current_step = 0

        for response in state.grpc_client.stub.GenerateImage(request):
            # Handle progress updates
            if response.HasField("currentSignpost"):
                signpost = response.currentSignpost
                if signpost.HasField("sampling"):
                    current_step = signpost.sampling.step
                    progress_pct = current_step / steps
                    progress(
                        progress_pct, desc=f"Sampling: step {current_step}/{steps}"
                    )
                elif signpost.HasField("textEncoded"):
                    progress(0.05, desc="Text encoded")
                elif signpost.HasField("imageEncoded"):
                    progress(0.95, desc="Image encoded")
                elif signpost.HasField("imageDecoded"):
                    progress(0.98, desc="Image decoded")

            # Handle preview images (live preview during sampling)
            # Yield image only, gr.update() leaves status/progress untouched
            if (
                live_preview
                and response.HasField("previewImage")
                and response.previewImage
            ):
                preview_pil = decode_preview(response.previewImage)
                if preview_pil is not None:
                    yield preview_pil, gr.update()

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
            metadata.add_text("comment", prompt)
            metadata.add_text(
                "negative_prompt", negative_prompt if negative_prompt else ""
            )
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

            # Add Draw Things compatible XMP metadata
            xmp_metadata = build_draw_things_xmp(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else "",
                model=model_file,
                width=width,
                height=height,
                steps=steps,
                sampler_name=sampler_name,
                cfg_scale=cfg_scale,
                seed=actual_seed,
                shift=final_shift,
                lora1=lora1,
                lora1_weight=lora1_weight,
                lora2=lora2,
                lora2_weight=lora2_weight,
                strength=1.0,
                seed_mode=seed_mode,
                clip_skip=clip_skip,
            )
            metadata.add_text("XML:com.adobe.xmp", xmp_metadata)
            metadata.add_text("xmp", xmp_metadata)

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

            yield pil_image, final_status
        else:
            yield None, status + "❌ No images generated"

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        yield (
            None,
            f"❌ Error during generation:\n{str(e)}\n\nDetails:\n{error_details}",
        )


def _save_video_from_tensors(
    frames: List[bytes],
    output_path: str,
    fps: int = 24,
    audio: Optional[bytes] = None,
    audio_sample_rate: int = 44100,
) -> str:
    """Assemble decoded Draw Things tensor frames into an MP4 video.

    imageio's imread cannot read raw tensor chunks, so we decode each frame
    with tensor_to_pil() and write numpy arrays via imageio. Audio muxing
    (sanitization + WAV intermediate + ffmpeg) is delegated to the connector.
    """
    import imageio
    import numpy as np

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame_bytes in frames:
        pil_frame = tensor_to_pil(frame_bytes)
        # Ensure RGB for consistent video encoding
        if pil_frame.mode != "RGB":
            pil_frame = pil_frame.convert("RGB")
        writer.append_data(np.array(pil_frame))
    writer.close()

    if audio:
        try:
            DrawThingsClient.mux_audio_into_video(
                video_path=str(output),
                audio=audio,
                audio_sample_rate=audio_sample_rate,
                video_duration=len(frames) / max(fps, 1),
            )
        except Exception as e:
            print(f"Warning: Could not mux audio: {e}")

    return str(output)


def generate_video(
    prompt: str,
    model: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler_name: str,
    seed: int,
    negative_prompt: str,
    num_frames: int,
    fps: int,
    shift: float,
    hires_fix: bool = False,
    hires_fix_width: int = 640,
    hires_fix_height: int = 384,
    hires_fix_strength: float = 0.7,
    start_image=None,
    progress=gr.Progress(),
):
    """Generate a video using Draw Things gRPC with optional starting image.

    Yields (frame_preview, status_message) tuples for Gradio streaming.
    """
    if not state.grpc_client:
        yield None, "❌ gRPC not initialized. Configure in Settings tab first."
        return

    if not prompt:
        yield None, "❌ No prompt provided"
        return

    if not model:
        yield None, "❌ No model selected"
        return

    model_file = state.model_name_to_file.get(model, model)

    # LTX video models have a fixed 25 fps output regardless of UI setting
    is_ltx = "ltx" in model.lower() if model else False
    actual_fps = 25 if is_ltx else fps

    try:
        # Use the new ImageGenerationConfig to leverage video fields
        sampler_id = SAMPLERS.get(sampler_name, 0)
        actual_seed = seed if seed is not None and seed != -1 else random_module.randint(0, 2**32 - 1)

        config = ImageGenerationConfig(
            model=model_file,
            steps=steps,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            scheduler=sampler_name,
            seed=actual_seed,
            seed_mode=2,
            clip_skip=1,
            shift=shift,
            batch_count=1,
            batch_size=1,
            num_frames=num_frames,
            fps_id=actual_fps,
            motion_bucket_id=127,
            compression_artifacts=0,
            hires_fix=hires_fix,
            hires_fix_start_width=hires_fix_width // 64,
            hires_fix_start_height=hires_fix_height // 64,
            hires_fix_strength=hires_fix_strength,
        )

        status = (
            f"🎬 Generating video...\n"
            f"Prompt: {prompt[:80]}...\n"
            f"Model: {model}\n"
            f"Size: {width}x{height}, Frames: {num_frames}, FPS: {actual_fps}\n"
            f"Seed: {actual_seed}"
        )
        if hires_fix:
            status += (
                f"\n🎬 Hires fix ON: {hires_fix_width}x{hires_fix_height} "
                f"→ {width}x{height} (strength {hires_fix_strength})"
            )
        if is_ltx and fps != actual_fps:
            status += f"\n⚠️ LTX uses fixed {actual_fps} fps; ignoring UI setting {fps}"
        yield None, status

        input_image = None
        if start_image is not None:
            try:
                img = Image.fromarray(start_image).convert("RGB")
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                input_image = img
                status += "\n📎 Using starting image as first frame"
                yield None, status
            except Exception as e:
                print(f"Warning: Could not process starting image: {e}")

        result = state.grpc_client.generate_media(
            prompt=prompt,
            config=config,
            negative_prompt=negative_prompt,
            input_image=input_image,
            reference_images=None,
            progress_callback=lambda stage, step: progress(step / max(steps, 1), desc=stage),
        )

        if not result.images:
            yield None, status + "\n❌ No video frames generated"
            return

        # Save frames and assemble video
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_short = "_".join(prompt.split()[:3]).replace("/", "-")[:30]
        video_path = output_dir / f"{timestamp}_{model.replace(' ', '_')}_{prompt_short}_s{actual_seed}.mp4"

        if result.audio:
            import struct as _struct
            first = result.audio[0]
            n = len(result.audio)
            sizes = [len(c) for c in result.audio]
            if len(first) >= 4:
                magic = _struct.unpack_from("<I", first, 0)[0]
                status += (
                    f"\n[AudioDebug] {n} chunk(s), sizes={sizes}, "
                    f"magic={magic} "
                    f"({'CCV tensor' if magic == 1012247 else 'raw/PCM'})"
                )

        # Decode audio chunks via the connector (CCV tensor blobs with fpzip
        # float32 payload, same wrapper as video frames). Decode each chunk
        # separately BEFORE joining so headers do not pollute the waveform.
        try:
            audio_bytes = state.grpc_client.decode_audio(result.audio)
        except RuntimeError as e:
            yield None, f"❌ Audio decode error: {e}"
            return
        except Exception as e:
            print(f"Warning: skipping malformed audio chunk ({e})")
            audio_bytes = None

        saved_video = _save_video_from_tensors(
            frames=result.images,
            output_path=str(video_path),
            fps=actual_fps,
            audio=audio_bytes,
            audio_sample_rate=44100,
        )

        final_status = (
            status
            + f"\n✅ Video complete!\n"
            f"💾 Saved: {video_path.name}\n"
            f"🎞️  Frames: {len(result.images)}"
        )
        if audio_bytes:
            final_status += "\n🔊 Audio included"

        # Return first frame as preview
        preview_image = tensor_to_pil(result.images[0]) if result.images else None
        yield preview_image, final_status

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        yield None, f"❌ Error during video generation:\n{str(e)}\n\nDetails:\n{error_details}"


def edit_image(
    input_image,
    instruction: str,
    model: str,
    steps: int,
    cfg_scale: float,
    sampler_name: str,
    strength: float,
    lora1: str,
    lora1_weight: float,
    lora2: str,
    lora2_weight: float,
    negative_prompt: str,
    seed: int,
    clip_skip: int,
    shift: float,
    res_dependent_shift: bool,
    tcd_gamma: float,
    live_preview: bool = False,
    progress=gr.Progress(),
) -> Tuple[any, str]:
    """Edit an image using AI instructions (generator for live preview)"""
    if not state.grpc_client:
        yield None, "❌ gRPC not initialized. Configure in Settings tab first."
        return

    if input_image is None:
        yield None, "❌ No input image provided"
        return

    if not instruction or not instruction.strip():
        yield None, "❌ No edit instruction provided"
        return

    if not model:
        yield None, "❌ No model selected"
        return

    import math

    start_time = time.time()

    try:
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(input_image.astype("uint8"), "RGB")
        original_size = pil_img.size

        # Translate display name to actual filename
        model_file = state.model_name_to_file.get(model, model)

        status = f"🎨 Editing image...\n\n"
        status += f"📝 Instruction: {instruction}\n"
        status += f"🤖 Model: {model}\n"
        status += f"⚙️  Steps: {steps}, CFG: {cfg_scale}\n"
        status += f"🎲 Sampler: {sampler_name}\n"
        status += f"💪 Strength: {strength}\n"
        status += f"📐 Input: {original_size[0]}×{original_size[1]} pixels\n"

        # Cap to max resolution (edit models work at 1024-1536px, not 4K)
        MAX_EDIT_PIXELS = 2048
        longest_side = max(pil_img.width, pil_img.height)
        if longest_side > MAX_EDIT_PIXELS:
            scale_down = MAX_EDIT_PIXELS / longest_side
            new_w = int(pil_img.width * scale_down)
            new_h = int(pil_img.height * scale_down)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            status += (
                f"📐 Downscaled to: {new_w}×{new_h} pixels (max {MAX_EDIT_PIXELS}px)\n"
            )

        # Round to nearest 64 pixels to match model requirements
        target_width = ((pil_img.width + 32) // 64) * 64
        target_height = ((pil_img.height + 32) // 64) * 64

        if pil_img.size != (target_width, target_height):
            status += f"📐 Aligned to: {target_width}×{target_height} pixels (64-pixel aligned)\n\n"
            pil_img = pil_img.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )
        else:
            status += f"📐 Image already 64-pixel aligned\n\n"

        progress(0.1, desc="Encoding input image...")

        # Handle seed (-1 = random)
        actual_seed = seed if seed >= 0 else random_module.randint(0, 2**32 - 1)

        # Calculate resolution-dependent shift if enabled (FLUX models)
        if res_dependent_shift:
            resolution_factor = (target_width * target_height) / 256
            final_shift = math.exp(
                ((resolution_factor - 256) * (1.15 - 0.5) / (4096 - 256)) + 0.5
            )
            status += f"📐 Resolution-dependent shift: {final_shift:.2f} (calculated from {target_width}×{target_height})\n"
        else:
            final_shift = shift

        # Build LoRA list
        loras = []
        for lora_name, lw in [(lora1, lora1_weight), (lora2, lora2_weight)]:
            if lora_name and lora_name != "None" and lora_name.strip():
                if hasattr(state, "lora_name_to_file") and state.lora_name_to_file:
                    lora_file = state.lora_name_to_file.get(lora_name, lora_name)
                else:
                    lora_file = lora_name
                loras.append(LoRAConfig(file=lora_file, weight=lw))
                status += f"🎨 LoRA: {lora_name} (weight: {lw})\n"

        status += f"⚙️ Seed: {actual_seed}, Shift: {final_shift:.3f}, CLIP Skip: {clip_skip}\n\n"

        progress(0.2, desc="Building configuration...")

        # Build generation config using the client API (for FlatBuffer serialization)
        gen_config = ImageGenerationConfig(
            model=model_file,
            steps=steps,
            width=target_width,
            height=target_height,
            cfg_scale=cfg_scale,
            scheduler=sampler_name,
            seed=actual_seed,
            strength=strength,
            clip_skip=clip_skip,
            shift=final_shift,
            image_guidance_scale=1.5,
            original_image_width=target_width,
            original_image_height=target_height,
            target_image_width=target_width,
            target_image_height=target_height,
            stochastic_sampling_gamma=tcd_gamma,
            resolution_dependent_shift=res_dependent_shift,
            loras=loras,
        )

        # Serialize config to FlatBuffer bytes
        config_bytes = gen_config.to_flatbuffer()

        # Encode input image using client's encoder
        image_tensor = state.grpc_client._encode_image(
            pil_img, target_width, target_height
        )
        image_hash = hashlib.sha256(image_tensor).digest()

        status += "📡 Sending to server...\n"
        progress(0.3, desc="Sending image + instruction...")

        # Build gRPC request with image (content-addressable via hash)
        request = imageService_pb2.ImageGenerationRequest(
            prompt=instruction,
            negativePrompt=negative_prompt if negative_prompt else "",
            configuration=config_bytes,
            scaleFactor=1,
            user="MuddleMeThis",
            device=imageService_pb2.LAPTOP,
            chunked=True,
            image=image_hash,
            contents=[image_tensor],
        )

        # Stream response with live preview support
        generated_images = []
        image_chunks = []
        image_was_encoded = False
        current_step = 0

        for response in state.grpc_client.stub.GenerateImage(request):
            # Handle progress signposts
            if response.HasField("currentSignpost"):
                signpost = response.currentSignpost
                if signpost.HasField("sampling"):
                    current_step = signpost.sampling.step
                    progress(
                        0.3 + (current_step / steps) * 0.6,
                        desc=f"Editing: step {current_step}/{steps}",
                    )
                elif signpost.HasField("textEncoded"):
                    progress(0.35, desc="Text encoded")
                elif signpost.HasField("imageEncoded"):
                    image_was_encoded = True
                    status += "✅ Server received input image\n"
                    progress(0.40, desc="Input image encoded")
                elif signpost.HasField("imageDecoded"):
                    progress(0.98, desc="Result decoded")

            # Handle preview images (live preview during sampling)
            # Yield image only, gr.update() leaves status/progress untouched
            if (
                live_preview
                and response.HasField("previewImage")
                and response.previewImage
            ):
                preview_pil = decode_preview(response.previewImage)
                if preview_pil is not None:
                    yield preview_pil, gr.update()

            # Handle chunked responses
            if response.generatedImages:
                for img_data in response.generatedImages:
                    image_chunks.append(img_data)

                if response.chunkState == imageService_pb2.LAST_CHUNK:
                    if len(image_chunks) > 1:
                        combined = b"".join(image_chunks)
                        generated_images.append(combined)
                    elif len(image_chunks) == 1:
                        generated_images.append(image_chunks[0])
                    image_chunks = []

        if generated_images:
            progress(0.99, desc="Decoding result...")

            edited_pil = tensor_to_pil(generated_images[0])
            elapsed = time.time() - start_time

            # Save to outputs
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_instruction = instruction[:50].replace(" ", "_").replace("/", "_")
            filename = f"edit_{timestamp}_{safe_instruction}.png"
            filepath = outputs_dir / filename

            metadata = PngInfo()
            metadata.add_text("prompt", instruction)
            metadata.add_text("comment", instruction)
            metadata.add_text("strength", str(strength))

            # Add Draw Things compatible XMP metadata for edits
            edit_xmp = build_draw_things_xmp(
                prompt=instruction,
                negative_prompt="",
                model=model,
                width=width,
                height=height,
                steps=steps,
                sampler_name=sampler_name,
                cfg_scale=cfg_scale,
                seed=actual_seed,
                shift=final_shift,
                strength=strength,
                seed_mode=seed_mode,
                clip_skip=clip_skip,
            )
            metadata.add_text("XML:com.adobe.xmp", edit_xmp)
            metadata.add_text("xmp", edit_xmp)
            metadata.add_text("model", model)
            metadata.add_text("steps", str(steps))
            metadata.add_text("cfg_scale", str(cfg_scale))
            metadata.add_text("sampler", sampler_name)
            metadata.add_text("seed", str(actual_seed))
            metadata.add_text("shift", str(final_shift))
            metadata.add_text("edit_time", f"{elapsed:.1f}s")
            edited_pil.save(filepath, pnginfo=metadata)

            final_status = status + f"\n✅ Edit complete in {elapsed:.1f}s!\n"
            final_status += f"💾 Saved: {filename}\n"

            if not image_was_encoded:
                final_status += "\n⚠️ WARNING: Server didn't encode input image!\n"
                final_status += (
                    "   This means it generated from scratch, not editing.\n"
                )
                final_status += "   → Check model is an edit model (Qwen Edit, Flux Kontext, etc.)\n"

            progress(1.0, desc="Complete!")
            yield edited_pil, final_status
        else:
            yield None, status + "❌ No images generated"

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        yield None, f"❌ Error during editing:\n{str(e)}\n\nDetails:\n{error_details}"


def check_for_updates() -> Tuple[str, str]:
    """
    Check if updates are available from GitHub.

    Returns:
        Tuple[str, str]: (status_message, version_info)
    """
    try:
        # Check if .git directory exists
        git_dir = Path(__file__).parent / ".git"
        if not git_dir.exists():
            return (
                "❌ Not installed via git clone.\n\n"
                "To enable auto-updates, please reinstall using:\n"
                "git clone https://github.com/AlexTheStampede/MuddleMeThis.git",
                f"Current Version: {APP_VERSION}",
            )

        # Fetch latest changes from remote
        result = subprocess.run(
            ["git", "fetch", "origin", "main"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).parent,
        )

        if result.returncode != 0:
            return (
                f"❌ Unable to check for updates.\n\n"
                f"Error: {result.stderr}\n\n"
                f"Please check your internet connection.",
                f"Current Version: {APP_VERSION}",
            )

        # Get local commit hash
        local_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
        local_commit = local_result.stdout.strip()

        # Get remote commit hash
        remote_result = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
        remote_commit = remote_result.stdout.strip()

        # Compare commits
        if local_commit == remote_commit:
            return (
                "✅ Already up to date!\n\n"
                "You have the latest version of MuddleMeThis.",
                f"Current Version: {APP_VERSION} | Status: Up to date",
            )

        # Get commit log to show what's new
        log_result = subprocess.run(
            ["git", "log", "--oneline", "HEAD..origin/main"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )

        commit_count = (
            len(log_result.stdout.strip().split("\n"))
            if log_result.stdout.strip()
            else 0
        )
        commit_list = log_result.stdout.strip()

        status_msg = f"✅ Updates available!\n\n"
        status_msg += f"New commits ({commit_count}):\n{commit_list}\n\n"
        status_msg += "Click 'Update Now' to install updates."

        return (
            status_msg,
            f"Current Version: {APP_VERSION} | Updates: {commit_count} commits available",
        )

    except subprocess.TimeoutExpired:
        return (
            "❌ Request timed out.\n\n"
            "Please check your internet connection and try again.",
            f"Current Version: {APP_VERSION}",
        )
    except Exception as e:
        return (
            f"❌ Error checking for updates:\n{str(e)}",
            f"Current Version: {APP_VERSION}",
        )


def apply_update() -> str:
    """
    Apply updates by running git pull.

    Returns:
        str: Status message
    """
    try:
        # Check if .git directory exists
        git_dir = Path(__file__).parent / ".git"
        if not git_dir.exists():
            return (
                "❌ Not installed via git clone.\n\n"
                "To enable auto-updates, please reinstall using:\n"
                "git clone https://github.com/AlexTheStampede/MuddleMeThis.git"
            )

        # Check for uncommitted changes (excluding settings/config.json which is gitignored)
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )

        # Filter out gitignored files (settings/config.json)
        uncommitted_changes = [
            line
            for line in status_result.stdout.strip().split("\n")
            if line and "settings/config.json" not in line
        ]

        if uncommitted_changes:
            files_list = "\n".join(uncommitted_changes)
            return (
                f"❌ Update blocked: Local changes detected.\n\n"
                f"Modified files:\n{files_list}\n\n"
                f"Please backup your changes and resolve conflicts before updating.\n"
                f"Or run manually: git stash && git pull && git stash pop"
            )

        # Perform git pull
        pull_result = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent,
        )

        if pull_result.returncode != 0:
            return (
                f"❌ Update failed!\n\n"
                f"Error: {pull_result.stderr}\n\n"
                f"You may need to resolve conflicts manually.\n"
                f"Run: git pull origin main"
            )

        # Success!
        return (
            "✅ Update complete!\n\n"
            f"Output:\n{pull_result.stdout}\n\n"
            f"⚠️ Please restart the application to use the new version:\n"
            f"  - Close this window\n"
            f"  - Run: ./launch.sh (or launch.bat on Windows)\n"
            f"  - Or: python app.py"
        )

    except subprocess.TimeoutExpired:
        return (
            "❌ Update timed out.\n\n"
            "The update process took too long. Please try again or update manually:\n"
            "git pull origin main"
        )
    except Exception as e:
        return f"❌ Error during update:\n{str(e)}\n\nTry updating manually: git pull origin main"


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
                        # Load available expanders
                        expander_names = list(settings.load_expander_prompts().keys())
                        if "default" not in expander_names:
                            expander_names.insert(0, "default")
                        
                        expander_dropdown = gr.Dropdown(
                            label="Expander",
                            choices=expander_names,
                            value="default",
                            info="Choose an expander algorithm (add custom .txt files to settings/expanders/)",
                        )
                        
                        # Aspect ratio dropdown (optional, for Ernie-style expanders)
                        default_aspects = [
                            label for label, _, _ in settings.load_aspect_ratios(1024)
                        ]
                        expand_aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio (Optional)",
                            choices=["(none)"] + default_aspects,
                            value="(none)",
                            info="Required for Ernie expanders, optional for others",
                        )
                        
                        expand_input = gr.Textbox(
                            label="Brief Prompt",
                            placeholder="Enter a short prompt (e.g., 'a peaceful garden')",
                            lines=5,
                        )
                        expand_btn = gr.Button(
                            "🚀 Expand Prompt", variant="primary", size="lg"
                        )

                    with gr.Column(scale=3):
                        expand_output = gr.Textbox(
                            label="Expanded Prompt", lines=15, interactive=True
                        )
                        with gr.Row():
                            expand_send_to_refine = gr.Button(
                                "➡️ Send to Refine", size="sm"
                            )
                            expand_send_to_edit = gr.Button(
                                "🎨 Send to Edit", size="sm"
                            )

                expand_btn.click(
                    fn=expand_prompt_advanced, 
                    inputs=[expand_input, expander_dropdown, expand_aspect_ratio], 
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
                        extract_btn = gr.Button(
                            "🔍 Extract Prompt", variant="primary", size="lg"
                        )

                    with gr.Column(scale=3):
                        extract_output = gr.Textbox(
                            label="Extracted Prompt", lines=12, interactive=True
                        )
                        with gr.Row():
                            extract_send_to_refine = gr.Button(
                                "➡️ Send to Refine", size="sm"
                            )
                            extract_send_to_edit = gr.Button(
                                "🎨 Send to Edit", size="sm"
                            )

                extract_btn.click(
                    fn=extract_prompt, inputs=[extract_image], outputs=extract_output
                )

            # ==================================================================
            # TAB 3: Bofonchio MC's Restyler
            # ==================================================================
            with gr.Tab("🎭 Bofonchio MC's Restyler"):
                gr.Markdown("### Copy the style from any image")
                gr.Markdown(
                    "*Upload an image and get a detailed style description to use in your prompts*"
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        style_image = gr.Image(label="Upload Image", type="numpy")
                        style_btn = gr.Button(
                            "🎨 Analyze Style", variant="primary", size="lg"
                        )

                    with gr.Column(scale=3):
                        style_output = gr.Textbox(
                            label="Style Description", lines=12, interactive=True
                        )
                        with gr.Row():
                            style_send_to_refine = gr.Button(
                                "➡️ Send to Refine", size="sm"
                            )
                            style_send_to_direct = gr.Button(
                                "📝 Send to Direct", size="sm"
                            )

                style_btn.click(
                    fn=copy_style, inputs=[style_image], outputs=style_output
                )

            # ==================================================================
            # TAB 4: Prompt Refinement
            # ==================================================================
            with gr.Tab("✏️ Refine Prompt"):
                gr.Markdown("### Modify an existing prompt with specific instructions")

                with gr.Row():
                    with gr.Column(scale=2):
                        refine_current = gr.Textbox(
                            label="Current Prompt",
                            placeholder="Paste your current prompt here...",
                            lines=6,
                        )
                        refine_instruction = gr.Textbox(
                            label="Refinement Instruction",
                            placeholder="e.g., 'change the hair to red' or 'add sunset lighting'",
                            lines=3,
                        )
                        refine_btn = gr.Button(
                            "🔧 Refine Prompt", variant="primary", size="lg"
                        )

                    with gr.Column(scale=3):
                        refine_output = gr.Textbox(
                            label="Refined Prompt", lines=12, interactive=True
                        )

                refine_btn.click(
                    fn=refine_prompt,
                    inputs=[refine_current, refine_instruction],
                    outputs=refine_output,
                )

            # ==================================================================
            # TAB 5: Direct Mode
            # ==================================================================
            with gr.Tab("✍️ Direct Mode"):
                gr.Markdown("### Write your prompt directly and generate")

                direct_prompt = gr.Textbox(
                    label="Your Prompt",
                    placeholder="Enter your complete prompt...",
                    lines=10,
                )
                gr.Markdown(
                    "*Use the Image Generation section below to create the image*"
                )

            # ==================================================================
            # TAB 6: Edit Image
            # ==================================================================
            with gr.Tab("🎨 Edit Image"):
                gr.Markdown("### Edit an image using AI instructions")
                gr.Markdown(
                    "*Use edit models like **Qwen Image Edit**, **Flux Kontext**, or **Flux Klein** for best results*"
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        edit_image_input = gr.Image(
                            label="Image to Edit",
                            type="numpy",
                            sources=["upload", "clipboard"],
                        )
                        edit_instruction = gr.Textbox(
                            label="Edit Instruction",
                            placeholder="e.g., 'Make it sunset', 'Add snow', 'Change hair to red'",
                            lines=3,
                            interactive=True,
                        )

                        with gr.Accordion("Generation Settings", open=True):
                            edit_model = gr.Dropdown(
                                label="Model",
                                choices=[],
                                value="",
                                interactive=True,
                                allow_custom_value=True,
                                info="Use Qwen Image Edit or similar edit models",
                            )
                            edit_preset = gr.Dropdown(
                                label="Preset",
                                choices=[
                                    "Custom (no preset)"
                                ],  # Will be populated when model selected
                                value="Custom (no preset)",
                                interactive=True,
                            )
                            edit_preset_info = gr.Textbox(
                                label="Preset Info",
                                value="Select a model first to see available presets",
                                interactive=False,
                                lines=2,
                            )
                            with gr.Row():
                                edit_steps = gr.Slider(
                                    1, 100, 28, step=1, label="Steps"
                                )
                                edit_cfg = gr.Slider(
                                    0.0, 20.0, 5.0, step=0.1, label="CFG Scale"
                                )
                            edit_sampler = gr.Dropdown(
                                choices=SAMPLER_NAMES,
                                value=SAMPLER_DEFAULT,
                                label="Sampler",
                            )
                            edit_tcd_gamma = gr.Slider(
                                0.0,
                                1.0,
                                0.3,
                                label="TCD Strategic Stochastic Sampling",
                                step=0.05,
                                visible=False,
                                info="Strategic Stochastic Sampling gamma for TCD sampler (higher = more stochastic)",
                            )
                            edit_strength = gr.Slider(
                                0.0,
                                1.0,
                                1.0,
                                label="Strength",
                                info="How much to modify (1.0=full edit, 0.75=moderate, 0.5=subtle)",
                            )

                            # LoRA Support
                            gr.Markdown("**LoRAs (optional)**")
                            with gr.Row():
                                edit_lora1 = gr.Dropdown(
                                    label="LoRA 1",
                                    choices=["None"],
                                    value="None",
                                    interactive=True,
                                    allow_custom_value=True,
                                    scale=3,
                                )
                                edit_lora1_weight = gr.Slider(
                                    0.0, 2.0, 1.0, step=0.05, label="Weight", scale=1
                                )
                            with gr.Row():
                                edit_lora2 = gr.Dropdown(
                                    label="LoRA 2",
                                    choices=["None"],
                                    value="None",
                                    interactive=True,
                                    allow_custom_value=True,
                                    scale=3,
                                )
                                edit_lora2_weight = gr.Slider(
                                    0.0, 2.0, 1.0, step=0.05, label="Weight", scale=1
                                )

                        with gr.Accordion("Advanced Settings", open=False):
                            edit_negative = gr.Textbox(
                                label="Negative Prompt",
                                value="blurry, low quality, distorted",
                                lines=2,
                            )
                            with gr.Row():
                                edit_seed = gr.Number(
                                    label="Seed (-1 = random)", value=-1, precision=0
                                )
                                edit_clip_skip = gr.Slider(
                                    1, 12, 1, step=1, label="CLIP Skip"
                                )
                            edit_shift = gr.Slider(
                                0.0,
                                10.0,
                                3.0,
                                step=0.1,
                                label="Shift",
                                info="Qwen Edit default: 3.0",
                            )
                            edit_res_shift = gr.Checkbox(
                                value=False,
                                label="Resolution-Dependent Shift",
                                info="Auto-calculate shift based on resolution (for FLUX models)",
                            )

                        edit_btn = gr.Button(
                            "✨ Edit Image", variant="primary", size="lg"
                        )

                    with gr.Column(scale=3):
                        edit_result_image = gr.Image(label="Edited Result")
                        edit_status = gr.Textbox(label="Status", lines=12)

            # ==================================================================
            # TAB 7: Video Generation
            # ==================================================================
            with gr.Tab("🎬 Video"):
                gr.Markdown("### Generate a video from a prompt")
                gr.Markdown(
                    "*Requires a video-capable Draw Things model (e.g., LTX 2.3). Optional starting image for first-frame conditioning.*"
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        video_start_image = gr.Image(
                            label="Starting Image (optional)",
                            type="numpy",
                            sources=["upload", "clipboard"],
                        )

                        # Load video-capable presets for the dropdown
                        video_preset_choices = [name for name, _ in load_video_presets()]
                        video_preset = gr.Dropdown(
                            label="Video Preset",
                            choices=video_preset_choices,
                            value="Custom (no preset)",
                            interactive=True,
                        )
                        video_preset_info = gr.Textbox(
                            label="Preset Info",
                            value="Select a video preset to apply official settings",
                            interactive=False,
                            lines=2,
                        )

                        video_prompt = gr.Textbox(
                            label="Video Prompt",
                            placeholder="Describe the video you want to generate",
                            lines=4,
                        )
                        video_negative = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, distorted, low quality",
                            lines=2,
                        )

                        with gr.Row():
                            video_width = gr.Number(
                                label="Width", value=512, precision=0, step=64
                            )
                            video_height = gr.Number(
                                label="Height", value=512, precision=0, step=64
                            )

                        with gr.Row():
                            video_steps = gr.Slider(
                                1, 100, 16, step=1, label="Steps"
                            )
                            video_cfg = gr.Slider(
                                0.0, 20.0, 1.0, step=0.1, label="CFG Scale"
                            )

                        video_sampler = gr.Dropdown(
                            choices=SAMPLER_NAMES,
                            value=SAMPLER_DEFAULT,
                            label="Sampler",
                        )

                        video_shift = gr.Slider(
                            0.0,
                            10.0,
                            1.0,
                            step=0.1,
                            label="Shift",
                            info="Timestep shift for video generation",
                        )

                        with gr.Row():
                            video_frames = gr.Slider(
                                1, 257, 14, step=1, label="Frames"
                            )
                            video_fps = gr.Slider(
                                1, 60, 25, step=1, label="FPS",
                                info="LTX is fixed at 25 fps"
                            )
                            video_seed = gr.Number(
                                label="Seed (-1 = random)", value=-1, precision=0
                            )

                        video_hires_fix = gr.Checkbox(
                            label="High Resolution Fix (LTX spatial upscaler)",
                            value=False,
                            info="Two-pass latent upscaling. 1st pass must be 1/2 or 2/3 of final resolution.",
                        )
                        with gr.Row():
                            video_hires_fix_width = gr.Number(
                                label="Hires 1st Pass Width",
                                value=640,
                                precision=0,
                                step=64,
                                info="Pixels (preset converts to scale units)",
                            )
                            video_hires_fix_height = gr.Number(
                                label="Hires 1st Pass Height",
                                value=384,
                                precision=0,
                                step=64,
                                info="Pixels (preset converts to scale units)",
                            )
                            video_hires_fix_strength = gr.Slider(
                                0.0,
                                1.0,
                                0.7,
                                step=0.05,
                                label="Hires Fix Strength",
                                info="Second-pass denoising strength",
                            )

                        video_model = gr.Dropdown(
                            label="Model",
                            choices=[],
                            value="",
                            interactive=True,
                            allow_custom_value=True,
                            info="Select a video model from the Draw Things server",
                        )

                        video_btn = gr.Button(
                            "🎬 Generate Video", variant="primary", size="lg"
                        )

                    with gr.Column(scale=3):
                        video_preview = gr.Image(label="First Frame Preview")
                        video_status = gr.Textbox(label="Status", lines=12)

            # ==================================================================
            # TAB 8: Settings
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
                        value=settings.get("llm_provider", "LM Studio"),
                        label="LLM Provider",
                        info="Choose your LLM backend",
                    )
                    llm_server = gr.Textbox(
                        label="LLM Server URL",
                        value=settings.get("llm_server", "http://localhost:1234"),
                        placeholder="LM Studio: http://localhost:1234 | Ollama: http://localhost:11434",
                    )
                    llm_model_dropdown = gr.Dropdown(
                        label="Text Model",
                        choices=[],
                        value=settings.get("llm_model", ""),
                        allow_custom_value=True,
                        interactive=True,
                        info="For prompt expansion and refinement",
                    )
                    llm_vision_model_dropdown = gr.Dropdown(
                        label="Vision Model (for image analysis)",
                        choices=[],
                        value=settings.get("llm_vision_model", ""),
                        allow_custom_value=True,
                        interactive=True,
                        info="Required for 'Extract from Image' tab. Leave empty to use text model if it supports vision.",
                    )
                    llm_connect_btn = gr.Button("Connect to LLM Server")
                    llm_status = gr.Textbox(label="Status", interactive=False, lines=5)

                    llm_connect_btn.click(
                        fn=init_llm,
                        inputs=[
                            llm_server,
                            llm_model_dropdown,
                            llm_vision_model_dropdown,
                            llm_provider,
                        ],
                        outputs=[
                            llm_status,
                            llm_model_dropdown,
                            llm_vision_model_dropdown,
                        ],
                    )

                    # When user selects a model, initialize it
                    def on_llm_model_change(
                        server_url: str, model: str, vision_model: str, provider: str
                    ) -> str:
                        if not model:
                            return "Please select a model"
                        try:
                            provider_map = {
                                "LM Studio": "lm_studio",
                                "Ollama": "ollama",
                            }
                            provider_key = provider_map.get(provider, "lm_studio")

                            settings.update_config(
                                llm_model=model,
                                llm_vision_model=vision_model,
                                llm_provider=provider,
                            )
                            (
                                state.llm_client,
                                state.text_processor,
                                state.vision_processor,
                            ) = create_ai_client(
                                provider=provider_key,
                                base_url=server_url,
                                text_model=model,
                                vision_model=vision_model
                                or model,  # Use text model if no vision model
                            )
                            vision_info = (
                                f" / Vision: {vision_model}" if vision_model else ""
                            )
                            return (
                                f"✅ Models loaded: {model}{vision_info} ({provider})"
                            )
                        except Exception as e:
                            return f"❌ Failed to load model: {str(e)}"

                    llm_model_dropdown.change(
                        fn=on_llm_model_change,
                        inputs=[
                            llm_server,
                            llm_model_dropdown,
                            llm_vision_model_dropdown,
                            llm_provider,
                        ],
                        outputs=llm_status,
                    )

                    llm_vision_model_dropdown.change(
                        fn=on_llm_model_change,
                        inputs=[
                            llm_server,
                            llm_model_dropdown,
                            llm_vision_model_dropdown,
                            llm_provider,
                        ],
                        outputs=llm_status,
                    )

                with gr.Accordion("gRPC Settings", open=True):
                    grpc_server = gr.Textbox(
                        label="gRPC Server Address",
                        value=settings.get("grpc_server", "localhost:7859"),
                        placeholder="localhost:7859",
                    )
                    grpc_connect_btn = gr.Button("Connect to gRPC Server")
                    grpc_status = gr.Textbox(
                        label="Status", interactive=False, lines=10
                    )
                    grpc_model_dropdown = gr.Dropdown(
                        label="Available Models",
                        choices=[],
                        allow_custom_value=True,
                        filterable=True,
                    )
                    grpc_lora_dropdown = gr.Dropdown(
                        label="Available LoRAs",
                        choices=[],
                        allow_custom_value=True,
                        filterable=True,
                    )

                    # Note: We'll wire up the connection button outputs below after creating gen_model and gen_lora

                with gr.Accordion("Updates", open=False):
                    gr.Markdown("""
                    **Git-Based Updates**

                    Check for and install updates directly from GitHub. Requires installation via `git clone`.
                    """)

                    # Version and status display
                    update_version_info = gr.Textbox(
                        label="Version Information",
                        value=f"Current Version: {APP_VERSION}",
                        interactive=False,
                        lines=2,
                    )

                    # Check button
                    update_check_btn = gr.Button("Check for Updates", size="sm")

                    # Status textbox
                    update_status = gr.Textbox(
                        label="Status", interactive=False, lines=8
                    )

                    # Update button
                    update_apply_btn = gr.Button(
                        "Update Now", variant="primary", size="sm"
                    )

                    # Wire up update callbacks
                    update_check_btn.click(
                        fn=check_for_updates,
                        inputs=[],
                        outputs=[update_status, update_version_info],
                    )

                    update_apply_btn.click(
                        fn=apply_update, inputs=[], outputs=[update_status]
                    )

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
                    lines=8,
                )

                gen_model = gr.Dropdown(
                    label="Model",
                    choices=[],
                    value=settings.get("last_used_model", ""),
                    allow_custom_value=True,
                    filterable=True,
                )

                gen_preset = gr.Dropdown(
                    label="Preset",
                    choices=["Custom (no preset)"],
                    value="Custom (no preset)",
                    interactive=True,
                )

                with gr.Row():
                    gen_lora1 = gr.Dropdown(
                        label="LoRA 1 (optional)",
                        choices=["None"],
                        value="None",
                        allow_custom_value=True,
                        filterable=True,
                        scale=3,
                    )
                    gen_lora1_weight = gr.Slider(
                        0.0, 2.0, 1.0, label="Weight", step=0.05, scale=1
                    )

                with gr.Row():
                    gen_lora2 = gr.Dropdown(
                        label="LoRA 2 (optional)",
                        choices=["None"],
                        value="None",
                        allow_custom_value=True,
                        filterable=True,
                        scale=3,
                    )
                    gen_lora2_weight = gr.Slider(
                        0.0, 2.0, 1.0, label="Weight", step=0.05, scale=1
                    )

                with gr.Row():
                    gen_steps = gr.Slider(
                        1, 150, settings.get("default_steps", 16), label="Steps", step=1
                    )
                    gen_cfg = gr.Slider(
                        1.0,
                        20.0,
                        settings.get("default_cfg", 7.0),
                        label="CFG Scale",
                        step=0.5,
                    )

                gen_sampler = gr.Dropdown(
                    choices=SAMPLER_NAMES, value=SAMPLER_DEFAULT, label="Sampler"
                )

                gen_tcd_gamma = gr.Slider(
                    0.0,
                    1.0,
                    0.3,
                    label="TCD Strategic Stochastic Sampling",
                    step=0.05,
                    visible=False,
                    info="Strategic Stochastic Sampling gamma for TCD sampler (higher = more stochastic)",
                )

                gen_clip_skip = gr.Slider(
                    1,
                    12,
                    1,
                    label="CLIP Skip",
                    step=1,
                    info="CLIP layers to skip (1=default, Pony/Illustrious need 2)",
                )

                # Pre-populate with 1024 base resolution defaults
                default_aspects = [
                    label for label, _, _ in settings.load_aspect_ratios(1024)
                ]
                gen_aspect = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=default_aspects,  # Pre-populated, will update when model selected
                    value=settings.get(
                        "default_aspect_ratio",
                        default_aspects[4]
                        if len(default_aspects) > 4
                        else default_aspects[0],
                    ),
                )

                gen_resolution_scale = gr.Dropdown(
                    label="Resolution Scale",
                    choices=["0.5x", "1x", "1.5x", "2x", "2.5x", "3x", "4x"],
                    value="1x",
                    info="Multiply aspect ratio resolution (useful for high-res fix)",
                )

                gen_seed = gr.Number(
                    label="Seed (-1 for random)", value=-1, precision=0
                )

                # Load negative prompt presets
                negative_prompt_presets = settings.load_negative_prompts()
                negative_prompt_choices = sorted(negative_prompt_presets.keys())

                gen_negative_preset = gr.Dropdown(
                    label="Negative Prompt Preset",
                    choices=negative_prompt_choices,
                    value=None,
                    interactive=True,
                    info="Quick select common negative prompts",
                )

                gen_negative = gr.Textbox(
                    label="Negative Prompt", placeholder="What to avoid...", lines=3
                )

                gen_preset_info = gr.Textbox(
                    label="Model Preset Info", interactive=False, lines=2
                )

                # Advanced Settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    gr.Markdown("*Optional advanced generation parameters*")

                    gen_shift = gr.Slider(
                        0.0,
                        10.0,
                        1.0,
                        label="Shift",
                        step=0.1,
                        info="Timestep shift for generation (1.0 is default)",
                    )

                    gen_res_shift = gr.Checkbox(
                        value=False,
                        label="Resolution Dependent Shift",
                        info="Automatically adjust shift based on resolution (calculated client-side)",
                    )

                    gen_seed_mode = gr.Slider(
                        0,
                        5,
                        2,
                        label="Seed Mode",
                        step=1,
                        info="Random seed generation mode (2 is default)",
                    )

                    # High-Res Fix settings
                    gen_hires = gr.Checkbox(
                        value=False,
                        label="Enable High-Res Fix",
                        info="Two-pass generation: low-res composition + high-res refinement (better quality)",
                    )

                    with gr.Row(visible=False) as gen_hires_row:
                        gen_hires_start_width = gr.Number(
                            value=512,
                            label="Start Width (pixels)",
                            info="Starting width for first pass (e.g., 512px for SD 1.5)",
                            step=64,
                        )
                        gen_hires_start_height = gr.Number(
                            value=512,
                            label="Start Height (pixels)",
                            info="Starting height for first pass (e.g., 512px for SD 1.5)",
                            step=64,
                        )
                        gen_hires_strength = gr.Slider(
                            0.0,
                            1.0,
                            0.7,
                            label="Refinement Strength",
                            info="How much to modify in second pass (0.7 recommended)",
                        )

                    # Performance optimizations
                    gen_tea_cache = gr.Checkbox(
                        value=False,
                        label="Enable TeaCache",
                        info="Timestep Embedding Aware Cache - accelerates generation (training-free)",
                    )

                    gen_live_preview = gr.Checkbox(
                        value=False,
                        label="Live Preview",
                        info="Show latent preview during sampling (experimental)",
                    )

                    # Hidden placeholders for removed settings (kept for preset compatibility)
                    gen_cfg_zero = gr.Checkbox(value=False, visible=False)

                gen_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")

                gr.Markdown(
                    "<small>🧙 **JSON-ify for Ideogram 4**: restructures the prompt as Ideogram JSON, often helping avoid 'Image blocked by safety filter'.</small>"
                )
                jsonify_btn = gr.Button("🧙 JSON-ify for Ideogram 4", size="sm")

            with gr.Column(scale=3):
                gen_image = gr.Image(label="Generated Image")
                gen_status = gr.Textbox(label="Generation Status", lines=8)

        # Model selection updates aspect ratios and shows available presets
        gen_model.change(
            fn=on_model_selected,
            inputs=[gen_model],
            outputs=[gen_preset, gen_preset_info, gen_aspect],
        )

        # Preset selection updates all settings
        gen_preset.change(
            fn=on_preset_selected,
            inputs=[gen_preset],
            outputs=[
                gen_steps,
                gen_cfg,
                gen_preset_info,
                gen_sampler,
                gen_shift,
                gen_res_shift,
                gen_seed_mode,
                gen_cfg_zero,
                gen_hires,
                gen_hires_start_width,
                gen_hires_start_height,
                gen_hires_strength,
                gen_clip_skip,
                gen_tea_cache,
                gen_aspect,
            ],
        )

        # Toggle hires fix controls visibility
        gen_hires.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[gen_hires],
            outputs=[gen_hires_row],
        )

        # Toggle TCD gamma slider visibility based on sampler
        gen_sampler.change(
            fn=lambda sampler: gr.update(visible=(sampler == "TCD")),
            inputs=[gen_sampler],
            outputs=[gen_tcd_gamma],
        )

        # Negative prompt preset selection
        gen_negative_preset.change(
            fn=on_negative_prompt_preset_selected,
            inputs=[gen_negative_preset],
            outputs=[gen_negative],
        )

        # JSON-ify prompt for Ideogram 4
        jsonify_btn.click(
            fn=jsonify_prompt_for_ideogram,
            inputs=[gen_prompt, gen_aspect],
            outputs=gen_prompt,
        )

        # Connect outputs to generation
        gen_btn.click(
            fn=generate_image,
            inputs=[
                gen_prompt,
                gen_model,
                gen_lora1,
                gen_lora1_weight,
                gen_lora2,
                gen_lora2_weight,
                gen_steps,
                gen_cfg,
                gen_sampler,
                gen_aspect,
                gen_resolution_scale,
                gen_seed,
                gen_negative,
                gen_shift,
                gen_res_shift,
                gen_seed_mode,
                gen_cfg_zero,
                gen_hires,
                gen_hires_start_width,
                gen_hires_start_height,
                gen_hires_strength,
                gen_clip_skip,
                gen_tea_cache,
                gen_tcd_gamma,
                gen_live_preview,
            ],
            outputs=[gen_image, gen_status],
        )

        # Link outputs from tabs to the generation prompt field
        expand_output.change(lambda x: x, inputs=expand_output, outputs=gen_prompt)
        extract_output.change(lambda x: x, inputs=extract_output, outputs=gen_prompt)
        style_output.change(lambda x: x, inputs=style_output, outputs=gen_prompt)
        refine_output.change(lambda x: x, inputs=refine_output, outputs=gen_prompt)
        direct_prompt.change(lambda x: x, inputs=direct_prompt, outputs=gen_prompt)

        # Send to Refine buttons
        expand_send_to_refine.click(
            lambda x: x, inputs=expand_output, outputs=refine_current
        )
        extract_send_to_refine.click(
            lambda x: x, inputs=extract_output, outputs=refine_current
        )
        style_send_to_refine.click(
            lambda x: x, inputs=style_output, outputs=refine_current
        )

        # Send to Direct buttons
        style_send_to_direct.click(
            lambda x: x, inputs=style_output, outputs=direct_prompt
        )

        # Send to Edit buttons
        expand_send_to_edit.click(
            lambda x: x, inputs=expand_output, outputs=edit_instruction
        )
        extract_send_to_edit.click(
            lambda img, prompt: (img, prompt),
            inputs=[extract_image, extract_output],
            outputs=[edit_image_input, edit_instruction],
        )

        # Edit Image button
        edit_btn.click(
            fn=edit_image,
            inputs=[
                edit_image_input,
                edit_instruction,
                edit_model,
                edit_steps,
                edit_cfg,
                edit_sampler,
                edit_strength,
                edit_lora1,
                edit_lora1_weight,
                edit_lora2,
                edit_lora2_weight,
                edit_negative,
                edit_seed,
                edit_clip_skip,
                edit_shift,
                edit_res_shift,
                edit_tcd_gamma,
                gen_live_preview,
            ],
            outputs=[edit_result_image, edit_status],
        )

        # Edit tab model selection updates preset choices
        edit_model.change(
            fn=lambda model_name: on_model_selected(model_name)[
                0:2
            ],  # Return preset dropdown and info only
            inputs=[edit_model],
            outputs=[edit_preset, edit_preset_info],
        )

        # Toggle TCD gamma slider visibility in Edit tab
        edit_sampler.change(
            fn=lambda sampler: gr.update(visible=(sampler == "TCD")),
            inputs=[edit_sampler],
            outputs=[edit_tcd_gamma],
        )

        # Edit tab preset selection updates settings (only the ones that exist in Edit tab)
        def on_edit_preset_selected(preset_name: str):
            """Apply preset to Edit tab (subset of controls)"""
            result = on_preset_selected(preset_name)
            # Extract: steps, cfg, info, sampler, shift, res_shift, clip_skip
            # Skip: seed_mode, cfg_zero, hires, hires_width, hires_height, hires_strength, tea_cache
            return (
                result[0],
                result[1],
                result[2],
                result[3],
                result[4],
                result[5],
                result[12],
            )  # steps, cfg, info, sampler, shift, res_shift, clip_skip

        edit_preset.change(
            fn=on_edit_preset_selected,
            inputs=[edit_preset],
            outputs=[
                edit_steps,
                edit_cfg,
                edit_preset_info,
                edit_sampler,
                edit_shift,
                edit_res_shift,
                edit_clip_skip,
            ],
        )

        # Wire up gRPC connection to update ALL dropdowns (settings, generation, and edit)
        # This is done here because gen_model, edit_model, etc. are created after grpc_connect_btn
        def init_grpc_all(server_url):
            """Initialize gRPC and return updates for all dropdowns"""
            status, models_dropdown, loras_dropdown = init_grpc(server_url)
            # Return: status, settings_models, settings_loras, gen_models, gen_lora1, gen_lora2,
            #         edit_model, edit_lora1, edit_lora2, video_model
            return (
                status,
                models_dropdown,
                loras_dropdown,
                models_dropdown,
                loras_dropdown,
                loras_dropdown,
                models_dropdown,
                loras_dropdown,
                loras_dropdown,
                models_dropdown,
            )

        # Wire video generation button
        video_btn.click(
            fn=generate_video,
            inputs=[
                video_prompt,
                video_model,
                video_width,
                video_height,
                video_steps,
                video_cfg,
                video_sampler,
                video_seed,
                video_negative,
                video_frames,
                video_fps,
                video_shift,
                video_hires_fix,
                video_hires_fix_width,
                video_hires_fix_height,
                video_hires_fix_strength,
                video_start_image,
            ],
            outputs=[video_preview, video_status],
        )

        # Wire video preset selection
        video_preset.change(
            fn=on_video_preset_selected,
            inputs=[video_preset],
            outputs=[
                video_width,
                video_height,
                video_steps,
                video_cfg,
                video_sampler,
                video_shift,
                video_frames,
                video_hires_fix,
                video_hires_fix_width,
                video_hires_fix_height,
                video_hires_fix_strength,
                video_preset_info,
            ],
        )

        grpc_connect_btn.click(
            fn=init_grpc_all,
            inputs=[grpc_server],
            outputs=[
                grpc_status,
                grpc_model_dropdown,
                grpc_lora_dropdown,
                gen_model,
                gen_lora1,
                gen_lora2,
                edit_model,
                edit_lora1,
                edit_lora2,
                video_model,
            ],
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
        pwa=True,
    )
