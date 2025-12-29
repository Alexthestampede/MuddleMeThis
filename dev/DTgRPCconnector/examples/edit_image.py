#!/usr/bin/env python3
"""
Image editing with Qwen Edit 2511 + 4-step lightning LoRA.

Usage:
    python edit_image.py <input_image.png> "edit prompt"

Example:
    python edit_image.py test_flux_auto.png "it's sunset"
"""

import argparse
import time
import hashlib
from PIL import Image
import io
import numpy as np
import struct
import flatbuffers
import GenerationConfiguration
import LoRA
import imageService_pb2
from drawthings_client import DrawThingsClient, StreamingProgressHandler
from tensor_decoder import save_tensor_image

def load_image_for_editing(image_path: str, target_size: int = 1024) -> bytes:
    """
    Load and prepare an image for editing.

    The server expects pixel-space tensors (not VAE-encoded latents).
    The tensor is compressed with fpzip to reduce size from 6MB+ to ~500KB.

    Args:
        image_path: Path to input image
        target_size: Target size (will resize to square)

    Returns:
        Compressed tensor bytes ready for Draw Things server
    """
    import fpzip

    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to square
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img, dtype=np.uint8)  # Shape: (H, W, 3)
    height, width, channels = img_array.shape

    # Convert from uint8 [0, 255] to float32 [-1, 1]
    # Note: fpzip requires float32, not float16
    float_array = (img_array.astype(np.float32) / 127.0) - 1.0

    # Add batch dimension: (H, W, 3) -> (1, H, W, 3)
    # fpzip expects/produces tensors with batch dimension
    float_array = np.expand_dims(float_array, axis=0)

    # Create tensor header (68 bytes)
    header = bytearray(68)
    magic = 1012247  # fpzip compression magic number

    struct.pack_into('<I', header, 0, magic)      # header[0] = magic
    struct.pack_into('<I', header, 24, height)    # header[6] = height
    struct.pack_into('<I', header, 28, width)     # header[7] = width
    struct.pack_into('<I', header, 32, channels)  # header[8] = channels

    # Compress pixel data with fpzip (requires float32)
    # Shape is now (1, H, W, 3) which matches what fpzip.decompress produces
    pixel_data = fpzip.compress(float_array, order='C')

    return bytes(header) + pixel_data

def edit_image(
    input_image_path: str,
    prompt: str,
    output: str = 'edited_output.png',
    model: str = 'qwen_image_edit_2511_q6p.ckpt',
    lora_file: str = None,  # Optional LoRA
    lora_weight: float = 1.0,
    size: int = 1024,
    steps: int = 4,
    cfg: float = 1.0,
    strength: float = 1.0,  # 1.0 = full edit, 0.5 = subtle edit
    seed: int = None,
    server: str = '192.168.2.150:7859',
):
    """
    Edit an image using Qwen Edit 2511.

    Args:
        input_image_path: Path to input image
        prompt: Edit instruction (e.g., "it's sunset")
        output: Output filename
        model: Qwen Edit model name
        lora_file: Lightning LoRA filename
        lora_weight: LoRA strength (0.0 to 1.0)
        size: Image size
        steps: Number of diffusion steps
        cfg: CFG scale
        strength: Edit strength (1.0 = full edit, lower = more subtle)
        seed: Random seed
        server: Server address
    """

    print(f"🎨 Image Editing with Qwen Edit 2511")
    print(f"=" * 70)
    print(f"Input: {input_image_path}")
    print(f"Prompt: {prompt}")
    print(f"Model: {model}")
    if lora_file:
        print(f"LoRA: {lora_file} (weight={lora_weight})")
    else:
        print(f"LoRA: None")
    print(f"Strength: {strength} (1.0=full edit, 0.5=subtle)")
    print(f"Steps: {steps}, CFG: {cfg}")
    print(f"=" * 70 + "\n")

    # Load and prepare input image
    print("📂 Loading input image...")
    image_bytes = load_image_for_editing(input_image_path, size)
    image_sha256 = hashlib.sha256(image_bytes).digest()
    print(f"   Size: {len(image_bytes):,} bytes")
    print(f"   SHA256: {image_sha256.hex()[:16]}...\n")

    # Connect to server
    client = DrawThingsClient(server, insecure=False, verify_ssl=False)

    # Build FlatBuffer configuration
    builder = flatbuffers.Builder(2048)

    # Create model string
    model_offset = builder.CreateString(model)

    # Create LoRA if specified
    loras_vector = None
    if lora_file:
        lora_file_offset = builder.CreateString(lora_file)
        LoRA.Start(builder)
        LoRA.AddFile(builder, lora_file_offset)
        LoRA.AddWeight(builder, lora_weight)
        lora_offset = LoRA.End(builder)

        # Create LoRAs vector
        GenerationConfiguration.StartLorasVector(builder, 1)
        builder.PrependUOffsetTRelative(lora_offset)
        loras_vector = builder.EndVector()

    # Build GenerationConfiguration
    # Qwen Edit uses latent_size=64, so scale = 1024 ÷ 64 = 16
    scale = size // 64

    if seed is None:
        import random
        seed = random.randint(0, 2**32-1)

    GenerationConfiguration.Start(builder)
    GenerationConfiguration.AddId(builder, 0)
    GenerationConfiguration.AddStartWidth(builder, scale)
    GenerationConfiguration.AddStartHeight(builder, scale)
    GenerationConfiguration.AddSeed(builder, seed)
    GenerationConfiguration.AddSteps(builder, steps)
    GenerationConfiguration.AddGuidanceScale(builder, cfg)
    GenerationConfiguration.AddStrength(builder, strength)  # Edit strength
    GenerationConfiguration.AddModel(builder, model_offset)
    GenerationConfiguration.AddSampler(builder, 18)  # UniPC Trailing
    GenerationConfiguration.AddBatchCount(builder, 1)
    GenerationConfiguration.AddBatchSize(builder, 1)
    if loras_vector is not None:
        GenerationConfiguration.AddLoras(builder, loras_vector)
    GenerationConfiguration.AddShift(builder, 3.0)

    config = GenerationConfiguration.End(builder)
    builder.Finish(config)

    # Create gRPC request with image
    request = imageService_pb2.ImageGenerationRequest(
        image=image_sha256,  # SHA256 reference to input image
        prompt=prompt,
        configuration=bytes(builder.Output()),
        scaleFactor=1,
        user='PythonClient',
        device=imageService_pb2.LAPTOP,
        contents=[image_bytes],  # Actual image data
        chunked=False
    )

    # Generate!
    start_time = time.time()
    progress = StreamingProgressHandler(steps)

    try:
        generated_images = []

        for response in client.stub.GenerateImage(request):
            # Handle progress updates
            if response.HasField('currentSignpost'):
                signpost = response.currentSignpost
                if signpost.HasField('sampling'):
                    progress.on_progress('Editing', signpost.sampling.step)
                elif signpost.HasField('textEncoded'):
                    print('📝 Text encoded')
                elif signpost.HasField('imageEncoded'):
                    print('🎨 Input image encoded')
                elif signpost.HasField('imageDecoded'):
                    elapsed = time.time() - start_time
                    print(f'✨ Image decoded ({elapsed:.1f}s)')

            # Collect generated images
            if response.generatedImages:
                generated_images.extend(response.generatedImages)

        if generated_images:
            elapsed = time.time() - start_time
            print(f'\n🎉 Edit complete in {elapsed:.1f}s!')
            print(f'📦 Received {len(generated_images[0]):,} bytes')

            # Decode and save
            print(f'🔧 Decoding tensor...')
            save_tensor_image(generated_images[0], output)

            return output
        else:
            print('❌ No images generated')
            return None

    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return None
    finally:
        client.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Edit images with Qwen Edit 2511',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic edit
  python edit_image.py input.png "it's sunset"

  # Subtle edit
  python edit_image.py input.png "add snow" --strength 0.5

  # Custom output
  python edit_image.py photo.jpg "make it rainy" --output rainy.png
        '''
    )

    parser.add_argument('input', help='Input image path')
    parser.add_argument('prompt', help='Edit instruction')
    parser.add_argument('--output', '-o', default='edited_output.png', help='Output filename')
    parser.add_argument('--strength', type=float, default=1.0, help='Edit strength (0.0-1.0, default 1.0)')
    parser.add_argument('--steps', type=int, default=4, help='Number of steps (default 4)')
    parser.add_argument('--cfg', type=float, default=1.0, help='CFG scale (default 1.0)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--size', type=int, default=1024, help='Image size (default 1024)')
    parser.add_argument('--server', default='192.168.2.150:7859', help='Server address')
    parser.add_argument('--model', default='qwen_image_edit_2511_q6p.ckpt', help='Edit model to use')
    parser.add_argument('--no-lora', action='store_true', help='Disable lightning LoRA')

    args = parser.parse_args()

    # Use lightning LoRA matching the model version unless --no-lora is specified
    if args.no_lora:
        lora = None
    elif '2511' in args.model:
        lora = 'qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt'
    elif '2509' in args.model:
        lora = 'qwen_image_edit_2509_lightning_8_step_v1.0_lora_f16.ckpt'
    else:
        lora = None  # No default LoRA for other models

    result = edit_image(
        input_image_path=args.input,
        prompt=args.prompt,
        output=args.output,
        model=args.model,
        lora_file=lora,
        size=args.size,
        steps=args.steps,
        cfg=args.cfg,
        strength=args.strength,
        seed=args.seed,
        server=args.server,
    )

    if result:
        print(f'\n✅ Success! Edited image saved to: {result}')
    else:
        print('\n❌ Failed to edit image')
        exit(1)
