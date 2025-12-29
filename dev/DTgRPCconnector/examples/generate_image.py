#!/usr/bin/env python3
"""
Complete example: Generate image with Draw Things gRPC server and decode to PNG.

Usage:
    python generate_image.py "a cute puppy" --output puppy.png --size 512 --steps 20
"""

import argparse
import time
import flatbuffers
import GenerationConfiguration
import imageService_pb2
from drawthings_client import DrawThingsClient, StreamingProgressHandler
from tensor_decoder import save_tensor_image
from model_metadata import ModelMetadata

def generate_image(
    prompt: str,
    server: str = '192.168.2.150:7859',
    output: str = 'output.png',
    negative_prompt: str = '',
    model: str = 'realdream_15sd15_q6p_q8p.ckpt',
    size: int = 512,
    steps: int = 16,
    cfg_scale: float = 5.0,
    seed: int = None,
    sampler: int = 18,  # UniPC AYS
    latent_size: int = None,  # Override auto-detection
):
    """
    Generate an image using Draw Things gRPC server.

    Args:
        prompt: Text description of image to generate
        server: Server address (default: 192.168.2.150:7859)
        output: Output filename (default: output.png)
        negative_prompt: What to avoid in the image
        model: Model checkpoint name
        size: Image size in pixels (e.g., 512, 768, 1024)
        steps: Number of diffusion steps
        cfg_scale: CFG guidance scale
        seed: Random seed (None for random)
        sampler: Sampler type (18=UniPC AYS)
        latent_size: Override latent size (64 for SD1.5, 128 for SDXL, None for auto-detect)
    """

    # Calculate scale factor from pixel size
    # SD 1.5: 512px ÷ 64 latent = 8
    # SD 1.5: 768px ÷ 64 latent = 12
    # SDXL: 1024px ÷ 128 latent = 8
    # FLUX/Qwen/Z-Image: 1024px ÷ 64 latent = 16
    if latent_size is None:
        # Try to fetch from server metadata
        try:
            metadata = ModelMetadata(server)
            model_info = metadata.get_latent_info(model)
            if 'error' not in model_info:
                latent_size = model_info['latent_size']
                print(f"✓ Fetched from server metadata: latent_size={latent_size} ({model_info['version']})")
            else:
                # Fallback to naive detection
                latent_size = 128 if 'xl' in model.lower() else 64
                print(f"⚠️  Server metadata not available, using naive detection: latent_size={latent_size}")
        except Exception as e:
            # Fallback to naive detection
            latent_size = 128 if 'xl' in model.lower() else 64
            print(f"⚠️  Could not fetch metadata ({e}), using naive detection: latent_size={latent_size}")

        print(f"   Use --latent-size to override if incorrect")

    scale = size // latent_size
    
    print(f"Generating {size}×{size} image (scale factor: {scale})")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    if negative_prompt:
        print(f"Negative: {negative_prompt}")
    print(f"Steps: {steps}, CFG: {cfg_scale}, Seed: {seed or 'random'}\n")
    
    # Connect to server
    client = DrawThingsClient(server, insecure=False, verify_ssl=False)
    
    # Build FlatBuffer configuration
    builder = flatbuffers.Builder(512)
    
    # Create strings
    model_offset = builder.CreateString(model)
    
    # Build configuration
    GenerationConfiguration.Start(builder)
    GenerationConfiguration.AddId(builder, 0)
    GenerationConfiguration.AddStartWidth(builder, scale)   # Scale factor!
    GenerationConfiguration.AddStartHeight(builder, scale)  # Not pixel size!
    if seed is not None:
        GenerationConfiguration.AddSeed(builder, seed)
    else:
        import random
        GenerationConfiguration.AddSeed(builder, random.randint(0, 2**32-1))
    GenerationConfiguration.AddSteps(builder, steps)
    GenerationConfiguration.AddGuidanceScale(builder, cfg_scale)
    GenerationConfiguration.AddStrength(builder, 1.0)
    GenerationConfiguration.AddModel(builder, model_offset)
    GenerationConfiguration.AddSampler(builder, sampler)
    GenerationConfiguration.AddBatchCount(builder, 1)
    GenerationConfiguration.AddBatchSize(builder, 1)
    
    config = GenerationConfiguration.End(builder)
    builder.Finish(config)
    config_bytes = bytes(builder.Output())
    
    # Create gRPC request
    request = imageService_pb2.ImageGenerationRequest(
        prompt=prompt,
        negativePrompt=negative_prompt,
        configuration=config_bytes,
        scaleFactor=1,
        user='PythonClient',
        device=imageService_pb2.LAPTOP,
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
                    progress.on_progress('Sampling', signpost.sampling.step)
                elif signpost.HasField('textEncoded'):
                    print('📝 Text encoded')
                elif signpost.HasField('imageEncoded'):
                    print('🎨 Image encoded')
                elif signpost.HasField('imageDecoded'):
                    elapsed = time.time() - start_time
                    print(f'✨ Image decoded ({elapsed:.1f}s)')
            
            # Collect generated images
            if response.generatedImages:
                generated_images.extend(response.generatedImages)
        
        if generated_images:
            elapsed = time.time() - start_time
            print(f'\n🎉 Generation complete in {elapsed:.1f}s!')
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
        description='Generate images with Draw Things gRPC server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # List available models first
  python list_models.py

  # Basic usage
  python generate_image.py "a cute puppy"

  # Specify size and output
  python generate_image.py "sunset over mountains" --size 768 --output sunset.png

  # More steps and negative prompt
  python generate_image.py "portrait of a cat" --steps 30 --negative "blurry, bad quality"

  # SDXL model
  python generate_image.py "futuristic city" --model juggernaut_xl_v9_q6p_q8p.ckpt --size 1024

  # Non-SD model with manual latent size override
  python generate_image.py "Japanese garden" --model z_image_turbo_1.0_q6p.ckpt --size 1024 --steps 8 --latent-size 64
        '''
    )
    
    parser.add_argument('prompt', help='Text description of image to generate')
    parser.add_argument('--server', default='192.168.2.150:7859', help='Server address')
    parser.add_argument('--output', '-o', default='output.png', help='Output filename')
    parser.add_argument('--negative', default='', help='Negative prompt')
    parser.add_argument('--model', default='realdream_15sd15_q6p_q8p.ckpt', help='Model name')
    parser.add_argument('--size', type=int, default=512, help='Image size (512, 768, 1024)')
    parser.add_argument('--steps', type=int, default=16, help='Number of steps')
    parser.add_argument('--cfg', type=float, default=5.0, help='CFG scale')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--latent-size', type=int, choices=[64, 128], default=None,
                       help='Latent grid size (64=SD1.5, 128=SDXL, auto-detect if not specified)')
    
    args = parser.parse_args()
    
    result = generate_image(
        prompt=args.prompt,
        server=args.server,
        output=args.output,
        negative_prompt=args.negative,
        model=args.model,
        size=args.size,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        latent_size=args.latent_size,
    )
    
    if result:
        print(f'\n✅ Success! Image saved to: {result}')
    else:
        print('\n❌ Failed to generate image')
        exit(1)
