#!/usr/bin/env python3
"""Test Image Turbo model with specific settings"""

import flatbuffers
import GenerationConfiguration
import imageService_pb2
from drawthings_client import DrawThingsClient, StreamingProgressHandler
from tensor_decoder import save_tensor_image
import time

# Settings from the user
settings = {
    "model": "z_image_turbo_1.0_q6p.ckpt",
    "width": 1024,
    "height": 1024,
    "steps": 8,
    "sampler": 17,
    "guidanceScale": 1,
    "strength": 1,
    "seed": 1930557564,
    "seedMode": 2,
    "shift": 3,
    "maskBlur": 1.5,
    "batchSize": 1,
    "batchCount": 1,
    # Additional settings
    "sharpness": 0,
    "maskBlurOutset": 0,
    "tiledDiffusion": False,
    "tiledDecoding": False,
    "hiresFix": False,
    "preserveOriginalAfterInpaint": True,
    "resolutionDependentShift": False,
    "cfgZeroStar": False,
    "cfgZeroInitSteps": 0,
    "causalInferencePad": 0,
}

# Determine architecture and scale factor
# Turbo models are often SDXL-based (128 latent) or SD 1.5-based (64 latent)
# Let's try SDXL first (1024 ÷ 128 = 8)
latent_size = 128  # SDXL
scale_w = settings["width"] // latent_size
scale_h = settings["height"] // latent_size

print(f"Testing {settings['model']}")
print(f"Resolution: {settings['width']}×{settings['height']}")
print(f"Scale factors: {scale_w}×{scale_h}")
print(f"Steps: {settings['steps']} (turbo model!)")
print(f"Sampler: {settings['sampler']}")
print(f"CFG Scale: {settings['guidanceScale']}")
print(f"Shift: {settings['shift']}\n")

# Connect
client = DrawThingsClient('192.168.2.150:7859', insecure=False, verify_ssl=False)

# Build configuration
builder = flatbuffers.Builder(1024)
model_offset = builder.CreateString(settings['model'])

GenerationConfiguration.Start(builder)
GenerationConfiguration.AddId(builder, 0)
GenerationConfiguration.AddStartWidth(builder, scale_w)
GenerationConfiguration.AddStartHeight(builder, scale_h)
GenerationConfiguration.AddSeed(builder, settings['seed'])
GenerationConfiguration.AddSteps(builder, settings['steps'])
GenerationConfiguration.AddGuidanceScale(builder, settings['guidanceScale'])
GenerationConfiguration.AddStrength(builder, settings['strength'])
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddSampler(builder, settings['sampler'])
GenerationConfiguration.AddBatchCount(builder, settings['batchCount'])
GenerationConfiguration.AddBatchSize(builder, settings['batchSize'])
GenerationConfiguration.AddSeedMode(builder, settings['seedMode'])
GenerationConfiguration.AddShift(builder, settings['shift'])
GenerationConfiguration.AddMaskBlur(builder, settings['maskBlur'])
GenerationConfiguration.AddSharpness(builder, settings['sharpness'])
GenerationConfiguration.AddMaskBlurOutset(builder, settings['maskBlurOutset'])

config = GenerationConfiguration.End(builder)
builder.Finish(config)
config_bytes = bytes(builder.Output())

print(f"Configuration size: {len(config_bytes)} bytes\n")

# Create request
request = imageService_pb2.ImageGenerationRequest(
    prompt='a serene Japanese garden with cherry blossoms, koi pond, stone lanterns, peaceful atmosphere',
    negativePrompt='blurry, bad quality, distorted',
    configuration=config_bytes,
    scaleFactor=1,
    user='TurboTest',
    device=imageService_pb2.LAPTOP,
    chunked=False
)

# Generate!
start_time = time.time()
progress = StreamingProgressHandler(settings['steps'])

try:
    generated_images = []
    
    for response in client.stub.GenerateImage(request):
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
                print(f'✨ Decoded ({elapsed:.1f}s)')
        
        if response.generatedImages:
            generated_images.extend(response.generatedImages)
    
    if generated_images:
        elapsed = time.time() - start_time
        print(f'\n🎉 Generation complete in {elapsed:.1f}s!')
        print(f'📦 Size: {len(generated_images[0]):,} bytes')
        
        # Decode and save
        print('🔧 Decoding...')
        save_tensor_image(generated_images[0], 'output/turbo_garden.png')
        
        print(f'\n✅ Success! Turbo model at 1024×1024 in {elapsed:.1f}s!')
        
except Exception as e:
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()
finally:
    client.close()
