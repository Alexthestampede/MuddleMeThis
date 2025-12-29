#!/usr/bin/env python3
"""
Test with SDXL model (Juggernaut XL 9) using proper SDXL dimensions.
SDXL native resolution is 1024x1024, latent is 128x128 (1024/8).
"""

from drawthings_client import DrawThingsClient
import flatbuffers
import GenerationConfiguration
import imageService_pb2
from drawthings_client import StreamingProgressHandler

# NOTE: Update this with the actual model filename!
# Common variations: juggernaut_xl_v9.safetensors, juggernautXL_v9.safetensors, etc.
SDXL_MODEL = "juggernaut_xl_v9_q6p_q8p.ckpt"

# Create client
client = DrawThingsClient('192.168.2.150:7859', insecure=False, verify_ssl=False)

# Build FlatBuffer config for SDXL
builder = flatbuffers.Builder(512)

# Create string offsets
model_offset = builder.CreateString(SDXL_MODEL)

# Create empty arrays
GenerationConfiguration.StartControlsVector(builder, 0)
controls_offset = builder.EndVector()

GenerationConfiguration.StartLorasVector(builder, 0)
loras_offset = builder.EndVector()

# SDXL configuration:
# - start: latent dimensions (128x128 for 1024x1024 output)
# - original: conditioning for SDXL (set to output size)
# - target: conditioning for SDXL (set to output size)
# BUT let's test with 512x512 output first (64x64 latent)
GenerationConfiguration.Start(builder)
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddStartWidth(builder, 64)    # Latent for 512x512: 512/8 = 64
GenerationConfiguration.AddStartHeight(builder, 64)   # Latent for 512x512: 512/8 = 64
# SDXL conditioning (original/target set to desired output)
GenerationConfiguration.AddOriginalImageWidth(builder, 512)
GenerationConfiguration.AddOriginalImageHeight(builder, 512)
GenerationConfiguration.AddTargetImageWidth(builder, 512)
GenerationConfiguration.AddTargetImageHeight(builder, 512)
GenerationConfiguration.AddSeed(builder, 3300843894)
GenerationConfiguration.AddSteps(builder, 20)  # SDXL typically needs more steps
GenerationConfiguration.AddGuidanceScale(builder, 7.0)  # SDXL typically uses higher CFG
GenerationConfiguration.AddStrength(builder, 1.0)
GenerationConfiguration.AddSampler(builder, 18)  # UniPC ays
GenerationConfiguration.AddBatchCount(builder, 1)
GenerationConfiguration.AddBatchSize(builder, 1)
GenerationConfiguration.AddSeedMode(builder, 2)  # ScaleAlike
GenerationConfiguration.AddClipSkip(builder, 1)
GenerationConfiguration.AddControls(builder, controls_offset)
GenerationConfiguration.AddLoras(builder, loras_offset)
GenerationConfiguration.AddShift(builder, 1.0)
GenerationConfiguration.AddHiresFix(builder, False)
GenerationConfiguration.AddMaskBlur(builder, 2.5)
GenerationConfiguration.AddMaskBlurOutset(builder, 0)
GenerationConfiguration.AddSharpness(builder, 0.0)
GenerationConfiguration.AddTiledDecoding(builder, False)
GenerationConfiguration.AddTiledDiffusion(builder, False)
GenerationConfiguration.AddPreserveOriginalAfterInpaint(builder, True)
GenerationConfiguration.AddCfgZeroStar(builder, False)
GenerationConfiguration.AddCfgZeroInitSteps(builder, 0)
GenerationConfiguration.AddCausalInferencePad(builder, 0)

config = GenerationConfiguration.End(builder)
builder.Finish(config)
config_bytes = bytes(builder.Output())

print(f'Testing SDXL model: {SDXL_MODEL}')
print('Config: start=64 (latent), original=512, target=512')
print('This should help us understand if the dimension fields work differently for SDXL')

# Create request
request = imageService_pb2.ImageGenerationRequest(
    prompt='a basket full of kittens',
    configuration=config_bytes,
    scaleFactor=1,
    user='Test',
    device=imageService_pb2.LAPTOP,
    chunked=False
)

# Generate!
import time
start_time = time.time()
progress = StreamingProgressHandler(20)
try:
    generated_images = []
    for response in client.stub.GenerateImage(request):
        if response.HasField('currentSignpost'):
            signpost = response.currentSignpost
            if signpost.HasField('sampling'):
                progress.on_progress('Sampling', signpost.sampling.step)
            elif signpost.HasField('textEncoded'):
                print('Text Encoded')
            elif signpost.HasField('imageEncoded'):
                print('Image Encoded')
            elif signpost.HasField('imageDecoded'):
                elapsed = time.time() - start_time
                print(f'Image Decoded! ({elapsed:.1f}s)')

        if response.generatedImages:
            generated_images.extend(response.generatedImages)
            elapsed = time.time() - start_time
            print(f'\nGot image: {len(response.generatedImages[0]):,} bytes in {elapsed:.1f}s')

    if generated_images:
        elapsed = time.time() - start_time
        print(f'\n🎉 SUCCESS with SDXL in {elapsed:.1f}s!')

        import os
        os.makedirs('output', exist_ok=True)
        with open('output/kitten_sdxl.png', 'wb') as f:
            f.write(generated_images[0])
        print('✓ Saved to output/kitten_sdxl.png')

except Exception as e:
    elapsed = time.time() - start_time
    print(f'\nError after {elapsed:.1f}s: {e}')
    import traceback
    traceback.print_exc()

client.close()
