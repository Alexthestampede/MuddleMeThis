#!/usr/bin/env python3
"""
Test with start dimensions set to LATENT space (64x64) only, no target dimensions.
Let the model's VAE handle the decoding to 512x512 automatically.
"""

from drawthings_client import DrawThingsClient
import flatbuffers
import GenerationConfiguration
import imageService_pb2
from drawthings_client import StreamingProgressHandler

# Create client
client = DrawThingsClient('192.168.2.150:7859', insecure=False, verify_ssl=False)

# Build FlatBuffer config
builder = flatbuffers.Builder(512)

# Create string offsets
model_offset = builder.CreateString('realdream_15sd15_q6p_q8p.ckpt')

# Create empty arrays
GenerationConfiguration.StartControlsVector(builder, 0)
controls_offset = builder.EndVector()

GenerationConfiguration.StartLorasVector(builder, 0)
loras_offset = builder.EndVector()

# Build configuration - start=latent (64), NO target/original dimensions
# SD 1.5 VAE should automatically decode 64x64 latent to 512x512 pixels
GenerationConfiguration.Start(builder)
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddStartWidth(builder, 64)    # Latent: 512/8 = 64
GenerationConfiguration.AddStartHeight(builder, 64)   # Latent: 512/8 = 64
# DO NOT set original or target dimensions - let VAE handle it
GenerationConfiguration.AddSeed(builder, 3300843894)
GenerationConfiguration.AddSteps(builder, 16)
GenerationConfiguration.AddGuidanceScale(builder, 5.0)
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

print('Testing with start=64 (latent) ONLY, no target/original dimensions')
print('Expected: VAE should decode 64x64 latent to 512x512 pixels automatically')

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
progress = StreamingProgressHandler(16)
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
        print(f'\n🎉 SUCCESS in {elapsed:.1f}s!')
        if elapsed < 15:
            print('✓✓✓ FAST generation! This is the right config!')
        else:
            print(f'⚠ Slower than expected ({elapsed:.1f}s > 15s)')

        import os
        os.makedirs('output', exist_ok=True)
        with open('output/kitten_latent_only.png', 'wb') as f:
            f.write(generated_images[0])
        print('✓ Saved to output/kitten_latent_only.png')

except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()

client.close()
