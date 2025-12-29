#!/usr/bin/env python3
"""
Absolutely minimal configuration - only required fields from official schema.
"""

from drawthings_client import DrawThingsClient
import flatbuffers
import GenerationConfiguration
import imageService_pb2
from drawthings_client import StreamingProgressHandler

# Create client
client = DrawThingsClient('192.168.2.150:7859', insecure=False, verify_ssl=False)

# Build MINIMAL FlatBuffer config - only required fields
builder = flatbuffers.Builder(256)

# Create string offset
model_offset = builder.CreateString('realdream_15sd15_q6p_q8p.ckpt')

# Build configuration with ONLY core required fields
GenerationConfiguration.Start(builder)
GenerationConfiguration.AddId(builder, 0)
GenerationConfiguration.AddStartWidth(builder, 512)
GenerationConfiguration.AddStartHeight(builder, 512)
GenerationConfiguration.AddSeed(builder, 3300843894)
GenerationConfiguration.AddSteps(builder, 16)
GenerationConfiguration.AddGuidanceScale(builder, 5.0)
GenerationConfiguration.AddStrength(builder, 1.0)
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddSampler(builder, 18)  # UniPC ays
GenerationConfiguration.AddBatchCount(builder, 1)
GenerationConfiguration.AddBatchSize(builder, 1)

config = GenerationConfiguration.End(builder)
builder.Finish(config)
config_bytes = bytes(builder.Output())

print(f'Minimal config size: {len(config_bytes)} bytes')
print('Fields: id, start_width=512, start_height=512, seed, steps, guidance_scale,')
print('        strength, model, sampler, batch_count, batch_size')

# Create request
request = imageService_pb2.ImageGenerationRequest(
    prompt='a basket full of kittens',
    configuration=config_bytes,
    scaleFactor=1,  # No scaling
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

        import os
        os.makedirs('output', exist_ok=True)
        with open('output/kitten_minimal.png', 'wb') as f:
            f.write(generated_images[0])
        print('✓ Saved to output/kitten_minimal.png')

except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()

client.close()
