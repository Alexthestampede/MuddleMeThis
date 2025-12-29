#!/usr/bin/env python3
"""
Test with NO dimension fields set at all.
Maybe the server computes dimensions automatically from the model?
"""

from drawthings_client import DrawThingsClient
import flatbuffers
import GenerationConfiguration
import imageService_pb2
from drawthings_client import StreamingProgressHandler

# Create client
client = DrawThingsClient('192.168.2.150:7859', insecure=False, verify_ssl=False)

# Build FlatBuffer config with NO dimension fields
builder = flatbuffers.Builder(256)

# Create string offset
model_offset = builder.CreateString('realdream_15sd15_q6p_q8p.ckpt')

# Build configuration - ABSOLUTELY NO DIMENSION FIELDS
GenerationConfiguration.Start(builder)
GenerationConfiguration.AddId(builder, 0)
# DO NOT set start_width/height
# DO NOT set original_width/height
# DO NOT set target_width/height
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

print(f'Config with NO dimension fields (size: {len(config_bytes)} bytes)')
print('Let the server auto-detect dimensions from the model')

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
            print('✓✓✓ FAST! Auto-detection works!')

        import os
        os.makedirs('output', exist_ok=True)
        with open('output/kitten_no_dims.png', 'wb') as f:
            f.write(generated_images[0])
        print('✓ Saved to output/kitten_no_dims.png')

except Exception as e:
    elapsed = time.time() - start_time
    print(f'\nError after {elapsed:.1f}s: {e}')

client.close()
