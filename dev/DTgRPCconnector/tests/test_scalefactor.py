#!/usr/bin/env python3
"""
Test with scaleFactor=8 hypothesis:
Maybe scaleFactor tells the server the VAE downsampling ratio,
so start_width=512, scaleFactor=8 means latent=512/8=64?
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

# HYPOTHESIS: start_width=output_size, scaleFactor=VAE_downsampling
# Server computes: latent_width = start_width / scaleFactor = 512 / 8 = 64
GenerationConfiguration.Start(builder)
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddStartWidth(builder, 512)   # Output size
GenerationConfiguration.AddStartHeight(builder, 512)  # Output size
# DO NOT set original/target dimensions
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

print('HYPOTHESIS TEST: scaleFactor=8 as VAE downsampling ratio')
print('Config: start_width/height=512, scaleFactor=8')
print('Expected: Server computes latent=512/8=64 internally')

# Create request with scaleFactor=8
request = imageService_pb2.ImageGenerationRequest(
    prompt='a basket full of kittens',
    configuration=config_bytes,
    scaleFactor=8,  # VAE downsampling factor for SD 1.5
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
        print(f'\n🎉🎉🎉 SUCCESS in {elapsed:.1f}s!')
        if elapsed < 15:
            print('✓✓✓ FAST! This is the correct configuration!')
        else:
            print(f'⚠ Still slow ({elapsed:.1f}s)')

        import os
        os.makedirs('output', exist_ok=True)
        with open('output/kitten_scalefactor8.png', 'wb') as f:
            f.write(generated_images[0])
        print('✓ Saved to output/kitten_scalefactor8.png')

except Exception as e:
    elapsed = time.time() - start_time
    print(f'\nError after {elapsed:.1f}s: {e}')
    import traceback
    traceback.print_exc()

client.close()
