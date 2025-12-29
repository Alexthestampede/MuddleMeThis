#!/usr/bin/env python3
"""
Simple test with minimal configuration - just start dimensions.
"""

from drawthings_client import DrawThingsClient
import flatbuffers
import GenerationConfiguration
import imageService_pb2
from drawthings_client import StreamingProgressHandler

# Create client
client = DrawThingsClient('192.168.2.150:7859', insecure=False, verify_ssl=False)

# Build minimal FlatBuffer config
builder = flatbuffers.Builder(512)

# Create string offsets
model_offset = builder.CreateString('realdream_15sd15_q6p_q8p.ckpt')

# Create empty arrays
GenerationConfiguration.StartControlsVector(builder, 0)
controls_offset = builder.EndVector()

GenerationConfiguration.StartLorasVector(builder, 0)
loras_offset = builder.EndVector()

# Build configuration - ONLY set start dimensions to 512
GenerationConfiguration.Start(builder)
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddStartWidth(builder, 512)   # Output size (NOT latent!)
GenerationConfiguration.AddStartHeight(builder, 512)  # Output size (NOT latent!)
# DO NOT set original or target dimensions for txt2img
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

print('Testing with SIMPLE configuration: start_width/height = 512 ONLY')
print('No target dimensions - let model handle latent space internally')

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
                print('Image Decoded!')

        if response.generatedImages:
            generated_images.extend(response.generatedImages)
            print(f'\nGot image: {len(response.generatedImages[0]):,} bytes')

    if generated_images:
        print(f'\n🎉 SUCCESS! Fast txt2img generation')
        import os
        os.makedirs('output', exist_ok=True)
        with open('output/kitten_simple.png', 'wb') as f:
            f.write(generated_images[0])
        print('✓ Saved to output/kitten_simple.png')

except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()

client.close()
