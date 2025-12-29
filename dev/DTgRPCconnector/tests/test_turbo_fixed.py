#!/usr/bin/env python3
"""Test with SD 1.5 scale factor (1024 ÷ 64 = 16)"""

import flatbuffers
import GenerationConfiguration
import imageService_pb2
from drawthings_client import DrawThingsClient, StreamingProgressHandler
from tensor_decoder import save_tensor_image
import time

# Try SD 1.5 architecture: 1024 ÷ 64 = 16
scale = 16

print(f"Testing z_image_turbo_1.0_q6p.ckpt")
print(f"Resolution: 1024×1024")
print(f"Scale factor: {scale} (SD 1.5 architecture)\n")

client = DrawThingsClient('192.168.2.150:7859', insecure=False, verify_ssl=False)

builder = flatbuffers.Builder(1024)
model_offset = builder.CreateString('z_image_turbo_1.0_q6p.ckpt')

GenerationConfiguration.Start(builder)
GenerationConfiguration.AddId(builder, 0)
GenerationConfiguration.AddStartWidth(builder, scale)
GenerationConfiguration.AddStartHeight(builder, scale)
GenerationConfiguration.AddSeed(builder, 1930557564)
GenerationConfiguration.AddSteps(builder, 8)
GenerationConfiguration.AddGuidanceScale(builder, 1.0)
GenerationConfiguration.AddStrength(builder, 1.0)
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddSampler(builder, 17)
GenerationConfiguration.AddBatchCount(builder, 1)
GenerationConfiguration.AddBatchSize(builder, 1)
GenerationConfiguration.AddSeedMode(builder, 2)
GenerationConfiguration.AddShift(builder, 3.0)

config = GenerationConfiguration.End(builder)
builder.Finish(config)

request = imageService_pb2.ImageGenerationRequest(
    prompt='a serene Japanese garden with cherry blossoms',
    configuration=bytes(builder.Output()),
    scaleFactor=1,
    user='Test',
    device=imageService_pb2.LAPTOP,
    chunked=False
)

start_time = time.time()
progress = StreamingProgressHandler(8)

try:
    for response in client.stub.GenerateImage(request):
        if response.HasField('currentSignpost'):
            signpost = response.currentSignpost
            if signpost.HasField('sampling'):
                progress.on_progress('Sampling', signpost.sampling.step)
            elif signpost.HasField('textEncoded'):
                print('Text encoded')
        
        if response.generatedImages:
            elapsed = time.time() - start_time
            print(f'\n✅ Done in {elapsed:.1f}s!')
            save_tensor_image(response.generatedImages[0], 'output/turbo_1024.png')
            break
            
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
finally:
    client.close()
