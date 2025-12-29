#!/usr/bin/env python3
"""
Test Qwen Edit 2511 with 4-step lightning LoRA, 8 steps, UniPC Trailing
"""

import flatbuffers
import GenerationConfiguration
import LoRA
import imageService_pb2
from drawthings_client import DrawThingsClient, StreamingProgressHandler
from tensor_decoder import save_tensor_image
import time

# Configuration
MODEL = 'qwen_image_edit_2511_q6p.ckpt'
LORA_FILE = 'qwen_image_edit_2511_lightning_4_step_v1.0_lora_f16.ckpt'
LORA_WEIGHT = 1.0  # Full strength
PROMPT = 'a serene zen garden with cherry blossoms and a small stone lantern'
STEPS = 4
SAMPLER = 18  # UniPCTrailing (from the schema)
SIZE = 1024
CFG = 1.0
SEED = 42

print(f"🚀 Testing Qwen Image Edit 2511 with 4-Step Lightning LoRA")
print(f"=" * 70)
print(f"Model: {MODEL}")
print(f"LoRA: {LORA_FILE} (weight={LORA_WEIGHT})")
print(f"Sampler: UniPC Trailing")
print(f"Resolution: {SIZE}×{SIZE}")
print(f"Steps: {STEPS}, CFG: {CFG}, Seed: {SEED}")
print(f"Prompt: {PROMPT}")
print(f"=" * 70 + "\n")

client = DrawThingsClient('192.168.2.150:7859', insecure=False, verify_ssl=False)

# Build FlatBuffer configuration
builder = flatbuffers.Builder(2048)

# Create model string
model_offset = builder.CreateString(MODEL)

# Create LoRA
lora_file_offset = builder.CreateString(LORA_FILE)
LoRA.Start(builder)
LoRA.AddFile(builder, lora_file_offset)
LoRA.AddWeight(builder, LORA_WEIGHT)
lora_offset = LoRA.End(builder)

# Create LoRAs vector
GenerationConfiguration.StartLorasVector(builder, 1)
builder.PrependUOffsetTRelative(lora_offset)
loras_vector = builder.EndVector()

# Build GenerationConfiguration
# For Qwen Image (latent_size=64): scale = 1024 ÷ 64 = 16
scale = 16

GenerationConfiguration.Start(builder)
GenerationConfiguration.AddId(builder, 0)
GenerationConfiguration.AddStartWidth(builder, scale)
GenerationConfiguration.AddStartHeight(builder, scale)
GenerationConfiguration.AddSeed(builder, SEED)
GenerationConfiguration.AddSteps(builder, STEPS)
GenerationConfiguration.AddGuidanceScale(builder, CFG)
GenerationConfiguration.AddStrength(builder, 1.0)
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddSampler(builder, SAMPLER)
GenerationConfiguration.AddBatchCount(builder, 1)
GenerationConfiguration.AddBatchSize(builder, 1)
GenerationConfiguration.AddLoras(builder, loras_vector)
GenerationConfiguration.AddShift(builder, 3.0)  # Qwen uses shift

config = GenerationConfiguration.End(builder)
builder.Finish(config)

# Create gRPC request
request = imageService_pb2.ImageGenerationRequest(
    prompt=PROMPT,
    configuration=bytes(builder.Output()),
    scaleFactor=1,
    user='Test',
    device=imageService_pb2.LAPTOP,
    chunked=False
)

start_time = time.time()
progress = StreamingProgressHandler(STEPS)

try:
    for response in client.stub.GenerateImage(request):
        if response.HasField('currentSignpost'):
            signpost = response.currentSignpost
            if signpost.HasField('sampling'):
                progress.on_progress('Sampling', signpost.sampling.step)
            elif signpost.HasField('textEncoded'):
                print('📝 Text encoded')
            elif signpost.HasField('imageDecoded'):
                elapsed = time.time() - start_time
                print(f'✨ Image decoded ({elapsed:.1f}s)')

        if response.generatedImages:
            elapsed = time.time() - start_time
            print(f'\n🎉 Generation complete in {elapsed:.1f}s!')
            save_tensor_image(response.generatedImages[0], 'output/qwen_lora_test.png')
            print(f'✅ Saved to output/qwen_lora_test.png')
            break

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
finally:
    client.close()
