#!/usr/bin/env python3
from drawthings_client import DrawThingsClient
import flatbuffers
import GenerationConfiguration
import imageService_pb2
from drawthings_client import StreamingProgressHandler
import time

client = DrawThingsClient('192.168.2.150:7859', insecure=False, verify_ssl=False)

builder = flatbuffers.Builder(256)
model_offset = builder.CreateString('realdream_15sd15_q6p_q8p.ckpt')

GenerationConfiguration.Start(builder)
GenerationConfiguration.AddId(builder, 0)
GenerationConfiguration.AddStartWidth(builder, 8)  # Scale factor!
GenerationConfiguration.AddStartHeight(builder, 8)
GenerationConfiguration.AddSeed(builder, 3300843894)
GenerationConfiguration.AddSteps(builder, 16)
GenerationConfiguration.AddGuidanceScale(builder, 5.0)
GenerationConfiguration.AddStrength(builder, 1.0)
GenerationConfiguration.AddModel(builder, model_offset)
GenerationConfiguration.AddSampler(builder, 18)
GenerationConfiguration.AddBatchCount(builder, 1)
GenerationConfiguration.AddBatchSize(builder, 1)

config = GenerationConfiguration.End(builder)
builder.Finish(config)
config_bytes = bytes(builder.Output())

# Try with chunked=True
request = imageService_pb2.ImageGenerationRequest(
    prompt='a cute puppy',
    configuration=config_bytes,
    scaleFactor=1,
    user='Test',
    device=imageService_pb2.LAPTOP,
    chunked=True  # <-- Set to True
)

start_time = time.time()
progress = StreamingProgressHandler(16)
try:
    all_chunks = []
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
                print(f'Image Decoded!')

        if response.generatedImages:
            for img_data in response.generatedImages:
                all_chunks.append(img_data)
                print(f'Received chunk: {len(img_data):,} bytes, chunk_state: {response.chunkState}')

    # Combine all chunks
    if all_chunks:
        full_data = b''.join(all_chunks)
        print(f'\n✅ Got {len(all_chunks)} chunk(s), total: {len(full_data):,} bytes')
        
        # Check what format it is
        if full_data[:8] == b'\x89PNG\r\n\x1a\n':
            print('Format: PNG')
            ext = 'png'
        elif full_data[:3] == b'\xff\xd8\xff':
            print('Format: JPEG')
            ext = 'jpg'
        else:
            print(f'Format: Unknown (first 20 bytes: {full_data[:20].hex()})')
            ext = 'bin'
        
        filename = f'output/puppy_chunked.{ext}'
        with open(filename, 'wb') as f:
            f.write(full_data)
        print(f'Saved to {filename}')
        
        # Try to verify with PIL
        if ext in ['png', 'jpg']:
            from PIL import Image
            img = Image.open(filename)
            print(f'Image dimensions: {img.size}')

except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()

client.close()
