# Third-Party Tool Integration: Direct gRPC Server Access

## Overview

This guide explains how third-party applications can directly interface with the Draw Things gRPC server (gRPCServerCLI). This enables custom clients, automation tools, and integrations to leverage GPU resources without going through the Draw Things app or cloud service.

## Architecture

```
Third-Party Tool → gRPC Server → GPU Processing → Response
         ↑                                            ↓
         └───────────── Direct Streaming ─────────────┘
```

Direct integration provides:
- Full control over generation parameters
- No cloud dependency (local network only)
- Custom authentication via shared secret
- Lower latency (no proxy hop)
- Complete model and LoRA management

## gRPC Protocol Definition

The server implements the `ImageGenerationService` protocol defined in `imageService.proto`:

### Service Methods

```protobuf
service ImageGenerationService {
  // Generate images with streaming progress
  rpc GenerateImage(ImageGenerationRequest)
    returns (stream ImageGenerationResponse);

  // Check which models/LoRAs are available
  rpc FilesExist(FileListRequest)
    returns (FileExistenceResponse);

  // Upload models to server
  rpc UploadFile(stream FileUploadRequest)
    returns (stream UploadResponse);

  // Ping server and get available models
  rpc Echo(EchoRequest)
    returns (EchoReply);

  // Get public key (for proxy mode)
  rpc Pubkey(PubkeyRequest)
    returns (PubkeyResponse);

  // Get compute unit thresholds (for proxy mode)
  rpc Hours(HoursRequest)
    returns (HoursResponse);
}
```

## Connection Setup

### Basic Connection

```python
import grpc
from generated import imageService_pb2, imageService_pb2_grpc

# Connect to server
channel = grpc.insecure_channel('localhost:7859')
stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)

print("Connected to Draw Things gRPC server")
```

### TLS Connection

```python
import grpc
from generated import imageService_pb2, imageService_pb2_grpc

# Load server certificate
with open('server.crt', 'rb') as f:
    cert = f.read()

credentials = grpc.ssl_channel_credentials(cert)
channel = grpc.secure_channel('server.example.com:7859', credentials)
stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)

print("Connected via TLS")
```

### With Shared Secret

```python
import grpc
from generated import imageService_pb2, imageService_pb2_grpc

channel = grpc.insecure_channel('localhost:7859')
stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)

# Include shared secret in requests
shared_secret = "YOUR_SECRET_HERE"

request = imageService_pb2.EchoRequest(
    name="MyClient",
    sharedSecret=shared_secret
)

response = stub.Echo(request)
print(response.message)
```

## Discovering Available Models

### Echo Request (Recommended)

The `Echo` call returns server information including available models:

```python
def get_available_models(stub, shared_secret=None):
    """Get list of available models, LoRAs, and ControlNets"""
    request = imageService_pb2.EchoRequest(name="ModelDiscovery")

    if shared_secret:
        request.sharedSecret = shared_secret

    response = stub.Echo(request)

    # Parse metadata override (JSON encoded)
    import json

    models = []
    loras = []
    controlnets = []
    textual_inversions = []
    upscalers = []

    if response.override.models:
        models = json.loads(response.override.models)

    if response.override.loras:
        loras = json.loads(response.override.loras)

    if response.override.controlNets:
        controlnets = json.loads(response.override.controlNets)

    if response.override.textualInversions:
        textual_inversions = json.loads(response.override.textualInversions)

    if response.override.upscalers:
        upscalers = json.loads(response.override.upscalers)

    return {
        'models': models,
        'loras': loras,
        'controlnets': controlnets,
        'textual_inversions': textual_inversions,
        'upscalers': upscalers,
        'files': list(response.files),
        'server_identifier': response.serverIdentifier
    }

# Usage
available = get_available_models(stub, "my_secret")

print("Available Models:")
for model in available['models']:
    print(f"  - {model['name']} ({model['file']})")

print("\nAvailable LoRAs:")
for lora in available['loras']:
    print(f"  - {lora['name']} ({lora['file']})")
```

Model specification format (JSON):
```json
{
  "name": "FLUX.1 Schnell",
  "file": "flux_1_schnell_f16.ckpt",
  "prefix": "flux_1_",
  "version": "flux1",
  "upcastAttention": false,
  "defaultScale": 8,
  "textEncoder": "clip_vit_l_f16.ckpt",
  "autoencoder": "flux_1_vae_f16.ckpt",
  "t5Encoder": "t5_xxl_encoder_f16.ckpt"
}
```

LoRA specification format (JSON):
```json
{
  "name": "LCM LoRA (SDXL)",
  "file": "lcm_sd_xl_base_1.0_lora_f16.ckpt",
  "prefix": "lcm_",
  "version": "sdxlBase",
  "weight": {
    "value": 1.0,
    "lowerBound": -1.5,
    "upperBound": 2.5
  }
}
```

### FilesExist Request

Check if specific files are downloaded:

```python
def check_files_exist(stub, filenames, compute_hashes=False, shared_secret=None):
    """Check which files exist on the server"""
    request = imageService_pb2.FileListRequest(
        files=filenames
    )

    if compute_hashes:
        request.filesWithHash.extend(filenames)

    if shared_secret:
        request.sharedSecret = shared_secret

    response = stub.FilesExist(request)

    results = []
    for i, filename in enumerate(response.files):
        result = {
            'file': filename,
            'exists': response.existences[i]
        }

        if i < len(response.hashes) and response.hashes[i]:
            result['sha256'] = response.hashes[i].hex()

        results.append(result)

    return results

# Usage
files_to_check = [
    "flux_1_schnell_f16.ckpt",
    "qwen_image_edit_2511_q8p.ckpt",
    "lcm_sd_xl_base_1.0_lora_f16.ckpt"
]

results = check_files_exist(stub, files_to_check, compute_hashes=True)

for r in results:
    status = "✓" if r['exists'] else "✗"
    print(f"{status} {r['file']}")
    if 'sha256' in r:
        print(f"    SHA256: {r['sha256']}")
```

## Generating Images

### Basic Text-to-Image

```python
import hashlib
from io import BytesIO
from PIL import Image

def generate_image(stub, prompt, negative_prompt="", config=None):
    """Generate an image from text prompt"""

    # Default configuration
    if config is None:
        config = {
            'model': 'flux_1_schnell_f16.ckpt',
            'width': 1024,
            'height': 1024,
            'steps': 4,
            'guidanceScale': 0.0,
            'seed': 42,
            'batchCount': 1,
            'sampler': 1  # EULER_A
        }

    # Serialize configuration (FlatBuffer format)
    # Note: You'll need to implement FlatBuffer serialization
    # or use the provided DataModels library
    configuration_bytes = serialize_configuration(config)

    # Create metadata override
    override = imageService_pb2.MetadataOverride()
    # Leave empty to use server's built-in model zoo

    # Create request
    request = imageService_pb2.ImageGenerationRequest(
        prompt=prompt,
        negativePrompt=negative_prompt,
        configuration=configuration_bytes,
        scaleFactor=1,
        override=override,
        user="ThirdPartyClient",
        device=imageService_pb2.LAPTOP,
        chunked=True
    )

    # Generate with streaming
    images = []
    current_chunk = b''

    for response in stub.GenerateImage(request):
        # Handle progress updates
        if response.HasField('currentSignpost'):
            signpost = response.currentSignpost
            if signpost.HasField('sampling'):
                print(f"Sampling step {signpost.sampling.step}")
            elif signpost.HasField('textEncoded'):
                print("Text encoded")
            elif signpost.HasField('imageDecoded'):
                print("Image decoded")

        # Handle preview images
        if response.previewImage:
            # Decompress and display preview
            preview = decompress_tensor(response.previewImage)
            # Display preview...

        # Handle final images
        if response.generatedImages:
            for image_data in response.generatedImages:
                if response.chunkState == imageService_pb2.LAST_CHUNK:
                    # Complete image or final chunk
                    full_data = current_chunk + image_data
                    current_chunk = b''

                    # Decompress tensor
                    tensor = decompress_tensor(full_data)
                    images.append(tensor)

                elif response.chunkState == imageService_pb2.MORE_CHUNKS:
                    # Accumulate chunks
                    current_chunk += image_data

    return images

# Usage
images = generate_image(
    stub,
    prompt="A serene landscape with mountains",
    negative_prompt="blurry, low quality"
)

print(f"Generated {len(images)} images")
```

### Image-to-Image Generation

```python
def img2img(stub, input_image_path, prompt, strength=0.75, config=None):
    """Generate image from input image and prompt"""

    # Load and encode input image
    with Image.open(input_image_path) as img:
        # Convert to tensor and compress
        tensor = image_to_tensor(img)
        compressed = compress_tensor(tensor)

        # Compute SHA256 hash
        hash_digest = hashlib.sha256(compressed).digest()

    # Configuration
    if config is None:
        config = {
            'model': 'flux_1_dev_f16.ckpt',
            'width': img.width,
            'height': img.height,
            'steps': 28,
            'guidanceScale': 3.5,
            'strength': strength,
            'seed': 42
        }
    else:
        config['strength'] = strength

    configuration_bytes = serialize_configuration(config)

    # Create request
    request = imageService_pb2.ImageGenerationRequest(
        image=hash_digest,  # SHA256 reference
        prompt=prompt,
        negativePrompt="blurry, distorted",
        configuration=configuration_bytes,
        scaleFactor=1,
        override=imageService_pb2.MetadataOverride(),
        user="ThirdPartyClient",
        device=imageService_pb2.LAPTOP,
        contents=[compressed],  # Actual image data
        chunked=True
    )

    # Generate
    images = []
    for response in stub.GenerateImage(request):
        # Handle responses (same as txt2img)
        if response.generatedImages and response.chunkState == imageService_pb2.LAST_CHUNK:
            for img_data in response.generatedImages:
                tensor = decompress_tensor(img_data)
                images.append(tensor)

    return images
```

### Inpainting

```python
def inpaint(stub, input_image_path, mask_image_path, prompt, config=None):
    """Inpaint masked region with new content"""

    # Load input image
    with Image.open(input_image_path) as img:
        image_tensor = image_to_tensor(img)
        image_compressed = compress_tensor(image_tensor)
        image_hash = hashlib.sha256(image_compressed).digest()

    # Load mask (white = inpaint, black = keep)
    with Image.open(mask_image_path) as mask_img:
        # Convert to uint8 tensor (0-255)
        mask_tensor = mask_to_tensor(mask_img)
        mask_compressed = compress_tensor(mask_tensor)
        mask_hash = hashlib.sha256(mask_compressed).digest()

    # Configuration
    if config is None:
        config = {
            'model': 'flux_1_dev_f16.ckpt',
            'width': img.width,
            'height': img.height,
            'steps': 28,
            'guidanceScale': 3.5,
            'strength': 1.0,  # Full inpainting
            'imagePriorSteps': 0,  # Don't preserve original
            'maskBlur': 4.0,
            'seed': 42
        }

    configuration_bytes = serialize_configuration(config)

    # Create request
    request = imageService_pb2.ImageGenerationRequest(
        image=image_hash,
        mask=mask_hash,
        prompt=prompt,
        negativePrompt="blurry, low quality",
        configuration=configuration_bytes,
        scaleFactor=1,
        override=imageService_pb2.MetadataOverride(),
        user="ThirdPartyClient",
        device=imageService_pb2.LAPTOP,
        contents=[image_compressed, mask_compressed],
        chunked=True
    )

    # Generate
    images = []
    for response in stub.GenerateImage(request):
        if response.generatedImages and response.chunkState == imageService_pb2.LAST_CHUNK:
            for img_data in response.generatedImages:
                tensor = decompress_tensor(img_data)
                images.append(tensor)

    return images
```

### Using LoRAs

```python
def generate_with_lora(stub, prompt, lora_specs):
    """Generate with LoRA applied

    Args:
        lora_specs: List of dicts with 'file' and 'weight' keys
    """

    # Encode LoRA specifications as JSON
    import json

    loras_json = json.dumps([
        {
            'file': lora['file'],
            'weight': lora.get('weight', 1.0),
            'version': lora.get('version', 'sdxlBase')
        }
        for lora in lora_specs
    ]).encode('utf-8')

    override = imageService_pb2.MetadataOverride(
        loras=loras_json
    )

    config = {
        'model': 'sd_xl_base_1.0_f16.ckpt',
        'width': 1024,
        'height': 1024,
        'steps': 4,  # Using LCM LoRA
        'guidanceScale': 1.5,
        'sampler': 6  # LCM sampler
    }

    configuration_bytes = serialize_configuration(config)

    request = imageService_pb2.ImageGenerationRequest(
        prompt=prompt,
        negativePrompt="",
        configuration=configuration_bytes,
        scaleFactor=1,
        override=override,
        user="ThirdPartyClient",
        device=imageService_pb2.LAPTOP,
        chunked=True
    )

    images = []
    for response in stub.GenerateImage(request):
        if response.generatedImages and response.chunkState == imageService_pb2.LAST_CHUNK:
            for img_data in response.generatedImages:
                images.append(decompress_tensor(img_data))

    return images

# Usage
images = generate_with_lora(
    stub,
    prompt="A futuristic city",
    lora_specs=[
        {'file': 'lcm_sd_xl_base_1.0_lora_f16.ckpt', 'weight': 1.0}
    ]
)
```

### Using ControlNet

```python
def generate_with_controlnet(stub, input_image_path, prompt, controlnet_type="canny"):
    """Generate with ControlNet guidance"""

    # Load and process control image
    with Image.open(input_image_path) as img:
        # Extract control hint (e.g., Canny edges)
        control_tensor = extract_control_hint(img, controlnet_type)
        control_compressed = compress_tensor(control_tensor)
        control_hash = hashlib.sha256(control_compressed).digest()

    # Create hint
    hint = imageService_pb2.HintProto(
        hintType=controlnet_type,  # "canny", "depth", "pose", etc.
        tensors=[
            imageService_pb2.TensorAndWeight(
                tensor=control_hash,
                weight=1.0
            )
        ]
    )

    # ControlNet specification
    import json
    controlnets_json = json.dumps([{
        'file': f'controlnet_v1.1_{controlnet_type}_f16.ckpt',
        'weight': 1.0
    }]).encode('utf-8')

    override = imageService_pb2.MetadataOverride(
        controlNets=controlnets_json
    )

    config = {
        'model': 'sd_v1.5_f16.ckpt',
        'width': img.width,
        'height': img.height,
        'steps': 28,
        'guidanceScale': 7.5
    }

    request = imageService_pb2.ImageGenerationRequest(
        hints=[hint],
        prompt=prompt,
        negativePrompt="blurry, low quality",
        configuration=serialize_configuration(config),
        scaleFactor=1,
        override=override,
        user="ThirdPartyClient",
        device=imageService_pb2.LAPTOP,
        contents=[control_compressed],
        chunked=True
    )

    images = []
    for response in stub.GenerateImage(request):
        if response.generatedImages and response.chunkState == imageService_pb2.LAST_CHUNK:
            for img_data in response.generatedImages:
                images.append(decompress_tensor(img_data))

    return images
```

### Moodboard/Style Reference

The moodboard feature uses the "shuffle" hint type:

```python
def generate_with_style_reference(stub, reference_images, prompt, weights=None):
    """Generate with style reference images (moodboard)

    Args:
        reference_images: List of PIL Image objects
        weights: List of weights (default 1.0 for each)
    """

    if weights is None:
        weights = [1.0] * len(reference_images)

    # Prepare reference tensors
    tensors_and_weights = []
    contents = []

    for img, weight in zip(reference_images, weights):
        tensor = image_to_tensor(img)
        compressed = compress_tensor(tensor)
        hash_digest = hashlib.sha256(compressed).digest()

        tensors_and_weights.append(
            imageService_pb2.TensorAndWeight(
                tensor=hash_digest,
                weight=weight
            )
        )
        contents.append(compressed)

    # Create shuffle hint
    hint = imageService_pb2.HintProto(
        hintType="shuffle",
        tensors=tensors_and_weights
    )

    config = {
        'model': 'flux_1_dev_f16.ckpt',
        'width': 1024,
        'height': 1024,
        'steps': 28,
        'guidanceScale': 3.5
    }

    request = imageService_pb2.ImageGenerationRequest(
        hints=[hint],
        prompt=prompt,
        negativePrompt="",
        configuration=serialize_configuration(config),
        scaleFactor=1,
        override=imageService_pb2.MetadataOverride(),
        user="ThirdPartyClient",
        device=imageService_pb2.LAPTOP,
        contents=contents,
        chunked=True
    )

    images = []
    for response in stub.GenerateImage(request):
        if response.generatedImages and response.chunkState == imageService_pb2.LAST_CHUNK:
            for img_data in response.generatedImages:
                images.append(decompress_tensor(img_data))

    return images

# Usage
from PIL import Image

refs = [
    Image.open("style_ref_1.jpg"),
    Image.open("style_ref_2.jpg")
]

images = generate_with_style_reference(
    stub,
    reference_images=refs,
    prompt="A portrait in this style",
    weights=[1.5, 1.0]  # First ref has more influence
)
```

### Edit Models (Qwen Image Edit)

```python
def edit_image(stub, input_image_path, instruction):
    """Edit image using instruction-based model (Qwen Image Edit)

    Args:
        input_image_path: Path to image to edit
        instruction: Text instruction for edits
    """

    # Load input image
    with Image.open(input_image_path) as img:
        tensor = image_to_tensor(img)
        compressed = compress_tensor(tensor)
        hash_digest = hashlib.sha256(compressed).digest()

    config = {
        'model': 'qwen_image_edit_2511_q8p.ckpt',
        'width': img.width,
        'height': img.height,
        'steps': 28,
        'guidanceScale': 5.0,
        'strength': 0.75,  # How much to modify
        'sampler': 0  # DPMPP_2M_KARRAS
    }

    request = imageService_pb2.ImageGenerationRequest(
        image=hash_digest,
        prompt=instruction,
        negativePrompt="blurry, low quality, distorted",
        configuration=serialize_configuration(config),
        scaleFactor=1,
        override=imageService_pb2.MetadataOverride(),
        user="ThirdPartyClient",
        device=imageService_pb2.LAPTOP,
        contents=[compressed],
        chunked=True
    )

    images = []
    for response in stub.GenerateImage(request):
        if response.generatedImages and response.chunkState == imageService_pb2.LAST_CHUNK:
            for img_data in response.generatedImages:
                images.append(decompress_tensor(img_data))

    return images

# Usage
edited = edit_image(
    stub,
    "portrait.jpg",
    "Make the background a sunset scene"
)
```

Edit model variants:
- `qwen_image_edit_2511_q8p.ckpt`: Standard 8-bit (recommended)
- `qwen_image_edit_2511_q6p.ckpt`: 6-bit (smaller, faster)
- `qwen_image_edit_2511_bf16_q8p.ckpt`: BF16 for higher quality

## Uploading Models

For servers with blob storage configured, you can upload custom models:

```python
def upload_model(stub, local_path, remote_filename, shared_secret=None, chunk_size=4*1024*1024):
    """Upload a model file to the server

    Args:
        local_path: Path to local .ckpt file
        remote_filename: Filename on server
        chunk_size: Chunk size in bytes (default 4MB)
    """

    import hashlib

    # Compute file info
    with open(local_path, 'rb') as f:
        data = f.read()
        total_size = len(data)
        sha256_hash = hashlib.sha256(data).digest()

    print(f"Uploading {remote_filename} ({total_size} bytes)")

    def request_generator():
        # Initial request
        init_request = imageService_pb2.InitUploadRequest(
            filename=remote_filename,
            sha256=sha256_hash,
            totalSize=total_size
        )

        upload_request = imageService_pb2.FileUploadRequest(
            initRequest=init_request
        )

        if shared_secret:
            upload_request.sharedSecret = shared_secret

        yield upload_request

        # Send chunks
        offset = 0
        while offset < total_size:
            chunk_end = min(offset + chunk_size, total_size)
            chunk_data = data[offset:chunk_end]

            chunk = imageService_pb2.FileChunk(
                content=chunk_data,
                filename=remote_filename,
                offset=offset
            )

            upload_request = imageService_pb2.FileUploadRequest(
                chunk=chunk
            )

            if shared_secret:
                upload_request.sharedSecret = shared_secret

            yield upload_request

            offset = chunk_end
            print(f"  Uploaded {offset}/{total_size} bytes ({offset*100//total_size}%)")

    # Upload with streaming
    for response in stub.UploadFile(request_generator()):
        if response.chunkUploadSuccess:
            print(f"  Server: {response.message}")
        else:
            print(f"  Error: {response.message}")
            return False

    print(f"Upload complete: {remote_filename}")
    return True

# Usage
success = upload_model(
    stub,
    local_path="/path/to/custom_model.ckpt",
    remote_filename="custom_model.ckpt",
    shared_secret="my_secret"
)
```

## Tensor Data Format

The server uses a custom tensor compression format with fpzip and zlib:

### Compression

```python
import struct
import zlib
import numpy as np

def compress_tensor(tensor_data, use_fpzip=True):
    """Compress tensor data

    Args:
        tensor_data: numpy array (float32 or uint8)
        use_fpzip: Use fpzip for float compression

    Returns:
        Compressed bytes with header
    """

    if tensor_data.dtype == np.float32 and use_fpzip:
        # For fpzip: use external fpzip library
        # This is a placeholder - you'll need actual fpzip bindings
        compressed = fpzip_compress(tensor_data)
        codec = 0x03  # zip | fpzip
    elif tensor_data.dtype == np.uint8:
        # For masks
        compressed = zlib.compress(tensor_data.tobytes())
        codec = 0x01  # zip only
    else:
        compressed = zlib.compress(tensor_data.astype(np.float32).tobytes())
        codec = 0x01

    # Build header
    shape = tensor_data.shape
    ndim = len(shape)

    header = struct.pack('<B', codec)  # 1 byte: codec
    header += struct.pack('<B', ndim)  # 1 byte: dimensions

    for dim in shape:
        header += struct.pack('<I', dim)  # 4 bytes per dimension

    return header + compressed

def decompress_tensor(data):
    """Decompress tensor data

    Returns:
        numpy array
    """

    # Parse header
    codec = struct.unpack('<B', data[0:1])[0]
    ndim = struct.unpack('<B', data[1:2])[0]

    offset = 2
    shape = []
    for _ in range(ndim):
        dim = struct.unpack('<I', data[offset:offset+4])[0]
        shape.append(dim)
        offset += 4

    compressed = data[offset:]

    # Decompress based on codec
    if codec & 0x02:  # fpzip
        # Use fpzip decompression
        decompressed = fpzip_decompress(compressed, shape)
    else:  # zlib only
        decompressed = zlib.decompress(compressed)
        array = np.frombuffer(decompressed, dtype=np.float32)
        array = array.reshape(shape)

    return array
```

### Image to Tensor Conversion

```python
def image_to_tensor(image):
    """Convert PIL Image to tensor format

    Returns CHW format, normalized to [-1, 1]
    """

    # Convert to RGB
    img = image.convert('RGB')

    # To numpy array (HWC)
    array = np.array(img, dtype=np.float32)

    # Normalize [0, 255] -> [-1, 1]
    array = (array / 127.5) - 1.0

    # HWC -> CHW
    array = np.transpose(array, (2, 0, 1))

    return array

def tensor_to_image(tensor):
    """Convert tensor to PIL Image

    Args:
        tensor: CHW format, range [-1, 1]

    Returns:
        PIL Image
    """

    # CHW -> HWC
    array = np.transpose(tensor, (1, 2, 0))

    # Denormalize [-1, 1] -> [0, 255]
    array = ((array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    return Image.fromarray(array, mode='RGB')

def mask_to_tensor(mask_image):
    """Convert mask image to uint8 tensor

    White (255) = inpaint, Black (0) = keep
    """

    # Convert to grayscale
    mask = mask_image.convert('L')

    # To numpy array
    array = np.array(mask, dtype=np.uint8)

    # Add batch dimension: HW -> 1HW
    array = array[np.newaxis, :, :]

    return array
```

## Configuration Serialization

The configuration uses FlatBuffers format. Here's a simplified example:

```python
# You'll need to use the actual FlatBuffer schema from DataModels
# This is a conceptual example

def serialize_configuration(config):
    """Serialize configuration to FlatBuffer format

    This is a simplified example. In practice, use the
    GenerationConfiguration FlatBuffer schema from DataModels library.
    """

    # Import FlatBuffer generated code
    from datamodels import GenerationConfiguration as GC

    builder = flatbuffers.Builder(1024)

    # Build model string
    model_str = builder.CreateString(config.get('model', ''))

    # Start building
    GC.Start(builder)

    if 'model' in config:
        GC.AddModel(builder, model_str)

    GC.AddWidth(builder, config.get('width', 1024))
    GC.AddHeight(builder, config.get('height', 1024))
    GC.AddSteps(builder, config.get('steps', 28))
    GC.AddGuidanceScale(builder, config.get('guidanceScale', 7.5))
    GC.AddSeed(builder, config.get('seed', 0))
    GC.AddStrength(builder, config.get('strength', 1.0))
    GC.AddBatchCount(builder, config.get('batchCount', 1))
    GC.AddSampler(builder, config.get('sampler', 1))

    # Add more fields as needed...

    config_fb = GC.End(builder)
    builder.Finish(config_fb)

    return bytes(builder.Output())
```

## Command-Line Tools

### Using grpcurl

```bash
# Echo (discover models)
grpcurl -plaintext \
  -d '{"name": "grpcurl-client"}' \
  localhost:7859 \
  ImageGenerationService/Echo

# With shared secret
grpcurl -plaintext \
  -d '{"name": "client", "sharedSecret": "SECRET"}' \
  localhost:7859 \
  ImageGenerationService/Echo

# Check files exist
grpcurl -plaintext \
  -d '{
    "files": [
      "flux_1_schnell_f16.ckpt",
      "qwen_image_edit_2511_q8p.ckpt"
    ]
  }' \
  localhost:7859 \
  ImageGenerationService/FilesExist

# Check files with hashes
grpcurl -plaintext \
  -d '{
    "files": ["flux_1_schnell_f16.ckpt"],
    "filesWithHash": ["flux_1_schnell_f16.ckpt"]
  }' \
  localhost:7859 \
  ImageGenerationService/FilesExist
```

### Using grpc_cli

```bash
# List services
grpc_cli ls localhost:7859

# List methods
grpc_cli ls localhost:7859 ImageGenerationService

# Describe method
grpc_cli type localhost:7859 ImageGenerationRequest

# Call echo
grpc_cli call localhost:7859 ImageGenerationService.Echo \
  'name: "test-client"'
```

## Complete Python Example

```python
#!/usr/bin/env python3
"""
Complete example: Text-to-image generation with Draw Things gRPC server
"""

import grpc
import hashlib
import numpy as np
from PIL import Image
from generated import imageService_pb2, imageService_pb2_grpc

def main():
    # Connect to server
    channel = grpc.insecure_channel('localhost:7859')
    stub = imageService_pb2_grpc.ImageGenerationServiceStub(channel)

    print("Connected to Draw Things server")

    # Check server status
    echo_req = imageService_pb2.EchoRequest(name="PythonClient")
    echo_resp = stub.Echo(echo_req)
    print(f"Server: {echo_resp.message}")
    print(f"Server ID: {echo_resp.serverIdentifier}")

    # Configuration
    config = {
        'model': 'flux_1_schnell_f16.ckpt',
        'width': 1024,
        'height': 1024,
        'steps': 4,
        'guidanceScale': 0.0,
        'seed': 42,
        'batchCount': 1,
        'sampler': 1  # EULER_A
    }

    # Serialize configuration (implement this based on FlatBuffer schema)
    config_bytes = serialize_configuration(config)

    # Create request
    request = imageService_pb2.ImageGenerationRequest(
        prompt="A serene mountain landscape at sunset",
        negativePrompt="blurry, low quality",
        configuration=config_bytes,
        scaleFactor=1,
        override=imageService_pb2.MetadataOverride(),
        user="PythonClient",
        device=imageService_pb2.LAPTOP,
        chunked=True
    )

    print("\nGenerating image...")

    # Stream generation
    images = []
    current_chunk = b''

    for response in stub.GenerateImage(request):
        # Progress updates
        if response.HasField('currentSignpost'):
            sig = response.currentSignpost
            if sig.HasField('textEncoded'):
                print("  ✓ Text encoded")
            elif sig.HasField('sampling'):
                print(f"  → Sampling step {sig.sampling.step}")
            elif sig.HasField('imageDecoded'):
                print("  ✓ Image decoded")

        # Download size notification
        if response.downloadSize > 0:
            size_mb = response.downloadSize / (1024 * 1024)
            print(f"  Receiving {size_mb:.2f} MB")

        # Generated images
        if response.generatedImages:
            for img_data in response.generatedImages:
                if response.chunkState == imageService_pb2.LAST_CHUNK:
                    full_data = current_chunk + img_data
                    current_chunk = b''

                    # Decompress and save
                    tensor = decompress_tensor(full_data)
                    image = tensor_to_image(tensor)
                    images.append(image)

                elif response.chunkState == imageService_pb2.MORE_CHUNKS:
                    current_chunk += img_data

    print(f"\n✓ Generated {len(images)} image(s)")

    # Save results
    for i, img in enumerate(images):
        filename = f"output_{i}.png"
        img.save(filename)
        print(f"  Saved: {filename}")

if __name__ == '__main__':
    main()
```

## Best Practices

### Connection Management

```python
class DrawThingsClient:
    """Reusable client with connection pooling"""

    def __init__(self, host='localhost', port=7859, use_tls=False,
                 shared_secret=None):
        if use_tls:
            # Load credentials
            with open('server.crt', 'rb') as f:
                cert = f.read()
            creds = grpc.ssl_channel_credentials(cert)
            self.channel = grpc.secure_channel(f'{host}:{port}', creds)
        else:
            self.channel = grpc.insecure_channel(f'{host}:{port}')

        self.stub = imageService_pb2_grpc.ImageGenerationServiceStub(
            self.channel
        )
        self.shared_secret = shared_secret

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.channel.close()

    def get_available_models(self):
        """Get all available models"""
        req = imageService_pb2.EchoRequest(name="Client")
        if self.shared_secret:
            req.sharedSecret = self.shared_secret

        resp = self.stub.Echo(req)
        # Parse and return...
        return resp

# Usage
with DrawThingsClient(shared_secret="secret") as client:
    models = client.get_available_models()
    # Use client...
```

### Error Handling

```python
import grpc

def safe_generate(stub, request):
    """Generate with proper error handling"""

    try:
        images = []

        for response in stub.GenerateImage(request):
            # Check for errors in metadata
            metadata = response._metadata

            # Process response
            if response.generatedImages:
                for img in response.generatedImages:
                    if response.chunkState == imageService_pb2.LAST_CHUNK:
                        images.append(decompress_tensor(img))

        return images

    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:
            print("Authentication failed - check shared secret")
        elif e.code() == grpc.StatusCode.UNAVAILABLE:
            print("Server unavailable")
        elif e.code() == grpc.StatusCode.CANCELLED:
            print("Request cancelled")
        else:
            print(f"RPC error: {e.code()} - {e.details()}")

        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### Batch Processing

```python
def batch_generate(stub, prompts, config):
    """Generate multiple images efficiently"""

    results = []

    for i, prompt in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}: {prompt[:50]}...")

        # Vary seed for each
        config_copy = config.copy()
        config_copy['seed'] = config['seed'] + i

        request = create_request(prompt, config_copy)

        images = []
        for response in stub.GenerateImage(request):
            if response.generatedImages and \
               response.chunkState == imageService_pb2.LAST_CHUNK:
                for img_data in response.generatedImages:
                    images.append(decompress_tensor(img_data))

        results.append({
            'prompt': prompt,
            'images': images
        })

    return results
```

## Troubleshooting

### Connection refused
```bash
# Check if server is running
grpcurl -plaintext localhost:7859 list

# Check firewall
sudo ufw status
sudo ufw allow 7859/tcp
```

### Authentication errors
- Verify shared secret matches server configuration
- Check if server requires TLS
- Ensure server is not in bridge mode (doesn't accept direct connections)

### Model not found errors
- Use `Echo` or `FilesExist` to verify model availability
- Check model filename exactly matches (case-sensitive)
- Ensure model is in server's models directory

### Compression errors
- Verify fpzip library is installed correctly
- Check tensor dimensions and data types
- Ensure proper byte order (little-endian)

### Out of memory
- Reduce batch size
- Lower image dimensions
- Use quantized models (q6p, q8p instead of f16)
- Check server GPU VRAM

## Security Considerations

1. **Shared Secret**: Use strong random secrets (12+ characters)
2. **TLS**: Always use TLS for remote connections
3. **Network Exposure**: Bind to specific interfaces, use firewall
4. **Input Validation**: Validate all inputs before sending
5. **Resource Limits**: Implement client-side rate limiting
6. **Error Handling**: Don't expose sensitive info in error messages

## Next Steps

- Review proto definition for complete API
- Implement FlatBuffer serialization for configuration
- Add fpzip compression library
- Build reusable client library
- Implement model caching
- Add progress callbacks
- Create GUI wrapper

## Resources

- gRPC Python Tutorial: https://grpc.io/docs/languages/python/
- FlatBuffers Python: https://google.github.io/flatbuffers/flatbuffers_guide_use_python.html
- Protocol Buffer Spec: imageService.proto
- Model Specifications: ModelZoo.swift, LoRAZoo.swift
