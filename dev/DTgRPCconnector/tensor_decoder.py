#!/usr/bin/env python3
"""
Draw Things Tensor Decoder - handles both compressed and uncompressed tensor formats.
Based on: https://github.com/kcjerrell/dt-grpc-ts
"""

import struct
import numpy as np
from PIL import Image
from typing import Tuple

# Try to import fpzip for compressed responses
try:
    import fpzip
    FPZIP_AVAILABLE = True
except ImportError:
    FPZIP_AVAILABLE = False
    print("Warning: fpzip not available. Only uncompressed responses will work.")
    print("Install with: pip install fpzip")

MAGIC_COMPRESSED = 1012247  # intBuffer[0] value indicating compression
HEADER_SIZE = 68

def decode_tensor(data: bytes) -> Tuple[np.ndarray, int, int, int]:
    """
    Decode Draw Things tensor format to numpy array.
    
    Args:
        data: Raw tensor data from ImageGenerationResponse.generatedImages
        
    Returns:
        Tuple of (image_array, width, height, channels)
        image_array is uint8 RGB in range [0, 255]
    """
    # Read header as 32-bit unsigned integers
    header = struct.unpack_from('<32I', data, 0)

    magic = header[0]
    height = header[6]
    width = header[7]
    channels = header[8]

    is_compressed = (magic == MAGIC_COMPRESSED)

    # DEBUG: Log full header to understand server format
    print(f"[DECODER] Magic: {magic}, Height: {height}, Width: {width}, Channels: {channels}")
    print(f"[DECODER] Header bytes 0-4: {header[0:5]}")
    
    print(f"Image: {width}×{height}×{channels}")
    print(f"Compressed: {is_compressed}")
    print(f"Input size: {len(data):,} bytes")
    
    # Get float data
    if is_compressed:
        if not FPZIP_AVAILABLE:
            raise RuntimeError("Response is compressed but fpzip is not available")
        
        # Decompress fpzip data (after 68-byte header)
        compressed_data = data[HEADER_SIZE:]
        print(f"Decompressing {len(compressed_data):,} bytes...")
        
        # fpzip.decompress returns float32 with shape (1, height, width, channels)
        float_data = fpzip.decompress(compressed_data, order='C')
        print(f"Decompressed shape: {float_data.shape}, dtype: {float_data.dtype}")
        
        # Remove batch dimension if present and ensure correct shape
        if float_data.ndim == 4 and float_data.shape[0] == 1:
            float_data = float_data[0]  # Remove batch dimension
        
        # Convert to float16 to match uncompressed format
        tensor = float_data.astype(np.float16)
            
    else:
        # Read directly as float16 from offset 68
        expected_floats = width * height * channels
        float_data = np.frombuffer(
            data[HEADER_SIZE:HEADER_SIZE + expected_floats * 2], 
            dtype=np.float16
        )
        # Reshape to image dimensions
        tensor = float_data.reshape((height, width, channels))
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Convert from [-1, 1] to [0, 255]
    # CRITICAL: Must use 127.5 for proper symmetry with encoder
    # -1.0 → 0, 0.0 → 127.5 → 128, 1.0 → 255
    image_array = ((tensor + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    return image_array, width, height, channels

def tensor_to_pil(data: bytes) -> Image.Image:
    """
    Convert Draw Things tensor data to PIL Image.
    
    Args:
        data: Raw tensor data from ImageGenerationResponse.generatedImages
        
    Returns:
        PIL Image object
    """
    image_array, width, height, channels = decode_tensor(data)
    
    if channels == 3:
        return Image.fromarray(image_array, 'RGB')
    elif channels == 4:
        return Image.fromarray(image_array, 'RGBA')
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

def save_tensor_image(data: bytes, filename: str):
    """
    Decode and save tensor data as PNG.
    
    Args:
        data: Raw tensor data from ImageGenerationResponse.generatedImages
        filename: Output filename (e.g., 'output.png')
    """
    img = tensor_to_pil(data)
    img.save(filename)
    print(f"✅ Saved to {filename}")
    return img

if __name__ == '__main__':
    # Test with our generated image
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tensor_decoder.py <tensor_file> [output.png]")
        print("\nExample:")
        print("  python tensor_decoder.py output/kitten_scale8.png output/decoded.png")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'decoded_output.png'
    
    with open(input_file, 'rb') as f:
        data = f.read()
    
    save_tensor_image(data, output_file)
