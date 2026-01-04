#!/usr/bin/env python3
"""
Encode images to Draw Things tensor format.

This is the reverse of tensor_decoder.py - it converts images TO the tensor
format that Draw Things expects for input images.
"""

import numpy as np
from PIL import Image
import struct
import fpzip

def encode_image_to_tensor(image_path: str, compress: bool = True, for_server: bool = False) -> bytes:
    """
    Encode an image to Draw Things tensor format.

    Args:
        image_path: Path to image file
        compress: Whether to use fpzip compression (default True)
        for_server: If True, encode as raw fpzip WITHOUT header (for gRPC server)
                   If False, include 68-byte header (for client-side decoding)

    Returns:
        Bytes in tensor format ready for Draw Things
    """
    # Load image
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert to numpy array
    img_array = np.array(img, dtype=np.uint8)  # Shape: (H, W, 3)

    height, width, channels = img_array.shape

    # Convert from uint8 [0, 255] to float32 [-1, 1]
    # Note: fpzip requires float32, NOT float16
    # CRITICAL: Must use 127.5 to get exact [-1, 1] range
    float_array = (img_array.astype(np.float32) / 127.5) - 1.0

    # Debug: verify value range BEFORE compression
    print(f"[ENCODER] Input value range: [{float_array.min():.6f}, {float_array.max():.6f}]")

    # Add batch dimension: (H, W, 3) -> (1, H, W, 3)
    # fpzip expects/produces tensors with batch dimension
    float_array = np.expand_dims(float_array, axis=0)

    # Compress pixel data
    if compress:
        # fpzip compression (requires float32)
        pixel_data = fpzip.compress(float_array, order='C')
    else:
        # For uncompressed, convert to float16 to save space
        float16_array = float_array.astype(np.float16)
        pixel_data = float16_array.tobytes()

    # Create proper 68-byte header matching server format
    header = bytearray(68)
    magic = 1012247 if compress else 0

    # Pack header as 32-bit unsigned integers to match server format exactly
    struct.pack_into('<I', header, 0, magic)      # header[0] = magic (1012247 for compressed)
    struct.pack_into('<I', header, 4, 1)          # header[1] = 1 (batch size)
    struct.pack_into('<I', header, 8, 2)          # header[2] = 2 (data type: float32)
    struct.pack_into('<I', header, 12, height * width * channels * 4)  # header[3] = total bytes
    struct.pack_into('<I', header, 16, 0)         # header[4] = 0 (reserved)
    struct.pack_into('<I', header, 20, 0)         # header[5] = 0 (reserved)
    struct.pack_into('<I', header, 24, height)    # header[6] = height
    struct.pack_into('<I', header, 28, width)     # header[7] = width
    struct.pack_into('<I', header, 32, channels)  # header[8] = channels

    # For server: use complete header (matches server's tensor format)
    # For client: same format (our decoder already handles it)
    return bytes(header) + pixel_data

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tensor_encoder.py <image.jpg>")
        print("\nConverts an image to Draw Things tensor format")
        sys.exit(1)

    image_path = sys.argv[1]
    tensor_bytes = encode_image_to_tensor(image_path)

    print(f"Encoded {image_path} to tensor format")
    print(f"Size: {len(tensor_bytes):,} bytes")

    # Save to file
    output_path = image_path.rsplit('.', 1)[0] + '.tensor'
    with open(output_path, 'wb') as f:
        f.write(tensor_bytes)
    print(f"Saved to: {output_path}")
