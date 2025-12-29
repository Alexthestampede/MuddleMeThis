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

def encode_image_to_tensor(image_path: str, compress: bool = True) -> bytes:
    """
    Encode an image to Draw Things tensor format.

    Args:
        image_path: Path to image file
        compress: Whether to use fpzip compression (default True)

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

    # Convert from uint8 [0, 255] to float16 [-1, 1]
    # Reverse of: uint8 = (float16 + 1) × 127
    float_array = (img_array.astype(np.float32) / 127.0) - 1.0
    float_array = float_array.astype(np.float16)

    # Create header (68 bytes)
    header = bytearray(68)

    # Magic number for compression
    magic = 1012247 if compress else 0

    # Pack header as 32-bit unsigned integers
    struct.pack_into('<I', header, 0, magic)      # header[0] = magic
    struct.pack_into('<I', header, 24, height)    # header[6] = height
    struct.pack_into('<I', header, 28, width)     # header[7] = width
    struct.pack_into('<I', header, 32, channels)  # header[8] = channels

    # Compress or append pixel data
    if compress:
        # fpzip compression
        pixel_data = fpzip.compress(float_array, order='C')
    else:
        pixel_data = float_array.tobytes()

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
