#!/usr/bin/env python3
"""
Encode images to Draw Things tensor format (CCV NNC tensor).

The CCV tensor header is 68 bytes (17 x uint32):
  header[0] = type flags (compression magic OR datatype+memory flags)
  header[1] = CCV_TENSOR_CPU_MEMORY (0x1)
  header[2] = CCV_TENSOR_FORMAT_NHWC (0x02)
  header[3] = datatype (CCV_16F=0x20000, CCV_32F=0x10000)
  header[4] = reserved (0)
  header[5] = batch (1)
  header[6] = height
  header[7] = width
  header[8] = channels
  header[9..16] = reserved (0)

For fpzip-compressed tensors, header[0] = 1012247 (magic).
For uncompressed float16, header[0] = 0.
"""

import numpy as np
from PIL import Image
import struct

# CCV tensor format constants
CCV_TENSOR_CPU_MEMORY = 0x1
CCV_TENSOR_FORMAT_NHWC = 0x02
CCV_16F = 0x20000
CCV_32F = 0x10000
FPZIP_MAGIC = 1012247
HEADER_SIZE = 68


def encode_image_to_tensor(image_path: str, compress: bool = True) -> bytes:
    """
    Encode an image to Draw Things CCV tensor format.

    Args:
        image_path: Path to image file
        compress: If True, use fpzip compression (float32).
                  If False, use uncompressed float16 (matches Swift reference client).

    Returns:
        Bytes in CCV tensor format ready for Draw Things server
    """
    img = Image.open(image_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = np.array(img, dtype=np.uint8)  # Shape: (H, W, 3)
    height, width, channels = img_array.shape

    # Normalize to [-1, 1] range
    # Swift client uses: Float(uint8Value) / 255.0 * 2.0 - 1.0
    # Which is equivalent to: uint8 / 127.5 - 1.0
    float_array = (img_array.astype(np.float32) / 127.5) - 1.0

    print(f"[ENCODER] Input value range: [{float_array.min():.6f}, {float_array.max():.6f}]")

    # Build CCV tensor header (68 bytes = 17 x uint32)
    header = bytearray(HEADER_SIZE)

    if compress:
        import fpzip

        # fpzip compressed format
        # Add batch dimension for fpzip: (H, W, C) -> (1, H, W, C)
        float_array_4d = np.expand_dims(float_array, axis=0)
        pixel_data = fpzip.compress(float_array_4d, order='C')

        struct.pack_into('<I', header, 0, FPZIP_MAGIC)            # [0] fpzip magic
        struct.pack_into('<I', header, 4, CCV_TENSOR_CPU_MEMORY)  # [1] memory type
        struct.pack_into('<I', header, 8, CCV_TENSOR_FORMAT_NHWC) # [2] format
        struct.pack_into('<I', header, 12, CCV_32F)               # [3] float32 datatype
        # header[4] = 0 (reserved)
        struct.pack_into('<I', header, 20, 1)                     # [5] batch = 1
        struct.pack_into('<I', header, 24, height)                # [6] height
        struct.pack_into('<I', header, 28, width)                 # [7] width
        struct.pack_into('<I', header, 32, channels)              # [8] channels
    else:
        # Uncompressed float16 format (matches Swift reference client exactly)
        float16_array = float_array.astype(np.float16)
        pixel_data = float16_array.tobytes()

        struct.pack_into('<I', header, 0, 0)                      # [0] no compression
        struct.pack_into('<I', header, 4, CCV_TENSOR_CPU_MEMORY)  # [1] memory type
        struct.pack_into('<I', header, 8, CCV_TENSOR_FORMAT_NHWC) # [2] format
        struct.pack_into('<I', header, 12, CCV_16F)               # [3] float16 datatype
        # header[4] = 0 (reserved)
        struct.pack_into('<I', header, 20, 1)                     # [5] batch = 1
        struct.pack_into('<I', header, 24, height)                # [6] height
        struct.pack_into('<I', header, 28, width)                 # [7] width
        struct.pack_into('<I', header, 32, channels)              # [8] channels

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

    output_path = image_path.rsplit('.', 1)[0] + '.tensor'
    with open(output_path, 'wb') as f:
        f.write(tensor_bytes)
    print(f"Saved to: {output_path}")
