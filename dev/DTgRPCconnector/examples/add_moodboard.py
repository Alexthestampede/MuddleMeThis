#!/usr/bin/env python3
"""
Add moodboard/reference image support to Draw Things client.

This demonstrates how to use IP-Adapter for style/composition reference.
"""

from PIL import Image
import io
import hashlib
from typing import List, Tuple

def load_reference_image(image_path: str, max_size: int = 1024) -> bytes:
    """
    Load and preprocess a reference image.

    Args:
        image_path: Path to the reference image
        max_size: Maximum dimension (will resize if larger)

    Returns:
        JPEG bytes of the processed image
    """
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize if too large (preserving aspect ratio)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convert to JPEG bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()

def get_image_sha256(image_bytes: bytes) -> bytes:
    """Calculate SHA256 hash of image data."""
    return hashlib.sha256(image_bytes).digest()

def prepare_reference_images(image_paths: List[str]) -> Tuple[List[bytes], List[Tuple[bytes, float]]]:
    """
    Prepare reference images for IP-Adapter.

    Args:
        image_paths: List of paths to reference images

    Returns:
        Tuple of (contents_array, hints_data)
        - contents_array: List of image bytes for the contents field
        - hints_data: List of (sha256, weight) tuples for the hints field
    """
    contents = []
    hints_data = []

    for i, path in enumerate(image_paths):
        # Load image
        image_bytes = load_reference_image(path)

        # Calculate SHA256
        sha256 = get_image_sha256(image_bytes)

        # Add to arrays
        contents.append(image_bytes)

        # Equal weight for all images (can be customized)
        weight = 1.0 / len(image_paths)
        hints_data.append((sha256, weight))

        print(f"✓ Prepared reference image {i+1}/{len(image_paths)}: {path}")
        print(f"  Size: {len(image_bytes):,} bytes")
        print(f"  SHA256: {sha256.hex()[:16]}...")
        print(f"  Weight: {weight:.2f}")

    return contents, hints_data

# Example usage
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python add_moodboard.py <image1.jpg> [image2.jpg] ...")
        print("\nExample:")
        print("  python add_moodboard.py reference1.jpg reference2.jpg")
        print("\nThis will prepare the images for use as moodboard/reference images.")
        sys.exit(1)

    image_paths = sys.argv[1:]

    print("Preparing reference images for moodboard...")
    print("=" * 70)

    contents, hints_data = prepare_reference_images(image_paths)

    print("\n" + "=" * 70)
    print(f"Ready to use {len(image_paths)} reference image(s)!")
    print("\nTo use in generation:")
    print("1. Add to ImageGenerationRequest.contents = contents")
    print("2. Create HintProto with hintType='ipadapterplus'")
    print("3. Add TensorAndWeight entries with sha256 and weights")
    print("\nSee generate_with_moodboard.py for a complete example.")
