#!/usr/bin/env python3
"""
Generate image with moodboard/reference images support.

This example demonstrates how to use multiple reference images with edit models
like FLUX.2 Klein. Reference images let you say things like:
- "The person from image 3, wearing clothes from image 2, on the beach from image 1"

Reference types:
- "shuffle" (default): For edit/kontext models - images are VAE-encoded and fed as visual tokens
- "ipadapterplus": For IP-Adapter style conditioning
- "ipadapterfull": Full IP-Adapter
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from drawthings_client import DrawThingsClient, ImageGenerationConfig, ReferenceImage


def generate_with_references(
    prompt: str,
    model: str,
    reference_paths: list,
    server: str = "192.168.2.150:7859",
    output: str = "output/moodboard_result.png",
):
    """Generate image with reference/moodboard images."""

    # Create reference image configs
    # Each image gets equal weight by default
    reference_images = [
        ReferenceImage(
            image=path,
            weight=1.0 / len(reference_paths),
            hint_type="shuffle",  # Use "shuffle" for edit models like FLUX.2 Klein
        )
        for path in reference_paths
    ]

    config = ImageGenerationConfig(
        model=model,
        steps=20,
        width=1024,
        height=1024,
        cfg_scale=1.0,
        scheduler="Euler A Trailing",
        strength=1.0,  # Required for edit models
        guidance_embed=3.5,
        t5_text_encoder=True,
    )

    print(f"Connecting to {server}...")
    with DrawThingsClient(server, insecure=False, verify_ssl=False) as client:
        print(f"Generating with prompt: {prompt}")
        print(f"Using {len(reference_images)} reference image(s)")

        images = client.generate_image(
            prompt=prompt,
            config=config,
            reference_images=reference_images,
        )

        if images:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "wb") as f:
                f.write(images[0])
            print(f"✓ Saved to {output}")
        else:
            print("✗ No images generated")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate image with reference/moodboard images"
    )
    parser.add_argument("prompt", help="Generation prompt")
    parser.add_argument(
        "reference_images", nargs="+", help="One or more reference image paths"
    )
    parser.add_argument(
        "--model",
        default="flux_klein_q6p.ckpt",
        help="Model to use (default: flux_klein_q6p.ckpt)",
    )
    parser.add_argument(
        "--server",
        default="192.168.2.150:7859",
        help="Server address (default: 192.168.2.150:7859)",
    )
    parser.add_argument(
        "--output",
        default="output/moodboard_result.png",
        help="Output path (default: output/moodboard_result.png)",
    )

    args = parser.parse_args()

    generate_with_references(
        prompt=args.prompt,
        model=args.model,
        reference_paths=args.reference_images,
        server=args.server,
        output=args.output,
    )


if __name__ == "__main__":
    # Example usage:
    # python generate_with_references.py "the person from reference 2 wearing outfit from reference 1" person.jpg outfit.jpg
    main()
