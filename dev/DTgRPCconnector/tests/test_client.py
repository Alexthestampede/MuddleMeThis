#!/usr/bin/env python3
"""
Draw Things gRPC Client Test Script

This script demonstrates how to use the Draw Things Python client library
to generate images from the Draw Things server.

Usage:
    python test_client.py
"""

import sys
from pathlib import Path
from drawthings_client import (
    DrawThingsClient,
    ImageGenerationConfig,
    StreamingProgressHandler,
    quick_generate
)


def test_echo(client: DrawThingsClient):
    """Test server connectivity with echo request."""
    print("\n" + "="*60)
    print("Testing server connectivity...")
    print("="*60)

    try:
        response = client.echo("Python Client Test")
        print(f"✓ Server responded: {response.message}")
        print(f"  Server Identifier: {response.serverIdentifier}")
        if response.files:
            print(f"  Available files: {len(response.files)}")
        return True
    except Exception as e:
        print(f"✗ Echo test failed: {e}")
        return False


def test_basic_generation(client: DrawThingsClient):
    """Test basic image generation with specified parameters."""
    print("\n" + "="*60)
    print("Testing image generation...")
    print("="*60)

    # Configuration matching requirements:
    # - Model: realDream
    # - Steps: 16
    # - Resolution: 512x512
    # - CFG: 5.0
    # - Scheduler: UniPC ays
    config = ImageGenerationConfig(
        model="realDream",
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays"
    )

    print(f"\nConfiguration:")
    print(f"  Model: {config.model}")
    print(f"  Steps: {config.steps}")
    print(f"  Resolution: {config.width}x{config.height}")
    print(f"  CFG Scale: {config.cfg_scale}")
    print(f"  Scheduler: {config.scheduler}")
    print(f"  Seed: {config.seed}")

    prompt = "A serene Japanese garden with a koi pond, cherry blossoms, and a traditional stone lantern, highly detailed, photorealistic"
    negative_prompt = "blurry, low quality, distorted, ugly, deformed"

    print(f"\nPrompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")

    # Create progress handler
    progress_handler = StreamingProgressHandler(config.steps)

    try:
        print("\nGenerating image...")
        images = client.generate_image(
            prompt=prompt,
            config=config,
            negative_prompt=negative_prompt,
            progress_callback=progress_handler.on_progress
        )

        progress_handler.on_complete()

        if images:
            print(f"\n✓ Generated {len(images)} image(s)")

            # Save images
            output_dir = Path("/home/alexthestampede/Aish/gRPC/output")
            saved_files = client.save_images(
                images,
                output_dir=str(output_dir),
                prefix="test_generation"
            )

            print(f"\nSaved images:")
            for filepath in saved_files:
                print(f"  - {filepath}")

            return True
        else:
            print("✗ No images generated")
            return False

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_images(client: DrawThingsClient):
    """Test generating multiple images with different prompts."""
    print("\n" + "="*60)
    print("Testing multiple image generation...")
    print("="*60)

    prompts = [
        "A majestic mountain landscape at sunset",
        "A futuristic city with flying cars",
        "A cute robot playing with a cat"
    ]

    config = ImageGenerationConfig(
        model="realDream",
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays"
    )

    output_dir = Path("/home/alexthestampede/Aish/gRPC/output")

    for i, prompt in enumerate(prompts):
        print(f"\nGenerating image {i+1}/{len(prompts)}: {prompt}")

        try:
            images = client.generate_image(
                prompt=prompt,
                config=config,
                progress_callback=lambda stage, step: print(f"\r  {stage}: {step}", end="", flush=True)
            )

            if images:
                saved_files = client.save_images(
                    images,
                    output_dir=str(output_dir),
                    prefix=f"multi_test_{i+1}"
                )
                print(f"\n  ✓ Saved: {saved_files[0]}")
            else:
                print("\n  ✗ No image generated")

        except Exception as e:
            print(f"\n  ✗ Failed: {e}")

    return True


def test_quick_generate():
    """Test the convenience quick_generate function."""
    print("\n" + "="*60)
    print("Testing quick_generate convenience function...")
    print("="*60)

    try:
        output_path = quick_generate(
            server_address="192.168.2.150:7859",
            prompt="A beautiful butterfly on a flower, macro photography",
            model="realDream",
            steps=16,
            width=512,
            height=512,
            cfg_scale=5.0,
            scheduler="UniPC ays",
            output_path="/home/alexthestampede/Aish/gRPC/output/quick_test.png",
            show_progress=True
        )

        print(f"\n✓ Image saved to: {output_path}")
        return True

    except Exception as e:
        print(f"\n✗ Quick generate failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("\n" + "="*70)
    print(" Draw Things gRPC Client Test Suite")
    print("="*70)

    SERVER_ADDRESS = "192.168.2.150:7859"
    print(f"\nServer: {SERVER_ADDRESS}")

    # Create output directory
    output_dir = Path("/home/alexthestampede/Aish/gRPC/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Test 1: Echo test
    with DrawThingsClient(SERVER_ADDRESS) as client:
        echo_success = test_echo(client)

        if not echo_success:
            print("\n✗ Server connectivity test failed. Please check:")
            print("  1. Server is running at 192.168.2.150:7859")
            print("  2. Network connectivity")
            print("  3. Firewall settings")
            sys.exit(1)

        # Test 2: Basic generation with specified parameters
        basic_success = test_basic_generation(client)

        # Test 3: Multiple images
        if basic_success:
            test_multiple_images(client)

    # Test 4: Quick generate function
    test_quick_generate()

    print("\n" + "="*70)
    print(" Test Suite Complete")
    print("="*70)
    print("\nCheck the output directory for generated images:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
