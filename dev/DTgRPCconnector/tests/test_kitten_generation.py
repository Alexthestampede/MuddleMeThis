#!/usr/bin/env python3
"""
Test Draw Things gRPC client with TLS and compression.
Generates a test image of "a basket full of kittens" with specific parameters.
"""

import grpc
from pathlib import Path
from drawthings_client import (
    DrawThingsClient,
    ImageGenerationConfig,
    StreamingProgressHandler,
)


def main():
    """Test client with TLS and compression enabled."""
    SERVER_ADDRESS = "192.168.2.150:7859"

    print("="*70)
    print(" Testing Draw Things gRPC Client")
    print("="*70)
    print(f"\nServer: {SERVER_ADDRESS}")
    print("Configuration: TLS enabled, compression enabled")

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test with TLS enabled (secure=True)
    print("\n" + "-"*70)
    print("Step 1: Testing connection with TLS...")
    print("-"*70)

    try:
        # Create client with TLS enabled, accepting self-signed certificates
        client = DrawThingsClient(
            SERVER_ADDRESS,
            insecure=False,          # Use TLS
            verify_ssl=False,        # Accept self-signed certificates
            enable_compression=False  # Server doesn't support compression
        )

        # Test echo to get model list
        print("\nSending echo request...")
        response = client.echo("Python TLS Test")

        print(f"✓ Server responded: {response.message}")
        print(f"  Server ID: {response.serverIdentifier}")

        if response.files:
            print(f"\n  Available models/files: {len(response.files)}")
            # Look for realdream model
            realdream_found = False
            for filename in response.files:
                if "realdream" in filename.lower():
                    print(f"  ✓ Found: {filename}")
                    realdream_found = True

            if not realdream_found:
                print("\n  Warning: 'realdream' model not found in file list")
                print("  First 10 files:")
                for filename in response.files[:10]:
                    print(f"    - {filename}")

        # Generate test image
        print("\n" + "-"*70)
        print("Step 2: Generating test image...")
        print("-"*70)

        # Try with the exact model filename (without .ckpt extension)
        config = ImageGenerationConfig(
            model="realdream_15sd15_q6p_q8p",
            steps=16,
            width=512,
            height=512,
            cfg_scale=5.0,
            scheduler="UniPC ays"
        )

        print(f"\nGeneration parameters:")
        print(f"  Model: {config.model}")
        print(f"  Prompt: a basket full of kittens")
        print(f"  Resolution: {config.width}x{config.height}")
        print(f"  Steps: {config.steps}")
        print(f"  CFG Scale: {config.cfg_scale}")
        print(f"  Scheduler: {config.scheduler}")
        print(f"  Seed: {config.seed}")

        progress_handler = StreamingProgressHandler(config.steps)

        print("\nGenerating...")
        images = client.generate_image(
            prompt="a basket full of kittens",
            config=config,
            negative_prompt="",
            progress_callback=progress_handler.on_progress
        )

        progress_handler.on_complete()

        if images:
            print(f"\n✓ Generated {len(images)} image(s)")

            # Save images
            saved_files = client.save_images(
                images,
                output_dir=str(output_dir),
                prefix="kitten_test"
            )

            print(f"\nSaved to:")
            for filepath in saved_files:
                print(f"  - {filepath}")
        else:
            print("\n✗ No images generated")

        client.close()

        print("\n" + "="*70)
        print(" Test completed successfully!")
        print("="*70)

    except grpc.RpcError as e:
        print(f"\n✗ gRPC Error: {e.code()}")
        print(f"   Details: {e.details()}")
        print("\nThis might indicate:")
        print("  - TLS certificate issues")
        print("  - Server not configured for TLS")
        print("  - Connection/network issues")
        print("\nTrying with insecure connection...")
        test_insecure(SERVER_ADDRESS, output_dir)

    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nTrying with insecure connection...")
        test_insecure(SERVER_ADDRESS, output_dir)


def test_insecure(server_address, output_dir):
    """Test with insecure connection as fallback."""
    print("\n" + "="*70)
    print(" Testing with insecure connection (no TLS)")
    print("="*70)

    try:
        client = DrawThingsClient(
            server_address,
            insecure=True,
            enable_compression=False
        )

        print("\nTesting echo...")
        response = client.echo("Python Insecure Test")
        print(f"✓ Server responded: {response.message}")

        if response.files:
            print(f"  Available files: {len(response.files)}")
            for filename in response.files[:10]:
                if "realdream" in filename.lower():
                    print(f"  ✓ Found: {filename}")

        # Generate test image
        print("\nGenerating test image...")
        config = ImageGenerationConfig(
            model="realdream_15sd15_q6p_q8p",
            steps=16,
            width=512,
            height=512,
            cfg_scale=5.0,
            scheduler="UniPC ays"
        )

        progress_handler = StreamingProgressHandler(config.steps)

        images = client.generate_image(
            prompt="a basket full of kittens",
            config=config,
            progress_callback=progress_handler.on_progress
        )

        progress_handler.on_complete()

        if images:
            print(f"\n✓ Generated {len(images)} image(s)")
            saved_files = client.save_images(
                images,
                output_dir=str(output_dir),
                prefix="kitten_test_insecure"
            )
            print(f"Saved to: {saved_files[0]}")

        client.close()
        print("\n✓ Insecure connection works!")

    except Exception as e:
        print(f"\n✗ Insecure connection also failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
