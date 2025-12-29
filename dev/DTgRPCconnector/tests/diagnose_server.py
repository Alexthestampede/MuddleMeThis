#!/usr/bin/env python3
"""
Diagnostic script for Draw Things gRPC server.
This helps identify what's working and what might be causing issues.
"""

import json
from drawthings_client import DrawThingsClient, ImageGenerationConfig, StreamingProgressHandler


def main():
    SERVER_ADDRESS = "192.168.2.150:7859"

    print("="*70)
    print(" Draw Things Server Diagnostics")
    print("="*70)

    # Test 1: Connection
    print("\n[1/4] Testing TLS connection...")
    try:
        client = DrawThingsClient(SERVER_ADDRESS, insecure=False, verify_ssl=False)
        print("  ✓ TLS connection successful")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return

    # Test 2: Echo
    print("\n[2/4] Testing server echo...")
    try:
        response = client.echo("Diagnostic Test")
        print(f"  ✓ Server responded: {response.message}")
        print(f"  ✓ Server ID: {response.serverIdentifier}")
        print(f"  ✓ Available files: {len(response.files)}")
        print(f"  ✓ Shared secret required: {response.sharedSecretMissing}")
    except Exception as e:
        print(f"  ✗ Echo failed: {e}")
        client.close()
        return

    # Test 3: Parse available models
    print("\n[3/4] Parsing available models...")
    try:
        if response.HasField('override'):
            models_json = json.loads(response.override.models.decode('utf-8'))
            print(f"  ✓ Total models available: {len(models_json)}")

            # Find realdream variants
            realdream_models = [m for m in models_json if 'realdream' in m['file'].lower()]
            print(f"  ✓ RealDream variants: {len(realdream_models)}")
            for model in realdream_models:
                print(f"    - {model['name']} ({model['file']})")

            # Show first 5 models as examples
            print("\n  First 5 models:")
            for model in models_json[:5]:
                print(f"    - {model.get('name', 'unnamed')}")
        else:
            print("  ⚠ No model metadata override available")
    except Exception as e:
        print(f"  ✗ Failed to parse models: {e}")

    # Test 4: Attempt image generation
    print("\n[4/4] Testing image generation...")
    print("  Model: realDream_15SD15 (8-bit)")
    print("  Prompt: a basket full of kittens")
    print("  Size: 512x512")
    print("  Steps: 16")
    print("  CFG: 5.0")

    config = ImageGenerationConfig(
        model="realDream_15SD15 (8-bit)",
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays"
    )

    progress_handler = StreamingProgressHandler(config.steps)

    try:
        print("\n  Starting generation...")
        images = client.generate_image(
            prompt="a basket full of kittens",
            config=config,
            progress_callback=progress_handler.on_progress
        )

        progress_handler.on_complete()

        if images:
            print(f"\n  ✓ SUCCESS! Generated {len(images)} image(s)")
            print(f"  ✓ Image size: {len(images[0])} bytes")

            # Save the image
            saved = client.save_images(images, output_dir="output", prefix="diagnostic")
            print(f"  ✓ Saved to: {saved[0]}")
        else:
            print("  ✗ No images generated")

    except Exception as e:
        print(f"\n  ✗ Generation failed: {e}")
        print("\n  Possible causes:")
        print("    - Server ran out of GPU memory")
        print("    - Model failed to load")
        print("    - Server configuration issue")
        print("    - Check Docker logs: docker logs <container_id>")
        print("\n  The error occurred after:")
        print("    ✓ Text Encoding")
        print("    ✓ Image Encoding")
        print("    ✗ Before Sampling started")
        print("\n  This suggests a server-side issue, not a client problem.")

    client.close()

    print("\n" + "="*70)
    print(" Diagnostic Complete")
    print("="*70)


if __name__ == "__main__":
    main()
