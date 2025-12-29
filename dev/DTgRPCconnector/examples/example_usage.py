#!/usr/bin/env python3
"""
Simple examples of using the Draw Things Python client library.

Run this file to see different usage patterns.
"""

from drawthings_client import (
    DrawThingsClient,
    ImageGenerationConfig,
    quick_generate
)


def example_1_simple_generation():
    """Example 1: Simple one-liner image generation."""
    print("\n" + "="*60)
    print("Example 1: Simple One-Liner Generation")
    print("="*60)

    image_path = quick_generate(
        server_address="192.168.2.150:7859",
        prompt="A beautiful mountain landscape at sunset",
        output_path="output/simple_example.png"
    )

    print(f"✓ Image saved to: {image_path}")


def example_2_with_config():
    """Example 2: Generation with custom configuration."""
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)

    # Create custom configuration
    config = ImageGenerationConfig(
        model="realDream",
        steps=20,  # More steps for higher quality
        width=768,
        height=768,
        cfg_scale=7.0,  # Higher guidance
        scheduler="UniPC ays",
        seed=42  # Fixed seed for reproducibility
    )

    with DrawThingsClient("192.168.2.150:7859") as client:
        images = client.generate_image(
            prompt="A cyberpunk city at night, neon lights, rain, highly detailed",
            negative_prompt="blurry, low quality, distorted",
            config=config
        )

        saved = client.save_images(images, output_dir="output", prefix="custom_config")
        print(f"✓ Saved: {saved[0]}")


def example_3_with_progress():
    """Example 3: Generation with progress tracking."""
    print("\n" + "="*60)
    print("Example 3: With Progress Tracking")
    print("="*60)

    config = ImageGenerationConfig(
        model="realDream",
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays"
    )

    def show_progress(stage, step):
        """Custom progress callback."""
        if stage == "Sampling":
            percent = (step / config.steps) * 100
            bar_length = 30
            filled = int(bar_length * step / config.steps)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r  [{bar}] {step}/{config.steps} ({percent:.0f}%)", end="", flush=True)
        else:
            print(f"\n  {stage}...", flush=True)

    with DrawThingsClient("192.168.2.150:7859") as client:
        images = client.generate_image(
            prompt="A serene Japanese garden with koi pond",
            config=config,
            progress_callback=show_progress
        )

        print("\n")  # New line after progress bar
        saved = client.save_images(images, output_dir="output", prefix="with_progress")
        print(f"✓ Saved: {saved[0]}")


def example_4_batch_generation():
    """Example 4: Generate multiple different images."""
    print("\n" + "="*60)
    print("Example 4: Batch Generation")
    print("="*60)

    prompts = [
        "A majestic dragon flying over mountains",
        "A futuristic spaceship in deep space",
        "A cozy cottage in a forest clearing"
    ]

    config = ImageGenerationConfig(
        model="realDream",
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays"
    )

    with DrawThingsClient("192.168.2.150:7859") as client:
        for i, prompt in enumerate(prompts, 1):
            print(f"\nGenerating {i}/{len(prompts)}: {prompt}")

            images = client.generate_image(
                prompt=prompt,
                config=config,
                progress_callback=lambda s, st: print(".", end="", flush=True)
            )

            saved = client.save_images(images, output_dir="output", prefix=f"batch_{i}")
            print(f" ✓ {saved[0]}")


def example_5_test_connection():
    """Example 5: Test server connection."""
    print("\n" + "="*60)
    print("Example 5: Test Server Connection")
    print("="*60)

    with DrawThingsClient("192.168.2.150:7859") as client:
        response = client.echo("Python Client Test")

        print(f"Server Message: {response.message}")
        print(f"Server ID: {response.serverIdentifier}")

        if response.files:
            print(f"Available files: {len(response.files)}")
            # Print first few files as example
            for filename in response.files[:5]:
                print(f"  - {filename}")
            if len(response.files) > 5:
                print(f"  ... and {len(response.files) - 5} more")


def example_6_different_schedulers():
    """Example 6: Try different schedulers."""
    print("\n" + "="*60)
    print("Example 6: Different Schedulers")
    print("="*60)

    schedulers = [
        "UniPC ays",
        "Euler A",
        "DPMPP2M Karras"
    ]

    base_config = {
        "model": "realDream",
        "steps": 16,
        "width": 512,
        "height": 512,
        "cfg_scale": 5.0,
        "seed": 12345  # Same seed for comparison
    }

    prompt = "A fantasy castle on a floating island"

    with DrawThingsClient("192.168.2.150:7859") as client:
        for scheduler in schedulers:
            print(f"\nTrying scheduler: {scheduler}")

            config = ImageGenerationConfig(**base_config, scheduler=scheduler)

            images = client.generate_image(
                prompt=prompt,
                config=config
            )

            safe_name = scheduler.replace(" ", "_").lower()
            saved = client.save_images(
                images,
                output_dir="output",
                prefix=f"scheduler_{safe_name}"
            )
            print(f"  ✓ Saved: {saved[0]}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" Draw Things Python Client - Usage Examples")
    print("="*70)

    try:
        # Run examples
        # Uncomment the ones you want to run

        example_5_test_connection()  # Always good to test connection first
        example_1_simple_generation()
        example_3_with_progress()

        # These take longer, uncomment if you want to run them:
        # example_2_with_config()
        # example_4_batch_generation()
        # example_6_different_schedulers()

        print("\n" + "="*70)
        print(" All examples completed!")
        print("="*70)
        print("\nCheck the 'output/' directory for generated images.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("  1. The Draw Things server is running at 192.168.2.150:7859")
        print("  2. The 'realDream' model is available on the server")
        print("  3. Network connectivity is working")


if __name__ == "__main__":
    main()
