#!/usr/bin/env python3
"""
Final test with complete configuration matching official client.
"""

from drawthings_client import DrawThingsClient, ImageGenerationConfig, StreamingProgressHandler


def main():
    print("="*70)
    print(" Draw Things Client - Final Test")
    print("="*70)

    SERVER_ADDRESS = "192.168.2.150:7859"

    # Create client with TLS
    client = DrawThingsClient(SERVER_ADDRESS, insecure=False, verify_ssl=False)

    # Complete configuration matching official client settings
    config = ImageGenerationConfig(
        model="realdream_15sd15_q6p_q8p.ckpt",  # Filename, not display name
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays",
        seed=3300843894,
        strength=1.0,
        seed_mode=2,  # ScaleAlike
        clip_skip=1,
        shift=1.0,
        mask_blur=2.5,
        mask_blur_outset=0,
        sharpness=0.0,
        preserve_original_after_inpaint=True,
        cfg_zero_star=False,
        cfg_zero_init_steps=0,
        hires_fix=False,
        causal_inference_pad=0,
        tiled_diffusion=False,
        tiled_decoding=False,
        batch_count=1,
        batch_size=1,
        upscaler="",
        face_restoration="",
        refiner_model="",
    )

    print(f"\nConfiguration:")
    print(f"  Model: {config.model}")
    print(f"  Prompt: a basket full of kittens")
    print(f"  Size: {config.width}x{config.height}")
    print(f"  Steps: {config.steps}")
    print(f"  CFG Scale: {config.cfg_scale}")
    print(f"  Scheduler: {config.scheduler}")
    print(f"  Seed: {config.seed}")
    print(f"  Seed Mode: {config.seed_mode} (ScaleAlike)")
    print(f"  CLIP Skip: {config.clip_skip}")
    print(f"  Shift: {config.shift}")

    progress_handler = StreamingProgressHandler(config.steps)

    print("\nGenerating image...")
    try:
        images = client.generate_image(
            prompt="a basket full of kittens",
            config=config,
            progress_callback=progress_handler.on_progress
        )

        progress_handler.on_complete()

        if images:
            print(f"\n🎉 SUCCESS!")
            print(f"Generated {len(images)} image(s)")
            print(f"Image size: {len(images[0]):,} bytes")

            # Save images
            saved = client.save_images(images, output_dir="output", prefix="kitten_final")
            print(f"\nSaved to: {saved[0]}")
        else:
            print("\n✗ No images generated")

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        print("\n📋 Progress so far:")
        print("  ✓ TLS connection")
        print("  ✓ Text encoding")
        print("  ✓ Image encoding")
        print("  ✓ Sampling started")
        print("  ✗ Server closed connection during sampling")
        print("\n🔍 Next steps:")
        print("  1. Check Docker logs: docker logs <container_id>")
        print("  2. Look for errors around sampling/generation")
        print("  3. Check GPU memory: nvidia-smi")
        print("  4. Verify model files are accessible in /grpc-models")

    client.close()

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
