#!/usr/bin/env python3
"""Quick 121-frame LTX video generation test with audio validation."""

import sys
import numpy as np

sys.path.insert(0, "dev/DTgRPCconnector")
from drawthings_client import DrawThingsClient, ImageGenerationConfig  # noqa: E402

SERVER = "192.168.2.150:7859"
ROOT_CA = "dev/DTgRPCconnector/root_ca.crt"

client = DrawThingsClient(SERVER, insecure=False, verify_ssl=False, ssl_cert_path=ROOT_CA)

echo = client.echo("test")
print(f"Connected: {echo.message}")

config = ImageGenerationConfig(
    model="ltx_2.3_22b_distilled_1.1_q6p.ckpt",
    steps=8,
    width=1280,
    height=768,
    cfg_scale=1.0,
    scheduler="Euler A Trailing",
    seed=42,
    seed_mode=2,
    clip_skip=1,
    shift=5,
    batch_count=1,
    batch_size=1,
    num_frames=121,
    fps_id=25,
    motion_bucket_id=127,
    compression_artifacts=0,
    hires_fix=False,
)

print("Generating 121-frame video...")
result = client.generate_media(
    prompt="a red car driving down a desert highway",
    config=config,
    progress_callback=lambda stage, step: print(f"  {stage}: {step}"),
)

print(f"\nFrames: {len(result.images)}")
print(f"Audio chunks: {len(result.audio)}")
for i, chunk in enumerate(result.audio):
    magic = int.from_bytes(chunk[:4], "little") if len(chunk) >= 4 else -1
    print(f"  chunk {i}: {len(chunk):,} bytes, magic={magic}")

if result.audio:
    audio_bytes = client.decode_audio(result.audio)
    n = len(audio_bytes) // 4
    arr = np.frombuffer(audio_bytes, dtype=np.float32)
    print(f"Decoded audio: {n} float32 samples = {n / 2 / 44100:.2f}s stereo")
    print(f"Peak: {np.max(np.abs(arr)):.4f}, NaN: {np.isnan(arr).sum()}")

    from tensor_decoder import tensor_to_pil

    out = client.save_video(
        result.images,
        output_path="outputs/test_121frame.mp4",
        fps=25,
        audio=audio_bytes,
        frame_decoder=tensor_to_pil,
    )
    print(f"Saved: {out}")