#!/usr/bin/env python3
"""FastH3 Omni test: 73 frames (~3s @ 25fps), 8 steps, tiled decoding OFF."""

import sys

import numpy as np

sys.path.insert(0, "dev/DTgRPCconnector")
from drawthings_client import DrawThingsClient, ImageGenerationConfig  # noqa: E402

SERVER = "192.168.2.150:7859"
ROOT_CA = "dev/DTgRPCconnector/root_ca.crt"

MODEL = "fasth3_omni_i4x.ckpt"
NUM_FRAMES = 73  # ~3s at 25fps
FPS = 24
AUDIO_RATE = 32000  # H3 delivers 32 kHz at 24 fps
PROMPT = (
    'A woman looks into the camera and says clearly: "Testing one two three. '
    'The quick brown fox jumps over the lazy dog." Close-up, natural lighting.'
)

client = DrawThingsClient(SERVER, insecure=False, verify_ssl=False, ssl_cert_path=ROOT_CA)
print("Connected:", client.echo("test").message)

config = ImageGenerationConfig(
    model=MODEL,
    steps=8,  # FastH3 wants 8, not the 50-step MiniMax default
    width=832,
    height=512,
    cfg_scale=1.0,
    scheduler="DDIM Trailing",
    seed=42,
    seed_mode=2,
    clip_skip=1,
    shift=12.0,
    batch_count=1,
    batch_size=1,
    num_frames=NUM_FRAMES,
    fps_id=FPS,
    motion_bucket_id=127,
    compression_artifacts=0,
    tiled_decoding=False,  # preset tiling was a phone-hardware artifact
    shift_for_audio=3.0,
)

print(f"Generating {NUM_FRAMES} frames with {MODEL} (8 steps, no tiling)...")
result = client.generate_media(
    prompt=PROMPT,
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
    print(f"Decoded audio: {n} samples = {n / 2 / AUDIO_RATE:.2f}s stereo @ {AUDIO_RATE}Hz")
    print(f"Expected video: {NUM_FRAMES / FPS:.2f}s | peak {np.max(np.abs(arr)):.4f} | NaN {np.isnan(arr).sum()}")

from tensor_decoder import tensor_to_pil  # noqa: E402

out = client.save_video(
    result.images,
    output_path="outputs/test_fasth3_73.mp4",
    fps=FPS,
    audio=audio_bytes,
    audio_sample_rate=AUDIO_RATE,
    frame_decoder=tensor_to_pil,
)
print(f"Saved: {out}")
