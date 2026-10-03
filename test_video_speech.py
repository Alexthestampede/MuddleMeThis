#!/usr/bin/env python3
"""121-frame LTX video generation test with speech audio validation.

Speech prompts are used deliberately: engine noise sounds plausible at any
pitch, but wrong sample-rate/pitch makes words obviously janky.
"""

import sys

import numpy as np

sys.path.insert(0, "dev/DTgRPCconnector")
from drawthings_client import DrawThingsClient, ImageGenerationConfig  # noqa: E402

SERVER = "192.168.2.150:7859"
ROOT_CA = "dev/DTgRPCconnector/root_ca.crt"

# Speech prompt: wrong pitch/sync is immediately audible in words
PROMPT = (
    'A woman looks into the camera and says clearly: "Testing one two three. '
    'The quick brown fox jumps over the lazy dog." Close-up, natural lighting.'
)

NUM_FRAMES = 121  # 4.84s @ 25fps — enough for a couple of spoken sentences
FPS = 25  # LTX output is fixed at 25 fps
AUDIO_RATE = 48000  # LTX audio is 48 kHz (upstream ModelZoo default is 24000)

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
    num_frames=NUM_FRAMES,
    fps_id=FPS,
    motion_bucket_id=127,
    compression_artifacts=0,
    hires_fix=False,
)

print(f"Generating {NUM_FRAMES}-frame video...")
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
    dur = n / 2 / AUDIO_RATE
    print(f"Decoded audio: {n} float32 samples = {dur:.2f}s stereo @ {AUDIO_RATE}Hz")
    print(f"Expected video duration: {NUM_FRAMES / FPS:.2f}s")
    print(f"Peak: {np.max(np.abs(arr)):.4f}, NaN: {np.isnan(arr).sum()}")

    if abs(dur - NUM_FRAMES / FPS) > 0.15:
        print("WARNING: Audio duration deviates from video duration by > 0.15s")

    from tensor_decoder import tensor_to_pil

    out = client.save_video(
        result.images,
        output_path="outputs/test_video_speech.mp4",
        fps=FPS,
        audio=audio_bytes,
        audio_sample_rate=AUDIO_RATE,
        frame_decoder=tensor_to_pil,
    )
    print(f"Saved: {out}")