#!/usr/bin/env python3
"""Long H3 run: 294 frames, 256px tiled decoding (operator's saved preset).

Purpose is twofold: produce a clip and record the memory envelope for a
much longer sequence than the 124-frame baseline.

    python test_h3_long.py

Writes outputs/h3_294_tiled.mp4 plus sampled frames/montage and prints
per-frame statistics (a memory-starved run still returns frames).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "dev/DTgRPCconnector")
from drawthings_client import DrawThingsClient, ImageGenerationConfig  # noqa: E402
from tensor_decoder import tensor_to_pil  # noqa: E402

SERVER = "192.168.2.150:7859"
ROOT_CA = "dev/DTgRPCconnector/root_ca.crt"
MODEL = "fasth3_omni_i4x.ckpt"
NUM_FRAMES = 294  # operator's preset; ~12.25 s at 24 fps
FPS = 24
AUDIO_RATE = 32000
TILE_PX = 256  # operator's preset (4 scale units)
OVERLAP_PX = 64
OUT_DIR = Path("outputs/h3_294_check")

PROMPT = (
    'A woman looks into the camera and says clearly: "Testing one two three. '
    'The quick brown fox jumps over the lazy dog." Close-up, natural lighting.'
)

client = DrawThingsClient(SERVER, insecure=False, verify_ssl=False, ssl_cert_path=ROOT_CA)
print("echo:", client.echo("t").message, flush=True)

config = ImageGenerationConfig(
    model=MODEL,
    steps=8,
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
    tiled_decoding=True,
    decoding_tile_width=TILE_PX // 64,
    decoding_tile_height=TILE_PX // 64,
    decoding_tile_overlap=OVERLAP_PX // 64,
    shift_for_audio=3.0,
)

print(f"start {time.strftime('%H:%M:%S')} | {NUM_FRAMES} frames | 832x512 | "
      f"{TILE_PX}px tiles/{OVERLAP_PX}px overlap", flush=True)
t0 = time.time()
result = client.generate_media(
    prompt=PROMPT,
    config=config,
    preview_callback=lambda _b: None,
)
elapsed = time.time() - t0
print(f"generated {len(result.images)} frames in {elapsed:.0f}s "
      f"({elapsed/max(len(result.images),1):.1f}s/frame)", flush=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
means, stds, thumbs = [], [], []
for i, frame in enumerate(result.images):
    import contextlib, os
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        pil = tensor_to_pil(frame).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    gray = arr.mean(axis=2)
    means.append(float(gray.mean()))
    stds.append(float(gray.std()))
    if i % 24 == 0:
        pil.save(OUT_DIR / f"frame_{i:03d}.png")
        thumbs.append(pil.resize((160, 98)))

if thumbs:
    from PIL import Image

    strip = Image.new("RGB", (160 * len(thumbs), 98))
    for k, t in enumerate(thumbs):
        strip.paste(t, (160 * k, 0))
    strip.save(OUT_DIR / "montage.png")

print("\n--- frame statistics ---", flush=True)
if means:
    print(f"frames decoded: {len(means)}")
    print(f"brightness: min {min(means):.1f} max {max(means):.1f}")
    print(f"detail std: min {min(stds):.1f} max {max(stds):.1f}")
    flat = sum(1 for s in stds if s < 5)
    print(f"near-uniform frames: {flat}")
print(f"audio chunks: {len(result.audio)}")
if result.audio:
    audio = client.decode_audio(result.audio)
    print(f"audio: {len(audio)//8/AUDIO_RATE:.2f}s @ {AUDIO_RATE}Hz stereo")
    out = "outputs/test_h3_294_tiled.mp4"
    client.save_video(result.images, out, fps=FPS, audio=audio,
                      audio_sample_rate=AUDIO_RATE)
    print("saved:", out, flush=True)
