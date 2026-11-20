#!/usr/bin/env python3
"""H3 124-frame run with frame-quality capture.

Generates, then saves sample frames + a montage strip and prints per-frame
statistics so a broken ("fakes working") output is visible without watching.
"""

import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "dev/DTgRPCconnector")
from drawthings_client import DrawThingsClient, ImageGenerationConfig  # noqa: E402
from tensor_decoder import tensor_to_pil  # noqa: E402

SERVER = "192.168.2.150:7859"
ROOT_CA = "dev/DTgRPCconnector/root_ca.crt"
OUT = "outputs/h3_124_check"
NUM_FRAMES = 124
FPS = 24
PROMPT = 'A woman looks into the camera and says clearly: "Testing one two three."'

client = DrawThingsClient(SERVER, insecure=False, verify_ssl=False, ssl_cert_path=ROOT_CA)
print("echo:", client.echo("t").message, flush=True)

cfg = ImageGenerationConfig(
    model="fasth3_omni_i4x.ckpt", steps=8, width=832, height=512, cfg_scale=1.0,
    scheduler="DDIM Trailing", seed=42, seed_mode=2, clip_skip=1, shift=12.0,
    batch_count=1, batch_size=1, num_frames=NUM_FRAMES, fps_id=FPS,
    motion_bucket_id=127, compression_artifacts=0, tiled_decoding=False,
    shift_for_audio=3.0,
)

print(f"start {time.strftime('%H:%M:%S')} {NUM_FRAMES} frames", flush=True)
t0 = time.time()
result = client.generate_media(prompt=PROMPT, config=cfg)
print(f"generated {len(result.images)} frames in {time.time()-t0:.1f}s", flush=True)

from pathlib import Path
Path(OUT).mkdir(parents=True, exist_ok=True)

stats = []
thumbs = []
for i, frame in enumerate(result.images):
    with open("/dev/null", "w") as devnull:
        import contextlib
        with contextlib.redirect_stdout(devnull):
            pil = tensor_to_pil(frame).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32)
    gray = arr.mean(axis=2)
    stats.append({"i": i, "mean": float(gray.mean()), "std": float(gray.std())})
    if i % 12 == 0:
        pil.save(f"{OUT}/frame_{i:03d}.png")
        thumbs.append(pil.resize((208, 128)))

# Montage strip of sampled frames
if thumbs:
    strip = Image.new("RGB", (208 * len(thumbs), 128))
    for k, t in enumerate(thumbs):
        strip.paste(t, (208 * k, 0))
    strip.save(f"{OUT}/montage.png")

means = np.array([s["mean"] for s in stats])
stds = np.array([s["std"] for s in stats])
diffs = np.abs(np.diff(means))

print("\n--- frame statistics ---")
print(f"mean brightness: min {means.min():.1f} max {means.max():.1f}")
print(f"std (detail):    min {stds.min():.1f} max {stds.max():.1f}")
print(f"frame-to-frame mean drift: max {diffs.max():.2f}, mean {diffs.mean():.2f}")

flags = []
if stds.min() < 5:
    flags.append(f"{int((stds < 5).sum())} near-uniform frame(s)")
if means.min() < 8:
    flags.append(f"{int((means < 8).sum())} near-black frame(s)")
if stds.max() > 200:
    flags.append("frame(s) with extreme contrast (possible noise)")
if diffs.max() > 40:
    flags.append(f"large brightness jump at frame {int(diffs.argmax())+1}")

print("flags:", ", ".join(flags) if flags else "none")
print(f"audio chunks: {len(result.audio)}")
if result.audio:
    audio = client.decode_audio(result.audio)
    per_ch = len(audio) // 8
    print(f"audio: {per_ch/32000:.2f}s @32kHz stereo")
print("saved:", OUT)
