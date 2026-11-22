#!/usr/bin/env python3
"""Diagnose LTX audio tensor layout: header dims, fpzip shape, rate analysis."""

import struct
import sys

import numpy as np

sys.path.insert(0, "dev/DTgRPCconnector")
from drawthings_client import DrawThingsClient, ImageGenerationConfig  # noqa: E402
import fpzip  # noqa: E402

SERVER = "192.168.2.150:7859"
ROOT_CA = "dev/DTgRPCconnector/root_ca.crt"

client = DrawThingsClient(SERVER, insecure=False, verify_ssl=False, ssl_cert_path=ROOT_CA)
print("Connected:", client.echo("test").message)

NUM_FRAMES = 121

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
    fps_id=25,
    motion_bucket_id=127,
    compression_artifacts=0,
)

result = client.generate_media(
    prompt="a red car driving down a desert highway",
    config=config,
)

video_duration = NUM_FRAMES / 25
print(f"\n=== VIDEO: {len(result.images)} frames, {video_duration:.3f}s @ 25fps ===")

for i, chunk in enumerate(result.audio):
    header = struct.unpack_from("<32I", chunk, 0)
    print(f"\n=== AUDIO CHUNK {i}: {len(chunk):,} bytes ===")
    print(f"  header[0] magic: {header[0]}")
    print(f"  header[4] dim0/batch: {header[4]}")
    print(f"  header[5] dim1: {header[5]}")
    print(f"  header[6] dim2: {header[6]}")
    print(f"  header[7] dim3: {header[7]}")
    print(f"  header[8] dim4: {header[8]}")
    print(f"  header[9..12]: {header[9:13]}")

    arr = fpzip.decompress(chunk[68:], order="C")
    print(f"  fpzip shape: {arr.shape}, dtype: {arr.dtype}")

    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    print(f"  flat samples: {flat.size}")
    print(f"  peak: {np.max(np.abs(flat)):.4f}, NaN: {np.isnan(flat).sum()}")

    # Duration hypotheses
    for label, frames, rate in [
        ("interleaved stereo @44100", flat.size // 2, 44100),
        ("interleaved stereo @48000", flat.size // 2, 48000),
        ("planar 2ch @44100 (per chan)", flat.size // 2, 44100),
        ("planar 2ch @48000 (per chan)", flat.size // 2, 48000),
        ("mono @44100", flat.size, 44100),
        ("mono @48000", flat.size, 48000),
    ]:
        dur = frames / rate
        match = " <<<" if abs(dur - video_duration) < 0.15 else ""
        print(f"  {label}: {dur:.3f}s{match}")

    # Channel similarity if 2D-ish shape: check if first half correlates with second
    half = flat.size // 2
    a, b = flat[:half], flat[half:]
    if a.size == b.size and np.std(a) > 1e-9 and np.std(b) > 1e-9:
        corr = np.corrcoef(a, b)[0, 1]
        print(f"  corr(first half, second half): {corr:.4f}  (planar L/R would be ~1)")

    # Autocorrelation at lag = half size (repetition check)
    if flat.size > 4:
        seg1 = flat[: flat.size // 4].astype(np.float64)
        seg2 = flat[flat.size // 4 : flat.size // 2].astype(np.float64)
        n = min(len(seg1), len(seg2))
        if np.std(seg1[:n]) > 1e-9 and np.std(seg2[:n]) > 1e-9:
            c = np.corrcoef(seg1[:n], seg2[:n])[0, 1]
            print(f"  corr(quarter1, quarter2): {c:.4f}  (repetition would be high)")