#!/usr/bin/env python3
"""Fast smoke test: FLUX.2 Klein 4B (kontext, 4 steps).

Chosen as the default test model because it is small, fast (~12 s for 1024²
on .150) and an edit/kontext model, so it exercises the reference-image path
too. Small models are not more reliable than SD1.5 - which is the point: a
quick run you will actually look at beats a slow one you skip.

A request can "succeed" and return noise (unresolved model name, wrong
sampler, missing metadata conditioning), so this script prints image
statistics and saves the PNG. LOOK AT IT.

Usage:
    python test_smoke.py            # 1024², untiled
    python test_smoke.py --tiled    # 1024² with 512px tiles
    python test_smoke.py --edit     # also exercise the reference-image path
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "dev/DTgRPCconnector")
from drawthings_client import DrawThingsClient, ImageGenerationConfig  # noqa: E402
from tensor_decoder import tensor_to_pil  # noqa: E402

SERVER = "192.168.2.150:7859"
ROOT_CA = "dev/DTgRPCconnector/root_ca.crt"
MODEL = "flux_2_klein_4b_i8x.ckpt"  # exact server filename - case matters
PROMPT = "a red ball on a white table"
PROMPT_EDIT = "make the ball bright blue, keep everything else the same"


def generate(client, tiled=False, reference=None, prompt=PROMPT):
    config = ImageGenerationConfig(
        model=MODEL,
        steps=4,  # Klein distilled: 4 steps, CFG 1, DDIM Trailing, shift 3
        width=1024,
        height=1024,
        cfg_scale=1.0,
        scheduler="DDIM Trailing",
        seed=11,
        shift=3.0,
        clip_skip=2,
        strength=1.0,  # kontext/edit models require strength 1.0
        tiled_decoding=tiled,
        decoding_tile_width=8,  # 512 px in scale units (px // 64)
        decoding_tile_height=8,
        decoding_tile_overlap=1,
    )
    started = time.time()
    images = client.generate_image(
        prompt=prompt, config=config, input_image=reference
    )
    return images, time.time() - started


def report(label, images, elapsed):
    if not images:
        print(f"{label:16s} FAILED: no images returned")
        return False
    image = tensor_to_pil(images[0]).convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    out = Path("/tmp/opencode") / f"smoke_{label.replace(' ', '_')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    # A correct render of a ball on a table: plenty of detail, mid brightness.
    # Pure noise also scores high on std, so the number is a tripwire for
    # black/flat/empty output - the eye is the real test.
    print(f"{label:16s} {elapsed:5.1f}s std={array.std():5.1f} mean={array.mean():5.1f} -> {out}")
    if array.std() < 5:
        print("  ⚠️ near-uniform image (flat/blank) - likely a failed run")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiled", action="store_true", help="enable 512px tiled decoding")
    parser.add_argument("--edit", action="store_true", help="also run a reference-image edit")
    args = parser.parse_args()

    client = DrawThingsClient(SERVER, insecure=False, verify_ssl=False, ssl_cert_path=ROOT_CA)
    print("echo:", client.echo("smoke").message)

    ok = report("untiled", *generate(client))
    if args.tiled:
        ok &= report("tiled 512px", *generate(client, tiled=True))
    if args.edit:
        base_images, _ = generate(client)
        if not base_images:
            print("edit (reference)  skipped: base generation failed")
            return 1 if ok else 1
        reference = tensor_to_pil(base_images[0]).convert("RGB")
        ok &= report(
            "edit (reference)",
            *generate(client, reference=reference, prompt=PROMPT_EDIT),
        )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
