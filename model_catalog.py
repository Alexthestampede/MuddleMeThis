#!/usr/bin/env python3
"""Model catalog: server models enriched with family, capability and preset.

The Draw Things server reports only ``file``/``name``/``version``. Everything
else an operator needs before generating - image vs video, which preset fits,
how many steps - is derived here, once, at connect time.

Name matching is deliberately strict (exact, or front-containment of the
normalized names). A looser rule once matched ``qwen_image_1.0`` onto
``Qwen Image 2.1 Turbo`` and silently applied the wrong steps.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

VIDEO_VERSIONS = {
    "ltx2",
    "ltx2.3",
    "minimax_h3",
    "wan_v2.1_1.3b",
    "wan_v2.1_14b",
    "wan_v2.2_5b",
    "hunyuan_video",
    "svd_i2v",
    "longcat_video_avatar_v1.5",
}

UPSCALER_VERSIONS = {"seedvr2_3b", "seedvr2_7b"}

# Version -> short family label used when no preset matches.
FAMILY_BY_VERSION = {
    "v1": "SD 1.5",
    "v2": "SD 2.x",
    "sdxl": "SDXL",
    "sdxl_base_v0.9": "SDXL",
    "sdxl_refiner_v0.9": "SDXL Refiner",
    "flux1": "FLUX.1",
    "flux2_4b": "FLUX.2 4B",
    "flux2_9b": "FLUX.2 9B",
    "flux3_5b": "FLUX.3 Klein",
    "z_image": "Z-Image",
    "qwen_image": "Qwen Image",
    "qwen_image_2.1": "Qwen Image 2.1",
    "ernie_image": "Ernie",
    "moss_image": "MOSS",
    "cosmos2.5_2b": "Cosmos 2.5 2B",
    "cosmos2.5_2.9b": "Cosmos 2.5 2.9B",
    "ideogram_4": "Ideogram 4",
    "krea_2": "Krea 2",
    "hidream_i1": "HiDream i1",
    "hidream_o1": "HiDream o1",
    "sd3": "SD3",
    "sd3_large": "SD3 Large",
    "pixart": "PixArt",
    "auraflow": "AuraFlow",
    "kandinsky2.1": "Kandinsky 2.1",
    "ssd_1b": "SSD 1B",
    "ltx2": "LTX 2",
    "ltx2.3": "LTX 2.3",
    "minimax_h3": "MiniMax H3",
    "minimax_h3_image": "MiniMax H3 Image",
    "wan_v2.1_1.3b": "Wan 2.1 1.3B",
    "wan_v2.1_14b": "Wan 2.1 14B",
    "wan_v2.2_5b": "Wan 2.2 5B",
    "hunyuan_video": "Hunyuan Video",
    "svd_i2v": "SVD I2V",
    "longcat_video_avatar_v1.5": "LongCat Video",
    "seedvr2_3b": "SeedVR2 3B",
    "seedvr2_7b": "SeedVR2 7B",
}

VIDEO_HINTS = ("vid", "video", "ltx", "h3", "wan", "hunyuan", "svd")
UPSCALER_HINTS = ("upscaler", "seedvr")


def normalize(text: str) -> str:
    """Lowercase alphanumeric tokens only, for safe name comparison."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


_QUANT_TOKEN = re.compile(
    r"\b(?:q\d+p|i\d+x|f16|f32|bf16|\d+\s*bit|ckpt|checkpoint|official|model)\b"
)


def canonical(text: str) -> str:
    """Strip quantisation noise, then normalize; bracket *contents* survive.

    Brackets often carry the meaningful name ("FLUX.3 [klein] 5.6B"), so they
    become spaces rather than being deleted. Quant tokens inside them
    ("(6-bit)", "(4-bit S)") are removed separately.
    """
    text = re.sub(r"[\(\[\{\)\]\}]", " ", (text or "").lower())
    text = _QUANT_TOKEN.sub(" ", text)
    return normalize(text)


def classify(version: str, name: str = "", file: str = "") -> str:
    """Return a coarse family key (used only when no preset matches)."""
    version = (version or "").lower()
    label = normalize(f"{name} {file}")

    if version in FAMILY_BY_VERSION:
        return version
    if version.startswith("qwen_image"):
        return "qwen_image_2.1" if "2.1" in version else "qwen_image"
    if "chroma" in label:
        return "chroma"
    if any(tok in label for tok in ("pony", "illustrious", "noob")):
        return "pony"
    if "schnell" in label:
        return "flux1"
    if any(tok in label for tok in UPSCALER_HINTS):
        return "seedvr2_7b"
    return "generic"


def supports(version: str, name: str = "", file: str = "") -> str:
    """Return 'image', 'video' or 'upscaler'."""
    version = (version or "").lower()
    label = normalize(f"{name} {file}")
    # MiniMax H3 has both a video model and an image model; the *_image
    # variant must not be swept up by the "h3" video hint below.
    if version.endswith("_image") or " h3 image" in f" {label} ":
        return "image"
    if version in UPSCALER_VERSIONS or any(tok in label for tok in UPSCALER_HINTS):
        return "upscaler"
    if version in VIDEO_VERSIONS or any(tok in label for tok in VIDEO_HINTS):
        return "video"
    return "image"


def preset_model_file(preset: Dict[str, Any]) -> str:
    """Preset schemas in this repo differ; find the bound checkpoint if any."""
    for key in ("model", "file", "model_file"):
        value = preset.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _preset_numeric(preset: Dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in preset and preset[key] is not None:
            return preset[key]
    return default


def match_preset(
    entry: Dict[str, Any], presets: Dict[str, Dict[str, Any]]
) -> Optional[tuple]:
    """Find the preset for a model entry. Returns (preset_key, preset) or None.

    Ranking: exact canonical equality beats containment, and among containment
    candidates the longest preset name wins - so "Qwen Image 2.1 Turbo"
    resolves to the turbo preset rather than the base one.
    """
    file = (entry.get("file") or "").lower()
    canon_model = canonical(entry.get("name") or file)

    # 1. Explicit binding (presets that carry a "model" filename). Compare
    #    canonically so quant variants of the same checkpoint still bind.
    canon_file = canonical(file)
    for key, preset in presets.items():
        bound = preset_model_file(preset)
        if bound and canonical(bound) and canonical(bound) == canon_file:
            return key, preset

    best_key, best_preset, best_score = None, None, 0
    for key, preset in presets.items():
        canon_preset = canonical(
            preset.get("name") or preset_model_file(preset) or key
        )
        if not canon_preset or not canon_model:
            continue
        if canon_preset == canon_model:
            score = 1000 + len(canon_preset)
        elif canon_preset in canon_model:
            score = len(canon_preset)
        elif canon_model in canon_preset:
            score = len(canon_model)
        else:
            continue
        if len(canon_preset if score < 1000 else canon_model) < 4:
            continue  # too generic to be meaningful
        if score > best_score:
            best_key, best_preset, best_score = key, preset, score

    if best_key:
        return best_key, best_preset
    return None


def _token_overlap(canon_model: str, canon_preset: str) -> int:
    """Count distinct shared distinctive tokens (>=3 chars, not pure digits)."""
    model_tokens = {t for t in canon_model.split() if len(t) >= 3 and not t.isdigit()}
    hits = 0
    for token in set(canon_preset.split()):
        if len(token) < 3 or token.isdigit():
            continue
        if any(token in mt for mt in model_tokens):
            hits += 1
    return hits


def is_video_preset(preset: Dict[str, Any]) -> bool:
    if _preset_numeric(preset, "num_frames", "numFrames") is not None:
        return True
    return "video model" in str(preset.get("notes", "")).lower()


def match_preset_loose(entry: Dict[str, Any], presets: Dict[str, Dict[str, Any]]):
    """Token-overlap fallback for renamed community variants (e.g. FastH3).

    Requires at least two shared distinctive tokens, so a lone "krea" cannot
    bind "Krea 2 Raw" to the combined "Flux dev/Krea/Kontext" preset.
    """
    canon_model = canonical(entry.get("name") or entry.get("file") or "")
    want_video = entry.get("supports") == "video"
    best = None
    best_score = 0
    for key, preset in presets.items():
        if is_video_preset(preset) != want_video:
            continue
        canon_preset = canonical(preset.get("name") or preset_model_file(preset) or key)
        score = _token_overlap(canon_model, canon_preset)
        if score >= 2 and score > best_score:
            best, best_score = (key, preset), score
    return best


def match_by_family(entry: Dict[str, Any], presets: Dict[str, Dict[str, Any]]):
    """Match by server version -> family label -> preset name.

    Works for renamed community variants: "FastH3 Omni" reports version
    minimax_h3, whose family label canonicalises to the "MiniMax H3" preset.
    """
    family = FAMILY_BY_VERSION.get(
        classify(entry.get("version", ""), entry.get("name", ""), entry.get("file", ""))
    )
    if not family:
        return None
    canon_family = canonical(family)
    want_video = entry.get("supports") == "video"
    for key, preset in presets.items():
        if is_video_preset(preset) != want_video:
            continue
        if canonical(preset.get("name") or "") == canon_family:
            return key, preset
    return None


def build_catalog(
    raw_models: Iterable[Dict[str, Any]],
    presets: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Enrich server model entries for the picker."""
    catalog: List[Dict[str, Any]] = []
    for entry in raw_models or []:
        file = entry.get("file")
        if not file:
            continue
        name = entry.get("name") or file
        version = entry.get("version") or ""
        kind = supports(version, name, file)
        lookup = {"file": file, "name": name, "supports": kind, "version": version}
        match = (
            match_preset(lookup, presets)
            or match_preset_loose(lookup, presets)
            or match_by_family(lookup, presets)
        )
        preset_key = match[0] if match else ""
        preset = match[1] if match else {}

        steps = _preset_numeric(preset, "steps", "recommended_steps")
        cfg = _preset_numeric(preset, "guidanceScale", "recommended_cfg")
        sampler = preset.get("sampler")
        shift = preset.get("shift")
        family = FAMILY_BY_VERSION.get(classify(version, name, file), "Generic")

        badges = [family]
        if kind == "video":
            badges.append("video")
        elif kind == "upscaler":
            badges.append("upscaler")
        if steps:
            badges.append(f"{steps} steps")

        catalog.append(
            {
                "file": file,
                "name": name,
                "version": version,
                "supports": kind,
                "family": family,
                "preset": preset_key,
                "preset_name": preset.get("name", "") if preset else "",
                "steps": steps,
                "cfg": cfg,
                "sampler": sampler,
                "shift": shift,
                "notes": preset.get("notes", "") if preset else "",
                "label": f"{name} · " + " · ".join(badges),
            }
        )
    catalog.sort(key=lambda e: (e["name"] or e["file"]).lower())
    return catalog


def filter_catalog(
    catalog: List[Dict[str, Any]],
    query: str = "",
    hide_non_image: bool = False,
    only_video: bool = False,
) -> List[Dict[str, Any]]:
    q = normalize(query)
    out = []
    for entry in catalog:
        if only_video and entry["supports"] != "video":
            continue
        if hide_non_image and entry["supports"] != "image":
            continue
        if q:
            haystack = normalize(
                f"{entry['name']} {entry['file']} {entry['family']} {entry.get('version','')}"
            )
            if q not in haystack:
                continue
        out.append(entry)
    return out


def find(catalog: List[Dict[str, Any]], name_or_file: str) -> Optional[Dict[str, Any]]:
    if not name_or_file:
        return None
    for entry in catalog:
        if name_or_file in (entry["name"], entry["file"]):
            return entry
    return None


def model_info_markdown(entry: Optional[Dict[str, Any]]) -> str:
    if not entry:
        return ""
    lines = [f"**{entry['name']}**"]
    lines.append(f"`{entry['file']}`")
    meta = [entry["family"]]
    if entry["supports"] != "image":
        meta.append(entry["supports"])
    if entry.get("steps"):
        meta.append(f"{entry['steps']} steps")
    if entry.get("cfg") is not None:
        meta.append(f"cfg {entry['cfg']}")
    if entry.get("sampler"):
        meta.append(str(entry["sampler"]))
    if entry.get("shift") is not None:
        meta.append(f"shift {entry['shift']}")
    lines.append(" · ".join(meta))
    if entry.get("preset_name"):
        lines.append(f"Preset: **{entry['preset_name']}**")
    else:
        lines.append("_No matching preset — custom settings apply._")
    if entry.get("notes"):
        lines.append(entry["notes"])
    return "\n\n".join(lines)
