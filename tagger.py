#!/usr/bin/env python3
"""Local Danbooru tagger (WD tagger v3, ONNX Runtime, CPU).

Turns an image into booru tags for tag-trained models (Pony, Illustrious,
NoobAI) instead of prose. Reference behaviour mirrors the WaifuDiffusion
tagger demo: white-composite RGBA, pad to square, BICUBIC resize to the
model's input size, float32, then RGB -> BGR.

Getting that order wrong still yields plausible-looking tags, just worse
ones, so do not "simplify" prepare_image().
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# WD tagger v3 repos (Apache-2.0). Camie is GPL-3.0 and deliberately unused.
MODELS = {
    "vit": {
        "repo": "SmilingWolf/wd-vit-tagger-v3",
        "label": "WD ViT v3 (fast, 379 MB)",
    },
    "eva02_large": {
        "repo": "SmilingWolf/wd-eva02-large-tagger-v3",
        "label": "WD EVA02-large v3 (better, 1.3 GB)",
    },
}
DEFAULT_MODEL = "vit"

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# Kaomojis must keep their underscores ("0_0" is not the words "0 0").
KAOMOJIS = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=",
    ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|",
    "||_||",
}

# WD v3 label categories
CAT_GENERAL = 0
CAT_CHARACTER = 4
CAT_RATING = 9

DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85


class TaggerError(RuntimeError):
    """Raised when the tagger cannot be loaded or run."""


def availability() -> dict:
    """Report whether the local tagger can run, with an actionable fix.

    Checks the actual interpreter running the app (``sys.executable``), not
    whatever ``pip`` happens to be on PATH — a common failure on second
    machines is installing into a different venv than the one serving the UI.
    """
    import platform
    import sys

    problems = []
    detail = []

    try:
        import onnxruntime as ort  # noqa: F401

        detail.append(f"onnxruntime {ort.__version__}")
    except Exception as e:  # ImportError, OSError (libstdc++/glibc), etc.
        problems.append(("onnxruntime", e))

    try:
        import huggingface_hub as hh

        detail.append(f"huggingface_hub {hh.__version__}")
    except Exception as e:
        problems.append(("huggingface_hub", e))

    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    arch = platform.machine()
    install = f'"{sys.executable}" -m pip install onnxruntime huggingface_hub'

    hint = ""
    if sys.version_info < (3, 11):
        hint = (
            f"Python {py} has no onnxruntime 1.32 wheel; "
            "use: pip install 'onnxruntime<1.23'"
        )
    if any(isinstance(err, OSError) for _, err in problems):
        hint = (
            "onnxruntime installed but its binary failed to load — needs "
            "glibc >= 2.28 and libstdc++ (e.g. 'apt install libstdc++6')."
        )

    if problems:
        detail_txt = "; ".join(f"{name}: {type(err).__name__}: {err}" for name, err in problems)
        message = (
            f"Local tagger unavailable ({detail_txt}).\n"
            f"Fix with: {install}\n"
            f"Interpreter: {sys.executable} (Python {py}, {arch})"
        )
        if hint:
            message += f"\nNote: {hint}"
        return {"ok": False, "message": message, "install": install, "detail": detail}

    return {
        "ok": True,
        "message": f"Local tagger ready ({', '.join(detail)}; Python {py}, {arch})",
        "install": install,
        "detail": detail,
    }


class DanbooruTagger:
    """Lazy-loading wrapper around a WD tagger ONNX model."""

    def __init__(self, model_key: str = DEFAULT_MODEL, models_dir: Optional[Path] = None):
        if model_key not in MODELS:
            raise TaggerError(
                f"Unknown tagger model '{model_key}'. Available: {', '.join(MODELS)}"
            )
        self.model_key = model_key
        self.models_dir = Path(models_dir or _default_models_dir())
        self._session = None
        self._input_name: Optional[str] = None
        self._target_size: Optional[int] = None
        self._tag_names: List[str] = []
        self._general_idx: List[int] = []
        self._character_idx: List[int] = []
        self._rating_idx: List[int] = []

    # -- loading ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - dependency guard
            raise TaggerError(
                "onnxruntime is required for the local tagger. "
                "Install: pip install onnxruntime"
            ) from e

        model_path, labels_path = self._download_files()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        with open(labels_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        names: List[str] = []
        general: List[int] = []
        character: List[int] = []
        rating: List[int] = []
        for i, row in enumerate(rows):
            raw = row["name"]
            names.append(raw if raw in KAOMOJIS else raw.replace("_", " "))
            try:
                cat = int(row["category"])
            except (KeyError, TypeError, ValueError):
                cat = CAT_GENERAL
            if cat == CAT_RATING:
                rating.append(i)
            elif cat == CAT_CHARACTER:
                character.append(i)
            else:
                general.append(i)

        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inp = session.get_inputs()[0]
        shape = inp.shape  # NHWC: [1, H, W, 3]

        self._target_size = int(shape[1])
        self._input_name = inp.name
        self._tag_names = names
        self._general_idx = general
        self._character_idx = character
        self._rating_idx = rating
        self._session = session

    def _download_files(self) -> Tuple[Path, Path]:
        repo = MODELS[self.model_key]["repo"]
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover - dependency guard
            raise TaggerError(
                "huggingface_hub is required to fetch tagger weights. "
                "Install: pip install huggingface_hub"
            ) from e

        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Cache both files under a stable directory so repeat runs are offline.
        model_path = hf_hub_download(
            repo_id=repo, filename=MODEL_FILENAME, cache_dir=str(self.models_dir)
        )
        labels_path = hf_hub_download(
            repo_id=repo, filename=LABEL_FILENAME, cache_dir=str(self.models_dir)
        )
        return Path(model_path), Path(labels_path)

    # -- inference -------------------------------------------------------
    def prepare_image(self, image: Image.Image) -> np.ndarray:
        """Match the reference WD tagger preprocessing exactly."""
        self._ensure_loaded()
        assert self._target_size is not None

        if image.mode != "RGBA":
            image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        rgb = canvas.convert("RGB")

        w, h = rgb.size
        max_dim = max(w, h)
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(rgb, ((max_dim - w) // 2, (max_dim - h) // 2))

        if max_dim != self._target_size:
            padded = padded.resize(
                (self._target_size, self._target_size), Image.BICUBIC
            )

        arr = np.asarray(padded, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR (reference client does this)
        return np.expand_dims(arr, axis=0)

    def tag(
        self,
        image: Image.Image,
        general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
        character_threshold: float = DEFAULT_CHARACTER_THRESHOLD,
        mcut_general: bool = False,
        mcut_character: bool = False,
    ) -> Dict[str, object]:
        """Tag an image. Returns rating, character, general and rating_scores."""
        self._ensure_loaded()
        assert self._session is not None and self._input_name is not None

        batch = self.prepare_image(image)
        probs = self._session.run(None, {self._input_name: batch})[0][0]

        def collect(indexes: List[int], threshold: float, use_mcut: bool):
            values = [(self._tag_names[i], float(probs[i])) for i in indexes]
            if use_mcut and values:
                threshold = _mcut_threshold(np.array([v for _, v in values]))
            picked = [(name, score) for name, score in values if score >= threshold]
            picked.sort(key=lambda kv: kv[1], reverse=True)
            return picked

        rating_scores = {
            self._tag_names[i]: float(probs[i]) for i in self._rating_idx
        }
        rating = (
            max(rating_scores, key=rating_scores.get) if rating_scores else None
        )

        return {
            "rating": rating,
            "rating_scores": rating_scores,
            "character": collect(
                self._character_idx, character_threshold, mcut_character
            ),
            "general": collect(self._general_idx, general_threshold, mcut_general),
        }

    def format_tags(
        self,
        tags: Dict[str, object],
        include_character: bool = True,
        quality_prefix: str = "",
        trigger_word: str = "",
        max_general: int = 75,
    ) -> str:
        """Build a booru-style prompt: [trigger,] [quality,] character, general."""
        parts: List[str] = []
        if trigger_word.strip():
            parts.append(trigger_word.strip())
        if quality_prefix.strip():
            parts.extend(t.strip() for t in quality_prefix.split(",") if t.strip())
        if include_character:
            parts.extend(name for name, _ in tags.get("character", []) or [])
        general = tags.get("general", []) or []
        parts.extend(name for name, _ in general[:max_general])

        # De-duplicate while preserving order
        seen = set()
        out: List[str] = []
        for p in parts:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
        return ", ".join(out)


def _mcut_threshold(probs: np.ndarray) -> float:
    """Maximum Cut Thresholding (Largeron et al. 2012)."""
    if probs.size < 2:
        return float(probs[0]) if probs.size else 1.0
    ordered = np.sort(probs)[::-1]
    diffs = ordered[:-1] - ordered[1:]
    i = int(np.argmax(diffs))
    return float((ordered[i] + ordered[i + 1]) / 2)


def _default_models_dir() -> Path:
    return Path(__file__).parent / "models" / "tagger"


_TAGGER_CACHE: Dict[str, DanbooruTagger] = {}


def get_tagger(model_key: str = DEFAULT_MODEL) -> DanbooruTagger:
    """Return a cached tagger for ``model_key`` (weights load on first use)."""
    if model_key not in _TAGGER_CACHE:
        _TAGGER_CACHE[model_key] = DanbooruTagger(model_key)
    return _TAGGER_CACHE[model_key]
