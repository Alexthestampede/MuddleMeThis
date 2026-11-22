# AGENTS.md - MuddleMeThis

Gradio app connecting vision LLMs (LM Studio/Ollama) to Draw Things gRPC for AI prompt engineering, image, and video (LTX 2.3 with audio) generation.

## Setup (Non-obvious)

```bash
# 1. Main deps
pip install -r requirements.txt

# 2. DTgRPCconnector (gRPC client for Draw Things) - MUST install separately
pip install -r dev/DTgRPCconnector/requirements.txt
# Python 3.13+ aarch64: pip install "flatbuffers>=24.3.0"  # override for piwheels

# 3. ModuLLe (LLM abstraction) - MUST install as editable
cd dev/ModuLLe && pip install -e . && cd ../..
```

**Why**: `dev/` packages are not installed transitively. Both must be present or the app fails at runtime.

## Running

```bash
./launch.sh        # Preferred - auto-detects and activates venv
python app.py      # Direct (requires manual venv activation)
```

Access: http://localhost:7860

## Entry Points & Structure

- `app.py` — Main Gradio app (~3500 lines). Video tab wired near the end (`create_ui`). Actor Sheet tab renders the Cutscene Artist reference-sheet prompt and passes the uploaded photo as an edit reference.
- `settings_manager.py` — Config persistence (JSON).
- `dev/DTgRPCconnector/` — gRPC client. **Synced from dtline (latest: 27c4b31 / 20261121.2)**; keep our audio-mux + keepalive patches when re-syncing: `save_images()` decodes CCV tensors (incl. RGBA), `save_video()` auto-detects tensors, `decode_audio()`/`mux_audio_into_video()` are ours. `drawthings_client.py` (client + audio muxing), `tensor_decoder.py` (`tensor_to_pil`, `decode_audio_tensor`), `GenerationConfiguration.py` (FlatBuffer schema; slots 86+ = newer server fields).
- `dev/ModuLLe/` — LLM provider abstraction (Ollama/LM Studio/OpenAI/Gemini/Claude), synced from github.com/Alexthestampede/ModuLLe (calendar-versioned). Editable install; its pyproject pulls `beautifulsoup4`, `html2text`, `ddgs`, `lxml` for web/tools modules (`modulle/__init__` imports lazily, so the app works without them).
- `settings/config.json` — User settings (gitignored, auto-created); reference: `config.example.json`.
- `settings/presets/*.json` — Model presets. **Our schema stores Draw Things keys** (`guidanceScale`, `hiresFix`, `resolutionDependentShift`) — dtline uses different keys (`recommended_cfg`, `hires_fix`); don't mix.
- `settings/prompts/*.txt` — System prompts (expand.txt, extract.txt, refine.txt).

## Critical Domain Knowledge

### gRPC Scale Factors (NOT pixels)
Draw Things gRPC config uses scale factors: `scale = pixels ÷ 64`. The client converts `ImageGenerationConfig(width=...)` from pixels internally — but `hires_fix_start_width/height` are passed raw, so divide by 64 yourself (app.py does `hires_fix_width // 64`).

Wrong scale crashes the server: `[cnnp_reshape_build] dim: (2, 262144, 320)`.

### LTX Video + Audio (hard-earned, verified empirically)
- **Audio chunks are NOT raw PCM**: each `generatedAudio` entry is a reassembled fpzip CCV tensor (68-byte header, magic `1012247`), shape `(1, 1, 2, samples)` with **planar** channels (all L, then all R). Decode via `decode_audio_tensor()` / `client.decode_audio()` BEFORE joining — blindly treating as PCM = full-scale static.
- **Audio rate is 48 kHz** (not 44.1). samples/channel = `1920×num_frames − 1440`. Playing 48k at 44.1k = sped-up/high-pitched audio.
- **Frame rate fixed at 25 fps** regardless of `fps_id`. Valid frame counts: `(n−1) % 8 == 0`, range 9–257 (9, 17, 57, 121...). Official preset uses 121.
- **First-frame conditioning (I2V)**: pass the starting image as `input_image` with `strength=1.0`. `reference_images` + `hint_type="shuffle"` is for edit/kontext models only — LTX ignores it.
- **Hires fix for LTX** = two-stage latent spatial upscaling (LTX-native ×2/×1.5 upscaler, no external upscaler file). Auto-triggers when first-pass size is exactly 1/2 or 2/3 of final. Preset: 640×384 → 1280×768.
- Muxing: NaN/Inf audio chunks happen; sanitize, ensure even float32 sample counts, mux via intermediate WAV file — `DrawThingsClient.mux_audio_into_video()`.

### Actor Sheet tab (prompt-only Storyflow method)
- One photo → 2048×1152 sheet (face left, three views right) from Cutscene Artist's 4-section prompt. The photo is passed as a `ReferenceImage` hint (`hint_type="shuffle"`), so **only edit/kontext models can see it** — a text-only checkpoint renders a stranger.
- Reference images are letterboxed to the render aspect (`_fit_on_white`) because the connector resizes hint tensors to the config size; stretching a portrait squashes the face.
- `describe_actor()` fills section 2 via the vision processor, capped at 256 tokens. Costume is free text with the demo's wildcard list and a randomiser.
- Saved sheets are ordinary PNGs in `outputs/` with Draw Things XMP; they are meant to be re-attached as moodboard/reference for LTX/H3 video.

### TLS / gRPC connection
TLS is the default. Self-signed certs need the localhost name override — already in our connector: `grpc.ssl_target_name_override = "localhost"` + `ssl_cert_path` to `dev/DTgRPCconnector/root_ca.crt` (app.py lines ~743).

### Local Danbooru tagger (`tagger.py`)
Image → booru tags for Pony/Illustrious/NoobAI, offline via ONNX Runtime (CPU, no torch).
- Models: `wd-vit-tagger-v3` (default, 379 MB, ~0.5 s/image) / `wd-eva02-large-tagger-v3` (opt-in, 1.3 GB). Both Apache-2.0. **Camie tagger is GPL-3.0 — do not add it to this MIT repo.**
- Weights download to `models/tagger/` (gitignored) through `huggingface_hub`.
- **Preprocessing must stay exact**: white RGBA composite → pad to square → BICUBIC resize to the model input (448) → float32 → **RGB→BGR**. Wrong order still returns plausible tags, just worse ones.
- Labels: keep kaomoji underscores (`0_0`, `>_<`), convert all other `_` to spaces. Categories: 0 general, 4 character, 9 rating. Defaults 0.35 general / 0.85 character; MCut optional.
- Ratings are computed but deliberately **not emitted** into prompts.

### LLM provider timeouts (`dev/ModuLLe`)
Ollama unloads models after 5 min idle, so every cold call pays model load + vision prefill (measured on .150: **29 s cold vs 0.03 s warm** for a 1.7B model). On a laptop that easily exceeds the HTTP timeout.
- Generation calls use `request_timeout * 3`; base default 60 s → 180 s.
- App passes `keep_alive` (default `30m`) and `think=False` for Ollama vision calls (`_ollama_extras()`), plus a configurable `llm_timeout` from Settings.
- **LM Studio does not accept those kwargs** — `_ollama_extras()` returns `{}` for it; keep provider branching when adding new options.
- ModuLLe's `generate()`/`chat()`/`analyze_image()` swallow exceptions and return `None`, so the app measures elapsed time to distinguish a timeout from an empty response (`_llm_failure_message`).

**Every LLM call must cap output** (`llm_max_tokens`, sent as `num_predict`, default 700). Nothing capped generation before, so a small model that started looping echoed the entire instruction block back until it exhausted the context — the "loops the whole context" hang. Extraction/style also run at low temperature (0.2–0.3) and send the instruction file as a **system** prompt, not as the user prompt.

**Shared GPU: release LLM VRAM before generating.** When Ollama and Draw Things run on the same host, a warm model (keep_alive) can hold ~2 GB and Draw Things dies mid-run with `Stream removed (Socket closed)` — verified on .150: generation failed with a model resident, succeeded in 9.4 s right after unloading. `_release_llm_vram()` posts `keep_alive: 0` to `/api/generate` for every model in `/api/ps` before each image/edit/video run, but only when `llm_server` and `grpc_server` share a host (`release_llm_before_generate`, default on).

### Video models: LTX vs MiniMax H3
| | LTX 2.3 | MiniMax H3 / FastH3 Omni |
|---|---|---|
| fps | **25** fixed | **24** |
| audio rate | **48 kHz** | **32 kHz** |
| max frames | 257, `(n−1) % 8 == 0` | **362** (73 ≈ 3 s) |
| steps | 8 (distilled preset) | 50 official / **8 for FastH3 Omni** |
| audio samples/ch | `1920×frames − 1440` | ~1337×frames at 32 kHz |

- Both accept **reference images** (`reference_images`, VAE-encoded like edit refs) in addition to a first-frame `input_image`.
- **Tiled decoding in the phone-exported H3 preset is a local-hardware artifact** — keep it off server-side (`tiled_decoding=False`, no tile dims). We keep the fields in the app but the presets no longer set them.
- Muxing must NOT use `-shortest` when the audio was already trimmed to the video duration: with `-c:v copy` it truncates the video tail (measured 73 → 70 frames). `mux_audio_into_video()` omits it when `video_duration > 0`.

**Memory envelope (measured on .150, RTX 3060 12 GB + 24 GB RAM):** H3 at 124 frames fills VRAM to 11,925/12,288 MB (6 MB free) and then **spills into system RAM**, with the text encoder pushing total RAM toward ~18 GB plus swap growth. Idle VRAM is ~190 MB, SD1.5 1024² peaks ~5.3 GB. 124 frames is fine on an otherwise idle machine, but the failure is binary under any co-tenant (one Ollama model at 1.9 GB is enough) — and H3's bad failure mode is *not* a crash: it returns frames that look plausible but are broken. Verify output, don't trust exit status.

**H3 and LTX are exclusive workloads.** They need the entire GPU: a co-tenant — another generation, an agent's second job, or even a 0.8B LLM on the GPU — is enough to kill the run. There is no useful multitasking on .150 while one is rendering.
- `_single_flight` decorator serializes `generate_image`, `edit_image` and `generate_video` inside the app, so concurrent UI events cannot stack jobs.
- `_ensure_grpc_alive()` pings before each run (waiting up to 10 s for a restarting server) and fails fast with instructions; previously a dead server burned the whole request.
- `_release_llm_vram()` frees Ollama first; everything else must be stopped by hand, which the Video tab says in the UI.

**Who actually kills the server:** `systemd-oomd` on .150. After a heavy H3 run the box shows `gRPCServerCLI` alive but spinning at ~94% CPU with 730 MB RSS and **no VRAM**, then the process disappears — i.e. the OOM killer reaps it at the decode stage, which is why "it crashes the server at the end and you get nothing". With mcpee: check `vram_free_mb`, `compute_apps` and `systemd-oomd`/`systemd` in `top_cpu`; an empty process list means the server is dead and must be restarted before any run. Diagnose on the host with `journalctl -k | grep -i oom` and `journalctl -u systemd-oomd`.

**A blank preview means the run is already dead.** A memory-starved H3 render still advances the progress bar while emitting empty latents: the low-res preview is flat gray, often with a single colored dot (measured std 0.0–0.5, versus ~49 for real content). `generate_video` watches previews and flips the progress text to "blank previews (likely memory pressure)" after three consecutive flat ones, ~3 minutes before the crash.

### Model picker (`model_catalog.py`)
Model lists are enriched at gRPC connect: `{file, name, family, supports(image|video|upscaler), preset, steps, cfg, sampler, notes, label}`.
- **Picking a model auto-applies its matching preset** (steps/cfg/sampler/shift/hires) — prevents the "wrong preset for this checkpoint" class of bug (Klein base vs distilled, Qwen 2.1 vs turbo).
- Matching order: explicit `preset["model"]` filename (quant-tolerant canonical compare) → exact/containment canonical names → token overlap (≥2 tokens, same video-ness) → version→family label.
- Canonicalisation **keeps bracket contents** (`FLUX.3 [klein] 5.6B`) and strips only quant noise (`q4p`/`i4x`/`4-bit`/`official`/`ckpt`). Deleting bracket contents breaks Klein/LTX matching.
- Tabs filter by capability: Generate/Edit hide video+upscalers (checkbox), Video tab shows video only (checkbox to override).
- Verified on .150: 33/77 models bind a preset; `FastH3 Omni` (version `minimax_h3`) binds to the MiniMax H3 preset via the family rule.

### Model-Specific Settings
- SD 1.5: clip_skip=1, base_res=512
- Pony/SDXL: clip_skip=2, base_res=1024
- FLUX: clip_skip=1, shift=1.0, use `guidance_embed` not cfg_scale
- Chroma: clip_skip=2 (per official raw preset)
- LTX 2.3 distilled: steps=8, cfg=1.0, shift=5, TCD Trailing (sampler id 19)
- **Qwen Image 2.1 (+Turbo)**: steps 40 (base) / 8 (turbo `qwen_image_2.1_turbo_i4x`), cfg=1.0, DDIM Trailing, 1536×1536. Base uses shift=1 + resolutionDependentShift; turbo uses shift=6.5 — **but not from the same preset file** (`qwen2_official` vs `qwen21_turbo`). The Edit-tab strength=1.0 force covers it via server metadata `modifier='kontext'` even though the name has no edit keyword.
- **FLUX.3 Klein 5.6B**: 16 steps, CFG 1.0, shift 16 (distilled); `*_base` variant = 50 steps, CFG 4.0, shift 16. **Export quirk**: the non-Base app export references the *base* checkpoint with distilled settings — check the filename before trusting step counts. Both are kontext edit models.
- **MOSS Image SFT** (`moss_image_sft_i4x`): 30 steps, **CFG 3.0** (guidance-distilled — unlike the usual CFG 1.0 turbo models), DDIM Trailing, shift 5.0.

## Testing

No test framework — standalone scripts requiring a live Draw Things server (192.168.2.150:7859):

```bash
# Video + speech audio test (speech exposes wrong pitch/sync; engine noise doesn't)
./venv/bin/python test_video_speech.py

# Audio layout diagnostics (tensor header, shape, rate hypotheses)
./venv/bin/python test_audio_diag.py

# Connector examples
cd dev/DTgRPCconnector
python examples/list_models.py --server 192.168.2.150:7859
```

Pre-commit check:
```bash
python3 -m py_compile app.py settings_manager.py
find . -name "*.py" -exec python3 -m py_compile {} \;
```

## Conventions

- **Versioning**: calendar-based `APP_VERSION = "YYYYMMDD.N"` in app.py (like dtline).
- **dtline repo** (github.com/Alexthestampede/dtline) is the upstream source for connector updates — port from there, keep our mux fixes.
- Related repos: `dev/ModuLLe` and `dev/DTgRPCconnector` are also Alexthestampede's.

## Resources

- README.md: Full documentation
- CLAUDE.md: Detailed architecture notes
- dev/DTgRPCconnector/AGENTS.md: gRPC/TLS specifics, tensor format, video notes