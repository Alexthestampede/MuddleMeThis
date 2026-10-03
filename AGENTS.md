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

- `app.py` — Main Gradio app (~3500 lines). Video tab wired near the end (`create_ui`).
- `settings_manager.py` — Config persistence (JSON).
- `dev/DTgRPCconnector/` — gRPC client. **Canonical copy synced from dtline repo (calendar-versioned, v26.0910.1+)**. `drawthings_client.py` (client + audio muxing), `tensor_decoder.py` (`tensor_to_pil`, `decode_audio_tensor`), `GenerationConfiguration.py` (FlatBuffer schema; slots 86+ = newer server fields).
- `dev/ModuLLe/` — LLM provider abstraction (Ollama/LM Studio).
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

### TLS / gRPC connection
TLS is the default. Self-signed certs need the localhost name override — already in our connector: `grpc.ssl_target_name_override = "localhost"` + `ssl_cert_path` to `dev/DTgRPCconnector/root_ca.crt` (app.py lines ~743).

### Model-Specific Settings
- SD 1.5: clip_skip=1, base_res=512
- Pony/SDXL: clip_skip=2, base_res=1024
- FLUX: clip_skip=1, shift=1.0, use `guidance_embed` not cfg_scale
- Chroma: clip_skip=2 (per official raw preset)
- LTX 2.3 distilled: steps=8, cfg=1.0, shift=5, TCD Trailing (sampler id 19)

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