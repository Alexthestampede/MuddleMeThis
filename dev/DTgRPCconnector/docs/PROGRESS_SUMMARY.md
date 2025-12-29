# Draw Things gRPC Client - Progress Summary

**Date:** 2025-12-27
**Status:** Client connects successfully but image generation has performance issues

---

## ✅ Accomplishments

### 1. **TLS Connection Working**
- Server uses self-signed certificate from "Draw Things Root CA"
- Extracted certificate to `server_cert.pem`
- Updated `DrawThingsClient` to accept self-signed certs with `verify_ssl=False`
- Connection established successfully to `192.168.2.150:7859`

### 2. **FlatBuffers Configuration**
- Found official schema at: https://github.com/drawthingsai/draw-things-community/blob/main/Libraries/DataModels/Sources/config.fbs
- Generated `GenerationConfiguration.py` from schema
- Added all required fields including:
  - Basic: id, seed, steps, guidance_scale, strength, model, sampler
  - Dimensions: start_width, start_height, original_image_width/height, target_image_width/height
  - Advanced: seed_mode, clip_skip, shift, mask_blur, tiled_decoding/diffusion, etc.
  - Arrays: controls (empty), loras (empty)

### 3. **Model Discovery**
- Successfully listed 623 model files from server
- Found target model: `realdream_15sd15_q6p_q8p.ckpt` (SD 1.5)
- Found SDXL model: `juggernaut_xl_v9_q6p_q8p.ckpt`

### 4. **Basic Request Structure**
- Protocol: `ImageGenerationRequest` with FlatBuffers configuration
- Fields set: prompt, configuration, scaleFactor, user, device, chunked
- Server responds with streaming progress: TextEncoded → ImageEncoded → Sampling → ImageDecoded

---

## ❌ Current Issues

### **Primary Issue: Incorrect Tensor Dimensions**

The server is using **pixel-space dimensions** instead of **latent-space dimensions** for attention operations, causing:
- Extremely slow generation (should be <10s, taking minutes)
- High VRAM usage
- Server crashes with certain configurations

#### Evidence from Debug Logs:

**Our failing configurations** (`debug/debug5.txt`, `debug/fishyoutput.txt`):
```
SD 1.5:  [2x262144x320]  where 262144 = 512×512 (WRONG - should be 64×64 = 4096)
SDXL:    [2x1048576x1280] where 1048576 = 1024×1024 (WRONG - should be 128×128 = 16384)
```

**Successful generation** (`debug/successdebug.txt`):
```
Latent output: [1x4x512x512]  (Expected: [1x4x64x64] for 512×512 output)
Final output:  [1x512x512x3] (Correct)
Generation time: 10 seconds (Correct)
```

**The mystery:** The successful log also shows wrong latent dimensions but completes fast!

---

## 🧪 Experiments Tried

### Configuration Attempts:

| Config | start_w/h | original_w/h | target_w/h | scaleFactor | Result |
|--------|-----------|--------------|------------|-------------|---------|
| 1 | 512 | - | - | 1 | ❌ Crashes (322GB buffer allocation) |
| 2 | 64 | 512 | 512 | 1 | ⚠️ Works but VERY slow (uses 512×512 tensors) |
| 3 | 64 | - | 512 | 1 | ⚠️ Works but VERY slow |
| 4 | 64 | - | - | 1 | ⚠️ Works but VERY slow |
| 5 | 512 | 512 | 512 | 1 | ❌ Crashes |
| 6 | - | - | - | 1 | ❌ Crashes |
| 7 | 512 | - | - | 8 | ❌ Hangs (no output) |
| 8 | Minimal fields only | 1 | ❌ Crashes |

### Key Findings:
- Setting `start_width/height = 512` always causes crashes
- Setting `start_width/height = 64` (latent dims) doesn't crash but is extremely slow
- Server ignores latent dimensions and processes in pixel space regardless
- Both SD 1.5 and SDXL show same issue

---

## 📊 Debug Output Analysis

### Tensor Dimension Patterns:

**SD 1.5 (should use 64×64 = 4,096 tokens):**
```
[cnnp_reshape_build] dim: (2, 262144, 8, 160)
[cnnp_reshape_build] input: (2, 262144, 1280)
```
→ Using 512×512 = 262,144 tokens ❌

**SDXL (should use 128×128 = 16,384 tokens):**
```
[cnnp_reshape_build] dim: (2, 1024, 1024, 1280)
[cnnp_reshape_build] input: (2, 1048576, 1280)
```
→ Using 1024×1024 = 1,048,576 tokens ❌

**Successful generation:**
```
CCV_NNC_CONVOLUTION_FORWARD [121]: [3] -> [1]
|-> 2. [4x128x3x3]  (UNet final layer)
|<- 1. [1x4x512x512]  (Latent output - should be [1x4x64x64])
```

---

## 🔍 Hypotheses

### 1. **Missing Field/Flag**
- Maybe there's a field in the schema that tells the server "use latent space" vs "use pixel space"
- We haven't set all 89 fields in the schema - some might be critical

### 2. **Dimension Field Semantics**
- `start_width/height` might mean something different than we think
- Maybe they're output dimensions and server computes latent automatically?
- Maybe scaleFactor has a different purpose?

### 3. **Model Detection Issue**
- Server might be detecting model type incorrectly
- Could be treating SD 1.5 as a different architecture

### 4. **Request Field Interaction**
- Maybe `scaleFactor` in `ImageGenerationRequest` affects dimension calculation
- Maybe the `image` field (optional bytes) being empty triggers wrong behavior

---

## 📁 Files Created

### Working Code:
- `drawthings_client.py` - Main client library with TLS support
- `GenerationConfiguration.py` - FlatBuffers schema (89 fields)
- `imageService_pb2.py` - gRPC protobuf definitions
- `server_cert.pem` - Extracted server certificate

### Test Scripts:
- `test_final.py` - Complete configuration test
- `test_simple.py` - Start dims only (crashes)
- `test_all_512.py` - All dims = 512 (crashes)
- `test_no_original.py` - No original dims (slow)
- `test_latent_only.py` - Latent dims only (slow)
- `test_sdxl.py` - SDXL model test (slow)
- `test_scalefactor.py` - scaleFactor=8 test (hangs)
- `test_minimal.py` - Minimal fields (crashes)
- `test_no_dimensions.py` - No dimension fields (crashes)

### Debug Logs:
- `debug/crashdebug.txt` - Server crash with 322GB buffer allocation
- `debug/debug2.txt` - Signal 4 (illegal instruction)
- `debug/debug3.txt` - 4096×4096 tensor crash
- `debug/debug5.txt` - Shows 262,144 and 1,048,576 token tensors
- `debug/fishyoutput.txt` - Shows oversized attention tensors
- `debug/successdebug.txt` - Successful generation (10 seconds, but shows wrong dims?)
- `debug/conf.txt` - Server command-line options

---

## 🎯 Next Steps

### Immediate:
1. **Capture full debug log from official client**
   - Run Docker with `docker logs -f` redirected to file
   - Generate 512×512 image with official Draw Things client
   - Compare complete debug output with our attempts
   - Look for attention operations in early UNet phase

2. **Find the dimension calculation**
   - Search for where server computes attention tensor dimensions
   - Identify what configuration fields actually control this
   - Check if official client sets fields we're missing

### Investigation:
3. **Review complete schema**
   - Go through all 89 fields in config.fbs
   - Identify fields we haven't set
   - Check for boolean flags related to image encoding/latent space

4. **Test minimal successful config**
   - Once we see what official client sends, replicate exactly
   - Start with bare minimum fields that work
   - Add fields incrementally to understand each one's effect

5. **Source code review**
   - Look at Draw Things server source code (if available)
   - Find how it reads GenerationConfiguration
   - Understand dimension calculation logic

---

## 💡 User Ideas
User mentioned having ideas for next steps (not yet detailed).

---

## 📚 Resources

- **Schema:** https://github.com/drawthingsai/draw-things-community/blob/main/Libraries/DataModels/Sources/config.fbs
- **Protocol:** https://github.com/drawthingsai/draw-things-community/blob/main/Libraries/GRPC/Models/Sources/imageService/imageService.proto
- **Server:** Docker image `drawthingsai/draw-things-grpc-server-cli:latest`
- **Model:** `realdream_15sd15_q6p_q8p.ckpt` on server at `192.168.2.150:7859`

---

## 🔧 Server Configuration

```bash
docker run -v /dt/models:/grpc-models -p 7859:7859 --gpus all \
  drawthingsai/draw-things-grpc-server-cli:latest \
  gRPCServerCLI /grpc-models --model-browser --cpu-offload --supervised --debug
```

**Flags:**
- `--model-browser` - Enable model browsing
- `--cpu-offload` - Offload weights to CPU during inference
- `--supervised` - Restart on crash
- `--debug` - Verbose model inference logging

---

## ✨ Success Criteria

When this is working correctly:
- ✅ 512×512 generation completes in <15 seconds
- ✅ Attention tensors use 4,096 tokens (64×64) for SD 1.5
- ✅ Attention tensors use 16,384 tokens (128×128) for SDXL
- ✅ No server crashes or hangs
- ✅ Reasonable VRAM usage

---

*Last updated: 2025-12-27*
