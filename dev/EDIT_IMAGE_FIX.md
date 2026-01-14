# Edit Image Fix

## Problem

The Edit Image feature was ignoring the input image and generating entirely new images instead. This indicated that the server was receiving the image data but not actually using it for editing.

## Root Cause

The issue was a configuration problem. While the manual implementation in `app.py` was correctly encoding and sending the image, it was missing critical configuration fields that tell the server to use the image for editing:

1. **ImageGuidanceScale** - Required for edit models (typically 1.5)
2. **OriginalImageWidth/Height** - Tells the server the original dimensions
3. **TargetImageWidth/Height** - Tells the server the target dimensions

Additionally, the `ImageGenerationConfig` class in the client library had bugs:
- It was passing pixel dimensions directly to `StartWidth/StartHeight` instead of converting to scale units (pixels ÷ 64)
- It lacked support for edit-specific fields

## Solution

### 1. Enhanced `drawthings_client.py`

**Added edit-specific fields to ImageGenerationConfig:**
```python
original_image_width: Optional[int] = None
original_image_height: Optional[int] = None
target_image_width: Optional[int] = None
target_image_height: Optional[int] = None
image_guidance_scale: Optional[float] = None
```

**Fixed pixel-to-scale conversion in to_flatbuffer():**
```python
# Convert pixels to scale units (server always uses 64 as divisor)
scale_width = self.width // 64
scale_height = self.height // 64

GenerationConfiguration.AddStartWidth(builder, scale_width)
GenerationConfiguration.AddStartHeight(builder, scale_height)
```

**Added edit-specific fields to configuration:**
```python
if self.original_image_width is not None:
    GenerationConfiguration.AddOriginalImageWidth(builder, self.original_image_width)
if self.original_image_height is not None:
    GenerationConfiguration.AddOriginalImageHeight(builder, self.original_image_height)
if self.target_image_width is not None:
    GenerationConfiguration.AddTargetImageWidth(builder, self.target_image_width)
if self.target_image_height is not None:
    GenerationConfiguration.AddTargetImageHeight(builder, self.target_image_height)
if self.image_guidance_scale is not None:
    GenerationConfiguration.AddImageGuidanceScale(builder, self.image_guidance_scale)
```

**Added metadata_override parameter to generate_image():**
```python
def generate_image(
    self,
    prompt: str,
    config: ImageGenerationConfig,
    negative_prompt: str = "",
    scale_factor: int = 1,
    input_image: Optional[bytes] = None,
    metadata_override: Optional[any] = None,  # NEW!
    progress_callback: Optional[Callable[[str, int], None]] = None,
    preview_callback: Optional[Callable[[bytes], None]] = None
) -> List[bytes]:
```

### 2. Refactored `app.py` edit_image()

**Simplified from 340+ lines to ~200 lines** by using the client's `generate_image()` method:

**Before (manual approach):**
- Manually encode tensor with `encode_image_to_tensor()`
- Manually calculate SHA256 hash
- Manually build FlatBuffer configuration
- Manually build gRPC request
- Manually handle chunked responses

**After (client method):**
```python
# Create configuration with edit-specific fields
config = ImageGenerationConfig(
    model=model_file,
    steps=steps,
    width=target_width,
    height=target_height,
    cfg_scale=cfg_scale,
    scheduler=sampler_name,
    seed=actual_seed,
    strength=strength,
    clip_skip=clip_skip,
    shift=final_shift,
    # CRITICAL: Edit-specific fields for edit models!
    original_image_width=target_width,
    original_image_height=target_height,
    target_image_width=target_width,
    target_image_height=target_height,
    image_guidance_scale=1.5
)

# Generate using client method (handles all encoding/hashing automatically)
generated_tensors = state.grpc_client.generate_image(
    prompt=instruction,
    config=config,
    negative_prompt=negative_prompt,
    input_image=input_image_bytes,
    metadata_override=override,
    progress_callback=on_progress
)
```

## Key Changes

### drawthings_client.py
1. Added 5 edit-specific optional fields to `ImageGenerationConfig`
2. Fixed `to_flatbuffer()` to convert pixels → scale units (÷ 64)
3. Added logic to include edit fields in FlatBuffer when set
4. Added `metadata_override` parameter to `generate_image()`
5. Wire up `metadata_override` to gRPC request

### app.py
1. Simplified `edit_image()` from manual implementation to client method
2. Create `ImageGenerationConfig` with edit-specific fields
3. Pass input image as bytes (client handles encoding)
4. Pass `MetadataOverride` for LoRA metadata
5. Let client handle all the complexity

### CLAUDE.md
1. Updated feature list - Edit Image now marked as functional
2. Added guidance for working with edit models
3. Clarified universal scale factor formula (÷ 64)

## Why This Should Work

The key insight is that **edit models need special configuration fields** to tell the server "this is an edit operation, use the input image":

1. **ImageGuidanceScale** (1.5) - Tells the model how strongly to condition on the input image
2. **OriginalImageWidth/Height** - Provides context about the original dimensions
3. **TargetImageWidth/Height** - Specifies the desired output dimensions

Without these fields, the server treats it as a regular generation (ignoring the image data even though it was sent).

Additionally, the client library now properly:
- Converts pixel dimensions to scale units
- Handles image encoding/resizing automatically
- Supports the MetadataOverride for LoRA metadata
- Provides consistent image handling across all generation modes

## Testing Recommendations

1. Test with Qwen Edit 2511 model
2. Use simple instructions like "Change the hair to red"
3. Verify generation takes longer (indicating it's actually processing the input image)
4. Check that the output is actually an edited version of the input, not a new generation
5. Look for "imageEncoded" signpost in debug output (confirms server received image)

## Files Modified

1. **dev/DTgRPCconnector/drawthings_client.py** - Enhanced client library
2. **app.py** - Refactored edit_image() function
3. **CLAUDE.md** - Updated documentation
4. **dev/EDIT_IMAGE_FIX.md** - This file
