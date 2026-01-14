# Resolution-Dependent Shift Bug Fix

## Summary

Fixed critical bugs in the resolution-dependent shift calculation that caused:
1. Absurdly large shift values (1.47e+19 instead of ~3.12)
2. Different results based on the manual shift value when resolution-dependent shift was enabled
3. Incorrect behavior compared to the official Draw Things app

## The Bugs

### Bug #1: Multiplying by Manual Shift Value
**Location:** `app.py:605` (generate image function)

**Problem:**
```python
if shift != 1.0:
    final_shift = calculated_shift * shift  # ❌ WRONG!
else:
    final_shift = calculated_shift
```

When the user had shift=1 and enabled resolution-dependent shift, it calculated correctly as 3.12.
But when shift=3, it calculated 3.12 × 3 = 9.36, which is completely wrong!

**Fix:**
```python
# When resolution-dependent shift is enabled, the calculated value replaces the manual shift
final_shift = calculated_shift
```

The resolution-dependent shift should **replace** the manual shift, not multiply it.

### Bug #2: Wrong Resolution Factor Calculation (Generate)
**Location:** `app.py:591-595`

**Problem:**
```python
latent_width = width // 8   # ❌ Assumes all models use divisor of 8
latent_height = height // 8
resolution_factor = (latent_height * latent_width) * 16
```

For 1152×896:
- Buggy calculation: `(1152//8) × (896//8) × 16 = 144 × 112 × 16 = 258,048`
- With this huge number: `exp(44.14) = 1.47e+19` ❌

**Fix:**
```python
# Resolution factor: pixel area divided by 256
resolution_factor = (width * height) / 256
```

For 1152×896:
- Correct calculation: `(1152 × 896) / 256 = 4032`
- With correct number: `exp(1.139) = 3.12` ✅

### Bug #3: Using Scale Units Instead of Pixels (Edit Image)
**Location:** `app.py:1018-1020`

**Problem:**
```python
latent_h = scale_height  # ❌ These are scale units (already divided by 64!)
latent_w = scale_width
resolution_factor = (latent_h * latent_w) * 16
```

Scale units are already divided by 64, so this produced completely wrong results.

**Fix:**
```python
# Use target dimensions (in pixels), not scale units!
resolution_factor = (target_width * target_height) / 256
```

## The Correct Formula

The resolution-dependent shift calculation is **universal and model-independent**:

```python
resolution_factor = (pixel_width * pixel_height) / 256
shift = exp(((resolution_factor - 256) * (1.15 - 0.5) / (4096 - 256)) + 0.5)
```

### Key Points:
1. **Always use pixel dimensions** - never use latent dimensions or scale units
2. **Divide by 256** - this is a constant, regardless of model latent size
3. **The calculated shift replaces the manual shift** - don't multiply them!
4. **Works for all models** - FLUX, SDXL, SD1.5, Qwen, etc.

## Test Results

Using the verification script (`dev/verify_shift_calculation.py`):

```
✓ PASS User's reported Z-Image Turbo case (1152×896 → 3.12)
✓ PASS Standard 1024×1024 → 3.16
✓ PASS Standard 512×512 → 1.88
✓ PASS High-res 1536×1536 → 7.51
✓ PASS Mid-res 768×768 → 2.33
```

All tests pass! The calculation now matches Draw Things exactly.

## Files Modified

1. **app.py** (lines 586-608): Fixed generate image shift calculation
2. **app.py** (lines 1015-1024): Fixed edit image shift calculation
3. **CLAUDE.md** (lines 95-101): Updated documentation with correct formula
4. **dev/verify_shift_calculation.py**: Created verification script
5. **dev/SHIFT_BUG_FIX.md**: This file

## How to Verify

Run the verification script:
```bash
python dev/verify_shift_calculation.py
```

Or test in the app:
1. Set resolution to 1152×896
2. Enable "Resolution Dependent Shift"
3. Check the console output - should show "Calculated Shift: 3.12"
4. The value should be **identical** regardless of what the manual Shift slider is set to

## Technical Details

The formula comes from Draw Things `ModelZoo.swift:2358-2360` and maps resolution to shift range 0.5-1.15:

- **Low res (256×256)**: resolution_factor = 256, shift = 0.5
- **Medium res (1024×1024)**: resolution_factor = 4096, shift = 3.16
- **High res (4096×4096)**: resolution_factor = 65536, shift = 1.15

The exponential curve ensures smooth transition across different resolutions, matching the official Draw Things behavior.
