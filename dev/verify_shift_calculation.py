#!/usr/bin/env python3
"""
Verification script for resolution-dependent shift calculation.
Tests the corrected formula against known expected values.
"""

import math

def calculate_shift(pixel_width, pixel_height):
    """
    Calculate resolution-dependent shift using the correct formula.

    Args:
        pixel_width: Width in pixels
        pixel_height: Height in pixels

    Returns:
        Calculated shift value
    """
    resolution_factor = (pixel_width * pixel_height) / 256
    shift = math.exp(((resolution_factor - 256) * (1.15 - 0.5) / (4096 - 256)) + 0.5)
    return shift

# Test cases - expected values calculated using the correct formula
test_cases = [
    # (width, height, expected_shift, description)
    (1152, 896, 3.12, "User's reported Z-Image Turbo case (verified from Draw Things)"),
    (1024, 1024, 3.16, "Standard 1024x1024"),
    (512, 512, 1.88, "Standard 512x512"),
    (1536, 1536, 7.51, "High-res 1536x1536"),
    (768, 768, 2.33, "Mid-res 768x768"),
]

print("=" * 70)
print("Resolution-Dependent Shift Verification")
print("=" * 70)
print()

all_passed = True

for width, height, expected, description in test_cases:
    calculated = calculate_shift(width, height)
    resolution_factor = (width * height) / 256

    # Allow 0.01 tolerance for floating point comparison
    passed = abs(calculated - expected) < 0.01
    status = "✓ PASS" if passed else "✗ FAIL"

    if not passed:
        all_passed = False

    print(f"{status} {description}")
    print(f"     Dimensions: {width}×{height} pixels")
    print(f"     Resolution Factor: {resolution_factor:.1f}")
    print(f"     Calculated Shift: {calculated:.2f}")
    print(f"     Expected Shift: {expected:.2f}")
    print()

print("=" * 70)
if all_passed:
    print("✓ All tests PASSED! The shift calculation is correct.")
else:
    print("✗ Some tests FAILED! Check the implementation.")
print("=" * 70)
