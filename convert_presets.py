#!/usr/bin/env python3
"""
Convert official Draw Things presets from dev/*.txt to settings/presets/*.json format
"""

import json
from pathlib import Path

# Preset metadata
preset_info = {
    'flux.txt': {
        'name': 'FLUX Dev (Official)',
        'description': 'Official FLUX Dev settings from Draw Things',
        'notes': 'Powerful all-rounder for FLUX models'
    },
    'schnell.txt': {
        'name': 'FLUX Schnell (Official)',
        'description': 'Official FLUX Schnell settings from Draw Things',
        'notes': 'Fast generation with FLUX Schnell'
    },
    'ponysdxl.txt': {
        'name': 'Pony/SDXL (Official)',
        'description': 'Official Pony Diffusion / SDXL settings from Draw Things',
        'notes': 'For Pony, Illustrious, NoobXL, and other SDXL-based models'
    },
    'sd15.txt': {
        'name': 'SD 1.5 (Official)',
        'description': 'Official SD 1.5 settings from Draw Things',
        'notes': 'For Stable Diffusion 1.5 and 2.x models'
    },
    'qwenimage.txt': {
        'name': 'Qwen Image (Official)',
        'description': 'Official Qwen Image settings from Draw Things',
        'notes': 'For Qwen visual models'
    },
    'chroma.txt': {
        'name': 'Chroma (Official)',
        'description': 'Official Chroma settings from Draw Things',
        'notes': 'For Chroma models'
    },
}

dev_folder = Path('dev')
presets_folder = Path('settings/presets')

for txt_file in dev_folder.glob('*.txt'):
    # Skip non-preset files
    if txt_file.name not in preset_info and txt_file.stem not in ['plans', 'z-image-turbo']:
        continue

    if txt_file.name in ['plans.txt', 'z-image-turbo.txt']:
        continue

    try:
        # Read Draw Things format
        with open(txt_file, 'r') as f:
            dt_settings = json.load(f)

        # Get preset metadata
        info = preset_info.get(txt_file.name, {
            'name': txt_file.stem.title(),
            'description': f'Official {txt_file.stem} settings',
            'notes': 'Converted from Draw Things'
        })

        # Convert to MuddleMeThis preset format
        preset = {
            'name': info['name'],
            'description': info['description'],
            'base_resolution': dt_settings.get('width', 1024),  # Use width as base

            # Core generation settings
            'steps': dt_settings.get('steps', 20),
            'guidanceScale': dt_settings.get('guidanceScale', 7.0),
            'sampler': dt_settings.get('sampler', 0),  # Keep as ID
            'strength': dt_settings.get('strength', 1.0),

            # Advanced settings
            'shift': dt_settings.get('shift', 1.0),
            'sharpness': dt_settings.get('sharpness', 0.0),
            'seedMode': dt_settings.get('seedMode', 2),
            'resolutionDependentShift': dt_settings.get('resolutionDependentShift', False),

            # Quality enhancement
            'hiresFix': dt_settings.get('hiresFix', False),
            'tiledDiffusion': dt_settings.get('tiledDiffusion', False),
            'tiledDecoding': dt_settings.get('tiledDecoding', False),
            'cfgZeroStar': dt_settings.get('cfgZeroStar', False),
            'cfgZeroInitSteps': dt_settings.get('cfgZeroInitSteps', 0),

            # Inpainting settings
            'maskBlur': dt_settings.get('maskBlur', 1.5),
            'maskBlurOutset': dt_settings.get('maskBlurOutset', 0),
            'preserveOriginalAfterInpaint': dt_settings.get('preserveOriginalAfterInpaint', True),

            # Batch settings
            'batchCount': dt_settings.get('batchCount', 1),
            'batchSize': dt_settings.get('batchSize', 1),

            # Optional models
            'upscaler': dt_settings.get('upscaler', ''),
            'refinerModel': dt_settings.get('refinerModel', ''),
            'faceRestoration': dt_settings.get('faceRestoration', ''),

            # Other settings
            'causalInferencePad': dt_settings.get('causalInferencePad', 0),
            'clip_skip': dt_settings.get('clipSkip', 1),
            'loras': [],
            'controls': [],

            'notes': info['notes']
        }

        # Write to presets folder
        output_file = presets_folder / f"{txt_file.stem}_official.json"
        with open(output_file, 'w') as f:
            json.dump(preset, f, indent=2)

        print(f"✅ Converted {txt_file.name} → {output_file.name}")

    except Exception as e:
        print(f"❌ Failed to convert {txt_file.name}: {e}")

print(f"\n✅ Preset conversion complete!")
