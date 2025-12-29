# Changelog

## v0.2.0 - Complete UI Refactor (2024-12-28)

### ✅ All Requested Improvements Implemented

#### UI/UX
- ✅ **Calibri Font**: Set as primary font with proper fallbacks (Segoe UI, Tahoma, sans-serif)
- ✅ **PWA Support**: Enabled with manifest.json and mobile web app meta tags
- ✅ **Removed Temperature Controls**: Temperature now uses server defaults
- ✅ **No Hardcoded Servers**: All settings loaded from `settings/config.json`

#### Settings System
- ✅ **Settings Manager**: New `settings_manager.py` module
- ✅ **Auto-Generated Config**: Creates `settings/config.json` on first run (gitignored)
- ✅ **System Prompts from Files**:
  - `settings/extract.txt` - Image prompt extraction
  - `settings/expand.txt` - Prompt expansion
  - `settings/refine.txt` - Prompt refinement
- ✅ **Aspect Ratio Presets**: Loaded from `settings/aspectratio.txt`
- ✅ **Base Resolution Detection**: Automatically scales ratios for 512 vs 1024 base models
- ✅ **Model Presets**: JSON-based preset system in `settings/presets/`

#### Image Generation
- ✅ **Aspect Ratio Dropdown**: Replaced free-form width/height sliders
- ✅ **Steps Range**: Increased from 50 to 150 maximum
- ✅ **Model Metadata Integration**: Fetches latent_size from gRPC server
- ✅ **Automatic Resolution Scaling**: Aspect ratios scale based on model's base resolution
- ✅ **LoRA Support**: Dropdown for LoRA selection with weight control
- ✅ **Model Presets**: Auto-applies recommended settings when model selected

### New Files Created

#### Core Application
- `settings_manager.py` - Settings management system
- `manifest.json` - PWA manifest

#### Settings Directory
- `settings/expand.txt` - Prompt expansion system prompt
- `settings/refine.txt` - Prompt refinement system prompt
- `settings/extract.txt` - Already existed (image prompt extraction)
- `settings/aspectratio.txt` - Already existed (aspect ratio presets)
- `settings/config.example.json` - Example configuration
- `settings/presets/example.json` - Example model preset

#### Documentation
- `CHANGELOG.md` - This file

### Modified Files
- `app.py` - Complete refactor with all improvements
- `.gitignore` - Added settings/config.json exclusion
- `CLAUDE.md` - Updated with settings system documentation

### Key Features

#### Settings Management
```python
# Settings auto-load from files
settings.load_system_prompt('extract')  # Loads settings/extract.txt
settings.load_aspect_ratios(1024)  # Loads and scales from aspectratio.txt
settings.get_model_preset('model.ckpt')  # Loads from presets/*.json
```

#### Aspect Ratio System
- Reads from `settings/aspectratio.txt`
- Automatically scales for 512 or 1024 base resolution
- Format: `ratio_label widthxheight`
- Example: `1:1 1024x1024` becomes `1:1 512x512` for SD 1.5 models

#### Model Presets
- JSON files in `settings/presets/`
- Auto-applied when model selected
- Contains: base_resolution, clip_skip, steps, CFG, sampler, notes
- Presets are gitignored (except example.json)

### Architecture Improvements

1. **Separation of Concerns**
   - Settings logic → `settings_manager.py`
   - UI logic → `app.py`
   - External config → `settings/` folder

2. **User-Friendly Configuration**
   - All settings editable as text/JSON files
   - No code changes needed for customization
   - Config auto-generated on first run

3. **Smart Defaults**
   - Server addresses default to localhost
   - Settings persist across sessions
   - Last used models/LoRAs remembered

### Next Steps

1. **Implement Image Generation**: Complete the `generate_image()` function with actual DrawThingsClient call
2. **Add Icons**: Create icon-192.png and icon-512.png for PWA
3. **Create More Presets**: Add presets for common models
4. **Test PWA**: Verify PWA installation on mobile devices

### Breaking Changes
- None - this is a new implementation

### Migration Guide
- No migration needed - first run auto-creates config
- Copy any custom server addresses to Settings tab and save
