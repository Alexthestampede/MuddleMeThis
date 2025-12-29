"""
Settings Manager for MuddleMeThis

Handles loading/saving user configuration and system prompts from the settings/ folder.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SettingsManager:
    """Manages application settings and system prompts"""

    def __init__(self):
        self.settings_dir = Path(__file__).parent / "settings"
        self.config_file = self.settings_dir / "config.json"
        self.config_example = self.settings_dir / "config.example.json"
        self.presets_dir = self.settings_dir / "presets"

        # Ensure directories exist
        self.settings_dir.mkdir(exist_ok=True)
        self.presets_dir.mkdir(exist_ok=True)

        # Load or create config
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Load config from file, or create from example if doesn't exist"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Create from example
            if self.config_example.exists():
                with open(self.config_example, 'r') as f:
                    default_config = json.load(f)
            else:
                # Fallback defaults
                default_config = {
                    "llm_server": "http://localhost:1234",
                    "llm_model": "",
                    "grpc_server": "localhost:7859",
                    "last_used_model": "",
                    "last_used_lora": "",
                    "default_steps": 16,
                    "default_cfg": 7.0,
                    "default_aspect_ratio": "1:1 1024x1024"
                }

            # Save the initial config
            self.save_config(default_config)
            return default_config

    def save_config(self, config: Optional[Dict] = None):
        """Save config to file"""
        if config is None:
            config = self.config

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def update_config(self, **kwargs):
        """Update config values and save"""
        self.config.update(kwargs)
        self.save_config()

    def get(self, key: str, default=None):
        """Get config value"""
        return self.config.get(key, default)

    def load_system_prompt(self, prompt_type: str) -> str:
        """Load system prompt from settings folder

        Args:
            prompt_type: One of 'extract', 'expand', 'refine'

        Returns:
            System prompt text, or empty string if file doesn't exist
        """
        prompt_file = self.settings_dir / f"{prompt_type}.txt"

        if prompt_file.exists():
            return prompt_file.read_text().strip()
        else:
            print(f"Warning: System prompt file not found: {prompt_file}")
            return ""

    def load_aspect_ratios(self, base_resolution: int = 1024) -> List[Tuple[str, int, int]]:
        """Load aspect ratio presets from aspectratio.txt

        Args:
            base_resolution: Base resolution (1024 or 512) to scale ratios

        Returns:
            List of (label, width, height) tuples
        """
        aspect_file = self.settings_dir / "aspectratio.txt"

        if not aspect_file.exists():
            # Fallback defaults for 1024
            return [
                ("1:1 1024x1024", 1024, 1024),
                ("16:9 1024x576", 1024, 576),
                ("4:3 1024x768", 1024, 768),
            ]

        ratios = []
        scale_factor = base_resolution / 1024.0  # Scale from 1024 base

        for line in aspect_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue

            # Parse format: "ratio_name widthxheight"
            parts = line.split()
            if len(parts) >= 2:
                label = parts[0]
                dimensions = parts[1]

                if 'x' in dimensions:
                    try:
                        width, height = dimensions.split('x')
                        width = int(int(width) * scale_factor)
                        height = int(int(height) * scale_factor)

                        # Create display label
                        display_label = f"{label} {width}x{height}"
                        ratios.append((display_label, width, height))
                    except ValueError:
                        continue

        return ratios

    def load_model_presets(self) -> Dict[str, Dict]:
        """Load all model presets from presets/ folder

        Returns:
            Dict mapping model names to their preset configurations
        """
        presets = {}

        if not self.presets_dir.exists():
            return presets

        for preset_file in self.presets_dir.glob("*.json"):
            try:
                with open(preset_file, 'r') as f:
                    preset_data = json.load(f)

                    # Use filename (without .json) as key if no name in file
                    preset_name = preset_data.get('name', preset_file.stem)
                    presets[preset_name] = preset_data
            except Exception as e:
                print(f"Warning: Failed to load preset {preset_file}: {e}")

        return presets

    def get_model_preset(self, model_name: str) -> Optional[Dict]:
        """Get preset for a specific model

        Args:
            model_name: Name of the model (with or without .ckpt extension)

        Returns:
            Preset dict or None if not found
        """
        presets = self.load_model_presets()

        # Try exact match first
        if model_name in presets:
            return presets[model_name]

        # Try without extension
        base_name = model_name.replace('.ckpt', '').replace('.safetensors', '')
        if base_name in presets:
            return presets[base_name]

        # Try partial match
        for preset_name, preset_data in presets.items():
            if base_name in preset_name or preset_name in base_name:
                return preset_data

        return None


# Global instance
settings = SettingsManager()
