#!/usr/bin/env python3
"""
Query and decode model metadata from the Draw Things gRPC server.
"""

import imageService_pb2
from drawthings_client import DrawThingsClient
import json
from typing import Dict, Optional

class ModelMetadata:
    """Handles model metadata from Draw Things server."""

    def __init__(self, server='192.168.2.150:7859'):
        self.server = server
        self._models_cache = None
        self._loras_cache = None

    def _fetch_metadata(self):
        """Fetch metadata from server."""
        if self._models_cache is not None:
            return

        client = DrawThingsClient(self.server, insecure=False, verify_ssl=False)
        try:
            echo_request = imageService_pb2.EchoRequest(name='list_files')
            response = client.stub.Echo(echo_request)

            if response.HasField('override'):
                self._models_cache = json.loads(response.override.models)
                self._loras_cache = json.loads(response.override.loras) if response.override.loras else []
            else:
                self._models_cache = []
                self._loras_cache = []
        finally:
            client.close()

    def get_model_info(self, model_filename: str) -> Optional[Dict]:
        """
        Get metadata for a specific model.

        Args:
            model_filename: The model filename (e.g., 'realdream_15sd15_q6p_q8p.ckpt')

        Returns:
            Dictionary with model metadata or None if not found
        """
        self._fetch_metadata()

        for model in self._models_cache:
            if model.get('file') == model_filename:
                return model
        return None

    def get_latent_info(self, model_filename: str) -> Dict:
        """
        Get latent space information for a model.

        The key formula is: scale = desired_pixels ÷ latent_size
        Where latent_size depends on the VAE compression ratio.

        Most VAEs use 8x compression:
        - SD 1.5: latent_size = 64 (for 512px output)
        - SDXL: latent_size = 128 (for 1024px output)
        - FLUX/Qwen/Z-Image: latent_size = 64 (for 1024px output with scale=16)

        Returns:
            Dict with model metadata and calculated latent parameters
        """
        info = self.get_model_info(model_filename)
        if not info:
            return {
                'default_scale': None,
                'version': None,
                'vae_compression': None,
                'latent_size': None,
                'error': 'Model not found in metadata'
            }

        version = info.get('version', 'unknown')
        default_scale = info.get('default_scale', 8)
        autoencoder = info.get('autoencoder', '')

        # All standard VAEs use 8x compression (8 pixels → 1 latent pixel)
        vae_compression = 8

        # Determine latent_size based on autoencoder and version
        # latent_size is what to use for the --latent-size parameter
        latent_size_map = {
            # SD 1.5/2.x use 64-pixel latent space
            'v1': 64,
            'v2': 64,
            # SDXL uses 128-pixel latent space
            'sdxl_base_v0.9': 128,
            'sdxl': 128,
            # FLUX, Z-Image, and Qwen use FLUX-style VAE with 64-pixel latent
            'flux1': 64,
            'z_image': 64,
            'qwen_image': 64,
        }

        latent_size = latent_size_map.get(version, 64)

        # Calculate the default output size from default_scale
        # Formula: pixel_size = scale × latent_size
        default_output_size = default_scale * latent_size

        return {
            'default_scale': default_scale,
            'version': version,
            'vae_compression': vae_compression,
            'latent_size': latent_size,
            'default_output_size': default_output_size,
            'name': info.get('name', model_filename),
            'autoencoder': autoencoder,
            'formula': f'{default_output_size}px = scale({default_scale}) × latent_size({latent_size})'
        }

    def list_all_models(self) -> list:
        """List all models with metadata."""
        self._fetch_metadata()
        return self._models_cache

if __name__ == '__main__':
    import sys

    metadata = ModelMetadata()

    if len(sys.argv) > 1:
        # Query specific model
        model_file = sys.argv[1]
        info = metadata.get_latent_info(model_file)

        if 'error' in info:
            print(f"Error: {info['error']}")
        else:
            print(f"Model: {info['name']}")
            print(f"Version: {info['version']}")
            print(f"Default scale: {info['default_scale']}")
            print(f"VAE compression: {info['vae_compression']}x")
            print(f"Latent size: {info['latent_size']}")
            print(f"Default output: {info['default_output_size']}×{info['default_output_size']}")
            print(f"Formula: {info['formula']}")
            print(f"Autoencoder: {info['autoencoder']}")
    else:
        # List all models
        print("Usage: python model_metadata.py <model_filename>")
        print("\nExample:")
        print("  python model_metadata.py z_image_turbo_1.0_q8p.ckpt")
