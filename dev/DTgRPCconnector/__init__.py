"""
Draw Things Python gRPC Client

A modular Python client library for the Draw Things image generation server.

Example usage:
    from drawthings_client import DrawThingsClient, ImageGenerationConfig

    config = ImageGenerationConfig(
        model="realDream",
        steps=16,
        width=512,
        height=512,
        cfg_scale=5.0,
        scheduler="UniPC ays"
    )

    with DrawThingsClient("192.168.2.150:7859") as client:
        images = client.generate_image("A beautiful sunset", config)
        client.save_images(images, output_dir="output")
"""

from drawthings_client import (
    DrawThingsClient,
    ImageGenerationConfig,
    StreamingProgressHandler,
    quick_generate,
    SCHEDULER_MAP
)

__version__ = "1.0.0"
__author__ = "Draw Things Community"
__all__ = [
    "DrawThingsClient",
    "ImageGenerationConfig",
    "StreamingProgressHandler",
    "quick_generate",
    "SCHEDULER_MAP"
]
