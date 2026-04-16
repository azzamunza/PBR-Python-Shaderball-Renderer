"""
simple_render.py – Minimal working example.

Renders a gold shaderball using the CPU path tracer and saves render.png.
"""

import logging
import sys
from pathlib import Path

# Ensure the package root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python_renderer.core.material import OpenPBRMaterial, RenderConfig
from python_renderer.loaders.model_loader import ModelLoader
from python_renderer.gpu.pathtracer import GPUPathtracer
from python_renderer.utils.logging import setup_logging

setup_logging(logging.INFO)
log = logging.getLogger(__name__)


def make_gold_material() -> OpenPBRMaterial:
    """Create an OpenPBR gold material."""
    return OpenPBRMaterial(
        name="Gold",
        base_color=(1.0, 0.766, 0.336),
        base_weight=1.0,
        base_metalness=1.0,
        base_diffuse_roughness=0.0,
        specular_weight=1.0,
        specular_color=(1.0, 0.766, 0.336),
        specular_roughness=0.1,
        specular_ior=0.47,
        emission_weight=0.0,
        geometry_opacity=1.0,
        geometry_thin_walled=False,
    )


def main() -> None:
    # ---- configuration -------------------------------------------------------
    config = RenderConfig(
        width=512,
        height=512,
        samples=32,          # low sample count for a quick demo
        max_bounces=6,
        use_gpu=False,       # CPU path tracer – works without a GPU
        output_path="render.png",
    )

    # ---- material ------------------------------------------------------------
    material = make_gold_material()
    log.info("Material: %s", material.name)

    # ---- geometry ------------------------------------------------------------
    loader = ModelLoader()
    mesh   = loader.create_shaderball(lat_segments=32, lon_segments=32)

    # ---- render --------------------------------------------------------------
    pt = GPUPathtracer(config)  # auto-falls-back to CPU
    pt.setup_gl()
    pt.load_scene(mesh, material)

    log.info("Starting render …")
    image = pt.render()

    # ---- save ----------------------------------------------------------------
    pt.save(config.output_path, image)
    log.info("Saved: %s", config.output_path)


if __name__ == "__main__":
    main()
