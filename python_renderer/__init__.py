"""PBR Python Shaderball Renderer – GPU-accelerated OpenPBR v1.2 pathtracer."""

__version__ = "1.0.0"
__author__  = "azzamunza"

from python_renderer.core.material import OpenPBRMaterial, RenderConfig
from python_renderer.core.geometry import Mesh, Triangle, Ray
from python_renderer.gpu.pathtracer import GPUPathtracer

__all__ = [
    "OpenPBRMaterial",
    "RenderConfig",
    "Mesh",
    "Triangle",
    "Ray",
    "GPUPathtracer",
]
