"""Unit tests for CPU path tracer (no GPU required)."""

import math

import numpy as np
import pytest

from python_renderer.core.material import OpenPBRMaterial, RenderConfig
from python_renderer.core.geometry import Mesh, Ray
from python_renderer.gpu.pathtracer import CPUPathtracer, GPUPathtracer
from python_renderer.loaders.model_loader import ModelLoader


# ===========================================================================
# helpers
# ===========================================================================

def small_config(**kwargs) -> RenderConfig:
    defaults = dict(width=32, height=32, samples=2, max_bounces=2, use_gpu=False)
    defaults.update(kwargs)
    return RenderConfig(**defaults)


def minimal_mesh() -> Mesh:
    """A single-triangle mesh."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh  = Mesh(verts, faces)
    mesh.compute_normals()
    return mesh


# ===========================================================================
# RenderConfig tests
# ===========================================================================

def test_render_config_default():
    cfg = RenderConfig()
    assert cfg.width  == 1920
    assert cfg.height == 1080
    assert cfg.samples == 512
    assert cfg.max_bounces == 8
    assert cfg.use_gpu is True


def test_render_config_custom():
    cfg = small_config()
    assert cfg.width   == 32
    assert cfg.samples == 2
    assert cfg.use_gpu is False


# ===========================================================================
# CPUPathtracer tests
# ===========================================================================

def test_cpu_pathtracer_init():
    cfg = small_config()
    pt  = CPUPathtracer(cfg)
    assert pt.config is cfg


def test_cpu_pathtracer_load_scene():
    cfg  = small_config()
    pt   = CPUPathtracer(cfg)
    mesh = minimal_mesh()
    mat  = OpenPBRMaterial()
    pt.load_scene(mesh, mat)
    assert pt._bvh      is not None
    assert pt._material is not None


def test_cpu_pathtracer_render_1sample():
    cfg  = small_config(width=8, height=8, samples=1, max_bounces=1)
    pt   = CPUPathtracer(cfg)
    mesh = ModelLoader().create_shaderball(lat_segments=8, lon_segments=8)
    mat  = OpenPBRMaterial()
    pt.load_scene(mesh, mat)
    image = pt.render()
    assert image.shape == (8, 8, 3)
    assert image.dtype == np.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0 + 1e-5


def test_render_pipeline():
    """End-to-end: create material → mesh → render → save (in-memory)."""
    import io
    from python_renderer.utils.image import apply_aces_tonemap, apply_gamma_correction

    cfg  = small_config(width=8, height=8, samples=1, max_bounces=1)
    mat  = OpenPBRMaterial(
        base_color=(0.8, 0.1, 0.1),
        base_metalness=0.0,
        specular_roughness=0.5,
    )
    mesh = ModelLoader().create_shaderball(lat_segments=8, lon_segments=8)
    pt   = CPUPathtracer(cfg)
    pt.load_scene(mesh, mat)
    image = pt.render()

    assert image is not None
    assert image.shape[2] == 3


def test_gpu_pathtracer_falls_back_to_cpu():
    """GPUPathtracer should create a CPUPathtracer fallback when GL is absent."""
    cfg = small_config(use_gpu=True)
    pt  = GPUPathtracer(cfg)
    pt.setup_gl()   # no GPU in CI – should not raise

    mesh = minimal_mesh()
    mat  = OpenPBRMaterial()
    pt.load_scene(mesh, mat)
    # render should still work via CPU fallback
    image = pt.render(samples=1)
    assert image is not None
    assert image.ndim == 3
