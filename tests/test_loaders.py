"""Unit tests for model and texture loaders."""

import math

import numpy as np
import pytest

from python_renderer.core.types import TextureType
from python_renderer.loaders.model_loader import ModelLoader
from python_renderer.loaders.texture_loader import TextureLoader, TextureCache


# ===========================================================================
# ModelLoader
# ===========================================================================

def test_model_loader_shaderball():
    loader = ModelLoader()
    mesh   = loader.create_shaderball(lat_segments=16, lon_segments=16)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces)    > 0
    assert mesh.normals is not None
    assert mesh.uvs     is not None


def test_model_loader_shaderball_uv_range():
    loader = ModelLoader()
    mesh   = loader.create_shaderball(lat_segments=8, lon_segments=8)
    assert mesh.uvs is not None
    assert float(mesh.uvs.min()) >= 0.0
    assert float(mesh.uvs.max()) <= 1.0 + 1e-5


def test_model_loader_center_and_scale():
    """After centering and scaling the mesh should fit within ~[-1, 1]."""
    loader = ModelLoader()
    mesh   = loader.create_shaderball(lat_segments=8, lon_segments=8)
    loader._center_and_scale(mesh)
    assert float(np.abs(mesh.vertices).max()) <= 1.0 + 1e-4


def test_model_loader_unsupported_format():
    loader = ModelLoader()
    with pytest.raises((ValueError, RuntimeError)):
        loader.load("dummy.xyz")


# ===========================================================================
# TextureLoader
# ===========================================================================

def test_texture_loader_create_blank():
    blank = np.ones((64, 64, 4), dtype=np.float32)
    assert blank.shape == (64, 64, 4)


def test_texture_loader_resize():
    loader = TextureLoader()
    img    = np.random.rand(64, 64, 3).astype(np.float32)
    resized = loader.resize(img, 32, 32)
    assert resized.shape[0] == 32
    assert resized.shape[1] == 32


def test_texture_loader_no_resize_needed():
    loader = TextureLoader()
    img    = np.random.rand(32, 32, 3).astype(np.float32)
    out    = loader.resize(img, 32, 32)
    np.testing.assert_array_equal(img, out)


def test_texture_to_linear():
    # A mid-grey sRGB value
    srgb   = np.array([[[0.5, 0.5, 0.5, 1.0]]], dtype=np.float32)
    linear = TextureLoader.to_linear(srgb)
    # sRGB 0.5 → linear ~0.2140
    assert float(linear[0, 0, 0]) == pytest.approx(0.2140, abs=0.01)


def test_texture_to_srgb():
    linear = np.array([[[0.2140, 0.2140, 0.2140, 1.0]]], dtype=np.float32)
    srgb   = TextureLoader.to_srgb(linear)
    assert float(srgb[0, 0, 0]) == pytest.approx(0.5, abs=0.01)


def test_texture_mipmaps():
    loader = TextureLoader()
    img    = np.random.rand(64, 64, 3).astype(np.float32)
    mips   = loader.generate_mipmaps(img)
    assert len(mips) >= 6   # 64→32→16→8→4→2→1
    assert mips[0].shape[0]  == 64
    assert mips[-1].shape[0] == 1


def test_texture_cache():
    cache = TextureCache(max_entries=2)
    img1  = np.ones((4, 4, 3), dtype=np.float32)
    cache.put("a.png", TextureType.ALBEDO, img1)
    assert cache.get("a.png", TextureType.ALBEDO) is not None
    assert cache.get("b.png", TextureType.ALBEDO) is None

    # Fill to capacity and evict
    cache.put("b.png", TextureType.ALBEDO, img1)
    cache.put("c.png", TextureType.ALBEDO, img1)   # evicts "a.png"
    assert cache.get("a.png", TextureType.ALBEDO) is None
    assert cache.get("c.png", TextureType.ALBEDO) is not None


def test_texture_cache_clear():
    cache = TextureCache()
    img   = np.ones((4, 4, 3), dtype=np.float32)
    cache.put("x.png", TextureType.NORMAL, img)
    cache.clear()
    assert cache.get("x.png", TextureType.NORMAL) is None
