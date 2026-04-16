"""Unit tests for OpenPBRMaterial."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from python_renderer.core.material import OpenPBRMaterial, RenderConfig
from python_renderer.core.types import TextureType, OutputFormat


# ===========================================================================
# helpers
# ===========================================================================

def gold_material() -> OpenPBRMaterial:
    return OpenPBRMaterial(
        name="Gold",
        base_color=(1.0, 0.766, 0.336),
        base_metalness=1.0,
        specular_roughness=0.1,
        specular_ior=0.47,
    )


# ===========================================================================
# tests
# ===========================================================================

def test_default_material():
    mat = OpenPBRMaterial()
    assert mat.base_weight == 1.0
    assert mat.specular_ior == 1.5
    assert mat.geometry_opacity == 1.0
    assert mat.emission_weight == 0.0
    assert mat.geometry_thin_walled is False


def test_from_dict():
    d = {
        "base": {"weight": 0.9, "color": [0.5, 0.5, 0.5], "metalness": 0.0, "diffuse_roughness": 0.1},
        "specular": {"weight": 1.0, "roughness": 0.4, "ior": 1.6},
        "emission": {"weight": 0.0, "luminance": 1.0, "color": [1.0, 1.0, 1.0]},
        "geometry": {"opacity": 1.0, "thin_walled": False},
    }
    mat = OpenPBRMaterial.from_dict(d)
    assert math.isclose(mat.base_weight, 0.9)
    assert math.isclose(mat.specular_ior, 1.6)
    assert mat.base_color == (0.5, 0.5, 0.5)


def test_to_dict_roundtrip():
    mat  = gold_material()
    d    = mat.to_dict()
    mat2 = OpenPBRMaterial.from_dict(d)
    assert mat.base_color     == mat2.base_color
    assert mat.base_metalness == mat2.base_metalness
    assert math.isclose(mat.specular_ior, mat2.specular_ior)


def test_save_load_json(tmp_path: Path):
    mat  = gold_material()
    path = str(tmp_path / "gold.json")
    mat.save_json(path)
    mat2 = OpenPBRMaterial.load_json(path)
    assert mat2.name == "Gold"
    assert math.isclose(mat2.base_metalness, 1.0)
    assert mat2.base_color == mat.base_color


def test_save_load_yaml(tmp_path: Path):
    mat  = gold_material()
    path = str(tmp_path / "gold.yaml")
    mat.save_yaml(path)
    mat2 = OpenPBRMaterial.load_yaml(path)
    assert math.isclose(mat2.specular_roughness, 0.1)


def test_pack_for_gpu():
    mat = gold_material()
    buf = mat.pack_for_gpu()
    assert buf.dtype == np.float32
    assert buf.shape == (64,)
    # base_color at indices 0-2
    assert math.isclose(buf[0], 1.0, abs_tol=1e-5)
    assert math.isclose(buf[1], 0.766, abs_tol=1e-3)
    # metalness at index 11
    assert math.isclose(buf[11], 1.0, abs_tol=1e-5)
    # specular_roughness at index 8
    assert math.isclose(buf[8], 0.1, abs_tol=1e-5)


def test_gold_material():
    mat = gold_material()
    assert mat.name == "Gold"
    assert mat.base_metalness == 1.0
    assert mat.specular_ior == pytest.approx(0.47)


def test_render_config_defaults():
    cfg = RenderConfig()
    assert cfg.width  == 1920
    assert cfg.height == 1080
    assert cfg.samples == 512
    assert cfg.use_gpu is True
    assert cfg.output_format == OutputFormat.PNG


def test_render_config_roundtrip(tmp_path: Path):
    cfg  = RenderConfig(width=512, height=512, samples=64, denoising=True)
    path = str(tmp_path / "config.yaml")
    cfg.save_yaml(path)
    cfg2 = RenderConfig.load_yaml(path)
    assert cfg2.width   == 512
    assert cfg2.samples == 64
    assert cfg2.denoising is True
