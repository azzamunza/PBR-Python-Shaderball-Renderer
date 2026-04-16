"""OpenPBR v1.2 material definition and render configuration."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import yaml

from python_renderer.core.types import TextureType, OutputFormat


def _tuples_to_lists(obj: Any) -> Any:
    """Recursively convert tuples to lists for YAML safe-serialisation."""
    if isinstance(obj, tuple):
        return [_tuples_to_lists(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tuples_to_lists(v) for v in obj]
    return obj


@dataclass
class OpenPBRMaterial:
    """
    Full OpenPBR v1.2 material definition.

    Implements all material layers as specified by the OpenPBR v1.2 specification:
    base, specular, transmission, subsurface, coat, fuzz, emission,
    thin_film, and geometry.
    """

    # ---- base layer ---------------------------------------------------------
    base_weight: float = 1.0
    base_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    base_diffuse_roughness: float = 0.0
    base_metalness: float = 0.0

    # ---- specular layer -----------------------------------------------------
    specular_weight: float = 1.0
    specular_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    specular_roughness: float = 0.3
    specular_anisotropy: float = 0.0
    specular_ior: float = 1.5
    specular_haze: float = 0.0
    specular_retroreflectivity: float = 0.0

    # ---- transmission layer -------------------------------------------------
    transmission_weight: float = 0.0
    transmission_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    transmission_depth: float = 0.0
    transmission_scatter: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    transmission_dispersion: float = 0.0

    # ---- subsurface layer ---------------------------------------------------
    subsurface_weight: float = 0.0
    subsurface_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    subsurface_radius: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    subsurface_anisotropy: float = 0.0

    # ---- coat layer ---------------------------------------------------------
    coat_weight: float = 0.0
    coat_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    coat_roughness: float = 0.0
    coat_ior: float = 1.5
    coat_darkening: float = 1.0

    # ---- fuzz layer ---------------------------------------------------------
    fuzz_weight: float = 0.0
    fuzz_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    fuzz_roughness: float = 0.5

    # ---- emission layer -----------------------------------------------------
    emission_weight: float = 0.0
    emission_luminance: float = 1.0
    emission_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # ---- thin-film layer ----------------------------------------------------
    thin_film_weight: float = 0.0
    thin_film_thickness: float = 500.0   # nanometres
    thin_film_ior: float = 1.5

    # ---- geometry parameters ------------------------------------------------
    geometry_opacity: float = 1.0
    geometry_thin_walled: bool = False

    # ---- texture maps -------------------------------------------------------
    texture_maps: Dict[TextureType, str] = field(default_factory=dict)

    # ---- optional metadata --------------------------------------------------
    name: str = "OpenPBR Material"

    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a nested dictionary matching OpenPBR layer grouping."""
        d = asdict(self)
        # Convert TextureType keys to strings for serialisation
        d["texture_maps"] = {k.name: v for k, v in self.texture_maps.items()}
        # Convert tuples → lists so YAML safe_load can round-trip without tags
        d = _tuples_to_lists(d)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenPBRMaterial":
        """Deserialise from a nested or flat dictionary."""
        flat: Dict[str, Any] = {}

        def _flatten(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _flatten(f"{prefix}_{k}" if prefix else k, v)
            else:
                flat[prefix] = obj

        for k, v in data.items():
            if k == "texture_maps":
                flat["texture_maps"] = {TextureType[tk]: tv for tk, tv in v.items()}
            else:
                _flatten(k, v)

        # Convert list → tuple for colour fields
        for key, val in flat.items():
            if isinstance(val, list):
                flat[key] = tuple(val)

        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in flat.items() if k in valid}
        return cls(**filtered)

    # ---- persistence --------------------------------------------------------
    def save_json(self, path: str) -> None:
        """Save material to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "OpenPBRMaterial":
        """Load material from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save_yaml(self, path: str) -> None:
        """Save material to a YAML file."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str) -> "OpenPBRMaterial":
        """Load material from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    # ---- GPU packing --------------------------------------------------------
    def pack_for_gpu(self) -> np.ndarray:
        """
        Pack material parameters into a flat float32 array suitable for
        uploading to a GPU shader storage buffer.

        Layout (std430, vec4-aligned):
          [0-2]   base_color          [3]   base_weight
          [4-6]   specular_color      [7]   specular_weight
          [8]     specular_roughness  [9]   specular_anisotropy
          [10]    specular_ior        [11]  base_metalness
          [12-14] transmission_color  [15]  transmission_weight
          [16]    transmission_depth  [17]  transmission_dispersion
          [18-19] padding
          [20-22] subsurface_color    [23]  subsurface_weight
          [24-26] subsurface_radius   [27]  subsurface_anisotropy
          [28-30] coat_color          [31]  coat_weight
          [32]    coat_roughness      [33]  coat_ior
          [34]    coat_darkening      [35]  padding
          [36-38] fuzz_color          [39]  fuzz_weight
          [40]    fuzz_roughness      [41]  emission_weight
          [42]    emission_luminance  [43]  geometry_opacity
          [44-46] emission_color      [47]  thin_film_weight
          [48]    thin_film_thickness [49]  thin_film_ior
          [50]    base_diffuse_roughness [51] geometry_thin_walled (0/1)
          [52-63] padding to 64 floats
        """
        buf = np.zeros(64, dtype=np.float32)
        buf[0:3]   = self.base_color
        buf[3]     = self.base_weight
        buf[4:7]   = self.specular_color
        buf[7]     = self.specular_weight
        buf[8]     = self.specular_roughness
        buf[9]     = self.specular_anisotropy
        buf[10]    = self.specular_ior
        buf[11]    = self.base_metalness
        buf[12:15] = self.transmission_color
        buf[15]    = self.transmission_weight
        buf[16]    = self.transmission_depth
        buf[17]    = self.transmission_dispersion
        buf[20:23] = self.subsurface_color
        buf[23]    = self.subsurface_weight
        buf[24:27] = self.subsurface_radius
        buf[27]    = self.subsurface_anisotropy
        buf[28:31] = self.coat_color
        buf[31]    = self.coat_weight
        buf[32]    = self.coat_roughness
        buf[33]    = self.coat_ior
        buf[34]    = self.coat_darkening
        buf[36:39] = self.fuzz_color
        buf[39]    = self.fuzz_weight
        buf[40]    = self.fuzz_roughness
        buf[41]    = self.emission_weight
        buf[42]    = self.emission_luminance
        buf[43]    = self.geometry_opacity
        buf[44:47] = self.emission_color
        buf[47]    = self.thin_film_weight
        buf[48]    = self.thin_film_thickness
        buf[49]    = self.thin_film_ior
        buf[50]    = self.base_diffuse_roughness
        buf[51]    = float(self.geometry_thin_walled)
        return buf


@dataclass
class RenderConfig:
    """Configuration for a render job."""

    width: int = 1920
    height: int = 1080
    samples: int = 512
    max_bounces: int = 8
    hdri_path: Optional[str] = None
    output_path: str = "render.png"
    output_format: OutputFormat = OutputFormat.PNG
    use_gpu: bool = True
    denoising: bool = False
    displacement_scale: float = 1.0
    tessellation_level: int = 4
    firefly_clamp: float = 10.0

    # Camera settings
    camera_fov: float = 45.0
    camera_distance: float = 3.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output_format"] = self.output_format.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderConfig":
        data = dict(data)
        if "output_format" in data:
            val = data["output_format"]
            if isinstance(val, str):
                data["output_format"] = OutputFormat(val.lower())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]

    def save_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str) -> "RenderConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))
