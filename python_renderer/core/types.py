"""Core enumerations and type definitions for the PBR renderer."""

from enum import Enum, auto


class TextureType(Enum):
    """Texture map types supported by the renderer."""
    ALBEDO = auto()
    NORMAL = auto()
    ROUGHNESS = auto()
    METALLIC = auto()
    AO = auto()
    DISPLACEMENT = auto()
    EMISSION = auto()
    HEIGHT = auto()


class DisplacementMode(Enum):
    """Displacement mapping coordinate space."""
    OBJECT_SPACE = auto()
    TANGENT_SPACE = auto()


class RenderMode(Enum):
    """Rendering backend selection."""
    PATHTRACER = auto()
    RASTERIZER = auto()


class OutputFormat(Enum):
    """Image output file formats."""
    PNG = "png"
    EXR = "exr"
    HDR = "hdr"
