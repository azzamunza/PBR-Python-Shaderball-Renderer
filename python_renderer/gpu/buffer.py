"""OpenGL GPU buffer management for the pathtracer."""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from OpenGL.GL import (
        glGenBuffers, glBindBuffer, glBufferData, glGetBufferSubData,
        glDeleteBuffers, glBindBufferBase,
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
        glActiveTexture, glDeleteTextures,
        GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, GL_STATIC_DRAW,
        GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        GL_LINEAR, GL_RGBA32F, GL_RGBA, GL_FLOAT,
        GL_TEXTURE0,
    )
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False

    # Minimal stubs so the module loads without OpenGL
    GL_SHADER_STORAGE_BUFFER = 0x90D2


class GPUBuffer:
    """
    Wrapper around an OpenGL Shader Storage Buffer Object (SSBO).

    Falls back to a plain numpy array when OpenGL is unavailable.
    """

    def __init__(
        self,
        data: np.ndarray,
        buffer_type: int = GL_SHADER_STORAGE_BUFFER,  # type: ignore[assignment]
        dynamic: bool = True,
    ) -> None:
        self._data   = np.ascontiguousarray(data, dtype=np.float32)
        self._type   = buffer_type
        self._handle: Optional[int] = None

        if _GL_AVAILABLE:
            usage = GL_DYNAMIC_DRAW if dynamic else GL_STATIC_DRAW
            self._handle = int(glGenBuffers(1))
            glBindBuffer(self._type, self._handle)
            glBufferData(self._type, self._data.nbytes, self._data, usage)
            glBindBuffer(self._type, 0)

    def bind(self, binding_point: int) -> None:
        """Bind the SSBO to the given binding point."""
        if _GL_AVAILABLE and self._handle is not None:
            glBindBufferBase(self._type, binding_point, self._handle)

    def update(self, data: np.ndarray) -> None:
        """Upload new data to the buffer."""
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        if _GL_AVAILABLE and self._handle is not None:
            glBindBuffer(self._type, self._handle)
            glBufferData(self._type, self._data.nbytes, self._data, GL_DYNAMIC_DRAW)
            glBindBuffer(self._type, 0)

    def read(self) -> np.ndarray:
        """Download buffer contents back to CPU."""
        if _GL_AVAILABLE and self._handle is not None:
            glBindBuffer(self._type, self._handle)
            raw = glGetBufferSubData(self._type, 0, self._data.nbytes)
            glBindBuffer(self._type, 0)
            return np.frombuffer(raw, dtype=np.float32).copy()
        return self._data.copy()

    def delete(self) -> None:
        """Release the OpenGL resource."""
        if _GL_AVAILABLE and self._handle is not None:
            glDeleteBuffers(1, [self._handle])
            self._handle = None

    def __del__(self) -> None:
        self.delete()


class TextureBuffer:
    """
    Wrapper around an OpenGL 2-D texture for environment maps and material
    texture maps.
    """

    def __init__(self) -> None:
        self._handle: Optional[int] = None
        self._width  = 0
        self._height = 0

    def upload(self, image_array: np.ndarray) -> None:
        """
        Upload an RGBA float32 image to a GL_TEXTURE_2D.

        image_array must be (H, W, 4) float32.
        """
        if not _GL_AVAILABLE:
            self._cpu_data = image_array.copy()
            return

        img = np.ascontiguousarray(image_array, dtype=np.float32)
        if img.ndim == 2:
            # Greyscale → RGBA
            tmp = np.zeros((*img.shape, 4), dtype=np.float32)
            tmp[:, :, 0] = img
            tmp[:, :, 1] = img
            tmp[:, :, 2] = img
            tmp[:, :, 3] = 1.0
            img = tmp
        elif img.shape[2] == 3:
            tmp = np.ones((*img.shape[:2], 4), dtype=np.float32)
            tmp[:, :, :3] = img
            img = tmp

        self._height, self._width = img.shape[:2]

        if self._handle is None:
            self._handle = int(glGenTextures(1))

        glBindTexture(GL_TEXTURE_2D, self._handle)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F,
            self._width, self._height, 0,
            GL_RGBA, GL_FLOAT, img,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    def bind(self, unit: int = 0) -> None:
        """Bind the texture to the given texture unit."""
        if _GL_AVAILABLE and self._handle is not None:
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, self._handle)

    def delete(self) -> None:
        """Release the OpenGL resource."""
        if _GL_AVAILABLE and self._handle is not None:
            glDeleteTextures(1, [self._handle])
            self._handle = None

    def __del__(self) -> None:
        self.delete()
