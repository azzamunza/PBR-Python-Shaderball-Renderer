"""Texture loading, conversion, and caching utilities."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from python_renderer.core.types import TextureType

log = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import imageio.v3 as iio
    _IMAGEIO_AVAILABLE = True
except ImportError:
    try:
        import imageio as iio  # type: ignore[no-redef]
        _IMAGEIO_AVAILABLE = True
    except ImportError:
        _IMAGEIO_AVAILABLE = False


class TextureCache:
    """LRU-style in-memory texture cache keyed on (path, texture_type)."""

    def __init__(self, max_entries: int = 64) -> None:
        self._cache: Dict[str, np.ndarray] = {}
        self._max   = max_entries

    def _key(self, path: str, tex_type: TextureType) -> str:
        return f"{path}::{tex_type.name}"

    def get(self, path: str, tex_type: TextureType) -> Optional[np.ndarray]:
        return self._cache.get(self._key(path, tex_type))

    def put(self, path: str, tex_type: TextureType, data: np.ndarray) -> None:
        if len(self._cache) >= self._max:
            # Evict oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[self._key(path, tex_type)] = data

    def clear(self) -> None:
        self._cache.clear()


class TextureLoader:
    """Load, convert, and resize texture maps for use in the renderer."""

    def __init__(self, cache: Optional[TextureCache] = None) -> None:
        self._cache = cache or TextureCache()

    # ---- primary loading ----------------------------------------------------
    def load(self, path: str, texture_type: TextureType = TextureType.ALBEDO) -> np.ndarray:
        """
        Load a texture from disk and return a float32 numpy array.

        The returned array has shape (H, W, C) with values in [0, 1] in
        linear colour space.
        """
        cached = self._cache.get(path, texture_type)
        if cached is not None:
            return cached

        img = self._load_image(path)
        if texture_type == TextureType.ALBEDO:
            img = self.to_linear(img)
        self._cache.put(path, texture_type, img)
        return img

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from disk, normalised to float32 [0, 1]."""
        ext = Path(path).suffix.lower()

        if ext in (".hdr", ".exr", ".rgbe"):
            return self.load_hdr(path)

        if _PIL_AVAILABLE:
            pil = PILImage.open(path).convert("RGBA")
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            return arr

        if _IMAGEIO_AVAILABLE:
            arr = np.asarray(iio.imread(path), dtype=np.float32)
            if arr.max() > 1.0:
                arr /= 255.0
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            if arr.shape[2] == 3:
                alpha = np.ones((*arr.shape[:2], 1), dtype=np.float32)
                arr   = np.concatenate([arr, alpha], axis=2)
            return arr

        raise RuntimeError(
            f"Cannot load texture '{path}': install Pillow or imageio."
        )

    def load_hdr(self, path: str) -> np.ndarray:
        """
        Load an HDR or OpenEXR environment map.

        Returns a float32 (H, W, 3) array in linear light.
        """
        if _IMAGEIO_AVAILABLE:
            try:
                arr = np.asarray(iio.imread(path), dtype=np.float32)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                return arr[:, :, :3]
            except Exception as exc:
                log.warning("imageio failed for '%s': %s", path, exc)

        if _PIL_AVAILABLE:
            try:
                pil = PILImage.open(path)
                arr = np.asarray(pil, dtype=np.float32)
                if arr.max() > 1.0:
                    arr /= arr.max()
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                return arr[:, :, :3]
            except Exception as exc:
                log.warning("PIL failed for '%s': %s", path, exc)

        # Minimal fallback: 1×1 white
        log.warning("load_hdr: using 1x1 white fallback for '%s'", path)
        return np.ones((1, 1, 3), dtype=np.float32)

    # ---- resize & conversion ------------------------------------------------
    def resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image to (height, width) using bilinear interpolation."""
        if image.shape[0] == height and image.shape[1] == width:
            return image
        if _PIL_AVAILABLE:
            channels = image.shape[2] if image.ndim == 3 else 1
            pil = PILImage.fromarray(
                (np.clip(image, 0, 1) * 255).astype(np.uint8)
                if channels == 4 else
                (np.clip(image, 0, 1) * 255).astype(np.uint8)
            )
            pil = pil.resize((width, height), PILImage.BILINEAR)
            return np.asarray(pil, dtype=np.float32) / 255.0

        # Simple nearest-neighbour fallback
        y = (np.arange(height) * image.shape[0] / height).astype(int)
        x = (np.arange(width)  * image.shape[1] / width ).astype(int)
        return image[np.ix_(y, x)]

    @staticmethod
    def to_linear(image: np.ndarray) -> np.ndarray:
        """Convert sRGB [0,1] → linear float32."""
        img = np.clip(image, 0.0, 1.0)
        return np.where(
            img <= 0.04045,
            img / 12.92,
            ((img + 0.055) / 1.055) ** 2.4,
        ).astype(np.float32)

    @staticmethod
    def to_srgb(image: np.ndarray) -> np.ndarray:
        """Convert linear float32 → sRGB [0,1]."""
        img = np.clip(image, 0.0, None)
        return np.where(
            img <= 0.0031308,
            img * 12.92,
            1.055 * np.power(img, 1.0 / 2.4) - 0.055,
        ).astype(np.float32)

    def generate_mipmaps(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generate a full mipmap chain down to 1×1.

        Returns a list of numpy arrays from full resolution to 1×1.
        """
        mips = [image]
        current = image
        while current.shape[0] > 1 or current.shape[1] > 1:
            new_h = max(1, current.shape[0] // 2)
            new_w = max(1, current.shape[1] // 2)
            current = self.resize(current, new_w, new_h)
            mips.append(current)
        return mips
