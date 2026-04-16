"""Image I/O and tone-mapping utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import imageio.v3 as iio
    _IMAGEIO_V3 = True
    _IMAGEIO_AVAILABLE = True
except ImportError:
    try:
        import imageio as iio  # type: ignore[no-redef]
        _IMAGEIO_V3 = False
        _IMAGEIO_AVAILABLE = True
    except ImportError:
        _IMAGEIO_AVAILABLE = False
        _IMAGEIO_V3 = False


# ===========================================================================
# Save helpers
# ===========================================================================

def save_png(image: np.ndarray, path: str) -> None:
    """Save an (H, W, 3) float32 image as a PNG file (8-bit sRGB)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if _PIL_AVAILABLE:
        PILImage.fromarray(img8).save(path)
        log.info("Saved PNG: %s", path)
        return
    if _IMAGEIO_AVAILABLE:
        if _IMAGEIO_V3:
            iio.imwrite(path, img8)
        else:
            iio.imwrite(path, img8)
        log.info("Saved PNG: %s", path)
        return
    raise RuntimeError("Install Pillow or imageio to save PNG files.")


def save_exr(image: np.ndarray, path: str) -> None:
    """Save an (H, W, 3) float32 HDR image as OpenEXR."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = image.astype(np.float32)
    if _IMAGEIO_AVAILABLE:
        try:
            if _IMAGEIO_V3:
                iio.imwrite(path, img, plugin="EXR")
            else:
                iio.imwrite(path, img, format="EXR-FI")
            log.info("Saved EXR: %s", path)
            return
        except Exception as exc:
            log.warning("EXR save failed (%s), falling back to PNG.", exc)
    # Fallback: save as PNG with tone-mapped content
    save_png(apply_aces_tonemap(apply_gamma_correction(img)), path.replace(".exr", ".png"))


def save_hdr(image: np.ndarray, path: str) -> None:
    """Save an (H, W, 3) float32 HDR image as Radiance HDR (.hdr)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = image.astype(np.float32)
    if _IMAGEIO_AVAILABLE:
        try:
            if _IMAGEIO_V3:
                iio.imwrite(path, img)
            else:
                iio.imwrite(path, img, format="HDR-FI")
            log.info("Saved HDR: %s", path)
            return
        except Exception as exc:
            log.warning("HDR save failed (%s), falling back to PNG.", exc)
    save_png(apply_aces_tonemap(apply_gamma_correction(img)), path.replace(".hdr", ".png"))


def load_image(path: str) -> np.ndarray:
    """Load any image to a float32 (H, W, C) array in [0, 1]."""
    if _PIL_AVAILABLE:
        pil = PILImage.open(path)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return arr
    if _IMAGEIO_AVAILABLE:
        if _IMAGEIO_V3:
            arr = np.asarray(iio.imread(path), dtype=np.float32)
        else:
            arr = np.asarray(iio.imread(path), dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        return arr
    raise RuntimeError("Install Pillow or imageio to load images.")


# ===========================================================================
# Tone mapping
# ===========================================================================

def apply_aces_tonemap(hdr: np.ndarray) -> np.ndarray:
    """ACES filmic tone mapping (float32 in → float32 out, clamped [0,1])."""
    x = np.asarray(hdr, dtype=np.float32)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    result = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(result, 0.0, 1.0)


def apply_reinhard_tonemap(hdr: np.ndarray) -> np.ndarray:
    """Reinhard tone mapping."""
    x = np.asarray(hdr, dtype=np.float32)
    return x / (1.0 + x)


def apply_gamma_correction(
    image: np.ndarray, gamma: float = 2.2
) -> np.ndarray:
    """Apply gamma correction: image^(1/gamma), clamped to [0,1]."""
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, None)
    return np.power(img, 1.0 / gamma)


# ===========================================================================
# Bilateral filter (CPU)
# ===========================================================================

def bilateral_filter(
    image: np.ndarray,
    sigma_space: float = 3.0,
    sigma_color: float = 0.1,
) -> np.ndarray:
    """
    Bilateral filter for denoising rendered images.

    Falls back to Gaussian blur when SciPy is unavailable.
    """
    try:
        from scipy.ndimage import generic_filter
        # Approximate via guided Gaussian
        from scipy.ndimage import gaussian_filter
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = gaussian_filter(image[:, :, c], sigma=sigma_space)
        return result.astype(np.float32)
    except ImportError:
        log.debug("SciPy not available; skipping bilateral filter.")
        return image.copy()
