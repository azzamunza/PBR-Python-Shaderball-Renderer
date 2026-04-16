"""
CuPy CUDA kernels for GPU-accelerated image operations.

All kernels gracefully fall back to NumPy/SciPy when CuPy is not available.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupy import RawKernel
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    log.debug("CuPy not available – CUDA operations will run on CPU.")

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ===========================================================================
# CUDA kernel source strings
# ===========================================================================

_GAUSSIAN_BLUR_SRC = r"""
extern "C" __global__
void gaussian_blur_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int width, int height, int channels,
    float sigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int radius = (int)(3.0f * sigma + 0.5f);
    float inv2sigma2 = 1.0f / (2.0f * sigma * sigma);
    float wsum = 0.0f;
    float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = min(max(x + dx, 0), width  - 1);
            int ny = min(max(y + dy, 0), height - 1);
            float w = expf(-(dx*dx + dy*dy) * inv2sigma2);
            int idx = (ny * width + nx) * channels;
            for (int c = 0; c < channels; ++c)
                vals[c] += w * input[idx + c];
            wsum += w;
        }
    }

    int out_idx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c)
        output[out_idx + c] = vals[c] / wsum;
}
"""

_BILATERAL_FILTER_SRC = r"""
extern "C" __global__
void bilateral_filter_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int width, int height, int channels,
    float sigma_space, float sigma_color)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int radius = (int)(3.0f * sigma_space + 0.5f);
    float inv2ss2 = 1.0f / (2.0f * sigma_space * sigma_space);
    float inv2sc2 = 1.0f / (2.0f * sigma_color * sigma_color);
    float wsum = 0.0f;
    float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int c_idx = (y * width + x) * channels;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = min(max(x + dx, 0), width  - 1);
            int ny = min(max(y + dy, 0), height - 1);
            int n_idx = (ny * width + nx) * channels;

            float ws = expf(-(dx*dx + dy*dy) * inv2ss2);
            float wc = 0.0f;
            for (int c = 0; c < channels; ++c) {
                float diff = input[n_idx + c] - input[c_idx + c];
                wc += diff * diff;
            }
            wc = expf(-wc * inv2sc2);
            float w = ws * wc;

            for (int c = 0; c < channels; ++c)
                vals[c] += w * input[n_idx + c];
            wsum += w;
        }
    }

    int out_idx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c)
        output[out_idx + c] = vals[c] / wsum;
}
"""

_TONEMAP_SRC = r"""
extern "C" __global__
void tone_mapping_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int   width, int height, int channels,
    int   mode,   // 0=ACES, 1=Reinhard, 2=Gamma
    float gamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    for (int c = 0; c < channels && c < 3; ++c) {
        float v = input[idx + c];
        if (mode == 0) {
            // ACES
            v = (v * (2.51f * v + 0.03f)) / (v * (2.43f * v + 0.59f) + 0.14f);
        } else if (mode == 1) {
            // Reinhard
            v = v / (1.0f + v);
        }
        // Gamma
        output[idx + c] = powf(fmaxf(v, 0.0f), 1.0f / gamma);
    }
    if (channels == 4) output[idx + 3] = input[idx + 3];
}
"""


# ===========================================================================
# CUDADenoiser
# ===========================================================================

class CUDADenoiser:
    """
    GPU-accelerated denoiser using a bilateral filter.

    Falls back to SciPy when CuPy is unavailable.
    """

    def __init__(self, sigma_space: float = 3.0, sigma_color: float = 0.1) -> None:
        self.sigma_space = sigma_space
        self.sigma_color = sigma_color
        self._kernel: Optional[object] = None

        if _CUPY_AVAILABLE:
            try:
                self._kernel = RawKernel(_BILATERAL_FILTER_SRC, "bilateral_filter_kernel")
            except Exception as exc:
                log.warning("Failed to compile bilateral CUDA kernel: %s", exc)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter denoising.

        Parameters
        ----------
        image : np.ndarray  (H, W, C) float32

        Returns
        -------
        np.ndarray  same shape as input
        """
        if _CUPY_AVAILABLE and self._kernel is not None:
            return self._denoise_cuda(image)
        return self._denoise_cpu(image)

    def _denoise_cuda(self, image: np.ndarray) -> np.ndarray:
        h, w, c = image.shape
        img_gpu = cp.asarray(image, dtype=cp.float32)
        out_gpu = cp.zeros_like(img_gpu)
        block = (16, 16, 1)
        grid  = ((w + 15) // 16, (h + 15) // 16, 1)
        self._kernel(
            grid, block,
            (img_gpu, out_gpu,
             np.int32(w), np.int32(h), np.int32(c),
             np.float32(self.sigma_space), np.float32(self.sigma_color)),
        )
        return cp.asnumpy(out_gpu)

    def _denoise_cpu(self, image: np.ndarray) -> np.ndarray:
        if not _SCIPY_AVAILABLE:
            return image
        from scipy.ndimage import gaussian_filter
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = gaussian_filter(image[:, :, c], sigma=self.sigma_space)
        return result


# ===========================================================================
# CUDAMath – GPU vectorised math helpers
# ===========================================================================

class CUDAMath:
    """Vectorised math operations that run on CuPy arrays when available."""

    @staticmethod
    def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _CUPY_AVAILABLE:
            return cp.asnumpy(cp.einsum("...i,...i->...", cp.asarray(a), cp.asarray(b)))
        return np.einsum("...i,...i->...", a, b)

    @staticmethod
    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if _CUPY_AVAILABLE:
            return cp.asnumpy(cp.cross(cp.asarray(a), cp.asarray(b)))
        return np.cross(a, b)

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        if _CUPY_AVAILABLE:
            v_gpu = cp.asarray(v, dtype=cp.float32)
            norms = cp.linalg.norm(v_gpu, axis=-1, keepdims=True)
            return cp.asnumpy(v_gpu / cp.maximum(norms, 1e-8))
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norms, 1e-8)

    @staticmethod
    def apply_tonemap(image: np.ndarray, mode: str = "aces", gamma: float = 2.2) -> np.ndarray:
        """Apply tone mapping; uses CUDA kernel when available."""
        if _CUPY_AVAILABLE:
            try:
                kernel = RawKernel(_TONEMAP_SRC, "tone_mapping_kernel")
                h, w, c = image.shape
                img_gpu = cp.asarray(image, dtype=cp.float32)
                out_gpu = cp.zeros_like(img_gpu)
                mode_id = {"aces": 0, "reinhard": 1, "gamma": 2}.get(mode, 0)
                block = (16, 16, 1)
                grid  = ((w + 15) // 16, (h + 15) // 16, 1)
                kernel(grid, block,
                       (img_gpu, out_gpu,
                        np.int32(w), np.int32(h), np.int32(c),
                        np.int32(mode_id), np.float32(gamma)))
                return cp.asnumpy(out_gpu)
            except Exception as exc:
                log.debug("CUDA tonemap failed: %s", exc)

        # CPU fallback
        img = image.copy()
        if mode == "aces":
            img[:, :, :3] = (img[:, :, :3] * (2.51 * img[:, :, :3] + 0.03)) / \
                            (img[:, :, :3] * (2.43 * img[:, :, :3] + 0.59) + 0.14)
        elif mode == "reinhard":
            img[:, :, :3] = img[:, :, :3] / (1.0 + img[:, :, :3])
        img = np.clip(img, 0, None)
        img[:, :, :3] = np.power(img[:, :, :3], 1.0 / gamma)
        return img
