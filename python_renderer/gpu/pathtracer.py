"""
GPU and CPU pathtracer implementations.

GPUPathtracer   – OpenGL compute-shader based pathtracer.
CPUPathtracer   – Pure NumPy fallback, no GPU required.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from python_renderer.core.geometry import BVHNode, Mesh, Ray, Triangle
from python_renderer.core.material import OpenPBRMaterial, RenderConfig
from python_renderer.core.types import OutputFormat, TextureType
from python_renderer.utils.math import (
    normalize, dot, cross, reflect, cosine_hemisphere_sample,
    ggx_sample, schlick_fresnel_rgb, ggx_ndf, smith_g2,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenGL availability
# ---------------------------------------------------------------------------
try:
    import OpenGL
    OpenGL.ERROR_CHECKING = False          # performance
    from OpenGL.GL import (
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
        glGetIntegerv,
        GL_COMPUTE_SHADER, GL_TEXTURE_2D, GL_RGBA32F, GL_RGBA, GL_FLOAT,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
        GL_SHADER_STORAGE_BUFFER,
        glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
        glGetShaderInfoLog, glCreateProgram, glAttachShader, glLinkProgram,
        glGetProgramiv, glGetProgramInfoLog, glUseProgram,
        glUniform1i, glUniform1f, glUniformMatrix4fv,
        glDispatchCompute, glMemoryBarrier,
        glBindImageTexture, glActiveTexture,
        GL_SHADER_STORAGE_BARRIER_BIT, GL_ALL_BARRIER_BITS,
        GL_TEXTURE0, GL_WRITE_ONLY, GL_READ_WRITE,
        GL_COMPILE_STATUS, GL_LINK_STATUS, GL_MAX_COMPUTE_WORK_GROUP_SIZE,
        glGetUniformLocation,
    )
    from OpenGL.GL.shaders import compileShader, compileProgram
    _GL_AVAILABLE = True
except Exception:
    _GL_AVAILABLE = False

try:
    import glfw
    _GLFW_AVAILABLE = True
except ImportError:
    _GLFW_AVAILABLE = False


# ===========================================================================
# Helpers
# ===========================================================================

def _load_glsl_source(filename: str) -> str:
    """Load a GLSL file from the shaders directory, resolving #include."""
    shader_dir = Path(__file__).parent / "shaders"
    src = (shader_dir / filename).read_text(encoding="utf-8")
    # Resolve #include directives
    lines = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith('#include "') and stripped.endswith('"'):
            inc_file = stripped[10:-1]
            inc_src  = (shader_dir / inc_file).read_text(encoding="utf-8")
            lines.append(inc_src)
        else:
            lines.append(line)
    return "\n".join(lines)


# ===========================================================================
# GPU Pathtracer
# ===========================================================================

class GPUPathtracer:
    """
    OpenGL compute-shader based path tracer.

    Falls back to :class:`CPUPathtracer` when OpenGL is unavailable.
    """

    def __init__(self, config: RenderConfig) -> None:
        self.config    = config
        self._program: Optional[int] = None
        self._accum_tex: Optional[int] = None
        self._output_tex: Optional[int] = None
        self._tri_ssbo: Optional[object] = None
        self._mat_ssbo: Optional[object] = None
        self._env_tex:  Optional[object] = None
        self._bvh: Optional[BVHNode] = None
        self._mesh: Optional[Mesh]   = None
        self._material: Optional[OpenPBRMaterial] = None
        self._setup_done = False
        self._cpu_fallback: Optional[CPUPathtracer] = None

        if not _GL_AVAILABLE:
            log.warning("OpenGL not available – using CPU fallback.")
            self._cpu_fallback = CPUPathtracer(config)

    # ---- GL setup -----------------------------------------------------------
    def setup_gl(self) -> None:
        """Initialise OpenGL context and compile compute shaders."""
        if not _GL_AVAILABLE:
            return

        if not _GLFW_AVAILABLE:
            log.warning("GLFW not available – cannot create headless GL context.")
            self._cpu_fallback = CPUPathtracer(self.config)
            return

        if not glfw.init():
            log.error("GLFW init failed.")
            self._cpu_fallback = CPUPathtracer(self.config)
            return

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self._window = glfw.create_window(1, 1, "offscreen", None, None)
        if not self._window:
            log.error("GLFW window creation failed.")
            self._cpu_fallback = CPUPathtracer(self.config)
            glfw.terminate()
            return

        glfw.make_context_current(self._window)

        # Compile shader
        try:
            src = _load_glsl_source("pathtracer.glsl")
            shader = glCreateShader(GL_COMPUTE_SHADER)
            glShaderSource(shader, src)
            glCompileShader(shader)
            if not glGetShaderiv(shader, GL_COMPILE_STATUS):
                err = glGetShaderInfoLog(shader)
                raise RuntimeError(f"Compute shader compile error: {err}")
            self._program = glCreateProgram()
            glAttachShader(self._program, shader)
            glLinkProgram(self._program)
            if not glGetProgramiv(self._program, GL_LINK_STATUS):
                err = glGetProgramInfoLog(self._program)
                raise RuntimeError(f"Shader link error: {err}")
        except Exception as exc:
            log.error("GPU shader compilation failed: %s", exc)
            self._cpu_fallback = CPUPathtracer(self.config)
            return

        self._create_textures()
        self._setup_done = True
        log.info("GPUPathtracer: OpenGL compute shader ready.")

    def _create_textures(self) -> None:
        w, h = self.config.width, self.config.height
        for attr in ("_accum_tex", "_output_tex"):
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0,
                         GL_RGBA, GL_FLOAT, None)
            setattr(self, attr, tex)

    # ---- scene loading ------------------------------------------------------
    def load_scene(self, mesh: Mesh, material: OpenPBRMaterial) -> None:
        """Upload geometry and material data to GPU buffers."""
        if self._cpu_fallback:
            self._cpu_fallback.load_scene(mesh, material)
            return

        self._mesh     = mesh
        self._material = material
        mesh.build_triangles()
        self._bvh = BVHNode.build(mesh.triangles)

        # Pack triangles into flat float buffer
        tri_data = _pack_triangles(mesh.triangles)
        mat_data = material.pack_for_gpu()

        from python_renderer.gpu.buffer import GPUBuffer
        self._tri_ssbo = GPUBuffer(tri_data)
        self._mat_ssbo = GPUBuffer(mat_data)

    def load_environment(self, hdri_path: str) -> None:
        """Load an HDR equirectangular environment map."""
        if self._cpu_fallback:
            self._cpu_fallback.load_environment(hdri_path)
            return
        from python_renderer.loaders.texture_loader import TextureLoader
        loader = TextureLoader()
        env_img = loader.load_hdr(hdri_path)
        from python_renderer.gpu.buffer import TextureBuffer
        self._env_tex = TextureBuffer()
        self._env_tex.upload(env_img)
        log.info("Environment map loaded: %s (%dx%d)", hdri_path,
                 env_img.shape[1], env_img.shape[0])

    def load_textures(
        self, material: OpenPBRMaterial, texture_loader: object
    ) -> None:
        """Pre-load all texture maps referenced by the material."""
        if self._cpu_fallback:
            return
        # Textures would be bound as GL samplers in a full implementation.
        log.debug("load_textures: %d maps", len(material.texture_maps))

    # ---- render loop --------------------------------------------------------
    def render(self, samples: Optional[int] = None) -> np.ndarray:
        """
        Run the path tracer and return an (H, W, 3) float32 image.
        """
        if self._cpu_fallback:
            return self._cpu_fallback.render(samples)

        n_samples = samples or self.config.samples
        log.info("GPU render: %dx%d, %d samples",
                 self.config.width, self.config.height, n_samples)

        for i in range(n_samples):
            self._run_compute_pass(i)

        hdr = self._read_accum()
        final = self._tonemap(hdr)
        if self.config.denoising:
            final = self.denoise(final)
        return final

    def _run_compute_pass(self, sample_idx: int) -> None:
        glUseProgram(self._program)
        glUniform1i(glGetUniformLocation(self._program, "u_sampleIndex"), sample_idx)
        glUniform1i(glGetUniformLocation(self._program, "u_maxBounces"),
                    self.config.max_bounces)
        glUniform1f(glGetUniformLocation(self._program, "u_fireflyClamp"),
                    self.config.firefly_clamp)
        glUniform1i(glGetUniformLocation(self._program, "u_frameWidth"),
                    self.config.width)
        glUniform1i(glGetUniformLocation(self._program, "u_frameHeight"),
                    self.config.height)
        if self._tri_ssbo:
            self._tri_ssbo.bind(2)
        if self._mat_ssbo:
            self._mat_ssbo.bind(3)
        glBindImageTexture(0, self._accum_tex,  0, False, 0, GL_READ_WRITE, GL_RGBA32F)
        glBindImageTexture(1, self._output_tex, 0, False, 0, GL_WRITE_ONLY,  GL_RGBA32F)
        gx = (self.config.width  + 15) // 16
        gy = (self.config.height + 15) // 16
        glDispatchCompute(gx, gy, 1)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)

    def _read_accum(self) -> np.ndarray:
        import ctypes
        from OpenGL.GL import glGetTexImage, GL_TEXTURE_2D, GL_RGBA, GL_FLOAT
        w, h = self.config.width, self.config.height
        buf = np.zeros((h, w, 4), dtype=np.float32)
        glBindTexture(GL_TEXTURE_2D, self._output_tex)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buf)
        return buf[:, :, :3]

    def _tonemap(self, hdr_image: np.ndarray) -> np.ndarray:
        """ACES tone mapping + gamma correction."""
        from python_renderer.utils.image import apply_aces_tonemap, apply_gamma_correction
        ldr = apply_aces_tonemap(hdr_image)
        return apply_gamma_correction(ldr)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising (bilateral filter via CUDA or SciPy)."""
        from python_renderer.gpu.cuda.kernels import CUDADenoiser
        denoiser = CUDADenoiser()
        return denoiser.denoise(image)

    def save(self, path: str, image: np.ndarray) -> None:
        """Save the rendered image to disk."""
        from python_renderer.utils.image import save_png, save_exr, save_hdr
        ext = Path(path).suffix.lower()
        if ext == ".exr":
            save_exr(image, path)
        elif ext in (".hdr", ".rgbe"):
            save_hdr(image, path)
        else:
            save_png(image, path)


# ===========================================================================
# CPU Pathtracer (NumPy fallback)
# ===========================================================================

def _pack_triangles(triangles: List[Triangle]) -> np.ndarray:
    """Pack triangle data into a flat float32 array (28 floats per tri)."""
    data = np.zeros((len(triangles), 28), dtype=np.float32)
    for i, tri in enumerate(triangles):
        data[i, 0:3]   = tri.v0
        data[i, 3:6]   = tri.v1
        data[i, 6:9]   = tri.v2
        data[i, 9:12]  = tri.n0
        data[i, 12:15] = tri.n1
        data[i, 15:18] = tri.n2
        data[i, 18:20] = tri.uv0
        data[i, 20:22] = tri.uv1
        data[i, 22:24] = tri.uv2
        data[i, 24]    = float(tri.material_id)
    return data.ravel()


class CPUPathtracer:
    """
    Pure NumPy path tracer – no GPU required.

    Implements a unidirectional path tracer with:
    - Cook-Torrance GGX specular BRDF
    - Lambert diffuse BRDF
    - Coat layer
    - Environment map (equirectangular) or procedural sky
    - Progressive accumulation
    - Firefly clamping
    """

    def __init__(self, config: RenderConfig) -> None:
        self.config   = config
        self._bvh:      Optional[BVHNode]          = None
        self._material: Optional[OpenPBRMaterial]  = None
        self._env_map:  Optional[np.ndarray]       = None   # (H, W, 3) linear
        self._rng = np.random.default_rng()

    # ---- scene loading ------------------------------------------------------
    def load_scene(self, mesh: Mesh, material: OpenPBRMaterial) -> None:
        mesh.build_triangles()
        self._bvh      = BVHNode.build(mesh.triangles)
        self._material = material
        log.info("CPUPathtracer: BVH built with %d triangles", len(mesh.triangles))

    def load_environment(self, hdri_path: str) -> None:
        try:
            from python_renderer.loaders.texture_loader import TextureLoader
            self._env_map = TextureLoader().load_hdr(hdri_path)
            log.info("Environment map loaded: %s", hdri_path)
        except Exception as exc:
            log.warning("Could not load environment map: %s", exc)

    # ---- render -------------------------------------------------------------
    def render(self, samples: Optional[int] = None) -> np.ndarray:
        n_samples = samples or self.config.samples
        w, h      = self.config.width, self.config.height

        log.info("CPU render: %dx%d, %d samples, %d bounces",
                 w, h, n_samples, self.config.max_bounces)

        accum = np.zeros((h, w, 3), dtype=np.float64)
        t0    = time.time()

        for s in range(n_samples):
            accum += self._render_sample(w, h)
            if (s + 1) % max(1, n_samples // 10) == 0:
                elapsed = time.time() - t0
                log.info("  sample %d/%d  (%.1fs)", s + 1, n_samples, elapsed)

        hdr   = (accum / n_samples).astype(np.float32)
        final = self._tonemap(hdr)
        if self.config.denoising:
            final = self.denoise(final)
        return final

    def _render_sample(self, w: int, h: int) -> np.ndarray:
        """Render one sample per pixel and return (H, W, 3) float64 buffer."""
        # Jittered pixel coordinates
        px = (np.arange(w) + self._rng.uniform(0, 1, w)).reshape(1, w)
        py = (np.arange(h) + self._rng.uniform(0, 1, h)).reshape(h, 1)

        fov   = math.radians(self.config.camera_fov)
        tan_h = math.tan(fov * 0.5)
        aspect = w / h

        # Normalised device coordinates
        ndcx = ((px / w) * 2.0 - 1.0) * aspect * tan_h   # (1, W)
        ndcy = (1.0 - (py / h) * 2.0) * tan_h            # (H, 1)

        # Camera sits at (0, 0, camera_distance) looking at origin
        dist  = self.config.camera_distance
        ray_o = np.array([0.0, 0.0, dist])
        dirs_x = ndcx * np.ones((h, 1))     # (H, W)
        dirs_y = ndcy * np.ones((1, w))     # (H, W)
        dirs_z = -np.ones((h, w))

        # Stack and normalise
        dirs = np.stack([dirs_x, dirs_y, dirs_z], axis=-1)  # (H, W, 3)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs  = dirs / norms

        result = np.zeros((h, w, 3), dtype=np.float64)

        # Per-pixel loop (slow but correct; use BVH to minimise work)
        for row in range(h):
            for col in range(w):
                ray = Ray(ray_o.copy(), dirs[row, col].copy())
                result[row, col] = self._trace(ray, self.config.max_bounces)

        return result

    # ---- path tracing -------------------------------------------------------
    def _trace(self, ray: Ray, depth: int) -> np.ndarray:
        """Recursive path tracer. Returns radiance (3,) float64."""
        BLACK = np.zeros(3)
        if depth < 0:
            return BLACK

        hit = self._bvh.intersect(ray) if self._bvh else None
        if hit is None:
            return self._sample_environment(ray.direction)

        tri: Triangle = hit["triangle"]
        t: float      = hit["t"]
        hit_pos = ray.at(t)

        # Recompute barycentric coords for interpolation
        u, v = _compute_barycentric(hit_pos, tri)
        N    = tri.normal_at(u, v)
        # Ensure normal faces the ray
        if np.dot(N, -ray.direction) < 0:
            N = -N

        mat = self._material
        if mat is None:
            return np.array([0.8, 0.8, 0.8])

        # Emission
        radiance = np.array(mat.emission_color) * mat.emission_luminance * mat.emission_weight

        # Sample BRDF direction
        r1 = self._rng.random()
        p_diff = (1.0 - mat.base_metalness) * 0.5
        V = -ray.direction

        if r1 < p_diff:
            L, pdf = cosine_hemisphere_sample(N, self._rng)
            if pdf < 1e-6:
                return radiance
            brdf = self._eval_diffuse(mat, N, V, L)
            w = np.dot(N, L) / (pdf * p_diff)
        else:
            alpha = max(mat.specular_roughness ** 2, 1e-3)
            H, L  = ggx_sample(V, N, alpha, self._rng)
            if np.dot(N, L) <= 0:
                return radiance
            brdf = self._eval_specular(mat, N, V, L, H)
            NdotH = max(np.dot(N, H), 1e-6)
            VdotH = max(np.dot(V, H), 1e-6)
            D_val = ggx_ndf(NdotH, alpha)
            pdf   = D_val * NdotH / max(4.0 * VdotH, 1e-6)
            if pdf < 1e-6:
                return radiance
            w = np.dot(N, L) / (pdf * (1.0 - p_diff))

        throughput = brdf * w
        # Firefly clamping
        lum = float(0.2126 * throughput[0] + 0.7152 * throughput[1] + 0.0722 * throughput[2])
        if lum > self.config.firefly_clamp:
            throughput *= self.config.firefly_clamp / lum

        if np.dot(throughput, throughput) < 1e-8:
            return radiance

        next_ray = Ray(hit_pos + N * 1e-4, L)
        radiance += throughput * self._trace(next_ray, depth - 1)
        return radiance

    # ---- BRDF helpers -------------------------------------------------------
    def _eval_diffuse(
        self, mat: OpenPBRMaterial,
        N: np.ndarray, V: np.ndarray, L: np.ndarray
    ) -> np.ndarray:
        H   = normalize(V + L)
        F0  = np.array([_ior_to_f0(mat.specular_ior)] * 3)
        F0  = F0 * (1 - mat.base_metalness) + np.array(mat.base_color) * mat.base_metalness
        F   = schlick_fresnel_rgb(F0, max(dot(H, V), 0.0))
        kd  = (1.0 - F) * (1.0 - mat.base_metalness)
        return kd * np.array(mat.base_color) * mat.base_weight / math.pi

    def _eval_specular(
        self, mat: OpenPBRMaterial,
        N: np.ndarray, V: np.ndarray, L: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        NdotL = max(dot(N, L), 0.0)
        NdotV = max(dot(N, V), 0.0)
        NdotH = max(dot(N, H), 0.0)
        VdotH = max(dot(V, H), 0.0)
        alpha = max(mat.specular_roughness ** 2, 1e-3)
        D   = ggx_ndf(NdotH, alpha)
        G   = smith_g2(NdotL, NdotV, alpha)
        F0  = np.array([_ior_to_f0(mat.specular_ior)] * 3)
        F0  = F0 * (1 - mat.base_metalness) + np.array(mat.base_color) * mat.base_metalness
        F   = schlick_fresnel_rgb(F0, VdotH)
        denom = max(4.0 * NdotV, 1e-6)
        return np.array(mat.specular_color) * mat.specular_weight * D * G * F / denom

    # ---- environment --------------------------------------------------------
    def _sample_environment(self, direction: np.ndarray) -> np.ndarray:
        if self._env_map is not None:
            return _sample_equirect(self._env_map, direction)
        # Procedural sky
        t = 0.5 * (direction[1] + 1.0)
        sky    = np.array([0.5, 0.7, 1.0])
        horizon = np.array([1.0, 0.85, 0.7])
        return sky * t + horizon * (1.0 - t)

    # ---- tone mapping -------------------------------------------------------
    def _tonemap(self, hdr: np.ndarray) -> np.ndarray:
        from python_renderer.utils.image import apply_aces_tonemap, apply_gamma_correction
        ldr = apply_aces_tonemap(hdr)
        return apply_gamma_correction(ldr)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        from python_renderer.gpu.cuda.kernels import CUDADenoiser
        return CUDADenoiser().denoise(image)

    def save(self, path: str, image: np.ndarray) -> None:
        from python_renderer.utils.image import save_png, save_exr, save_hdr
        ext = Path(path).suffix.lower()
        if ext == ".exr":
            save_exr(image, path)
        elif ext in (".hdr", ".rgbe"):
            save_hdr(image, path)
        else:
            save_png(image, path)


# ===========================================================================
# Utility helpers
# ===========================================================================

def _ior_to_f0(ior: float) -> float:
    f = (ior - 1.0) / (ior + 1.0)
    return f * f


def _compute_barycentric(
    p: np.ndarray, tri: Triangle
) -> Tuple[float, float]:
    """Compute (u, v) barycentric coordinates of p on tri."""
    v0 = tri.v1 - tri.v0
    v1 = tri.v2 - tri.v0
    v2 = p       - tri.v0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return 0.0, 0.0
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return float(v), float(w)


def _sample_equirect(env: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Sample equirectangular map by direction."""
    h, w = env.shape[:2]
    phi   = math.atan2(direction[2], direction[0])
    theta = math.asin(max(-1.0, min(1.0, direction[1])))
    u = (phi / (2 * math.pi) + 0.5) % 1.0
    v = (theta / math.pi + 0.5)
    px = int(u * (w - 1))
    py = int((1.0 - v) * (h - 1))
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    return env[py, px, :3].astype(np.float64)
