"""
Math utilities for 3-D vector operations used throughout the renderer.

All functions operate on NumPy arrays and are optimised for
array broadcasting where possible.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ===========================================================================
# Basic vector operations
# ===========================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of *v*."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two 1-D vectors."""
    return float(np.dot(a, b))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-D vectors."""
    return np.cross(a, b)


def reflect(i: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Reflect vector *i* about normal *n*."""
    return i - 2.0 * dot(i, n) * n


def refract(i: np.ndarray, n: np.ndarray, eta: float) -> np.ndarray:
    """
    Refract vector *i* through surface with normal *n* and IOR ratio *eta*.

    Returns the zero vector for total internal reflection.
    """
    cos_i = -dot(n, i)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t >= 1.0:
        return np.zeros(3)
    cos_t = math.sqrt(1.0 - sin2_t)
    return eta * i + (eta * cos_i - cos_t) * n


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between *a* and *b*."""
    return a + t * (b - a)


def clamp(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clamp each element of *x* to [min_val, max_val]."""
    return np.clip(x, min_val, max_val)


# ===========================================================================
# Coordinate conversions
# ===========================================================================

def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    """
    Convert spherical (theta, phi) to Cartesian.

    theta : polar angle from +Y axis [0, pi]
    phi   : azimuthal angle in XZ plane [0, 2pi]
    """
    sin_t = math.sin(theta)
    return np.array([sin_t * math.cos(phi), math.cos(theta), sin_t * math.sin(phi)])


def cartesian_to_spherical(v: np.ndarray) -> Tuple[float, float]:
    """Return (theta, phi) for unit vector *v*."""
    v = normalize(v)
    theta = math.acos(max(-1.0, min(1.0, float(v[1]))))
    phi   = math.atan2(float(v[2]), float(v[0])) % (2.0 * math.pi)
    return theta, phi


# ===========================================================================
# Matrix helpers
# ===========================================================================

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Build a 3×3 rotation matrix for *angle* radians around *axis*.

    Uses Rodrigues' rotation formula.
    """
    axis = normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ], dtype=np.float64)


def look_at_matrix(
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Build a 4×4 view matrix (camera-to-world)."""
    f = normalize(eye - center)           # forward (points toward camera)
    r = normalize(cross(up, f))           # right
    u = cross(f, r)                       # up
    M = np.eye(4, dtype=np.float64)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = f
    M[0, 3]  = -dot(r, eye)
    M[1, 3]  = -dot(u, eye)
    M[2, 3]  = -dot(f, eye)
    return np.linalg.inv(M)               # world-to-camera → camera-to-world


def perspective_matrix(
    fov: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    """
    Build a 4×4 OpenGL-style perspective projection matrix.

    fov    : vertical field-of-view in radians
    aspect : width / height
    """
    f   = 1.0 / math.tan(fov * 0.5)
    nf  = near - far
    return np.array([
        [f / aspect, 0, 0,                          0             ],
        [0,          f, 0,                          0             ],
        [0,          0, (far + near) / nf,          2*far*near/nf ],
        [0,          0, -1,                         0             ],
    ], dtype=np.float64)


# ===========================================================================
# BRDF helpers (CPU implementations mirroring GLSL)
# ===========================================================================

def cosine_hemisphere_sample(
    N: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    Sample a direction in the hemisphere around *N* with cosine weighting.

    Returns (direction, pdf).
    """
    r1, r2 = rng.random(), rng.random()
    phi        = 2.0 * math.pi * r1
    sin_theta  = math.sqrt(r2)
    cos_theta  = math.sqrt(1.0 - r2)
    local = np.array([
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta,
    ])
    direction = normalize(_to_world(local, N))
    pdf       = max(dot(N, direction), 1e-8) / math.pi
    return direction, pdf


def ggx_sample(
    V: np.ndarray,
    N: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a GGX microfacet half-vector H and compute reflection direction L.

    Returns (H, L) both in world space.
    """
    r1, r2 = rng.random(), rng.random()
    phi = 2.0 * math.pi * r1
    cos_theta = math.sqrt((1.0 - r2) / (1.0 + (alpha * alpha - 1.0) * r2))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    local_H = np.array([
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta,
    ])
    H = normalize(_to_world(local_H, N))
    L = reflect(-V, H)
    return H, L


def ggx_ndf(cos_theta_h: float, alpha: float) -> float:
    """GGX Normal Distribution Function value."""
    a2    = alpha * alpha
    denom = cos_theta_h * cos_theta_h * (a2 - 1.0) + 1.0
    return a2 / (math.pi * denom * denom + 1e-12)


def _smith_lambda(cos_theta: float, alpha: float) -> float:
    a2   = alpha * alpha
    c2   = cos_theta * cos_theta
    tan2 = max((1.0 - c2) / max(c2, 1e-12), 0.0)
    return 0.5 * (-1.0 + math.sqrt(1.0 + a2 * tan2))


def smith_g2(cos_l: float, cos_v: float, alpha: float) -> float:
    """Smith height-correlated G2 masking-shadowing function."""
    return 1.0 / (1.0 + _smith_lambda(cos_l, alpha) + _smith_lambda(cos_v, alpha))


def schlick_fresnel(F0: float, cos_theta: float) -> float:
    """Schlick Fresnel for scalar F0."""
    x = 1.0 - cos_theta
    return F0 + (1.0 - F0) * x ** 5


def schlick_fresnel_rgb(F0: np.ndarray, cos_theta: float) -> np.ndarray:
    """Schlick Fresnel for RGB F0."""
    x = 1.0 - cos_theta
    return F0 + (1.0 - F0) * x ** 5


# ===========================================================================
# Internal helpers
# ===========================================================================

def _to_world(local: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Transform a local vector into world space aligned with normal *N*."""
    up = np.array([0.0, 1.0, 0.0]) if abs(N[1]) < 0.9999 else np.array([1.0, 0.0, 0.0])
    T  = normalize(cross(up, N))
    B  = cross(N, T)
    return local[0] * T + local[1] * B + local[2] * N
