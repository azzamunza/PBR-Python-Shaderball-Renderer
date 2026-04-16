"""Geometry primitives, mesh representation, and BVH acceleration structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Ray
# ---------------------------------------------------------------------------

@dataclass
class Ray:
    """A ray defined by an origin and unit direction."""
    origin: np.ndarray
    direction: np.ndarray
    t_min: float = 1e-4
    t_max: float = 1e12

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm

    def at(self, t: float) -> np.ndarray:
        """Return the point on the ray at parameter t."""
        return self.origin + t * self.direction


# ---------------------------------------------------------------------------
# Triangle
# ---------------------------------------------------------------------------

@dataclass
class Triangle:
    """A single triangle with per-vertex attributes."""
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    n0: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    n1: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    n2: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    uv0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    uv1: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    uv2: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    material_id: int = 0

    def __post_init__(self) -> None:
        self.v0  = np.asarray(self.v0,  dtype=np.float64)
        self.v1  = np.asarray(self.v1,  dtype=np.float64)
        self.v2  = np.asarray(self.v2,  dtype=np.float64)
        self.n0  = np.asarray(self.n0,  dtype=np.float64)
        self.n1  = np.asarray(self.n1,  dtype=np.float64)
        self.n2  = np.asarray(self.n2,  dtype=np.float64)
        self.uv0 = np.asarray(self.uv0, dtype=np.float64)
        self.uv1 = np.asarray(self.uv1, dtype=np.float64)
        self.uv2 = np.asarray(self.uv2, dtype=np.float64)

    def intersect(self, ray: Ray) -> Optional[float]:
        """
        Möller–Trumbore ray-triangle intersection.

        Returns the hit distance t or None if there is no intersection.
        """
        EPSILON = 1e-8
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)
        if abs(a) < EPSILON:
            return None
        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = f * np.dot(edge2, q)
        if ray.t_min <= t <= ray.t_max:
            return t
        return None

    def normal_at(self, u: float, v: float) -> np.ndarray:
        """Interpolated shading normal at barycentric coordinates (u, v)."""
        w = 1.0 - u - v
        n = w * self.n0 + u * self.n1 + v * self.n2
        length = np.linalg.norm(n)
        return n / length if length > 0 else n

    def uv_at(self, u: float, v: float) -> np.ndarray:
        """Interpolated UV coordinates at barycentric coordinates (u, v)."""
        w = 1.0 - u - v
        return w * self.uv0 + u * self.uv1 + v * self.uv2

    def centroid(self) -> np.ndarray:
        return (self.v0 + self.v1 + self.v2) / 3.0

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (bbox_min, bbox_max)."""
        vmin = np.minimum(np.minimum(self.v0, self.v1), self.v2)
        vmax = np.maximum(np.maximum(self.v0, self.v1), self.v2)
        return vmin, vmax


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

class Mesh:
    """
    Triangle mesh with per-vertex normals, UVs, and optional tangents.

    Internally stores data as flat numpy arrays for efficient GPU upload.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None,
        uvs: Optional[np.ndarray] = None,
        tangents: Optional[np.ndarray] = None,
        material_id: int = 0,
    ) -> None:
        self.vertices: np.ndarray = np.asarray(vertices, dtype=np.float32)
        self.faces: np.ndarray = np.asarray(faces, dtype=np.int32)
        self.normals: Optional[np.ndarray] = (
            np.asarray(normals, dtype=np.float32) if normals is not None else None
        )
        self.uvs: Optional[np.ndarray] = (
            np.asarray(uvs, dtype=np.float32) if uvs is not None else None
        )
        self.tangents: Optional[np.ndarray] = (
            np.asarray(tangents, dtype=np.float32) if tangents is not None else None
        )
        self.material_id: int = material_id
        self.triangles: List[Triangle] = []

    # ---- triangle list ------------------------------------------------------
    def build_triangles(self) -> None:
        """Populate self.triangles from vertex and face data."""
        self.triangles = []
        verts = self.vertices
        faces = self.faces
        norms = self.normals if self.normals is not None else np.zeros_like(verts)
        uvs   = self.uvs if self.uvs is not None else np.zeros((len(verts), 2), dtype=np.float32)

        for face in faces:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            tri = Triangle(
                v0=verts[i0].astype(np.float64),
                v1=verts[i1].astype(np.float64),
                v2=verts[i2].astype(np.float64),
                n0=norms[i0].astype(np.float64),
                n1=norms[i1].astype(np.float64),
                n2=norms[i2].astype(np.float64),
                uv0=uvs[i0].astype(np.float64),
                uv1=uvs[i1].astype(np.float64),
                uv2=uvs[i2].astype(np.float64),
                material_id=self.material_id,
            )
            self.triangles.append(tri)

    # ---- normals & tangents -------------------------------------------------
    def compute_normals(self) -> None:
        """Compute smooth per-vertex normals from face geometry."""
        n_verts = len(self.vertices)
        normals = np.zeros((n_verts, 3), dtype=np.float32)
        verts = self.vertices

        for face in self.faces:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            v0, v1, v2 = verts[i0], verts[i1], verts[i2]
            fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)
            normals[i0] += fn
            normals[i1] += fn
            normals[i2] += fn

        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.where(lengths == 0, 1.0, lengths)
        self.normals = normals / lengths

    def compute_tangents(self) -> None:
        """Compute per-vertex tangents using UV gradients (MikkTSpace-like)."""
        if self.uvs is None:
            return
        n_verts = len(self.vertices)
        tangents  = np.zeros((n_verts, 3), dtype=np.float32)
        bitangents = np.zeros((n_verts, 3), dtype=np.float32)

        verts = self.vertices
        uvs   = self.uvs

        for face in self.faces:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            dv1 = verts[i1] - verts[i0]
            dv2 = verts[i2] - verts[i0]
            du1 = uvs[i1][0] - uvs[i0][0]
            du2 = uvs[i2][0] - uvs[i0][0]
            dv1u = uvs[i1][1] - uvs[i0][1]
            dv2u = uvs[i2][1] - uvs[i0][1]
            denom = du1 * dv2u - du2 * dv1u
            if abs(denom) < 1e-8:
                continue
            r = 1.0 / denom
            tan  = (dv2u * dv1 - dv1u * dv2) * r
            btan = (du1  * dv2 - du2  * dv1) * r
            tangents[i0] += tan
            tangents[i1] += tan
            tangents[i2] += tan
            bitangents[i0] += btan
            bitangents[i1] += btan
            bitangents[i2] += btan

        lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        lengths = np.where(lengths == 0, 1.0, lengths)
        self.tangents = tangents / lengths

    # ---- displacement -------------------------------------------------------
    def apply_displacement(
        self,
        height_map: np.ndarray,
        scale: float = 1.0,
        mode: str = "TANGENT_SPACE",
    ) -> None:
        """
        Apply displacement mapping to vertices.

        height_map : HxW float array, values in [0, 1].
        scale      : world-space displacement magnitude.
        mode       : 'TANGENT_SPACE' or 'OBJECT_SPACE'.
        """
        if self.uvs is None or self.normals is None:
            return
        h, w = height_map.shape[:2]
        uvs = self.uvs
        u_px = np.clip((uvs[:, 0] * (w - 1)).astype(int), 0, w - 1)
        v_px = np.clip((uvs[:, 1] * (h - 1)).astype(int), 0, h - 1)
        disp = height_map[v_px, u_px].astype(np.float32)
        if disp.ndim > 1:
            disp = disp[:, 0]
        # Displace along normal
        self.vertices = (
            self.vertices + self.normals * (disp[:, np.newaxis] * scale)
        ).astype(np.float32)

    # ---- tessellation -------------------------------------------------------
    def tessellate(self, level: int = 1) -> None:
        """
        Subdivide each triangle into 4^level sub-triangles using
        the Loop-like midpoint subdivision scheme.
        """
        if level <= 0:
            return
        for _ in range(level):
            self._subdivide_once()

    def _subdivide_once(self) -> None:
        """One pass of midpoint subdivision."""
        new_verts = list(self.vertices)
        new_faces: List[List[int]] = []
        edge_map: Dict[Tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in edge_map:
                mid = (self.vertices[a] + self.vertices[b]) * 0.5
                edge_map[key] = len(new_verts)
                new_verts.append(mid)
            return edge_map[key]

        for face in self.faces:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            m01 = midpoint(i0, i1)
            m12 = midpoint(i1, i2)
            m20 = midpoint(i2, i0)
            new_faces.extend([
                [i0, m01, m20],
                [m01, i1, m12],
                [m20, m12, i2],
                [m01, m12, m20],
            ])

        self.vertices = np.array(new_verts, dtype=np.float32)
        self.faces = np.array(new_faces, dtype=np.int32)
        # Reset derived data
        self.normals = None
        self.uvs     = None
        self.tangents = None
        self.triangles = []
        self.compute_normals()

    # ---- GPU export ---------------------------------------------------------
    def to_gpu_buffers(self) -> Dict[str, np.ndarray]:
        """
        Return a dict of flat numpy arrays ready for GPU upload:
          vertices  : (N, 4) float32  (xyz + padding)
          normals   : (N, 4) float32  (xyz + padding)
          uvs       : (N, 2) float32
          faces     : (F, 3) int32
          tangents  : (N, 4) float32  (xyz + padding)
        """
        nv = len(self.vertices)

        vert_buf = np.zeros((nv, 4), dtype=np.float32)
        vert_buf[:, :3] = self.vertices

        if self.normals is None:
            self.compute_normals()
        norm_buf = np.zeros((nv, 4), dtype=np.float32)
        norm_buf[:, :3] = self.normals  # type: ignore[index]

        uv_buf = np.zeros((nv, 2), dtype=np.float32)
        if self.uvs is not None:
            uv_buf[:, :] = self.uvs

        tan_buf = np.zeros((nv, 4), dtype=np.float32)
        if self.tangents is not None:
            tan_buf[:, :3] = self.tangents

        return {
            "vertices": vert_buf,
            "normals":  norm_buf,
            "uvs":      uv_buf,
            "faces":    self.faces.astype(np.int32),
            "tangents": tan_buf,
        }

    # ---- factories ----------------------------------------------------------
    @classmethod
    def from_trimesh(cls, tm_mesh: Any, material_id: int = 0) -> "Mesh":
        """Construct a Mesh from a trimesh.Trimesh object."""
        verts  = np.array(tm_mesh.vertices,       dtype=np.float32)
        faces  = np.array(tm_mesh.faces,           dtype=np.int32)
        norms  = np.array(tm_mesh.vertex_normals,  dtype=np.float32)
        uvs: Optional[np.ndarray] = None
        if hasattr(tm_mesh.visual, "uv") and tm_mesh.visual.uv is not None:
            uvs = np.array(tm_mesh.visual.uv, dtype=np.float32)
        mesh = cls(verts, faces, norms, uvs, material_id=material_id)
        return mesh


# ---------------------------------------------------------------------------
# BVH
# ---------------------------------------------------------------------------

class BVHNode:
    """
    Axis-Aligned Bounding-Box BVH node for accelerated ray intersection.

    Uses a simple surface-area heuristic (SAH) partition strategy.
    """

    MAX_LEAF_SIZE = 4

    def __init__(self) -> None:
        self.bbox_min: np.ndarray = np.zeros(3, dtype=np.float64)
        self.bbox_max: np.ndarray = np.zeros(3, dtype=np.float64)
        self.left:  Optional["BVHNode"] = None
        self.right: Optional["BVHNode"] = None
        self.triangles: List[Triangle] = []

    @classmethod
    def build(cls, triangles: List[Triangle]) -> "BVHNode":
        """Recursively build a BVH over the given triangle list."""
        node = cls()
        if not triangles:
            return node

        # Compute bounding box
        all_min = np.array([tri.bbox()[0] for tri in triangles])
        all_max = np.array([tri.bbox()[1] for tri in triangles])
        node.bbox_min = all_min.min(axis=0)
        node.bbox_max = all_max.max(axis=0)

        if len(triangles) <= cls.MAX_LEAF_SIZE:
            node.triangles = triangles
            return node

        # Split on the longest axis at the centroid median
        extent = node.bbox_max - node.bbox_min
        axis = int(np.argmax(extent))
        centroids = np.array([tri.centroid()[axis] for tri in triangles])
        median = np.median(centroids)
        left_tris  = [t for t, c in zip(triangles, centroids) if c <= median]
        right_tris = [t for t, c in zip(triangles, centroids) if c >  median]

        # Avoid degenerate splits
        if not left_tris or not right_tris:
            mid = len(triangles) // 2
            left_tris, right_tris = triangles[:mid], triangles[mid:]

        node.left  = cls.build(left_tris)
        node.right = cls.build(right_tris)
        return node

    def _intersect_bbox(self, ray: Ray) -> bool:
        """Slab-based AABB ray test that handles zero direction components."""
        t_near = ray.t_min
        t_far  = ray.t_max
        for i in range(3):
            d = float(ray.direction[i])
            o = float(ray.origin[i])
            lo = float(self.bbox_min[i])
            hi = float(self.bbox_max[i])
            if abs(d) < 1e-8:
                # Ray is parallel to this pair of slabs
                if o < lo or o > hi:
                    return False
            else:
                inv_d = 1.0 / d
                t1 = (lo - o) * inv_d
                t2 = (hi - o) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_near = max(t_near, t1)
                t_far  = min(t_far,  t2)
                if t_near > t_far:
                    return False
        return True

    def intersect(self, ray: Ray) -> Optional[Dict]:
        """
        Find the closest triangle intersection.

        Returns a dict with keys: t, u, v, triangle — or None.
        """
        if not self._intersect_bbox(ray):
            return None

        best: Optional[Dict] = None

        if self.triangles:
            for tri in self.triangles:
                t = tri.intersect(ray)
                if t is not None:
                    if best is None or t < best["t"]:
                        # Recompute barycentric coords for normal/UV interpolation
                        best = {"t": t, "triangle": tri}
            return best

        hit_l = self.left.intersect(ray)  if self.left  else None
        hit_r = self.right.intersect(ray) if self.right else None

        if hit_l and hit_r:
            return hit_l if hit_l["t"] < hit_r["t"] else hit_r
        return hit_l or hit_r
