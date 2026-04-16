"""Unit tests for geometry primitives."""

import math

import numpy as np
import pytest

from python_renderer.core.geometry import BVHNode, Mesh, Ray, Triangle


# ===========================================================================
# Ray tests
# ===========================================================================

def test_ray_creation():
    ray = Ray(origin=[0, 0, 0], direction=[1, 0, 0])
    assert ray.origin.shape   == (3,)
    assert ray.direction.shape == (3,)
    assert math.isclose(np.linalg.norm(ray.direction), 1.0, abs_tol=1e-8)


def test_ray_at():
    ray = Ray(origin=[0, 0, 0], direction=[0, 1, 0])
    p   = ray.at(3.0)
    assert math.isclose(p[1], 3.0, abs_tol=1e-8)


# ===========================================================================
# Triangle tests
# ===========================================================================

@pytest.fixture
def unit_triangle():
    return Triangle(
        v0=np.array([0.0, 0.0, 0.0]),
        v1=np.array([1.0, 0.0, 0.0]),
        v2=np.array([0.0, 1.0, 0.0]),
        n0=np.array([0.0, 0.0, 1.0]),
        n1=np.array([0.0, 0.0, 1.0]),
        n2=np.array([0.0, 0.0, 1.0]),
    )


def test_triangle_intersect(unit_triangle):
    ray = Ray(origin=[0.25, 0.25, -1.0], direction=[0.0, 0.0, 1.0])
    t   = unit_triangle.intersect(ray)
    assert t is not None
    assert math.isclose(t, 1.0, abs_tol=1e-6)


def test_triangle_no_intersect_behind(unit_triangle):
    ray = Ray(origin=[0.25, 0.25, 1.0], direction=[0.0, 0.0, 1.0])
    t   = unit_triangle.intersect(ray)
    assert t is None


def test_triangle_no_intersect_miss(unit_triangle):
    # Ray that clearly misses the triangle
    ray = Ray(origin=[2.0, 2.0, -1.0], direction=[0.0, 0.0, 1.0])
    t   = unit_triangle.intersect(ray)
    assert t is None


def test_triangle_normal_at(unit_triangle):
    n = unit_triangle.normal_at(0.0, 0.0)
    assert math.isclose(n[2], 1.0, abs_tol=1e-6)


def test_triangle_uv_at():
    tri = Triangle(
        v0=[0, 0, 0], v1=[1, 0, 0], v2=[0, 1, 0],
        uv0=[0, 0], uv1=[1, 0], uv2=[0, 1],
    )
    uv = tri.uv_at(0.5, 0.0)
    assert math.isclose(uv[0], 0.5, abs_tol=1e-6)


# ===========================================================================
# Mesh tests
# ===========================================================================

@pytest.fixture
def simple_mesh():
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return Mesh(verts, faces)


def test_mesh_from_vertices(simple_mesh):
    assert len(simple_mesh.vertices) == 4
    assert len(simple_mesh.faces)    == 4


def test_mesh_compute_normals(simple_mesh):
    simple_mesh.compute_normals()
    assert simple_mesh.normals is not None
    assert simple_mesh.normals.shape == (4, 3)
    # All normals should be unit length
    lengths = np.linalg.norm(simple_mesh.normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


def test_mesh_build_triangles(simple_mesh):
    simple_mesh.compute_normals()
    simple_mesh.build_triangles()
    assert len(simple_mesh.triangles) == 4
    for tri in simple_mesh.triangles:
        assert isinstance(tri, Triangle)


def test_mesh_tessellate(simple_mesh):
    original_faces = len(simple_mesh.faces)
    simple_mesh.tessellate(level=1)
    assert len(simple_mesh.faces) == original_faces * 4


def test_mesh_to_gpu_buffers(simple_mesh):
    bufs = simple_mesh.to_gpu_buffers()
    assert "vertices" in bufs
    assert "normals"  in bufs
    assert "faces"    in bufs
    assert bufs["vertices"].dtype == np.float32
    assert bufs["faces"].dtype    == np.int32


# ===========================================================================
# BVH tests
# ===========================================================================

def _make_triangles(n: int):
    """Make n flat triangles along the Z=0 plane at positions (i, 0)."""
    tris = []
    for i in range(n):
        tris.append(Triangle(
            v0=[i, 0, 0], v1=[i + 1, 0, 0], v2=[i, 1, 0],
            n0=[0, 0, 1], n1=[0, 0, 1], n2=[0, 0, 1],
        ))
    return tris


def test_bvh_build():
    tris = _make_triangles(8)
    bvh  = BVHNode.build(tris)
    assert bvh is not None
    # bbox should encompass all triangles
    assert bvh.bbox_min[0] < 0.5
    assert bvh.bbox_max[0] > 7.5


def test_bvh_intersect():
    tris = _make_triangles(4)
    bvh  = BVHNode.build(tris)
    # Ray hitting triangle at i=2 (x in [2,3])
    ray  = Ray(origin=[2.2, 0.2, -1.0], direction=[0.0, 0.0, 1.0])
    hit  = bvh.intersect(ray)
    assert hit is not None
    assert math.isclose(hit["t"], 1.0, abs_tol=1e-4)


def test_bvh_no_intersect():
    tris = _make_triangles(4)
    bvh  = BVHNode.build(tris)
    ray  = Ray(origin=[10.0, 0.5, -1.0], direction=[0.0, 0.0, 1.0])
    hit  = bvh.intersect(ray)
    assert hit is None
