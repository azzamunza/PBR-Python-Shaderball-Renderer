"""3-D model loader supporting glTF, OBJ, STL, FBX, and procedural meshes."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from python_renderer.core.geometry import Mesh

log = logging.getLogger(__name__)

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False
    log.debug("trimesh not available – only procedural mesh creation supported.")


class ModelLoader:
    """Load 3-D models from disk or generate procedural geometry."""

    SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl", ".fbx", ".ply"}

    def load(self, path: str) -> Mesh:
        """
        Load a mesh from *path*.

        Supported formats: glTF (.glb/.gltf), OBJ, STL, FBX, PLY.
        """
        p   = Path(path)
        ext = p.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        if not _TRIMESH_AVAILABLE:
            raise RuntimeError(
                "trimesh is required to load model files. "
                "Install it with:  pip install trimesh"
            )
        log.info("Loading model: %s", path)
        tm = trimesh.load(str(p), force="mesh", process=True)
        mesh = Mesh.from_trimesh(tm)
        self._center_and_scale(mesh)
        log.info("Loaded %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        return mesh

    # ---- extension-specific loaders (thin wrappers) -------------------------
    def _load_gltf(self, path: str) -> Mesh:
        return self.load(path)

    def _load_obj(self, path: str) -> Mesh:
        return self.load(path)

    def _load_stl(self, path: str) -> Mesh:
        return self.load(path)

    def _load_fbx(self, path: str) -> Mesh:
        return self.load(path)

    # ---- procedural geometry ------------------------------------------------
    def create_shaderball(
        self,
        radius: float = 1.0,
        lat_segments: int = 64,
        lon_segments: int = 64,
    ) -> Mesh:
        """
        Create a UV sphere suitable for material preview (shaderball).

        Parameters
        ----------
        radius       : sphere radius
        lat_segments : number of latitude divisions
        lon_segments : number of longitude divisions
        """
        vertices: list  = []
        normals:  list  = []
        uvs:      list  = []
        faces:    list  = []

        for lat in range(lat_segments + 1):
            theta = math.pi * lat / lat_segments
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            for lon in range(lon_segments + 1):
                phi   = 2.0 * math.pi * lon / lon_segments
                sin_p = math.sin(phi)
                cos_p = math.cos(phi)
                x = sin_t * cos_p
                y = cos_t
                z = sin_t * sin_p
                vertices.append([x * radius, y * radius, z * radius])
                normals.append([x, y, z])
                uvs.append([lon / lon_segments, 1.0 - lat / lat_segments])

        for lat in range(lat_segments):
            for lon in range(lon_segments):
                i0 = lat * (lon_segments + 1) + lon
                i1 = i0 + 1
                i2 = i0 + (lon_segments + 1)
                i3 = i2 + 1
                faces.append([i0, i2, i1])
                faces.append([i1, i2, i3])

        mesh = Mesh(
            vertices=np.array(vertices, dtype=np.float32),
            faces=np.array(faces, dtype=np.int32),
            normals=np.array(normals, dtype=np.float32),
            uvs=np.array(uvs, dtype=np.float32),
        )
        log.info(
            "Shaderball created: %d vertices, %d faces",
            len(mesh.vertices), len(mesh.faces),
        )
        return mesh

    # ---- UV mapping helpers -------------------------------------------------
    @staticmethod
    def _generate_spherical_uvs(mesh: Mesh) -> None:
        """Overwrite mesh UVs with a spherical projection."""
        verts = mesh.vertices
        uvs   = np.zeros((len(verts), 2), dtype=np.float32)
        for i, v in enumerate(verts):
            r = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            if r < 1e-8:
                continue
            phi   = math.atan2(v[2], v[0])
            theta = math.acos(max(-1.0, min(1.0, v[1] / r)))
            uvs[i, 0] = (phi / (2.0 * math.pi)) % 1.0
            uvs[i, 1] = 1.0 - theta / math.pi
        mesh.uvs = uvs

    @staticmethod
    def _center_and_scale(mesh: Mesh) -> None:
        """Translate mesh to origin and scale to unit cube."""
        vmin = mesh.vertices.min(axis=0)
        vmax = mesh.vertices.max(axis=0)
        center = (vmin + vmax) * 0.5
        scale  = np.max(vmax - vmin)
        if scale < 1e-8:
            return
        mesh.vertices = (mesh.vertices - center) / scale
