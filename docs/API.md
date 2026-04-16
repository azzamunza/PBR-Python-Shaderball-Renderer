# API Reference

## `python_renderer`

```python
import python_renderer
python_renderer.__version__   # "1.0.0"
```

---

## `python_renderer.core.material`

### `OpenPBRMaterial`

Full OpenPBR v1.2 material definition.

```python
from python_renderer.core.material import OpenPBRMaterial

mat = OpenPBRMaterial(
    name="Gold",
    base_color=(1.0, 0.766, 0.336),
    base_metalness=1.0,
    specular_roughness=0.1,
    specular_ior=0.47,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_weight` | float | 1.0 | Base layer weight |
| `base_color` | tuple[float,float,float] | (0.8,0.8,0.8) | Base albedo |
| `base_diffuse_roughness` | float | 0.0 | Oren-Nayar roughness |
| `base_metalness` | float | 0.0 | Metal/dielectric blend |
| `specular_weight` | float | 1.0 | Specular layer weight |
| `specular_color` | tuple | (1,1,1) | Specular tint |
| `specular_roughness` | float | 0.3 | GGX roughness |
| `specular_anisotropy` | float | 0.0 | Anisotropy [0,1] |
| `specular_ior` | float | 1.5 | Index of refraction |
| `transmission_weight` | float | 0.0 | Glass/transmission |
| `subsurface_weight` | float | 0.0 | Subsurface scattering |
| `coat_weight` | float | 0.0 | Clear coat layer |
| `coat_roughness` | float | 0.0 | Coat GGX roughness |
| `coat_ior` | float | 1.5 | Coat IOR |
| `fuzz_weight` | float | 0.0 | Fabric/velvet fuzz |
| `fuzz_roughness` | float | 0.5 | Sheen roughness |
| `emission_weight` | float | 0.0 | Emissive strength |
| `emission_luminance` | float | 1.0 | Emission luminance |
| `emission_color` | tuple | (1,1,1) | Emission colour |
| `thin_film_weight` | float | 0.0 | Iridescence |
| `thin_film_thickness` | float | 500.0 | Film thickness (nm) |
| `geometry_opacity` | float | 1.0 | Alpha / cutout |
| `geometry_thin_walled` | bool | False | Two-sided shading |

#### Methods

```python
mat.to_dict() -> dict
mat.from_dict(d: dict) -> OpenPBRMaterial          # classmethod
mat.save_json(path: str)
mat.load_json(path: str) -> OpenPBRMaterial        # classmethod
mat.save_yaml(path: str)
mat.load_yaml(path: str) -> OpenPBRMaterial        # classmethod
mat.pack_for_gpu() -> np.ndarray                   # 64 x float32
```

---

### `RenderConfig`

```python
from python_renderer.core.material import RenderConfig

cfg = RenderConfig(
    width=1920, height=1080,
    samples=512, max_bounces=8,
    use_gpu=True, denoising=False,
    output_path="render.png",
)
```

---

## `python_renderer.core.geometry`

### `Ray`

```python
from python_renderer.core.geometry import Ray
import numpy as np

ray = Ray(origin=np.array([0,0,3]), direction=np.array([0,0,-1]))
p   = ray.at(2.5)   # point along ray
```

### `Triangle`

```python
tri = Triangle(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2)
t   = tri.intersect(ray)           # Möller–Trumbore, returns float or None
n   = tri.normal_at(u, v)          # interpolated normal
uv  = tri.uv_at(u, v)             # interpolated UV
```

### `Mesh`

```python
from python_renderer.core.geometry import Mesh

mesh = Mesh(vertices, faces, normals, uvs)
mesh.compute_normals()
mesh.compute_tangents()
mesh.tessellate(level=1)
mesh.build_triangles()
bufs = mesh.to_gpu_buffers()       # dict of numpy arrays
```

### `BVHNode`

```python
from python_renderer.core.geometry import BVHNode

bvh = BVHNode.build(triangles)
hit = bvh.intersect(ray)           # returns dict{t, triangle} or None
```

---

## `python_renderer.gpu.pathtracer`

### `GPUPathtracer`

```python
from python_renderer.gpu.pathtracer import GPUPathtracer

pt = GPUPathtracer(config)
pt.setup_gl()
pt.load_scene(mesh, material)
pt.load_environment("studio.hdr")
image = pt.render(samples=512)     # (H, W, 3) float32
pt.save("render.png", image)
```

### `CPUPathtracer`

Drop-in CPU replacement, no GPU required.

```python
from python_renderer.gpu.pathtracer import CPUPathtracer

pt = CPUPathtracer(config)
pt.load_scene(mesh, material)
image = pt.render()
```

---

## `python_renderer.loaders`

### `ModelLoader`

```python
from python_renderer.loaders import ModelLoader

loader = ModelLoader()
mesh   = loader.load("model.glb")
ball   = loader.create_shaderball(lat_segments=64, lon_segments=64)
```

### `TextureLoader`

```python
from python_renderer.loaders import TextureLoader
from python_renderer.core.types import TextureType

loader = TextureLoader()
albedo = loader.load("texture.png", TextureType.ALBEDO)   # linear float32
env    = loader.load_hdr("studio.hdr")
resized = loader.resize(albedo, 512, 512)
linear  = TextureLoader.to_linear(srgb_image)
mips    = loader.generate_mipmaps(albedo)
```

---

## `python_renderer.utils`

```python
from python_renderer.utils import (
    save_png, save_exr, save_hdr, load_image,
    apply_aces_tonemap, apply_gamma_correction,
    normalize, dot, cross, reflect, refract,
    setup_logging, get_logger, ProgressBar,
)
```

---

## `python_renderer.ui`

### CLI

```python
from python_renderer.ui.cli import main
main()   # invoke as Click group
```

### Viewport

```python
from python_renderer.ui.viewport import Viewport

with Viewport(800, 600) as vp:
    vp.show(image)
    while vp.update():
        pass
```
