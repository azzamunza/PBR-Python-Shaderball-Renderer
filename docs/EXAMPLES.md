# Examples

## 1. Simple Gold Shaderball

```python
from python_renderer import OpenPBRMaterial, RenderConfig, GPUPathtracer
from python_renderer.loaders import ModelLoader

material = OpenPBRMaterial(
    name="Gold",
    base_color=(1.0, 0.766, 0.336),
    base_metalness=1.0,
    specular_roughness=0.1,
    specular_ior=0.47,
)
config = RenderConfig(width=1920, height=1080, samples=512, use_gpu=False)
mesh   = ModelLoader().create_shaderball()

pt = GPUPathtracer(config)
pt.setup_gl()
pt.load_scene(mesh, material)
image = pt.render()
pt.save("gold.png", image)
```

---

## 2. Clear-Coated Red Plastic

```python
material = OpenPBRMaterial(
    base_color=(0.8, 0.05, 0.05),
    base_metalness=0.0,
    specular_roughness=0.2,
    coat_weight=1.0,
    coat_roughness=0.05,
    coat_ior=1.6,
)
```

---

## 3. Blue Fabric with Sheen

```python
material = OpenPBRMaterial(
    base_color=(0.1, 0.2, 0.7),
    base_diffuse_roughness=0.8,
    specular_weight=0.1,
    specular_roughness=0.9,
    fuzz_weight=1.0,
    fuzz_color=(0.15, 0.25, 0.75),
    fuzz_roughness=0.7,
)
```

---

## 4. Transparent Glass

```python
material = OpenPBRMaterial(
    base_color=(1.0, 1.0, 1.0),
    base_weight=0.0,
    specular_ior=1.52,
    transmission_weight=1.0,
    transmission_color=(0.9, 0.95, 1.0),
    geometry_thin_walled=False,
)
```

---

## 5. Emissive Material (Area Light)

```python
material = OpenPBRMaterial(
    base_weight=0.0,
    emission_weight=1.0,
    emission_color=(1.0, 0.9, 0.7),
    emission_luminance=10.0,
)
```

---

## 6. Loading from JSON

```python
material = OpenPBRMaterial.load_json("examples/materials/gold.json")
```

---

## 7. With HDR Environment Lighting

```python
config = RenderConfig(
    width=1920, height=1080,
    samples=1024,
    hdri_path="studio.hdr",
)
pt = GPUPathtracer(config)
pt.setup_gl()
pt.load_scene(mesh, material)
pt.load_environment("studio.hdr")
image = pt.render()
```

---

## 8. Displacement Mapping

```python
import numpy as np
from python_renderer.loaders import ModelLoader, TextureLoader

mesh     = ModelLoader().create_shaderball(lat_segments=64, lon_segments=64)
height   = TextureLoader().load("displacement.png")[:, :, 0]
mesh.tessellate(level=2)
mesh.compute_normals()
mesh.apply_displacement(height, scale=0.1, mode="TANGENT_SPACE")
```

---

## 9. Custom Camera Config

```python
config = RenderConfig(
    width=2560, height=1440,
    samples=2048,
    camera_fov=35.0,       # telephoto-ish
    camera_distance=4.0,
    max_bounces=12,
    denoising=True,
)
```

---

## 10. CLI Quick-Start

```bash
# Download and render
git clone https://github.com/azzamunza/PBR-Python-Shaderball-Renderer.git
cd PBR-Python-Shaderball-Renderer
pip install -e .
pbr-render render \
    -m python_renderer/examples/materials/gold.json \
    -o gold_render.png \
    --samples 256 \
    --no-gpu
```
