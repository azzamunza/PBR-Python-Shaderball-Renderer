# PBR Python Shaderball Renderer

A **GPU-accelerated, standalone Python path tracer** that renders physically-based
materials defined using the **OpenPBR v1.2** specification.

> **Inspired by** [OpenPBR-viewer](https://github.com/portsmouth/OpenPBR-viewer) by
> [portsmouth](https://github.com/portsmouth) – a WebGL-based interactive viewer for the
> OpenPBR surface shading model that served as the conceptual foundation for this project.

---

## Features

- Full **OpenPBR v1.2** material model
  - Base / Specular / Transmission / Subsurface / Coat / Fuzz / Emission / Thin-Film layers
- **GPU path tracing** via OpenGL 4.3 compute shaders
- **CPU fallback** using pure NumPy – no GPU required for basic use
- Optional **CUDA acceleration** (CuPy) for denoising and post-processing
- **BVH acceleration** structure for fast CPU ray traversal
- Supports **.glb, .gltf, .obj, .stl, .fbx, .ply** mesh formats via trimesh
- HDR/EXR environment maps (equirectangular)
- ACES, Reinhard, and filmic tone mapping
- Bilateral filter denoising
- Procedural **shaderball** geometry (UV sphere) for material preview
- Displacement mapping (tangent-space and object-space)
- Click-based **CLI** (`pbr-render`)
- Optional real-time **viewport** (pygame)
- Full **pytest** test suite (CPU only, no GPU required for tests)

---

## Quick Start

```bash
pip install pbr-python-renderer
pbr-render render --material examples/materials/gold.json --output gold.png
```

---

## Installation

See [INSTALL.md](INSTALL.md) for full instructions.

```bash
git clone https://github.com/azzamunza/PBR-Python-Shaderball-Renderer.git
cd PBR-Python-Shaderball-Renderer
pip install -e ".[all]"
```

---

## Usage

```bash
# Render with default shaderball
pbr-render render -m gold.json -o render.png --samples 512

# Quick 64-sample preview
pbr-render preview -m plastic.json

# Show material info
pbr-render info gold.json

# Convert between JSON and YAML
pbr-render convert-material gold.json gold.yaml

# Benchmark GPU performance
pbr-render benchmark --width 1920 --height 1080 --samples 64
```

See [USAGE.md](USAGE.md) for full documentation.

---

## Python API

```python
from python_renderer import OpenPBRMaterial, RenderConfig, GPUPathtracer
from python_renderer.loaders import ModelLoader

material = OpenPBRMaterial(
    base_color=(1.0, 0.766, 0.336),
    base_metalness=1.0,
    specular_roughness=0.1,
)
config = RenderConfig(width=1920, height=1080, samples=512)
mesh   = ModelLoader().create_shaderball()

pt = GPUPathtracer(config)
pt.setup_gl()
pt.load_scene(mesh, material)
image = pt.render()
pt.save("render.png", image)
```

---

## Directory Structure

```
python_renderer/
├── core/          # Material, geometry, and type definitions
├── gpu/           # OpenGL compute shaders and CPU path tracer
│   ├── shaders/   # GLSL compute shaders
│   └── cuda/      # CuPy CUDA kernels (optional)
├── loaders/       # Model and texture loaders
├── utils/         # Image I/O, math, logging
└── ui/            # CLI and viewport
tests/             # pytest test suite
docs/              # Documentation
```

---

## Screenshot

*(Rendered with 2048 samples, ACES tone mapping)*

![shaderball preview](https://placeholder.com/shaderball.png)

---

## Acknowledgements

This project was inspired by and is based on the excellent work of
**[portsmouth](https://github.com/portsmouth)** on the
**[OpenPBR-viewer](https://github.com/portsmouth/OpenPBR-viewer)** – a WebGL-based
interactive path tracer that implements the full
[OpenPBR v1.2](https://academysoftwarefoundation.github.io/OpenPBR/) surface shading model.

Key contributions from the original project that informed this Python implementation:

| Area | Reference in OpenPBR-viewer |
|---|---|
| OpenPBR v1.2 material layer stack | `src/materials/`, shader GLSL sources |
| Cook-Torrance BRDF with GGX NDF | `src/shaders/brdf.glsl` |
| Importance-sampled IBL (equirectangular HDR) | `src/shaders/env_map.glsl` |
| Progressive path tracing with MIS | `src/pathtracer/` |
| Shaderball (Standard Shader Ball asset) | `assets/shaderball/` |

A sincere **thank you** to portsmouth and all contributors to OpenPBR-viewer for making
such a clean and well-documented reference implementation available under an open-source
license.

---

## License

MIT License – see [LICENSE](../LICENSE).
