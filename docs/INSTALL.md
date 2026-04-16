# Installation Guide

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| numpy  | ≥ 1.21 |
| Pillow | ≥ 9.0  |
| PyYAML | ≥ 6.0  |
| click  | ≥ 8.0  |
| tqdm   | ≥ 4.60 |
| scipy  | ≥ 1.7  |
| imageio | ≥ 2.9 |
| trimesh | ≥ 3.15 |

## Optional Requirements

| Package | Purpose |
|---|---|
| PyOpenGL ≥ 3.1.5 | GPU path tracing via OpenGL compute shaders |
| glfw ≥ 2.5 | Headless OpenGL context creation |
| cupy-cuda11x ≥ 10.0 | CUDA GPU acceleration (requires NVIDIA GPU + CUDA) |
| pygame ≥ 2.0 | Real-time viewport |
| OpenEXR ≥ 1.3 | Native OpenEXR I/O |

## Installation

### From PyPI (when published)

```bash
pip install pbr-python-renderer
```

### From Source

```bash
git clone https://github.com/azzamunza/PBR-Python-Shaderball-Renderer.git
cd PBR-Python-Shaderball-Renderer
pip install -e .
```

### With All Optional Dependencies

```bash
pip install -e ".[all]"
```

### With GPU Support

```bash
pip install -e ".[gpu]"
```

### With CUDA Support (NVIDIA GPU Required)

```bash
# Install CuPy matching your CUDA version, e.g. CUDA 11.x:
pip install cupy-cuda11x
pip install -e ".[gpu]"
```

## Verifying Installation

```bash
python -c "import python_renderer; print(python_renderer.__version__)"
pbr-render --help
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

All tests pass without a GPU (CPU path-tracer fallback).
