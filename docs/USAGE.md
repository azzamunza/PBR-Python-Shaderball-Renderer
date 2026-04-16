# Usage Guide

## Command-Line Interface

The `pbr-render` command is the primary entry point.

```
pbr-render [OPTIONS] COMMAND [ARGS]...
```

### Global Options

| Option | Description |
|---|---|
| `--verbose` / `-v` | Enable DEBUG logging |

---

## Commands

### `render`

Render a material preview image.

```bash
pbr-render render [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--material` / `-m` PATH | — | Material JSON or YAML file |
| `--model` PATH | shaderball | Custom 3-D model file |
| `--hdri` PATH | — | HDR environment map |
| `--output` / `-o` PATH | `render.png` | Output image path |
| `--width` INT | `1920` | Image width |
| `--height` INT | `1080` | Image height |
| `--samples` / `-s` INT | `512` | Samples per pixel |
| `--bounces` INT | `8` | Max ray bounce depth |
| `--config` / `-c` PATH | — | YAML configuration file |
| `--gpu` / `--no-gpu` | GPU | Enable/disable GPU |
| `--denoise` | off | Apply denoising |
| `--displacement-scale` FLOAT | `1.0` | Displacement strength |

**Examples:**

```bash
# Render gold material at 1080p, 512 spp
pbr-render render -m materials/gold.json -o gold.png

# Use a custom model and environment
pbr-render render -m materials/plastic.json \
    --model model.glb --hdri studio.hdr -o result.png

# Render using a config file
pbr-render render -m gold.json -c configs/production.yaml

# CPU-only render
pbr-render render -m gold.json --no-gpu --samples 128

# Render to EXR
pbr-render render -m gold.json -o render.exr
```

---

### `preview`

Quick 64-sample, 512×512 preview render.

```bash
pbr-render preview -m gold.json -o preview.png
```

---

### `info`

Display the contents of a material or configuration file.

```bash
pbr-render info gold.json
pbr-render info configs/production.yaml
```

---

### `convert-material`

Convert a material file between JSON and YAML formats.

```bash
pbr-render convert-material gold.json gold.yaml
pbr-render convert-material plastic.yaml plastic.json
```

---

### `benchmark`

Benchmark rendering performance.

```bash
pbr-render benchmark --width 1920 --height 1080 --samples 64
```

Output:
```
Benchmark: 1920x1080, 64 spp
  Time   : 12.34s
  MRays/s: 10.72
```

---

## Configuration Files (YAML)

```yaml
# configs/production.yaml
width: 1920
height: 1080
samples: 2048
max_bounces: 12
use_gpu: true
denoising: true
output_path: production.png
output_format: png
firefly_clamp: 10.0
displacement_scale: 1.0
tessellation_level: 4
camera_fov: 45.0
camera_distance: 3.0
```

---

## Material Files

### JSON format

```json
{
  "name": "Gold",
  "base":     {"weight": 1.0, "color": [1.0, 0.766, 0.336], "metalness": 1.0},
  "specular": {"weight": 1.0, "roughness": 0.1, "ior": 0.47},
  "emission": {"weight": 0.0, "luminance": 1.0, "color": [1.0, 1.0, 1.0]},
  "geometry": {"opacity": 1.0, "thin_walled": false}
}
```

### YAML format

```yaml
name: Gold
base:
  weight: 1.0
  color: [1.0, 0.766, 0.336]
  metalness: 1.0
specular:
  roughness: 0.1
  ior: 0.47
```
