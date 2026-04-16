"""
Click-based command-line interface for the PBR Python Renderer.

Entry point:  pbr-render   (configured in setup.py)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click

from python_renderer.utils.logging import setup_logging, get_logger

log = get_logger(__name__)


@click.group()
@click.option(
    "--verbose", "-v", is_flag=True, default=False,
    help="Enable verbose (DEBUG) logging.",
)
def main(verbose: bool) -> None:
    """PBR Python Shaderball Renderer – GPU-accelerated OpenPBR v1.2 path tracer."""
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level)


# ===========================================================================
# render
# ===========================================================================

@main.command()
@click.option("--material", "-m", "material_path",
              type=click.Path(exists=True),
              help="Material JSON or YAML file.")
@click.option("--model", "model_path",
              type=click.Path(exists=True), default=None,
              help="3-D model file (.glb, .obj, .stl, .fbx).  Uses a shaderball by default.")
@click.option("--hdri", "hdri_path",
              type=click.Path(exists=True), default=None,
              help="HDR equirectangular environment map.")
@click.option("--output", "-o", "output_path", default="render.png",
              help="Output image file path.")
@click.option("--width",  default=1920, show_default=True, help="Image width in pixels.")
@click.option("--height", default=1080, show_default=True, help="Image height in pixels.")
@click.option("--samples", "-s", default=512, show_default=True,
              help="Number of path-tracing samples per pixel.")
@click.option("--config", "-c", "config_path",
              type=click.Path(exists=True), default=None,
              help="YAML render configuration file.")
@click.option("--gpu/--no-gpu", "use_gpu", default=True, show_default=True,
              help="Enable or disable GPU acceleration.")
@click.option("--denoise", "denoising", is_flag=True, default=False,
              help="Apply denoising to the final image.")
@click.option("--bounces", "max_bounces", default=8, show_default=True,
              help="Maximum ray bounce depth.")
@click.option("--displacement-scale", default=1.0, show_default=True,
              help="Displacement map scale factor.")
def render(
    material_path: Optional[str],
    model_path: Optional[str],
    hdri_path: Optional[str],
    output_path: str,
    width: int,
    height: int,
    samples: int,
    config_path: Optional[str],
    use_gpu: bool,
    denoising: bool,
    max_bounces: int,
    displacement_scale: float,
) -> None:
    """Render a material preview (shaderball) image."""
    from python_renderer.core.material import OpenPBRMaterial, RenderConfig
    from python_renderer.core.types import OutputFormat
    from python_renderer.loaders.model_loader import ModelLoader
    from python_renderer.gpu.pathtracer import GPUPathtracer, CPUPathtracer

    # ---- load config --------------------------------------------------------
    if config_path:
        config = RenderConfig.load_yaml(config_path)
        log.info("Loaded config: %s", config_path)
    else:
        ext = Path(output_path).suffix.lower()
        fmt = {".exr": OutputFormat.EXR, ".hdr": OutputFormat.HDR}.get(ext, OutputFormat.PNG)
        config = RenderConfig(
            width=width, height=height, samples=samples,
            max_bounces=max_bounces, hdri_path=hdri_path,
            output_path=output_path, output_format=fmt,
            use_gpu=use_gpu, denoising=denoising,
            displacement_scale=displacement_scale,
        )

    # ---- load material ------------------------------------------------------
    if material_path:
        mat_ext = Path(material_path).suffix.lower()
        if mat_ext == ".json":
            material = OpenPBRMaterial.load_json(material_path)
        elif mat_ext in (".yaml", ".yml"):
            material = OpenPBRMaterial.load_yaml(material_path)
        else:
            click.echo(f"Unsupported material format: {mat_ext}", err=True)
            sys.exit(1)
        log.info("Loaded material: %s", material_path)
    else:
        material = OpenPBRMaterial()
        log.info("Using default material (white plastic).")

    # ---- load / create mesh -------------------------------------------------
    loader = ModelLoader()
    if model_path:
        mesh = loader.load(model_path)
    else:
        mesh = loader.create_shaderball()

    # ---- render -------------------------------------------------------------
    if config.use_gpu:
        pt = GPUPathtracer(config)
        pt.setup_gl()
    else:
        from python_renderer.gpu.pathtracer import CPUPathtracer
        pt = CPUPathtracer(config)  # type: ignore[assignment]

    pt.load_scene(mesh, material)
    if hdri_path or config.hdri_path:
        pt.load_environment(hdri_path or config.hdri_path)  # type: ignore[arg-type]

    t0 = time.time()
    image = pt.render()
    elapsed = time.time() - t0
    click.echo(f"Render complete in {elapsed:.1f}s – saving to {config.output_path}")
    pt.save(config.output_path, image)


# ===========================================================================
# preview
# ===========================================================================

@main.command()
@click.option("--material", "-m", "material_path",
              type=click.Path(exists=True), default=None)
@click.option("--output", "-o", "output_path", default="preview.png")
def preview(material_path: Optional[str], output_path: str) -> None:
    """Quick low-quality preview render (64 samples, 512×512)."""
    from python_renderer.core.material import OpenPBRMaterial, RenderConfig
    from python_renderer.loaders.model_loader import ModelLoader
    from python_renderer.gpu.pathtracer import CPUPathtracer

    material = (
        OpenPBRMaterial.load_json(material_path)
        if material_path and material_path.endswith(".json")
        else OpenPBRMaterial.load_yaml(material_path)
        if material_path
        else OpenPBRMaterial()
    )
    config = RenderConfig(width=512, height=512, samples=64,
                          max_bounces=4, output_path=output_path)
    mesh = ModelLoader().create_shaderball()
    pt   = CPUPathtracer(config)
    pt.load_scene(mesh, material)
    image = pt.render()
    pt.save(output_path, image)
    click.echo(f"Preview saved to {output_path}")


# ===========================================================================
# info
# ===========================================================================

@main.command()
@click.argument("path", type=click.Path(exists=True))
def info(path: str) -> None:
    """Show information about a material or config file."""
    from python_renderer.core.material import OpenPBRMaterial, RenderConfig

    ext = Path(path).suffix.lower()
    try:
        if ext == ".json":
            mat = OpenPBRMaterial.load_json(path)
            click.echo(json.dumps(mat.to_dict(), indent=2))
        elif ext in (".yaml", ".yml"):
            try:
                cfg = RenderConfig.load_yaml(path)
                click.echo(json.dumps(cfg.to_dict(), indent=2))
            except Exception:
                mat = OpenPBRMaterial.load_yaml(path)
                click.echo(json.dumps(mat.to_dict(), indent=2))
        else:
            click.echo(f"Unsupported file type: {ext}", err=True)
    except Exception as exc:
        click.echo(f"Error reading {path}: {exc}", err=True)
        sys.exit(1)


# ===========================================================================
# convert_material
# ===========================================================================

@main.command("convert-material")
@click.argument("input_path",  type=click.Path(exists=True))
@click.argument("output_path")
def convert_material(input_path: str, output_path: str) -> None:
    """Convert a material file between JSON and YAML formats."""
    from python_renderer.core.material import OpenPBRMaterial

    in_ext  = Path(input_path).suffix.lower()
    out_ext = Path(output_path).suffix.lower()

    if in_ext == ".json":
        mat = OpenPBRMaterial.load_json(input_path)
    elif in_ext in (".yaml", ".yml"):
        mat = OpenPBRMaterial.load_yaml(input_path)
    else:
        click.echo(f"Unsupported input format: {in_ext}", err=True)
        sys.exit(1)

    if out_ext == ".json":
        mat.save_json(output_path)
    elif out_ext in (".yaml", ".yml"):
        mat.save_yaml(output_path)
    else:
        click.echo(f"Unsupported output format: {out_ext}", err=True)
        sys.exit(1)

    click.echo(f"Converted {input_path} → {output_path}")


# ===========================================================================
# benchmark
# ===========================================================================

@main.command()
@click.option("--width",   default=1920, show_default=True)
@click.option("--height",  default=1080, show_default=True)
@click.option("--samples", default=64,   show_default=True)
def benchmark(width: int, height: int, samples: int) -> None:
    """Benchmark GPU (or CPU) pathtracer performance."""
    from python_renderer.core.material import OpenPBRMaterial, RenderConfig
    from python_renderer.loaders.model_loader import ModelLoader
    from python_renderer.gpu.pathtracer import GPUPathtracer

    config   = RenderConfig(width=width, height=height, samples=samples,
                            max_bounces=4, use_gpu=True)
    material = OpenPBRMaterial()
    mesh     = ModelLoader().create_shaderball(lat_segments=32, lon_segments=32)
    pt       = GPUPathtracer(config)
    pt.setup_gl()
    pt.load_scene(mesh, material)

    t0    = time.time()
    image = pt.render()
    elapsed = time.time() - t0

    total_rays = width * height * samples
    mrays_s    = total_rays / elapsed / 1e6
    click.echo(
        f"Benchmark: {width}x{height}, {samples} spp\n"
        f"  Time   : {elapsed:.2f}s\n"
        f"  MRays/s: {mrays_s:.2f}"
    )


if __name__ == "__main__":
    main()
