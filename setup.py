"""Package setup for PBR Python Shaderball Renderer."""

from pathlib import Path
from setuptools import setup, find_packages

long_description = (Path(__file__).parent / "docs" / "README.md").read_text(encoding="utf-8")

setup(
    name="pbr-python-renderer",
    version="1.0.0",
    author="azzamunza",
    description="GPU-accelerated OpenPBR v1.2 path tracer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/azzamunza/PBR-Python-Shaderball-Renderer",
    packages=find_packages(exclude=["tests*", "docs*"]),
    package_data={
        "python_renderer": [
            "gpu/shaders/*.glsl",
            "examples/materials/*.json",
            "examples/configs/*.yaml",
        ]
    },
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "imageio>=2.9.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "gpu": [
            "PyOpenGL>=3.1.5",
            "PyGLM>=2.6.0",
            "glfw>=2.5.0",
        ],
        "cuda": [
            "cupy-cuda11x>=10.0.0",
        ],
        "loader": [
            "trimesh>=3.15.0",
        ],
        "viewer": [
            "pygame>=2.0.0",
        ],
        "all": [
            "PyOpenGL>=3.1.5",
            "PyGLM>=2.6.0",
            "glfw>=2.5.0",
            "trimesh>=3.15.0",
            "pygame>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pbr-render=python_renderer.ui.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
)
