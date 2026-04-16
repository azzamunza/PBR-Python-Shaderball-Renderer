"""
Optional real-time viewport for displaying rendered images.

Requires pygame; gracefully disabled when not available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False
    log.debug("pygame not available – Viewport will operate in headless mode.")


class Viewport:
    """
    Display a rendered image in an interactive window.

    When pygame is not available the class silently no-ops,
    allowing headless rendering to proceed unaffected.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "PBR Renderer",
    ) -> None:
        self.width  = width
        self.height = height
        self.title  = title
        self._screen: Optional[object] = None
        self._surface: Optional[object] = None
        self._running = False

        if _PYGAME_AVAILABLE:
            pygame.init()
            self._screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(title)
            self._running = True
            log.info("Viewport opened: %dx%d", width, height)

    def show(self, image: np.ndarray) -> None:
        """
        Display *image* in the viewport.

        image : (H, W, 3) float32 in [0, 1].
        """
        if not _PYGAME_AVAILABLE or self._screen is None:
            return

        # Convert float → uint8 and flip for pygame (top-left origin)
        img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if img8.shape[0] != self.height or img8.shape[1] != self.width:
            # Resize to viewport
            from python_renderer.loaders.texture_loader import TextureLoader
            loader = TextureLoader()
            img_f  = image.astype(np.float32)
            img_f  = loader.resize(img_f, self.width, self.height)
            img8   = (np.clip(img_f, 0, 1) * 255).astype(np.uint8)

        surf = pygame.surfarray.make_surface(img8.swapaxes(0, 1))
        self._screen.blit(surf, (0, 0))  # type: ignore[union-attr]
        pygame.display.flip()

    def update(self) -> bool:
        """
        Process pygame events.

        Returns False when the window has been closed, True otherwise.
        """
        if not _PYGAME_AVAILABLE:
            return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
        return self._running

    def save_screenshot(self, path: str) -> None:
        """Save the current viewport contents as a PNG."""
        if not _PYGAME_AVAILABLE or self._screen is None:
            log.warning("save_screenshot: pygame not available.")
            return
        pygame.image.save(self._screen, str(path))  # type: ignore[arg-type]
        log.info("Screenshot saved: %s", path)

    def close(self) -> None:
        """Close the viewport window."""
        if _PYGAME_AVAILABLE:
            pygame.quit()
        self._running = False

    def __enter__(self) -> "Viewport":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
