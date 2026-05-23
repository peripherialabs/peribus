"""
rio.font_loader
===============

Load bundled font files into Qt's font database at app startup.

Why bundle?
  - Reproducible look across Linux/macOS/Windows (no "looks great on my
    machine, ugly on yours" because of system font availability).
  - No install step for end users.
  - License-clean: all bundled fonts are SIL OFL.

Usage:
    from rio.font_loader import load_bundled_fonts
    # ...inside your QApplication setup, BEFORE creating any widgets:
    load_bundled_fonts()

The loader is idempotent — calling it twice does nothing the second time.
Failures (missing files, bad format) are logged but never raise.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Set

from PySide6.QtGui import QFontDatabase

logger = logging.getLogger(__name__)

# Resolve the fonts/ directory relative to this file so it works no matter
# how the package is launched.
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

_LOADED: Set[str] = set()  # font filenames already passed to QFontDatabase
_FAMILIES: Dict[str, List[str]] = {}  # filename -> the family names Qt registered


def load_bundled_fonts(font_dir: str = None) -> Dict[str, List[str]]:
    """Register every .ttf/.otf in ``font_dir`` with Qt's font database.

    Returns a {filename: [family_names]} map of what Qt actually
    registered, useful for debugging.  An empty list against a filename
    means Qt rejected the file (corrupt, unsupported format, etc.).

    Safe to call multiple times — already-loaded files are skipped.
    """
    directory = font_dir or _FONT_DIR
    if not os.path.isdir(directory):
        logger.warning(f"[font_loader] font directory not found: {directory}")
        return {}

    for entry in sorted(os.listdir(directory)):
        if not entry.lower().endswith((".ttf", ".otf")):
            continue
        if entry in _LOADED:
            continue
        path = os.path.join(directory, entry)
        font_id = QFontDatabase.addApplicationFont(path)
        if font_id < 0:
            logger.warning(f"[font_loader] Qt rejected font: {entry}")
            _FAMILIES[entry] = []
        else:
            families = QFontDatabase.applicationFontFamilies(font_id)
            _FAMILIES[entry] = list(families)
            logger.info(f"[font_loader] loaded {entry} -> {families}")
        _LOADED.add(entry)

    return dict(_FAMILIES)


def loaded_families() -> Set[str]:
    """Set of every family name registered via the loader so far.

    Useful for sanity checks — e.g. assert "IBM Plex Sans" in loaded_families()
    before assuming the theme can use it.
    """
    out: Set[str] = set()
    for fams in _FAMILIES.values():
        out.update(fams)
    return out