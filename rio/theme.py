"""
rio.theme
=========

Central registry of visual themes for Rio.

A *theme* describes the surface vocabulary of the scene: the canvas
background, how panel frames are drawn (stroke, fill, radius),
whether and how drop-shadows render under proxies, and the colours
that animated dark-mode transitions interpolate towards.

Two themes ship by default:

  * ``"glass"`` — the original look: 5-px-rounded translucent
    panels floating on the canvas with chunky 45-px drop shadows.
    Looks great when shadows can carry the depth.

  * ``"paper"`` — flat editorial style inspired by print:
    hairline 1-px borders, opaque panel fills, no rounding,
    no shadows.  Modern and minimal.  Holds up in any lighting
    because it doesn't lean on translucency.

Switching themes at runtime is animated (interpolated colours +
geometry over ~800 ms) by ``RioWindow.set_theme``; the theme
registry itself is a passive description.

Usage from outside this module::

    from rio.theme import THEMES, get_theme
    t = get_theme("paper")
    t.scene_bg(dark_mode=False)              # -> QColor
    t.frame_stylesheet(dark_mode=False)      # -> str   (QSS)
    t.shadow_for(dark_mode=False)            # -> ShadowSpec | None

A theme is intentionally a dataclass of simple values — no Qt
imports at module load — so it can be inspected, serialised,
diffed, and tested without a running QApplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# We deliberately keep PySide6 out of the module *body*; only helpers
# that build QColor/QBrush/etc. import it lazily, so this file can
# be imported in headless contexts (tests, docs build) without
# pulling in Qt.

RGBA = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Sub-specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShadowSpec:
    """Drop-shadow parameters for a proxy.

    A theme returns ``None`` instead of a ShadowSpec when it wants
    *no* shadow — call sites must check for that and skip
    ``QGraphicsDropShadowEffect`` setup entirely (don't just create
    a transparent shadow; that still costs Qt a render pass).
    """
    blur_radius: float
    offset_x: float
    offset_y: float
    color: RGBA            # light-mode shadow tint
    color_dark: RGBA       # dark-mode shadow tint (typically near-white)


@dataclass(frozen=True)
class FrameSpec:
    """How a terminal/panel frame is drawn.

    Border + fill are kept as RGBA tuples (not QColor) so this
    dataclass remains pure data — see module docstring.
    """
    border_width: int                 # px — 0 means no border
    border_rgba: RGBA                 # light-mode stroke
    border_rgba_dark: RGBA            # dark-mode stroke
    fill_rgba_idle: RGBA              # bg when not focused (often alpha=0)
    fill_rgba_focused: RGBA           # bg when focused
    fill_rgba_idle_dark: RGBA
    fill_rgba_focused_dark: RGBA
    radius: int                       # px — 0 for sharp, editorial corners
    inner_padding: int = 10           # contentMargins inside the frame


@dataclass(frozen=True)
class TextSpec:
    """Text colour palette used by terminal output streams."""
    default_rgba: RGBA                # main user text, light mode
    default_rgba_dark: RGBA           # main user text, dark mode
    selection_rgba: RGBA
    selection_rgba_dark: RGBA


@dataclass(frozen=True)
class InputSpec:
    """Command-input area styling.

    The input has an idle alpha of 0 (invisible bg) and animates
    to ``focus_alpha`` on focus.  Theme controls both the RGB
    base and the focus-target alpha.
    """
    bg_rgb: RGBA                      # alpha is the idle alpha; usually 0
    bg_rgb_dark: RGBA
    focus_alpha: int                  # target alpha on focus (light)
    focus_alpha_dark: int
    text_rgba: RGBA
    text_rgba_dark: RGBA
    border_radius: int                # 0 for editorial; 3-5 for glass


@dataclass(frozen=True)
class FontSpec:
    """Font families used by a theme.

    Each field is a comma-separated CSS-style font stack so that if the
    primary family wasn't bundled or installed, Qt walks down to the
    fallbacks.  Always end with a generic family (``sans-serif`` /
    ``monospace``) so we never hit a "Qt picks Times by default" trap.

    UI vs. mono are separate because terminal output should remain
    monospace even when a theme wants its menus / labels in a sans.
    """
    ui_family: str       # used for menus, labels, dialogs
    mono_family: str     # used for terminal output and command input


# ---------------------------------------------------------------------------
# Top-level theme
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Theme:
    """A complete visual theme for a Rio scene."""
    name: str
    scene_bg_rgb: Tuple[int, int, int]       # light mode canvas
    scene_bg_rgb_dark: Tuple[int, int, int]  # dark mode canvas
    frame: FrameSpec
    text: TextSpec
    input: InputSpec
    shadow: Optional[ShadowSpec]             # None disables shadows entirely
    # ``font`` is optional with a sensible legacy default so older
    # Theme instances and external code that doesn't know about FontSpec
    # keep rendering correctly.  Themes that care about typography
    # should override this — see PAPER for an editorial example.
    font: FontSpec = field(default_factory=lambda: FontSpec(
        ui_family="'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
        mono_family="'Consolas', 'Monaco', monospace",
    ))
    # Menu CSS is theme-specific because the editorial/paper look
    # wants square menus with hairline borders, while glass wants
    # rounded translucent ones.
    menu_css_light: str = ""
    menu_css_dark: str = ""

    # ------------------------------------------------------------------
    # Convenience accessors  (return Qt objects)
    # ------------------------------------------------------------------

    def scene_bg(self, dark_mode: bool):
        """Return the scene background as a QColor."""
        from PySide6.QtGui import QColor
        rgb = self.scene_bg_rgb_dark if dark_mode else self.scene_bg_rgb
        return QColor(*rgb)

    def shadow_for(self, dark_mode: bool):
        """Return (blur, offset_x, offset_y, QColor) or None."""
        if self.shadow is None:
            return None
        from PySide6.QtGui import QColor
        rgba = self.shadow.color_dark if dark_mode else self.shadow.color
        return (
            self.shadow.blur_radius,
            self.shadow.offset_x,
            self.shadow.offset_y,
            QColor(*rgba),
        )

    def frame_stylesheet(self, dark_mode: bool, focus_alpha: int = 0) -> str:
        """Return QSS for a terminal_frame given mode and current focus alpha.

        The focus alpha is interpolated by the focus-fade animation;
        the call site passes the current value, this helper just
        composes the final QSS string.

        Two policies:
          * **Glass** (translucent panels): the focus animation drives
            the literal background alpha — at rest the panel is invisible,
            on focus it fades in to its target opacity.
          * **Paper** (opaque editorial): the panel is *always* shown
            at its full fill alpha; the focus animation acts as an
            additional tint laid over the surface.  ``focus_alpha`` here
            controls the strength of that tint, not the panel itself.

        We distinguish the two by checking whether the theme's idle fill
        has alpha ≥ 200 — opaque themes opt into the second policy.
        """
        f = self.frame
        if dark_mode:
            border_rgba = f.border_rgba_dark
            idle_fill = f.fill_rgba_idle_dark
            focused_fill = f.fill_rgba_focused_dark
        else:
            border_rgba = f.border_rgba
            idle_fill = f.fill_rgba_idle
            focused_fill = f.fill_rgba_focused

        # Use the focused fill's RGB throughout the animation so hue
        # doesn't drift mid-fade.  Use idle's RGB when nothing else
        # has been chosen.
        fr, fgc, fbc, focused_alpha = focused_fill
        _, _, _, idle_alpha = idle_fill
        br, bg, bb, _ba = border_rgba

        # Opaque-theme policy: clamp displayed alpha to the fill's own,
        # so the panel never goes transparent during a focus fade.
        is_opaque_theme = idle_alpha >= 200 and focused_alpha >= 200
        if is_opaque_theme:
            displayed_alpha = focused_alpha  # always show the fill at full strength
        else:
            # Glass-style: literal focus_alpha is the displayed alpha.
            # Idle is 0; focused tween goes up to focused_alpha.
            displayed_alpha = focus_alpha

        if f.border_width <= 0:
            border_css = "border: none;"
        else:
            border_css = (
                f"border: {f.border_width}px solid "
                f"rgba({br}, {bg}, {bb}, {_ba});"
            )
        return (
            "QFrame {\n"
            f"    background-color: rgba({fr}, {fgc}, {fbc}, {displayed_alpha});\n"
            f"    {border_css}\n"
            f"    border-radius: {f.radius}px;\n"
            "}\n"
        )


# ---------------------------------------------------------------------------
# Built-in themes
# ---------------------------------------------------------------------------

# The "glass" theme reproduces the original Rio look exactly so existing
# scenes keep rendering identically when no explicit theme is set.
GLASS = Theme(
    name="glass",
    scene_bg_rgb=(250, 250, 250),
    scene_bg_rgb_dark=(18, 18, 25),
    frame=FrameSpec(
        border_width=2,
        border_rgba=(150, 150, 150, 200),
        border_rgba_dark=(200, 200, 200, 220),
        fill_rgba_idle=(255, 255, 255, 0),
        fill_rgba_focused=(255, 255, 255, 230),
        fill_rgba_idle_dark=(40, 42, 52, 0),
        fill_rgba_focused_dark=(40, 42, 52, 180),
        radius=5,
        inner_padding=10,
    ),
    text=TextSpec(
        default_rgba=(0, 0, 0, 230),
        default_rgba_dark=(230, 230, 230, 240),
        selection_rgba=(100, 100, 255, 100),
        selection_rgba_dark=(100, 100, 255, 120),
    ),
    input=InputSpec(
        bg_rgb=(255, 255, 255, 0),
        bg_rgb_dark=(40, 40, 50, 0),
        focus_alpha=230,
        focus_alpha_dark=180,
        text_rgba=(0, 0, 0, 255),
        text_rgba_dark=(230, 230, 230, 255),
        border_radius=3,
    ),
    shadow=ShadowSpec(
        blur_radius=25.0,
        offset_x=30.0,
        offset_y=30.0,
        color=(0, 0, 0, 120),
        color_dark=(255, 255, 255, 160),
    ),
    menu_css_light=(
        "QMenu { background-color: rgba(255,255,255,200); border: 1px solid #000000;"
        " padding: 2px; }"
        " QMenu::item { color: #000000; padding: 4px 20px 4px 10px; }"
        " QMenu::item:selected { background-color: rgba(0,0,0,242); color: #ffffff; }"
        " QMenu::separator { height: 1px; background: #000000; margin: 2px 4px; }"
    ),
    menu_css_dark=(
        "QMenu { background-color: rgba(0,0,0,242); border: 1px solid #000000;"
        " padding: 2px; }"
        " QMenu::item { color: #ffffff; padding: 4px 20px 4px 10px; }"
        " QMenu::item:selected { background-color: rgba(255,255,255,242); color: #000000; }"
        " QMenu::separator { height: 1px; background: #ffffff; margin: 2px 4px; }"
    ),
    # Glass keeps the legacy Consolas/Monaco for the terminal so the
    # default look is unchanged from before themes existed.  UI font
    # falls back through the standard system stack.
    font=FontSpec(
        ui_family="'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
        mono_family="'Consolas', 'Monaco', 'Menlo', monospace",
    ),
)


# The "paper" theme: warm off-white canvas, hairline near-black borders,
# zero radius, opaque fills, no drop shadows.  Designed to look like
# editorial print on paper — see e.g. typographic comparison tables
# from product pages.
#
# Tuning notes:
#   - 0xF0EDE6 light bg (warm grey-cream); 0x1A1A1A dark bg (near-black ink).
#   - 1px hairline borders at near-black give the print-like decisiveness.
#   - Fills are fully opaque — when shadows go away, translucency
#     loses its purpose, and opaque panels read as crisper.
#   - 0px radius is a deliberate choice; 2px would already start
#     looking like a friendly product UI rather than editorial.
PAPER = Theme(
    name="paper",
    scene_bg_rgb=(240, 237, 230),
    scene_bg_rgb_dark=(26, 26, 26),
    frame=FrameSpec(
        border_width=1,
        border_rgba=(42, 42, 42, 255),
        border_rgba_dark=(180, 180, 180, 220),
        # Idle == focused for paper: opaque fill always, no fade-in.
        # The focus-alpha animation is still wired up but the values
        # collapse so it visually does nothing — input still gets
        # the focus highlight (see InputSpec).
        fill_rgba_idle=(250, 247, 240, 255),
        fill_rgba_focused=(250, 247, 240, 255),
        fill_rgba_idle_dark=(34, 34, 34, 255),
        fill_rgba_focused_dark=(34, 34, 34, 255),
        radius=0,
        inner_padding=14,
    ),
    text=TextSpec(
        default_rgba=(26, 26, 26, 255),
        default_rgba_dark=(232, 230, 224, 255),
        selection_rgba=(180, 200, 180, 140),     # sage tint, like the reference
        selection_rgba_dark=(180, 200, 180, 100),
    ),
    input=InputSpec(
        bg_rgb=(250, 247, 240, 0),
        bg_rgb_dark=(34, 34, 34, 0),
        # Subtle focus tint — a paper UI shouldn't shout, but the
        # input still needs to feel "live" when typed in.
        focus_alpha=40,
        focus_alpha_dark=60,
        text_rgba=(26, 26, 26, 255),
        text_rgba_dark=(232, 230, 224, 255),
        border_radius=0,
    ),
    shadow=None,  # <-- the headline of the theme
    menu_css_light=(
        "QMenu { background-color: #FAF7F0; border: 1px solid #2A2A2A;"
        " padding: 0px; }"
        " QMenu::item { color: #1A1A1A; padding: 5px 22px 5px 12px;"
        " font-family: 'IBM Plex Sans', 'Inter', 'Segoe UI', sans-serif; }"
        " QMenu::item:selected { background-color: #1A1A1A; color: #FAF7F0; }"
        " QMenu::separator { height: 1px; background: #2A2A2A; margin: 0px; }"
    ),
    menu_css_dark=(
        "QMenu { background-color: #222222; border: 1px solid #B4B4B4;"
        " padding: 0px; }"
        " QMenu::item { color: #E8E6E0; padding: 5px 22px 5px 12px;"
        " font-family: 'IBM Plex Sans', 'Inter', 'Segoe UI', sans-serif; }"
        " QMenu::item:selected { background-color: #E8E6E0; color: #222222; }"
        " QMenu::separator { height: 1px; background: #B4B4B4; margin: 0px; }"
    ),
    # Paper opts into IBM Plex (SIL OFL, free, editorial-leaning).
    # Falls back to Inter, then system sans/mono.  If neither bundled
    # nor installed, Qt's substitution gives the standard system font —
    # never a worst-case "Times New Roman" surprise because we end the
    # stack with a generic family.
    font=FontSpec(
        ui_family="'IBM Plex Sans', 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
        mono_family="'IBM Plex Mono', 'JetBrains Mono', 'Consolas', 'Menlo', monospace",
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

THEMES: Dict[str, Theme] = {
    "glass": GLASS,
    "paper": PAPER,
}

# The default theme used when the scene initialises.  Kept as
# "glass" for backward compatibility — existing scenes don't change
# unless the caller asks for "paper".
DEFAULT_THEME_NAME = "glass"


def get_theme(name: str) -> Theme:
    """Look up a theme by name.  Falls back to the default if unknown."""
    return THEMES.get(name, THEMES[DEFAULT_THEME_NAME])


def register_theme(theme: Theme) -> None:
    """Register a custom theme.  Overwrites any existing theme with the same name."""
    THEMES[theme.name] = theme