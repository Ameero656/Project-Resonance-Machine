"""Look and feel: colours, fonts and small drawing helpers shared by the panels.

Instrument-panel styles (themes below): one pixel monospace face (Departure Mono, SIL
Open Font License, see fonts/LICENSE-DepartureMono.txt), hairline rules, square
corners and one accent colour. Every piece of text is fitted to a width, and
each panel is clipped to its own rectangle, so nothing spills onto the plate.
If the font files are missing it falls back to pygame's built-in font.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

THEMES = {
    # Dark instrument panel with one amber accent.
    "Amber": dict(
        BG=(6, 7, 9), SELECTED=(17, 19, 24), SEL_OUTLINE=None, EDGE=(32, 35, 42), LINE=(32, 35, 42),
        BORDER=None,
        TICK=(70, 76, 88), TEXT=(214, 218, 226), DIM=(112, 119, 132), FAINT=(62, 67, 78),
        ACCENT=(255, 176, 64), WARN=(240, 100, 90),
        GRID=(34, 38, 49), AXIS=(74, 80, 98), LABEL=(128, 135, 150),
        CHANNEL={"1": (255, 176, 64), "2": (90, 200, 255)}, HEAT=(70, 90, 160),
    ),
    # Pure black with white outlines and greys; colour only for the two scope channels.
    "Mono": dict(
        BG=(0, 0, 0), SELECTED=None, SEL_OUTLINE=(255, 255, 255), EDGE=(255, 255, 255), LINE=(255, 255, 255),
        BORDER=(255, 255, 255),
        TICK=(255, 255, 255), TEXT=(255, 255, 255), DIM=(150, 150, 150), FAINT=(85, 85, 85),
        ACCENT=(255, 255, 255), WARN=(255, 150, 40),
        GRID=(38, 38, 38), AXIS=(110, 110, 110), LABEL=(150, 150, 150),
        CHANNEL={"1": (255, 150, 40), "2": (70, 170, 255)}, HEAT=(70, 90, 160),
    ),
    # Mono with no colour at all: white and greys only, both scope channels white.
    "White": dict(
        BG=(0, 0, 0), SELECTED=None, SEL_OUTLINE=(255, 255, 255), EDGE=(255, 255, 255), LINE=(255, 255, 255),
        BORDER=(255, 255, 255),
        TICK=(255, 255, 255), TEXT=(255, 255, 255), DIM=(150, 150, 150), FAINT=(85, 85, 85),
        ACCENT=(255, 255, 255), WARN=(255, 255, 255),
        GRID=(38, 38, 38), AXIS=(110, 110, 110), LABEL=(150, 150, 150),
        CHANNEL={"1": (255, 255, 255), "2": (255, 255, 255)}, HEAT=(90, 90, 90),
    ),
}
THEME_NAMES = tuple(THEMES)


class Theme:
    """The current palette. Read colours as ui.T.TEXT etc. at draw time."""

    def __init__(self):
        self.CHANNEL = {}
        self.use("Amber")

    def use(self, name):
        for k, v in THEMES[name].items():
            if k == "CHANNEL":
                self.CHANNEL.clear()
                self.CHANNEL.update(v)  # same dict object, so scope.COLOURS follows
            else:
                setattr(self, k, v)
        self.name = name


T = Theme()


class Fonts:
    """Departure Mono, a pixel font: drawn without smoothing, at sizes where its
    pixels stay square (11 and 22 px, and 16 px, which renders as a 2-pixel stroke)."""

    FILE = os.path.join(FONT_DIR, "DepartureMono-Regular.otf")

    def __init__(self, pygame):
        found = os.path.exists(self.FILE)
        self.antialias = not found  # pygame's fallback font needs smoothing

        def load(size):
            if found:
                return pygame.font.Font(self.FILE, size)
            return pygame.font.Font(None, round(size * 1.4))  # the default font runs small

        self.caption = load(11)   # UPPERCASE labels
        self.tick = load(11)      # scope axis labels
        self.body = load(11)
        self.value = load(16)
        self.title = load(16)
        self.big = load(22)


class Painter:
    """Draws text and shapes onto one surface, caching rendered text."""

    def __init__(self, pygame, surface, fonts):
        self.pg, self.surface, self.fonts = pygame, surface, fonts
        self.cache = {}

    def fit(self, s, font, width):
        """s, cut short with an ellipsis if it's wider than `width` pixels."""
        if width is None or font.size(s)[0] <= width:
            return s
        while s and font.size(s + "…")[0] > width:
            s = s[:-1]
        return s.rstrip() + "…"

    def render(self, s, font, colour):
        key = (s, id(font), colour)
        surf = self.cache.get(key)
        if surf is None:
            if len(self.cache) > 800:
                self.cache.clear()
            surf = self.cache[key] = font.render(s, self.fonts.antialias, colour)
        return surf

    def text(self, s, x, y, font=None, colour=None, anchor="topleft", width=None):
        """Draw text no wider than `width`; returns its height."""
        font = font or self.fonts.body
        colour = colour or T.TEXT
        surf = self.render(self.fit(s, font, width), font, colour)
        rect = surf.get_rect(**{anchor: (round(x), round(y))})
        self.surface.blit(surf, rect)
        return rect.height

    def caption(self, s, x, y, colour=None, anchor="topleft", width=None):
        colour = colour or T.DIM
        return self.text(s.upper(), x, y, self.fonts.caption, colour, anchor, width)

    def width(self, s, font):
        return font.size(s)[0]

    def rect(self, rect, colour):
        self.pg.draw.rect(self.surface, colour, rect)

    def frame(self, rect, colour=None):
        colour = colour or T.LINE
        self.pg.draw.rect(self.surface, colour, rect, 1)

    def line(self, x0, y0, x1, y1, colour=None):
        colour = colour or T.LINE
        self.pg.draw.line(self.surface, colour, (x0, y0), (x1, y1))

    def rule(self, label, x, y, right, colour=None):
        """A section heading: LABEL followed by a hairline to `right`. Returns the y below it."""
        colour = colour or T.DIM
        w = self.width(label.upper(), self.fonts.caption)
        self.caption(label, x, y, colour)
        mid = y + self.fonts.caption.get_height() // 2
        self.line(x + w + 6, mid, right, mid)
        return y + self.fonts.caption.get_height() + 6

    def pair(self, label, value, x, y, right, colour=None, font=None):
        """LABEL on the left, value right-aligned on the same line. Returns the line height."""
        font = font or self.fonts.body
        vw = min(self.width(value, font), right - x - 30)
        self.caption(label, x, y + 1, T.DIM, width=right - x - vw - 8)
        return self.text(value, right, y, font, colour, anchor="topright", width=right - x - 30)

    def corners(self, rect, colour=None, size=8):
        """Crop marks at the four corners of a rectangle."""
        colour = colour or T.TICK
        x, y, w, h = rect
        r, b = x + w - 1, y + h - 1
        for cx, cy, dx, dy in ((x, y, 1, 1), (r, y, -1, 1), (x, b, 1, -1), (r, b, -1, -1)):
            self.line(cx, cy, cx + dx * size, cy, colour)
            self.line(cx, cy, cx, cy + dy * size, colour)

    def square(self, x, y, size, colour):
        self.pg.draw.rect(self.surface, colour, (round(x), round(y), size, size))

    def clip(self, rect):
        self.surface.set_clip(None if rect is None else self.pg.Rect(rect))
