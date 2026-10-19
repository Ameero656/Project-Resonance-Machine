"""The adjustable parameters. K1 picks one, K2 changes it."""


class Param:
    """`steps`, coarse to fine, makes K1 press cycle the step size (e.g. 1000 -> 100 -> 10)."""

    def __init__(self, name, value, lo, hi, step, fmt="{}", wrap=False, steps=None):
        self.name, self.value = name, value
        self.lo, self.hi = lo, hi
        self.fmt, self.wrap = fmt, wrap
        self.steps = tuple(steps) if steps else (step,)
        self.level = 0  # index into steps; starts coarsest

    @property
    def step(self):
        return self.steps[self.level]

    @property
    def coarse(self):
        """True if K1 press can change the step size."""
        return len(self.steps) > 1

    def cycle_step(self):
        self.level = (self.level + 1) % len(self.steps)

    def nudge(self, clicks):
        v = self.value + clicks * self.step
        if self.wrap:
            span = self.hi - self.lo + self.step
            v = self.lo + (v - self.lo) % span
        else:
            v = min(self.hi, max(self.lo, v))
        self.value = round(v, 6)

    def text(self):
        return self.fmt.format(self.value)


CHANNELS = ("A", "B")
CHANNEL_KEYS = ("Show", "Particles", "Type", "Mass", "Colour", "Size")
TYPES = ("Sand", "Powder")
COLOURS = [
    ("Sand", (236, 214, 168)),
    ("Cyan", (110, 205, 235)),
    ("Rose", (245, 130, 160)),
    ("Lime", (175, 225, 110)),
    ("White", (240, 240, 245)),
]

# (Show, Particles, Type, Mass, Colour, Size). Channel A starts as white 1 px sand;
# channel B is set up as fine powder but starts hidden (turn Show on to add it).
CHANNEL_DEFAULTS = {"A": (1, 6000, 0, 1.0, 4, 1), "B": (0, 2500, 1, 1.0, 1, 1)}


def channel_params(ch):
    show, count, kind, mass, colour, size = CHANNEL_DEFAULTS[ch]
    return [
        # Off hides the channel: its sand isn't drawn or moved, but keeps its count and settings.
        Param(f"{ch} Show", show, 0, 1, 1, wrap=True),
        Param(f"{ch} Particles", count, 0, 20000, 1000, steps=(1000, 100, 10)),
        Param(f"{ch} Type", kind, 0, len(TYPES) - 1, 1, wrap=True),
        Param(f"{ch} Mass", mass, 0.25, 4.0, 0.25, steps=(0.25, 0.05)),
        Param(f"{ch} Colour", colour, 0, len(COLOURS) - 1, 1, wrap=True),
        # Drawn dot diameter in screen pixels. Display only: it doesn't change the physics.
        Param(f"{ch} Size", size, 1, 6, 1),
    ]


def default_params():
    """Plate parameters first, then each channel's (named "A Mass", "B Type", ...)."""
    plate = [
        Param("Uniformity", 0.7, 0.0, 1.0, 0.1, "{:.0%}", steps=(0.1, 0.01)),
        # How strongly the sand is pulled onto its lines: 25% drifts lazily, 300% snaps.
        Param("Adherence", 1.0, 0.25, 3.0, 0.25, steps=(0.25, 0.05)),
        # 0 is allowed: with free edges cos(0) = 1, which gives straight-line modes.
        Param("n", 3, 0, 12, 1),
        Param("m", 5, 0, 12, 1),
        # Weight of the second term: +1 / -1 are the classic +/- forms.
        Param("Mix", 1.0, -1.0, 1.0, 0.1, steps=(0.1, 0.01)),
        # 1 = free edges (cos form), 0 = fixed edges (sin form from the sketch).
        Param("Edges", 1, 0, 1, 1, wrap=True),
    ]
    settings = [
        Param("Display", 0, 0, 1, 1, wrap=True),          # 0 Resonance, 1 Oscilloscope
        Param("Sound", 1, 0, 1, 1, wrap=True),            # the mode tone, Off / On
        Param("Volume", 0.5, 0.0, 1.0, 0.1, steps=(0.1, 0.01)),
        Param("Base pitch", 80, 40, 160, 10, steps=(10, 1)),             # tone of mode (1, 1), in Hz
        Param("Live mic", 0, 0, 1, 1, wrap=True),
        Param("Theme", 0, 0, 2, 1, wrap=True),           # 0 Amber, 1 Mono, 2 White (ui.THEME_NAMES)
        # Music / live mic: how loud a note must be to count, and how much loudness
        # (and beats) drive the shaking: 0% = steady shaking, 100% = follows the volume.
        Param("Sensitivity", 0.5, 0.0, 1.0, 0.1, steps=(0.1, 0.01)),
        Param("Reactivity", 1.0, 0.0, 1.0, 0.1, steps=(0.1, 0.01)),
    ]
    return plate + [p for ch in CHANNELS for p in channel_params(ch)] + settings


def make_param(name, value):
    """A fresh Param with the standard range for `name`, set to `value`."""
    if name == "Hold":
        return Param("Hold", value, 1, 120, 10, steps=(10, 1))
    if name == "Reset":
        return Param("Reset", value, 0, 1, 1, wrap=True)
    p = next(p for p in default_params() if p.name == name)
    p.value = value
    return p


def format_value(name, value):
    if name[:2] in (f"{ch} " for ch in CHANNELS):
        name = name[2:]  # "A Mass" -> "Mass"
    if name == "Type":
        return TYPES[int(value)]
    if name == "Colour":
        return COLOURS[int(value)][0]
    if name == "Size":
        return f"{value} px"
    if name == "Mass":
        return f"{value:g}x"
    if name == "Display":
        return ("Resonance", "Oscilloscope")[int(value)]
    if name == "Theme":
        return ("Amber", "Mono", "White")[int(value)]
    if name in ("Sound", "Live mic", "Show"):
        return "On" if value else "Off"
    if name == "Base pitch":
        return f"{value:g} Hz"
    if name in ("Volume", "Sensitivity", "Reactivity"):
        return f"{value:.0%}"
    if name == "Mix":
        return f"{value:+.2f}"
    if name == "Reset":
        return "On" if value else "Off"
    if name == "Edges":
        return "Free" if value else "Fixed"
    if name == "Adherence":
        return f"{value:.0%}"
    if name == "Uniformity":
        return f"{value:.0%}"
    if name == "Hold":
        return f"{value:g} s"
    return str(value)


def format_step(name, step):
    """The K2 step size as shown on a row, e.g. "±100", "±1%"."""
    if name[:2] in (f"{ch} " for ch in CHANNELS):
        name = name[2:]
    if name in ("Uniformity", "Volume", "Sensitivity", "Reactivity", "Adherence"):
        return f"±{step:.0%}"
    if name == "Base pitch":
        return f"±{step:g} Hz"
    if name == "Hold":
        return f"±{step:g} s"
    return f"±{step:g}"
