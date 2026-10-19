"""Two-channel oscilloscope display, with a built-in function generator.

Each channel generates a sine, square or triangle wave at its own frequency
and amplitude, or shows real audio: the live microphone or the song playing
in Music (Amplitude is then the gain on that signal). The graph is 10 divisions across and 8 down, like a bench
scope: Time/div (shared by both channels, as on a real scope) sets the
horizontal scale, and each channel's Volts/div and Position set its vertical
scale and move it up or down. The sweep is triggered on channel 1's rising
zero crossing, so channel 1 stands still and channel 2 drifts unless its
frequency is a multiple of channel 1's. With audio on channel 1 the sweep
starts at the latest rising zero crossing that still fits a full screen.

Mode X-Y plots channel 1 across and channel 2 up instead of either against
time, as a scope does with two signal generators: sines at simple frequency
ratios (1:2, 2:3, ...) draw Lissajous figures, Phase on a channel turns them,
and a slight detune makes the figure roll. Time/div then sets how much of the
recent signal is drawn, with the oldest part fading like scope phosphor.
"""

from fractions import Fraction

import numpy as np

from params import Param
from ui import T

WAVES = ("Off", "Sine", "Square", "Triangle", "Live mic", "Music")
AUDIO = ("Live mic", "Music")   # sources that come from a sound rather than the generator
TIME_DIVS = (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
VOLT_DIVS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
MODES = ("Y-T", "X-Y")
H_DIVS, V_DIVS = 10, 8
XY_DIVS = 8      # the X-Y graticule is square: 8 divisions each way
XY_FADE = 12     # the curve is drawn in this many pieces, oldest dimmest
CHANNELS = ("1", "2")
COLOURS = T.CHANNEL  # follows the UI theme


def default_params():
    ps = [Param("Mode", 0, 0, len(MODES) - 1, 1, wrap=True),
          Param("Time/div", TIME_DIVS.index(0.001), 0, len(TIME_DIVS) - 1, 1)]
    defaults = {"1": (1, 500.0, 1.0), "2": (0, 500.0, 1.0)}
    for ch in CHANNELS:
        wave, freq, amp = defaults[ch]
        ps += [
            Param(f"CH{ch} Wave", wave, 0, len(WAVES) - 1, 1, wrap=True),
            Param(f"CH{ch} Frequency", freq, 1.0, 20000.0, 1000, steps=(1000, 100, 10, 1)),
            Param(f"CH{ch} Amplitude", amp, 0.1, 5.0, 1, steps=(1, 0.1, 0.01)),
            Param(f"CH{ch} Phase", 0.0, 0.0, 359.0, 15, wrap=True, steps=(15, 1)),
            Param(f"CH{ch} Volts/div", VOLT_DIVS.index(0.5), 0, len(VOLT_DIVS) - 1, 1),
            Param(f"CH{ch} Position", 0.0, -4.0, 4.0, 1, steps=(1, 0.1)),
        ]
    return ps


KEYS = tuple(p.name for p in default_params())


def format_time(s):
    if s >= 1:
        return f"{s:g} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3g} ms"
    return f"{s * 1e6:.3g} us"


def format_volts(v):
    return f"{v * 1e3:.3g} mV" if 0 < abs(v) < 1 else f"{v:.3g} V"


def format_freq(f):
    if f >= 1000:
        return f"{f / 1e3:.3f}".rstrip("0").rstrip(".") + " kHz"
    return f"{f:.4g} Hz"


def format_step(name, step):
    key = name.split(" ", 1)[-1]
    if key == "Frequency":
        return f"±{step:g} Hz"
    if key == "Amplitude":
        return f"±{step:g} V"
    if key == "Position":
        return f"±{step:g} div"
    if key == "Phase":
        return f"±{step:g}°"
    return f"±{step:g}"


def format_value(name, value):
    key = name.split(" ", 1)[-1]
    if name == "Mode":
        return MODES[int(value)]
    if name == "Time/div":
        return format_time(TIME_DIVS[int(value)]) + "/div"
    if key == "Phase":
        return f"{value:g}°"
    if key == "Wave":
        return WAVES[int(value)]
    if key == "Frequency":
        return format_freq(value)
    if key == "Amplitude":
        return format_volts(value)
    if key == "Volts/div":
        return format_volts(VOLT_DIVS[int(value)]) + "/div"
    if key == "Position":
        return f"{value:+.1f} div"
    return str(value)


def wave(kind, cycles):
    """Unit-amplitude waveform at `cycles` (time x frequency), starting at 0 and rising."""
    p = np.mod(cycles, 1.0)
    if kind == "Sine":
        return np.sin(2 * np.pi * p)
    if kind == "Square":
        return np.where(p < 0.5, 1.0, -1.0)
    if kind == "Triangle":
        return 1.0 - 4.0 * np.abs(np.mod(p + 0.25, 1.0) - 0.5)
    return np.zeros_like(p)


def ratio(f1, f2, most=9):
    """f1:f2 as small whole numbers, e.g. "2:3"; "~2:3" if slightly detuned (the
    figure rolls), or None if no simple ratio is close."""
    q = f1 / f2
    r = Fraction(q).limit_denominator(most)
    if r.numerator == 0 or r.numerator > most:
        return None
    off = abs(r - q) / q
    if off > 0.01:
        return None
    return ("" if off < 1e-3 else "~") + f"{r.numerator}:{r.denominator}"


def rising_edge(x, last):
    """Index of the latest rising zero crossing in x[:last + 1], or None.

    A short moving average first, so noise near zero doesn't cause false triggers."""
    if len(x) < 8 or last < 1:
        return None
    k = np.ones(5) / 5
    y = np.convolve(x[:last + 1], k, mode="same")
    up = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0)) + 1
    return int(up[-1]) if len(up) else None


class Scope:
    def __init__(self, params):
        self.p = {p.name: p for p in params if p.name in KEYS}
        # audio(kind) -> (samples, rate) with the newest sample = now, or None if
        # that source has nothing (no mic, no song). Set by the app.
        self.audio = lambda kind: None

    def value(self, name):
        return self.p[name].value

    def channel(self, ch):
        return {
            "wave": WAVES[int(self.value(f"CH{ch} Wave"))],
            "freq": self.value(f"CH{ch} Frequency"),
            "amp": self.value(f"CH{ch} Amplitude"),
            "vdiv": VOLT_DIVS[int(self.value(f"CH{ch} Volts/div"))],
            "pos": self.value(f"CH{ch} Position"),
            "phase": self.value(f"CH{ch} Phase") / 360.0,  # in cycles
        }

    @property
    def xy(self):
        return MODES[int(self.value("Mode"))] == "X-Y"

    @property
    def time_div(self):
        return TIME_DIVS[int(self.value("Time/div"))]

    def trigger(self, now):
        """Start of the sweep as (generator time, seconds before the newest audio sample)."""
        span = self.time_div * H_DIVS
        c1 = self.channel("1")
        if c1["wave"] in AUDIO:
            got = self.audio(c1["wave"])
            if got is not None:
                x, rate = got
                i = rising_edge(x, len(x) - int(np.ceil(span * rate)))
                if i is not None:
                    back = (len(x) - i) / rate
                    return now - back, back
            return now - span, span  # nothing to trigger on: free-run
        if c1["wave"] == "Off":
            return now, span
        # The rising zero crossing of channel 1, after its own phase shift.
        f, ph = c1["freq"], c1["phase"]
        return (np.floor(now * f + ph) - ph) / f, span

    def trace(self, ch, now, points):
        """Sample times (s from the trigger) and volts for one sweep, or None if there's no signal."""
        start, back = self.trigger(now)
        t = np.linspace(0.0, self.time_div * H_DIVS, points)
        v = self.volts(ch, start, back, t)
        return None if v is None else (t, v)

    def volts(self, ch, start, back, t):
        """A channel's volts at times t after `start` (generator time), which is
        `back` seconds before the newest audio sample. None if there's no signal."""
        c = self.channel(ch)
        if c["wave"] in AUDIO:
            got = self.audio(c["wave"])
            if got is None:
                return None
            x, rate = got
            idx = len(x) - back * rate + t * rate
            return c["amp"] * np.interp(idx, np.arange(len(x)), x, left=0.0, right=0.0)
        return c["amp"] * wave(c["wave"], (start + t) * c["freq"] + c["phase"])

    def xy_trace(self, now, points):
        """Channel 1 against channel 2 over the last Time/div x 10 seconds, oldest first,
        in divisions from the centre. None if neither channel has a signal."""
        span = self.time_div * H_DIVS
        t = np.linspace(0.0, span, points)
        out = []
        for ch in CHANNELS:
            c = self.channel(ch)
            v = None if c["wave"] == "Off" else self.volts(ch, now - span, span, t)
            out.append(None if v is None else v / c["vdiv"] + c["pos"])
        if out[0] is None and out[1] is None:
            return None
        return tuple(np.zeros(points) if d is None else d for d in out)

    # --- drawing -------------------------------------------------------

    def draw(self, pygame, surface, rect, small, now):
        if self.xy:
            return self.draw_xy(pygame, surface, rect, small, now)
        x0, y0, w, h = rect
        # Margins for the labels: CH1 volts on the left, CH2 on the right, time below.
        gx, gy = x0 + 44, y0 + 26
        gw, gh = w - 88, h - 62
        dx, dy = gw / H_DIVS, gh / V_DIVS

        for i in range(H_DIVS + 1):
            x = round(gx + i * dx)
            pygame.draw.line(surface, T.AXIS if i == H_DIVS // 2 else T.GRID, (x, gy), (x, gy + gh))
        for j in range(V_DIVS + 1):
            y = round(gy + j * dy)
            pygame.draw.line(surface, T.AXIS if j == V_DIVS // 2 else T.GRID, (gx, y), (gx + gw, y))
        # Minor ticks on the centre lines.
        for i in range(H_DIVS * 5):
            x = round(gx + i * dx / 5)
            pygame.draw.line(surface, T.AXIS, (x, gy + gh / 2 - 3), (x, gy + gh / 2 + 3))
        for j in range(V_DIVS * 5):
            y = round(gy + j * dy / 5)
            pygame.draw.line(surface, T.AXIS, (gx + gw / 2 - 3, y), (gx + gw / 2 + 3, y))

        def label(text, x, y, colour, anchor):
            s = small.render(text, True, colour)
            r = s.get_rect(**{anchor: (round(x), round(y))})
            surface.blit(s, r)

        # Time axis: every other division, from the trigger point.
        td = self.time_div
        for i in range(0, H_DIVS + 1, 2):
            label("0" if i == 0 else format_time(i * td), gx + i * dx, gy + gh + 6, T.LABEL, "midtop")
        label(f"time ({format_time(td)}/div)", gx + gw / 2, gy + gh + 22, T.LABEL, "midtop")

        # Voltage axes, one per channel, offset by that channel's Position.
        for ch, side in (("1", "left"), ("2", "right")):
            c = self.channel(ch)
            if c["wave"] == "Off":
                continue
            for j in range(0, V_DIVS + 1, 2):
                volts = (V_DIVS / 2 - j - c["pos"]) * c["vdiv"]
                x = gx - 6 if side == "left" else gx + gw + 6
                anchor = "midright" if side == "left" else "midleft"
                label(f"{volts:+.3g}", x, gy + j * dy, COLOURS[ch], anchor)
            label(f"CH{ch} V", gx - 6 if side == "left" else gx + gw + 6, y0 + 2, COLOURS[ch],
                  "topright" if side == "left" else "topleft")

        # Traces.
        surface.set_clip(pygame.Rect(gx, gy, gw + 1, gh + 1))
        points = int(gw) * 2
        for ch in CHANNELS:
            c = self.channel(ch)
            if c["wave"] == "Off":
                continue
            tr = self.trace(ch, now, points)
            if tr is None:
                continue
            t, v = tr
            xs = gx + t / (td * H_DIVS) * gw
            ys = gy + gh / 2 - (v / c["vdiv"] + c["pos"]) * dy
            ys = np.clip(ys, gy - 10, gy + gh + 10)
            pygame.draw.lines(surface, COLOURS[ch], False, list(zip(xs, ys)), 2)
            # Ground marker: where 0 V sits for this channel.
            zero = gy + gh / 2 - c["pos"] * dy
            mx = gx + 1 if ch == "1" else gx + gw - 7
            pygame.draw.polygon(surface, COLOURS[ch], [(mx, zero - 4), (mx + 6, zero), (mx, zero + 4)]
                                if ch == "1" else [(mx + 6, zero - 4), (mx, zero), (mx + 6, zero + 4)])
        surface.set_clip(None)

    def draw_xy(self, pygame, surface, rect, small, now):
        x0, y0, w, h = rect
        size = min(w - 88, h - 62)
        size -= size % XY_DIVS
        gx, gy = x0 + (w - size) / 2, y0 + 26
        d = size / XY_DIVS
        cx, cy = gx + size / 2, gy + size / 2

        for i in range(XY_DIVS + 1):
            mid = i == XY_DIVS // 2
            pygame.draw.line(surface, T.AXIS if mid else T.GRID, (round(gx + i * d), gy), (round(gx + i * d), gy + size))
            pygame.draw.line(surface, T.AXIS if mid else T.GRID, (gx, round(gy + i * d)), (gx + size, round(gy + i * d)))
        for i in range(XY_DIVS * 5):
            k = i * d / 5
            pygame.draw.line(surface, T.AXIS, (gx + k, cy - 3), (gx + k, cy + 3))
            pygame.draw.line(surface, T.AXIS, (cx - 3, gy + k), (cx + 3, gy + k))

        def label(text, x, y, colour, anchor):
            s = small.render(text, True, colour)
            surface.blit(s, s.get_rect(**{anchor: (round(x), round(y))}))

        c1, c2 = self.channel("1"), self.channel("2")
        for i in range(0, XY_DIVS + 1, 2):  # volts at every other line, as in Y-T
            label(f"{(i - XY_DIVS / 2 - c1['pos']) * c1['vdiv']:+.3g}", gx + i * d, gy + size + 6, COLOURS["1"], "midtop")
            label(f"{(XY_DIVS / 2 - i - c2['pos']) * c2['vdiv']:+.3g}", gx - 6, gy + i * d, COLOURS["2"], "midright")
        label(f"x  CH1 ({format_volts(c1['vdiv'])}/div)", cx, gy + size + 22, COLOURS["1"], "midtop")
        label(f"y  CH2 ({format_volts(c2['vdiv'])}/div)", gx - 6, y0 + 2, COLOURS["2"], "topleft")

        xy = self.xy_trace(now, 3000)
        if xy is None:
            return
        surface.set_clip(pygame.Rect(gx, gy, size + 1, size + 1))
        xs = np.clip(cx + xy[0] * d, gx - 10, gx + size + 10)
        ys = np.clip(cy - xy[1] * d, gy - 10, gy + size + 10)
        pts = list(zip(xs, ys))
        # Oldest pieces dimmest, like the afterglow on a phosphor screen.
        n = len(pts)
        for k in range(XY_FADE):
            a, b = k * n // XY_FADE, min(n, (k + 1) * n // XY_FADE + 1)
            f = 0.2 + 0.8 * (k + 1) / XY_FADE
            colour = tuple(round(T.BG[j] + (T.ACCENT[j] - T.BG[j]) * f) for j in range(3))
            pygame.draw.lines(surface, colour, False, pts[a:b], 2)
        surface.set_clip(None)

    def readouts(self, now=0.0):
        """Side panel readouts: one (channel, signal, [(label, value), ...]) per channel."""
        out = []
        for ch in CHANNELS:
            c = self.channel(ch)
            rows = []
            if c["wave"] in AUDIO:
                got = self.audio(c["wave"])
                if got is None:
                    rows.append(("status", "no mic" if c["wave"] == "Live mic" else "no song"))
                else:
                    import live
                    x, rate = got
                    pitch = live.detect_pitch(x[-int(0.13 * rate):], rate)
                    tr = self.trace(ch, now, 400)
                    rows += [("pitch", f"~{format_freq(pitch)}" if pitch else "--"),
                             ("vpp", format_volts(float(np.ptp(tr[1])) if tr is not None else 0.0)),
                             ("gain", f"{c['amp']:g}x")]
            elif c["wave"] != "Off":
                rows += [("freq", format_freq(c["freq"])),
                         ("period", format_time(1 / c["freq"])),
                         ("vpp", format_volts(2 * c["amp"]))]
                if c["phase"]:
                    rows.append(("phase", f"{c['phase'] * 360:g}°"))
            if c["wave"] != "Off":
                rows += [("scale", f"{format_volts(c['vdiv'])}/div"),
                         ("offset", f"{c['pos']:+.1f} div")]
            out.append((ch, c["wave"], rows))
        return out
