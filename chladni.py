"""Chladni plate equation for a square plate, vectorised with NumPy.

Fixed edges (the form in the project sketch):
    C(x, y) = sin(n*pi*x) * sin(m*pi*y)  +  mix * sin(m*pi*x) * sin(n*pi*y)
Free edges (a real plate bolted at its centre):
    C(x, y) = cos(n*pi*x) * cos(m*pi*y)  +  mix * cos(m*pi*x) * cos(n*pi*y)

mix = +1 or -1 gives the classic +/- forms. Values in between blend the two
standing waves unequally, which real plates do too: it is what bends straight
nodal lines into the wavy ones seen in photographs of Chladni figures.

x and y are in [0, 1]. Sand collects where C = 0 (the nodal lines) because the
plate is still there; everywhere else it gets shaken off.

With fixed edges the whole border is a nodal line in every mode, so it is a
permanent trap: each mode change strands a few more grains there and the
sand slowly ends up around the rim. With free edges the border vibrates like
the rest of the plate, which is how a physical Chladni plate behaves.
"""

import numpy as np

PI = np.pi


def field(x, y, n, m, mix=1, free=True):
    """Plate displacement C(x, y) for vibration mode (n, m), mix in [-1, 1]."""
    f = np.cos if free else np.sin
    return f(n * PI * x) * f(m * PI * y) + mix * f(m * PI * x) * f(n * PI * y)


def gradient(x, y, n, m, mix=1, free=True):
    """Analytic partial derivatives (dC/dx, dC/dy)."""
    if free:
        f, df = np.cos, lambda a: -np.sin(a)
    else:
        f, df = np.sin, np.cos
    dx = (n * PI * df(n * PI * x) * f(m * PI * y)
          + mix * m * PI * df(m * PI * x) * f(n * PI * y))
    dy = (m * PI * f(n * PI * x) * df(m * PI * y)
          + mix * n * PI * f(m * PI * x) * df(n * PI * y))
    return dx, dy


def grid(size, n, m, mix=1, free=True):
    """C sampled on a size x size grid, indexed [x, y] to match pygame.surfarray."""
    s = np.linspace(0.0, 1.0, size)
    x, y = np.meshgrid(s, s, indexing="ij")
    return field(x, y, n, m, mix, free)


def is_degenerate(n, m, mix, free=True):
    """True when the plate has no nodal lines to draw.

    n == m with mix = -1 cancels to C = 0 everywhere; with fixed edges a 0 mode
    makes sin(0) = 0 everywhere; with free edges (0, 0) is a constant.
    """
    if n == m and mix <= -1:
        return True
    if not free:
        return n == 0 or m == 0
    return n == 0 and m == 0


def relative_frequency(n, m):
    """Resonant frequency of a square plate scales with n^2 + m^2.

    Returned relative to the (1, 1) mode, so the display can show how the
    driving frequency would change between modes once real hardware is
    calibrated.
    """
    return (n * n + m * m) / 2.0
