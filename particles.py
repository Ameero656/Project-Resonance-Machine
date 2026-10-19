"""2D sand particle system driven by the Chladni field.

Each frame every grain gets:
  * a random kick proportional to how hard the plate moves under it (|C|),
    so grains on antinodes bounce around and grains on nodal lines sit still;
  * a small drift down the slope of C^2, which pulls grains onto nodal lines;
  * background jitter controlled by `uniformity` (0 = chaotic, 1 = orderly);
  * `adherence` scales the drift, i.e. how strongly grains follow the pattern.

`powder=True` models very fine, light dust instead (Faraday's lycopodium
powder): air currents over the vibrating plate carry it to the antinodes, so
it drifts *up* the slope of C^2 and only wanders where the plate is still.

`mass` scales how hard grains are thrown about (1/mass) more than how
strongly they are pulled onto the lines (1/sqrt(mass)), so heavy grains
settle more slowly but into thinner, sharper lines, and light grains make
wide, fuzzy ones.
"""

import numpy as np

import chladni

PI2 = np.pi ** 2
# Crowding: the plate is split into CROWD_CELLS^2 cells; a cell holding more than
# CROWD_FACTOR times the average (and at least CROWD_MIN grains) is overcrowded.
CROWD_CELLS = 64
CROWD_FACTOR = 10.0
CROWD_MIN = 12
CROWD_PUSH = 0.01


class SandPlate:
    def __init__(self, count=5000, n=3, m=5, mix=1, uniformity=0.7, free=True,
                 mass=1.0, powder=False, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n, self.m, self.mix = n, m, mix
        self.free = free
        self.uniformity = uniformity
        # How strongly grains are pulled onto their lines (1 = normal): high snaps
        # the sand into thin lines fast, low lets it drift there lazily.
        self.adherence = 1.0
        self.mass, self.powder = mass, powder
        # How hard the plate is being driven, 0..1. Live mode lowers it when the
        # mic pitch is off-resonance, and to 0 in silence (the sand stays put).
        self.drive = 1.0
        # A short burst of extra shaking (set by shake()), decaying each step: it
        # knocks loose sand that settled on the old nodal lines when the mode changes.
        self.agitation = 0.0
        # Crowded spots push their grains outward, so sand can't pile up without limit.
        self.crowding = True
        self.pos = self.rng.random((count, 2))

    @property
    def count(self):
        return len(self.pos)

    def set_count(self, count):
        """Grow by scattering new grains, or shrink by dropping grains at random."""
        count = max(0, int(count))
        if count > self.count:
            self.pos = np.vstack([self.pos, self.rng.random((count - self.count, 2))])
        elif count < self.count:
            keep = self.rng.choice(self.count, count, replace=False)
            self.pos = self.pos[keep]

    def shake(self, amount=1.0):
        self.agitation = max(self.agitation, amount)

    def crowd_push(self):
        """Push grains out of overcrowded cells, down the slope of the crowding.

        A grain held on a nodal line slides along it (the drift keeps it on the
        line), so blobs where many modes share a node flow back out into lines
        instead of growing."""
        cells = CROWD_CELLS
        ij = np.minimum((self.pos * cells).astype(np.int32), cells - 1)
        counts = np.bincount(ij[:, 0] * cells + ij[:, 1], minlength=cells * cells)
        cap = max(CROWD_MIN, CROWD_FACTOR * self.count / (cells * cells))
        excess = np.maximum(counts.reshape(cells, cells) / cap - 1.0, 0.0)
        if not excess.any():
            return 0.0
        # Blur the excess a little so its slope reaches the edges of a blob.
        e = np.pad(excess, 1, mode="edge")
        e = (e[:-2, 1:-1] + e[2:, 1:-1] + e[1:-1, :-2] + e[1:-1, 2:] + 2 * e[1:-1, 1:-1]) / 6
        gx, gy = np.gradient(e)
        i, j = ij[:, 0], ij[:, 1]
        push = -np.column_stack([gx[i, j], gy[i, j]]) * CROWD_PUSH
        return np.clip(push, -0.01, 0.01)

    def scatter(self):
        """Throw all the sand back onto the plate at random."""
        self.pos = self.rng.random((self.count, 2))

    def step(self, dt=1.0):
        if self.count == 0 or self.drive <= 0:
            return
        x, y = self.pos[:, 0], self.pos[:, 1]
        n, m, s = self.n, self.m, self.mix

        if chladni.is_degenerate(n, m, s, self.free):
            c = np.zeros_like(x)
            gx = gy = c
        else:
            c = chladni.field(x, y, n, m, s, self.free)
            gx, gy = chladni.gradient(x, y, n, m, s, self.free)

        # |C| peaks at 2; normalise the amplitude to roughly 0..1.
        amp = np.abs(c) * 0.5
        chaos = 1.0 - self.uniformity

        throw = 1.0 / self.mass
        pull = 1.0 / np.sqrt(self.mass)

        # Shake. Sand is thrown hardest on antinodes and sits still on nodes;
        # powder is held at the antinodes and only wanders where the plate is still.
        # (Powder also keeps a little motion everywhere, so it forms soft
        # heaps around each antinode rather than single dots.)
        shaken = 0.15 + (1.0 - amp) * 0.5 if self.powder else amp
        kick = 0.012 * throw * shaken[:, None] * self.rng.standard_normal((self.count, 2))

        # Drift along grad(C^2 / 2) = C * grad C: down it for sand (to the
        # nodes), up it for powder (to the antinodes). Scaled so higher modes
        # (steeper gradients) don't move faster than low ones.
        k = PI2 * (n * n + m * m)
        direction = 1.0 if self.powder else -1.0
        drift = direction * 0.35 * pull * self.adherence * (c / k)[:, None] * np.column_stack([gx, gy])

        # Background jitter, independent of the plate: this is what "uniformity" tunes.
        # The cubic term leaves high uniformity as before but makes low settings
        # properly chaotic, enough to show through music's jolts and mode changes.
        noise = (0.004 * chaos + 0.012 * chaos ** 3) * throw
        background = noise * self.rng.standard_normal((self.count, 2))
        jitter = 0.0

        # Mode-change jolt: plain jitter that dies away over about half a second.
        if self.agitation > 0.01:
            jitter = jitter + 0.03 * self.agitation * self.rng.standard_normal((self.count, 2))
            self.agitation *= 0.88
        else:
            self.agitation = 0.0
        if self.crowding:
            jitter = jitter + self.crowd_push()

        # The background shaking follows the square root of the drive: quiet passages
        # calm it much less than the rest, but silence (drive -> 0) still stops it.
        move = (kick + drift + jitter) * (dt * self.drive) + background * (dt * np.sqrt(self.drive))
        np.clip(move, -0.02, 0.02, out=move)
        self.pos += move

        # Reflect off the plate edges.
        self.pos = np.abs(self.pos)
        self.pos = 1.0 - np.abs(1.0 - self.pos)
        np.clip(self.pos, 0.0, 1.0, out=self.pos)

