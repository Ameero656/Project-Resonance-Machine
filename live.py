"""Live mode: listen to a microphone and put the plate in the mode that
pitch would excite.

  1. PitchTracker finds the dominant pitch of the latest audio
     (autocorrelation), and smooths it over about half a second so a held
     note holds a mode while speech mostly blurs.
  2. nearest_mode() picks the (n, m) whose tone, base * (n^2 + m^2) / 2, is
     closest to that pitch.
  3. resonance() says how close the pitch is to that mode. A real plate
     barely moves between its resonances, so the app shakes the sand less
     the further off-pitch you are, and not at all in silence.

Mic needs the `sounddevice` package (and PortAudio: on Raspberry Pi OS,
`sudo apt install libportaudio2`). Without it, or without a microphone,
Mic.available is False and live mode says "no mic".
"""

from collections import deque

import numpy as np

RATE = 16000
WINDOW = 2048          # samples analysed per estimate (~0.13 s)
FMIN, FMAX = 60.0, 2000.0
SILENCE_RMS = 0.01     # quieter than this counts as silence
VOICING = 0.5          # autocorrelation peak needed to call it a pitch
SMOOTH = 5             # estimates in the median (~0.17 s at 30 fps)
RESONANCE_CENTS = 60   # how far off-pitch the plate still responds well


def detect_pitch(x, rate=RATE):
    """Dominant pitch of a block of samples in Hz, or None if silent/unpitched."""
    x = np.asarray(x, float)
    x = x - x.mean()
    if np.sqrt(np.mean(x * x)) < SILENCE_RMS:
        return None
    n = len(x)
    spec = np.fft.rfft(x * np.hanning(n), 2 * n)
    r = np.fft.irfft(spec * np.conj(spec))[:n]
    if r[0] <= 0:
        return None
    r = r / r[0]
    lo, hi = int(rate / FMAX), min(n - 2, int(rate / FMIN))
    # Skip the central peak: a low note stays correlated with itself for many
    # samples, so start looking only once the autocorrelation has gone negative.
    below = np.nonzero(r[lo:hi] < 0)[0]
    if len(below) == 0:
        return None
    lo += int(below[0])
    seg = r[lo:hi]
    best = seg.max()
    if best < VOICING:
        return None
    # The first peak close to the best one: avoids picking a multiple of the
    # true period (an octave too low).
    lag = lo + int(np.argmax(seg >= 0.9 * best))
    while lag + 2 < hi and r[lag + 1] > r[lag]:
        lag += 1
    a, b, c = r[lag - 1], r[lag], r[lag + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
    shift = float(np.clip(shift, -1.0, 1.0))
    return rate / (lag + shift)


def candidate_modes(free=True, top=12):
    """One (n, m) per distinct n^2 + m^2, sorted by frequency.

    Several pairs share a frequency; prefer n != m (n == m cancels when
    Mix is -1) and then the most two-dimensional pair (largest n).
    """
    best = {}
    start = 0 if free else 1
    for n in range(start, top + 1):
        for m in range(max(n, 1), top + 1):
            k = n * n + m * m
            key = (n != m, n)
            if k not in best or key > best[k][0]:
                best[k] = (key, (n, m))
    return [best[k][1] for k in sorted(best)]


def nearest_mode(freq, base, free=True):
    """The (n, m) whose tone is closest to freq (in musical distance)."""
    modes = candidate_modes(free)
    target = np.log(2.0 * freq / base)
    return min(modes, key=lambda nm: abs(np.log(nm[0] ** 2 + nm[1] ** 2) - target))


def resonance(freq, n, m, base):
    """1 exactly on the mode's pitch, falling towards 0 as you go off it."""
    cents = 1200 * np.log2(freq / (base * (n * n + m * m) / 2.0))
    return float(np.exp(-(cents / RESONANCE_CENTS) ** 2) * 0.8 + 0.2)


class PitchTracker:
    def __init__(self, smooth=SMOOTH):
        self.recent = deque(maxlen=smooth)

    def update(self, samples, rate=RATE):
        """Feed the latest window; return the smoothed pitch or None."""
        self.recent.append(detect_pitch(samples, rate))
        voiced = [f for f in self.recent if f is not None]
        # Need the note held for at least half the window to count.
        if len(voiced) * 2 < len(self.recent):
            return None
        return float(np.median(voiced))


HISTORY = RATE   # samples kept (1 s): the scope's slowest sweep is 0.5 s


class Mic:
    """Keeps the most recent second from the default microphone."""

    def __init__(self):
        self.available = False
        self.buffer = np.zeros(HISTORY, np.float32)
        try:
            import sounddevice
            self.stream = sounddevice.InputStream(
                samplerate=RATE, channels=1, dtype="float32", callback=self._callback)
            self.stream.start()
        except Exception:
            return  # no sounddevice package, no PortAudio, or no microphone
        self.available = True

    def _callback(self, indata, frames, time, status):
        data = indata[:, 0]
        if len(data) >= HISTORY:
            self.buffer = data[-HISTORY:].copy()
        else:
            self.buffer = np.concatenate([self.buffer[len(data):], data])

    def read(self):
        """The latest WINDOW samples, for pitch tracking."""
        return self.buffer[-WINDOW:].copy()

    def recent(self):
        """The whole history, for the oscilloscope."""
        return self.buffer.copy()

    def close(self):
        if self.available:
            self.stream.stop()
            self.stream.close()
            self.available = False
