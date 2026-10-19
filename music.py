"""Music mode: play a folder of songs and drive the plate from them.

Songs (mp3, ogg, wav, flac) are read from the music/ folder next to the app.
While a song plays, its strongest note is found about 30 times a second and
the plate switches to the mode whose tone is closest, the same way live mode
follows the microphone. How hard the sand is shaken follows the song's
loudness and how close the note is to that mode, so quiet passages and
breaks freeze the pattern. Songs play one after another and loop.

Music is mostly chords, so instead of the autocorrelation used for a single
voice this uses a harmonic product spectrum, which picks out the note whose
harmonics line up best.
"""

import os

import numpy as np

import live

EXTENSIONS = (".mp3", ".ogg", ".wav", ".flac")
WINDOW_S = 0.19        # audio analysed per note estimate, centred on the playhead
BEAT_S = 0.05          # audio used for loudness and beats: short, so the sand moves with the rhythm
FMIN, FMAX = 60.0, 2000.0
HARMONICS = 3
SMOOTH = 3             # estimates in the median (~0.1 s at 30 fps): tight sync over steadiness
LOUD_RMS = 0.12        # RMS that counts as full drive (at 50% Sensitivity)
GATE = 0.1             # notes quieter than this fraction of full loudness are ignored


def list_tracks(folder):
    try:
        names = sorted(os.listdir(folder), key=str.lower)
    except OSError:
        return []
    return [os.path.join(folder, n) for n in names if n.lower().endswith(EXTENSIONS)]


def dominant_note(x, rate):
    """Strongest note in a block of samples, in Hz, or None if silent."""
    x = np.asarray(x, float)
    if len(x) < 256 or np.sqrt(np.mean(x * x)) < live.SILENCE_RMS:
        return None
    size = 1 << int(np.ceil(np.log2(len(x) * 2)))
    mag = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), size))
    hps = mag.copy()
    for h in range(2, HARMONICS + 1):
        down = mag[::h]
        hps[:len(down)] *= down
        hps[len(down):] = 0
    lo, hi = int(FMIN * size / rate), int(FMAX * size / rate)
    k = lo + int(np.argmax(hps[lo:hi]))
    if hps[k] <= 0:
        return None
    a, b, c = np.log(hps[k - 1:k + 2] + 1e-12)
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
    return (k + float(np.clip(shift, -1, 1))) * rate / size


class Song:
    """A decoded song: plays through pygame and exposes its samples for analysis."""

    def __init__(self, path, pygame):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.pygame = pygame
        self.sound = pygame.mixer.Sound(path)
        rate, _, channels = pygame.mixer.get_init()
        a = pygame.sndarray.array(self.sound).astype(np.float32) / 32768.0
        self.samples = a.mean(axis=1) if a.ndim == 2 else a
        self.rate = rate
        self.length = len(self.samples) / rate
        self.started = None

    def play(self, channel):
        channel.play(self.sound)
        self.started = self.pygame.time.get_ticks()

    def position(self):
        return 0.0 if self.started is None else (self.pygame.time.get_ticks() - self.started) / 1000.0

    def window(self, t, seconds=WINDOW_S):
        """`seconds` of audio centred on time t. The whole song is in memory, so
        the analysis can look a little ahead and line up with what you hear."""
        half = int(seconds * self.rate / 2)
        mid = int(t * self.rate)
        return self.samples[max(0, mid - half):max(0, mid + half)]


class NoteTracker:
    def __init__(self, smooth=SMOOTH):
        from collections import deque
        self.recent = deque(maxlen=smooth)

    def update(self, x, rate):
        self.recent.append(dominant_note(x, rate))
        notes = [f for f in self.recent if f is not None]
        if len(notes) * 2 < len(self.recent):
            return None
        # Median in log space: robust to the odd octave slip.
        return float(np.exp(np.median(np.log(notes))))


def full_rms(sensitivity=0.5):
    """RMS that counts as full loudness: 50% sensitivity is LOUD_RMS, 0% needs
    10x louder, 100% only a tenth as loud."""
    return LOUD_RMS * 10 ** ((0.5 - sensitivity) * 2)


def loudness_drive(x, sensitivity=0.5):
    """Loudness of x from 0 to 1, relative to the sensitivity's full level."""
    if len(x) == 0:
        return 0.0
    return float(np.clip(np.sqrt(np.mean(np.square(x))) / full_rms(sensitivity), 0.0, 1.0))


def format_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


class BeatDetector:
    """Spots onsets (drum hits, new notes) as a jump in short-window energy."""

    def __init__(self, history=15, jump=1.6, gap=4):
        from collections import deque
        self.recent = deque(maxlen=history)
        self.jump, self.gap = jump, gap
        self.since = gap

    def update(self, x):
        """Feed the latest short window; True on a beat."""
        energy = float(np.mean(np.square(x))) if len(x) else 0.0
        before = np.mean(self.recent) if self.recent else energy
        self.recent.append(energy)
        self.since += 1
        loud = energy > (live.SILENCE_RMS * 2) ** 2
        if loud and self.since >= self.gap and energy > self.jump * before:
            self.since = 0
            return True
        return False
