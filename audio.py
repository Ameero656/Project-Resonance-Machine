"""The mode tone: a sine at the frequency that would drive the current mode.

On a square plate the resonant frequency scales with n^2 + m^2, so
f = base * (n^2 + m^2) / 2, which puts (1, 1) at the base pitch (a setting,
80 Hz by default). When the mode changes the pitch glides to the new note.

Audio is generated with NumPy in short chunks and queued on a pygame mixer
channel. If there is no sound device, Synth.available is False and nothing
else changes.
"""

import numpy as np

DEFAULT_BASE_HZ = 80.0
MAX_HZ = 6000.0
CHUNK_S = 0.06   # length of each queued chunk
GLIDE = 0.25     # fraction of the way to the new pitch per chunk


def mode_frequency(n, m, base=DEFAULT_BASE_HZ):
    return float(np.clip(base * (n * n + m * m) / 2.0, 20.0, MAX_HZ))


class Voice:
    """Pure signal generation, so it can be tested and recorded without a sound card."""

    def __init__(self, rate=44100):
        self.rate = rate
        self.freq = None
        self.phase = 0.0
        self.amp = 0.0

    def chunk(self, target_freq, level, seconds=CHUNK_S):
        """Next block of float samples in [-1, 1].

        target_freq None means silence (the plate has no mode). level is the
        target loudness 0..1; pitch and loudness ramp across the block so
        there are no clicks.
        """
        n = max(1, int(self.rate * seconds))
        if target_freq is None:
            level = 0.0
            target_freq = self.freq or DEFAULT_BASE_HZ
        if self.freq is None:
            self.freq = target_freq
        f0 = self.freq
        f1 = f0 + (target_freq - f0) * GLIDE
        # Glide in log space so big jumps sound like a smooth slide.
        freqs = np.exp(np.linspace(np.log(f0), np.log(f1), n, endpoint=False))
        phase = self.phase + np.cumsum(2 * np.pi * freqs / self.rate)
        self.phase = float(phase[-1] % (2 * np.pi))
        self.freq = f1
        amp = np.linspace(self.amp, level, n, endpoint=False)
        self.amp = level
        return np.clip(0.5 * amp * np.sin(phase), -1.0, 1.0)


class Synth:
    """Feeds a Voice to the sound card."""

    def __init__(self, enabled=True):
        self.available = False
        self.voice = None
        if not enabled:
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            rate, _, channels = pygame.mixer.get_init()
            self.channel = pygame.mixer.Channel(0)
        except Exception:
            return  # no sound device: run silently
        self.pygame = pygame
        self.channels = channels
        self.voice = Voice(rate)
        self.available = True

    def update(self, freq, level):
        """Call once a frame; tops up the queue when it runs low."""
        if not self.available or self.channel.get_queue() is not None:
            return
        pcm = (self.voice.chunk(freq, level) * 32767).astype(np.int16)
        if self.channels > 1:
            pcm = np.repeat(pcm, self.channels)
        sound = self.pygame.mixer.Sound(buffer=pcm.tobytes())
        if self.channel.get_busy():
            self.channel.queue(sound)
        else:
            self.channel.play(sound)
