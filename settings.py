"""General settings, shared by both displays (sound, live mic, ...) plus
which display is showing. Saved to settings.json next to the app so they
survive a restart. Unlike plate parameters they aren't part of program
states."""

import json
import os

KEYS = ("Display", "Sound", "Volume", "Base pitch", "Live mic", "Sensitivity", "Reactivity", "Theme")


def load(path, params):
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return
    for p in params:
        if p.name in KEYS and isinstance(saved.get(p.name), (int, float)):
            p.value = min(p.hi, max(p.lo, saved[p.name]))


def save(path, params):
    data = {p.name: p.value for p in params if p.name in KEYS}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
