"""Programs: a list of parameter states the plate cycles through on its own.

A program runs S1 -> S2 -> ... -> Sn -> S1, holding each state for its own
number of seconds. Switching state just changes the mode, so the sand
migrates to the new nodal lines the way it would on a real plate.

Programs are saved as JSON (programs.json next to the app) so they survive
a restart.
"""

import json
import os

from params import CHANNELS

# Parameters a state remembers. Particle counts are left out on purpose: they
# are how much sand is on the plate, not how it vibrates.
STATE_KEYS = ("n", "m", "Mix", "Edges", "Uniformity", "Adherence") + tuple(
    f"{ch} {k}" for ch in CHANNELS for k in ("Show", "Type", "Mass", "Colour", "Size"))
# Optional keys a state may also set. Programs you build don't save these, so
# they keep whatever particle counts are on the plate, but a built-in program
# can use them to set up the plate the way it wants.
OPTIONAL_KEYS = tuple(f"{ch} Particles" for ch in CHANNELS)
# Per-state settings that aren't plate parameters.
DEFAULT_HOLD = 8
# Reset = 1 throws the sand back on the plate when the program enters the state,
# so the new pattern forms from scratch instead of from the previous one.
DEFAULT_RESET = 0


def new_state(values, hold=DEFAULT_HOLD, reset=DEFAULT_RESET):
    state = {k: values[k] for k in STATE_KEYS if k in values}
    state["Hold"] = hold
    state["Reset"] = reset
    return state


def _state(n, m, mix, edges=1, hold=8, reset=0):
    return {"n": n, "m": m, "Mix": mix, "Edges": edges, "Uniformity": 0.7,
            "Hold": hold, "Reset": reset}


# The 40 reference figures from the project sheet, in reading order (row by
# row, left to right), each matched to the closest mode of the equation.
# Rings (1,3  2,3  5,4 and 8,4) only appear in the fixed-edge form, so
# those use Edges = 0. Fixed edges leave sand on the rim, so each of those
# states, and the one after, starts with a reset. A few figures (6,2  8,1
# 8,2  8,4) come from real plates and are only approximated by this equation.
REFERENCE = [
    # (n, m, mix, edges)
    (1, 1, 1.0, 1), (0, 2, -1.0, 1), (1, 3, 1.0, 1), (1, 2, 0.0, 1), (1, 2, -0.3, 1),
    (0, 3, 0.0, 1), (0, 2, 0.5, 1), (1, 4, 1.0, 1), (2, 2, 1.0, 1), (2, 3, 0.3, 1),
    (1, 3, -1.0, 1), (1, 3, 1.0, 1), (2, 3, 0.0, 1), (3, 2, -1.0, 1), (2, 3, -1.0, 1),
    (0, 4, -1.0, 1), (0, 4, -0.5, 1), (0, 4, 1.0, 1), (0, 4, 0.6, 1), (1, 4, 0.0, 1),
    (1, 4, 1.0, 1), (3, 3, 1.0, 1), (2, 4, -1.0, 1), (3, 5, 1.0, 1), (2, 5, -0.7, 1),
    (1, 5, -1.0, 1), (1, 5, -0.5, 1), (3, 4, 0.0, 1), (3, 4, -1.0, 1), (4, 3, -1.0, 1),
    (2, 5, 0.0, 1), (2, 5, 1.0, 1), (4, 4, 1.0, 1), (3, 5, -1.0, 1), (3, 5, 0.9, 1),
    (0, 6, -0.6, 1), (1, 6, 0.2, 1), (2, 6, -0.2, 1), (1, 5, 1.0, 1), (2, 6, 0.0, 1),
]


def reference_tour(hold=2):
    # Plain sand only, so the figures read like the sheet: channel A at double
    # the default count with 2 px dots, channel B off.
    look = {"A Show": 1, "A Particles": 12000, "A Type": 0, "A Mass": 1.0, "A Size": 1,
            "B Show": 0, "B Particles": 0}
    states = []
    for i, (n, m, mix, edges) in enumerate(REFERENCE):
        after_fixed = i > 0 and REFERENCE[i - 1][3] == 0
        reset = int(i == 0 or edges == 0 or after_fixed)
        states.append(dict(_state(n, m, mix, edges, hold, 1), **look))
    return {"name": "Reference tour", "states": states}


def default_programs():
    tour = [_state(3, 5, 1), _state(2, 7, -1), _state(1, 4, 1), _state(4, 6, -1), _state(5, 8, 1)]
    return [{"name": "Tour", "states": tour}, reference_tour()]


def _migrate(programs):
    """Bring states saved by older versions up to date."""
    for prog in programs:
        for state in prog.get("states", []):
            if "Sign" in state:  # Sign (+1 / -1) became Mix (-1 .. +1)
                state.setdefault("Mix", float(state.pop("Sign")))
            state.setdefault("Reset", DEFAULT_RESET)
            for ch in CHANNELS:  # channels could always be seen before Show existed
                state.setdefault(f"{ch} Show", 1)
        if prog.get("name") == "Reference tour" and all(
                s.get("A Size") == 2 for s in prog.get("states", [])):
            for s in prog["states"]:  # the built-in tour went back to 1 px dots
                s["A Size"] = 1
    return programs


class Store:
    def __init__(self, path):
        self.path = path
        self.programs, self.builtins_seen = self._load()
        # Built-in programs added in later versions appear once; deleting one
        # afterwards keeps it deleted.
        for prog in default_programs():
            if prog["name"] not in self.builtins_seen:
                self.builtins_seen.append(prog["name"])
                if all(p["name"] != prog["name"] for p in self.programs):
                    self.programs.append(prog)

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            programs = data["programs"]
            assert isinstance(programs, list)
            # Files from before built-ins were tracked had only "Tour".
            seen = data.get("builtins_seen", ["Tour"])
            return _migrate(programs), list(seen)
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            return [], []

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"programs": self.programs, "builtins_seen": self.builtins_seen}, f, indent=2)
        os.replace(tmp, self.path)  # atomic, so a power cut can't leave half a file

    def create(self, values):
        taken = {p["name"] for p in self.programs}
        i = len(self.programs) + 1
        while f"Program {i}" in taken:
            i += 1
        prog = {"name": f"Program {i}", "states": [new_state(values)]}
        self.programs.append(prog)
        return prog

    def delete(self, prog):
        self.programs.remove(prog)


class Sequencer:
    """Steps through a program's states in a loop."""

    def __init__(self):
        self.program = None
        self.index = 0
        self.elapsed = 0.0

    @property
    def running(self):
        return self.program is not None

    @property
    def current(self):
        return self.program["states"][self.index]

    @property
    def remaining(self):
        return max(0.0, self.current["Hold"] - self.elapsed)

    def start(self, program):
        """Start at S1 and return it, or None if the program has no states."""
        if not program["states"]:
            return None
        self.program, self.index, self.elapsed = program, 0, 0.0
        return self.current

    def stop(self):
        self.program = None

    def tick(self, dt):
        """Advance the clock; return the new state when it changes, else None."""
        if not self.running:
            return None
        states = self.program["states"]
        if not states:  # every state was deleted while running
            self.stop()
            return None
        self.index %= len(states)
        self.elapsed += dt
        if self.elapsed < self.current["Hold"]:
            return None
        self.elapsed = 0.0
        self.index = (self.index + 1) % len(states)
        return self.current
