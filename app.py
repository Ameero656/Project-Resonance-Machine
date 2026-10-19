"""Project Resonance: Chladni plate sand simulation.

Layout matches the 4.3" DSI screen (800x480): the 480x480 plate sits in the
middle, parameters and program menus on the left, mode, frequency and
program status on the right.

    python3 app.py                 # windowed, keyboard controls
    python3 app.py --fullscreen    # on the Pi
    python3 app.py --screenshot out.png --frames 300   # render headless and save
    python3 app.py --run Tour      # start a saved program straight away
"""

import argparse
import os

import numpy as np
import pygame

import audio
import chladni
import controls
import live
import music
import program
import scope
import settings
import ui
from ui import T
from menu import Menu, Row
from params import CHANNELS, COLOURS, TYPES, default_params, format_step, format_value, make_param
from particles import SandPlate

SCREEN_W, SCREEN_H = 800, 480
PLATE = 480
PANEL_W = (SCREEN_W - PLATE) // 2
FPS = 30

PLATE_BG = (0, 0, 0)

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMS_FILE = os.path.join(HERE, "programs.json")
SETTINGS_FILE = os.path.join(HERE, "settings.json")
MUSIC_DIR = os.path.join(HERE, "music")
SETTLE_S = 0.1   # live/music: a new note must hold this long before the mode changes
BEAT_SHAKE = 0.6 # live/music: how hard a beat jolts the sand
ROW_H = 44
MENU_TOP = 64
FOOTER_H = 52
MENU_ROWS = (SCREEN_H - FOOTER_H - MENU_TOP) // ROW_H


def dot_offsets(size):
    """Pixel offsets that make a roughly round dot `size` pixels across."""
    c = (size - 1) / 2.0
    return [(dx - size // 2, dy - size // 2)
            for dx in range(size) for dy in range(size)
            if (dx - c) ** 2 + (dy - c) ** 2 <= (c + 0.5) ** 2]


class App:
    def __init__(self, fullscreen=False, programs_file=PROGRAMS_FILE, sound=True,
                 settings_file=SETTINGS_FILE, mic=None, music_dir=MUSIC_DIR):
        """mic: an object with .available and .read() to use instead of the real
        microphone (for tests); None opens the real one when live mode starts."""
        # Only display + font here; audio.Synth opens the mixer and runs silently without a sound card.
        pygame.display.init()
        pygame.font.init()
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
        pygame.display.set_caption("Project Resonance")
        if fullscreen:
            pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        self.fonts = ui.Fonts(pygame)  # (the theme is applied once settings load)
        self.paint = ui.Painter(pygame, self.screen, self.fonts)

        self.params = default_params() + scope.default_params()
        self.scope = scope.Scope(self.params)
        self.scope.audio = self.scope_audio
        self.scope_time = 0.0
        self.state_params = {}
        self.dt = 1.0 / FPS
        self.pending_mode, self.pending_time = None, 0.0
        self.beats = music.BeatDetector()
        self.settings_file = settings_file
        settings.load(settings_file, self.params)
        ui.T.use(ui.THEME_NAMES[int(self.value("Theme"))])
        self.heatmap = False
        self.store = program.Store(programs_file)
        self.seq = program.Sequencer()
        self.confirm = None  # row awaiting a second press to delete
        self.menus = [Menu("Display", self.root_rows)]
        self.plates = {ch: SandPlate(seed=i + 1) for i, ch in enumerate(CHANNELS)}
        self.apply_params()
        self.plate_surface = pygame.Surface((PLATE, PLATE))
        self.encoders = controls.Encoders()
        self.synth = audio.Synth(enabled=sound)
        self.mic = mic
        self.tracker = live.PitchTracker()
        self.live_pitch = None
        self.music_dir = music_dir
        self.song = None          # the music.Song playing, if any
        self.playlist = []
        self.track = 0
        self.music_note = None
        self.music_error = None
        self.running = True

    # --- state ---------------------------------------------------------

    def value(self, name):
        return next(p.value for p in self.params if p.name == name)

    @property
    def plate(self):
        """Channel A; both channels always share the plate's mode."""
        return self.plates[CHANNELS[0]]

    def apply_params(self):
        for ch, p in self.plates.items():
            p.set_count(self.value(f"{ch} Particles"))
            p.mass = self.value(f"{ch} Mass")
            p.powder = TYPES[int(self.value(f"{ch} Type"))] == "Powder"
            p.uniformity = self.value("Uniformity")
            p.adherence = self.value("Adherence")
            p.n, p.m, p.mix = int(self.value("n")), int(self.value("m")), self.value("Mix")
            p.free = bool(self.value("Edges"))
        self._heat = None  # field changed; rebuild heatmap lazily

    def param(self, name):
        return next(p for p in self.params if p.name == name)

    def apply_state(self, state):
        for k in program.STATE_KEYS + program.OPTIONAL_KEYS:
            if k in state:  # states saved before channels existed lack channel keys
                self.param(k).value = state[k]
        self.apply_params()

    def current_values(self):
        return {p.name: p.value for p in self.params}

    # --- menus ---------------------------------------------------------

    @property
    def menu(self):
        return self.menus[-1]

    def open(self, title, build):
        self.menus.append(Menu(title, build))

    def back(self):
        if len(self.menus) > 1:
            self.menus.pop()
            self.store.save()
        else:
            self.scatter()

    def confirmed(self, key):
        """Deleting takes two presses; the first only arms it."""
        if self.confirm == key:
            self.confirm = None
            return True
        self.confirm = key
        return False

    def scatter(self):
        for p in self.plates.values():
            p.scatter()

    def param_row(self, p, label=None):
        fmt = scope.format_value if p.name in scope.KEYS else format_value
        if not p.coarse:
            return Row(label or p.name, lambda: fmt(p.name, p.value),
                       adjust=lambda d: self.manual_adjust(p, d))
        # K1 press cycles the step size, coarse to fine (only on rows that open nothing).
        step = scope.format_step if p.name in scope.KEYS else format_step
        return Row(label or p.name, lambda: fmt(p.name, p.value),
                   adjust=lambda d: self.manual_adjust(p, d),
                   press=p.cycle_step, hint=lambda: step(p.name, p.step))

    @property
    def scope_on(self):
        return self.value("Display") == 1

    DISPLAYS = ("Resonance", "Oscilloscope")  # index = the Display setting

    def root_rows(self):
        """The two displays (opening one switches to it), then Settings, shared by both."""
        rows = []
        for i in (1, 0):
            name = self.DISPLAYS[i]
            rows.append(Row(name, press=lambda i=i: self.show(i),
                            badge=lambda i=i: "on" if self.value("Display") == i else ""))
        rows.append(Row("Settings", press=lambda: self.open("Settings", self.settings_rows),
                        badge=lambda: "mic" if self.live_on else ""))
        return rows

    def show(self, display):
        """Switch to a display and open its menu."""
        p = self.param("Display")
        if p.value != display:
            p.value = display
            settings.save(self.settings_file, self.params)
            self.apply_params()
        del self.menus[1:]
        build = self.oscilloscope_rows if display == 1 else self.resonance_rows
        self.open(self.DISPLAYS[display], build)

    def home(self):
        """Back to the current display's menu, to watch the plate."""
        del self.menus[2:]

    def oscilloscope_rows(self):
        rows = [self.param_row(self.param("Mode")), self.param_row(self.param("Time/div"))]
        for ch in scope.CHANNELS:
            rows.append(Row(f"Channel {ch}", lambda ch=ch: scope.WAVES[int(self.value(f"CH{ch} Wave"))],
                            press=lambda ch=ch: self.open(f"Channel {ch}", lambda: self.scope_rows(ch))))
        rows.append(self.music_row())  # a song to put on a channel set to Music
        return rows

    def scope_rows(self, ch):
        audio = scope.WAVES[int(self.value(f"CH{ch} Wave"))] in scope.AUDIO
        rows = []
        for p in self.params:
            if not p.name.startswith(f"CH{ch} "):
                continue
            key = p.name.split(" ", 1)[1]
            if audio and key in ("Frequency", "Phase"):
                continue  # the sound sets its own frequency and phase
            row = self.param_row(p, key)
            if key == "Wave":
                row.label = "Signal"
            if audio and key == "Amplitude":
                row.label, row.value = "Gain", lambda p=p: f"{p.value:g}x"
                row.hint = lambda p=p: f"±{p.step:g}x"
            rows.append(row)
        return rows

    def scope_audio(self, kind):
        """Recent samples for a scope channel showing real audio, newest = now."""
        if kind == "Live mic":
            if self.mic is None:
                self.mic = live.Mic()
            if not self.mic.available:
                return None
            x = self.mic.recent() if hasattr(self.mic, "recent") else self.mic.read()
            return x, live.RATE
        if kind == "Music" and self.song is not None:
            song = self.song
            end = int(min(song.position(), song.length) * song.rate)
            return song.samples[max(0, end - song.rate):end], song.rate
        return None

    SOURCE_KEYS = ("Uniformity", "Adherence", "n", "m", "Mix", "Edges")

    def source_rows(self):
        return [self.param_row(self.param(k)) for k in self.SOURCE_KEYS]

    def source_summary(self):
        edges = format_value("Edges", self.value("Edges"))
        return f"({self.value('n')},{self.value('m')}) {edges.lower()}"

    def resonance_rows(self):
        rows = [Row("Source", self.source_summary,
                    press=lambda: self.open("Source", self.source_rows))]
        for ch in CHANNELS:
            rows.append(Row(f"Channel {ch}", lambda ch=ch: self.channel_summary(ch),
                            press=lambda ch=ch: self.open(f"Channel {ch}", lambda: self.channel_rows(ch))))
        rows.append(Row("Programs", lambda: "running" if self.seq.running else "",
                        press=lambda: self.open("Programs", self.programs_rows)))
        rows.append(self.music_row())
        return rows

    def music_row(self):
        return Row("Music", lambda: self.song.name if self.song else "",
                   press=lambda: self.open("Music", self.music_rows))

    def settings_rows(self):
        rows = []
        for k in settings.KEYS:
            if k == "Display":
                continue  # chosen by opening Oscilloscope or Resonance
            row = self.param_row(self.param(k))
            if k == "Sound" and not self.synth.available:
                row.value = lambda: "no device"
            if k == "Live mic" and self.live_on and self.mic is not None and not self.mic.available:
                row.value = lambda: "no mic"
            rows.append(row)
        return rows

    @property
    def live_on(self):
        return bool(self.value("Live mic"))

    def channel_summary(self, ch):
        count = self.value(f"{ch} Particles")
        if not self.value(f"{ch} Show"):
            return "hidden"
        return f"{format_value('Type', self.value(f'{ch} Type'))} {count}" if count else "off"

    def channel_rows(self, ch):
        return [self.param_row(p, p.name.split(" ", 1)[1])
                for p in self.params if p.name.startswith(ch + " ")]

    def manual_adjust(self, p, clicks):
        p.nudge(clicks)
        if p.name in program.STATE_KEYS:
            self.seq.stop()  # turning a knob by hand takes over from the program
        if p.name in ("n", "m"):
            if self.live_on:
                self.set_live(False)  # ...and from the microphone
            self.stop_music()         # ...and from a song
        if p.name == "Live mic":
            self.set_live(bool(p.value))
        if p.name == "Theme":
            ui.T.use(ui.THEME_NAMES[int(p.value)])
            self._heat = None  # the heatmap is tinted by the theme
        if p.name in settings.KEYS:
            settings.save(self.settings_file, self.params)
        self.apply_params()

    def set_live(self, on):
        self.param("Live mic").value = int(on)
        if on:
            self.seq.stop()
            self.stop_music()
            if self.mic is None:
                self.mic = live.Mic()
        else:
            for plate in self.plates.values():
                plate.drive = 1.0
        self.live_pitch = None
        self.tracker = live.PitchTracker()
        settings.save(self.settings_file, self.params)

    def update_live(self):
        """Live mode: follow the microphone's pitch."""
        if not self.live_on:
            return
        if self.mic is None:
            self.mic = live.Mic()
        if not self.mic.available:
            return
        x = self.mic.read()
        self.live_pitch = pitch = self.tracker.update(x)
        self.follow_sound(pitch, x[-int(music.BEAT_S * live.RATE):])

    def follow_pitch(self, pitch, loudness=1.0):
        """Put the plate in the mode `pitch` would excite (shared by live and music)."""
        base = self.value("Base pitch")
        plate = self.plate
        if pitch is None:
            drive = plate.drive * 0.85  # silence: the plate stops, the sand stays put
            self.pending_mode = None
        else:
            mode = live.nearest_mode(pitch, base, plate.free)
            if mode == (plate.n, plate.m):
                self.pending_mode = None
            else:
                # Only change mode once the new note has held for SETTLE_S, so each
                # figure gets to form instead of the sand collecting where modes overlap.
                if self.pending_mode != mode:
                    self.pending_mode, self.pending_time = mode, 0.0
                self.pending_time += self.dt
                if self.pending_time >= SETTLE_S or plate.drive < 0.2:  # (or starting from quiet)
                    self.param("n").value, self.param("m").value = mode
                    self.apply_params()
                    self.pending_mode = None
                    for p in self.plates.values():
                        p.shake()
            drive = live.resonance(pitch, plate.n, plate.m, base) * loudness
        for p in self.plates.values():
            p.drive = drive

    # --- music -----------------------------------------------------------

    def music_rows(self):
        if not self.synth.available:
            return [Row("No sound device")]
        tracks = music.list_tracks(self.music_dir)
        rows = []
        if self.song:
            rows.append(Row("Stop", lambda: self.song.name if self.song else "", press=self.stop_music))
        if self.music_error:
            rows.append(Row("Couldn't play", lambda: self.music_error))
        for i, path in enumerate(tracks):
            name = os.path.splitext(os.path.basename(path))[0]
            playing = self.song is not None and self.song.path == path
            rows.append(Row(name[:18], lambda playing=playing: "playing" if playing else "",
                            press=lambda i=i, tracks=tracks: self.play_music(tracks, i)))
        if not tracks:
            rows.append(Row("No songs yet", lambda: "add to music/"))
        return rows

    def play_music(self, tracks, i, go_home=True):
        """Play tracks[i], then the rest of the folder in order, looping."""
        self.seq.stop()
        if self.live_on:
            self.set_live(False)
        self.stop_music()
        self.playlist, self.track = list(tracks), i
        self.music_error = None
        try:
            self.song = music.Song(tracks[i], pygame)
        except Exception:
            self.music_error = os.path.basename(tracks[i])[:18]
            return
        self.note_tracker = music.NoteTracker()
        self.song.play(pygame.mixer.Channel(1))
        if go_home:
            self.home()  # back to the display's menu to watch it

    def stop_music(self):
        if self.song is None:
            return
        pygame.mixer.Channel(1).stop()
        self.song = None
        self.music_note = None
        for p in self.plates.values():
            p.drive = 1.0

    def next_song(self):
        """At the end of a song, start the next one. True if it did."""
        song = self.song
        if song is not None and song.position() >= song.length:
            self.play_music(self.playlist, (self.track + 1) % len(self.playlist), go_home=False)
            return True
        return False

    def update_music(self):
        if self.song is None or self.next_song():
            return
        song = self.song
        t = song.position()
        self.music_note = note = self.note_tracker.update(song.window(t), song.rate)
        self.follow_sound(note, song.window(t, music.BEAT_S))

    def follow_sound(self, pitch, short):
        """Shared by music and live: `short` is the latest ~50 ms of audio.

        Sensitivity sets how loud a note must be to count (quieter ones are
        ignored); Reactivity sets how much the loudness and beats drive the
        shaking (0 = steady, 1 = follows the volume)."""
        level = music.loudness_drive(short, self.value("Sensitivity"))
        react = self.value("Reactivity")
        if level < music.GATE:
            pitch = None
        if self.beats.update(short) and level >= music.GATE and react > 0:
            for p in self.plates.values():
                p.shake(BEAT_SHAKE * react)
        self.follow_pitch(pitch, (1.0 - react) + react * level)

    def programs_rows(self):
        rows = []
        if self.seq.running:
            rows.append(Row("Stop", lambda: self.seq.program["name"], press=self.seq.stop))
        for prog in self.store.programs:
            rows.append(Row(prog["name"], lambda prog=prog: f"{len(prog['states'])} states",
                            press=lambda prog=prog: self.open_program(prog)))
        rows.append(Row("+ New program", press=self.create_program))
        return rows

    def create_program(self):
        prog = self.store.create(self.current_values())
        self.store.save()
        self.open_program(prog)

    def open_program(self, prog):
        self.open(prog["name"], lambda: self.program_rows(prog))

    def enter_state(self, state):
        """The program has moved to `state`."""
        self.apply_state(state)
        if state.get("Reset"):
            self.scatter()

    def run_program(self, prog):
        if self.live_on:
            self.set_live(False)
        self.stop_music()
        state = self.seq.start(prog)
        if state is not None:
            self.enter_state(state)
            self.store.save()
            self.home()  # back to the display's menu to watch it

    def program_rows(self, prog):
        rows = [Row("Run", press=lambda: self.run_program(prog))]
        for i, s in enumerate(prog["states"]):
            summary = f"({s['n']},{s['m']}) {s['Mix']:+.1f}"
            shown = "".join(ch for ch in CHANNELS if s.get(f"{ch} Show", 1)) or "--"
            badge = f"{shown} {s['Hold']:g}s" + (" R" if s.get("Reset") else "")
            rows.append(Row(f"State {i + 1}", lambda summary=summary: summary,
                            press=lambda i=i: self.open_state(prog, i), badge=lambda badge=badge: badge))
        rows.append(Row("+ Add state", press=lambda: self.add_state(prog)))
        armed = self.confirm == ("program", id(prog))
        rows.append(Row("Delete program", lambda: "press again" if armed else "",
                        press=lambda: self.delete_program(prog)))
        return rows

    def add_state(self, prog):
        # Start from what's on the plate now, so "set it up by hand, then add" works.
        prog["states"].append(program.new_state(self.current_values()))
        self.open_state(prog, len(prog["states"]) - 1)

    def delete_program(self, prog):
        if self.confirmed(("program", id(prog))):
            if self.seq.program is prog:
                self.seq.stop()
            self.store.delete(prog)
            self.back()

    def open_state(self, prog, i):
        self.seq.stop()  # preview the state being edited instead of the program
        values = self.current_values()
        for k in program.STATE_KEYS:
            prog["states"][i].setdefault(k, values[k])
        self.apply_state(prog["states"][i])
        self.open(f"{prog['name']} · S{i + 1}", lambda: self.state_rows(prog, i))

    def state_rows(self, prog, i):
        if i >= len(prog["states"]):
            return [Row("(deleted)")]
        state = prog["states"][i]
        rows = []
        optional = tuple(k for k in program.OPTIONAL_KEYS if k in state)
        for key in program.STATE_KEYS + optional + ("Hold", "Reset"):
            p = self.state_param(key)
            coarse = p.coarse
            rows.append(Row(key, lambda key=key: format_value(key, state[key]),
                            adjust=lambda d, key=key: self.edit_state(state, key, d),
                            press=p.cycle_step if coarse else None,
                            hint=(lambda p=p: format_step(p.name, p.step)) if coarse else None))
        armed = self.confirm == ("state", id(state))
        rows.append(Row("Delete state", lambda: "press again" if armed else "",
                        press=lambda: self.delete_state(prog, state)))
        return rows

    def state_param(self, key):
        """One Param per key for the state editor, so its step size is remembered."""
        if key not in self.state_params:
            self.state_params[key] = make_param(key, 0)
        return self.state_params[key]

    def edit_state(self, state, key, clicks):
        p = self.state_param(key)
        p.value = state[key]
        p.nudge(clicks)
        state[key] = p.value
        if key != "Hold":
            self.apply_state(state)

    def delete_state(self, prog, state):
        if self.confirmed(("state", id(state))):
            prog["states"].remove(state)
            self.back()

    # --- input ---------------------------------------------------------

    def handle(self, action, amount):
        if action not in (controls.PRESS, controls.HEATMAP):
            self.confirm = None
        if action == controls.QUIT:
            self.running = False
        elif action == controls.SELECT:
            self.menu.move(amount)
        elif action == controls.ADJUST:
            row = self.menu.current()
            if row.adjust:
                row.adjust(amount)
        elif action == controls.PRESS:
            row = self.menu.current()
            if row.press:
                row.press()
        elif action == controls.BACK:
            self.back()
        elif action == controls.SCATTER:
            self.scatter()
        elif action == controls.HEATMAP:
            self.heatmap = not self.heatmap

    # --- drawing -------------------------------------------------------

    def heat_rgb(self):
        if self._heat is None:
            p = self.plate
            amp = np.abs(chladni.grid(PLATE, p.n, p.m, p.mix, p.free)) * 0.5
            base = np.array(PLATE_BG, float)
            hot = np.array(T.HEAT, float)
            self._heat = (base + amp[..., None] * (hot - base)).astype(np.uint8)
        return self._heat

    def draw_plate(self):
        if self.heatmap:
            rgb = self.heat_rgb().copy()
        else:
            rgb = np.empty((PLATE, PLATE, 3), np.uint8)
            rgb[:] = PLATE_BG
        # Powder underneath, sand on top, as on a real plate.
        order = sorted(CHANNELS, key=lambda ch: not self.plates[ch].powder)
        for ch in order:
            if not self.value(f"{ch} Show"):
                continue
            colour = COLOURS[int(self.value(f"{ch} Colour"))][1]
            idx = (self.plates[ch].pos * (PLATE - 1)).astype(np.int32)
            for dx, dy in dot_offsets(int(self.value(f"{ch} Size"))):
                x = np.clip(idx[:, 0] + dx, 0, PLATE - 1)
                y = np.clip(idx[:, 1] + dy, 0, PLATE - 1)
                rgb[x, y] = colour
        pygame.surfarray.blit_array(self.plate_surface, rgb)
        self.screen.blit(self.plate_surface, (PANEL_W, 0))

    def draw_left(self):
        pt, f = self.paint, self.fonts
        pt.clip((0, 0, PANEL_W, SCREEN_H))
        pt.line(PANEL_W - 1, 0, PANEL_W - 1, SCREEN_H, T.EDGE)
        x, right = 12, PANEL_W - 13
        menu = self.menus[-1]
        # Header: the path to this menu, then its name.
        path = "/" + "/".join(m.title.lower().replace(" ", "-") for m in self.menus)
        while len(path) > 1 and pt.width(path, f.caption) > right - x:
            path = "…" + path[2:]
        pt.text(path, x, 12, f.caption, T.FAINT)
        pt.text(menu.title.upper(), x, 30, f.title, T.TEXT, width=right - x)
        pt.line(x, MENU_TOP - 10, right, MENU_TOP - 10)

        rows = menu.rows()
        menu.scroll_to_selection(MENU_ROWS)
        y = MENU_TOP
        for i in range(menu.top, min(len(rows), menu.top + MENU_ROWS)):
            row, sel = rows[i], i == menu.selected
            if sel:
                if T.SELECTED:
                    pt.rect((0, y - 4, PANEL_W - 1, ROW_H - 4), T.SELECTED)
                    pt.rect((0, y - 4, 2, ROW_H - 4), T.ACCENT)
                if T.SEL_OUTLINE:
                    pt.frame((4, y - 4, PANEL_W - 9, ROW_H - 4), T.SEL_OUTLINE)
            value = row.value()
            opens = row.press is not None and row.hint is None
            tag = row.hint() if row.hint else (row.badge() if row.badge else "")
            tag_w = pt.width(tag, f.caption) + 6 if tag else 0
            # Top line: index, label (when there's a value) and the tag on the right.
            pt.text(f"{i + 1:02d}", x, y, f.caption, T.ACCENT if sel else T.FAINT)
            if value:
                pt.caption(row.label, x + 20, y, T.ACCENT if sel else T.DIM, width=right - x - 20 - tag_w)
            if tag:
                pt.caption(tag, right, y, T.ACCENT if sel or tag in ("on", "mic") else T.DIM, anchor="topright")
            # Bottom line: the value, or the label for rows that only open or run something.
            main = value or row.label
            colour = T.WARN if value == "press again" else (T.ACCENT if sel and not value else T.TEXT)
            pt.text(main, x, y + 15, f.value, colour, width=right - x - (9 if opens else 0))
            if opens:
                pt.text(">", right, y + 18, f.body, T.ACCENT if sel else T.FAINT, anchor="topright")
            y += ROW_H
        if menu.top > 0:
            pt.text("▲", right, MENU_TOP - 26, f.caption, T.DIM, anchor="topright")
        if menu.top + MENU_ROWS < len(rows):
            pt.text("▼", right, MENU_TOP + MENU_ROWS * ROW_H - 14, f.caption, T.DIM, anchor="topright")

        top = SCREEN_H - FOOTER_H
        pt.line(x, top, right, top)
        pt.caption("K1 select / open", x, top + 10, T.FAINT, width=right - x)
        pt.caption("K2 adjust / back", x, top + 26, T.FAINT, width=right - x)
        pt.clip(None)

    def side(self):
        """Start the right panel; returns (left text x, right edge)."""
        left = PANEL_W + PLATE
        self.paint.clip((left, 0, SCREEN_W - left, SCREEN_H))
        self.paint.line(left, 0, left, SCREEN_H, T.EDGE)
        return left + 12, SCREEN_W - 12

    def draw_music_player(self, y):
        pt, f = self.paint, self.fonts
        x, right = self.side()
        p = self.plate
        if self.song:
            y = pt.rule("music", x, y, right)
            y += pt.text(self.song.name, x, y, f.body, T.ACCENT, width=right - x) + 4
            pos, length = self.song.position(), self.song.length
            pt.line(x, y, right, y, T.FAINT)
            pt.line(x, y, x + (right - x) * min(1.0, pos / max(length, 1e-6)), y, T.ACCENT)
            y += 6
            y += pt.pair("time", f"{music.format_time(pos)}/{music.format_time(length)}", x, y, right) + 10
        if self.seq.running:
            seq = self.seq
            y = pt.rule("program", x, y, right)
            y += pt.text(seq.program["name"], x, y, f.body, T.ACCENT, width=right - x) + 2
            y += pt.pair("state", f"{seq.index + 1}/{len(seq.program['states'])}", x, y, right) + 2
            y += pt.pair("next", f"{seq.remaining:.0f} s", x, y, right) + 2


    def draw_right(self):
        pt, f = self.paint, self.fonts
        x, right = self.side()
        p = self.plate

        y = pt.rule("mode", x, 12, right)
        y += pt.text(f"({p.n},{p.m})", x, y, f.big, T.TEXT) + 4
        y += pt.pair("rel freq", f"{chladni.relative_frequency(p.n, p.m):.1f}x", x, y, right) + 6
        status = None
        if self.song:
            status = "music " + ("--" if self.music_note is None else f"{self.music_note:.0f} Hz")
        elif self.live_on:
            if self.mic is not None and not self.mic.available:
                status = "no mic"
            elif self.live_pitch is None:
                status = "live listening"
            else:
                status = f"live {self.live_pitch:.0f} Hz"
        elif self.synth.available and self.value("Sound"):
            status = f"tone {audio.mode_frequency(p.n, p.m, self.value('Base pitch')):.0f} Hz"
        if status:
            pt.square(x, y + 4, 6, T.ACCENT)
            pt.caption(status, x + 12, y + 1, T.ACCENT, width=right - x - 12)
            y += 18
        y += 10

        y = pt.rule("equation", x, y, right)
        fn = "cos" if p.free else "sin"
        sign = "+" if p.mix >= 0 else "-"
        # Term, sign (and weight), term: one per line so cos(12πx)cos(11πy) still fits.
        eq = f.body
        weight = "" if abs(p.mix) == 1 else f" {abs(p.mix):.2g} ×"
        y += pt.text(f"{fn}({p.n}πx){fn}({p.m}πy)", x, y, eq, T.TEXT, width=right - x) + 1
        y += pt.text(f"{sign}{weight}", x, y, eq, T.DIM) + 1
        y += pt.text(f"{fn}({p.m}πx){fn}({p.n}πy)", x, y, eq, T.TEXT, width=right - x) + 6
        y += pt.pair("edges", "free" if p.free else "fixed", x, y, right) + 2
        if chladni.is_degenerate(p.n, p.m, p.mix, p.free):
            y += pt.caption("flat plate: no nodes", x, y, T.WARN, width=right - x) + 2
        y += 10

        y = pt.rule("channels", x, y, right)
        for ch in CHANNELS:
            colour = COLOURS[int(self.value(f"{ch} Colour"))][1]
            on = self.value(f"{ch} Particles") > 0 and self.value(f"{ch} Show")
            pt.square(x, y + 4, 7, colour if on else T.FAINT)
            pt.text(ch, x + 14, y, f.body, T.TEXT if on else T.FAINT)
            pt.text(self.channel_summary(ch), right, y, f.body, T.DIM, anchor="topright", width=right - x - 30)
            y += 18
        y += 10

        self.draw_music_player(y)
        self.draw_side_footer(x, right, "R scatter / H heat")

    def draw_side_footer(self, x, right, keys):
        pt, f = self.paint, self.fonts
        top = SCREEN_H - FOOTER_H
        pt.line(x, top, right, top)
        if keys:
            pt.caption(keys, x, top + 10, T.FAINT, width=right - x)
        pt.caption(f"{self.clock.get_fps():.0f} fps", x, top + 26, T.FAINT)
        pt.clip(None)

    def draw(self):
        self.screen.fill(T.BG)
        area = (PANEL_W, 0, PLATE, PLATE)
        if self.scope_on:
            pygame.draw.rect(self.screen, PLATE_BG, area)
            self.paint.clip(area)
            self.scope.draw(pygame, self.screen, area, self.fonts.tick, self.scope_time)
            self.paint.clip(None)
            if not T.BORDER:  # with Mono's white edges the marks would double them up
                self.paint.corners(area)
            self.draw_left()
            self.draw_scope_readouts()
        else:
            self.draw_plate()
            if not T.BORDER:  # with Mono's white edges the marks would double them up
                self.paint.corners(area)
            self.draw_left()
            self.draw_right()
        if T.BORDER:
            # Four 1px lines meeting square at the corners, around the whole window.
            r, b = SCREEN_W - 1, SCREEN_H - 1
            for x0, y0, x1, y1 in ((0, 0, r, 0), (0, b, r, b), (0, 0, 0, b), (r, 0, r, b)):
                self.paint.line(x0, y0, x1, y1, T.BORDER)

    def draw_scope_readouts(self):
        pt, f = self.paint, self.fonts
        x, right = self.side()
        sc = self.scope
        if sc.xy:
            y = pt.rule("x-y", x, 12, right)
            c1, c2 = sc.channel("1"), sc.channel("2")
            gen = all(c["wave"] not in scope.AUDIO + ("Off",) for c in (c1, c2))
            r = scope.ratio(c1["freq"], c2["freq"]) if gen else None
            y += pt.text(r or "free", x, y, f.big) + 4
            y += pt.pair("window", scope.format_time(sc.time_div * scope.H_DIVS), x, y, right) + 12
        else:
            y = pt.rule("timebase", x, 12, right)
            y += pt.text(f"{scope.format_time(sc.time_div)}/div", x, y, f.big) + 12
        for ch, signal, rows in self.scope.readouts(self.scope_time):
            colour = scope.COLOURS[ch]
            y = pt.rule(f"ch{ch}", x, y, right, colour)
            y += pt.text(signal.lower(), x, y, f.value, T.TEXT if signal != "Off" else T.FAINT, width=right - x) + 4
            for label, value in rows:
                y += pt.pair(label, value, x, y, right, T.WARN if label == "status" else T.TEXT) + 2
            y += 12
        self.draw_music_player(y)
        self.draw_side_footer(x, right, "")

    # --- loop ----------------------------------------------------------

    def tick(self, dt=None):
        """One frame. dt is seconds since the last frame; None means use the clock."""
        if dt is None:
            dt = self.clock.get_time() / 1000.0
        for event in pygame.event.get():
            for action in controls.keyboard_actions(event):
                self.handle(*action)
        for action in self.encoders.poll():
            self.handle(*action)
        if self.scope_on:
            self.scope_time += dt  # the plate pauses while the scope is showing
            self.next_song()       # but a song on the scope keeps playing through the folder
        else:
            self.dt = dt
            state = self.seq.tick(dt)
            if state is not None:
                self.enter_state(state)
            self.update_live()
            self.update_music()
            for ch, p in self.plates.items():
                if self.value(f"{ch} Show"):
                    p.step()
        self.update_sound()
        self.draw()
        pygame.display.flip()
        self.clock.tick(FPS)

    def update_sound(self):
        p = self.plate
        silent = (not self.value("Sound") or self.live_on  # the mic would hear the speaker
                  or self.song is not None                  # the song is the sound
                  or self.scope_on
                  or chladni.is_degenerate(p.n, p.m, p.mix, p.free))
        freq = None if silent else audio.mode_frequency(p.n, p.m, self.value("Base pitch"))
        self.synth.update(freq, self.value("Volume"))

    def run(self):
        while self.running:
            self.tick()
        self.store.save()
        if self.mic is not None and hasattr(self.mic, "close"):
            self.mic.close()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--screenshot", help="render headless and save a PNG here")
    ap.add_argument("--frames", type=int, default=300, help="frames to simulate before the screenshot")
    ap.add_argument("--n", type=int)
    ap.add_argument("--m", type=int)
    ap.add_argument("--mix", type=float, help="-1 .. 1, weight of the second term")
    ap.add_argument("--fixed-edges", action="store_true", help="use the sin (fixed edge) form")
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--run", metavar="PROGRAM", help="start this saved program")
    ap.add_argument("--mute", action="store_true", help="don't open the sound device")
    ap.add_argument("--music", metavar="DIR", default=MUSIC_DIR, help="folder of songs for Music mode")
    args = ap.parse_args()

    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    app = App(fullscreen=args.fullscreen, sound=not (args.mute or args.screenshot), music_dir=args.music)
    if args.screenshot:
        app.param("Live mic").value = 0  # never open the mic for an offline render
    for name, v in (("n", args.n), ("m", args.m), ("Mix", args.mix)):
        if v is not None:
            next(p for p in app.params if p.name == name).value = v
    if args.fixed_edges:
        next(p for p in app.params if p.name == "Edges").value = 0
    app.heatmap = args.heatmap
    app.apply_params()
    if args.run:
        prog = next((p for p in app.store.programs if p["name"] == args.run), None)
        if prog is None:
            ap.error(f"no program named {args.run!r}")
        app.run_program(prog)

    if args.screenshot:
        global FPS
        FPS = 0  # no frame cap when rendering offline
        for _ in range(args.frames):
            app.tick(1.0 / 30)  # simulated time: 30 frames = 1 second
        pygame.image.save(app.screen, args.screenshot)
        pygame.quit()
    else:
        app.run()


if __name__ == "__main__":
    main()
