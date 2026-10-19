"""Quick sanity checks: python3 test_chladni.py"""

import numpy as np

import chladni
from params import Param
from particles import SandPlate


def test_fixed_edges_are_nodal():
    s = np.linspace(0, 1, 50)
    for n, m in [(1, 2), (3, 5), (4, 7)]:
        for sign in (1, -1):
            assert np.allclose(chladni.field(0, s, n, m, sign, free=False), 0)
            assert np.allclose(chladni.field(s, 1, n, m, sign, free=False), 0)


def test_free_edges_vibrate():
    s = np.linspace(0, 1, 50)
    assert np.abs(chladni.field(0, s, 3, 5, 1, free=True)).max() > 1


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    x, y = rng.random(100), rng.random(100)
    h = 1e-6
    for free in (True, False):
        for sign in (1, -1):
            f = lambda a, b: chladni.field(a, b, 3, 5, sign, free)
            gx, gy = chladni.gradient(x, y, 3, 5, sign, free)
            fx = (f(x + h, y) - f(x - h, y)) / (2 * h)
            fy = (f(x, y + h) - f(x, y - h)) / (2 * h)
            assert np.allclose(gx, fx, atol=1e-4) and np.allclose(gy, fy, atol=1e-4)


def test_degenerate_mode():
    assert chladni.is_degenerate(4, 4, -1)
    assert np.allclose(chladni.grid(32, 4, 4, -1), 0)


def test_sand_settles_on_nodes():
    plate = SandPlate(count=3000, n=3, m=5, uniformity=1.0, seed=0)
    before = np.abs(chladni.field(*plate.pos.T, 3, 5)).mean()
    for _ in range(300):
        plate.step()
    after = np.abs(chladni.field(*plate.pos.T, 3, 5)).mean()
    assert after < before * 0.2, (before, after)
    assert plate.pos.min() >= 0 and plate.pos.max() <= 1


def test_sand_does_not_pile_up_on_free_edges():
    # Regression: with fixed edges ~70% of the sand ended up on the rim after
    # a few mode changes, because the border is a nodal line in every mode.
    rng = np.random.default_rng(3)
    plate = SandPlate(count=3000, seed=0)
    for _ in range(20):
        plate.n, plate.m = rng.integers(1, 9, 2)
        plate.mix = rng.choice([-1, 1])
        for _ in range(60):
            plate.step()
    near_edge = np.mean(np.minimum(plate.pos, 1 - plate.pos).min(1) < 0.02)
    assert near_edge < 0.25, near_edge


def test_particle_count_changes():
    plate = SandPlate(count=1000, seed=0)
    plate.set_count(2500)
    assert plate.count == 2500
    plate.set_count(400)
    assert plate.count == 400


def test_value_labels():
    from params import format_value
    assert format_value("Live mic", 1) == "On" and format_value("Base pitch", 80) == "80 Hz"
    assert format_value("A Mass", 2.0) == "2x" and format_value("B Type", 1) == "Powder"


def test_wrapping_param():
    p = Param("Edges", 1, 0, 1, 1, wrap=True)
    p.nudge(1)
    assert p.value == 0
    p.nudge(1)
    assert p.value == 1


def test_mix_blends_terms():
    x, y = np.random.default_rng(0).random((2, 50))
    t1 = chladni.field(x, y, 2, 5, 0.0)
    t2 = chladni.field(y, x, 2, 5, 0.0)       # second term is the first, transposed
    assert np.allclose(chladni.field(x, y, 2, 5, 0.4), t1 + 0.4 * t2)
    assert chladni.is_degenerate(0, 0, 0.5) and chladni.is_degenerate(0, 3, 1, free=False)
    assert not chladni.is_degenerate(0, 3, 1)



def test_sequencer_cycles_and_wraps():
    import program
    prog = {"name": "t", "states": [dict(program.new_state(
        {"n": n, "m": 2, "Mix": 1, "Edges": 1, "Uniformity": 0.5}), Hold=h) for n, h in ((1, 2), (3, 1))]}
    seq = program.Sequencer()
    assert seq.start(prog)["n"] == 1
    assert seq.tick(1.5) is None          # still holding S1 (2 s)
    assert seq.tick(0.6)["n"] == 3        # S2
    assert seq.tick(1.0)["n"] == 1        # wraps back to S1
    prog["states"].clear()
    assert seq.tick(5) is None and not seq.running


def test_program_menus_end_to_end():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import controls as c
    from app import App
    path = os.path.join(tempfile.mkdtemp(), "programs.json")
    app = App(programs_file=path, sound=False, settings_file=path + ".settings")
    app.show(0)                                        # open Resonance

    def go(*actions):
        for a in actions:
            app.handle(*(a if isinstance(a, tuple) else (a, 0)))

    def select(label):
        labels = [r.label for r in app.menu.rows()]
        go((c.SELECT, labels.index(label) - app.menu.selected))

    select("Programs"); go(c.PRESS)
    select("+ New program"); go(c.PRESS)             # creates "Program 3" with S1 = current plate
    name = app.store.programs[-1]["name"]
    assert app.menu.title == name == "Program 3"
    select("+ Add state"); go(c.PRESS)                # opens the S2 editor
    select("n"); go((c.ADJUST, 2))                     # n 3 -> 5, previewed live
    assert app.plate.n == 5
    select("Hold"); go(c.PRESS, (c.ADJUST, -5))        # step 10 s -> 1 s, then 8 s -> 3 s
    go(c.BACK)                                         # back to the program, saves
    select("Run"); go(c.PRESS)
    assert app.seq.running and len(app.menus) == 2 and app.plate.n == 3

    for _ in range(8 * 30 + 1):                        # S1 holds 8 s at 30 fps
        app.tick(1 / 30)
    assert app.plate.n == 5 and app.seq.index == 1

    select("Source"); go(c.PRESS)
    select("n"); go((c.ADJUST, 1))                     # a manual turn stops the program
    assert not app.seq.running

    import program
    saved = program.Store(path).programs
    p2 = next(p for p in saved if p["name"] == name)
    assert [s["n"] for s in p2["states"]] == [3, 5] and p2["states"][1]["Hold"] == 3

    go(c.BACK)                                         # Source -> Resonance
    select("Programs"); go(c.PRESS)
    select(name); go(c.PRESS)
    select("Delete program"); go(c.PRESS)
    assert any(p["name"] == name for p in app.store.programs)  # first press only arms
    go(c.PRESS)
    assert not any(p["name"] == name for p in app.store.programs)


def test_powder_collects_on_antinodes():
    plate = SandPlate(count=3000, n=3, m=5, powder=True, seed=0)
    for _ in range(300):
        plate.step()
    amp = np.abs(chladni.field(*plate.pos.T, 3, 5)) * 0.5
    assert amp.mean() > 0.6, amp.mean()   # sand settles near 0


def test_heavier_sand_makes_sharper_lines():
    spread = []
    for mass in (0.5, 4.0):
        plate = SandPlate(count=3000, n=3, m=5, mass=mass, seed=0)
        for _ in range(400):
            plate.step()
        spread.append(np.abs(chladni.field(*plate.pos.T, 3, 5)).mean())
    assert spread[1] < spread[0] * 0.5, spread


def test_channel_menus_and_old_states():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import controls as c
    from app import App
    app = App(programs_file=os.path.join(tempfile.mkdtemp(), "programs.json"), sound=False,
              settings_file=os.path.join(tempfile.mkdtemp(), "settings.json"))
    app.show(0)
    labels = [r.label for r in app.menu.rows()]
    app.handle(c.SELECT, labels.index("Channel B")); app.handle(c.PRESS, 0)
    assert [r.label for r in app.menu.rows()] == ["Show", "Particles", "Type", "Mass", "Colour", "Size"]
    assert app.plates["B"].powder
    app.handle(c.SELECT, 2); app.handle(c.ADJUST, 1)     # Type: Powder -> Sand
    assert not app.plates["B"].powder
    app.handle(c.SELECT, -1); app.handle(c.ADJUST, -5)   # Particles 2500 -> 0
    assert app.plates["B"].count == 0
    # The built-in Tour states predate channels; running one keeps the channel settings.
    app.run_program(app.store.programs[0])
    assert app.plates["B"].count == 0 and not app.plates["B"].powder


def test_dot_size_draws_bigger_dots():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from app import App
    app = App(programs_file=os.path.join(tempfile.mkdtemp(), "programs.json"), sound=False,
              settings_file=os.path.join(tempfile.mkdtemp(), "settings.json"))
    app.param("B Particles").value = 0
    app.param("A Particles").value = 500
    app.apply_params()
    lit = []
    for size in (1, 4):
        app.param("A Size").value = size
        app.draw_plate()
        import pygame
        arr = pygame.surfarray.array3d(app.plate_surface)
        lit.append(int((arr.sum(axis=2) > 300).sum()))
    assert lit[1] > lit[0] * 5, lit


def test_old_programs_migrate_and_get_new_builtins():
    import json
    import os
    import tempfile
    import program
    path = os.path.join(tempfile.mkdtemp(), "programs.json")
    old = {"programs": [{"name": "Mine", "states": [{"n": 2, "m": 3, "Sign": -1, "Edges": 1,
                                                       "Uniformity": 0.5, "Hold": 4}]}]}
    json.dump(old, open(path, "w"))
    store = program.Store(path)
    names = [p["name"] for p in store.programs]
    assert names == ["Mine", "Reference tour"]     # "Tour" was seen (and deleted) before
    state = store.programs[0]["states"][0]
    assert state["Mix"] == -1.0 and "Sign" not in state and state["Reset"] == 0
    store.delete(store.programs[1]); store.save()
    assert [p["name"] for p in program.Store(path).programs] == ["Mine"]  # stays deleted


def test_reference_tour_resets():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from app import App
    app = App(programs_file=os.path.join(tempfile.mkdtemp(), "programs.json"), sound=False,
              settings_file=os.path.join(tempfile.mkdtemp(), "settings.json"))
    tour = next(p for p in app.store.programs if p["name"] == "Reference tour")
    assert len(tour["states"]) == 40
    for s in tour["states"]:
        assert not __import__("chladni").is_degenerate(s["n"], s["m"], s["Mix"], s["Edges"])
    app.run_program(tour)
    assert app.plates["A"].count == 12000 and app.plates["B"].count == 0
    assert app.value("A Size") == 1
    hold = tour["states"][0]["Hold"]
    for _ in range(int(hold * 30) + 1):             # S1 -> S2
        app.tick(1 / 30)
    assert app.seq.index == 1
    app.seq.elapsed = hold - 0.01                   # jump to the edge of S3, which resets
    before = app.plate.pos.copy()
    app.tick(1 / 30)
    assert app.seq.index == 2 and tour["states"][2]["Reset"] == 1
    moved = np.abs(app.plate.pos - before).max(axis=1)
    assert np.mean(moved > 0.05) > 0.5               # most grains were thrown back on the plate


def test_tone_frequency_and_glide():
    import audio
    assert audio.mode_frequency(1, 1) == 80
    assert audio.mode_frequency(3, 5, base=100) == 1700
    v = audio.Voice(rate=44100)
    for _ in range(40):                                   # long enough to finish gliding
        out = v.chunk(1000.0, 1.0)
    crossings = np.sum(np.diff(np.sign(out)) != 0)
    assert abs(crossings / 2 / audio.CHUNK_S - 1000) < 30, crossings
    v.chunk(2000.0, 1.0)                                  # pitch moves part way, not instantly
    assert 1000 < v.freq < 2000
    for _ in range(5):
        quiet = v.chunk(None, 1.0)                        # no mode: silence
    assert np.abs(quiet).max() < 1e-6


def voice_like(f0, seconds=0.128, rate=16000):
    """A sung-note stand-in: a fundamental plus decaying harmonics and a little noise."""
    t = np.arange(int(seconds * rate)) / rate
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    return 0.3 * x + 0.01 * np.random.default_rng(0).standard_normal(len(t))


def test_pitch_detection():
    import live
    for f0 in (65, 80, 110, 196, 440, 880, 1400):
        f = live.detect_pitch(voice_like(f0))
        assert abs(f / f0 - 1) < 0.01, (f0, f)
    pure = 0.3 * np.sin(2 * np.pi * 80 * np.arange(2048) / 16000)    # a plain low sine
    assert abs(live.detect_pitch(pure) / 80 - 1) < 0.01
    assert live.detect_pitch(np.zeros(2048)) is None                     # silence
    noise = 0.3 * np.random.default_rng(1).standard_normal(2048)
    assert live.detect_pitch(noise) is None                             # unpitched


def test_pitch_to_mode():
    import live
    assert live.nearest_mode(1360, 80) == (3, 5)                        # 80 * 34 / 2
    assert live.nearest_mode(80, 80) == (1, 1)
    assert live.nearest_mode(40, 80) == (0, 1)                          # free edges allow 0
    assert live.nearest_mode(40, 80, free=False) == (1, 1)              # fixed edges don't
    assert live.resonance(1360, 3, 5, 80) == 1.0
    assert live.resonance(1360 * 1.06, 3, 5, 80) < 0.5                  # a semitone off


class FakeMic:
    available = True

    def __init__(self):
        self.signal = np.zeros(2048)

    def read(self):
        return self.signal


def test_live_mode_follows_the_mic():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import controls as c
    from app import App
    d = tempfile.mkdtemp()
    mic = FakeMic()
    app = App(programs_file=os.path.join(d, "p.json"), sound=False,
              settings_file=os.path.join(d, "s.json"), mic=mic)
    app.run_program(app.store.programs[0])
    app.back()                                         # up to the Display menu
    labels = [r.label for r in app.menu.rows()]
    app.handle(c.SELECT, labels.index("Settings")); app.handle(c.PRESS, 0)
    labels = [r.label for r in app.menu.rows()]
    app.handle(c.SELECT, labels.index("Live mic")); app.handle(c.ADJUST, 1)
    assert app.live_on and not app.seq.running         # live mode takes over from programs

    mic.signal = voice_like(1360)
    for _ in range(20):
        app.tick(1 / 30)
    assert (app.plate.n, app.plate.m) == (3, 5) and abs(app.live_pitch - 1360) < 15
    assert app.plate.drive > 0.9

    mic.signal = np.zeros(2048)                          # silence: the plate stops
    for _ in range(40):
        app.tick(1 / 30)
    assert app.live_pitch is None and app.plate.drive < 0.05
    before = app.plate.pos.copy()
    app.tick(1 / 30)
    assert np.abs(app.plate.pos - before).max() < 1e-3

    # Settings are saved and come back on the next start.
    app.handle(c.SELECT, -app.menu.selected); app.handle(c.ADJUST, -1)  # Sound off
    again = App(programs_file=os.path.join(d, "p.json"), sound=False,
                settings_file=os.path.join(d, "s.json"), mic=mic)
    assert again.value("Sound") == 0 and again.value("Live mic") == 1


def test_app_runs_with_dummy_audio():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    from app import App
    d = tempfile.mkdtemp()
    app = App(programs_file=os.path.join(d, "p.json"), settings_file=os.path.join(d, "s.json"))
    for _ in range(10):
        app.tick(1 / 30)
    if app.synth.available:
        assert app.synth.voice.freq == 80 * 34 / 2       # default mode (3, 5)
    assert [r.label for r in app.menu.rows()][-1] == "Settings"


def write_melody(path, notes, rate=44100):
    """A test song: one chord per second (melody note, bass two octaves down, a fifth)."""
    import wave
    t = np.arange(rate) / rate
    parts = []
    for f in notes:
        env = np.minimum(1, t * 20) * np.exp(-t * 1.5)
        mel = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 5))
        parts.append(env * (0.35 * mel + 0.2 * np.sin(2 * np.pi * f / 4 * t)
                            + 0.15 * np.sin(2 * np.pi * f * 1.5 * t)))
    x = np.concatenate(parts)
    x = (x / np.abs(x).max() * 0.8 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(x.tobytes())
    return x / 32767.0


def test_music_note_detection_on_chords():
    import music
    import os
    import tempfile
    d = tempfile.mkdtemp()
    notes = [220, 262, 330, 392, 523, 784]
    x = write_melody(os.path.join(d, "song.wav"), notes)
    for i, f in enumerate(notes):
        seg = x[int((i + 0.3) * 44100):int((i + 0.3 + music.WINDOW_S) * 44100)]
        assert abs(music.dominant_note(seg, 44100) / f - 1) < 0.01, f
    assert music.dominant_note(np.zeros(8000), 44100) is None
    open(os.path.join(d, "notes.txt"), "w").close()
    assert [os.path.basename(p) for p in music.list_tracks(d)] == ["song.wav"]


def test_music_mode_plays_a_folder():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import live
    from app import App
    d = tempfile.mkdtemp()
    songs = os.path.join(d, "music"); os.mkdir(songs)
    write_melody(os.path.join(songs, "a.wav"), [220, 330, 523])
    write_melody(os.path.join(songs, "b.wav"), [262])
    app = App(programs_file=os.path.join(d, "p.json"), settings_file=os.path.join(d, "s.json"),
              music_dir=songs)
    if not app.synth.available:
        return  # no audio driver at all on this machine
    app.show(0)
    labels = [r.label for r in app.menu.rows()]
    app.handle("select", labels.index("Music")); app.handle("press", 0)
    assert [r.label for r in app.menu.rows()] == ["a", "b"]
    app.handle("press", 0)                                 # play "a"
    assert app.song.name == "a" and len(app.menus) == 2
    app.song.position = lambda: 1.5                        # halfway through the 330 Hz chord
    for _ in range(10):
        app.tick(1 / 30)
    assert abs(app.music_note - 330) < 4
    assert (app.plate.n, app.plate.m) == live.nearest_mode(330, 80)
    assert app.plate.drive > 0.3
    app.song.position = lambda: 99                         # past the end: next song
    app.tick(1 / 30)
    assert app.song.name == "b"
    app.open("Source", app.source_rows)
    labels = [r.label for r in app.menu.rows()]
    app.handle("select", labels.index("n") - app.menu.selected); app.handle("adjust", 1)  # turn n by hand
    assert app.song is None and app.plate.drive == 1.0


def test_scope_waveforms_and_labels():
    import scope
    p = np.array([0, 0.25, 0.5, 0.75])
    assert np.allclose(scope.wave("Sine", p), [0, 1, 0, -1])
    assert np.allclose(scope.wave("Triangle", p), [0, 1, 0, -1])
    assert np.allclose(scope.wave("Square", p + 0.1), [1, 1, -1, -1])
    assert scope.format_value("Time/div", scope.TIME_DIVS.index(0.0005)) == "500 us/div"
    assert scope.format_value("CH2 Volts/div", scope.VOLT_DIVS.index(0.2)) == "200 mV/div"
    assert scope.format_freq(1500) == "1.5 kHz"
    f = next(p for p in scope.default_params() if p.name == "CH1 Frequency")
    assert [f.step] + [(f.cycle_step(), f.step)[1] for _ in range(4)] == [1000, 100, 10, 1, 1000]
    f.nudge(-1)
    assert f.value == 1.0                                # 440 - 1000 clamps to 1 Hz
    assert scope.format_freq(12345) == "12.345 kHz"


def test_scope_display_and_trigger():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import scope
    from app import App
    d = tempfile.mkdtemp()
    app = App(programs_file=os.path.join(d, "p.json"), sound=False, settings_file=os.path.join(d, "s.json"))
    assert [r.label for r in app.menu.rows()] == ["Oscilloscope", "Resonance", "Settings"]
    app.handle("press", 0)                               # open Oscilloscope: switches to the scope
    assert app.scope_on and app.menu.title == "Oscilloscope"
    assert [r.label for r in app.menu.rows()] == ["Mode", "Time/div", "Channel 1", "Channel 2", "Music"]
    app.back()
    assert app.menu.rows()[0].badge() == "on"
    before = app.plate.pos.copy()
    for _ in range(5):
        app.tick(1 / 30)
    assert np.array_equal(before, app.plate.pos)        # the plate pauses
    # Channel 1 is triggered: every sweep starts at 0 V and rising.
    for now in (0.0, 0.37, 1.234):
        t, v = app.scope.trace("1", now, 200)
        assert abs(v[0]) < 1e-9 and v[1] > 0
    # The display choice is remembered; Settings stay shared.
    again = App(programs_file=os.path.join(d, "p.json"), sound=False, settings_file=os.path.join(d, "s.json"))
    assert again.scope_on


def test_press_cycles_step_size():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from app import App
    d = tempfile.mkdtemp()
    app = App(programs_file=os.path.join(d, "p.json"), sound=False, settings_file=os.path.join(d, "s.json"))
    app.open("Channel A", lambda: app.channel_rows("A"))
    labels = [r.label for r in app.menu.rows()]
    app.menu.selected = labels.index("Particles")
    p = app.param("A Particles")
    start = p.value
    seen = []
    for _ in range(4):
        seen.append(app.menu.current().hint())
        app.handle("adjust", 1)
        app.handle("press", 1)
    assert seen == ["±1000", "±100", "±10", "±1000"], seen
    assert p.value == start + 1000 + 100 + 10 + 1000
    # Rows without a step choice (n) have no hint and press does nothing.
    app.back()
    app.open("Source", app.source_rows)
    assert [r.label for r in app.menu.rows()] == ["Uniformity", "Adherence", "n", "m", "Mix", "Edges"]
    row = next(r for r in app.menu.rows() if r.label == "n")
    assert row.hint is None and row.press is None


def test_scope_shows_live_audio():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import live
    import scope
    from app import App

    class Mic:
        available = True
        def recent(self):
            t = np.arange(live.RATE) / live.RATE
            return 0.2 * np.sin(2 * np.pi * 250 * t + 1.0)
        read = recent

    d = tempfile.mkdtemp()
    app = App(programs_file=os.path.join(d, "p.json"), sound=False,
              settings_file=os.path.join(d, "s.json"), mic=Mic())
    app.show(1)
    app.param("CH1 Wave").value = scope.WAVES.index("Live mic")
    app.param("CH1 Amplitude").value = 2.0               # gain
    t, v = app.scope.trace("1", 0.0, 400)
    assert abs(v[0]) < 0.05 and v[3] > v[0]               # triggered on a rising edge
    assert abs(np.ptp(v) - 0.8) < 0.02                    # 0.2 peak x gain 2 -> 0.8 Vpp
    app.open("Channel 1", lambda: app.scope_rows("1"))
    assert [r.label for r in app.menu.rows()] == ["Signal", "Gain", "Volts/div", "Position"]
    app.param("CH2 Wave").value = scope.WAVES.index("Music")
    assert app.scope.trace("2", 0.0, 50) is None          # no song playing yet
    (_, _, ch1), (_, _, ch2) = app.scope.readouts()
    assert ("status", "no song") in ch2
    assert dict(ch1)["pitch"].startswith("~25")           # ~250 Hz detected
    app.tick(1 / 30)                                       # draws without a song


def test_program_states_toggle_channels():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import program
    from app import App
    d = tempfile.mkdtemp()
    app = App(programs_file=os.path.join(d, "p.json"), sound=False, settings_file=os.path.join(d, "s.json"))
    assert app.value("A Show") == 1 and app.value("B Show") == 0   # B starts hidden
    app.param("B Show").value = 1
    prog = app.store.create(app.current_values())
    assert prog["states"][0]["A Show"] == 1 and prog["states"][0]["B Show"] == 1
    app.edit_state(prog["states"][0], "B Show", 1)          # B off in this state
    assert app.value("B Show") == 0 and app.channel_summary("B") == "hidden"
    before_b, before_a = app.plates["B"].pos.copy(), app.plates["A"].pos.copy()
    app.tick(1 / 30)
    assert np.array_equal(before_b, app.plates["B"].pos)      # hidden: not moved
    assert not np.array_equal(before_a, app.plates["A"].pos)
    app.open_program(prog)
    assert app.menu.rows()[1].badge().startswith("A ")       # state list shows which channels are on
    # Programs saved before Show existed get both channels on.
    old = [{"name": "old", "states": [{"n": 2, "m": 3, "Mix": 1.0, "Hold": 4}]}]
    assert program._migrate(old)[0]["states"][0]["A Show"] == 1


def test_music_keeps_patterns_from_bundling():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import live
    from app import App
    from particles import SandPlate
    # Crowding: a blob of sand on a nodal line spreads out along it.
    plate = SandPlate(4000, 2, 3, 1.0, seed=0)
    plate.pos[:] = 0.5 + 0.005 * plate.rng.standard_normal((4000, 2))
    def busiest(p):
        ij = np.minimum((p.pos * 64).astype(int), 63)
        return np.bincount(ij[:, 0] * 64 + ij[:, 1]).max()
    start = busiest(plate)
    for _ in range(150):
        plate.step()
    assert busiest(plate) < start / 4
    # Settling: a note shorter than SETTLE_S doesn't change the mode; a held one does, with a shake.
    d = tempfile.mkdtemp()
    app = App(programs_file=os.path.join(d, "p.json"), sound=False, settings_file=os.path.join(d, "s.json"))
    app.dt = 1 / 30
    for _ in range(30):
        app.follow_pitch(523.0)                          # settle on (2, 3)
    assert (app.plate.n, app.plate.m) == live.nearest_mode(523.0, 80)
    for _ in range(2):
        app.follow_pitch(659.0)                          # a quick passing note
    assert (app.plate.n, app.plate.m) == live.nearest_mode(523.0, 80)
    for _ in range(3):
        app.follow_pitch(659.0)                          # ...held past SETTLE_S
    assert (app.plate.n, app.plate.m) == live.nearest_mode(659.0, 80)
    assert app.plates["A"].agitation > 0.5


def test_beats_and_centred_window():
    import music
    rate = 8000
    beats = music.BeatDetector()
    hits = []
    for i in range(90):                                    # 3 s at 30 fps, a hit every 0.5 s
        loud = 0.5 if i % 15 == 0 else 0.03
        hits.append(beats.update(loud * np.ones(400)))
    assert [i for i, h in enumerate(hits) if h] == [15, 30, 45, 60, 75]

    class FakeSong(music.Song):
        def __init__(self):
            self.rate, self.samples = rate, np.arange(rate * 2, dtype=float)
    w = FakeSong().window(1.0, 0.1)
    assert w[0] == rate - 400 and w[-1] == rate + 399        # centred on the playhead


def test_sensitivity_and_reactivity():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import live
    from app import App
    d = tempfile.mkdtemp()
    app = App(programs_file=os.path.join(d, "p.json"), sound=False, settings_file=os.path.join(d, "s.json"))
    app.dt = 1 / 30
    start = (app.plate.n, app.plate.m)
    quiet = 0.005 * np.ones(800)                          # well under the gate at 50% sensitivity
    for _ in range(10):
        app.follow_sound(523.0, quiet)
    assert (app.plate.n, app.plate.m) == start             # too quiet to count
    app.param("Sensitivity").value = 1.0                  # 10x more sensitive: now it counts
    for _ in range(10):
        app.follow_sound(523.0, quiet)
    assert (app.plate.n, app.plate.m) == live.nearest_mode(523.0, 80)
    drive = app.plate.drive
    app.param("Reactivity").value = 0.0                   # steady shaking whatever the volume
    app.follow_sound(523.0, quiet)
    assert app.plate.drive > drive and app.plate.drive == live.resonance(523.0, app.plate.n, app.plate.m, 80)


def test_scope_xy_lissajous():
    import os
    import tempfile
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import scope
    from app import App
    d = tempfile.mkdtemp()
    app = App(programs_file=os.path.join(d, "p.json"), sound=False, settings_file=os.path.join(d, "s.json"))
    app.show(1)
    app.param("Mode").value = scope.MODES.index("X-Y")
    app.param("CH2 Wave").value = scope.WAVES.index("Sine")
    sc = app.scope
    # Equal sines in phase draw a diagonal line; 90 degrees apart, a circle of 2 div radius.
    x, y = sc.xy_trace(0.3, 2000)
    assert np.allclose(x, y)
    app.param("CH2 Phase").value = 90
    x, y = sc.xy_trace(0.3, 2000)
    assert np.allclose(np.hypot(x, y), 2.0)            # 1 V at 0.5 V/div
    # 2:3 is read off the side panel, and the frame draws without error.
    app.param("CH2 Frequency").value = 750.0
    assert scope.ratio(500.0, 750.0) == "2:3" and scope.ratio(500.0, 751.3) == "~2:3"
    assert scope.ratio(1000.0, 3141.6) is None
    app.draw()
    # The White theme has no colour anywhere: both channels are white.
    import ui
    app.param("Theme").value = ui.THEME_NAMES.index("White") - 1
    app.manual_adjust(app.param("Theme"), 1)
    assert app.value("Theme") == ui.THEME_NAMES.index("White") and ui.T.name == "White"
    assert scope.COLOURS["1"] == scope.COLOURS["2"] == (255, 255, 255)
    for k, v in ui.THEMES["White"].items():
        for c in (v.values() if isinstance(v, dict) else [v]):
            assert c is None or c[0] == c[1] == c[2], k
    app.draw()
    ui.T.use("Amber")
    # Phase is only for the generator: hidden on an audio channel.
    app.open("Channel 2", lambda: app.scope_rows("2"))
    assert "Phase" in [r.label for r in app.menu.rows()]
    app.param("CH2 Wave").value = scope.WAVES.index("Live mic")
    assert "Phase" not in [r.label for r in app.menu.rows()]
    # CH1 phase doesn't break the Y-T trigger: sweeps still start at 0 V rising.
    app.param("Mode").value = 0
    app.param("CH1 Phase").value = 45
    t, v = sc.trace("1", 0.37, 200)
    assert abs(v[0]) < 1e-9 and v[1] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
