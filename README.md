# Project Resonance: software

Chladni plate sand simulation in NumPy + Pygame, laid out for the 4.3" 800x480 DSI screen:
the 480x480 plate in the middle, parameters on the left, mode info on the right.

```
pip install -r requirements.txt
python3 app.py                # windowed
python3 app.py --fullscreen   # on the Pi
python3 test_chladni.py       # sanity checks
```

## Controls

| Knob | Keyboard | Does |
|---|---|---|
| K1 turn | Up / Down | select a row |
| K1 press | Enter | open / run the selected row; on a plain value, change its step size (coarse, finer, finest, back to coarse) |
| K2 turn | Left / Right | change the selected value |
| K2 press | Backspace / Space | back (on the Display menu: throw the sand back on the plate) |
| | R | throw the sand back on the plate |
| | H | toggle the vibration heatmap behind the sand |
| | Esc / Q | quit |

Values with a wide range show their current step size right of the label (e.g. Particles ±1000). K1 press on
them cycles it: Particles 1000 / 100 / 10, Uniformity, Mix and Volume 10% / 1% (0.1 / 0.01 for Mix),
Mass 0.25 / 0.05, Base pitch and Hold 10 / 1, scope Frequency 1000 / 100 / 10 / 1 Hz,
Amplitude 1 / 0.1 / 0.01 V, Position 1 / 0.1 div. The step always starts at the coarsest.

## Displays

The top menu, **Display**, has three rows:

```
Display
  Oscilloscope   Mode, Time/div, Channel 1, Channel 2, Music
  Resonance      Source, Channel A, Channel B, Programs, Music
  Settings       shared by both displays
```

Opening **Oscilloscope** or **Resonance** (K1 press) switches the screen to that display; the one showing is
marked "showing". The plate pauses while the scope is showing, and the display you were on is remembered next start.

## Parameters

**Resonance > Source** (shared by everything on the plate):

- **Uniformity** 0 to 100%. Random background jitter per grain: lower is fuzzier, and 0% is properly chaotic (it
  shows through music too, since it only partly calms down in quiet passages).
- **Adherence** 25% to 300% (default 100%). How strongly the sand is pulled onto its lines, with no added
  randomness: high snaps it into thin lines and follows every note change tightly; low drifts there lazily, so
  with music you see soft blends of the last few patterns. Saved in program states.
- **n, m** 1 to 12, the vibration mode.
- **Mix** −1 to +1, the weight of the second term: C = T(n,m) + Mix · T(m,n). ±1 are the classic ± forms; values in
  between bend straight nodal lines into wavy ones, as on real plates. n and m can be 0 (with free edges, cos 0 = 1
  gives straight-line modes). The right panel says when a setting leaves the plate with no nodal lines.
- **Edges** Free (default) uses C = cos(nπx)cos(mπy) + Mix · cos(mπx)cos(nπy), like a real plate bolted at its centre.
  Fixed uses the sin form from the sketch. With fixed edges the border is a nodal line in every mode,
  so sand slowly collects around the rim as you change modes (about 70% of it after a few changes).

**Channel A** and **Channel B** (K1 press to open) are two materials on the same plate, each with:

- **Show** On / Off. Off hides the channel (its sand isn't drawn or moved) but keeps its count and settings.
  Channel A starts on (white, 1 px); channel B starts hidden.
- **Particles** 0 to 20000 (0 turns the channel off). Defaults: A 6000, B 2500.
- **Type** *Sand* collects on the nodal lines. *Powder* is very fine dust, which air currents above the plate
  carry to the antinodes instead (Faraday, 1831), so it forms heaps in the spaces between the lines.
  A starts as sand and B as powder, so both patterns show at once.
- **Mass** 0.25x to 4x. Heavier grains settle more slowly into thinner, sharper lines; lighter ones make wide, fuzzy lines.
- **Colour** Sand, Cyan, Rose, Lime or White.
- **Size** 1 to 6 px, how big each grain is drawn. Display only; the physics is the same. Bigger dots cost a little
  frame rate, so with very high particle counts on the Pi keep it small.

## Programs

A program is a list of states S1 ... Sn. Each state stores n, m, Mix, Edges, Uniformity, each channel's
Show (On / Off: hide channel A, B or both in that state), Type, Mass, Colour and Size, plus a Hold time in seconds and a **Reset** switch (On throws the sand back on the plate when the state starts, so the
pattern forms from scratch), and a running program cycles S1 -> S2 -> ... -> Sn -> S1 forever. Particle
counts are not part of a state. Two programs are built in: a short "Tour", and "Reference tour", which steps through all 40 figures of the
project's reference sheet in reading order (1 s each, 40 s in all). It runs channel A only, as plain sand with 12000
particles drawn at 1 px, and hides channel B. Four of those figures are rings that only
the fixed-edge form makes, so the tour switches Edges for them and resets the sand on entering and leaving them.
A few figures came from real plates and are approximations. Built-in programs added by an update appear once;
delete one and it stays deleted.

Open **Resonance > Programs** (K1 press):

- **Run (set):** pick a program, then *Run*. You go back to the Resonance menu and the right panel shows the
  current state and the time until the next one. Turning a parameter by hand, or *Stop* in Programs, ends it.
- **Create:** *+ New program* makes a program whose S1 is whatever is on the plate now.
- **Edit:** open a program to see its states. Open a state to change its values with K2; the plate previews it
  live. *+ Add state* copies the current plate, so you can dial a pattern in by hand first and then add it.
  Deleting a state or a program takes two presses.

Programs are saved to `programs.json` next to the app (atomically, so a power cut can't corrupt it).
Run one at startup with `python3 app.py --run "Reference tour"`.

## Music

**Resonance > Music** (K1 press) lists the songs in the `music/` folder next to the app (mp3, ogg, wav or
flac; `--music DIR` uses another folder). Pick one to play it; the rest of the folder follows in order and loops.

While a song plays, its strongest note is found about 30 times a second and the plate switches to the mode whose
tone is closest (the same mapping as live mode, using **Base pitch**). The analysis window is centred on what
you're hearing (the whole song is in memory), so the plate follows the music within about a tenth of a second.
The sand is shaken harder the louder the music (measured over 50 ms, so it pulses with the rhythm) and the
closer the note is to that mode, so quiet passages freeze the pattern. Beats (sudden jumps in loudness, in
music and live mode) give the sand an extra jolt. Because songs are mostly
chords, notes are found with a harmonic product spectrum, which picks the note whose harmonics line up best:
melodies and bass lines with clear notes work best, dense or drum-heavy tracks will jump around more.

Three things stop the sand bundling up into a few blobs over a song (they apply in live mode too):

- **Settling.** The mode only changes once a new note has held for 0.1 s, so the briefest passing notes and
  detection glitches don't drag the sand around.
- **Shake on change.** Every mode change gives the sand a short jolt that knocks loose grains sitting on the old
  nodal lines (spots shared by many modes would otherwise trap sand forever).
- **Crowding.** Overcrowded spots push their sand outward, like a real pile that can only get so tall; on a
  nodal line this spreads blobs back out along the line. This is always on, so patterns are a little more even
  everywhere.

The mode tone is off while music plays. *Stop* in the Music menu, turning n or m by hand, running a program or
turning on live mode stops the song. Each song is decoded into memory when it starts, which takes a moment
(about half a second for a 3 minute MP3 on a desktop, longer on the Pi).

## Oscilloscope

A 10 x 8 division graph with labeled axes: time along the bottom, CH1 volts on the left (yellow) and CH2 volts
on the right (blue), each scale following that channel's Volts/div and Position. A small arrow marks each
channel's 0 V line. The trace is triggered on CH1's rising zero crossing, so it stands still.

- **Mode** Y-T (each channel against time) or X-Y (Lissajous, below).
- **Time/div** horizontal scale, shared by both channels: 0.1 ms to 50 ms per division.
- **Channel 1 / Channel 2** (K1 press) each have:
  - **Signal** Off, Sine, Square, Triangle, Live mic or Music.
  - **Frequency** 1 Hz to 20 kHz, in 1000 / 100 / 10 / 1 Hz steps (defaults 440 Hz and 660 Hz).
  - **Amplitude** 0.1 to 5 V peak. For Live mic and Music it's **Gain** instead (the raw signal is ±1 at full
    scale) and Frequency is hidden, since the sound sets it.
  - **Volts/div** vertical scale, 10 mV to 5 V per division.
  - **Position** moves the trace up or down, -4 to +4 divisions.
  - **Phase** 0 to 359 degrees, in 15 / 1 degree steps, for the generator (hidden for audio).

**Live mic** opens the microphone (it doesn't need the Live mic setting, which is for the plate).
**Music** shows the song playing: pick one from **Oscilloscope > Music** (the same list as Resonance > Music).
With audio on CH1 the sweep starts at its latest rising zero crossing, so steady notes stand still.

The right panel reads out each channel's frequency, period, peak-to-peak voltage, scale and offset; for audio,
the detected pitch and the measured Vpp.
The tone is silent on the scope display.

**X-Y mode** plots CH1 across and CH2 up on a square 8 x 8 graticule, as a real scope does with two signal
generators. Sines at simple frequency ratios draw Lissajous figures: 1:1 is a line, or a circle with CH2 Phase
at 90 degrees; 2:3 and 3:4 make knots. Detune one channel slightly and the figure rolls. Time/div sets how much
of the recent signal is drawn (10 divisions' worth), the oldest part fading like scope phosphor. The right panel
shows the ratio ("2:3", "~2:3" when slightly detuned, "free" otherwise). Audio works too: mic on one channel and
a generator on the other.

## Settings

**Settings** on the Display menu (K1 press) holds general options shared by both displays. They're saved to `settings.json` next to the app,
and aren't part of program states.

- The display showing (Oscilloscope or Resonance) is saved here too.
- **Sound** On / Off. A sine at the frequency that would drive the current mode. On a square plate it scales with
  n² + m², so f = Base pitch × (n² + m²) / 2, capped at 6 kHz. When the mode changes the pitch glides to the new
  note, and the right panel shows the tone frequency.
- **Volume** 0 to 100% (default 50%).
- **Base pitch** 40 to 160 Hz (default 80): the tone of mode (1, 1). Used by both the tone and live mode.
- **Live mic** On / Off. See below.
- **Sensitivity** 0 to 100% (default 50%): how loud a note must be to count, in music and live mode. Notes
  quieter than a tenth of "full" loudness are ignored (the pattern holds). 0% needs sound 10x louder than 50%,
  100% responds to sound a tenth as loud; raise it for a quiet microphone.
- **Reactivity** 0 to 100% (default 100%): how much the loudness and beats drive the shaking. 0% shakes steadily
  whatever the volume (only pitch matters); 100% follows the volume closely, with full beat jolts.
- **Theme** Amber (dark panels, amber highlights), Mono (black, white outlines and greys; orange CH1 and
  blue CH2 on the scope) or White (the same with no colour at all: both scope channels, their labels and the
  heatmap are white and grey).

On the Pi, plug a powered speaker (or a small amp board and speaker) into the 3.5 mm jack; you may need to pick
the headphone jack in `raspi-config` (System Options, Audio). Without a sound device the Sound row says
"no device" and the app runs silently. `--mute` skips the sound device entirely.

### Live mic

With **Live mic** on, the plate listens to a microphone instead of following the knobs or a program:

1. The dominant pitch is found about 30 times a second and smoothed over half a second, so a held note holds
   a mode.
2. The plate switches to the mode whose tone is closest to that pitch (so with an 80 Hz base, humming 200 Hz
   gives (1, 2) and whistling 1360 Hz gives (3, 5)).
3. The sand is shaken less the further you are from that mode's exact pitch, and not at all in silence, so
   the pattern freezes when you stop.

Steady sounds (humming, whistling, a held instrument note, a tone generator app) make clear patterns; speech and
music change pitch too fast and mostly blur. The tone is muted while live mode is on, otherwise the mic would hear
the speaker. Turning n or m by hand, or running a program, switches live mode off.

The Pi's audio jack is output only, so live mode needs a USB microphone (or a USB sound card with a mic input).
It also needs the `sounddevice` package and PortAudio:

```
sudo apt install libportaudio2
pip install sounddevice
```

Without them, or without a microphone, the right panel says "no mic found".

## Files

- `chladni.py` the equation (free and fixed edge forms), its analytic gradient, grid sampling, relative frequency (∝ n² + m²)
- `particles.py` `SandPlate`: one channel of grains. Sand is kicked in proportion to |C| and drifts down ∇(C²) onto nodal lines; powder drifts up it to the antinodes
- `params.py` parameter ranges and steps
- `program.py` program storage and the sequencer that cycles states
- `menu.py` the menu rows the knobs move through
- `audio.py` the mode tone
- `live.py` live mode: microphone input, pitch detection and pitch-to-mode mapping
- `music.py` Music mode: song decoding and note detection
- `scope.py` oscilloscope display: waveforms, scales and the labeled graph
- `ui.py` colours, fonts and drawing helpers for the panels
- `fonts/` Departure Mono, a pixel font (SIL Open Font License, `fonts/LICENSE-DepartureMono.txt`), bundled so the Pi looks the same
- `settings.py` saving and loading the Settings menu
- `controls.py` keyboard mapping and the gpiozero encoder reader
- `app.py` the Pygame display
