"""Input for the two knobs.

On a desktop the keyboard stands in for the encoders:
  K1 turn (select row):     Up / Down
  K1 press (open / run):    Enter
  K2 turn (change value):   Left / Right
  K2 press (back):          Backspace or Space  (on the main screen: re-scatter)
  R: re-scatter the sand    H: toggle heatmap    Esc / Q: quit

On the Pi, `Encoders` reads two rotary encoders over GPIO with gpiozero, with
the same meaning for each turn and press. Pins are BCM numbers and are
placeholders until the wiring is decided.
"""

import pygame

# Events the app understands.
SELECT, ADJUST, PRESS, BACK = "select", "adjust", "press", "back"
SCATTER, HEATMAP, QUIT = "scatter", "heatmap", "quit"

K1_PINS = dict(a=17, b=18, button=27)
K2_PINS = dict(a=22, b=23, button=24)


def keyboard_actions(event):
    """Translate one pygame event into (action, amount) pairs."""
    if event.type == pygame.QUIT:
        return [(QUIT, 0)]
    if event.type != pygame.KEYDOWN:
        return []
    keymap = {
        pygame.K_UP: (SELECT, -1), pygame.K_DOWN: (SELECT, 1),
        pygame.K_LEFT: (ADJUST, -1), pygame.K_RIGHT: (ADJUST, 1),
        pygame.K_RETURN: (PRESS, 0), pygame.K_KP_ENTER: (PRESS, 0),
        pygame.K_BACKSPACE: (BACK, 0), pygame.K_SPACE: (BACK, 0),
        pygame.K_r: (SCATTER, 0), pygame.K_h: (HEATMAP, 0),
        pygame.K_ESCAPE: (QUIT, 0), pygame.K_q: (QUIT, 0),
    }
    return [keymap[event.key]] if event.key in keymap else []


class Encoders:
    """Polls two rotary encoders. Falls back to nothing if gpiozero is missing."""

    def __init__(self):
        self.available = False
        try:
            from gpiozero import Button, RotaryEncoder
        except ImportError:
            return
        try:
            self.k1 = RotaryEncoder(K1_PINS["a"], K1_PINS["b"], max_steps=0)
            self.k2 = RotaryEncoder(K2_PINS["a"], K2_PINS["b"], max_steps=0)
            self.buttons = [Button(K1_PINS["button"]), Button(K2_PINS["button"])]
        except Exception:
            # Not on a Pi, or no GPIO access.
            return
        self._last = [0, 0]
        self._pressed = [False, False]
        self.available = True

    def poll(self):
        if not self.available:
            return []
        actions = []
        for i, (enc, action) in enumerate([(self.k1, SELECT), (self.k2, ADJUST)]):
            delta = enc.steps - self._last[i]
            if delta:
                self._last[i] = enc.steps
                actions.append((action, delta))
        for i, action in enumerate((PRESS, BACK)):
            pressed = self.buttons[i].is_pressed
            if pressed and not self._pressed[i]:
                actions.append((action, 0))
            self._pressed[i] = pressed
        return actions
