"""Menus for the two-knob interface.

A menu is a titled list of rows. K1 turns move between rows and K1 press
activates a row; K2 turns change a row's value and K2 press goes back.
Rows are rebuilt from a function every time they're needed, so a menu
always reflects the current programs and states.
"""


class Row:
    def __init__(self, label, value=None, adjust=None, press=None, hint=None, badge=None):
        self.label = label
        self.badge = badge    # small status tag right of the label (e.g. "on")
        self.hint = hint      # small text right of the label (e.g. the step size)
        self.value = value or (lambda: "")
        self.adjust = adjust  # called with +/- clicks from K2
        self.press = press    # called on K1 press


class Menu:
    def __init__(self, title, build):
        self.title = title
        self.build = build
        self.selected = 0
        self.top = 0  # first visible row when the list scrolls

    def rows(self):
        rows = self.build()
        self.selected = min(self.selected, len(rows) - 1)
        return rows

    def move(self, clicks):
        self.selected = (self.selected + clicks) % len(self.rows())

    def current(self):
        return self.rows()[self.selected]

    def scroll_to_selection(self, visible):
        if self.selected < self.top:
            self.top = self.selected
        elif self.selected >= self.top + visible:
            self.top = self.selected - visible + 1
