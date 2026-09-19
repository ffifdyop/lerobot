# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Display-independent keyboard input for interactive controls.

This module centralizes everything related to *discrete* keyboard controls
(end-episode-early, re-record, stop, and the rollout strategies' custom keys):

* environment detection — :func:`is_headless`, :func:`is_wayland`,
  :func:`pynput_can_capture` (the single predicate every call-site should use to
  decide whether ``pynput`` can actually capture keys here);
* a shared key mapping — :func:`apply_recording_control`; and
* two interchangeable backends behind one ``(listener, events)`` contract:
  the ``pynput`` global listener (X11 / trusted-macOS / Windows) and a
  standard-library :class:`TerminalKeyListener` that reads the controlling TTY
  (Wayland / headless-SSH-with-TTY / macOS without Accessibility permission).

NOTE: *continuous* key-state teleoperation ("hold a key to keep moving") is
deliberately NOT served here. A terminal in cbreak mode delivers only key-down
bytes — there is no key-release event — so the held-key model cannot be
reproduced. Those teleoperators stay on ``pynput`` and use
:func:`pynput_can_capture` to warn instead of silently doing nothing.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import platform
import select
import sys
import threading
import time
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

from .import_utils import _pynput_available

logger = logging.getLogger(__name__)

# POSIX-only terminal modules (absent on Windows, where the pynput backend is used).
if TYPE_CHECKING:
    import termios
    import tty

    _TERMIOS_AVAILABLE = True
else:
    try:
        import termios
        import tty

        _TERMIOS_AVAILABLE = True
    except ImportError:  # POSIX-only modules; unavailable on Windows
        termios = tty = None
        _TERMIOS_AVAILABLE = False

keyboard = None
if _pynput_available:
    try:
        from pynput import keyboard
    except Exception as e:  # e.g. no reachable X display on a headless Linux box
        logger.info("Could not import pynput keyboard backend: %s", e)


@cache
def is_headless() -> bool:
    """Return ``True`` when no display server is available.

    * Linux: headless when neither ``DISPLAY`` (X11) nor ``WAYLAND_DISPLAY`` is set.
    * macOS / Windows: a display is always assumed to be present. A genuinely GUI-less
    Mac/Windows CI host would be misclassified but it doesn't matter, because the
    sys.stdin.isatty() gate returns None there regardless.
    """
    if platform.system() == "Linux":
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


@cache
def is_wayland() -> bool:
    """Return ``True`` when running under a Wayland session.

    ``pynput`` relies on an X11 backend. Under Wayland it still imports (XWayland
    is usually present and ``$DISPLAY`` is set) but cannot capture *global*
    hotkeys, so the documented arrow/Esc shortcuts silently do nothing. This case
    is invisible to :func:`is_headless`, hence the dedicated check.
    """
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(
        os.environ.get("WAYLAND_DISPLAY")
    )


@cache
def pynput_can_capture() -> bool:
    """Return ``True`` when a ``pynput`` global listener can actually capture keys.

    This is the single predicate every keyboard call-site should use to choose
    between the ``pynput`` backend and a fallback. It is intentionally
    conservative:

    * Linux: only a real X11 session (a display is present *and* it is not Wayland).
    * macOS: ``True`` here — Accessibility / Input-Monitoring permission
      (``IS_TRUSTED``) can only be confirmed at runtime *after* starting a
      listener, so :func:`init_keyboard_listener` refines this with
      :func:`pynput_listener_is_trusted`.
    * Windows: ``True`` (the low-level global hook needs no special permission).

    Always ``False`` when ``pynput`` is not installed.
    """
    if not _pynput_available:
        return False
    if platform.system() == "Linux":
        return not is_headless() and not is_wayland()
    return True


def pynput_listener_is_trusted(listener, timeout_s: float = 1.0) -> bool:
    """Best-effort check that a freshly started ``pynput`` listener can capture.

    On macOS, ``pynput`` sets ``listener.IS_TRUSTED`` on its *listener thread*
    once the Quartz event tap is created; the class default is ``False``. We
    therefore wait for the thread to either flip it ``True`` (trusted) or for a
    short timeout to elapse (untrusted — it stays ``False`` forever). This biases
    toward the common trusted case (returns as soon as the flag flips) and only
    pays the full ``timeout_s`` on an already-broken untrusted machine.

    On non-macOS backends the attribute is absent and capture is assumed to work.
    """
    if platform.system() != "Darwin":
        return True
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if getattr(listener, "IS_TRUSTED", False):
            return True
        time.sleep(0.02)
    return bool(getattr(listener, "IS_TRUSTED", False))


def apply_recording_control(control: str, events: dict) -> None:
    """Apply a recording control-flow key press to the shared ``events`` dict.

    Centralizes the mapping so the ``pynput`` and terminal backends behave
    identically. ``control`` is one of ``"right"`` (end the loop early), ``"left"``
    (re-record the last episode), or ``"esc"`` (stop recording).
    """
    if control == "right":
        print("Right arrow key pressed. Exiting loop...")
        events["exit_early"] = True
    elif control == "left":
        print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif control == "esc":
        print("Escape key pressed. Stopping data recording...")
        events["stop_recording"] = True
        events["exit_early"] = True


# Terminal arrow keys arrive as a 3-byte escape sequence whose *final* byte identifies
# the direction. Two encodings exist depending on the terminal's cursor-key mode — CSI
# ("ESC [ X") and SS3 ("ESC O X", common over SSH/tmux) — but both share the same final
# byte, so this single table decodes either. Looked up by TerminalKeyListener._parse;
# an unknown final byte yields None (sequence ignored).
_ARROW_FINAL_BYTES = {"A": "up", "B": "down", "C": "right", "D": "left"}

# Function keys also come in the two encodings, and F1..F4 additionally reuse the
# arrow final bytes in SS3 form. The CSI form (``ESC [ <code> ~``) is the common one.
_FUNCTION_SS3_FINAL = {"P": "f1", "Q": "f2", "R": "f3", "S": "f4"}
# xterm's CSI ``n~`` function-key codes are non-contiguous after F5.  Include the
# extended range because the default View Mask bindings are defined for every
# policy-declared visual slot, not only the first eight cameras.
_FUNCTION_CSI_CODES = {
    11: "f1",
    12: "f2",
    13: "f3",
    14: "f4",
    15: "f5",
    17: "f6",
    18: "f7",
    19: "f8",
    20: "f9",
    21: "f10",
    23: "f11",
    24: "f12",
    25: "f13",
    26: "f14",
    28: "f15",
    29: "f16",
    31: "f17",
    32: "f18",
    33: "f19",
    34: "f20",
}


class TerminalKeyListener:
    """Display-independent keyboard listener that reads keys from the controlling TTY.

    Used as the Wayland / headless / macOS-untrusted equivalent of the ``pynput``
    listener for *discrete* controls. It puts the terminal into cbreak mode with
    echo disabled and reads bytes on a daemon thread, decoding them into logical
    key names that are passed to ``on_key``:

    * arrow keys (``ESC [ C`` / ``ESC O C`` …) -> ``"right"`` / ``"left"`` / ``"up"`` / ``"down"``
    * a bare ``ESC`` -> ``"esc"``
    * Enter / Tab / Space / Backspace -> ``"enter"`` / ``"tab"`` / ``"space"`` / ``"backspace"``
    * any other printable byte -> that character (e.g. ``"n"``, ``"s"``)

    Only key-down events are produced (terminals have no key-release), so this is
    suitable for discrete commands but NOT for continuous "hold-to-move" teleop.

    The terminal is restored on :meth:`stop` and also via an ``atexit`` hook, so a
    crash or Ctrl-C never leaves the shell in a no-echo cbreak state. POSIX-only
    (``termios`` / ``tty`` / ``select``); those modules are imported lazily so this
    file stays importable on Windows (where ``pynput`` is used instead).
    """

    def __init__(self, on_key: Callable[[str], None]):
        self._on_key = on_key
        self._running = False
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_attrs = None

    def _read_char(self, timeout: float) -> str | None:
        """Return one character from stdin within ``timeout`` seconds, or ``None``."""
        if self._fd is None:
            return None
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        try:
            data = os.read(self._fd, 1)
        except OSError:
            return None
        if not data:
            return None
        return data.decode(errors="ignore")

    def _parse(self, ch: str) -> str | None:
        """Decode one (possibly multi-byte) key starting at ``ch`` into a key name."""
        if ch == "\x1b":
            # Possible CSI / SS3 escape sequence (arrow or function keys) or a bare
            # ESC. Use short follow-up reads so a lone ESC is not mistaken for one.
            ch2 = self._read_char(timeout=0.02)
            if ch2 is None:
                return "esc"
            if ch2 == "O":
                # SS3: arrows share their final byte with F1–F4.
                ch3 = self._read_char(timeout=0.02) or ""
                return _ARROW_FINAL_BYTES.get(ch3) or _FUNCTION_SS3_FINAL.get(ch3)
            if ch2 == "[":
                return self._parse_csi()
            # Some other escape sequence (e.g. Alt+key); ignore it.
            return None
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\t":
            return "tab"
        if ch == " ":
            return "space"
        if ch in ("\x7f", "\x08"):
            return "backspace"
        if ch.isprintable():
            return ch
        return None

    def _parse_csi(self) -> str | None:
        """Decode the body of a CSI sequence after ``ESC [``.

        Handles the forms this module emits: a bare final byte (arrows, Home/End)
        and a numeric parameter terminated by ``~`` (function keys).  Anything
        longer or unknown is consumed and ignored rather than leaking into the
        key stream as stray characters.
        """
        buf = ""
        while len(buf) < 8:
            ch = self._read_char(timeout=0.02)
            if ch is None:
                return None
            if ch.isdigit() or ch == ";":
                buf += ch
                continue
            if ch == "~":
                try:
                    code = int(buf.split(";")[-1])
                except ValueError:
                    return None
                return _FUNCTION_CSI_CODES.get(code)
            if not buf:
                return _ARROW_FINAL_BYTES.get(ch)
            return None
        return None

    def _run(self) -> None:
        while self._running:
            ch = self._read_char(timeout=0.05)
            if ch is None:
                continue
            name = self._parse(ch)
            if name is None:
                continue
            try:
                self._on_key(name)
            except Exception as e:  # never let a handler error kill the reader thread
                logger.debug("Terminal key handler error: %s", e)

    def start(self) -> None:
        """Switch the terminal to cbreak mode (echo off) and read keys on a daemon thread.

        No-op when stdin is not a TTY (piped/redirected input) or on platforms
        without ``termios`` (e.g. Windows), so non-interactive runs are unaffected.
        """
        if not sys.stdin.isatty():
            return
        if not _TERMIOS_AVAILABLE:  # POSIX-only modules (e.g. unavailable on Windows)
            logger.warning("Terminal keyboard input is not supported on this platform.")
            return

        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        # Explicitly disable ECHO so arrow-key escape sequences (e.g. ^[[C) are not
        # echoed as garbage into the recording terminal. (Independent of the
        # version-specific behavior of tty.setcbreak.)
        new_attrs = termios.tcgetattr(self._fd)
        new_attrs[3] &= ~termios.ECHO  # index 3 == lflags
        termios.tcsetattr(self._fd, termios.TCSADRAIN, new_attrs)
        # Safety net: restore the terminal even if stop() is never reached (crash).
        atexit.register(self.stop)

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the reader thread and restore the original terminal attributes.

        Idempotent: safe to call multiple times (e.g. explicitly and via atexit).
        """
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
            self._thread = None
        if self._fd is not None and self._old_attrs is not None and _TERMIOS_AVAILABLE:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            finally:
                self._old_attrs = None
        with contextlib.suppress(Exception):
            atexit.unregister(self.stop)


# Map pynput key objects to the same canonical names TerminalKeyListener emits, so a
# single dispatch works across both backends. Empty when pynput is unavailable.
if keyboard is not None:
    _PYNPUT_KEY_NAMES = {
        keyboard.Key.right: "right",
        keyboard.Key.left: "left",
        keyboard.Key.up: "up",
        keyboard.Key.down: "down",
        keyboard.Key.esc: "esc",
        keyboard.Key.enter: "enter",
        keyboard.Key.tab: "tab",
        keyboard.Key.space: "space",
        keyboard.Key.backspace: "backspace",
    }
    # Function keys: the default View Mask hotkeys (F1, F2, F3…).  Added here rather
    # than in a view-ablation-only map so both backends spell them identically.
    for _index in range(1, 21):
        _fn_key = getattr(keyboard.Key, f"f{_index}", None)
        if _fn_key is not None:
            _PYNPUT_KEY_NAMES[_fn_key] = f"f{_index}"
else:
    _PYNPUT_KEY_NAMES = {}


def _resolve_pynput_key(key) -> str | None:
    """Resolve a pynput key event to the canonical name TerminalKeyListener also emits.

    Special keys map through :data:`_PYNPUT_KEY_NAMES`; character keys fall back to their
    ``.char`` (e.g. ``"n"``). Returns ``None`` for keys with no mapping and no character.
    """
    name = _PYNPUT_KEY_NAMES.get(key)
    if name is not None:
        return name
    # ``or None`` keeps the historical truthy-char semantics: an empty/None char is "no key".
    return getattr(key, "char", None) or None


def create_key_listener(dispatch: Callable[[str], None], *, controls_help: str = ""):
    """Start a keyboard listener that routes resolved key names to ``dispatch``.

    Shared backend selection used by recording and the rollout strategies:

    * the ``pynput`` global listener on X11 / trusted-macOS / Windows (on macOS the
      listener's ``IS_TRUSTED`` flag is checked after start, and an untrusted listener is
      stopped so the terminal backend is used instead);
    * the stdlib :class:`TerminalKeyListener` on Wayland / headless sessions with a TTY;
    * ``None`` when no backend is usable (non-interactive / piped runs).

    Both backends pass ``dispatch`` the same canonical key names ("right" / "left" / "up" /
    "down" / "esc" / "enter" / "tab" / "space" / "backspace", or a character), so one
    ``dispatch`` works regardless of backend. ``controls_help`` is an optional hint
    appended to the log messages.

    Returns the listener (exposing ``.stop()``) or ``None``.
    """
    suffix = f" ({controls_help})" if controls_help else ""

    if pynput_can_capture() and keyboard is not None:

        def on_press(key):
            with contextlib.suppress(Exception):
                name = _resolve_pynput_key(key)
                if name is not None:
                    dispatch(name)

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        if pynput_listener_is_trusted(listener):
            logger.info("Keyboard listener started%s.", suffix)
            return listener
        # macOS without Accessibility / Input-Monitoring permission: the listener never
        # fires. Stop it and fall through to the terminal backend.
        logger.warning(
            "pynput keyboard listener is not trusted (missing macOS Accessibility / "
            "Input Monitoring permission); falling back to terminal keyboard input."
        )
        listener.stop()

    if sys.stdin.isatty():
        listener = TerminalKeyListener(dispatch)
        listener.start()
        logger.info("Using terminal keyboard input — keep this terminal focused%s.", suffix)
        return listener

    logger.warning(
        "Keyboard controls unavailable: no usable display (Wayland/headless) and stdin is "
        "not an interactive terminal%s.",
        suffix,
    )
    return None


class LatchingKeyListener:
    """Wraps a key listener so a *held* key produces exactly one dispatch.

    Discrete controls want every key-down; latch-style controls (the View Mask
    hotkeys) want one event per independent press.  This filters the auto-repeat
    that ``pynput`` emits while a key is held, using the key-release events pynput
    provides — so F1 held down toggles its view once, and only a genuine second
    press toggles it back.

    Backends without key-release events (the terminal fallback) use a conservative
    repeat window.  Terminal input cannot represent key-up, so a second independent
    press must be separated from the first by that window; this is preferable to
    allowing the OS auto-repeat stream to toggle the mask continuously.
    """

    def __init__(
        self,
        dispatch: Callable[[str], None],
        releases_available: bool,
        terminal_repeat_window_s: float = 0.75,
    ) -> None:
        self._dispatch = dispatch
        self._releases_available = releases_available
        self._latched: set[str] = set()
        self._released: set[str] | None = set() if releases_available else None
        self._last_terminal_press: dict[str, float] = {}
        self._terminal_repeat_window_s = terminal_repeat_window_s
        self._lock = threading.Lock()

    def on_press(self, key) -> None:
        """Deliver a key-down exactly once per latch cycle."""
        name = key if isinstance(key, str) else _resolve_pynput_key(key)
        if name is None:
            return
        with self._lock:
            if self._releases_available:
                if name in self._latched:
                    return  # auto-repeat while held
            else:
                now = time.monotonic()
                previous = self._last_terminal_press.get(name)
                # Refresh on suppressed repeats too.  Terminal input has no key-up
                # event, so continuous auto-repeat must remain one latch cycle even
                # when the hold lasts longer than the debounce window.
                self._last_terminal_press[name] = now
                if previous is not None and now - previous < self._terminal_repeat_window_s:
                    return  # terminal auto-repeat while the key is held
            self._latched.add(name)
        self._dispatch(name)

    def on_release(self, key) -> None:
        """Clear the latch so the next press is a new toggle."""
        name = key if isinstance(key, str) else _resolve_pynput_key(key)
        if name is None:
            return
        with self._lock:
            self._latched.discard(name)
            self._last_terminal_press.pop(name, None)

    @property
    def supports_releases(self) -> bool:
        """Whether held-key auto-repeat is actually filtered on this backend."""
        return self._releases_available


def create_latching_key_listener(
    dispatch: Callable[[str], None],
    *,
    controls_help: str = "",
    pynput_factory=None,
    terminal_factory=None,
    terminal_enabled: bool = True,
):
    """Start a key listener whose key-downs are deduplicated while a key is held.

    Same backend selection as :func:`create_key_listener`, plus key-release
    tracking on pynput.  Used by the View Mask hotkeys, whose semantics are
    "one press toggles once".  Returns the listener (exposing ``.stop()``) or
    ``None`` when no backend is usable.

    ``pynput_factory`` / ``terminal_factory`` exist so a single-backend test can
    drive the latch logic without a display; production callers omit them.
    """
    suffix = f" ({controls_help})" if controls_help else ""

    if pynput_factory is not None or (pynput_can_capture() and keyboard is not None):
        listener_cls = keyboard.Listener if pynput_factory is None else pynput_factory
        latch: LatchingKeyListener | None = None

        if pynput_factory is None:
            latch = LatchingKeyListener(dispatch, releases_available=True)
            listener = listener_cls(on_press=latch.on_press, on_release=latch.on_release)
            listener.start()
            trusted = pynput_listener_is_trusted(listener)
            if trusted:
                logger.info("Keyboard listener started%s.", suffix)
                return listener
            logger.warning(
                "pynput keyboard listener is not trusted (missing macOS Accessibility / "
                "Input Monitoring permission); falling back to terminal keyboard input."
            )
            listener.stop()
        else:
            latch = LatchingKeyListener(dispatch, releases_available=False)
            listener = listener_cls(on_press=latch.on_press)
            listener.start()
            logger.info("Keyboard listener started%s.", suffix)
            return listener

    if terminal_enabled and (terminal_factory is not None or sys.stdin.isatty()):
        latch = LatchingKeyListener(dispatch, releases_available=False)
        listener = (
            terminal_factory(latch.on_press)
            if terminal_factory is not None
            else TerminalKeyListener(latch.on_press)
        )
        listener.start()
        logger.info("Using terminal keyboard input — keep this terminal focused%s.", suffix)
        return listener

    if not terminal_enabled:
        logger.info(
            "Terminal View Mask hotkeys are disabled because stdin is reserved for the "
            "interactive command reader%s.",
            suffix,
        )
    else:
        logger.warning(
            "Keyboard controls unavailable: no usable display (Wayland/headless) and stdin is "
            "not an interactive terminal%s.",
            suffix,
        )
    return None


def init_keyboard_listener():
    """Initialize a non-blocking keyboard listener for interactive recording controls.

    Backend selection:

    * ``pynput`` global listener when :func:`pynput_can_capture` is true (real
      X11, macOS, Windows). On macOS the listener's ``IS_TRUSTED`` flag is checked
      after start; if the process lacks Accessibility / Input-Monitoring
      permission, the listener is stopped and the terminal backend is used.
    * a :class:`TerminalKeyListener` reading the controlling TTY when ``pynput``
      cannot capture (Wayland / headless-SSH / macOS-untrusted) *and* stdin is a TTY.
    * otherwise no listener (non-interactive / piped runs) — recording relies on
      the episode/reset timers (or Ctrl+C).

    Both backends accept the same controls: Right/Left/Esc, plus the single-byte letter
    equivalents ``n`` (next), ``r`` (re-record) and ``q`` (quit). The letters are the most
    reliable choice over high-latency SSH/VNC links, where arrow-key escape sequences can
    be split, delayed, or intercepted by the terminal.

    Returns:
        A tuple ``(listener, events)`` where ``listener`` exposes ``.stop()`` or is
        ``None``, and ``events`` is the dict of flags (``exit_early``,
        ``rerecord_episode``, ``stop_recording``) set by key presses.
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    # Accept the single-byte letter equivalents n/r/q alongside the arrow/Esc keys: the
    # letters are immune to the escape-sequence split/delay/interception that affects arrows
    # over laggy SSH/VNC links. Case-insensitive so Shift+letter still works.
    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("right", "n"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", events)
        # other keys (incl. up/down) are intentionally ignored

    listener = create_key_listener(on_key, controls_help="Right/Left/Esc, or n=next, r=re-record, q=quit")
    return listener, events
