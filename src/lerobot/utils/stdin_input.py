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

"""Non-blocking stdin commands with optional TTY space hotkeys.

Unlike :mod:`lerobot.utils.keyboard_input`, which reads single raw bytes in cbreak
mode for hotkeys, :class:`StdinCommandListener` also assembles whole typed lines. By
default it leaves the terminal in canonical mode; an optional space callback enables
cbreak editing and hotkeys in the same reader. These listeners cannot share stdin.
The listener must be
the stream's *sole* consumer: it reads the file descriptor directly with ``os.read``,
so bytes already buffered by e.g. an earlier ``input()`` call are invisible to it.

Reading works over SSH, headless (no display server, unlike the ``pynput`` keyboard
backend) and from piped stdin.
"""

from __future__ import annotations

import atexit
import codecs
import logging
import os
import select
import sys
from collections.abc import Callable
from threading import Lock, Thread, current_thread
from time import monotonic
from typing import IO, Any

if os.name == "posix":
    import termios
    import tty

logger = logging.getLogger(__name__)


class StdinCommandListener:
    """Daemon thread that reads input lines and forwards them to a callback.

    On POSIX the reader polls the stream with ``select`` so ``stop()`` can end the
    thread promptly; elsewhere (or for file-like objects without a file descriptor) it
    falls back to a blocking ``readline`` daemon thread that dies with the process.
    Blank lines are skipped; end-of-file and unexpected read errors trigger ``on_eof``,
    so a dead command channel never leaves the consumer waiting.
    """

    # TTYs expose key-down bytes only. Suppress common auto-repeat bursts; a fresh
    # tap after this quiet interval is eligible again, even across a mode change.
    _SPACE_REPEAT_GAP_S = 0.75

    def __init__(
        self,
        on_line: Callable[[str], None],
        on_eof: Callable[[], None] | None = None,
        stream: IO[str] | IO[bytes] | None = None,
        poll_interval_s: float = 0.2,
        *,
        on_space: Callable[[], None] | None = None,
    ) -> None:
        self._on_line = on_line
        self._on_eof = on_eof
        self._on_space = on_space
        self._terminal_state: tuple[int, list[Any]] | None = None
        self._terminal_lock = Lock()
        self._last_space_at = float("-inf")
        # sys.stdin can itself be None (pythonw, daemonized processes).
        self._stream = stream if stream is not None else sys.stdin
        self._poll_interval_s = poll_interval_s
        self._running = False
        self._thread: Thread | None = None
        self._use_select = False
        if os.name == "posix":
            try:
                self._stream.fileno()
                self._use_select = True
            except (OSError, ValueError, AttributeError):
                pass

    @property
    def space_hotkey_enabled(self) -> bool:
        """Whether the active input stream supports immediate space-key handling."""
        return self._terminal_state is not None

    def _enable_hotkeys(self) -> None:
        if not self._use_select or self._on_space is None:
            return
        fd = self._stream.fileno()
        if not os.isatty(fd):
            return
        self._terminal_state = (fd, termios.tcgetattr(fd))
        try:
            # TCSANOW preserves already typed input; ISIG stays set for Ctrl-C.
            tty.setcbreak(fd, termios.TCSANOW)
        except BaseException:
            self._restore_terminal()
            raise
        atexit.register(self._restore_terminal)

    def _restore_terminal(self) -> None:
        with self._terminal_lock:
            state, self._terminal_state = self._terminal_state, None
            if state is not None:
                try:
                    termios.tcsetattr(state[0], termios.TCSANOW, state[1])
                except (OSError, termios.error):
                    logger.warning("Could not restore closed/unavailable input terminal")
        atexit.unregister(self._restore_terminal)

    def start(self) -> None:
        """Start the reader thread (idempotent).

        Callbacks fire on the reader thread — except with a missing stream
        (``sys.stdin`` is ``None``), where ``on_eof`` fires synchronously here.
        """
        if self._thread is not None:
            return
        if self._stream is None:
            logger.warning("No stdin available for command input — treating as EOF")
            self._emit_eof()
            return
        self._enable_hotkeys()
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name="StdinCommandListener")
        try:
            self._thread.start()
        except BaseException:
            self._restore_terminal()
            raise
        if not self._use_select:
            logger.info("stdin listener running in blocking mode (select unavailable for this stream)")

    def stop(self) -> None:
        """Stop the reader thread.

        Blocking-mode threads may be stuck inside ``readline`` and cannot be joined;
        they are daemons and die with the process.  Late lines are ignored either way.
        """
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not current_thread() and thread.is_alive() and self._use_select:
            thread.join(timeout=1.0)
        self._restore_terminal()

    def _run(self) -> None:
        try:
            if self.space_hotkey_enabled:
                self._run_terminal()
            elif self._use_select:
                self._run_select()
            else:
                self._run_blocking()
        except (OSError, ValueError):
            self._emit_read_error()
        finally:
            self._restore_terminal()

    def _run_terminal(self) -> None:
        """Read hotkeys and editable UTF-8 command lines from one cbreak TTY."""
        fd = self._stream.fileno()
        on_space = self._on_space
        assert on_space is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buffer = ""
        escape = ""
        while self._running:
            try:
                ready, _, _ = select.select([fd], [], [], self._poll_interval_s)
                if not ready:
                    continue
                data = os.read(fd, 4096)
            except (OSError, ValueError):
                self._emit_read_error()
                return
            if not self._running:
                return
            if not data:
                self._emit_line(buffer)
                self._emit_eof()
                return
            for char in decoder.decode(data):
                if not self._running:
                    return
                if escape:
                    # Consume CSI/SS3 sequences (arrows etc.) without adding garbage
                    # to the command. The editor supports append/backspace, not cursor motion.
                    if escape == "\x1b" and char in "[O":
                        escape += char
                    elif escape == "\x1b" or "@" <= char <= "~":
                        escape = ""
                    continue
                if char == "\x1b":
                    escape = char
                elif char in "\r\n":
                    self._echo("\n")
                    self._emit_line(buffer)
                    buffer = ""
                elif char in "\x7f\x08":
                    if buffer:
                        buffer = buffer[:-1]
                        self._echo("\b \b")
                elif char == "\x15":  # Ctrl-U: cancel an unfinished command.
                    self._echo("\b \b" * len(buffer))
                    buffer = ""
                elif char == "\x04":  # Ctrl-D: canonical-style EOF on an empty line.
                    if not buffer:
                        self._emit_eof()
                        return
                    self._emit_line(buffer)
                    buffer = ""
                elif char == " " and not buffer:
                    now = monotonic()
                    eligible = now - self._last_space_at >= self._SPACE_REPEAT_GAP_S
                    self._last_space_at = now
                    if eligible:
                        try:
                            on_space()
                        except Exception:
                            logger.exception("Error while handling space hotkey")
                elif char.isprintable():
                    buffer += char
                    self._echo(char)

    @staticmethod
    def _echo(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def _run_select(self) -> None:
        """Poll the file descriptor and split lines from raw bytes.

        Raw bytes rather than ``stream.readline()``: a buffered file object can slurp
        several lines at once, after which ``select`` reports the drained fd as
        not-ready and those lines would never be delivered.
        """
        try:
            fd = self._stream.fileno()
        except (OSError, ValueError):  # closed between construction and thread start
            self._emit_read_error()
            return
        buffer = b""
        while self._running:
            try:
                ready, _, _ = select.select([fd], [], [], self._poll_interval_s)
            except (OSError, ValueError):  # stream closed underneath us
                self._emit_read_error()
                return
            if not ready:
                continue
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                self._emit_read_error()
                return
            if not self._running:
                return
            if chunk == b"":  # EOF: Ctrl-D or the piped input ended
                # A final command without trailing newline still counts.
                self._emit_line(buffer.decode(errors="replace"))
                self._emit_eof()
                return
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                self._emit_line(raw.decode(errors="replace"))

    def _run_blocking(self) -> None:
        while self._running:
            try:
                line = self._stream.readline()
            except (OSError, ValueError, AttributeError):
                self._emit_read_error()
                return
            if not self._running:
                return
            if not line:  # EOF: "" on text streams, b"" on bytes streams
                self._emit_eof()
                return
            self._emit_line(line if isinstance(line, str) else line.decode(errors="replace"))

    def _emit_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            self._on_line(line)
        except Exception:  # never let a handler error kill the reader thread
            logger.exception("Error while handling input line %r", line)

    def _emit_eof(self) -> None:
        logger.info("Input stream closed (EOF)")
        if self._on_eof is not None:
            try:
                self._on_eof()
            except Exception:
                logger.exception("Error while handling input EOF")

    def _emit_read_error(self) -> None:
        """Treat an unexpected read failure like EOF so consumers shut down.

        A deliberate ``stop()`` clears ``_running`` first and does not reach this.
        """
        if self._running:
            logger.warning("Input stream failed — treating as EOF")
            self._emit_eof()
