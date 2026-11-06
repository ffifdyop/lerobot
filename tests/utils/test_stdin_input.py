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

"""Tests for the non-blocking stdin line reader."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from threading import Event

import pytest

from lerobot.utils.stdin_input import StdinCommandListener


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@contextlib.contextmanager
def _pipe_stream():
    """A held-open pipe so the listener never sees EOF until we close it."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r")
    writer = os.fdopen(write_fd, "w")
    try:
        yield reader, writer
    finally:
        with contextlib.suppress(OSError, ValueError):
            writer.close()
        with contextlib.suppress(OSError, ValueError):
            reader.close()


def test_stdin_listener_delivers_batched_lines_and_eof():
    """Several lines arriving in one chunk (paste, piped script) are all delivered — the reason
    the reader splits raw ``os.read`` bytes instead of calling ``readline()``."""
    lines: list[str] = []
    eof = Event()

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(lines.append, on_eof=eof.set, stream=reader)
        listener.start()
        writer.write("/start\n   \n/subtask grab the cube\n")  # one chunk; the blank line is skipped
        writer.flush()
        assert _wait_for(lambda: len(lines) == 2)
        assert lines == ["/start", "/subtask grab the cube"]

        writer.close()
        assert _wait_for(eof.is_set)
        listener.stop()


def test_stdin_listener_accumulates_partial_lines_and_flushes_the_last_one_at_eof():
    """A line split across reads is delivered once, whole, and a final command with no trailing
    newline (``printf '/stop'``) is delivered before on_eof."""
    lines: list[str] = []
    eof = Event()

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(lines.append, on_eof=eof.set, stream=reader)
        listener.start()
        writer.write("/subtask grab ")
        writer.flush()
        time.sleep(0.05)  # let the listener consume the partial chunk
        assert lines == []

        writer.write("the cube\n/stop")
        writer.flush()
        assert _wait_for(lambda: lines == ["/subtask grab the cube"])

        writer.close()
        assert _wait_for(eof.is_set)
        assert lines == ["/subtask grab the cube", "/stop"]
        listener.stop()


def test_stdin_listener_handler_errors_do_not_kill_reader():
    lines: list[str] = []

    def flaky(line: str) -> None:
        if line == "/boom":
            raise RuntimeError("boom")
        lines.append(line)

    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(flaky, stream=reader)
        listener.start()
        writer.write("/boom\n/start\n")
        writer.flush()
        assert _wait_for(lambda: lines == ["/start"])
        listener.stop()


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        (io.StringIO("/start\n\n/help\n"), ["/start", "/help"]),
        (io.BytesIO(b"/start\n"), ["/start"]),
    ],
    ids=["text", "bytes"],
)
def test_stdin_listener_blocking_fallback(stream, expected):
    """Streams without a file descriptor use the blocking readline path, where EOF is ``""`` on
    text streams and ``b""`` on bytes streams — missing the latter busy-looped forever."""
    lines: list[str] = []
    eof = Event()
    listener = StdinCommandListener(lines.append, on_eof=eof.set, stream=stream)
    assert not listener._use_select
    listener.start()
    assert _wait_for(eof.is_set)
    assert lines == expected
    assert _wait_for(lambda: not listener._thread.is_alive())
    listener.stop()


def test_stdin_listener_unusable_stream_treated_as_eof(monkeypatch):
    """A stream that blows up mid-run, or a missing sys.stdin (daemonized process), must fire
    on_eof instead of leaving the consumer waiting for commands that cannot arrive."""

    class _BrokenStream:
        def readline(self):
            raise AttributeError("broken")

    broken_eof = Event()
    broken = StdinCommandListener(lambda line: None, on_eof=broken_eof.set, stream=_BrokenStream())
    broken.start()
    assert _wait_for(broken_eof.is_set)
    broken.stop()

    monkeypatch.setattr(sys, "stdin", None)
    missing_eof = Event()
    missing = StdinCommandListener(lambda line: None, on_eof=missing_eof.set)
    missing.start()
    assert missing_eof.is_set()
    missing.stop()


@contextlib.contextmanager
def _terminal_stream():
    """A real POSIX pseudo-terminal, with no display server or keyboard permissions."""
    if os.name != "posix":
        pytest.skip("POSIX terminal required")
    master, slave = os.openpty()
    reader = os.fdopen(slave, "r", encoding="utf-8")
    try:
        yield master, reader
    finally:
        reader.close()
        os.close(master)


def test_terminal_space_is_immediate_and_repeats_are_suppressed(monkeypatch):
    termios = pytest.importorskip("termios")

    import lerobot.utils.stdin_input as stdin_input

    now = [10.0]
    monkeypatch.setattr(stdin_input, "monotonic", lambda: now[0])
    taps, lines = [], []
    with _terminal_stream() as (master, reader):
        original = termios.tcgetattr(reader.fileno())
        listener = StdinCommandListener(lines.append, stream=reader, on_space=lambda: taps.append(now[0]))
        try:
            listener.start()
            active = termios.tcgetattr(reader.fileno())
            assert not active[3] & (termios.ICANON | termios.ECHO)
            assert active[3] & termios.ISIG == original[3] & termios.ISIG
            assert listener.space_hotkey_enabled
            os.write(master, b" ")  # No newline: must dispatch immediately.
            assert _wait_for(lambda: len(taps) == 1)
            now[0] += 0.5  # Typical first key-repeat delay.
            os.write(master, b"   /help\n")
            assert _wait_for(lambda: lines == ["/help"])
            assert len(taps) == 1
            now[0] += 0.8
            os.write(master, b" ")
            assert _wait_for(lambda: len(taps) == 2)
        finally:
            listener.stop()
        assert termios.tcgetattr(reader.fileno()) == original


def test_terminal_command_spaces_utf8_editing_and_ctrl_d():
    termios = pytest.importorskip("termios")

    lines, taps = [], []
    eof = Event()
    with _terminal_stream() as (master, reader):
        original = termios.tcgetattr(reader.fileno())
        listener = StdinCommandListener(
            lines.append, stream=reader, on_space=lambda: taps.append(True), on_eof=eof.set
        )
        try:
            listener.start()
            os.write(master, "/autosteer 抓".encode()[:-1])  # Split inside a UTF-8 character.
            os.write(master, "/autosteer 抓".encode()[-1:] + " 取x".encode() + b"\x7f\n")
            os.write(master, b"/discard this\x15/subtask keep the cube\x1b[D\n")
            assert _wait_for(lambda: len(lines) == 2)
            assert lines == ["/autosteer 抓 取", "/subtask keep the cube"]
            assert not taps
            os.write(master, b"\x04")
            assert eof.wait(2)
            assert _wait_for(lambda: not listener._thread.is_alive())
            assert termios.tcgetattr(reader.fileno()) == original
        finally:
            listener.stop()


def test_space_callback_is_ignored_on_pipes():
    taps, lines = [], []
    with _pipe_stream() as (reader, writer):
        listener = StdinCommandListener(lines.append, stream=reader, on_space=lambda: taps.append(True))
        try:
            listener.start()
            assert not listener.space_hotkey_enabled
            writer.write(" \n/teleop\n/resume\n")
            writer.flush()
            assert _wait_for(lambda: len(lines) == 2)
            assert lines == ["/teleop", "/resume"]
            assert not taps
        finally:
            listener.stop()


def test_terminal_read_error_restores_mode(monkeypatch):
    termios = pytest.importorskip("termios")

    import lerobot.utils.stdin_input as stdin_input

    eof = Event()
    with _terminal_stream() as (_master, reader):
        original = termios.tcgetattr(reader.fileno())

        def broken_select(*args):
            raise OSError("lost input")

        monkeypatch.setattr(stdin_input.select, "select", broken_select)
        listener = StdinCommandListener(
            lambda line: None, stream=reader, on_space=lambda: None, on_eof=eof.set
        )
        try:
            listener.start()
            assert eof.wait(2)
            assert _wait_for(lambda: not listener._thread.is_alive())
            assert termios.tcgetattr(reader.fileno()) == original
        finally:
            listener.stop()
