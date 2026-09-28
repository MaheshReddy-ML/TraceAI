"""Exercise the animated home screen through an actual pseudo-terminal."""

import os
import re
import select
import struct
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name != "posix", reason="pseudo-terminal controls require POSIX")
@pytest.mark.parametrize("width", [52, 100])
def test_animated_menu_moves_and_restores_terminal(tmp_path, width):
    import fcntl
    import pty
    import termios

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["HF_HUB_CACHE"] = str(tmp_path / "empty-hub")
    env["OLLAMA_HOST"] = "http://127.0.0.1:9"
    env.pop("NO_COLOR", None)
    env.pop("TRACEAI_NO_ANIMATION", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "traceai.cli"],
        cwd=tmp_path,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
    )
    os.close(slave)
    output = bytearray()
    menu_seen_at = None
    sent = False
    deadline = time.monotonic() + 8
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.05)
            if ready:
                try:
                    output.extend(os.read(master, 65536))
                except OSError:
                    break
            if b"EXPLORE" in output and menu_seen_at is None:
                menu_seen_at = time.monotonic()
            if menu_seen_at is not None and not sent and time.monotonic() - menu_seen_at > 0.4:
                os.write(master, b"\x1b[B\r")  # Move from study setup to mock walkthrough.
                sent = True
            if process.poll() is not None:
                break
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        os.close(master)

    text = output.decode(errors="replace")
    plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    frames = set(re.findall(r"[▁▂▃▄▅▆▇█]{8,}", plain))
    assert process.returncode == 0, plain[-1200:]
    assert "\x1b[?1049h" in text and "\x1b[?1049l" in text
    assert len(frames) > 1
    assert (tmp_path / "experiment.yaml").is_file()
