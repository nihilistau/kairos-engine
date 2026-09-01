"""engine/launch.py — starting, stopping and identifying THE DAEMON. The sp-specific half
of serve.py, split out 2026-08-21 so the launcher runs without this directory at all
(the engine-agnostic Kairos export excludes engine/ wholesale). serve.py imports this
lazily, only when the profile's [engine].kind is "sp" (the default).

ONE COPY, TWO CALLERS still holds: the full launch and --daemon-only both come through
`launch_daemon`; `stop_daemon` kills by the profile's own image basenames; the
model-agreement guard (`daemon_model_of` / `running_daemon_model`) is the thing that
caught the 2026-08-01 wrong-profile outage and it lives here, next to the spawn.
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAR = os.path.join(ROOT, "var")


def wait_http(url: str, secs: int) -> bool:
    for _ in range(secs):
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def launch_daemon(c: dict, env: dict) -> bool:
    """Start the engine and wait for it to answer. True if it came up.

    APPEND, and stamp each boot — this was "w", truncating the previous boot's stdout
    on every launch. On a failed boot the traceback lives in var/daemon.boot.log, not
    in var/daemon.log (the engine's own runtime log). utf-8 EXPLICITLY: the default
    Windows encoding is cp1252 and the stamp crashed the very first boot after it was
    written (Popen writes the daemon's raw bytes through the fileno; the encoding
    governs only this header line)."""
    daemon_log = open(os.path.join(VAR, "daemon.boot.log"), "a", encoding="utf-8")
    daemon_log.write("\n-- boot %s --\n" % time.strftime("%Y-%m-%dT%H:%M:%S"))
    daemon_log.flush()
    subprocess.Popen(
        [c["paths"]["engine_exe"].replace("/", "\\"), "start",
         "--model", c["paths"]["model"], "--tokenizer", c["paths"]["tokenizer"],
         "--port", str(c["serve"]["port"])],
        env=env, stdout=daemon_log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return wait_http(f"http://127.0.0.1:{c['serve']['port']}/v1/metrics", 90)


def daemon_images(c=None) -> list:
    """The daemon's and the voice server's image basenames — by the PROFILE when we have
    one: killing hardcoded names worked by coincidence of all 14 profiles agreeing, and a
    renamed build made stop() a silent no-op."""
    images = ["sp-daemon.exe", "tts-server.exe"]
    if c:
        images = [os.path.basename(c["paths"]["engine_exe"]),
                  os.path.basename(c.get("tts", {}).get("server_exe") or "tts-server.exe")]
    return images


def stop_daemon(c=None) -> None:
    """Kill the daemon and the voice server only — never the gateway (the room's start
    button runs through here and must not kill the thing that pressed it)."""
    for img in daemon_images(c):
        subprocess.run(["taskkill", "/F", "/IM", img], capture_output=True)


def daemon_model_of(cfg) -> str:
    """The model path this profile would serve — the SAME expression the launch uses
    (`c["paths"]["model"]`). An empty result is "could not determine", never "fine"."""
    try:
        return str(cfg["paths"]["model"]).strip()
    except Exception:
        return ""


def running_daemon_model() -> str:
    """The model the LIVE daemon is actually serving, read from its own command line —
    ground truth, not a launch record that could be stale. "" when undeterminable, and
    the caller says so out loud."""
    try:
        import psutil
        for pr in psutil.process_iter(["name", "cmdline"]):
            nm = (pr.info.get("name") or "").lower()
            if "sp-daemon" not in nm and "sp_daemon" not in nm:
                continue
            cmd = pr.info.get("cmdline") or []
            for i, tok in enumerate(cmd):
                if tok == "--model" and i + 1 < len(cmd):
                    return cmd[i + 1].strip()
                if tok.startswith("--model="):
                    return tok.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""
