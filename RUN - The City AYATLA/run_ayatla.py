"""
run_ayatla.py - start the whole AYATLA system with ONE command.

    python run_ayatla.py                  hub + vision + bluetooth + game
    python run_ayatla.py --camera 1       pass a camera index to vision
    python run_ayatla.py --no-bluetooth   skip the Bluetooth process
    python run_ayatla.py --enable-object  turn on the object power-up in vision
    python run_ayatla.py --keys           game also accepts keyboard fakes (testing)
    python run_ayatla.py --verbose        also show MediaPipe / pygame start-up noise

(or double-click AYATLA.bat)

What it does
------------
1. Starts hub.py and waits until it is listening (the hub is the source of
   truth, so everyone else should find it on their first try).
2. Starts vision.py, bluetooth_id.py and game.py with the same Python.
3. Shows the output of all four processes in this one console, each line
   tagged [hub] / [vision] / [bt] / [game], and writes full logs to logs/.
4. Keeps them running together:
     * the player quits the game (Quit screen or closing the window)
           -> everything is shut down
     * any process CRASHES -> it is restarted after 2 s (the hub keeps the
       state, and the clients reconnect to it by themselves). A process that
       keeps crashing is given up after MAX_RESTARTS tries.
     * Ctrl+C here -> everything is shut down
Only the standard library is used.
"""

import argparse
import os
import socket
import subprocess
import sys
import threading
import time

from common import HOST, PORT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

# The project's own virtual environment (.venv) has the pinned MediaPipe that
# vision.py needs; fall back to the Python running this launcher.
VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "Scripts" if os.name == "nt" else "bin",
                           "python.exe" if os.name == "nt" else "python")
PYTHON = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

HUB_READY_TIMEOUT = 10.0   # seconds to wait for "Listening on ..." from the hub
RESTART_DELAY = 2.0        # seconds before restarting a crashed process
MAX_RESTARTS = 3           # crashes allowed per process within RESTART_WINDOW
RESTART_WINDOW = 60.0
STOP_TIMEOUT = 3.0         # seconds a process gets to exit before it is killed

# Harmless start-up chatter from MediaPipe / TensorFlow Lite / pygame, and the
# hub's log of every calibration message (10 per second). It still goes into
# logs/, it is just hidden from the console.
NOISE = ("W0000", "I0000", "WARNING: All log messages before absl",
         "INFO: Created TensorFlow Lite", "WARNING: Logging before InitGoogle",
         "pkg_resources", "from pkg_resources import", "pygame 2.",
         "Hello from the pygame community", "'type': 'calibration'")

# ANSI colours for the tags (violet / pink accents like the UI).
COLORS = {"hub": "\033[96m", "vision": "\033[95m", "bt": "\033[94m", "game": "\033[38;5;213m",
          "ayatla": "\033[97m"}
RESET = "\033[0m"
print_lock = threading.Lock()


def say(tag, text):
    with print_lock:
        print(f"{COLORS.get(tag, '')}[{tag:>6}]{RESET} {text}", flush=True)


def port_in_use():
    """True if something already accepts connections on the hub port."""
    try:
        with socket.create_connection((HOST, PORT), timeout=0.5):
            return True
    except OSError:
        return False


class Managed:
    """One child process: start, stream its output, restart, stop."""

    def __init__(self, tag, script, args=(), ready_text=None, verbose=False):
        self.tag = tag
        self.cmd = [PYTHON, "-u", os.path.join(BASE_DIR, script), *args]
        self.ready_text = ready_text          # line that means "I'm ready"
        self.ready = threading.Event()
        self.verbose = verbose
        self.proc = None
        self.crashes = []                     # times of recent crashes
        self.gave_up = False
        self.log = open(os.path.join(LOG_DIR, f"{tag}.log"), "a", encoding="utf-8")

    def start(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.ready.clear()
        self.log.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        self.proc = subprocess.Popen(
            self.cmd, cwd=BASE_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,      # one merged stream
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        say("ayatla", f"started {self.tag} (pid {self.proc.pid})")

    def _pump(self, proc):
        """Copy the child's output to the log file and (filtered) to the console."""
        for line in proc.stdout:
            line = line.rstrip()
            self.log.write(line + "\n")
            self.log.flush()
            if self.ready_text and self.ready_text in line:
                self.ready.set()
            if line and (self.verbose or not any(n in line for n in NOISE)):
                say(self.tag, line)

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self):
        return None if self.proc is None else self.proc.poll()

    def note_crash(self):
        """Remember a crash; return False if it crashed too often to retry."""
        now = time.monotonic()
        self.crashes = [t for t in self.crashes if now - t < RESTART_WINDOW] + [now]
        return len(self.crashes) <= MAX_RESTARTS

    def stop(self):
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()


def main():
    ap = argparse.ArgumentParser(description="Start the whole AYATLA system")
    ap.add_argument("--camera", type=int, default=None, help="camera index for vision")
    ap.add_argument("--no-bluetooth", action="store_true", help="do not start bluetooth_id.py")
    ap.add_argument("--enable-object", action="store_true", help="vision: object power-up")
    ap.add_argument("--keys", action="store_true", help="game: keyboard fakes as well")
    ap.add_argument("--verbose", action="store_true", help="show start-up noise too")
    args = ap.parse_args()

    os.system("")                  # turns on ANSI colours in old Windows consoles
    os.makedirs(LOG_DIR, exist_ok=True)

    vision_args = []
    if args.camera is not None:
        vision_args += ["--camera", str(args.camera)]
    if args.enable_object:
        vision_args.append("--enable-object")

    hub = Managed("hub", "hub.py", ready_text="Listening on", verbose=args.verbose)
    clients = [Managed("vision", "vision.py", vision_args, verbose=args.verbose)]
    if not args.no_bluetooth:
        clients.append(Managed("bt", "bluetooth_id.py", verbose=args.verbose))
    game = Managed("game", "game.py", ["--keys"] if args.keys else [], verbose=args.verbose)
    clients.append(game)
    everyone = [hub] + clients

    say("ayatla", "starting AYATLA...  (Ctrl+C here stops everything)")
    if PYTHON != VENV_PYTHON:
        say("ayatla", "WARNING: no .venv found - using this Python. If vision fails, create it: "
                      "python -m venv .venv  then  .venv\\Scripts\\python -m pip install -r requirements.txt")
    if port_in_use():
        say("ayatla", f"WARNING: something already listens on {HOST}:{PORT} "
                      "(an old hub.py?). Close it, otherwise two hubs fight over the port.")
    hub.start()
    if not hub.ready.wait(HUB_READY_TIMEOUT):
        say("ayatla", "hub did not report ready in time; starting the others anyway "
                      "(they retry every 2 s)")
    for c in clients:
        c.start()

    reason = "stopped"
    try:
        while True:
            time.sleep(0.5)
            for m in everyone:
                if m.running() or m.gave_up or m.proc is None:
                    continue
                code = m.exit_code()
                if m is game and code == 0:
                    reason = "the game was closed"
                    raise StopIteration
                if m.note_crash():
                    say("ayatla", f"{m.tag} exited (code {code}); restarting in {RESTART_DELAY:.0f} s...")
                    time.sleep(RESTART_DELAY)
                    m.start()
                else:
                    m.gave_up = True
                    say("ayatla", f"{m.tag} keeps crashing - giving up. See logs/{m.tag}.log")
                    if m is game or m is hub:
                        reason = f"{m.tag} cannot run"
                        raise StopIteration
    except StopIteration:
        pass
    except KeyboardInterrupt:
        reason = "Ctrl+C"

    say("ayatla", f"shutting down ({reason})...")
    for m in reversed(everyone):      # clients first, hub last
        m.stop()
    say("ayatla", "all processes stopped. Logs are in logs/.")


if __name__ == "__main__":
    main()
