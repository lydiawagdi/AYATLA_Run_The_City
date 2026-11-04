"""Shared AYATLA newline-delimited JSON protocol. Python 3.10+."""
import json
import logging
import socket
import threading
import weakref
from collections.abc import Iterator

HOST = "127.0.0.1"
PORT = 5000
CONTEXTS = frozenset({
    "CALIBRATION", "MAIN_MENU", "GESTURE_GUIDE", "SETTINGS", "PLAYER_MODE",
    "PLAYING", "PAUSED", "GAME_OVER", "QUIT_CONFIRM", "TRACKING_LOST",
    "BT_REGISTER",   # hands-free registration of a new Bluetooth device
})
GESTURES = frozenset({
    "jump", "squat", "lean_left", "lean_right", "open_palm", "thumbs_up", "ok_sign",
    "fingers_1", "fingers_2", "fingers_3", "fingers_4", "hold_x", "object_power_up",
    "zone_left", "zone_center", "zone_right",
})
DEFAULT_SETTINGS = {
    "gesture_sensitivity": 0.72, "hold_time": 2.0, "volume": 0.8,
    "camera_index": 0, "left_handed": False,
}
# One lock per socket prevents concurrent senders interleaving messages.
_send_locks = weakref.WeakKeyDictionary()
_locks_guard = threading.Lock()
_MAX_LINE = 1024 * 1024
log = logging.getLogger(__name__)


def send_msg(sock: socket.socket, obj: dict) -> None:
    """Send one object; caller handles OSError and reconnection."""
    if not isinstance(obj, dict):
        raise TypeError("Messages must be JSON objects")
    data = (json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with _locks_guard:
        lock = _send_locks.setdefault(sock, threading.Lock())
    with lock:
        sock.sendall(data)


def _reject_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


def recv_msgs(sock: socket.socket) -> Iterator[dict]:
    """One reader per socket. Skip bad lines; stop on EOF/disconnect.

    Buffer bytes so split UTF-8 characters and partial TCP chunks work.
    Timeouts mean idle, not disconnected. Oversized lines are discarded.
    """
    buffer = bytearray()
    discarding = False
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError:
            return
        if not chunk:
            return
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                if len(buffer) > _MAX_LINE:
                    if not discarding:
                        log.warning("Ignoring oversized JSON line")
                    discarding = True
                    buffer.clear()
                break
            line = bytes(buffer[:newline])
            del buffer[:newline + 1]
            if discarding or len(line) > _MAX_LINE:
                discarding = False
                log.warning("Ignored oversized JSON line")
                continue
            try:
                obj = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
                if not isinstance(obj, dict):
                    raise ValueError("Expected a JSON object")
            except (UnicodeDecodeError, ValueError, RecursionError) as exc:
                log.warning("Ignoring malformed JSON: %s", exc)
                continue
            yield obj
