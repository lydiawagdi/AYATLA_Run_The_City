"""Developer simulator only: the actual AYATLA game stays hands-free."""
import json
import socket
import threading
from common import HOST, PORT, CONTEXTS, GESTURES, send_msg, recv_msgs
from hub import valid_setting

HELP = """Commands:
  gesture <name>             e.g. gesture jump
  tracking lost|found
  mode bluetooth|guest
  user user_01|none
  context <NAME>             e.g. context PLAYING
  settings <key> <value>     e.g. settings hold_time 2.5
  calibration done
  help
  quit
Use lowercase JSON booleans: settings left_handed true
hold_x sends once with repeat=false (no automatic held-pose simulation).
"""


def parse_command(line):
    """Turn a short command into a protocol message."""
    parts = line.split()
    if len(parts) == 2:
        command, value = parts
        if command == "gesture" and value in GESTURES:
            body = value in ("jump", "squat", "lean_left", "lean_right")
            return {"type": "gesture", "name": value,
                    "score": 0.91 if body else 1.0, "repeat": False}
        if command == "tracking" and value in ("lost", "found"):
            return {"type": "tracking", "status": value}
        if command == "mode" and value in ("bluetooth", "guest"):
            return {"type": "mode", "value": value}
        if command == "user":
            return {"type": "user", "id": None if value == "none" else value}
        if command == "context" and value in CONTEXTS:
            return {"type": "context", "value": value}
        if command == "calibration" and value == "done":
            return {"type": "calibration", "camera": True, "body": True,
                    "hands": True, "progress": 1.0, "seconds_left": 0.0, "done": True}
    if len(parts) == 3 and parts[0] == "settings":
        key = parts[1]
        try:
            value = json.loads(parts[2])
        except ValueError:
            raise ValueError("Settings need a JSON number or true/false") from None
        if valid_setting(key, value):
            return {"type": "settings", key: value}
    raise ValueError("Unknown command or invalid value. Type help for examples.")


class TestClient:
    def __init__(self):
        self.sock = None
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.receiver = threading.Thread(target=self._receive_loop, daemon=True)

    def _receive_loop(self):
        # This thread owns reconnection and is the only socket reader.
        while not self.stopping.is_set():
            sock = None
            try:
                sock = socket.create_connection((HOST, PORT), timeout=2)
                send_msg(sock, {"type": "hello", "role": "game"})
                with self.lock:
                    if self.stopping.is_set():
                        break
                    self.sock = sock
                print("\nConnected as game.", flush=True)
                for message in recv_msgs(sock):
                    print("\nRECV " + json.dumps(message, ensure_ascii=False), flush=True)
            except OSError as exc:
                if not self.stopping.is_set():
                    print(f"\nHub unavailable: {exc}", flush=True)
            finally:
                with self.lock:
                    if self.sock is sock:
                        self.sock = None
                if sock is not None:
                    sock.close()
            if not self.stopping.is_set():
                print("\nDisconnected. Retrying in 2 seconds.", flush=True)
            self.stopping.wait(2)

    def send(self, message):
        with self.lock:
            sock = self.sock
        if sock is None:
            print("Not connected; command not sent. Try again after connection.")
            return
        try:
            send_msg(sock, message)
        except OSError:
            print("Send failed; delivery unknown. Reconnecting.")
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def stop(self):
        self.stopping.set()
        with self.lock:
            sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        self.receiver.join(timeout=3)


def main():
    client = TestClient()
    client.receiver.start()
    print(HELP)
    try:
        while True:
            line = input("> ").strip()
            if line == "quit":
                break
            if line == "help":
                print(HELP)
            elif line:
                try:
                    client.send(parse_command(line))
                except ValueError as exc:
                    print(exc)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        client.stop()


if __name__ == "__main__":
    main()
