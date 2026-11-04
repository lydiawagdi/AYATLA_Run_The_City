"""AYATLA state authority and TCP relay. Run: python hub.py."""
import copy
import logging
import math
import socket
import threading
from common import HOST, PORT, CONTEXTS, DEFAULT_SETTINGS, send_msg, recv_msgs

log = logging.getLogger("ayatla.hub")
ROLES = ("vision", "bluetooth", "game")
# bt_register (game -> bluetooth) and bt_devices (bluetooth -> game) are used
# by the hands-free device registration screen (context BT_REGISTER).
FORWARDED = ("gesture", "tracking", "calibration", "mode", "bt_status", "user", "settings",
             "bt_register", "bt_devices")


def valid_setting(key, value):
    """Known settings only; booleans must not pass as numbers."""
    if key == "left_handed":
        return type(value) is bool
    if key == "camera_index":
        return type(value) is int and value >= 0
    if type(value) not in (int, float):
        return False
    try:
        if not math.isfinite(value):
            return False
    except OverflowError:
        return False
    if key in ("gesture_sensitivity", "volume"):
        return 0 <= value <= 1
    return key == "hold_time" and value > 0


class Hub:
    def __init__(self, host=HOST, port=PORT):
        self.host, self.port = host, port
        self.state = {
            "user": None, "mode": None, "context": "CALIBRATION",
            "previous_context": None, "settings": DEFAULT_SETTINGS.copy(),
        }
        # Serialize both changes and notifications to preserve state ordering.
        self.lock = threading.RLock()
        self.clients = {}  # socket -> role (None until hello)
        self.workers = set()
        self.listener = None
        self.stopping = threading.Event()
        self.ready = threading.Event()

    def _remove(self, sock):
        with self.lock:
            self.clients.pop(sock, None)
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _send(self, sock, message):
        try:
            send_msg(sock, message)
        except OSError:
            log.info("[%s] disconnected during send", self.clients.get(sock))
            self._remove(sock)

    def _broadcast(self, message, exclude=None):
        # Called with the lock held. A failed send may remove a client.
        for sock, role in list(self.clients.items()):
            if role is not None and sock is not exclude:
                self._send(sock, message)

    def _state_message(self):
        return {"type": "state", **copy.deepcopy(self.state)}

    def _handle(self, sock, message):
        with self.lock:
            if sock not in self.clients:
                return
            role = self.clients[sock]
            log.info("[%s] %s", role or "unregistered", message)
            kind = message.get("type")
            if role is None:
                new_role = message.get("role")
                if kind != "hello" or new_role not in ROLES:
                    log.warning("[unregistered] First valid message must be hello")
                    self._remove(sock)
                    return
                self.clients[sock] = new_role
                threading.current_thread().name = f"client:{new_role}"
                self._send(sock, self._state_message())
                return

            before = copy.deepcopy(self.state)
            if kind == "context":
                value = message.get("value")
                if not isinstance(value, str) or value not in CONTEXTS:
                    log.warning("[%s] Ignoring invalid context", role)
                    return
                self.state["context"] = value
            elif kind == "mode":
                value = message.get("value")
                if value not in ("bluetooth", "guest"):
                    return
                self.state["mode"] = value
                if value == "guest":
                    self.state["user"] = None
            elif kind == "user":
                if "id" not in message:
                    return
                value = message["id"]
                if value is not None and not isinstance(value, str):
                    return
                self.state["user"] = value
            elif kind == "settings":
                updates = {k: v for k, v in message.items() if k != "type"}
                if not updates or not all(valid_setting(k, v) for k, v in updates.items()):
                    log.warning("[%s] Ignoring invalid settings", role)
                    return
                self.state["settings"].update(updates)
            elif kind == "tracking":
                status = message.get("status")
                context = self.state["context"]
                if status == "lost":
                    if context not in ("CALIBRATION", "TRACKING_LOST"):
                        self.state["previous_context"] = context
                        self.state["context"] = "TRACKING_LOST"
                elif status == "found":
                    if context == "TRACKING_LOST":
                        previous = self.state["previous_context"]
                        # A direct request can enter TRACKING_LOST without a
                        # saved context. Keep it until game requests a screen.
                        if previous is not None:
                            self.state["context"] = previous
                else:
                    return
            elif kind == "calibration":
                if self.state["context"] != "CALIBRATION":
                    return
            elif kind not in FORWARDED:
                return

            # Deliberately allow simulated messages from the game test client.
            if kind in FORWARDED:
                self._broadcast(message, exclude=sock)
            if before != self.state:
                self._broadcast(self._state_message())

    def _client_loop(self, sock):
        try:
            for message in recv_msgs(sock):
                if self.stopping.is_set():
                    break
                self._handle(sock, message)
        finally:
            with self.lock:
                role = self.clients.get(sock)
                self._remove(sock)
                self.workers.discard(threading.current_thread())
            log.info("[%s] connection closed", role or "unregistered")

    def serve_forever(self):
        """Retry listener failures every two seconds; one thread per client."""
        while not self.stopping.is_set():
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Windows SO_REUSEADDR can allow duplicate live listeners.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(0.5)
            self.listener = listener
            try:
                listener.bind((self.host, self.port))
                listener.listen()
                self.port = listener.getsockname()[1]  # port=0 for isolated tests
                self.ready.set()
                log.info("[hub] Listening on %s:%s", self.host, self.port)
                while not self.stopping.is_set():
                    try:
                        sock, address = listener.accept()
                    except socket.timeout:
                        continue
                    # Bound blocked sends; recv_msgs tolerates idle timeouts.
                    sock.settimeout(2)
                    worker = threading.Thread(target=self._client_loop, args=(sock,),
                                              name=f"client:unregistered:{address}", daemon=True)
                    with self.lock:
                        if self.stopping.is_set():
                            sock.close()
                            break
                        self.clients[sock] = None
                        self.workers.add(worker)
                        worker.start()
            except OSError as exc:
                if not self.stopping.is_set():
                    log.warning("[hub] %s; retrying in 2 seconds", exc)
            finally:
                self.ready.clear()
                listener.close()
            self.stopping.wait(2)

    def stop(self):
        """Close sockets to unblock receivers, then join client threads."""
        self.stopping.set()
        if self.listener is not None:
            self.listener.close()
        with self.lock:
            for sock in list(self.clients):
                self._remove(sock)
            workers = list(self.workers)
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(timeout=3)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(threadName)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    hub = Hub()
    try:
        hub.serve_forever()
    except KeyboardInterrupt:
        log.info("[hub] Shutting down")
    finally:
        hub.stop()


if __name__ == "__main__":
    main()
