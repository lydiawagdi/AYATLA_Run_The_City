"""Isolated integration tests: python test_hub.py (no real devices needed)."""
import queue
import socket
import threading
import unittest
from common import HOST, DEFAULT_SETTINGS, send_msg, recv_msgs
from hub import Hub
from test_client import parse_command


class FakeClient:
    """Every fake process listens continuously in its own receiver thread."""
    def __init__(self, port, role):
        self.sock = socket.create_connection((HOST, port), timeout=2)
        self.messages = queue.Queue()
        send_msg(self.sock, {"type": "hello", "role": role})
        self.receiver = threading.Thread(target=self._receive, daemon=True)
        self.receiver.start()

    def _receive(self):
        for message in recv_msgs(self.sock):
            self.messages.put(message)

    def send(self, message):
        send_msg(self.sock, message)

    def next(self):
        return self.messages.get(timeout=3)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.receiver.join(timeout=3)


class HubTests(unittest.TestCase):
    def setUp(self):
        # Random free port keeps tests independent of a running real hub.
        self.hub = Hub(port=0)
        self.server = threading.Thread(target=self.hub.serve_forever, daemon=True)
        self.server.start()
        self.addCleanup(self.shutdown)
        self.clients = []
        self.assertTrue(self.hub.ready.wait(3), "Hub did not start")
        for role in ("vision", "bluetooth", "game"):
            client = FakeClient(self.hub.port, role)
            self.clients.append(client)
            self.assertEqual(client.next(), {
                "type": "state", "user": None, "mode": None,
                "context": "CALIBRATION", "previous_context": None,
                "settings": DEFAULT_SETTINGS,
            })
        self.vision, self.bluetooth, self.game = self.clients

    def shutdown(self):
        for client in self.clients:
            client.close()
        self.hub.stop()
        self.server.join(timeout=3)
        self.assertFalse(self.server.is_alive())

    def states(self, **expected):
        for client in self.clients:
            state = client.next()
            self.assertEqual(state["type"], "state")
            for key, value in expected.items():
                self.assertEqual(state[key], value)

    def relay(self, sender, message):
        sender.send(message)
        for client in self.clients:
            if client is not sender:
                self.assertEqual(client.next(), message)

    def barrier(self):
        # A context request on the same socket proves earlier input was
        # processed; expected state also reveals unwanted queued messages.
        self.game.send({"type": "context", "value": "SETTINGS"})
        self.states(context="SETTINGS")

    def test_tracking_lost_found_restores_previous_context(self):
        self.game.send({"type": "context", "value": "PLAYING"})
        self.states(context="PLAYING", previous_context=None)
        self.relay(self.vision, {"type": "tracking", "status": "lost"})
        self.states(context="TRACKING_LOST", previous_context="PLAYING")
        # Repeated lost must not overwrite the remembered screen.
        self.relay(self.vision, {"type": "tracking", "status": "lost"})
        self.relay(self.vision, {"type": "tracking", "status": "found"})
        self.states(context="PLAYING", previous_context="PLAYING")

    def test_calibration_tracking_and_forwarding(self):
        self.relay(self.vision, {"type": "tracking", "status": "lost"})
        self.relay(self.vision, {"type": "tracking", "status": "found"})
        self.relay(self.vision, parse_command("calibration done"))
        self.relay(self.vision, parse_command("gesture jump"))
        self.relay(self.bluetooth, {"type": "bt_status", "status": "scanning"})
        self.barrier()  # No state changes or echoed gestures should be queued.

    def test_guest_clears_user_and_settings_merge(self):
        self.relay(self.bluetooth, {"type": "user", "id": "user_01"})
        self.states(user="user_01")
        self.relay(self.game, parse_command("mode bluetooth"))
        self.states(mode="bluetooth", user="user_01")
        self.relay(self.game, parse_command("mode guest"))
        self.states(mode="guest", user=None)
        self.relay(self.game, parse_command("settings hold_time 3.0"))
        self.states(settings={**DEFAULT_SETTINGS, "hold_time": 3.0})
        self.relay(self.game, parse_command("settings left_handed true"))
        self.states(settings={**DEFAULT_SETTINGS, "hold_time": 3.0, "left_handed": True})
        # A reconnect receives current state, not defaults.
        late = FakeClient(self.hub.port, "game")
        try:
            state = late.next()
            self.assertEqual(state["mode"], "guest")
            self.assertTrue(state["settings"]["left_handed"])
        finally:
            late.close()

    def test_malformed_invalid_and_fragmented_input(self):
        self.game.sock.sendall(b'not json\n[]\n\xff\n{"type":')
        self.game.sock.sendall(b'"context","value":"PLAYING"}\n')
        self.states(context="PLAYING")
        self.game.send({"type": "context", "value": ["bad"]})
        self.game.send({"type": "context", "value": "UNKNOWN"})
        self.game.send({"type": "settings", "volume": 5})
        self.game.send({"type": "settings", "left_handed": 1})
        self.game.send({"type": "settings", "unknown": 1})
        self.game.send(parse_command("calibration done"))  # Outside calibration.
        self.barrier()
        self.assertEqual(self.hub.state["settings"], DEFAULT_SETTINGS)

    def test_all_commands_and_rule_based_scores(self):
        for command in ("gesture open_palm", "gesture hold_x", "gesture fingers_4"):
            msg = parse_command(command)
            self.assertEqual(msg["score"], 1.0)
            self.assertFalse(msg["repeat"])
        self.assertIsNone(parse_command("user none")["id"])
        for command in ("gesture invalid", "tracking maybe", "settings volume true"):
            with self.assertRaises(ValueError):
                parse_command(command)


if __name__ == "__main__":
    unittest.main(verbosity=2)

