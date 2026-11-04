#!/usr/bin/env python3
"""
AYATLA - "RUN / THE CITY."  Game process (role "game").

    python game.py                 connect to the hub on 127.0.0.1:5000
    python game.py --host H --port P
    python game.py --no-hub        play alone with keyboard fakes (see KEYS)

Design rules (from the shared contract)
  * The hub is the source of truth: the screen shown is ALWAYS state.context.
    To change screens the game only SENDS a context request and waits for the
    new state. (The one exception is --no-hub, where LocalHub plays the hub.)
  * What a gesture MEANS comes from content/screens.json through
    ContextManager.resolve_action(state, gesture) - never hard-coded per key.
  * Colours come from the active profile (guest vs Bluetooth user).

PROTOCOL: every wire detail lives in the PROTOCOL block below and in
Game.handle_message(). One JSON object per line over TCP (shared contract).

  game -> hub   {"type":"hello","role":"game"}
                {"type":"context","value":"MAIN_MENU"}          (request)
                {"type":"mode","value":"bluetooth"|"guest"}
                {"type":"settings","gesture_sensitivity":0.72,"hold_time":2.0,
                 "volume":0.8,"camera_index":0,"left_handed":false}
  hub  -> game  {"type":"state","user":..,"mode":..,"context":..,
                 "previous_context":..,"settings":{...}}
                {"type":"gesture","name":"jump","score":0.91,"repeat":false}
                {"type":"calibration","progress":0..1,"seconds_left":2.4,
                 "camera":true,"body":true,"hands":false,"done":false}
                {"type":"bt_status","status":"scanning|found|not_found"}
                {"type":"user","id":"user_01"|null}
                {"type":"tracking","status":"lost|found"}

KEYS (--no-hub):  1-4 fingers_N | O ok_sign | X hold_x (hold the key) |
  Y thumbs_up (pause) | P open_palm | J jump | S squat | Left/Right lean | E object_power_up |
  T toggle tracking lost | B mode bluetooth | G mode guest |
  U toggle fake user_01 | C skip calibration | F1 key help | F11 fullscreen
"""

import argparse
import copy
import io
import json
import math
import queue
import random
import socket
import sys
import threading
import time
from array import array

import pygame

from context_manager import ContextManager

# --------------------------------------------------------------------------
# PROTOCOL (edit here if the shared contract differs)
# --------------------------------------------------------------------------
HUB_HOST, HUB_PORT = "127.0.0.1", 5000

CONTEXTS = ["CALIBRATION", "MAIN_MENU", "GESTURE_GUIDE", "PLAYER_MODE", "PLAYING",
            "PAUSED", "GAME_OVER", "SETTINGS", "QUIT_CONFIRM", "TRACKING_LOST", "BT_REGISTER"]


def msg_bt_register(action, address=None):
    """BT_REGISTER screen -> bluetooth process: start / cancel / confirm."""
    msg = {"type": "bt_register", "action": action}
    if address:
        msg["address"] = address
    return msg


def msg_hello():            return {"type": "hello", "role": "game"}
def msg_context(ctx):       return {"type": "context", "value": ctx}
def msg_mode(mode):         return {"type": "mode", "value": mode}


def msg_settings(settings):
    """Shared contract: flat fields, and ONLY the five known keys (the hub
    rejects the whole message if any key is unknown). Internally the game
    calls gesture_sensitivity "sensitivity" and also keeps a "camera" name."""
    return {
        "type": "settings",
        "gesture_sensitivity": float(settings.get("gesture_sensitivity",
                                                  settings.get("sensitivity", 0.72))),
        "hold_time": float(settings.get("hold_time", 2.0)),
        "volume": float(settings.get("volume", 0.8)),
        "camera_index": int(settings.get("camera_index", 0)),
        "left_handed": bool(settings.get("left_handed", False)),
    }


# --------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------
class NetClient(threading.Thread):
    """Background socket thread. Everything received goes into `inbox`
    (a thread-safe queue) and is drained once per frame by the main loop."""

    def __init__(self, host, port, inbox):
        super().__init__(daemon=True)
        self.host, self.port, self.inbox = host, port, inbox
        self.sock = None
        self.connected = False
        self._lock = threading.Lock()
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                s = socket.create_connection((self.host, self.port), timeout=3)
                s.settimeout(None)
                self.sock = s
                self.connected = True
                self.inbox.put({"type": "_connected"})
                self.send(msg_hello())
                buf = b""
                while not self._halt.is_set():
                    data = s.recv(4096)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            try:
                                self.inbox.put(json.loads(line.decode("utf-8")))
                            except ValueError:
                                pass
            except OSError:
                pass
            finally:
                was = self.connected
                self.connected = False
                with self._lock:
                    if self.sock:
                        try:
                            self.sock.close()
                        except OSError:
                            pass
                    self.sock = None
                if was:
                    self.inbox.put({"type": "_disconnected"})
            self._halt.wait(2.0)          # retry every 2 seconds (shared contract)

    def send(self, msg):
        with self._lock:
            if self.sock:
                try:
                    self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
                except OSError:
                    pass

    def stop(self):
        self._halt.set()


class LocalHub:
    """--no-hub: a tiny in-process hub that follows the same rules as the real
    one (state, context requests, mode -> bluetooth scan, tracking lost with
    automatic restore, calibration) so the game is fully playable alone."""

    def __init__(self, inbox):
        self.inbox = inbox
        self.connected = True
        self.state = {"context": "CALIBRATION", "mode": "guest", "user": None,
                      "settings": {}, "bt_status": None}
        self.prev_ctx = None
        self.fake_user = None
        self.timers = []
        self.calib_t = 0.0
        self.calib_push = 0.0

    def start(self):
        self.broadcast()

    # -- hub -> game
    def broadcast(self):
        self.inbox.put({"type": "state", "state": copy.deepcopy(self.state)})

    def push(self, msg):
        self.inbox.put(msg)

    # -- game -> hub
    def send(self, msg):
        t = msg.get("type")
        if t == "context":
            if self.state["context"] == "TRACKING_LOST":
                return
            ctx = msg.get("value")
            if ctx in CONTEXTS:
                self.state["context"] = ctx
                if ctx == "CALIBRATION":
                    self.calib_t = 0.0
                self.broadcast()
        elif t == "mode":
            mode = msg.get("value")
            self.state["mode"] = mode
            self.state["user"] = None
            if mode == "bluetooth":
                self.state["bt_status"] = "scanning"
                self.push({"type": "bt_status", "status": "scanning", "user": None})
                self.timers.append((time.time() + 1.5, self._bt_result))
            else:
                self.state["bt_status"] = None
            self.broadcast()
        elif t == "settings":
            self.state["settings"] = {k: v for k, v in msg.items() if k != "type"}
            self.broadcast()
        elif t == "bt_register":
            # Fake nearby devices; registering logs in as "test_player" (not saved).
            if msg.get("action") == "start":
                self.timers.append((time.time() + 1.0, lambda: self.push({
                    "type": "bt_devices", "status": "scanning", "nearest": "AA:BB:CC:00:00:01",
                    "devices": [{"address": "AA:BB:CC:00:00:01", "name": "Test-Phone", "paired": True,
                                 "rssi": -45, "registered": None},
                                {"address": "AA:BB:CC:00:00:02", "name": None, "rssi": -71, "registered": None},
                                {"address": "AA:BB:CC:00:00:03", "name": "JBL Flip 5", "rssi": -80, "registered": None}]})))
            elif msg.get("action") == "confirm":
                self.push({"type": "bt_devices", "status": "registered", "id": "test_player", "new": True,
                           "name": "Test-Phone", "matched_by": "name", "devices": []})
                self.state.update({"user": "test_player", "mode": "bluetooth", "bt_status": "found"})
                self.push({"type": "bt_status", "status": "found", "user": "test_player", "name": "Test-Phone"})
                self.broadcast()

    def _bt_result(self):
        if self.state["mode"] != "bluetooth":
            return
        if self.fake_user:
            self.state["user"] = self.fake_user
            self.state["bt_status"] = "found"
        else:
            self.state["bt_status"] = "not_found"
        self.push({"type": "bt_status", "status": self.state["bt_status"],
                   "user": self.state["user"]})
        self.broadcast()

    def toggle_tracking(self):
        if self.state["context"] != "TRACKING_LOST":
            self.prev_ctx = self.state["context"]
            self.state["context"] = "TRACKING_LOST"
            self.push({"type": "tracking", "status": "lost"})
        else:
            self.state["context"] = self.prev_ctx or "MAIN_MENU"
            self.push({"type": "tracking", "status": "strong"})
        self.broadcast()

    def skip_calibration(self):
        if self.state["context"] == "CALIBRATION":
            self.push({"type": "calibration", "progress": 1.0, "seconds_left": 0.0,
                       "camera": True, "body": True, "hands": True, "done": True})

    def toggle_fake_user(self):
        self.fake_user = None if self.fake_user else "user_01"
        return self.fake_user

    def tick(self, dt):
        now = time.time()
        for item in list(self.timers):
            if now >= item[0]:
                self.timers.remove(item)
                item[1]()
        if self.state["context"] == "CALIBRATION":
            self.calib_t += dt
            self.calib_push -= dt
            if self.calib_push <= 0:
                self.calib_push = 0.1
                total = 4.0
                prog = min(1.0, self.calib_t / total)
                self.push({"type": "calibration", "progress": prog,
                           "seconds_left": round(total * (1 - prog), 1),
                           "camera": self.calib_t > 0.2, "body": self.calib_t > 0.8,
                           "hands": self.calib_t > 1.4, "done": prog >= 1.0})


# --------------------------------------------------------------------------
# Colours / drawing helpers
# --------------------------------------------------------------------------
LW, LH = 1920, 1080
FPS = 60
WHITE = (242, 242, 252)


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Theme:
    """Colour scheme derived from the active profile (per-user colour scheme)."""

    def __init__(self, d):
        self.accent = hexrgb(d["accent"])
        self.accent2 = hexrgb(d["accent_2"])
        self.bg = hexrgb(d["background"])
        self.avatar = hexrgb(d["avatar"])
        self.panel = mix(self.bg, (255, 255, 255), 0.055)
        self.panel2 = mix(self.bg, self.accent, 0.16)
        self.border = mix(self.bg, self.accent, 0.30)
        self.text = WHITE
        self.muted = mix(WHITE, self.bg, 0.45)
        self.dim = mix(WHITE, self.bg, 0.72)
        self.gold = (255, 209, 102)
        self.danger = self.accent2


class Fonts:
    NAMES = "inter,segoeui,helveticaneue,helvetica,arial,dejavusans"
    cache = {}

    @classmethod
    def get(cls, size, bold=False, italic=False):
        key = (size, bold, italic)
        if key not in cls.cache:
            cls.cache[key] = pygame.font.SysFont(cls.NAMES, size, bold=bold, italic=italic)
        return cls.cache[key]


def text(surf, s, size, color, pos, anchor="topleft", bold=False, italic=False, alpha=None):
    img = Fonts.get(size, bold, italic).render(str(s), True, color)
    if alpha is not None:
        img.set_alpha(alpha)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def wrap(s, size, maxw, bold=False):
    font = Fonts.get(size, bold)
    lines = []
    for para in str(s).split("\n"):
        cur = ""
        for w in para.split(" "):
            t = (cur + " " + w).strip()
            if font.size(t)[0] <= maxw or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def rrect(surf, rect, color, radius=28, border=None, bw=2):
    """Rounded rect; color may be RGBA (translucent)."""
    rect = pygame.Rect(rect)
    if len(color) == 4 and color[3] < 255:
        tmp = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(tmp, color, tmp.get_rect(), border_radius=radius)
        surf.blit(tmp, rect.topleft)
    else:
        pygame.draw.rect(surf, color[:3], rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, border, rect, width=bw, border_radius=radius)


def pill(surf, center_or_topleft, label, color, bg, size=24, pad=22, h=48,
         anchor="topleft", border=None, icon=None, th=None):
    font = Fonts.get(size, True)
    w = font.size(label)[0] + pad * 2 + (size + 8 if icon else 0)
    r = pygame.Rect(0, 0, w, h)
    setattr(r, anchor, center_or_topleft)
    rrect(surf, r, bg, radius=h // 2, border=border or mix(bg, color, 0.35))
    x = r.x + pad
    if icon:
        draw_icon(surf, icon, x + size // 2, r.centery, size + 4, color)
        x += size + 8
    text(surf, label, size, color, (x, r.centery), "midleft", bold=True)
    return r


def dashed_poly(surf, pts, color, width=3, dash=14, gap=10):
    """Dashed line along a polyline."""
    on, left = True, dash
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg == 0:
            continue
        ux, uy = (x2 - x1) / seg, (y2 - y1) / seg
        pos = 0.0
        while pos < seg:
            step = min(left, seg - pos)
            if on:
                pygame.draw.line(surf, color, (x1 + ux * pos, y1 + uy * pos),
                                 (x1 + ux * (pos + step), y1 + uy * (pos + step)), width)
            pos += step
            left -= step
            if left <= 0:
                on = not on
                left = dash if on else gap


def rrect_points(rect, r, steps=8):
    rect = pygame.Rect(rect)
    pts = []
    corners = [(rect.right - r, rect.top + r, -90), (rect.right - r, rect.bottom - r, 0),
               (rect.left + r, rect.bottom - r, 90), (rect.left + r, rect.top + r, 180)]
    for cx, cy, a0 in corners:
        for k in range(steps + 1):
            a = math.radians(a0 + 90 * k / steps)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append(pts[0])
    return pts


def dot_ring(surf, center, radius, frac, on, off, n=72, dot=6):
    for i in range(n):
        a = -math.pi / 2 + 2 * math.pi * i / n
        p = (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
        pygame.draw.circle(surf, on if i / n < frac else off, p, dot)


# ---- icons ---------------------------------------------------------------
def draw_icon(surf, name, cx, cy, s, color, w=None):
    """Small vector icons, s = overall size in px."""
    w = w or max(2, s // 14)
    h = s / 2

    def P(x, y):
        return (cx + x * h, cy + y * h)

    if name == "check":
        pygame.draw.lines(surf, color, False, [P(-.7, 0), P(-.2, .55), P(.75, -.6)], w)
    elif name == "x":
        pygame.draw.line(surf, color, P(-.7, -.7), P(.7, .7), w)
        pygame.draw.line(surf, color, P(.7, -.7), P(-.7, .7), w)
    elif name == "arrow":
        pygame.draw.line(surf, color, P(-.8, 0), P(.8, 0), w)
        pygame.draw.lines(surf, color, False, [P(.2, -.55), P(.8, 0), P(.2, .55)], w)
    elif name == "arrow_left":
        pygame.draw.line(surf, color, P(.8, 0), P(-.8, 0), w)
        pygame.draw.lines(surf, color, False, [P(-.2, -.55), P(-.8, 0), P(-.2, .55)], w)
    elif name == "bluetooth":
        pygame.draw.lines(surf, color, False,
                          [P(-.55, -.45), P(.5, .45), P(0, .9), P(0, -.9), P(.5, -.45), P(-.55, .45)], w)
    elif name in ("person", "jump", "squat", "lean"):
        head = {"jump": P(0, -.55), "squat": P(-.1, -.2), "lean": P(.25, -.7), "person": P(0, -.7)}[name]
        pygame.draw.circle(surf, color, head, max(3, s // 9), w)
        if name == "person":
            pygame.draw.rect(surf, color, (cx - h * .3, cy - h * .45, h * .6, h * .9), w, border_radius=3)
        elif name == "jump":
            pygame.draw.line(surf, color, P(0, -.35), P(0, .25), w)
            pygame.draw.line(surf, color, P(0, -.25), P(-.6, -.75), w)
            pygame.draw.line(surf, color, P(0, -.25), P(.6, -.75), w)
            pygame.draw.line(surf, color, P(0, .25), P(-.45, .85), w)
            pygame.draw.line(surf, color, P(0, .25), P(.45, .85), w)
        elif name == "squat":
            pygame.draw.line(surf, color, P(-.1, 0), P(.1, .45), w)
            pygame.draw.lines(surf, color, False, [P(.1, .45), P(.6, .35), P(.55, .85)], w)
            pygame.draw.line(surf, color, P(-.05, .1), P(.45, .05), w)
        else:
            pygame.draw.line(surf, color, P(.2, -.45), P(-.05, .2), w)
            pygame.draw.line(surf, color, P(-.05, .2), P(-.5, .85), w)
            pygame.draw.line(surf, color, P(-.05, .2), P(.3, .85), w)
            pygame.draw.line(surf, color, P(.1, -.3), P(-.5, -.1), w)
            pygame.draw.line(surf, color, P(.1, -.3), P(.65, -.05), w)
    elif name in ("palm", "fingers"):
        up = 4 if name == "palm" else 2
        gold = (255, 209, 102)
        pygame.draw.rect(surf, gold, (cx - h * .5, cy - h * .05, h, h * .85), w, border_radius=int(h * .3))
        for i in range(up):
            x = cx - h * .38 + i * (h * .76 / max(1, up - 1)) if up > 1 else cx
            pygame.draw.line(surf, gold, (x, cy - h * .05), (x, cy - h * .85), w + 1)
    elif name == "thumb":
        gold = (255, 209, 102)
        # fist (curled fingers) with the thumb pointing up
        pygame.draw.rect(surf, gold, (cx - h * .35, cy - h * .15, h * .9, h * .95), w, border_radius=int(h * .25))
        for i in (1, 2, 3):
            y = cy - h * .15 + i * h * .95 / 4
            pygame.draw.line(surf, gold, (cx + h * .05, y), (cx + h * .55, y), max(1, w - 1))
        pygame.draw.line(surf, gold, P(-.2, -.15), P(-.2, -.95), w + 3)
        pygame.draw.circle(surf, gold, P(-.2, -.95), max(2, (w + 3) // 2))
    elif name == "ok":
        gold = (255, 209, 102)
        pygame.draw.circle(surf, gold, P(-.1, -.1), int(h * .38), w)
        for dx in (.25, .5, .75):
            pygame.draw.line(surf, gold, P(dx, -.1 + dx * .1), P(dx + .05, -.9 + dx * .35), w + 1)
        pygame.draw.rect(surf, gold, (cx - h * .55, cy + h * .25, h * 1.0, h * .6), w, border_radius=int(h * .2))
    elif name == "object":
        L = h * .9
        for sx in (-1, 1):
            for sy in (-1, 1):
                pygame.draw.lines(surf, color, False,
                                  [(cx + sx * L, cy + sy * L * .55), (cx + sx * L, cy + sy * L),
                                   (cx + sx * L * .55, cy + sy * L)], w)
        pygame.draw.polygon(surf, color, [P(0, -.45), P(.4, -.2), P(.4, .3), P(0, .55), P(-.4, .3), P(-.4, -.2)], w)


# ---- characters ------------------------------------------------------------
def blocky_figure(surf, cx, cy, s, lean, body, legs):
    """Front-view figure used in menus (like the Figma mannequin).
    lean in -1..1 shifts head/torso sideways."""
    ox = lean * 26 * s
    pygame.draw.circle(surf, body, (cx + ox * 1.4, cy - 150 * s), 28 * s)
    pygame.draw.rect(surf, body, (cx - 40 * s + ox, cy - 112 * s, 80 * s, 100 * s), border_radius=int(6 * s))
    pygame.draw.rect(surf, body, (cx - 66 * s + ox * 1.1, cy - 112 * s, 20 * s, 96 * s), border_radius=int(5 * s))
    pygame.draw.rect(surf, body, (cx + 46 * s + ox * 1.1, cy - 112 * s, 20 * s, 96 * s), border_radius=int(5 * s))
    pygame.draw.rect(surf, legs, (cx - 36 * s + ox * .3, cy - 6 * s, 30 * s, 100 * s), border_radius=int(5 * s))
    pygame.draw.rect(surf, legs, (cx + 6 * s + ox * .3, cy - 6 * s, 30 * s, 100 * s), border_radius=int(5 * s))


def runner(surf, x, foot_y, s, pose, t, body, legs):
    """Side-view runner (jump / slide / run). foot_y = ground contact."""
    def limb(hip, a1, a2, l1, l2, col, w):
        knee = (hip[0] + math.sin(a1) * l1, hip[1] + math.cos(a1) * l1)
        foot = (knee[0] + math.sin(a2) * l2, knee[1] + math.cos(a2) * l2)
        pygame.draw.line(surf, col, hip, knee, w)
        pygame.draw.line(surf, col, knee, foot, w)
        pygame.draw.circle(surf, col, knee, w // 2)
        return foot

    w = max(4, int(20 * s))
    if pose == "slide":
        hip = (x - 10 * s, foot_y - 36 * s)
        pygame.draw.line(surf, body, hip, (x + 62 * s, foot_y - 70 * s), int(42 * s))
        pygame.draw.circle(surf, body, (x + 92 * s, foot_y - 80 * s), 22 * s)
        pygame.draw.line(surf, body, (x + 40 * s, foot_y - 62 * s), (x + 100 * s, foot_y - 38 * s), w - 4)
        pygame.draw.line(surf, legs, hip, (x - 70 * s, foot_y - 22 * s), w)
        pygame.draw.line(surf, legs, hip, (x - 60 * s, foot_y - 6 * s), w)
        return
    hip = (x, foot_y - 90 * s)
    sh = (x + 12 * s, foot_y - 150 * s)
    if pose == "jump":
        limb(hip, -0.7, 0.2, 48 * s, 46 * s, legs, w)
        limb(hip, 0.8, -0.3, 48 * s, 46 * s, legs, w)
        pygame.draw.line(surf, body, hip, sh, int(40 * s))
        pygame.draw.circle(surf, body, (sh[0] + 8 * s, sh[1] - 34 * s), 22 * s)
        pygame.draw.line(surf, body, sh, (sh[0] - 54 * s, sh[1] - 40 * s), w - 4)
        pygame.draw.line(surf, body, sh, (sh[0] + 62 * s, sh[1] - 38 * s), w - 4)
        return
    ph = t * 11
    a = math.sin(ph) * 0.9
    b = math.sin(ph + math.pi) * 0.9
    limb(hip, a, a - 0.7 - 0.3 * math.cos(ph), 46 * s, 46 * s, legs, w)
    limb(hip, b, b - 0.7 - 0.3 * math.cos(ph + math.pi), 46 * s, 46 * s, legs, w)
    pygame.draw.line(surf, body, hip, sh, int(40 * s))
    pygame.draw.circle(surf, body, (sh[0] + 14 * s, sh[1] - 32 * s), 22 * s)
    pygame.draw.line(surf, body, sh, (sh[0] + math.sin(ph) * 50 * s, sh[1] + 44 * s), w - 4)
    pygame.draw.line(surf, body, sh, (sh[0] + math.sin(ph + math.pi) * 50 * s, sh[1] + 44 * s), w - 4)


# --------------------------------------------------------------------------
# Gameplay logic (no pygame inside: easy to test)
# --------------------------------------------------------------------------
class Run:
    def __init__(self, gp, rng=None):
        self.gp = gp
        self.mod = gp.get("mod", {})
        self.rng = rng or random.Random()
        pl = gp["player"]
        self.jump_dur = pl["jump_duration"]
        self.slide_dur = pl["slide_duration"]
        self.switch_time = pl["lane_switch_time"]
        self.lanes = gp["lane_count"]
        self.lane = self.lanes // 2
        self.lane_x = float(self.lane)
        self.jump_t = None
        self.slide_t = None
        self.t = 0.0
        self.dist = 0.0
        self.speed = gp["base_speed"] * self.mod.get("speed_mult", 1.0)
        self.score_f = 0.0
        self.coins = 0
        self.objs = []
        self.next_row_at = gp["spawn"]["first_row_at"]
        self.stored = self.mod.get("start_powerup")
        self.active = None
        self.active_left = 0.0
        self.over = False
        self.grace = 1.2
        self.events = []
        self.SPAWN_Z = 75.0

    # ---- read-only helpers
    @property
    def score(self):
        return int(self.score_f)

    @property
    def height(self):
        if self.jump_t is None:
            return 0.0
        return math.sin(math.pi * min(1.0, self.jump_t / self.jump_dur))

    @property
    def sliding(self):
        return self.slide_t is not None

    def score_mult(self):
        m = self.mod.get("score_mult", 1.0)
        return m * (2.0 if self.active == "double" else 1.0)

    def powerup_label(self, kind):
        return self.gp["powerups"]["types"][kind]["label"]

    # ---- player input
    def jump(self):
        if self.over or self.grace > 0:
            return
        self.slide_t = None
        if self.jump_t is None:
            self.jump_t = 0.0

    def slide(self):
        if self.over or self.grace > 0:
            return
        self.jump_t = None            # fast-fall
        self.slide_t = 0.0

    def move(self, d):
        if self.over or self.grace > 0:
            return
        self.lane = max(0, min(self.lanes - 1, self.lane + d))

    def set_lane(self, lane):
        """Absolute lane, from where the player stands in the camera picture."""
        if self.over:
            return False
        lane = max(0, min(self.lanes - 1, lane))
        if self.grace > 0:
            self.lane_x = float(lane)     # still "GET READY": just place the runner there
        changed = lane != self.lane
        self.lane = lane
        return changed

    def use_powerup(self):
        if self.stored and not self.over:
            self.active = self.stored
            self.active_left = self.gp["powerups"]["types"][self.stored]["duration"]
            self.stored = None
            self.events.append(("powerup_on", self.active))
            return True
        return False

    # ---- spawning
    def _gap(self):
        sp = self.gp["spawn"]
        g = sp["base_gap"] * self.mod.get("gap_mult", 1.0) - sp["gap_shrink_per_100m"] * (self.dist / 100.0)
        g = max(sp["min_gap"], g)
        return g + self.rng.uniform(0, 4)

    def _pick(self, table):
        names = list(table)
        weights = [table[n].get("weight", 1) for n in names]
        return self.rng.choices(names, weights)[0]

    def _spawn_row(self, z):
        sp = self.gp["spawn"]
        n = 2 if self.rng.random() < sp["double_row_chance"] else 1
        n = min(n, self.lanes - 1)                       # never wall all lanes
        lanes = self.rng.sample(range(self.lanes), n)
        for ln in lanes:
            kind = self._pick(self.gp["obstacle_types"])
            self.objs.append({"k": "obstacle", "type": kind, "lane": ln, "z": z,
                              "ok": True, "dead": False, "scored": False})
        co = self.gp["coins"]
        if self.rng.random() < co["trail_chance"] * self.mod.get("coin_mult", 1.0):
            ln = self.rng.randrange(self.lanes)
            for i in range(self.rng.randint(co["trail_min"], co["trail_max"])):
                self.objs.append({"k": "coin", "lane": ln, "z": z + 9 + i * co["spacing"], "dead": False})
        pu = self.gp["powerups"]
        if self.rng.random() < pu["spawn_chance"]:
            self.objs.append({"k": "power", "type": self._pick(pu["types"]),
                              "lane": self.rng.randrange(self.lanes), "z": z + 17, "dead": False})

    # ---- simulation
    def update(self, dt):
        if self.over:
            return
        if self.grace > 0:
            self.grace -= dt
            return
        gp = self.gp
        self.t += dt
        self.speed = min(gp["max_speed"], gp["base_speed"] + gp["speed_ramp"] * self.t) \
            * self.mod.get("speed_mult", 1.0)
        dz = self.speed * dt
        self.dist += dz
        self.score_f += dz * gp["scoring"]["per_meter"] * self.score_mult()
        for o in self.objs:
            o["z"] -= dz

        # lane glide
        step = dt / max(0.01, self.switch_time)
        if self.lane_x < self.lane:
            self.lane_x = min(self.lane, self.lane_x + step)
        elif self.lane_x > self.lane:
            self.lane_x = max(self.lane, self.lane_x - step)

        if self.jump_t is not None:
            self.jump_t += dt
            if self.jump_t >= self.jump_dur:
                self.jump_t = None
        if self.slide_t is not None:
            self.slide_t += dt
            if self.slide_t >= self.slide_dur:
                self.slide_t = None

        if self.active:
            self.active_left -= dt
            if self.active_left <= 0:
                self.events.append(("powerup_off", self.active))
                self.active = None

        while self.dist + self.SPAWN_Z >= self.next_row_at:
            self._spawn_row(self.next_row_at - self.dist)
            self.next_row_at += self._gap()

        bonus = gp["scoring"]["clear_bonus"]
        for o in self.objs:
            z = o["z"]
            same = abs(self.lane_x - o["lane"]) < 0.55
            if o["k"] == "obstacle":
                if -0.8 < z < 0.9 and same and not o["dead"]:
                    req = gp["obstacle_types"][o["type"]]["requires"]
                    safe = (req == "jump" and self.height > 0.35) or (req == "slide" and self.sliding)
                    if not safe:
                        if self.active == "shield":
                            o["dead"] = True
                            self.events.append(("shield_hit", o["type"]))
                        else:
                            o["ok"] = False
                            self.over = True
                            self.events.append(("crash", o["type"]))
                            return
                if z < -0.8 and not o["scored"]:
                    o["scored"] = True
                    req = gp["obstacle_types"][o["type"]]["requires"]
                    if o["ok"] and not o["dead"] and same and req in ("jump", "slide"):
                        pts = int(bonus * self.score_mult())
                        self.score_f += pts
                        self.events.append(("clear", o["type"], pts))
            elif o["k"] == "coin":
                near = abs(self.lane_x - o["lane"]) < 0.6 or (self.active == "magnet" and z < 16)
                if -0.8 < z < 0.9 and near:
                    o["dead"] = True
                    self.coins += 1
                    self.score_f += gp["coins"]["value"] * self.score_mult()
                    self.events.append(("coin",))
            elif o["k"] == "power":
                if -0.8 < z < 0.9 and same:
                    o["dead"] = True
                    self.stored = o["type"]
                    self.events.append(("powerup_get", o["type"]))
        self.objs = [o for o in self.objs if not o["dead"] and o["z"] > -4]


# --------------------------------------------------------------------------
# Sound (tiny synthesised beeps; silently disabled if no audio device)
# --------------------------------------------------------------------------
class Sfx:
    def __init__(self):
        self.ok = False
        self.sounds = {}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(44100, -16, 1, 512)
            self.ok = True
        except pygame.error:
            return
        spec = {"select": (660, 70), "confirm": (880, 120), "coin": (1320, 80), "jump": (520, 110),
                "slide": (300, 110), "hit": (130, 380), "clear": (990, 140), "save": (760, 160)}
        for name, (f, ms) in spec.items():
            n = int(44100 * ms / 1000)
            buf = array("h", (int(30000 * math.sin(2 * math.pi * f * i / 44100) * (1 - i / n)) for i in range(n)))
            try:
                self.sounds[name] = pygame.mixer.Sound(buffer=buf.tobytes())
            except pygame.error:
                pass

    def play(self, name, volume):
        s = self.sounds.get(name)
        if s and self.ok:
            s.set_volume(max(0.0, min(1.0, volume)))
            s.play()


# --------------------------------------------------------------------------
# The game
# --------------------------------------------------------------------------
GESTURE_KEYS = {
    pygame.K_1: "fingers_1", pygame.K_2: "fingers_2", pygame.K_3: "fingers_3", pygame.K_4: "fingers_4",
    pygame.K_o: "ok_sign", pygame.K_p: "open_palm", pygame.K_y: "thumbs_up", pygame.K_j: "jump",
    pygame.K_s: "squat", pygame.K_LEFT: "lean_left", pygame.K_RIGHT: "lean_right",
    pygame.K_e: "object_power_up",
}
INPUT_LOCK = 0.45          # ignore stray gestures right after a screen change
# Vision splits the camera picture into thirds; the player's third is their lane.
ZONES = ("zone_left", "zone_center", "zone_right")


def zone_lane(zone, lanes):
    """zone_left -> first lane, zone_center -> middle lane, zone_right -> last lane."""
    return {"zone_left": 0, "zone_center": lanes // 2, "zone_right": lanes - 1}[zone]
HORIZON, FOOT_Y, LANE_W, Z0 = 300.0, 960.0, 330.0, 9.0
ROAD_CX = LW / 2


class Game:
    def __init__(self, args):
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()
        pygame.display.set_caption("AYATLA · RUN / THE CITY.")
        pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        self.canvas = pygame.Surface((LW, LH))
        self.clock = pygame.time.Clock()
        self.sfx = Sfx()
        self.cm = ContextManager()
        self.inbox = queue.Queue()
        self.bt_name = None                 # name of the device bluetooth matched
        self.local = None
        if args.no_hub:
            self.local = LocalHub(self.inbox)
            self.net = self.local
        else:
            self.net = NetClient(args.host, args.port, self.inbox)
        self.keys_enabled = args.no_hub or args.keys
        self.show_help = args.no_hub

        # shared state (mirror of the hub's)
        self.state = {"context": "CALIBRATION", "mode": "guest", "user": None,
                      "settings": {}, "bt_status": None}
        self.context = "CALIBRATION"
        self.connected = args.no_hub
        self.ident = None
        self.profile = self.theme = self.gp = None
        self.active_settings = {}
        self.run = None
        self.content = {}
        self.refresh_identity(send=False)

        # ui state
        self.content = self.cm.get_screen_content(self.state)
        self.ctx_entered = time.time()
        self.focus = 0
        self.return_to = None
        self.requested = {}
        self.toast = None
        self.last_gesture = None            # (name, time)
        self.zone = None                    # zone_left / zone_center / zone_right (from vision)
        self.reg = {"devices": [], "nearest": None, "status": None, "message": "",
                    "picked": False, "result": None}   # BT_REGISTER screen
        self.guide_done = set()
        self.guide_back_at = None
        self.calib = {"progress": 0.0, "seconds_left": None, "camera": False,
                      "body": False, "hands": False, "done": False}
        self.tracking_strong = True
        self.bt_status = None
        self.awaiting_bt = False
        self.draft = {}
        self.dirty = False
        self.quit_remaining = 3.0
        self.quit_enter = time.time()
        self.quit_last_hold = None
        self.run = None
        self.last_result = None
        self.x_held = False
        self.x_timer = 0.0
        self.running = True
        self.camera_preview = None
        self.preview_socket = None
        try:
            self.preview_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.preview_socket.bind(("127.0.0.1", 5001))
            self.preview_socket.setblocking(False)
        except OSError as exc:
            if self.preview_socket is not None:
                self.preview_socket.close()
            self.preview_socket = None
            print(f"[game] in-game camera preview unavailable: {exc}")
        self.t = 0.0
        self.skyline = [(i * 150 - 40, 90 + (i * 53) % 150, 150 + (i * 97) % 260) for i in range(15)]

    # ------------------------------------------------------------ plumbing
    def send(self, msg):
        self.net.send(msg)

    def request(self, ctx):
        """Ask the hub to change screen. Never changes self.context itself."""
        now = time.time()
        if now - self.requested.get(ctx, 0) < 0.35:
            return
        self.requested[ctx] = now
        self.send(msg_context(ctx))

    def vol(self):
        return self.active_settings.get("volume", 0.7)

    def play(self, name):
        self.sfx.play(name, self.vol())

    def set_toast(self, title, sub="", dur=1.6):
        self.toast = {"title": title, "sub": sub, "t0": time.time(), "dur": dur}

    # ------------------------------------------------------ hub -> game
    def drain_inbox(self):
        while True:
            try:
                m = self.inbox.get_nowait()
            except queue.Empty:
                return
            self.handle_message(m)

    def handle_message(self, m):
        t = m.get("type")
        if t == "state":
            self.apply_state(m.get("state", m))
        elif t == "gesture":
            g = m.get("gesture") or m.get("name")
            if g:
                self.on_gesture(g, bool(m.get("repeat")))
        elif t == "calibration":
            p = m.get("progress", 0.0)
            if p > 1.0:
                p /= 100.0
            self.calib.update({"progress": p, "seconds_left": m.get("seconds_left"),
                               "camera": bool(m.get("camera")), "body": bool(m.get("body")),
                               "hands": bool(m.get("hands")), "done": bool(m.get("done"))})
            if self.calib["done"] and self.context == "CALIBRATION":
                self.request("MAIN_MENU")
        elif t == "bt_status":
            if m.get("name"):
                self.bt_name = m["name"]             # names a brand-new profile
            self.on_bt_status(m.get("status"), m.get("user"))
        elif t == "bt_devices":
            self.on_bt_devices(m)
        elif t == "tracking":
            st = m.get("status")
            self.tracking_strong = not (m.get("lost") or st in ("lost", "TRACKING_LOST"))
        elif t == "_connected":
            self.connected = True
        elif t == "_disconnected":
            self.connected = False

    def apply_state(self, new):
        new = {k: v for k, v in new.items() if k != "type"}
        old_ctx = self.context
        self.state.update(new)
        if "bt_status" in new and new["bt_status"] != self.bt_status:
            self.on_bt_status(new["bt_status"], None)
        if self.state.get("mode") != "bluetooth":
            self.bt_status = None
        self.refresh_identity()
        ctx = self.state.get("context")
        if ctx and ctx != old_ctx:
            self.context = ctx
            self.on_enter(ctx, old_ctx)
        self.content = self.cm.get_screen_content(self.state)

    def refresh_identity(self, send=True):
        uid = self.state.get("user")
        if self.state.get("mode") == "bluetooth" and uid and not self.cm.has_profile(uid):
            # First time this Bluetooth user plays: name the profile after the device.
            self.cm.create_profile(uid, self.bt_name or uid, persist=self.local is None)
        ident = self.cm.profile_id(self.state)
        if ident == self.ident:
            return
        first = self.ident is None
        self.ident = ident
        self.profile = self.cm.get_profile(self.state)
        self.theme = Theme(self.cm.get_theme(self.state))
        self.gp = self.cm.get_gameplay(self.state)
        self.active_settings = self.cm.get_settings(self.state)
        self.content = self.cm.get_screen_content(self.state)
        if send and not first:
            self.send(msg_settings(self.active_settings))      # vision uses these
        if self.run is not None and self.run.t < 0.01 and not self.run.over:
            self.run = Run(self.gp)                            # identity arrived late

    def on_bt_status(self, status, user):
        self.bt_status = status
        if user and not self.state.get("user"):
            self.state["user"] = user
            self.state["mode"] = "bluetooth"
            self.refresh_identity()
        if status == "found" and self.context in ("PLAYER_MODE", "BT_REGISTER") and self.awaiting_bt:
            self.awaiting_bt = False
            self.run = None
            self.request("PLAYING")
        elif status == "not_found" and self.context != "BT_REGISTER":
            self.awaiting_bt = False

    def on_bt_devices(self, m):
        """Messages from the bluetooth process while registering a device."""
        if self.context != "BT_REGISTER":
            return
        status = m.get("status")
        self.reg["status"] = status
        if status == "scanning":
            self.reg["devices"] = m.get("devices") or []
            self.reg["nearest"] = m.get("nearest")
            if not self.reg["picked"]:
                self.focus = 0                       # the strongest signal is listed first
            self.focus = min(self.focus, max(0, len(self.reg["devices"]) - 1))
        elif status == "registered":
            uid, name = m.get("id"), m.get("name")
            # A NEW registration always gets a fresh profile named after the
            # device (never an old one that happens to use the same id).
            self.cm.create_profile(uid, name, overwrite=bool(m.get("new")), persist=self.local is None)
            self.reg["result"] = {"id": uid, "name": name, "matched_by": m.get("matched_by")}
            self.play("save")
            self.set_toast("Device registered", f"Welcome, {name}!", 2.4)
        else:                                        # error / timeout
            self.reg["message"] = m.get("message", "")

    # ----------------------------------------------------- screen entry
    def on_enter(self, ctx, old):
        now = time.time()
        self.content = self.cm.get_screen_content(self.state)   # content of the NEW screen
        self.ctx_entered = now
        self.focus = 0
        self.toast = None
        if ctx in ("MAIN_MENU", "PLAYER_MODE", "GAME_OVER", "GESTURE_GUIDE"):
            self.run = None
        if old == "BT_REGISTER" and ctx != "TRACKING_LOST" and not self.reg["result"]:
            self.send(msg_bt_register("cancel"))     # left without registering
        if ctx == "BT_REGISTER":
            self.reg = {"devices": [], "nearest": None, "status": None, "message": "",
                        "picked": False, "result": None}
            self.awaiting_bt = True                  # log in + start once registered
            self.send(msg_mode("bluetooth"))
            self.send(msg_bt_register("start"))
        elif ctx == "PLAYER_MODE":
            self.bt_status = None
            self.awaiting_bt = False
        elif ctx == "PLAYING":
            if self.run is None:
                self.run = Run(self.gp)
                if self.zone:                        # start in the lane the player stands in
                    self.run.set_lane(zone_lane(self.zone, self.run.lanes))
            else:
                self.run.grace = 1.2                 # resume: short "GET READY"
        elif ctx == "SETTINGS":
            self.draft = copy.deepcopy(self.active_settings)
            self.dirty = False
        elif ctx == "QUIT_CONFIRM":
            self.quit_remaining = self.content["params"]["seconds"]
            self.quit_enter = now
            self.quit_last_hold = None
        elif ctx == "GESTURE_GUIDE":
            self.guide_back_at = None
        elif ctx == "CALIBRATION":
            self.calib["done"] = False
        self.content = self.cm.get_screen_content(self.state)

    # --------------------------------------------------------- gestures
    def on_gesture(self, name, repeat):
        now = time.time()
        if name in ZONES:
            self.zone = name                          # remembered on every screen
        if self.context == "TRACKING_LOST":
            return
        if not repeat:
            self.last_gesture = (name, now)
        if self.context == "QUIT_CONFIRM":
            if name == "hold_x":
                self.quit_last_hold = now            # repeat:true keeps it alive
            return
        if repeat:
            # Zone repeats keep the lane in sync with where the player stands;
            # other repeats only matter for quit.
            if name in ZONES and self.context == "PLAYING" and self.run is not None:
                self.run.set_lane(zone_lane(name, self.run.lanes))
            return
        if now - self.ctx_entered < INPUT_LOCK and self.context != "PLAYING":
            return
        action = self.cm.resolve_action(self.state, name)
        if action:
            self.do_action(action, name)

    def gesture_text(self, name):
        gl = self.content.get("gesture_labels", {}).get(name, {})
        return gl.get("detected", name)

    def do_action(self, action, gesture=None):
        c = self.content
        opts = c.get("options", [])
        ctx = self.context

        if ctx == "BT_REGISTER" and action.startswith("focus_") and action[6:].isdigit():
            idx = int(action[6:]) - 1
            if idx < len(self.reg["devices"]):
                self.focus, self.reg["picked"] = idx, True
                self.play("select")
        elif action == "register_device":
            devices = self.reg["devices"]
            if self.reg["result"] is None and 0 <= self.focus < len(devices):
                self.reg["status"] = "registering"
                self.play("confirm")
                self.send(msg_bt_register("confirm", devices[self.focus]["address"]))
        elif action.startswith("focus_") and action[6:].isdigit():
            self.focus_option(int(action[6:]) - 1)
        elif action == "focus_4_cycle":
            # Settings has 5 rows but only 4 finger counts: fingers_4 focuses
            # row 4 (camera); showing 4 again moves to row 5 (left-handed),
            # and again goes back to row 4.
            self.focus = 4 if self.focus == 3 else 3
            self.play("select")
        elif action == "focus_bluetooth":
            self.focus = 0
            self.play("select")
        elif action == "focus_guest":
            self.focus = 1
            self.bt_status = None
            self.play("select")
        elif action == "confirm":
            if 0 <= self.focus < len(opts):
                self.play("confirm")
                self.do_action(opts[self.focus]["action"])
        elif action.startswith("goto:"):
            target = action[5:]
            if target in ("SETTINGS", "QUIT_CONFIRM"):
                self.return_to = ctx
            self.request(target)
        elif action == "back":
            self.go_back()
        elif action == "resume":
            self.request("PLAYING")
        elif action == "restart":
            self.run = None
            self.request("PLAYING")
        elif action == "main_menu":
            self.request("MAIN_MENU")
        elif action == "start_bluetooth":
            self.bt_status = "scanning"
            self.awaiting_bt = True
            self.send(msg_mode("bluetooth"))
        elif action == "start_guest":
            self.awaiting_bt = False
            self.send(msg_mode("guest"))
            self.run = None
            self.request("PLAYING")
        elif action in ("jump", "slide", "lane_left", "lane_right", "lane_zone", "use_powerup", "pause"):
            self.gameplay_action(action, gesture)
        elif action == "setting_decrease":
            self.adjust_setting(-1)
        elif action == "setting_increase":
            self.adjust_setting(+1)
        elif action == "save_settings":
            self.save_settings()
        elif action == "reset_settings":
            self.draft = self.cm.default_settings()
            self.dirty = True
            self.set_toast("Defaults restored", c["notes"]["reset"], 2.4)
        elif action.startswith("practice:"):
            self.practice(action[9:], gesture)

    def focus_option(self, idx):
        if 0 <= idx < len(self.content.get("options", [])):
            self.focus = idx
            self.play("select")

    def go_back(self):
        target = self.content.get("back_to")
        if not target:
            return
        if target == "@return":
            self.request(self.return_to or "MAIN_MENU")
        elif target.startswith("action:"):
            self.do_action(target[7:])
        else:
            self.request(target)

    # ---- gameplay actions
    def gameplay_action(self, action, gesture):
        if action == "pause":
            self.request("PAUSED")
            return
        r = self.run
        if self.context != "PLAYING" or r is None or r.over:
            return
        if action == "jump":
            r.jump(); self.play("jump"); self.set_toast(self.gesture_text(gesture), "Jump!", 1.0)
        elif action == "slide":
            r.slide(); self.play("slide"); self.set_toast(self.gesture_text(gesture), "Slide!", 1.0)
        elif action == "lane_left":
            r.move(-1); self.set_toast(self.gesture_text(gesture), "Lane changed", 1.0)
        elif action == "lane_right":
            r.move(+1); self.set_toast(self.gesture_text(gesture), "Lane changed", 1.0)
        elif action == "lane_zone" and gesture in ZONES:
            if r.set_lane(zone_lane(gesture, r.lanes)):
                self.set_toast(self.gesture_text(gesture), "Lane changed", 1.0)
        elif action == "use_powerup":
            if not r.use_powerup():
                self.set_toast(self.gesture_text(gesture), "No power-up to activate", 1.2)

    def consume_run_events(self):
        r = self.run
        for ev in r.events:
            kind = ev[0]
            if kind == "coin":
                self.play("coin")
            elif kind == "clear":
                spec = self.gp["obstacle_types"][ev[1]]
                self.play("clear")
                self.set_toast(spec["label"], spec["sub"].format(bonus=ev[2]), 1.5)
            elif kind == "powerup_get":
                self.play("confirm")
                self.set_toast("Power-up collected", f"{r.powerup_label(ev[1])} ready · show your object", 2.0)
            elif kind == "powerup_on":
                self.play("save")
                self.set_toast("Power-up active", r.powerup_label(ev[1]), 1.5)
            elif kind == "powerup_off":
                self.set_toast("Power-up ended", "", 1.2)
            elif kind == "shield_hit":
                self.play("hit")
                self.set_toast("Shield absorbed the hit", "", 1.2)
            elif kind == "crash":
                self.play("hit")
        r.events = []

    def finish_run(self):
        r = self.run
        res = self.cm.save_score(self.state, r.score)         # guests: never saved
        self.last_result = {"score": r.score, "best": res["best"], "new_best": res["new_best"],
                            "km": r.dist / 1000.0, "guest": self.ident == "guest",
                            "name": self.profile.get("display_name", "")}
        r.reported = True
        self.request("GAME_OVER")

    # ---- gesture guide
    def practice(self, card_id, gesture):
        self.guide_done.add(card_id)
        self.play("confirm")
        if card_id == "hold_x":
            self.guide_back_at = time.time() + 0.9        # hold X returns to the menu

    # ---- settings
    def adjust_setting(self, d):
        opts = self.content["options"]
        if not (0 <= self.focus < len(opts)):
            return
        o = opts[self.focus]
        key, kind = o["key"], o["kind"]
        if kind == "level":
            v = self.draft.get(key, o["min"]) + d * o["step"]
            self.draft[key] = round(max(o["min"], min(o["max"], v)), 3)
        elif kind == "choice":
            ch = o["choices"]
            cur = self.draft.get(key)
            i = ch.index(cur) if cur in ch else 0
            i = (i + d) % len(ch)
            self.draft[key] = ch[i]
            self.draft["camera_index"] = i
        elif kind == "toggle":
            self.draft[key] = d > 0          # lean right = ON, lean left = OFF
        self.dirty = True
        self.sfx.play("select", self.draft.get("volume", self.vol()))

    def save_settings(self):
        n = self.content["notes"]
        self.active_settings = copy.deepcopy(self.draft)
        self.send(msg_settings(self.active_settings))
        stored = self.cm.save_settings(self.state, self.active_settings)   # Bluetooth only
        self.dirty = False
        self.play("save")
        self.set_toast("Settings saved", n["saved_profile"] if stored else n["saved_guest"], 2.4)

    def fmt_setting(self, o, v):
        f = o.get("format")
        if o["kind"] == "toggle":
            return "Enabled" if v else "Disabled"
        if o["kind"] == "choice":
            return str(v)
        if f == "level3":
            return "Low" if v < 0.34 else "Medium" if v < 0.67 else "High"
        if f == "seconds":
            return f"{v:.1f} sec"
        if f == "percent":
            return f"{int(round(v * 100))}%"
        return str(v)

    # ------------------------------------------------------------- camera preview
    def poll_camera_preview(self):
        """Keep only the newest JPEG frame from the local vision process."""
        if self.preview_socket is None:
            return
        newest = None
        while True:
            try:
                newest, _ = self.preview_socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                return
        if newest:
            try:
                decoded = pygame.image.load(io.BytesIO(newest)).convert()
                self.camera_preview = pygame.transform.scale(decoded, (222, 166))
            except (pygame.error, ValueError):
                pass

    # ------------------------------------------------------------- update
    def update(self, dt):
        self.poll_camera_preview()
        now = time.time()
        self.t += dt
        if self.local:
            self.local.tick(dt)
            if self.x_held:                   # hold the X key = repeating hold_x
                self.x_timer -= dt
                if self.x_timer <= 0:
                    self.x_timer = 0.5
                    self.inbox.put({"type": "gesture", "gesture": "hold_x", "repeat": True})
        if self.toast and now - self.toast["t0"] > self.toast["dur"]:
            self.toast = None
        ctx = self.context
        if ctx == "PLAYING" and self.run is not None and not self.run.over:
            self.run.update(dt)
            self.consume_run_events()
            if self.run.over and not getattr(self.run, "reported", False):
                self.finish_run()
        elif ctx == "QUIT_CONFIRM":
            p = self.content["params"]
            if self.quit_last_hold is None:
                if now - self.quit_enter > p["entry_grace"]:
                    self.go_back()                              # nobody crossed arms
            elif now - self.quit_last_hold > p["cancel_after_idle"]:
                self.go_back()                                  # X released: cancel
            else:
                self.quit_remaining -= dt
                if self.quit_remaining <= 0:
                    self.running = False                        # exit the game
        elif ctx == "GESTURE_GUIDE" and self.guide_back_at and now >= self.guide_back_at:
            self.guide_back_at = None
            self.request("MAIN_MENU")
        # TRACKING_LOST: nothing updates -> the run stays frozen.

    # -------------------------------------------------------------- input
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_F1:
                    self.show_help = not self.show_help
                elif self.keys_enabled:
                    self.fake_key(e.key, True)
            elif e.type == pygame.KEYUP and self.keys_enabled and e.key == pygame.K_x:
                self.x_held = False

    def fake_key(self, key, down):
        if key in GESTURE_KEYS:
            self.inbox.put({"type": "gesture", "gesture": GESTURE_KEYS[key], "repeat": False})
        elif key == pygame.K_x:
            self.x_held = True
            self.x_timer = 0.5
            self.inbox.put({"type": "gesture", "gesture": "hold_x", "repeat": False})
        elif self.local:
            if key == pygame.K_t:
                self.local.toggle_tracking()
            elif key == pygame.K_b:
                self.local.send(msg_mode("bluetooth"))
            elif key == pygame.K_g:
                self.local.send(msg_mode("guest"))
            elif key == pygame.K_u:
                u = self.local.toggle_fake_user()
                self.set_toast("Fake Bluetooth user", u or "none (device not found)", 1.5)
            elif key == pygame.K_c:
                self.local.skip_calibration()

    # --------------------------------------------------------------- main
    def run_loop(self):
        if not self.local:
            self.net.start()
        else:
            self.local.start()
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self.handle_events()
            self.drain_inbox()
            self.update(dt)
            self.draw()
        if not self.local:
            self.net.stop()
        if self.preview_socket is not None:
            self.preview_socket.close()
        pygame.quit()

    # ======================================================================
    # DRAWING
    # ======================================================================
    def draw(self):
        s = self.canvas
        ctx = self.context
        fn = {
            "CALIBRATION": self.draw_calibration, "MAIN_MENU": self.draw_main_menu,
            "GESTURE_GUIDE": self.draw_guide, "PLAYER_MODE": self.draw_player_mode,
            "PLAYING": self.draw_playing, "PAUSED": self.draw_paused,
            "GAME_OVER": self.draw_game_over, "SETTINGS": self.draw_settings,
            "QUIT_CONFIRM": self.draw_quit, "TRACKING_LOST": self.draw_tracking_lost,
            "BT_REGISTER": self.draw_bt_register,
        }.get(ctx, self.draw_main_menu)
        fn(s)
        self.draw_overlays(s)
        win = pygame.display.get_surface()
        ww, wh = win.get_size()
        sc = min(ww / LW, wh / LH)
        w, h = int(LW * sc), int(LH * sc)
        if (w, h) == (LW, LH):
            win.blit(s, (0, 0))
        else:
            win.fill((0, 0, 0))
            win.blit(pygame.transform.smoothscale(s, (w, h)), ((ww - w) // 2, (wh - h) // 2))
        pygame.display.flip()

    def draw_overlays(self, s):
        th = self.theme
        if not self.connected:
            pill(s, (LW // 2, 30), "Connecting to hub…", th.accent2, th.panel, size=22, h=40,
                 anchor="midtop")
        if self.show_help and self.keys_enabled:
            r = pygame.Rect(0, 0, LW, 40)
            rrect(s, (0, 0, 0, 170), (0, LH - 44, LW, 44), 0)
            msg = ("NO-HUB TEST  1-4 fingers · O ok · X hold · Y thumbs up (pause) · P palm · J jump · S squat · ←/→ lean · "
                   "E object · T tracking · B/G mode · U fake user · C skip calib · F1 hide")
            text(s, msg, 20, th.muted, (20, LH - 22), "midleft")
        if self.keys_enabled:
            text(s, f"{self.context} · {self.state.get('mode')} · {self.state.get('user')}",
                 18, th.dim, (LW - 20, LH - 22), "midright")

    # ---- shared pieces ------------------------------------------------
    def frame(self, s, status):
        th = self.theme
        s.fill(th.bg)
        rrect(s, (64, 56, 1792, 968), th.panel, 44, border=th.border, bw=2)
        rrect(s, (112, 92, 54, 54), th.panel2, 14, border=th.border)
        pygame.draw.lines(s, th.accent, False, [(125, 134), (139, 102), (153, 134)], 3)
        pygame.draw.line(s, th.accent2, (124, 124), (156, 120), 3)
        r = text(s, "AYA", 40, th.text, (182, 119), "midleft", bold=True, italic=True)
        text(s, "TLA", 40, th.accent, (r.right, 119), "midleft", bold=True, italic=True)
        if status:
            color = mix(th.accent, WHITE, 0.35)
            r = text(s, status, 22, color, (1808, 119), "midright", bold=True)
            pygame.draw.circle(s, color, (r.left - 16, 119), 5)

    def chips(self, s, hints, x, y, center=False):
        th = self.theme
        widths = []
        for h in hints:
            widths.append(Fonts.get(24, True).size(h["text"])[0] + 44 + 32)
        total = sum(widths) + 16 * (len(hints) - 1)
        if center:
            x = x - total // 2
        for h, w in zip(hints, widths):
            danger = h.get("tone") == "danger"
            col = th.accent2 if danger else mix(th.accent, WHITE, 0.3)
            bg = mix(th.bg, th.accent2 if danger else th.accent, 0.16)
            pill(s, (x, y), h["text"], col, bg, size=24, icon=h.get("icon"))
            x += w + 16

    def option_card(self, s, o, i, x, y, w, h, show_state=True):
        th = self.theme
        focused = (i == self.focus)
        rrect(s, (x, y, w, h), th.panel2 if focused else th.panel, 22,
              border=th.accent if focused else th.border, bw=3 if focused else 2)
        cy = y + h // 2
        rrect(s, (x + 24, cy - 36, 72, 72), mix(th.bg, th.accent, 0.22), 16)
        text(s, str(o.get("finger", i + 1)), 40, th.accent, (x + 60, cy), "center", bold=True)
        n = o.get("finger", i + 1)
        text(s, f"SHOW {n} FINGER{'S' if n > 1 else ''}", 20, th.muted, (x + 120, cy - 30), bold=True)
        text(s, o["label"], 44, th.text, (x + 120, cy + 6), "midleft", bold=True)
        if not show_state:
            return
        if focused:
            text(s, "PENDING", 22, th.accent, (x + w - 32, cy), "midright", bold=True)
        else:
            draw_icon(s, "arrow", x + w - 56, cy, 30, th.accent)

    # ---- CALIBRATION --------------------------------------------------
    def draw_calibration(self, s):
        th, c, cal = self.theme, self.content, self.calib
        self.frame(s, c["status"])
        rrect(s, (112, 200, 820, 760), th.panel2, 30)
        fr = pygame.Rect(232, 250, 580, 620)
        done = cal["progress"] >= 1.0
        dashed_poly(s, rrect_points(fr, 26), th.accent, 3)
        blocky_figure(s, fr.centerx, 540, 1.7, 0, th.avatar, th.accent2)
        text(s, c["frame_caption"], 22, th.muted, (fr.centerx, 915), "midtop", bold=True)
        ring = (800, 330)
        dot_ring(s, ring, 80, cal["progress"], th.accent, mix(th.panel2, th.bg, .5), n=60, dot=6)
        secs = cal["seconds_left"] if cal["seconds_left"] is not None else 0.0
        text(s, f"{secs:.1f}", 52, th.text, (ring[0], ring[1] - 6), "center", bold=True)
        text(s, "SECONDS", 18, th.muted, (ring[0], ring[1] + 34), "center", bold=True)
        x = 1010
        text(s, c["eyebrow"], 24, th.accent, (x, 330), bold=True)
        for i, ln in enumerate(wrap(c["title"], 58, 800, True)):
            text(s, ln, 58, th.text, (x, 380 + i * 70), bold=True)
        cx, cy = x, 590
        for chip in c["chips"]:
            ok = cal.get(chip["key"])
            lab = chip["ok"] if ok else chip["wait"]
            col = mix(th.accent, WHITE, .3) if ok else th.muted
            r = pill(s, (cx, cy), lab, col, th.panel2 if ok else th.panel, size=22,
                     icon="check" if ok else None)
            cx = r.right + 14
            if cx > 1700:
                cx, cy = x, cy + 60
        text(s, c["subtitle"], 32, th.muted, (x, 700))
        pygame.draw.rect(s, th.panel2, (x, 770, 800, 10), border_radius=5)
        pygame.draw.rect(s, th.accent, (x, 770, int(800 * cal["progress"]), 10), border_radius=5)
        text(s, c["footer"], 20, th.dim, (x, 810), bold=True)

    # ---- MAIN MENU ----------------------------------------------------
    def draw_main_menu(self, s):
        th, c = self.theme, self.content
        self.frame(s, c["status"])
        text(s, c["eyebrow"], 24, th.accent, (112, 268), bold=True)
        for i, ln in enumerate(c["title"].split("\n")):
            text(s, ln, 150, th.text, (104, 296 + i * 140), bold=True, italic=True)
        text(s, c["subtitle"], 32, th.muted, (112, 610))
        heights = [150, 230, 120, 200, 260, 140, 200, 110]
        for i, hh in enumerate(heights):
            hh = int(hh * (1 + 0.06 * math.sin(self.t * 1.3 + i)))
            r = pygame.Rect(112 + i * 92, 940 - hh, 84, hh)
            rrect(s, r, mix(th.bg, th.accent, 0.10 + 0.04 * (i % 2)), 4, border=th.border)
        pygame.draw.line(s, th.accent, (112, 940), (112 + 8 * 92, 940), 2)
        for i, o in enumerate(c["options"]):
            self.option_card(s, o, i, 1000, 300 + i * 140, 800, 122)
        self.chips(s, c["hints"], 1000, 880)

    # ---- GESTURE GUIDE ------------------------------------------------
    def draw_guide(self, s):
        th, c = self.theme, self.content
        self.frame(s, c["status"])
        text(s, c["eyebrow"], 24, th.accent, (112, 190), bold=True)
        text(s, c["title"], 64, th.text, (112, 224), bold=True)
        w, h, g = 405, 250, 24
        for i, card in enumerate(c["cards"]):
            x = 112 + (i % 4) * (w + g)
            y = 340 + (i // 4) * (h + g)
            done = card["id"] in self.guide_done
            rrect(s, (x, y, w, h), th.panel2 if done else th.panel, 24,
                  border=th.accent if done else th.border, bw=3 if done else 2)
            draw_icon(s, card["icon"], x + 62, y + 66, 64, th.accent, 4)
            if done:
                pill(s, (x + w - 20, y + 22), "Completed", mix(th.accent, WHITE, .3), th.panel2,
                     size=20, h=44, anchor="topright", icon="check")
            text(s, card["title"], 34, th.text, (x + 28, y + 140), bold=True)
            text(s, "→ " + card["text"], 26, th.muted, (x + 28, y + 192))
        by = 905
        rrect(s, (112, by, 1692, 96), mix(th.bg, th.accent, .14), 22)
        text(s, c["try_title"], 32, th.text, (144, by + 30), bold=True)
        text(s, c["subtitle"], 24, th.muted, (144, by + 66))
        lab = "Waiting for a gesture…"
        if self.last_gesture and time.time() - self.last_gesture[1] < 3.0:
            lab = self.gesture_text(self.last_gesture[0])
        pill(s, (1772, by + 48), lab, mix(th.accent, WHITE, .3), th.panel2, size=26, h=56,
             anchor="midright", icon="jump" if "Jump" in lab else None)

    # ---- PLAYER MODE --------------------------------------------------
    def draw_player_mode(self, s):
        th, c = self.theme, self.content
        self.frame(s, c["status"])
        text(s, c["eyebrow"], 24, th.accent, (112, 190), bold=True)
        text(s, c["title"], 64, th.text, (112, 224), bold=True)
        bt = c["bt_labels"]
        cards = [(112, 0), (1168, 1)]
        for x, i in cards:
            o = c["options"][i]
            focused = (self.focus == i)
            rect = pygame.Rect(x, 330, 640, 520)
            rrect(s, rect, th.panel2 if focused else th.panel, 30,
                  border=th.accent if focused else th.border, bw=3 if focused else 2)
            draw_icon(s, o["icon"], x + 80, 410, 70, th.accent, 4)
            if focused:
                st = bt["pending"]
                if i == 0 and self.bt_status:
                    st = bt.get(self.bt_status, bt["pending"])
                pill(s, (x + 610, 380), st, mix(th.accent, WHITE, .3), mix(th.bg, th.accent, .25),
                     size=22, h=46, anchor="topright")
            text(s, o["caption"], 22, th.accent, (x + 40, 500), bold=True)
            text(s, o["label"], 52, th.text, (x + 40, 535), bold=True)
            for k, ln in enumerate(wrap(o["text"], 30, 560)):
                text(s, ln, 30, th.muted, (x + 40, 615 + k * 40))
            if o.get("tag"):
                pill(s, (x + 40, 740), o["tag"], th.accent2, mix(th.bg, th.accent2, .18), size=22, h=46)
            if i == 0 and self.bt_status == "not_found":
                for k, ln in enumerate(wrap(bt["retry_hint"], 24, 560)):
                    text(s, ln, 24, th.accent2, (x + 40, 720 + k * 32))
            elif i == 0 and self.bt_status == "scanning":
                text(s, bt["scanning_hint"], 26, th.accent, (x + 40, 730), bold=True)
                a = self.t * 4
                for k in range(3):
                    pygame.draw.circle(s, th.accent, (x + 60 + k * 30, 790), 7 + 3 * math.sin(a + k))
            elif i == 0 and self.bt_status == "found":
                nm = self.profile.get("display_name", "")
                text(s, f"Welcome, {nm}!", 30, th.accent, (x + 40, 730), bold=True)
        lean = -1 if self.focus == 0 else 1
        blocky_figure(s, 960, 640, 1.15, lean * 0.8, th.avatar, th.accent2)
        draw_icon(s, "arrow_left", 820, 770, 34, th.accent)
        draw_icon(s, "arrow", 1100, 770, 34, th.accent)
        text(s, c["center_caption"], 22, th.muted, (960, 830), "midtop", bold=True)
        self.chips(s, c["hints"], 960, 900, center=True)

    # ---- BT REGISTER (hands-free device registration) ------------------
    def draw_bt_register(self, s):
        th, c = self.theme, self.content
        lab, reg = c["labels"], self.reg
        self.frame(s, c["status"])
        text(s, c["eyebrow"], 24, th.accent, (112, 190), bold=True)
        text(s, c["title"], 60, th.text, (112, 224), bold=True)
        # left: instructions with a phone next to a laptop
        rrect(s, (112, 330, 640, 520), th.panel2, 30, border=th.border)
        rrect(s, (222, 420, 220, 140), mix(th.bg, th.accent, .18), 14, border=th.accent, bw=3)
        pygame.draw.line(s, th.accent, (192, 572), (472, 572), 6)
        rrect(s, (520, 430, 80, 140), mix(th.bg, th.accent2, .2), 16, border=th.accent2, bw=3)
        draw_icon(s, "bluetooth", 560, 500, 46, th.accent2, 4)
        for k in range(3):                           # pulsing signal waves
            if (int(self.t * 3) + k) % 3:
                pygame.draw.arc(s, th.accent2, (470 - k * 22, 470 - k * 22, 60 + k * 44, 60 + k * 44),
                                2.4, 3.9, 3)
        for k, ln in enumerate(wrap(c["subtitle"], 28, 560)):
            text(s, ln, 28, th.muted, (152, 640 + k * 40))
        # right: up to 4 devices, strongest signal first
        devices = reg["devices"]
        if reg["result"]:
            r = reg["result"]
            text(s, lab["registered_as"].replace("{name}", r["name"] or r["id"]), 44, th.accent,
                 (840, 380), bold=True)
            if r.get("matched_by") == "mac":
                for k, ln in enumerate(wrap(lab["mac_note"], 26, 940)):
                    text(s, ln, 26, th.accent2, (840, 450 + k * 36))
            text(s, lab["starting"], 30, th.muted, (840, 560))
        elif reg["status"] in ("error", "timeout"):
            msg = lab["timeout"] if reg["status"] == "timeout" else lab["error"].replace("{message}", reg["message"])
            for k, ln in enumerate(wrap(msg, 30, 940)):
                text(s, ln, 30, th.accent2, (840, 380 + k * 42), bold=True)
        elif not devices:
            text(s, lab["searching"], 36, th.muted, (840, 380), bold=True)
            for k in range(3):
                pygame.draw.circle(s, th.accent, (860 + k * 34, 450), 8 + 3 * math.sin(self.t * 4 + k))
        else:
            for i, d in enumerate(devices):
                name = d.get("name") or f"{lab['unnamed']} …{d['address'][-5:]}"
                y = 330 + i * 130
                self.option_card(s, {"label": name[:22], "finger": i + 1}, i, 840, y, 960, 112,
                                 show_state=False)
                bars = max(0, min(4, (d["rssi"] + 95) // 10))       # -95 dBm .. -55 dBm
                for b in range(4):
                    col = th.accent if b < bars else th.border
                    pygame.draw.rect(s, col, (1712 + b * 16, y + 76 - b * 10, 10, 18 + b * 10))
                tags = [lab["nearest"]] if d["address"] == reg["nearest"] else []
                if d.get("paired"):
                    tags.append(lab["paired"])       # paired with this laptop: likely yours
                tag = " · ".join(tags)
                if d.get("registered"):
                    tag = lab["known"].replace("{id}", d["registered"])
                if tag:
                    pill(s, (1690, y + 36), tag, mix(th.accent, WHITE, .3), mix(th.bg, th.accent, .25),
                         size=20, h=40, anchor="topright")
            if reg["status"] == "registering":
                text(s, lab["registering"], 30, th.accent, (840, 860), bold=True)
        self.chips(s, c["hints"], 960, 900, center=True)

    # ---- PLAYING ------------------------------------------------------
    def proj(self, lane_f, z):
        sc = 1.0 / (1.0 + z / Z0)
        return ROAD_CX + (lane_f - 1) * LANE_W * sc, HORIZON + (FOOT_Y - HORIZON) * sc, sc

    def draw_scene(self, s):
        th, r = self.theme, self.run
        s.fill(th.bg)
        sway = (r.lane_x - 1) * -26 if r else 0
        for i, (x, w, h) in enumerate(self.skyline):
            rect = pygame.Rect(x + sway * (0.4 + (i % 3) * 0.3), HORIZON + 40 - h, w, h)
            rrect(s, rect, mix(th.bg, th.accent, 0.07 + 0.03 * (i % 3)), 4, border=mix(th.bg, th.accent, .16))
            for k in range(3):
                pygame.draw.line(s, mix(th.bg, th.accent, .25), (rect.x + 20, rect.y + 30 + k * 34),
                                 (rect.right - 20, rect.y + 30 + k * 34), 3)
        # road
        left = [self.proj(-0.5, z)[:2] for z in (200, 60, 20, 6, 0, -1.4)]
        right = [self.proj(2.5, z)[:2] for z in (200, 60, 20, 6, 0, -1.4)]
        pygame.draw.polygon(s, mix(th.bg, th.accent, .11), left + right[::-1])
        pygame.draw.lines(s, th.accent, False, left, 3)
        pygame.draw.lines(s, th.accent, False, right, 3)
        off = (r.dist % 5.0) if r else 0.0
        for lf in (0.5, 1.5):
            z = -off
            while z < 90:
                a, b = self.proj(lf, max(z, -1.2)), self.proj(lf, z + 2.4)
                pygame.draw.line(s, mix(th.accent, WHITE, .35), a[:2], b[:2], max(2, int(7 * a[2])))
                z += 5.0
        if r is None:
            return
        for o in sorted(r.objs, key=lambda q: -q["z"]):
            self.draw_obj(s, o)
        # player
        px, py, _ = self.proj(r.lane_x, 0)
        foot = py - r.height * 170
        pygame.draw.ellipse(s, mix(th.bg, (0, 0, 0), .5), (px - 60, py - 12, 120, 26))
        pose = "jump" if r.height > 0.02 else "slide" if r.sliding else "run"
        if r.active == "shield":
            ring = pygame.Surface((300, 300), pygame.SRCALPHA)
            pygame.draw.circle(ring, (*th.accent, 55), (150, 150), 130)
            pygame.draw.circle(ring, (*th.accent, 200), (150, 150), 130, 4)
            s.blit(ring, (px - 150, foot - 160))
        runner(s, px, foot, 1.0, pose, r.t if not r.over else 0, th.avatar, th.accent2)

    def draw_obj(self, s, o):
        th = self.theme
        x, y, sc = self.proj(o["lane"], o["z"])
        if sc > 1.6 or o["z"] < -3:
            return
        if o["k"] == "obstacle":
            pink = th.accent2
            if o["type"] == "barrier":
                r = pygame.Rect(0, 0, 230 * sc, 90 * sc)
                r.midbottom = (x, y)
            elif o["type"] == "overhead":
                r = pygame.Rect(0, 0, 260 * sc, 100 * sc)
                r.midbottom = (x, y - 120 * sc)
                for dx in (-0.45, 0.45):
                    pygame.draw.line(s, pink, (x + dx * 260 * sc, r.bottom), (x + dx * 260 * sc, y), max(2, int(5 * sc)))
            else:
                r = pygame.Rect(0, 0, 250 * sc, 230 * sc)
                r.midbottom = (x, y)
            rrect(s, r, (*pink, 70), max(3, int(10 * sc)), border=pink, bw=max(2, int(4 * sc)))
            if o["type"] == "block":
                pygame.draw.line(s, pink, r.topleft, r.bottomright, max(2, int(4 * sc)))
                pygame.draw.line(s, pink, r.topright, r.bottomleft, max(2, int(4 * sc)))
            else:
                for k in range(1, 5):
                    px = r.x + r.w * k / 5
                    pygame.draw.line(s, pink, (px, r.bottom), (px + r.h * .5, r.top), max(2, int(3 * sc)))
        elif o["k"] == "coin":
            rad = max(3, int(24 * sc))
            pygame.draw.circle(s, th.gold, (x, y - 50 * sc), rad)
            pygame.draw.circle(s, mix(th.gold, (0, 0, 0), .35), (x, y - 50 * sc), rad, max(2, int(3 * sc)))
        elif o["k"] == "power":
            rad = max(6, int(40 * sc))
            pygame.draw.circle(s, mix(th.bg, th.accent, .3), (x, y - 70 * sc), rad)
            pygame.draw.circle(s, th.accent, (x, y - 70 * sc), rad, max(2, int(4 * sc)))
            draw_icon(s, "object", x, y - 70 * sc, rad * 1.3, th.accent, max(2, int(3 * sc)))

    def hud_box(self, s, x, y, w, h, label, value, hot=False, vsize=52):
        th = self.theme
        rrect(s, (x, y, w, h), (*mix(th.bg, (0, 0, 0), .4), 235), 22,
              border=th.accent if hot else th.border, bw=3 if hot else 2)
        text(s, label, 20, th.accent if hot else th.muted, (x + 26, y + 22), bold=True)
        text(s, value, vsize, th.text, (x + 26, y + 52), bold=True)

    def draw_playing(self, s):
        th, r, c = self.theme, self.run, self.content
        self.draw_scene(s)
        if r is None:
            return
        L = c["labels"]
        lh = self.active_settings.get("left_handed", False)   # mirror the HUD

        def X(x, w):
            return LW - x - w if lh else x

        self.hud_box(s, X(60, 300), 50, 300, 126, L["score"], f"{r.score:,}")
        self.hud_box(s, X(380, 300), 50, 300, 126, L["distance"], f"{r.dist / 1000:.1f} KM")
        self.hud_box(s, X(700, 260), 50, 260, 126, L["coins"], f"× {r.coins}")
        if r.active:
            pv = f"{r.powerup_label(r.active)} · {max(0, math.ceil(r.active_left)):02d}s"
        elif r.stored:
            pv = f"{r.powerup_label(r.stored)} · {L['ready']}"
        else:
            pv = L["none"]
        self.hud_box(s, X(1460, 400), 50, 400, 126, L["powerup"], pv, hot=bool(r.active or r.stored), vsize=44)
        # gesture toast
        bx = X(60, 560)
        if self.toast:
            rrect(s, (bx, 900, 560, 112), (*mix(th.bg, (0, 0, 0), .4), 235), 22, border=th.border)
            draw_icon(s, "jump", bx + 62, 956, 54, th.accent, 4)
            text(s, self.toast["title"], 34, th.text, (bx + 120, 934), bold=True)
            text(s, self.toast["sub"], 24, th.muted, (bx + 120, 976))
        # Live camera preview: show the actual webcam image, never the game avatar.
        tx = X(1590, 270)
        rrect(s, (tx, 700, 270, 312), (*mix(th.bg, (0, 0, 0), .4), 235), 22, border=th.border)
        text(s, "LIVE CAMERA", 20, th.accent, (tx + 24, 722), bold=True)
        preview_rect = pygame.Rect(tx + 24, 756, 222, 166)
        rrect(s, preview_rect, mix(th.bg, th.accent, .12), 10)
        if self.camera_preview is not None:
            s.blit(self.camera_preview, preview_rect.topleft)
            pygame.draw.rect(s, th.accent, preview_rect, width=2, border_radius=10)
        else:
            text(s, "Waiting for webcam feed…", 16, th.muted,
                 preview_rect.center, "center")
        strong = self.tracking_strong
        text(s, L["strong"] if strong else L["lost"], 22, th.accent if strong else th.accent2,
             (tx + 24, 944), bold=True)
        lg = self.last_gesture[0] if self.last_gesture else "—"
        text(s, f"Last: {lg}", 20, th.muted, (tx + 24, 976))
        if r.grace > 0:
            text(s, L["get_ready"], 96, th.text, (LW // 2, 460), "center", bold=True, italic=True)

    # ---- PAUSED -------------------------------------------------------
    def draw_paused(self, s):
        th, c = self.theme, self.content
        self.draw_scene(s)
        rrect(s, (0, 0, LW, LH), (4, 4, 10, 215), 0)
        rrect(s, (380, 200, 1160, 700), (*mix(th.bg, (0, 0, 0), .5), 245), 36, border=th.border)
        text(s, c["status"], 22, mix(th.accent, WHITE, .35), (1808, 119), "midright", bold=True)
        text(s, c["eyebrow"], 22, th.accent, (430, 250), bold=True)
        text(s, c["title"], 56, th.text, (430, 282), bold=True)
        for i, o in enumerate(c["options"]):
            self.option_card(s, o, i, 430, 380 + i * 110, 640, 98)
        for k, sc in enumerate(c["side_cards"]):
            x = 1100 + k * 214
            rrect(s, (x, 440, 200, 250), th.panel, 24, border=th.border)
            draw_icon(s, sc["icon"], x + 100, 520, 72, th.accent2 if sc["icon"] == "x" else th.accent, 4)
            text(s, sc["title"], 30, th.text, (x + 100, 600), "center", bold=True)
            text(s, sc["text"], 22, th.muted, (x + 100, 645), "center")
        self.chips(s, c["hints"], 960, 800, center=True)

    # ---- GAME OVER ----------------------------------------------------
    def draw_game_over(self, s):
        th, c = self.theme, self.content
        self.frame(s, c["status"])
        res = self.last_result or {"score": 0, "best": None, "new_best": False, "km": 0.0, "guest": True}
        L = c["labels"]
        text(s, c["eyebrow"], 24, th.accent, (LW // 2, 215), "midtop", bold=True)
        text(s, f"{res['score']:,}", 250, th.text, (LW // 2, 390), "center", bold=True, italic=True)
        tag = L["final"] + (" · " + L["new_best"] if res["new_best"] else "")
        text(s, tag, 26, th.accent2, (LW // 2, 520), "midtop", bold=True)
        best = f"{res['best']:,}" if res["best"] is not None else "—"
        text(s, L["best"], 22, th.muted, (790, 590), bold=True)
        text(s, best, 46, th.text, (790, 620), bold=True)
        pygame.draw.line(s, th.border, (960, 590), (960, 670), 2)
        text(s, L["distance"], 22, th.muted, (1020, 590), bold=True)
        text(s, f"{res['km']:.1f} KM", 46, th.text, (1020, 620), bold=True)
        note = L["guest_note"] if res["guest"] else L["saved_note"]
        text(s, note, 22, th.dim, (LW // 2, 700), "midtop")
        for i, card in enumerate(c["cards"]):
            x = 150 + i * 580
            rrect(s, (x, 770, 540, 200), th.panel, 26, border=th.border)
            draw_icon(s, card["icon"], x + 52, 825, 46, th.accent2 if card["icon"] == "x" else th.accent, 3)
            text(s, card["gesture"], 24, th.muted, (x + 92, 825), "midleft")
            text(s, "→ " + card["result"], 44, th.text, (x + 34, 890), bold=True)

    # ---- SETTINGS -----------------------------------------------------
    def draw_settings(self, s):
        th, c = self.theme, self.content
        self.frame(s, c["status"])
        text(s, c["eyebrow"], 24, th.accent, (112, 262), bold=True)
        text(s, c["title"], 84, th.text, (112, 296), bold=True)
        for i, ln in enumerate(wrap(c["subtitle"], 30, 520)):
            text(s, ln, 30, th.muted, (112, 410 + i * 40))
        pts = [(112, 580), (160, 580), (200, 520), (240, 660), (280, 560), (330, 560)]
        pygame.draw.lines(s, th.accent, False, pts, 3)
        for i, ln in enumerate(wrap(c["help"], 24, 520)):
            text(s, ln, 24, th.muted, (112, 710 + i * 32))
        who = self.profile.get("display_name", "Guest")
        lab = f"Saving to {who}" if self.ident != "guest" else "Guest · this session only"
        pill(s, (112, 800), lab, mix(th.accent, WHITE, .3), th.panel2, size=22, h=44)
        if self.dirty:
            pill(s, (112, 856), c["unsaved"], th.accent2, mix(th.bg, th.accent2, .18), size=20, h=40)
        y = 200
        x, w = 700, 1104
        for i, o in enumerate(c["options"]):
            h = 124 if o["kind"] == "level" else 96
            focused = (i == self.focus)
            rrect(s, (x, y, w, h), th.panel2 if focused else th.panel, 22,
                  border=th.accent if focused else th.border, bw=3 if focused else 2)
            v = self.draft.get(o["key"])
            rrect(s, (x + 22, y + 24, 44, 44), mix(th.bg, th.accent, .22), 12)
            text(s, str(o["finger"]), 26, th.accent, (x + 44, y + 46), "center", bold=True)
            text(s, o["label"], 34, th.text, (x + 88, y + 46), "midleft", bold=True)
            if o["kind"] == "toggle":
                on = bool(v)
                text(s, "Enabled" if on else "Disabled", 24, th.muted, (x + w - 150, y + 46), "midright")
                sw = pygame.Rect(x + w - 128, y + 22, 96, 48)
                rrect(s, sw, th.accent if on else mix(th.bg, th.accent, .2), 24)
                pygame.draw.circle(s, WHITE, (sw.right - 24 if on else sw.left + 24, sw.centery), 18)
            else:
                text(s, self.fmt_setting(o, v), 32, mix(th.accent, WHITE, .25), (x + w - 32, y + 46), "midright")
            if o["kind"] == "level":
                frac = (v - o["min"]) / (o["max"] - o["min"])
                tx0, tx1 = x + 28, x + w - 120
                pygame.draw.rect(s, mix(th.bg, th.accent, .15), (tx0, y + 92, tx1 - tx0, 10), border_radius=5)
                pygame.draw.rect(s, th.accent, (tx0, y + 92, int((tx1 - tx0) * frac), 10), border_radius=5)
                pygame.draw.rect(s, WHITE, (tx0 + (tx1 - tx0) * frac - 5, y + 84, 10, 26), border_radius=3)
                text(s, f"{int(frac * 100)}%", 22, th.muted, (x + w - 32, y + 97), "midright")
            y += h + 14
        self.chips(s, c["hints"], x, 862)
        if self.toast:
            rrect(s, (560, 944, 800, 64), th.panel2, 32, border=th.accent)
            text(s, f"{self.toast['title']} · {self.toast['sub']}", 24, th.text, (960, 976), "center", bold=True)

    # ---- QUIT CONFIRM -------------------------------------------------
    def draw_quit(self, s):
        th, c = self.theme, self.content
        self.frame(s, c["status"])
        total = c["params"]["seconds"]
        started = self.quit_last_hold is not None
        rem = max(0.0, self.quit_remaining)
        center = (960, 440)
        dot_ring(s, center, 190, rem / total, th.accent2, mix(th.panel, th.bg, .4), n=72, dot=7)
        draw_icon(s, "x", center[0], center[1] - 80, 54, th.accent2, 4)
        n = max(1, math.ceil(rem))
        text(s, f"{n:02d}", 130, th.text, (center[0], center[1] + 10), "center", bold=True)
        text(s, "SECONDS", 22, th.muted, (center[0], center[1] + 90), "center", bold=True)
        text(s, c["eyebrow"], 24, th.accent2, (960, 680), "midtop", bold=True)
        title = c["title"].replace("{n}", str(n)).replace("{seconds_word}", "second" if n == 1 else "seconds")
        text(s, title, 58, th.text, (960, 724), "midtop", bold=True)
        text(s, c["subtitle"] if started else c["waiting"], 30, th.muted, (960, 806), "midtop")

    # ---- TRACKING LOST ------------------------------------------------
    def draw_tracking_lost(self, s):
        th, c = self.theme, self.content
        self.frame(s, c["status"])
        fr = pygame.Rect(790, 250, 340, 400)
        dashed_poly(s, rrect_points(fr, 26), th.accent2, 3)
        blocky_figure(s, 960, 470, 1.1, 0, mix(th.muted, th.bg, .3), th.accent2)
        sy = fr.y + 40 + int((math.sin(self.t * 2.2) * .5 + .5) * (fr.h - 80))
        pygame.draw.line(s, th.accent2, (fr.x - 24, sy), (fr.right + 24, sy), 4)
        text(s, c["eyebrow"], 24, th.accent2, (960, 700), "midtop", bold=True)
        text(s, c["title"], 68, th.text, (960, 745), "midtop", bold=True)
        text(s, c["subtitle"], 34, th.muted, (960, 835), "midtop")
        pill(s, (960, 910), c["chip"], th.accent2, mix(th.bg, th.accent2, .18), size=24, h=52, anchor="midtop")


def main():
    ap = argparse.ArgumentParser(description="AYATLA game process")
    ap.add_argument("--host", default=HUB_HOST)
    ap.add_argument("--port", type=int, default=HUB_PORT)
    ap.add_argument("--no-hub", action="store_true", help="play alone with keyboard fakes")
    ap.add_argument("--keys", action="store_true", help="enable keyboard fakes while connected to a hub")
    args = ap.parse_args()
    Game(args).run_loop()


if __name__ == "__main__":
    main()
