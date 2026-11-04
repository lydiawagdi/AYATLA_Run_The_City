#!/usr/bin/env python3
"""
Webcam vision module for the game.

Hub contract (AYATLA shared contract, host/port from common.py):
  * TCP connection to 127.0.0.1:5000, newline-delimited UTF-8 JSON.
  * First message after connecting: {"type":"hello","role":"vision"}.
  * Incoming messages are {"type":"state", ...} or {"type":"settings", ...}.
    State contains "context" and a nested "settings" object.
    Settings may contain camera_index, gesture_sensitivity, hold_time,
    left_handed.
  * Outgoing messages are newline-delimited JSON objects:
      {"type":"gesture","name":"jump","score":0.82,"repeat":false}
      {"type":"calibration","camera":true,"body":true,"hands":false,
       "progress":0.5,"seconds_left":1.5,"done":false}
      {"type":"tracking","status":"lost"}
  The exact message envelope/field names are isolated in HubClient.send().
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from dollarpy import Point, Recognizer, Template

from common import HOST, PORT

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
BODY_GESTURES = ("jump",)   # DollarPy body movement (lanes: ZoneTracker, squat: SquatLine)
POSE_LANDMARKS = mp.solutions.pose.PoseLandmark


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sensitivity_to_threshold(sensitivity: float) -> float:
    """Tune this mapping if your recorded templates score too low/high.

    DollarPy's score is 0..1. Higher sensitivity intentionally lowers the
    acceptance threshold: 0 -> 0.78 (strict), 1 -> 0.45 (permissive).
    """
    s = clamp(float(sensitivity), 0.0, 1.0)
    return 0.78 - 0.33 * s


def sensitivity_to_stable_frames(sensitivity: float) -> int:
    """Tune the hand debounce mapping: sensitive -> fewer frames."""
    s = clamp(float(sensitivity), 0.0, 1.0)
    return int(round(12 - 7 * s))  # 12 frames at 0; 5 frames at 1


class SharedState:
    """Thread-safe latest state/settings snapshot."""

    def __init__(self, camera_override: Optional[int] = None):
        self.lock = threading.RLock()
        self.state = {"context": ""}
        self.settings = {
            "camera_index": 0,
            "gesture_sensitivity": 0.5,
            "hold_time": 1.0,
            "left_handed": False,
            "hub_host": os.getenv("VISION_HUB_HOST", HOST),
            "hub_port": int(os.getenv("VISION_HUB_PORT", str(PORT))),
        }
        if camera_override is not None:
            self.settings["camera_index"] = camera_override
            self.camera_override = camera_override
        else:
            self.camera_override = None
        self.version = 0

    def update(self, message: dict) -> None:
        kind = message.get("type")
        with self.lock:
            if kind == "state":
                # Accept either a nested state object or top-level fields.
                incoming = message.get("state", message)
                if isinstance(incoming, dict):
                    self.state.update(incoming)
                    # The hub's state carries the current settings too.
                    if isinstance(incoming.get("settings"), dict):
                        self.settings.update(incoming["settings"])
                        if self.camera_override is not None:
                            self.settings["camera_index"] = self.camera_override
            elif kind == "settings":
                incoming = message.get("settings", message)
                if isinstance(incoming, dict):
                    self.settings.update(incoming)
                    if self.camera_override is not None:
                        self.settings["camera_index"] = self.camera_override
            self.version += 1

    def snapshot(self) -> Tuple[dict, dict, int]:
        with self.lock:
            return dict(self.state), dict(self.settings), self.version


class HubClient:
    """TCP JSON-lines hub client with a dedicated receiver thread.

    The socket is optional in --no-hub mode. Reconnection is automatic.
    """

    def __init__(self, shared: SharedState, enabled: bool = True):
        self.shared = shared
        self.enabled = enabled
        self.stop_event = threading.Event()
        self.sock: Optional[socket.socket] = None
        self.send_lock = threading.Lock()
        self.connect_lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.enabled:
            self.thread = threading.Thread(target=self._receiver, daemon=True, name="hub-receiver")
            self.thread.start()

    def _connect(self) -> Optional[socket.socket]:
        _, settings, _ = self.shared.snapshot()
        host = str(settings.get("hub_host", HOST))
        port = int(settings.get("hub_port", PORT))
        try:
            s = socket.create_connection((host, port), timeout=2.0)
            # The hub drops any client whose first message is not hello.
            s.sendall((json.dumps({"type": "hello", "role": "vision"}) + "\n").encode("utf-8"))
            s.settimeout(1.0)
            with self.connect_lock:
                self.sock = s
            print(f"[vision] connected to hub at {host}:{port}")
            return s
        except OSError as exc:
            print(f"[vision] hub unavailable ({exc}); retrying")
            return None

    def _receiver(self) -> None:
        buffer = b""
        while not self.stop_event.is_set():
            sock = self.sock
            if sock is None:
                sock = self._connect()
                if sock is None:
                    self.stop_event.wait(2.0)
                    continue
                buffer = b""
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("hub closed connection")
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    if not raw.strip():
                        continue
                    try:
                        msg = json.loads(raw.decode("utf-8"))
                        if isinstance(msg, dict):
                            self.shared.update(msg)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        print(f"[vision] ignored malformed hub message: {exc}")
            except socket.timeout:
                continue
            except OSError as exc:
                print(f"[vision] hub receive error: {exc}")
                self._close_socket()
                self.stop_event.wait(2.0)

    def send(self, payload: dict) -> None:
        """Change this method only if the shared project uses another envelope."""
        if payload.get("type") == "gesture" and "gesture" in payload:
            # Shared contract: the gesture is called "name" on the wire.
            payload = {"type": "gesture", "name": payload["gesture"],
                       "score": payload.get("score", 1.0), "repeat": payload.get("repeat", False)}
        line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        if not self.enabled:
            print("[vision -> hub]", line.decode("utf-8").rstrip())
            return
        with self.send_lock:
            sock = self.sock
            if sock is None:
                print("[vision -> hub queued/dropped: disconnected]", payload)
                return
            try:
                sock.sendall(line)
            except OSError as exc:
                print(f"[vision] hub send failed: {exc}")
                self._close_socket()

    def _close_socket(self) -> None:
        with self.connect_lock:
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None

    def close(self) -> None:
        self.stop_event.set()
        self._close_socket()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)


def landmark_xy(landmarks, index: int) -> Optional[Tuple[float, float]]:
    try:
        lm = landmarks[index]
        if getattr(lm, "visibility", 1.0) < 0.35:
            return None
        return float(lm.x), float(lm.y)
    except (IndexError, TypeError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Body movements - UPPER BODY ONLY (head, shoulders, wrists; legs not needed)
# ---------------------------------------------------------------------------
# Stored in every template file. Templates recorded with another feature
# layout (e.g. the old hip-based one) are skipped because they cannot match.
COORDINATE_SYSTEM = ("upper-body v2: nose, shoulder midpoint and wrists relative to the "
                     "neutral shoulder midpoint, in neutral shoulder widths (y down)")
STROKES = ("nose", "shoulder", "left_wrist", "right_wrist")   # DollarPy stroke ids 1..4
DOLLAR_OFFSET = 10.0    # dollarpy's scaling assumes coordinates >= 0, so shift them up
DOLLAR_N = 24           # resampled points; the library default (32) is ~2x slower

# Tune: lower = more sensitive, higher = fewer accidental gestures.
MIN_MOTION = 0.25       # head + shoulders must move at least this many shoulder widths
ARMS_UP_MIN = 0.4       # both wrists this far above the nose = "arms up" (counts as jump)
NEUTRAL_RADIUS = 0.2    # a movement must START this close to your neutral position
STILL_SPEED = 0.8       # slower than this (shoulder widths / s) = standing still
BASELINE_SECONDS = 2.0  # how quickly the neutral position follows you while still
BODY_COOLDOWN = 0.8     # the same movement is not sent twice within this time

# Only the upward movement (jump) is a DollarPy gesture. Lanes are decided by
# WHERE the player stands (ZoneTracker) and squats by HEAD HEIGHT (SquatLine).
DIRECTION_TO_GESTURE = {"up": "jump"}
GESTURE_TO_DIRECTION = {g: d for d, g in DIRECTION_TO_GESTURE.items()}

# The squat / jump line at SQUAT_LINE of the picture height:
#   * squat = the HEAD drops below the line (sent once; the head must come back
#     above SQUAT_LINE - SQUAT_REARM_MARGIN before the next squat counts)
#   * jump  = BOTH HANDS go above the line (sent once; a hand must come back
#     below SQUAT_LINE + SQUAT_REARM_MARGIN first). Both hands, because one
#     raised hand is a hand sign (OK, fingers, thumbs up).
SQUAT_LINE = 0.30              # Tune: fraction of the picture height from the top
SQUAT_STABLE_SECONDS = 0.1     # head / hands must stay past the line this long
SQUAT_REARM_MARGIN = 0.03      # stops a head or hand sitting on the line from re-firing
JUMP_COOLDOWN = 0.8            # one jump per movement (line or DollarPy, whichever is first)


def _blend(a, b, amount):
    return (a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount)


class UpperBodyFeatures:
    """Turns pose landmarks into the upper-body features used by DollarPy.

    Positions are measured from the player's NEUTRAL shoulder midpoint
    (learned while they stand still) and divided by their neutral shoulder
    width, so camera distance does not matter and whole-body moves (jump,
    crouch, lean or side step) stay visible. Only the head, shoulders and
    wrists are used, so hips and legs may be outside the frame.
    x is multiplied by the frame aspect ratio so x and y use the same units.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.baseline = None    # {"nose": (x, y), "shoulder": (x, y), "width": w}
        self.previous = None    # (time, nose, shoulder) of the previous frame
        self.speed = 0.0        # smoothed head + shoulder speed, shoulder widths / s

    def update(self, landmarks, now: float, aspect: float) -> Optional[dict]:
        if landmarks is None:
            self.reset()
            return None

        def xy(index, min_visibility=0.35):
            p = landmark_xy(landmarks, index)
            if p is None or getattr(landmarks[index], "visibility", 1.0) < min_visibility:
                return None
            return (p[0] * aspect, p[1])

        nose = xy(POSE_LANDMARKS.NOSE.value, 0.5)
        ls = xy(POSE_LANDMARKS.LEFT_SHOULDER.value, 0.5)
        rs = xy(POSE_LANDMARKS.RIGHT_SHOULDER.value, 0.5)
        if nose is None or ls is None or rs is None:
            self.reset()
            return None
        shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        width = math.dist(ls, rs)
        if width < 0.03:
            self.reset()
            return None

        dt = 0.0
        if self.previous is not None:
            dt = now - self.previous[0]
            if dt > 0:
                moved = (math.dist(nose, self.previous[1]) + math.dist(shoulder, self.previous[2])) / 2 / width
                self.speed += (moved / dt - self.speed) * min(1.0, dt / 0.2)
        self.previous = (now, nose, shoulder)

        if self.baseline is None:
            self.baseline = {"nose": nose, "shoulder": shoulder, "width": width}
        elif self.speed < STILL_SPEED and dt > 0 and nose[1] <= SQUAT_LINE:
            # (not while squatting: standing up again must not look like a jump)
            a = min(1.0, dt / BASELINE_SECONDS)
            self.baseline["nose"] = _blend(self.baseline["nose"], nose, a)
            self.baseline["shoulder"] = _blend(self.baseline["shoulder"], shoulder, a)
            self.baseline["width"] += (width - self.baseline["width"]) * a
        b = self.baseline

        def norm(p):
            if p is None:
                return None
            return ((p[0] - b["shoulder"][0]) / b["width"], (p[1] - b["shoulder"][1]) / b["width"])

        features = {
            "t": now,
            "nose": norm(nose),
            "shoulder": norm(shoulder),
            "left_wrist": norm(xy(POSE_LANDMARKS.LEFT_WRIST.value)),
            "right_wrist": norm(xy(POSE_LANDMARKS.RIGHT_WRIST.value)),
        }
        neutral_nose = norm(b["nose"])
        # "torso" = how far head and shoulders moved from their neutral spot.
        features["torso"] = ((features["nose"][0] - neutral_nose[0] + features["shoulder"][0]) / 2,
                             (features["nose"][1] - neutral_nose[1] + features["shoulder"][1]) / 2)
        lw, rw = features["left_wrist"], features["right_wrist"]
        # How far BOTH wrists are above the nose (positive = arms up).
        features["arms_up"] = (min(features["nose"][1] - lw[1], features["nose"][1] - rw[1])
                               if lw is not None and rw is not None else None)
        return features


def movement_direction(frames: Sequence[dict]) -> Tuple[str, float]:
    """Main vertical direction of the head + shoulder movement and its size
    (shoulder widths). Raising both arms above the head also counts as "up"
    (jump). Sideways movement is ignored: stepping to another lane must not
    look like a jump or a squat."""
    dys = [f["torso"][1] for f in frames]
    moves = {"up": -min(dys), "down": max(dys)}
    arms = [f["arms_up"] for f in frames if f["arms_up"] is not None]
    if arms and max(arms) >= ARMS_UP_MIN:
        return "up", max(moves["up"], MIN_MOTION)
    direction = max(moves, key=moves.get)
    return direction, moves[direction]


def starts_neutral(frames: Sequence[dict]) -> bool:
    """True if the first 0.15 s of the window are close to the neutral height
    (only height matters: the player may stand in any lane)."""
    t0 = frames[0]["t"]
    for f in frames:
        if f["t"] - t0 > 0.15:
            break
        if abs(f["torso"][1]) > NEUTRAL_RADIUS:
            return False
        if f["arms_up"] is not None and f["arms_up"] >= ARMS_UP_MIN:
            return False
    return True


# ---------------------------------------------------------------------------
# Lanes: the picture is split into thirds. The third that holds most of the
# player's visible landmarks is their zone (mirrored view: left third = the
# player's own left). Sent as gesture zone_left / zone_center / zone_right
# when it changes (repeat:false) and every ZONE_REPEAT_SECONDS while it stays
# the same (repeat:true) so the game's lane can never get out of sync.
# ---------------------------------------------------------------------------
ZONE_NAMES = ("zone_left", "zone_center", "zone_right")
ZONE_MAJORITY = 0.5        # Tune: share of the body that must be inside a third
ZONE_STABLE_SECONDS = 0.15 # Tune: a new zone must hold this long (stops flicker on a line)
ZONE_REPEAT_SECONDS = 1.0
ZONE_FORGET_SECONDS = 1.0  # body missing this long -> zone unknown


class ZoneTracker:
    def __init__(self):
        self.zone: Optional[int] = None     # 0 left, 1 center, 2 right
        self.candidate: Optional[int] = None
        self.since = 0.0
        self.last_send = 0.0
        self.last_seen = 0.0
        self.shares = (0.0, 0.0, 0.0)       # share of the body in each third (debug)

    def update(self, landmarks, now: float) -> Optional[Tuple[str, bool]]:
        """Return (zone gesture name, repeat) when something should be sent."""
        xs = [lm.x for lm in landmarks if getattr(lm, "visibility", 1.0) >= 0.5] if landmarks else []
        if len(xs) < 3:
            if self.zone is not None and now - self.last_seen > ZONE_FORGET_SECONDS:
                self.zone = self.candidate = None
            return None
        self.last_seen = now
        counts = [0, 0, 0]
        for x in xs:
            counts[0 if x < 1 / 3 else (1 if x < 2 / 3 else 2)] += 1
        self.shares = tuple(c / len(xs) for c in counts)
        best = max(range(3), key=lambda i: self.shares[i])
        wanted = best if self.shares[best] > ZONE_MAJORITY else self.zone   # no majority: stay
        if wanted is None:
            return None
        if wanted != self.zone:
            if wanted != self.candidate:
                self.candidate, self.since = wanted, now
            elif now - self.since >= ZONE_STABLE_SECONDS:
                self.zone, self.last_send = wanted, now
                return ZONE_NAMES[wanted], False
            return None
        self.candidate = wanted
        if now - self.last_send >= ZONE_REPEAT_SECONDS:
            self.last_send = now
            return ZONE_NAMES[wanted], True
        return None


def head_height(landmarks) -> Optional[float]:
    """Head position as a fraction of the picture height (0 = top), or None."""
    if landmarks is None:
        return None
    nose = landmarks[POSE_LANDMARKS.NOSE.value]
    if getattr(nose, "visibility", 1.0) >= 0.5:
        return float(nose.y)
    face = [landmarks[i].y for i in range(1, 11) if getattr(landmarks[i], "visibility", 1.0) >= 0.5]
    return sum(face) / len(face) if face else None


class SquatLine:
    """Squat = the head drops below SQUAT_LINE (see the constants above)."""

    def __init__(self):
        self.armed = False          # True once the head has been above the line
        self.below_since = None
        self.head = None            # last head height (debug)

    def update(self, landmarks, now: float) -> bool:
        """True exactly once per squat."""
        self.head = head_height(landmarks)
        if self.head is None:
            self.armed, self.below_since = False, None   # nobody: do not fire on return
            return False
        if self.head < SQUAT_LINE - SQUAT_REARM_MARGIN:
            self.armed, self.below_since = True, None
            return False
        if self.armed and self.head > SQUAT_LINE:
            if self.below_since is None:
                self.below_since = now
            if now - self.below_since >= SQUAT_STABLE_SECONDS:
                self.armed, self.below_since = False, None
                return True
        else:
            self.below_since = None
        return False

    def status(self) -> str:
        if self.head is None:
            return "no head"
        state = "SQUAT" if self.head > SQUAT_LINE else "standing"
        ready = "" if self.armed or self.head > SQUAT_LINE else " (raise your head above the line)"
        return f"head at {self.head * 100:.0f}% (line {SQUAT_LINE * 100:.0f}%): {state}{ready}"


def hand_height(landmarks, ids) -> Optional[float]:
    """Highest point of one hand (pose wrist + hand points), fraction of the
    picture height (can be < 0 above the picture), or None if not visible."""
    ys = [landmarks[i].y for i in ids if getattr(landmarks[i], "visibility", 1.0) >= 0.2]
    return min(ys) if ys else None


class HandsUpLine:
    """Jump = BOTH hands above SQUAT_LINE (see the constants above)."""
    LEFT = (15, 17, 19, 21)       # MediaPipe pose: wrist, pinky, index, thumb
    RIGHT = (16, 18, 20, 22)

    def __init__(self):
        self.armed = False          # True once a hand has been below the line
        self.above_since = None
        self.hands = (None, None)   # last hand heights (debug)

    def update(self, landmarks, now: float) -> bool:
        """True exactly once per "both hands up"."""
        if landmarks is None:
            self.armed, self.above_since, self.hands = False, None, (None, None)
            return False
        left, right = hand_height(landmarks, self.LEFT), hand_height(landmarks, self.RIGHT)
        self.hands = (left, right)
        # A hand that is not visible counts as down (e.g. hanging below the picture).
        if any(h is None or h > SQUAT_LINE + SQUAT_REARM_MARGIN for h in (left, right)):
            self.armed, self.above_since = True, None
            return False
        if self.armed and left < SQUAT_LINE and right < SQUAT_LINE:
            if self.above_since is None:
                self.above_since = now
            if now - self.above_since >= SQUAT_STABLE_SECONDS:
                self.armed, self.above_since = False, None
                return True
        else:
            self.above_since = None
        return False

    def status(self) -> str:
        def show(h):
            return "-" if h is None else f"{h * 100:.0f}%"
        up = all(h is not None and h < SQUAT_LINE for h in self.hands)
        return f"hands at {show(self.hands[0])} / {show(self.hands[1])} (line {SQUAT_LINE * 100:.0f}%): " \
               f"{'JUMP (both up)' if up else 'down'}"


def features_to_rows(frames: Sequence[dict]) -> List[List[float]]:
    """Frames -> [[x, y, stroke_id], ...], one stroke per landmark (the format
    stored in templates/*.json). Frames where a wrist is not visible are skipped."""
    rows = []
    for stroke_id, name in enumerate(STROKES, start=1):
        for f in frames:
            xy = f.get(name)
            if xy is not None:
                rows.append([round(float(xy[0]), 4), round(float(xy[1]), 4), stroke_id])
    return rows


def rows_to_points(rows) -> List[Point]:
    return [Point(float(x) + DOLLAR_OFFSET, float(y) + DOLLAR_OFFSET, int(sid)) for x, y, sid in rows]


def load_templates(folder: Path) -> List[Template]:
    """Load {gesture}.json; each file contains a list of point-cloud samples."""
    templates: List[Template] = []
    for gesture in BODY_GESTURES:
        path = folder / f"{gesture}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("coordinate_system") != COORDINATE_SYSTEM:
                print(f"[vision] skipping {path.name}: recorded with an older feature layout "
                      "(re-record with record_templates.py)")
                continue
            for sample in data.get("samples", []):
                pts = rows_to_points(sample)
                if len(pts) >= 4:
                    templates.append(Template(gesture, pts))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"[vision] could not load {path}: {exc}")
    print(f"[vision] loaded {len(templates)} body templates")
    return templates


def recognize(points: List[Point], templates: List[Template]) -> Tuple[Optional[str], float]:
    """DollarPy best match. dollarpy's recognize() inserts resampled points into
    the template lists it gets, so it is given fresh copies every time."""
    if not templates or len(points) < 4:
        return None, 0.0
    fresh = [Template(t.name, list(t)) for t in templates]
    try:
        name, score = Recognizer(fresh).recognize(list(points), DOLLAR_N)
    except (ZeroDivisionError, IndexError, ValueError):
        return None, 0.0          # e.g. no movement at all (zero path length)
    return name, float(score)


class BodyGestureDetector:
    """Rolling 1-second upper-body classifier: a cheap movement check first,
    then DollarPy (so standing still or moving one arm never fires)."""

    def __init__(self, template_dir: Path):
        self.templates = load_templates(template_dir)
        self.history: Deque[dict] = deque()
        self.last_classify = 0.0
        self.last_sent = defaultdict(lambda: -1e9)
        self.last_name = "none"
        self.last_score = 0.0
        self.status = "waiting"

    def update(self, features: Optional[dict], now: float, threshold: float) -> Optional[Tuple[str, float]]:
        if features is None:
            self.history.clear()
            return None
        self.history.append(features)
        while self.history and now - self.history[0]["t"] > 1.0:
            self.history.popleft()

        if now - self.last_classify < 0.15:
            return None
        self.last_classify = now
        if not self.templates:
            self.status = "no templates - run record_templates.py"
            return None
        frames = list(self.history)
        if len(frames) < 8:
            return None

        direction, amount = movement_direction(frames)
        if amount < MIN_MOTION:
            self.status = f"still (moved {amount:.2f})"
            return None
        if direction not in DIRECTION_TO_GESTURE:
            self.status = f"moved {direction} (squats use the head line)"
            return None
        if not starts_neutral(frames):
            self.status = "return to your neutral position"
            return None

        points = rows_to_points(features_to_rows(frames))
        expected = DIRECTION_TO_GESTURE[direction]
        name, score = recognize(points, self.templates)
        if name != expected:
            # DollarPy compares point clouds and ignores time order, so a jump
            # and a squat can look alike. The measured direction decides; the
            # movement must still match THAT gesture's templates well enough.
            name = expected
            score = recognize(points, [t for t in self.templates if t.name == expected])[1]
        self.last_name, self.last_score = name, score
        self.status = f"{name} {score:.2f} (need {threshold:.2f})"
        if score < threshold:
            return None
        if now - self.last_sent[name] < BODY_COOLDOWN:
            return None
        self.last_sent[name] = now
        self.history.clear()        # one movement = one gesture
        return name, score


def point_distance(a, b) -> float:
    return math.dist((a.x, a.y), (b.x, b.y))


def hand_pose_name(hand_landmarks) -> Optional[str]:
    """Rule-based sign classifier; landmarks are normalized 2D image points."""
    lm = hand_landmarks.landmark
    wrist = lm[0]
    palm_center = ((lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5,
                   (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5)
    palm_width = max(math.dist((lm[5].x, lm[5].y), (lm[17].x, lm[17].y)), 0.035)

    # For non-thumb fingers, a tip above (smaller y) its PIP means extended.
    # This image-y heuristic works best with palm facing camera; tune the
    # margin if signs are missed due to camera angle.
    extended = {}
    for finger, tip, pip in (
        ("index", 8, 6), ("middle", 12, 10), ("ring", 16, 14), ("pinky", 20, 18)
    ):
        extended[finger] = lm[tip].y < lm[pip].y - 0.015

    # Thumb is handled separately: lateral distance from palm center relative
    # to palm width. This is more stable than tip-vs-joint vertical position.
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_lateral = abs(thumb_tip.x - palm_center[0]) / palm_width
    thumb_extended = thumb_lateral > 0.48 and point_distance(thumb_tip, thumb_ip) / palm_width > 0.16

    # OK: thumb/index tips touch, with middle/ring/pinky extended.
    tip_gap = point_distance(thumb_tip, lm[8]) / palm_width
    other_three = extended["middle"] and extended["ring"] and extended["pinky"]
    if tip_gap < 0.32 and other_three:
        return "ok_sign"

    # Thumbs up: the four fingers are curled (each tip is not farther from the
    # wrist than its PIP joint) and the thumb tip is clearly the highest point
    # of the hand, with the thumb joints stacked upwards (image y grows down).
    def from_wrist(i):
        return math.dist((lm[i].x, lm[i].y), (wrist.x, wrist.y))
    curled = all(from_wrist(tip) <= from_wrist(pip) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
    highest_other = min(lm[i].y for i in range(5, 21))
    thumb_up = thumb_tip.y < thumb_ip.y < lm[2].y and highest_other - thumb_tip.y > 0.5 * palm_width
    if curled and thumb_up:
        return "thumbs_up"

    non_thumb_count = sum(extended.values())
    # Require an extended, spread hand for open_palm. This keeps four fingers
    # plus thumb distinct from fingers_4 (which requires folded thumb).
    spread = (
        math.dist((lm[8].x, lm[8].y), (lm[20].x, lm[20].y)) / palm_width > 1.45
        and math.dist((lm[12].x, lm[12].y), (lm[16].x, lm[16].y)) / palm_width > 0.12
    )
    if non_thumb_count == 4 and thumb_extended and spread:
        return "open_palm"
    if non_thumb_count in (1, 2, 3, 4) and not thumb_extended:
        return f"fingers_{non_thumb_count}"
    return None


# ---------------------------------------------------------------------------
# Recorded samples for hand signs, hold X and the power-up object colour
# (made with the recorder app, record_templates.py). The rules still work
# without any samples; recorded samples teach vision YOUR way of doing a sign.
# ---------------------------------------------------------------------------
HAND_SIGNS = ("open_palm", "thumbs_up", "ok_sign", "fingers_1", "fingers_2", "fingers_3", "fingers_4")
# Signs usually shown with the hand sideways (fingers not pointing up).
SIDEWAYS_SIGNS = ("thumbs_up",)
HAND_COORDINATES = ("hand v1: 21 landmarks, wrist at origin, wrist->middle knuckle pointing up, "
                    "in palm lengths, left hands mirrored")
XPOSE_COORDINATES = ("x-pose v1: nose, shoulders, elbows, wrists relative to the shoulder "
                     "midpoint, in shoulder widths")
OBJECT_COLOR_FILE = "object_power_up.json"
# Tune: how close (average landmark distance) a live hand / pose must be to a
# recorded sample to count. The recorder's live test shows these distances.
HAND_MATCH_DIST = 0.22    # palm lengths
XPOSE_MATCH_DIST = 0.30   # shoulder widths


def hand_vector(hand_landmarks, label: str, aspect: float,
                require_raised: bool = True) -> Optional[np.ndarray]:
    """21 hand landmarks -> 42 numbers: wrist at the origin, rotated so the
    wrist->middle knuckle direction points up, in palm lengths. Left hands are
    mirrored so both hands can be compared. None if the hand is not raised
    (unless require_raised is False, e.g. for a sideways thumbs up)."""
    pts = np.array([(lm.x * aspect, lm.y) for lm in hand_landmarks.landmark])
    v = pts[9] - pts[0]
    length = float(np.hypot(v[0], v[1]))
    if length < 1e-6:
        return None
    ux, uy = v / length
    if require_raised and uy > -0.3:
        return None                     # fingers not pointing up: hand is down
    d = (pts - pts[0]) / length
    x = -uy * d[:, 0] + ux * d[:, 1]
    y = -ux * d[:, 0] - uy * d[:, 1]
    if label == "Left":
        x = -x
    return np.column_stack([x, y]).ravel()


def xpose_vector(landmarks, aspect: float) -> Optional[np.ndarray]:
    """Nose, shoulders, elbows and wrists relative to the shoulder midpoint,
    in shoulder widths (14 numbers). None if an arm is not visible."""
    ids = POSE_LANDMARKS
    pts = []
    for lm_id in (ids.NOSE, ids.LEFT_SHOULDER, ids.RIGHT_SHOULDER, ids.LEFT_ELBOW,
                  ids.RIGHT_ELBOW, ids.LEFT_WRIST, ids.RIGHT_WRIST):
        lm = landmarks[lm_id.value]
        if getattr(lm, "visibility", 1.0) < 0.35:
            return None
        pts.append((lm.x * aspect, lm.y))
    pts = np.array(pts)
    middle = (pts[1] + pts[2]) / 2
    width = float(np.hypot(*(pts[1] - pts[2])))
    if width < 1e-6:
        return None
    return ((pts - middle) / width).ravel()


class SampleMatcher:
    """Nearest recorded example; distance = average landmark distance (RMS)."""

    def __init__(self, names: List[str], vectors: List[List[float]]):
        self.names = names
        self.vectors = np.array(vectors, dtype=float)
        self.points = self.vectors.shape[1] // 2

    def nearest(self, vector: np.ndarray, only: Optional[Sequence[str]] = None) -> Tuple[Optional[str], Optional[float]]:
        """Closest example (optionally only among the gestures in `only`)."""
        d = np.sqrt(((self.vectors - vector) ** 2).sum(axis=1) / self.points)
        if only is not None:
            d = np.where(np.isin(self.names, list(only)), d, np.inf)
        i = int(np.argmin(d))
        if not np.isfinite(d[i]):
            return None, None
        return self.names[i], float(d[i])


def load_samples(folder: Path, gestures: Sequence[str], coordinate_system: str) -> Optional[SampleMatcher]:
    """Recorded frames for these gestures, or None if there are none yet."""
    names, vectors = [], []
    for gesture in gestures:
        path = Path(folder) / f"{gesture}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"[vision] could not load {path}: {exc}")
            continue
        if data.get("coordinate_system") != coordinate_system:
            print(f"[vision] skipping {path.name}: different sample format")
            continue
        for sample in data.get("samples", []):
            for vec in sample.get("frames", []):
                names.append(gesture)
                vectors.append(vec)
    size = {len(v) for v in vectors}
    if not vectors or len(size) != 1:
        return None
    print(f"[vision] loaded {len(vectors)} recorded frames for {sorted(set(names))}")
    return SampleMatcher(names, vectors)


def choose_hand(hand_results, left_handed: bool):
    """(landmarks, label) of the preferred hand (right, or left in left-handed
    mode), else of the first hand; (None, None) if no hand is visible."""
    if not hand_results or not hand_results.multi_hand_landmarks:
        return None, None
    target = "Left" if left_handed else "Right"
    # Frames are mirrored before MediaPipe. Handedness labels therefore
    # follow the mirrored selfie-view convention (not raw camera pixels).
    labels = [h.classification[0].label for h in (hand_results.multi_handedness or [])]
    for idx, label in enumerate(labels):
        if label == target:
            return hand_results.multi_hand_landmarks[idx], label
    return hand_results.multi_hand_landmarks[0], (labels[0] if labels else target)


def classify_hand_sign(hand_landmarks, label: str, matcher: Optional[SampleMatcher], aspect: float):
    """Return (final sign, rules sign, nearest recorded sign, its distance).
    A recorded sample that matches closely wins; otherwise the rules decide."""
    rule = hand_pose_name(hand_landmarks)
    sample, dist = None, None
    if matcher is not None:
        vec = hand_vector(hand_landmarks, label, aspect)
        if vec is not None:
            sample, dist = matcher.nearest(vec)
        else:
            # Hand not raised: only signs that are shown sideways can match.
            vec = hand_vector(hand_landmarks, label, aspect, require_raised=False)
            if vec is not None:
                sample, dist = matcher.nearest(vec, only=SIDEWAYS_SIGNS)
    final = sample if sample is not None and dist <= HAND_MATCH_DIST else rule
    return final, rule, sample, dist


class HandGestureDetector:
    def __init__(self, matcher: Optional[SampleMatcher] = None):
        self.matcher = matcher
        self.candidate = None
        self.count = 0
        self.last_sent_name = None
        self.last_sent_time = 0.0

    def update(self, hand_results, left_handed: bool, stable_frames: int, now: float,
               aspect: float = 16 / 9):
        chosen, label = choose_hand(hand_results, left_handed)
        if chosen is None:
            self.candidate, self.count = None, 0
            return None
        name = classify_hand_sign(chosen, label, self.matcher, aspect)[0]
        if name is None:
            self.candidate, self.count = None, 0
            return None
        if name == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = name, 1
        if self.count < stable_frames:
            return None
        # Re-send only after sign changes or a one-second heartbeat.
        if name != self.last_sent_name or now - self.last_sent_time >= 1.0:
            self.last_sent_name, self.last_sent_time = name, now
            return name
        return None


def x_pose_by_rules(landmarks) -> bool:
    """Forearms crossed in front of the chest (rule-based)."""
    ids = POSE_LANDMARKS
    lw = landmark_xy(landmarks, ids.LEFT_WRIST.value)
    rw = landmark_xy(landmarks, ids.RIGHT_WRIST.value)
    ls = landmark_xy(landmarks, ids.LEFT_SHOULDER.value)
    rs = landmark_xy(landmarks, ids.RIGHT_SHOULDER.value)
    nose = landmark_xy(landmarks, ids.NOSE.value)
    if any(p is None for p in (lw, rw, ls, rs, nose)):
        return False
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    # Chest point without hips: below the shoulders by 80% of the
    # nose-to-shoulder height, so the legs/hips may be out of frame.
    chest = (shoulder_mid[0], shoulder_mid[1] + 0.8 * (shoulder_mid[1] - nose[1]))
    shoulder_width = max(math.dist(ls, rs), 0.05)
    wrists_close = math.dist(lw, rw) < shoulder_width * 0.55
    wrists_near_chest = (
        abs((lw[1] + rw[1]) / 2 - chest[1]) < shoulder_width * 0.8
        and abs((lw[0] + rw[0]) / 2 - chest[0]) < shoulder_width * 0.85
    )
    # Crossed forearms: each wrist is on the OTHER shoulder's side. Comparing
    # the two orders works however MediaPipe labels left/right when mirrored.
    crossed = (lw[0] - rw[0]) * (ls[0] - rs[0]) < 0
    return wrists_close and wrists_near_chest and crossed


def x_pose_by_samples(landmarks, matcher: Optional[SampleMatcher], aspect: float) -> Optional[float]:
    """Distance to the nearest recorded X pose, or None (no samples / arms hidden)."""
    if matcher is None:
        return None
    vec = xpose_vector(landmarks, aspect)
    return None if vec is None else matcher.nearest(vec)[1]


class HoldXDetector:
    def __init__(self, matcher: Optional[SampleMatcher] = None):
        self.matcher = matcher
        self.started: Optional[float] = None
        self.fired = False
        self.last_repeat = 0.0

    def update(self, landmarks, now: float, hold_time: float,
               aspect: float = 16 / 9) -> Optional[Tuple[str, bool]]:
        if landmarks is None:
            self.started, self.fired = None, False
            return None
        # The rules OR a close match to a recorded X pose.
        dist = x_pose_by_samples(landmarks, self.matcher, aspect)
        posed = x_pose_by_rules(landmarks) or (dist is not None and dist <= XPOSE_MATCH_DIST)
        if not posed:
            self.started, self.fired = None, False
            return None
        if self.started is None:
            self.started = now
            return None
        elapsed = now - self.started
        if not self.fired and elapsed >= max(0.1, hold_time):
            self.fired = True
            self.last_repeat = now
            return ("hold_x", False)
        if self.fired and now - self.last_repeat >= 0.5:
            self.last_repeat = now
            return ("hold_x", True)
        return None


class CalibrationTracker:
    # Upper body only: head and shoulders must be inside the frame (plus a hand).
    REQUIRED = (
        POSE_LANDMARKS.NOSE.value,
        POSE_LANDMARKS.LEFT_SHOULDER.value, POSE_LANDMARKS.RIGHT_SHOULDER.value,
    )

    def __init__(self):
        self.still_since: Optional[float] = None
        self.previous = None
        self.last_report = 0.0
        self.done_sent = False

    def update(self, landmarks, hand_results, now: float, frame_shape) -> Optional[dict]:
        hands_ok = bool(hand_results and hand_results.multi_hand_landmarks)
        body_ok = landmarks is not None
        current = []
        if body_ok:
            for idx in self.REQUIRED:
                lm = landmarks[idx]
                if lm.visibility < 0.65 or not (0.02 <= lm.x <= 0.98 and 0.02 <= lm.y <= 0.98):
                    body_ok = False
                    break
                current.extend((lm.x, lm.y))
        if body_ok:
            # Standing height: the head must be above the squat line, otherwise
            # the player would count as squatting all the time.
            head = head_height(landmarks)
            body_ok = head is not None and head < SQUAT_LINE - SQUAT_REARM_MARGIN
        valid = body_ok and hands_ok
        checks = {"camera": True, "body": body_ok, "hands": hands_ok}
        # Calibration requires hands to be still too, not merely visible.
        if valid:
            for hand in hand_results.multi_hand_landmarks:
                for point in hand.landmark:
                    current.extend((point.x, point.y))
        if not valid:
            self.still_since, self.previous, self.done_sent = None, None, False
            if now - self.last_report >= 0.1:
                self.last_report = now
                return {"type": "calibration", **checks, "progress": 0.0, "seconds_left": 3.0, "done": False}
            return None

        movement = 0.0
        if self.previous is not None:
            movement = sum(abs(a - b) for a, b in zip(current, self.previous)) / len(current)
        self.previous = current
        # Tune stillness threshold: lower is stricter; 0.006 is normalized image motion.
        if movement > 0.006:
            self.still_since = now
            self.done_sent = False
        elif self.still_since is None:
            self.still_since = now

        elapsed = now - self.still_since
        progress = clamp(elapsed / 3.0, 0.0, 1.0)
        if now - self.last_report >= 0.1:
            self.last_report = now
            payload = {
                "type": "calibration",
                **checks,
                "progress": progress,
                "seconds_left": round(max(0.0, 3.0 - elapsed), 2),
                "done": False,
            }
            if progress >= 1.0 and not self.done_sent:
                payload["done"] = True
                self.done_sent = True
            return payload
        return None


def color_sample_from_pixels(hsv_pixels: np.ndarray):
    """HSV pixels of the object -> (colour sample dict, None) or (None, reason).
    Hue is circular (red sits at both 0 and 179), so a circular mean is used."""
    h = hsv_pixels[:, 0].astype(float)
    s = hsv_pixels[:, 1].astype(float)
    v = hsv_pixels[:, 2].astype(float)
    if np.median(s) < 80 or np.median(v) < 60:
        return None, "not colourful enough - use a brightly coloured object"
    angle = h * (2 * math.pi / 180)
    center = (math.degrees(math.atan2(np.sin(angle).mean(), np.cos(angle).mean())) / 2) % 180
    spread = np.abs((h - center + 90) % 180 - 90)          # circular distance in hue units
    width = float(np.percentile(spread, 90)) + 4
    if width > 25:
        return None, "several colours in the box - fill the whole box with the object"
    return {"hue": round(center, 1), "hue_width": round(width, 1),
            "sat_min": int(max(60, np.percentile(s, 10) - 20)),
            "val_min": int(max(50, np.percentile(v, 10) - 30))}, None


def color_ranges(sample: dict):
    """One colour sample -> cv2.inRange (low, high) pairs (two if hue wraps)."""
    lo, hi = sample["hue"] - sample["hue_width"], sample["hue"] + sample["hue_width"]
    s, v = sample["sat_min"], sample["val_min"]
    if lo < 0:
        return [((0, s, v), (int(hi), 255, 255)), ((int(lo + 180), s, v), (179, 255, 255))]
    if hi > 179:
        return [((int(lo), s, v), (179, 255, 255)), ((0, s, v), (int(hi - 180), 255, 255))]
    return [((int(lo), s, v), (int(hi), 255, 255))]


def load_object_ranges(folder: Path):
    """HSV ranges recorded with the recorder app, or None if there are none."""
    path = Path(folder) / OBJECT_COLOR_FILE
    try:
        samples = json.loads(path.read_text(encoding="utf-8")).get("samples", [])
    except (OSError, ValueError, AttributeError):
        return None
    ranges = [r for sample in samples for r in color_ranges(sample)]
    if ranges:
        print(f"[vision] loaded {len(samples)} recorded object colour samples")
    return ranges or None


class ObjectPowerUpDetector:
    """Optional, deliberately small HSV-based bright-object detector."""

    def __init__(self, folder: Optional[Path] = None):
        self.started = None
        self.sent = False
        # Recorded colour samples, else the default: bright yellow.
        self.ranges = (load_object_ranges(folder) if folder else None) or [((20, 100, 100), (40, 255, 255))]

    def mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lower, upper in self.ranges:
            part = cv2.inRange(hsv, lower, upper)
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        return mask

    def update(self, frame, now):
        mask = self.mask(frame)
        h, w = mask.shape
        roi = mask[: int(h * 0.42), :]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = max((cv2.contourArea(c) for c in contours), default=0)
        visible = area > (w * h * 0.002)
        if visible:
            if self.started is None:
                self.started = now
            if not self.sent and now - self.started >= 0.5:
                self.sent = True
                return "object_power_up"
        else:
            self.started, self.sent = None, False
        return None


def draw_zones(frame, zones: ZoneTracker, squat: Optional[SquatLine] = None,
               hands: Optional[HandsUpLine] = None):
    """Thirds of the picture (current zone highlighted) and the squat / jump line."""
    h, w = frame.shape[:2]
    if squat is not None:
        y = int(SQUAT_LINE * h)
        below = squat.head is not None and squat.head > SQUAT_LINE
        up = hands is not None and all(v is not None and v < SQUAT_LINE for v in hands.hands)
        label = "SQUAT / JUMP LINE" + ("  - SQUAT" if below else "") + ("  - JUMP" if up else "")
        color = (107, 107, 255) if (below or up) else (166, 225, 92)
        cv2.line(frame, (0, y), (w, y), color, 2)
        cv2.putText(frame, label, (w - 430, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if zones.zone is not None:
        overlay = frame.copy()
        x0 = zones.zone * w // 3
        cv2.rectangle(overlay, (x0, 0), (x0 + w // 3, h), (232, 123, 208), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    for i in (1, 2):
        cv2.line(frame, (i * w // 3, 0), (i * w // 3, h), (255, 140, 140), 2)
    for i, name in enumerate(("LEFT", "CENTER", "RIGHT")):
        cv2.putText(frame, f"{name} {zones.shares[i] * 100:.0f}%", (i * w // 3 + 12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)


def draw_debug(frame, pose_results, hand_results, fps, last_gesture, score, context, body_status=""):
    if pose_results and pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )
    if hand_results and hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS
            )
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 230, 40), 2)
    cv2.putText(frame, f"Gesture: {last_gesture}  score: {score:.2f}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 220, 240), 2)
    cv2.putText(frame, f"Context: {context or '-'}", (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(frame, f"Body: {body_status}", (12, 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)


def main():
    parser = argparse.ArgumentParser(description="Game webcam vision module")
    parser.add_argument("--no-hub", action="store_true", help="print outgoing messages instead of connecting")
    parser.add_argument("--camera", type=int, default=None, help="override settings.camera_index")
    parser.add_argument("--enable-object", action="store_true", help="enable optional HSV object power-up")
    parser.add_argument("--templates", type=Path, default=TEMPLATE_DIR)
    parser.add_argument("--width", type=int, default=640, help="capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="capture height (default: 480)")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1), default=0,
                        help="MediaPipe model complexity: 0 is faster, 1 may be more accurate")
    parser.add_argument("--max-hands", type=int, choices=(1, 2), default=1,
                        help="maximum hands to track (default: 1)")
    parser.add_argument("--no-display", action="store_true",
                        help="disable the debug window to reduce CPU use (stop with Ctrl+C)")
    parser.add_argument("--stats", action="store_true",
                        help="print average pose/hand processing times every 5 seconds")
    args = parser.parse_args()
    if args.width < 160 or args.height < 120:
        parser.error("--width must be >= 160 and --height must be >= 120")

    shared = SharedState(camera_override=args.camera)
    hub = HubClient(shared, enabled=not args.no_hub)
    hub.start()

    upper_body = UpperBodyFeatures()
    zones = ZoneTracker()
    squat_line = SquatLine()
    hands_up = HandsUpLine()
    last_jump = -1e9
    body_detector = BodyGestureDetector(args.templates)
    hand_detector = HandGestureDetector(load_samples(args.templates, HAND_SIGNS, HAND_COORDINATES))
    hold_detector = HoldXDetector(load_samples(args.templates, ("hold_x",), XPOSE_COORDINATES))
    calibration = CalibrationTracker()
    object_detector = ObjectPowerUpDetector(args.templates) if args.enable_object else None

    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=args.model_complexity, smooth_landmarks=True,
        enable_segmentation=False, min_detection_confidence=0.55, min_tracking_confidence=0.55,
    )
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=args.max_hands, model_complexity=args.model_complexity,
        min_detection_confidence=0.55, min_tracking_confidence=0.55,
    )

    cap = None
    active_camera = None
    last_body_seen = 0.0
    tracking_lost_sent = False
    found_since = None
    last_gesture = "none"
    last_score = 0.0
    fps = 0.0
    last_frame_time = time.monotonic()
    stats_started = last_frame_time
    stats_frames = 0
    stats_pose_seconds = 0.0
    stats_hands_seconds = 0.0
    if args.no_display:
        print("[vision] debug window disabled; press Ctrl+C to quit")
    else:
        print("[vision] press q in the debug window to quit")

    try:
        while True:
            state, settings, _ = shared.snapshot()
            requested_camera = int(settings.get("camera_index", 0))
            if cap is None or requested_camera != active_camera:
                if cap is not None:
                    cap.release()
                # A camera switch invalidates all temporal gesture histories.
                upper_body.reset()
                body_detector.history.clear()
                hand_detector.candidate, hand_detector.count = None, 0
                hold_detector.started, hold_detector.fired = None, False
                calibration.still_since, calibration.previous = None, None
                if object_detector:
                    object_detector.started, object_detector.sent = None, False
                last_body_seen = 0.0
                found_since = None
                active_camera = requested_camera
                cap = cv2.VideoCapture(active_camera)
                # A one-frame capture buffer avoids processing stale frames when
                # MediaPipe briefly falls behind the camera.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                if not cap.isOpened():
                    print(f"[vision] cannot open camera {active_camera}; retrying")
                    cap.release()
                    cap = None
                    if state.get("context") == "CALIBRATION":
                        hub.send({"type": "calibration", "camera": False, "body": False, "hands": False,
                                  "progress": 0.0, "seconds_left": 3.0, "done": False})
                    time.sleep(1.0)
                    continue

            ok, frame = cap.read()
            now = time.monotonic()
            if not ok:
                print("[vision] camera frame unavailable; reopening")
                cap.release()
                cap = None
                time.sleep(0.25)
                continue

            # Selfie mirror: player's own left/right matches the displayed view.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_started = time.perf_counter()
            pose_results = pose.process(rgb)
            pose_elapsed = time.perf_counter() - pose_started
            hands_started = time.perf_counter()
            hand_results = hands.process(rgb)
            hands_elapsed = time.perf_counter() - hands_started
            rgb.flags.writeable = True
            stats_frames += 1
            stats_pose_seconds += pose_elapsed
            stats_hands_seconds += hands_elapsed

            pose_landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
            body_seen = pose_landmarks is not None
            hands_seen = bool(hand_results and hand_results.multi_hand_landmarks)

            if body_seen:
                last_body_seen = now
                if tracking_lost_sent:
                    if found_since is None:
                        found_since = now
                    if now - found_since >= 0.5:
                        hub.send({"type": "tracking", "status": "found"})
                        tracking_lost_sent = False
                        found_since = None
                else:
                    found_since = None
            else:
                found_since = None
                if not tracking_lost_sent and now - last_body_seen > 1.0:
                    hub.send({"type": "tracking", "status": "lost"})
                    tracking_lost_sent = True

            sensitivity = clamp(float(settings.get("gesture_sensitivity", 0.5)), 0.0, 1.0)
            threshold = sensitivity_to_threshold(sensitivity)
            stable_frames = sensitivity_to_stable_frames(sensitivity)
            aspect = frame.shape[1] / frame.shape[0]
            # Updated every frame (also while calibrating) so the neutral
            # position is learned while the player stands still.
            features = upper_body.update(pose_landmarks, now, aspect)
            # Lane zone is tracked all the time, but (like every gesture) not
            # sent while calibrating; the 1 s repeat tells the game afterwards.
            zone_msg = zones.update(pose_landmarks, now)
            squatted = squat_line.update(pose_landmarks, now)
            hands_jump = hands_up.update(pose_landmarks, now)

            if state.get("context") == "CALIBRATION":
                cal_msg = calibration.update(pose_landmarks, hand_results, now, frame.shape)
                if cal_msg:
                    hub.send(cal_msg)
            else:
                # Do not make gameplay gesture decisions while calibrating.
                calibration.still_since = None
                calibration.previous = None
                calibration.done_sent = False

                if zone_msg:
                    name, repeat = zone_msg
                    if not repeat:
                        last_gesture, last_score = name, 1.0
                    hub.send({"type": "gesture", "gesture": name, "score": 1.0, "repeat": repeat})

                if squatted:
                    last_gesture, last_score = "squat", 1.0
                    hub.send({"type": "gesture", "gesture": "squat", "score": 1.0, "repeat": False})

                # Jump: both hands above the line, or the DollarPy movement
                # classifier - whichever comes first (shared cooldown).
                body_match = body_detector.update(features, now, threshold)
                jump = (("jump", 1.0) if hands_jump else None) or body_match
                if jump and now - last_jump >= JUMP_COOLDOWN:
                    last_jump = now
                    name, score = jump
                    last_gesture, last_score = name, score
                    hub.send({"type": "gesture", "gesture": name, "score": round(score, 4), "repeat": False})

                hand_match = hand_detector.update(
                    hand_results, bool(settings.get("left_handed", False)), stable_frames, now, aspect
                )
                if hand_match:
                    last_gesture, last_score = hand_match, 1.0
                    hub.send({"type": "gesture", "gesture": hand_match, "score": 1.0, "repeat": False})

                hold_match = hold_detector.update(
                    pose_landmarks, now, float(settings.get("hold_time", 1.0)), aspect
                )
                if hold_match:
                    name, repeat = hold_match
                    last_gesture, last_score = name, 1.0
                    hub.send({"type": "gesture", "gesture": name, "score": 1.0, "repeat": repeat})

                if object_detector:
                    object_match = object_detector.update(frame, now)
                    if object_match:
                        last_gesture, last_score = object_match, 1.0
                        hub.send({"type": "gesture", "gesture": object_match, "score": 1.0, "repeat": False})

            dt = max(now - last_frame_time, 1e-6)
            last_frame_time = now
            instantaneous_fps = 1.0 / dt
            fps = instantaneous_fps if fps == 0 else (0.9 * fps + 0.1 * instantaneous_fps)
            if args.stats and now - stats_started >= 5.0:
                elapsed = max(now - stats_started, 1e-6)
                print(
                    f"[vision stats] {stats_frames / elapsed:.1f} FPS | "
                    f"Pose {stats_pose_seconds / max(stats_frames, 1) * 1000:.1f} ms | "
                    f"Hands {stats_hands_seconds / max(stats_frames, 1) * 1000:.1f} ms"
                )
                stats_started = now
                stats_frames = 0
                stats_pose_seconds = 0.0
                stats_hands_seconds = 0.0

            if not args.no_display:
                draw_zones(frame, zones, squat_line, hands_up)
                draw_debug(frame, pose_results, hand_results, fps, last_gesture, last_score,
                           str(state.get("context", "")), body_detector.status)
                cv2.imshow("Game Vision Debug (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        if cap is not None:
            cap.release()
        pose.close()
        hands.close()
        cv2.destroyAllWindows()
        hub.close()


if __name__ == "__main__":
    main()
