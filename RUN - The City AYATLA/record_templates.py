#!/usr/bin/env python3
"""
record_templates.py - AYATLA Gesture Recorder: ONE app to record samples for
every gesture. (A developer tool: it uses the mouse and keyboard. The game
itself stays completely hands-free.)

    python record_templates.py              (or double-click Recorder.bat)
    python record_templates.py --camera 1
    python record_templates.py ok_sign      (start with ok_sign selected)

What vision.py does with each recording
  * Body movement (jump): DollarPy templates - a second way to detect a jump
    (the movement itself). Lanes, squats and jumps also work with NO samples:
    the third of the picture you stand in is your lane, your head below the
    line at 30% of the picture height is a squat, and both hands above that
    line is a jump.
  * Hand signs (thumbs_up = pause, open_palm, ok_sign, fingers_1..4) and hold_x: your own
    examples. A close match wins; otherwise the built-in rules decide, so you
    only NEED samples for the signs that are not detected well.
  * object_power_up: the colour of your power-up object (used with
    --enable-object; without samples vision looks for bright yellow).

New samples are ADDED to templates/<gesture>.json, so you can come back and
record more later (also with other people). Restart vision.py / the game
afterwards so it loads the new samples.

Keys: SPACE record (or cancel), D delete last sample, T live test,
Up/Down choose a gesture.
"""

from __future__ import annotations

import argparse
import datetime
import json
import queue
import threading
import time
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

import vision as V

# Every gesture: (name, kind, how to perform it)
CATALOG = [
    ("jump", "body", "Jump, or raise BOTH arms above your head. (Both hands above the line "
                     "already count as a jump; these DollarPy samples are a second way to detect it.)"),
    ("thumbs_up", "hand", "Thumbs up: make a fist with your thumb pointing straight up. (Pauses the game.)"),
    ("open_palm", "hand", "Show an open, flat palm to the camera with the fingers spread."),
    ("ok_sign", "hand", "Touch your thumb and index fingertips (a circle); the other three fingers point up."),
    ("fingers_1", "hand", "One finger up (index), thumb folded."),
    ("fingers_2", "hand", "Two fingers up (index + middle), thumb folded."),
    ("fingers_3", "hand", "Three fingers up, thumb folded."),
    ("fingers_4", "hand", "Four fingers up, thumb folded across the palm."),
    ("hold_x", "pose", "Cross your forearms in an X in front of your chest."),
    ("object_power_up", "object", "Hold your coloured object so it fills the pink box at the top of the picture."),
]
KIND_OF = {name: kind for name, kind, _ in CATALOG}
HOW_TO = {name: how for name, _, how in CATALOG}
GROUPS = [("body", "Body movements (DollarPy)"), ("hand", "Hand signs"),
          ("pose", "Held pose"), ("object", "Power-up object")]
COORDS = {"body": V.COORDINATE_SYSTEM, "hand": V.HAND_COORDINATES,
          "pose": V.XPOSE_COORDINATES, "object": "hsv v1"}
RECOMMENDED = {"body": 8, "hand": 5, "pose": 5, "object": 3}

COUNTDOWN_SECONDS = 3.0
CAPTURE_SECONDS = {"body": 1.0, "hand": 1.5, "pose": 1.5, "object": 1.0}
GET_READY = {"body": "Stand still in your normal position",
             "hand": "Raise your hand and show the sign",
             "pose": "Get into the X pose",
             "object": "Hold the object inside the pink box"}
FRAMES_PER_SAMPLE = 10                 # hand / pose samples keep up to 10 frames
OBJECT_BOX = (0.42, 0.12, 0.58, 0.32)  # x0, y0, x1, y1 as fractions of the frame

# Dark navy UI with violet and pink accents (same palette as the game).
BG, PANEL, FIELD = "#0B0B14", "#151527", "#1E1E36"
TEXT, MUTED = "#F2F2FC", "#9A9AB8"
VIOLET, PINK, GREEN, RED = "#8C8CFF", "#E87BD0", "#5CE1A6", "#FF6B6B"
BGR = {"violet": (255, 140, 140), "pink": (208, 123, 232), "red": (107, 107, 255),
       "green": (166, 225, 92), "white": (252, 242, 242), "navy": (20, 11, 11)}

FILE_LOCK = threading.Lock()


# ===========================================================================
# templates/<gesture>.json
# ===========================================================================
def template_path(folder: Path, gesture: str) -> Path:
    return Path(folder) / f"{gesture}.json"


def sample_count(folder: Path, gesture: str) -> int:
    try:
        data = json.loads(template_path(folder, gesture).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(data, dict) or data.get("coordinate_system") != COORDS[KIND_OF[gesture]]:
        return 0
    return len(data.get("samples", []))


def read_file(folder: Path, gesture: str) -> dict:
    """Existing samples, or a new empty file. A file in an older format is
    moved to templates/old_format/ first (it could not be used anyway)."""
    kind = KIND_OF[gesture]
    path = template_path(folder, gesture)
    empty = {"gesture": gesture, "kind": kind, "coordinate_system": COORDS[kind], "samples": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return empty
    except (OSError, ValueError):
        data = None
    if (not isinstance(data, dict) or not isinstance(data.get("samples"), list)
            or data.get("coordinate_system") != COORDS[kind]):
        backup = path.parent / "old_format" / f"{path.stem}_{int(time.time())}.json"
        backup.parent.mkdir(exist_ok=True)
        path.replace(backup)
        print(f"[rec] {path.name} was in an older format; moved it to {backup}")
        return empty
    return data


def write_file(folder: Path, gesture: str, data: dict) -> None:
    path = template_path(folder, gesture)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)                  # never leaves half a file behind


def add_sample(folder: Path, gesture: str, sample) -> int:
    with FILE_LOCK:
        data = read_file(folder, gesture)
        data["samples"].append(sample)
        write_file(folder, gesture, data)
        return len(data["samples"])


def delete_samples(folder: Path, gesture: str, everything: bool = False) -> int:
    """Delete the last sample (or all of them); returns how many were removed."""
    with FILE_LOCK:
        data = read_file(folder, gesture)
        removed = len(data["samples"]) if everything else min(1, len(data["samples"]))
        if removed:
            data["samples"] = [] if everything else data["samples"][:-1]
            write_file(folder, gesture, data)
        return removed


# ===========================================================================
# Checking a finished recording and turning it into a sample
# ===========================================================================
def pick_evenly(items, n):
    if len(items) <= n:
        return list(items)
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, n)]


def stamp(person):
    return {"person": person or "anonymous",
            "recorded_at": datetime.datetime.now().isoformat(timespec="seconds")}


def build_sample(gesture, kind, items, person):
    """Return (sample, None, info) or (None, reason it was rejected, "")."""
    if kind == "body":
        frames = [f for f in items if f is not None]
        if len(frames) < 8:
            return None, "head and shoulders were not visible enough", ""
        direction, amount = V.movement_direction(frames)
        if amount < V.MIN_MOTION:
            return None, f"movement too small ({amount:.2f} < {V.MIN_MOTION}) - move more clearly", ""
        if direction != V.GESTURE_TO_DIRECTION[gesture]:
            return None, f"you moved {direction}, but {gesture} needs {V.GESTURE_TO_DIRECTION[gesture]}", ""
        return V.features_to_rows(frames), None, f"(moved {direction} {amount:.2f})"

    if kind in ("hand", "pose"):
        good = [i for i in items if i is not None and i[0] is not None]
        if len(good) < max(5, 0.6 * len(items)):
            if kind == "pose":
                reason = "both arms must be visible"
            elif gesture in V.SIDEWAYS_SIGNS:
                reason = "hand not seen clearly - keep your hand in view"
            else:
                reason = "hand not seen clearly - raise it with the fingers pointing up"
            return None, reason, ""
        sample = stamp(person)
        sample["frames"] = [[round(float(x), 4) for x in i[0]] for i in pick_evenly(good, FRAMES_PER_SAMPLE)]
        if kind == "hand":
            sample["hand"] = Counter(i[2] for i in good).most_common(1)[0][0]
            rules_saw = Counter(i[1] for i in good).most_common(1)[0][0]
            return sample, None, f"(the built-in rules saw: {rules_saw or 'no sign'})"
        # Sanity check so a wrong recording cannot turn normal poses into an X:
        # in xpose_vector the wrists are points 5 and 6, in shoulder widths.
        gaps = [float(np.hypot(i[0][10] - i[0][12], i[0][11] - i[0][13])) for i in good]
        if float(np.median(gaps)) > 1.0:
            return None, "your wrists are not close together - cross your forearms in front of your chest", ""
        rules_saw = sum(1 for i in good if i[1]) > len(good) / 2
        return sample, None, f"(the built-in rules {'also' if rules_saw else 'did NOT'} see an X)"

    pixels = np.concatenate([i for i in items if i is not None])
    colour, problem = V.color_sample_from_pixels(pixels)
    if problem:
        return None, problem, ""
    colour.update(stamp(person))
    return colour, None, f"(hue {colour['hue']:.0f})"


# ===========================================================================
# Camera + MediaPipe thread (keeps the window responsive)
# ===========================================================================
class CameraWorker(threading.Thread):
    def __init__(self, folder: Path, camera_index: int):
        super().__init__(daemon=True, name="camera")
        self.folder = Path(folder)
        self.camera_index = camera_index
        self.commands = queue.Queue()     # window -> worker
        self.events = queue.Queue()       # worker -> window
        self.lock = threading.Lock()
        self.frame = None                 # latest RGB preview frame
        self.stopping = threading.Event()
        self.selected = CATALOG[0][0]
        self.left_handed = False
        self.person = "anonymous"
        self.live_test = False
        self.phase = "idle"               # idle -> countdown -> recording -> idle
        self.phase_start = 0.0
        self.rec_gesture, self.rec_kind, self.captured = None, None, []
        self.upper = V.UpperBodyFeatures()
        self.zones = V.ZoneTracker()
        self.squat = V.SquatLine()
        self.hands_up = V.HandsUpLine()
        self.last_test_post = 0.0
        self.last_detect = None

    def send(self, *command):
        self.commands.put(command)

    def post(self, *event):
        self.events.put(event)

    # ---------------- commands from the window ----------------
    def handle_commands(self, now):
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return
            name = cmd[0]
            if name == "select" and self.phase == "idle":
                self.selected = cmd[1]
            elif name == "record":
                if self.phase == "idle":
                    self.person = cmd[1]
                    self.rec_gesture, self.rec_kind = self.selected, KIND_OF[self.selected]
                    self.phase, self.phase_start = "countdown", now
                    self.post("phase", "countdown")
                else:
                    self.phase = "idle"
                    self.post("phase", "idle")
                    self.post("status", "Recording cancelled.", MUTED)
            elif name == "camera":
                self.camera_index = cmd[1]
            elif name == "left_handed":
                self.left_handed = cmd[1]
            elif name == "test":
                self.live_test = cmd[1]
                if self.live_test:
                    self.reload_detectors()
            elif name == "reload":
                self.reload_detectors()
            elif name == "stop":
                self.stopping.set()

    def reload_detectors(self):
        self.body_detector = V.BodyGestureDetector(self.folder)
        self.hand_matcher = V.load_samples(self.folder, V.HAND_SIGNS, V.HAND_COORDINATES)
        self.x_matcher = V.load_samples(self.folder, ("hold_x",), V.XPOSE_COORDINATES)
        self.object_detector = V.ObjectPowerUpDetector(self.folder)
        self.last_detect = None

    # ---------------- main loop ----------------
    def run(self):
        pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                                      min_detection_confidence=0.55, min_tracking_confidence=0.55)
        hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                         min_detection_confidence=0.55, min_tracking_confidence=0.55)
        self.reload_detectors()
        cap, opened = None, None
        try:
            while not self.stopping.is_set():
                self.handle_commands(time.monotonic())
                if cap is None or opened != self.camera_index:
                    if cap is not None:
                        cap.release()
                    opened = self.camera_index
                    cap = cv2.VideoCapture(opened)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        self.post("status", f"Cannot open camera {opened}. Close the game / vision "
                                            "(they use the camera) or choose another camera.", RED)
                        self.stopping.wait(2.0)
                        continue
                    self.upper.reset()
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    self.stopping.wait(0.3)
                    continue
                self.process(cv2.flip(frame, 1), pose, hands)   # mirrored, like vision.py
        finally:
            if cap is not None:
                cap.release()
            pose.close()
            hands.close()

    def process(self, frame, pose, hands):
        now = time.monotonic()
        h, w = frame.shape[:2]
        aspect = w / h
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb)
        kind = KIND_OF[self.rec_gesture if self.phase != "idle" else self.selected]
        hand_results = hands.process(rgb) if (kind == "hand" or self.live_test) else None
        landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
        # Updated every frame: standing still during the countdown teaches it
        # your neutral position, exactly like vision.py does in the game.
        features = self.upper.update(landmarks, now, aspect)

        self.step_recording(now, frame, landmarks, hand_results, features, aspect)
        if self.live_test:
            self.run_live_test(now, frame, landmarks, hand_results, features, aspect)
        self.draw(frame, pose_results, hand_results, now, kind)
        with self.lock:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------------- recording ----------------
    def capture_item(self, frame, landmarks, hand_results, features, aspect):
        if self.rec_kind == "body":
            return features
        if self.rec_kind == "hand":
            chosen, label = V.choose_hand(hand_results, self.left_handed)
            if chosen is None:
                return None
            raised = self.rec_gesture not in V.SIDEWAYS_SIGNS     # a thumbs up is shown sideways
            return (V.hand_vector(chosen, label, aspect, require_raised=raised), V.hand_pose_name(chosen), label)
        if self.rec_kind == "pose":
            if landmarks is None:
                return None
            return (V.xpose_vector(landmarks, aspect), V.x_pose_by_rules(landmarks))
        h, w = frame.shape[:2]                 # object: the pixels inside the box
        x0, y0, x1, y1 = OBJECT_BOX
        box = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        return cv2.cvtColor(box, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    def step_recording(self, now, frame, landmarks, hand_results, features, aspect):
        if self.phase == "countdown" and now - self.phase_start >= COUNTDOWN_SECONDS:
            self.phase, self.phase_start, self.captured = "recording", now, []
            self.post("phase", "recording")
        elif self.phase == "recording":
            self.captured.append(self.capture_item(frame, landmarks, hand_results, features, aspect))
            if now - self.phase_start >= CAPTURE_SECONDS[self.rec_kind]:
                self.phase = "idle"
                self.post("phase", "idle")
                self.finish_recording()

    def finish_recording(self):
        g, kind = self.rec_gesture, self.rec_kind
        sample, problem, info = build_sample(g, kind, self.captured, self.person)
        if problem:
            self.post("status", f"NOT saved: {problem}. Press Record to try again.", RED)
            return
        try:
            total = add_sample(self.folder, g, sample)
        except OSError as exc:
            self.post("status", f"Could not save: {exc}", RED)
            return
        self.post("status", f"Saved {g} sample #{total} {info}", GREEN)
        self.post("counts")
        self.reload_detectors()

    # ---------------- live test ----------------
    def run_live_test(self, now, frame, landmarks, hand_results, features, aspect):
        hit = self.body_detector.update(features, now, V.sensitivity_to_threshold(0.72))
        if hit:
            self.last_detect = (f"{hit[0]} (score {hit[1]:.2f})", now)
        self.zones.update(landmarks, now)
        if self.squat.update(landmarks, now):
            self.last_detect = ("squat (head below the line)", now)
        if self.hands_up.update(landmarks, now):
            self.last_detect = ("jump (both hands above the line)", now)
        if now - self.last_test_post < 0.2:
            return
        self.last_test_post = now
        lines = [f"Body movement : {self.body_detector.status}"]
        if self.last_detect and now - self.last_detect[1] < 2.0:
            lines[0] += f"   ->  DETECTED {self.last_detect[0]}"
        zone = "-" if self.zones.zone is None else ("LEFT", "MIDDLE", "RIGHT")[self.zones.zone]
        shares = " / ".join(f"{s * 100:.0f}%" for s in self.zones.shares)
        lines.append(f"Lane (third)  : {zone}   [body share left / middle / right: {shares}]")
        lines.append(f"Squat         : {self.squat.status()}")
        lines.append(f"Jump (line)   : {self.hands_up.status()}")

        chosen, label = V.choose_hand(hand_results, self.left_handed)
        if chosen is None:
            lines.append("Hand sign     : no hand")
        else:
            final, rule, sample, dist = V.classify_hand_sign(chosen, label, self.hand_matcher, aspect)
            closest = "no samples" if sample is None else f"{sample} {dist:.2f} (needs <= {V.HAND_MATCH_DIST})"
            lines.append(f"Hand sign     : {final or '-'}   [rules: {rule or '-'} | closest sample: {closest}]")

        if landmarks is None:
            lines.append("Hold X        : no body")
        else:
            rule_x = V.x_pose_by_rules(landmarks)
            dist = V.x_pose_by_samples(landmarks, self.x_matcher, aspect)
            yes = rule_x or (dist is not None and dist <= V.XPOSE_MATCH_DIST)
            closest = "no samples" if dist is None else f"{dist:.2f} (needs <= {V.XPOSE_MATCH_DIST})"
            lines.append(f"Hold X        : {'YES' if yes else 'no'}   [rules: {'yes' if rule_x else 'no'} | "
                         f"closest sample: {closest}]")

        h, w = frame.shape[:2]
        mask = self.object_detector.mask(frame)[: int(h * 0.42)]
        share = cv2.countNonZero(mask) / float(w * h) * 100
        lines.append(f"Object colour : {share:.2f}% of the frame (vision needs about 0.2%)")
        self.post("test", "\n".join(lines))

    # ---------------- overlay on the preview ----------------
    def draw(self, frame, pose_results, hand_results, now, kind):
        h, w = frame.shape[:2]
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=BGR["pink"], thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=BGR["violet"], thickness=3))
        if hand_results and hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        if self.live_test:
            V.draw_zones(frame, self.zones, self.squat, self.hands_up)
        if kind == "object":
            x0, y0, x1, y1 = OBJECT_BOX
            cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), BGR["pink"], 3)

        if self.phase == "countdown":
            left = COUNTDOWN_SECONDS - (now - self.phase_start)
            centered(frame, str(int(left) + 1), h // 2, 5.0, BGR["pink"], 10)
            centered(frame, GET_READY[self.rec_kind], h // 2 + 80, 1.1, BGR["white"], 2)
        elif self.phase == "recording":
            progress = min(1.0, (now - self.phase_start) / CAPTURE_SECONDS[self.rec_kind])
            cv2.circle(frame, (40, 40), 14, BGR["red"], -1)
            centered(frame, "RECORDING", 60, 1.4, BGR["red"], 3)
            centered(frame, HOW_TO[self.rec_gesture], h - 60, 0.9, BGR["white"], 2)
            cv2.rectangle(frame, (0, h - 12), (int(w * progress), h), BGR["red"], -1)


def centered(frame, text, y, scale, color, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, y), font, scale, BGR["navy"], thickness + 6, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ===========================================================================
# The window
# ===========================================================================
class RecorderApp:
    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.folder = Path(args.templates)
        self.selected = args.gesture or CATALOG[0][0]
        self.busy = False
        self.testing = False
        self.photo = None

        root.title("AYATLA - Gesture Recorder")
        root.configure(bg=BG)
        root.geometry(f"{min(1400, root.winfo_screenwidth() - 40)}x{min(820, root.winfo_screenheight() - 80)}")
        root.minsize(1000, 600)
        self.setup_style()
        self.build(args)
        self.refresh_counts()
        self.select(self.selected)

        self.worker = CameraWorker(self.folder, args.camera)
        self.worker.selected = self.selected
        self.worker.start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        for key in ("<space>", "<KeyPress-d>", "<KeyPress-D>", "<KeyPress-t>", "<KeyPress-T>"):
            root.bind_all(key, self.on_key)
        root.after(30, self.tick)

    # ---------------- layout ----------------
    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=FIELD, fieldbackground=FIELD, foreground=TEXT,
                        rowheight=26, borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background=PANEL, foreground=MUTED,
                        font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("Treeview", background=[("selected", VIOLET)], foreground=[("selected", BG)])

    def button(self, parent, text, command, color):
        return tk.Button(parent, text=text, command=command, bg=color, fg=BG, activebackground=TEXT,
                         activeforeground=BG, relief="flat", bd=0, font=("Segoe UI", 11, "bold"),
                         padx=14, pady=9, cursor="hand2", takefocus=0)   # SPACE must not press it

    def label(self, parent, text, color=TEXT, size=10, bold=False, bg=PANEL, **kw):
        return tk.Label(parent, text=text, bg=bg, fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

    def build(self, args):
        left = tk.Frame(self.root, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)
        right = tk.Frame(self.root, bg=PANEL, width=440)
        right.pack(side="right", fill="y", padx=(8, 16), pady=16)
        right.pack_propagate(False)

        # Left: title, camera preview, status, live test read-out
        self.label(left, "AYATLA  ·  GESTURE RECORDER", VIOLET, 12, True, bg=BG).pack(anchor="w")
        holder = tk.Frame(left, bg="#000000")
        holder.pack(fill="both", expand=True, pady=8)
        holder.pack_propagate(False)
        self.preview_holder = holder
        self.preview = tk.Label(holder, bg="#000000", fg=MUTED, text="Starting camera...",
                                font=("Segoe UI", 14))
        self.preview.pack(fill="both", expand=True)
        self.status = self.label(left, "Choose a gesture on the right, then press Record (or SPACE).",
                                 TEXT, 13, bg=BG, anchor="w", justify="left", wraplength=900)
        self.status.pack(fill="x")
        self.test_box = tk.Label(left, text="", bg=PANEL, fg=TEXT, font=("Consolas", 10),
                                 anchor="w", justify="left", padx=12, pady=8)

        # Right: settings
        pad = {"padx": 16}
        self.label(right, "RECORDING SETTINGS", MUTED, 9, True).pack(anchor="w", pady=(16, 4), **pad)
        row = tk.Frame(right, bg=PANEL)
        row.pack(fill="x", **pad)
        self.label(row, "Person").pack(side="left")
        self.person = tk.Entry(row, bg=FIELD, fg=TEXT, insertbackground=TEXT, relief="flat",
                               font=("Segoe UI", 11), width=14)
        self.person.insert(0, args.person)
        self.person.pack(side="left", padx=(8, 16), ipady=4)
        self.label(row, "Camera").pack(side="left")
        self.camera = tk.Spinbox(row, from_=0, to=9, width=3, bg=FIELD, fg=TEXT, relief="flat",
                                 buttonbackground=FIELD, font=("Segoe UI", 11),
                                 command=self.change_camera)
        self.camera.delete(0, "end")
        self.camera.insert(0, str(args.camera))
        self.camera.pack(side="left", padx=8, ipady=3)
        self.left_handed = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Record with my LEFT hand (left-handed)", variable=self.left_handed,
                       command=lambda: self.worker.send("left_handed", self.left_handed.get()),
                       bg=PANEL, fg=TEXT, selectcolor=FIELD, activebackground=PANEL, activeforeground=TEXT,
                       font=("Segoe UI", 10), takefocus=0).pack(anchor="w", pady=(8, 0), **pad)

        # Right, bottom (packed first so it is always visible): instructions + buttons
        self.label(right, "New samples are added to the saved ones. Restart the game\n"
                          "(or vision.py) afterwards so it uses them.", MUTED, 9,
                   justify="left").pack(side="bottom", anchor="w", pady=(6, 14), **pad)
        buttons = tk.Frame(right, bg=PANEL)
        buttons.pack(side="bottom", fill="x", **pad)
        self.record_button = self.button(buttons, "●  Record sample  (SPACE)", self.record, PINK)
        self.record_button.pack(fill="x")
        row2 = tk.Frame(buttons, bg=PANEL)
        row2.pack(fill="x", pady=8)
        self.button(row2, "Delete last (D)", self.delete_last, MUTED).pack(side="left", fill="x", expand=True)
        self.button(row2, "Delete all", self.delete_all, MUTED).pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.test_button = self.button(buttons, "Live test: OFF  (T)", self.toggle_test, VIOLET)
        self.test_button.pack(fill="x")
        self.how = self.label(right, "", TEXT, 10, wraplength=400, justify="left", anchor="w")
        self.how.pack(side="bottom", fill="x", pady=(8, 8), **pad)

        # Right, middle: gesture list (takes the space that is left, scrolls if needed)
        self.label(right, "GESTURES", MUTED, 9, True).pack(anchor="w", pady=(14, 4), **pad)
        tree_frame = tk.Frame(right, bg=PANEL)
        tree_frame.pack(fill="both", expand=True, **pad)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        self.tree = ttk.Treeview(tree_frame, columns=("samples",), selectmode="browse",
                                 yscrollcommand=scrollbar.set)
        scrollbar.configure(command=self.tree.yview)
        self.tree.heading("#0", text="Gesture")
        self.tree.heading("samples", text="Samples")
        self.tree.column("#0", width=250)
        self.tree.column("samples", width=110, anchor="center")
        self.tree.tag_configure("group", foreground=VIOLET, font=("Segoe UI", 10, "bold"))
        self.tree.tag_configure("few", foreground=PINK)
        self.tree.tag_configure("enough", foreground=GREEN)
        for kind, title in GROUPS:
            self.tree.insert("", "end", iid=f"group:{kind}", text=title, open=True, tags=("group",))
            for name, k, _ in CATALOG:
                if k == kind:
                    self.tree.insert(f"group:{kind}", "end", iid=name, text=f"  {name}", values=("",))
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.focus_set()                 # Up/Down arrows choose a gesture

    # ---------------- actions ----------------
    def select(self, gesture):
        self.selected = gesture
        kind = KIND_OF[gesture]
        if self.tree.selection() != (gesture,):
            self.tree.selection_set(gesture)
            self.tree.see(gesture)
        rec = RECOMMENDED[kind]
        uses = {"body": "Used by DollarPy (the course's movement classifier). Optional for playing: "
                        "both hands above the line already count as a jump.",
                "hand": "Optional: your samples teach vision how YOU do this sign (the built-in rules still work).",
                "pose": "Optional: your samples teach vision how YOU do the X (the built-in rules still work).",
                "object": "Teaches vision the colour of your power-up object (game: --enable-object)."}[kind]
        self.how.configure(text=f"{gesture}\n\nHow: {HOW_TO[gesture]}\n\n{uses}\nRecommended: {rec}+ samples.")
        if hasattr(self, "worker"):
            self.worker.send("select", gesture)

    def on_tree_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("group:"):
            children = self.tree.get_children(iid)
            if children:
                self.select(children[0])
        elif iid != self.selected and not self.busy:
            self.select(iid)
        elif self.busy and iid != self.selected:
            self.tree.selection_set(self.selected)

    def record(self):
        self.worker.send("record", self.person.get().strip() or "anonymous")

    def delete_last(self):
        if self.busy:
            return
        removed = delete_samples(self.folder, self.selected)
        self.set_status(f"Deleted the last {self.selected} sample." if removed
                        else f"No {self.selected} samples to delete.", MUTED)
        self.after_change()

    def delete_all(self):
        if self.busy:
            return
        n = sample_count(self.folder, self.selected)
        if n and messagebox.askyesno("Delete all samples",
                                     f"Delete all {n} samples of {self.selected}? This cannot be undone."):
            delete_samples(self.folder, self.selected, everything=True)
            self.set_status(f"Deleted all {self.selected} samples.", MUTED)
            self.after_change()

    def after_change(self):
        self.refresh_counts()
        self.worker.send("reload")

    def toggle_test(self):
        self.testing = not self.testing
        self.worker.send("test", self.testing)
        self.test_button.configure(text=f"Live test: {'ON' if self.testing else 'OFF'}  (T)")
        if self.testing:
            self.test_box.configure(text="Live test: move, show a sign or cross your arms...")
            self.test_box.pack(fill="x", pady=(8, 0))
        else:
            self.test_box.pack_forget()

    def change_camera(self):
        try:
            self.worker.send("camera", int(self.camera.get()))
        except ValueError:
            pass

    def on_key(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, tk.Spinbox)):
            return None                       # typing a name: keys are text
        key = event.keysym.lower()
        if key == "space":
            self.record()
        elif key == "d":
            self.delete_last()
        elif key == "t":
            self.toggle_test()
        return "break"

    def set_status(self, text, color=TEXT):
        self.status.configure(text=text, fg=color)

    def refresh_counts(self):
        for name, kind, _ in CATALOG:
            n = sample_count(self.folder, name)
            rec = RECOMMENDED[kind]
            self.tree.item(name, values=(f"{n} / {rec}",), tags=("enough" if n >= rec else "few",))

    # ---------------- refresh loop ----------------
    def tick(self):
        with self.worker.lock:
            frame = self.worker.frame
        if frame is not None:
            bw, bh = self.preview_holder.winfo_width(), self.preview_holder.winfo_height()
            if bw > 20 and bh > 20:
                fh, fw = frame.shape[:2]
                scale = min(bw / fw, bh / fh)
                image = Image.fromarray(frame).resize((max(1, int(fw * scale)), max(1, int(fh * scale))),
                                                      Image.BILINEAR)
                self.photo = ImageTk.PhotoImage(image)
                self.preview.configure(image=self.photo, text="")
        while True:
            try:
                event = self.worker.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "status":
                self.set_status(event[1], event[2])
            elif kind == "counts":
                self.refresh_counts()
            elif kind == "phase":
                self.busy = event[1] != "idle"
                self.record_button.configure(
                    text="■  Cancel  (SPACE)" if self.busy else "●  Record sample  (SPACE)",
                    bg=RED if self.busy else PINK)
                if event[1] == "countdown":
                    self.set_status(f"Get ready: {GET_READY[KIND_OF[self.selected]]}...", PINK)
                elif event[1] == "recording":
                    self.set_status(f"RECORDING: {HOW_TO[self.selected]}", RED)
            elif kind == "test" and self.testing:
                self.test_box.configure(text=event[1])
        self.root.after(30, self.tick)

    def close(self):
        self.worker.send("stop")
        self.worker.join(timeout=3)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="AYATLA gesture recorder (all gestures in one app)")
    parser.add_argument("gesture", nargs="?", choices=[name for name, _, _ in CATALOG],
                        help="gesture to select at start")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--person", default="anonymous", help="name stored with each sample")
    parser.add_argument("--templates", type=Path, default=V.TEMPLATE_DIR)
    args = parser.parse_args()
    args.templates.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    RecorderApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
