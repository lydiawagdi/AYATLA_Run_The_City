"""
context_manager.py - context-based content for AYATLA.

Everything the game shows or does that depends on (screen, profile) comes from
here, driven by the JSON files in content/:

    content/screens.json         per-context title / subtitle / options / hint
                                 chips / gesture_map  (+ per-mode overrides)
    content/guest.json           default (guest) profile
    content/profiles/<id>.json   one file per Bluetooth user
    content/gameplay.json        lanes, speed, spawn rates, power-ups, difficulty

Public API (used by game.py):

    get_profile(state)              -> profile dict (guest unless Bluetooth user)
    get_screen_content(state)       -> content for state["context"] for that profile
    resolve_action(state, gesture)  -> action string or None
    save_score(state, score)        -> updates best_score for Bluetooth users ONLY

Extra helpers: get_theme, get_gameplay, save_settings, default_settings.

`state` is the hub state dict:
    {"context": "...", "mode": "guest"|"bluetooth", "user": "user_01"|None, ...}

Resolution order for a gesture_map (later wins):
    screens.json[context].gesture_map
      -> screens.json[context].by_mode[guest|bluetooth].gesture_map
      -> profile["gesture_overrides"][context]

so the SAME gesture means different things per context AND per profile.
"""

import copy
import fnmatch
import json
import os
import tempfile
from pathlib import Path

GUEST_ID = "guest"


class _SafeDict(dict):
    def __missing__(self, key):          # unknown {placeholders} stay untouched
        return "{" + key + "}"


def _deep_merge(base, over):
    """Return base merged with over. Dicts merge recursively, everything else
    (including lists) is replaced."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _format_tree(node, vars_):
    """Recursively apply str.format_map to every string in a JSON-like tree."""
    if isinstance(node, str):
        try:
            return node.format_map(_SafeDict(vars_))
        except (ValueError, IndexError):
            return node
    if isinstance(node, list):
        return [_format_tree(x, vars_) for x in node]
    if isinstance(node, dict):
        return {k: _format_tree(v, vars_) for k, v in node.items()}
    return node


class ContextManager:
    def __init__(self, content_dir=None):
        base = Path(__file__).resolve().parent
        self.dir = Path(content_dir) if content_dir else base / "content"
        self.reload()

    # ------------------------------------------------------------------ io
    def _read(self, rel):
        with open(self.dir / rel, "r", encoding="utf-8") as f:
            return json.load(f)

    def reload(self):
        self.screens_doc = self._read("screens.json")
        self.gameplay_doc = self._read("gameplay.json")
        self.guest = self._read("guest.json")
        self.profiles = {}
        pdir = self.dir / "profiles"
        if pdir.is_dir():
            for p in sorted(pdir.glob("*.json")):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                except (OSError, ValueError):
                    continue
                pid = raw.get("id") or p.stem
                # Missing fields fall back to the guest defaults.
                self.profiles[pid] = _deep_merge(self.guest, raw)
                self.profiles[pid]["id"] = pid

    def _write_json_atomic(self, path, data):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            return True
        except OSError:
            return False

    def _persist_profile(self, pid):
        """Write only the fields the profile file owns (not guest-merged ones
        that were never in the file would be harmless, so we write it all)."""
        prof = copy.deepcopy(self.profiles[pid])
        return self._write_json_atomic(self.dir / "profiles" / f"{pid}.json", prof)

    # ------------------------------------------------------------ identity
    @staticmethod
    def _user_id(state):
        u = (state or {}).get("user")
        if isinstance(u, dict):
            u = u.get("id") or u.get("user_id") or u.get("UserID")
        return str(u) if u else None

    def profile_id(self, state):
        """'guest' unless state.mode == 'bluetooth' AND state.user is not null."""
        state = state or {}
        uid = self._user_id(state)
        if state.get("mode") == "bluetooth" and uid:
            if uid not in self.profiles:     # first time we see this user
                fresh = copy.deepcopy(self.guest)
                fresh["id"] = uid
                fresh["display_name"] = uid
                fresh["best_score"] = 0
                self.profiles[uid] = fresh
            return uid
        return GUEST_ID

    # Colour schemes given to newly registered players (cycled).
    NEW_PLAYER_THEMES = [
        ("#4FE3C1", "#3DDBC0", "#FFB454", "#08121A"),
        ("#FF9F5A", "#FF8A4C", "#5CE1E6", "#150C10"),
        ("#B18CFF", "#9D7BFF", "#FF7BB8", "#0E0B18"),
        ("#6FD3FF", "#4CC3FF", "#FFD166", "#081320"),
    ]

    def has_profile(self, uid):
        """True if a saved profile file exists for this Bluetooth user."""
        return (self.dir / "profiles" / f"{uid}.json").exists()

    def create_profile(self, uid, display_name, overwrite=False, persist=True):
        """Profile for a Bluetooth player, named after their device. Keeps an
        existing saved profile unless overwrite=True (a brand-new registration
        must never inherit someone else's profile). Returns the profile."""
        if not overwrite and uid in self.profiles and self.has_profile(uid):
            return self.profiles[uid]
        fresh = copy.deepcopy(self.guest)
        avatar, accent, accent2, bg = self.NEW_PLAYER_THEMES[len(self.profiles) % len(self.NEW_PLAYER_THEMES)]
        fresh.update({"id": uid, "display_name": display_name or uid, "best_score": 0,
                      "avatar_color": avatar, "theme_accent": accent,
                      "theme_accent_2": accent2, "theme_bg": bg})
        self.profiles[uid] = fresh
        if persist:
            self._persist_profile(uid)
        return fresh

    def is_bluetooth_user(self, state):
        return self.profile_id(state) != GUEST_ID

    def get_profile(self, state):
        pid = self.profile_id(state)
        return self.guest if pid == GUEST_ID else self.profiles[pid]

    # ------------------------------------------------------------- content
    def _mode_key(self, state):
        return "guest" if self.profile_id(state) == GUEST_ID else "bluetooth"

    def _template_vars(self, state):
        p = self.get_profile(state)
        return {
            "display_name": p.get("display_name", "Guest"),
            "best_score": f"{int(p.get('best_score', 0)):,}",
            "difficulty": p.get("difficulty", "normal"),
            "profile_id": p.get("id", GUEST_ID),
        }

    def _effective_gesture_map(self, state):
        ctx = (state or {}).get("context")
        base = self.screens_doc["screens"].get(ctx, {})
        gm = dict(base.get("gesture_map", {}))
        gm.update(base.get("by_mode", {}).get(self._mode_key(state), {}).get("gesture_map", {}))
        gm.update(self.get_profile(state).get("gesture_overrides", {}).get(ctx, {}))
        return gm

    def get_screen_content(self, state):
        ctx = (state or {}).get("context")
        raw = self.screens_doc["screens"].get(ctx)
        if raw is None:
            raw = {"title": str(ctx), "subtitle": "", "options": [], "hints": [], "gesture_map": {}}
        raw = copy.deepcopy(raw)
        by_mode = raw.pop("by_mode", {})
        content = _deep_merge(raw, by_mode.get(self._mode_key(state), {}))
        profile = self.get_profile(state)
        content = _deep_merge(content, profile.get("screen_overrides", {}).get(ctx, {}))
        content["gesture_map"] = self._effective_gesture_map(state)
        content = _format_tree(content, self._template_vars(state))
        content["context"] = ctx
        content["profile_id"] = profile.get("id", GUEST_ID)
        content["gesture_labels"] = self.screens_doc.get("gesture_labels", {})
        return content

    def resolve_action(self, state, gesture):
        """Action string for this gesture in this context+profile, else None."""
        gm = self._effective_gesture_map(state)
        if gesture in gm:
            return gm[gesture] or None
        for pattern, action in gm.items():       # e.g. "fingers_*"
            if "*" in pattern and fnmatch.fnmatch(gesture, pattern):
                return action or None
        return None

    # --------------------------------------------------------------- theme
    def get_theme(self, state):
        p = self.get_profile(state)
        return {
            "accent": p.get("theme_accent", self.guest["theme_accent"]),
            "accent_2": p.get("theme_accent_2", self.guest["theme_accent_2"]),
            "background": p.get("theme_bg", self.guest["theme_bg"]),
            "avatar": p.get("avatar_color", self.guest["avatar_color"]),
        }

    # ------------------------------------------------------------ gameplay
    def get_gameplay(self, state):
        """gameplay.json with the profile's difficulty modifiers resolved."""
        gp = copy.deepcopy(self.gameplay_doc)
        diff = self.get_profile(state).get("difficulty", "normal")
        mods = gp.get("difficulty", {})
        gp["difficulty_name"] = diff if diff in mods else "normal"
        gp["mod"] = mods.get(gp["difficulty_name"], {})
        return gp

    # ------------------------------------------------------------ settings
    def default_settings(self):
        return copy.deepcopy(self.guest["settings"])

    def get_settings(self, state):
        return copy.deepcopy(self.get_profile(state).get("settings", self.default_settings()))

    def save_settings(self, state, settings):
        """Persist settings to the profile file. Bluetooth users only."""
        pid = self.profile_id(state)
        if pid == GUEST_ID:
            return False
        self.profiles[pid]["settings"] = copy.deepcopy(settings)
        return self._persist_profile(pid)

    # --------------------------------------------------------------- score
    def save_score(self, state, score):
        """Update best_score ONLY for a Bluetooth user. Guest scores are never
        stored anywhere. Returns a small result dict."""
        pid = self.profile_id(state)
        score = int(score)
        if pid == GUEST_ID:
            return {"saved": False, "new_best": False, "best": None, "previous_best": None}
        prof = self.profiles[pid]
        prev = int(prof.get("best_score", 0))
        if score > prev:
            prof["best_score"] = score
            ok = self._persist_profile(pid)
            return {"saved": ok, "new_best": True, "best": score, "previous_best": prev}
        return {"saved": False, "new_best": False, "best": prev, "previous_best": prev}
