"""
bluetooth_id.py - AYATLA Bluetooth player identification.

What this process does
----------------------
* Connects to the hub as role "bluetooth" and listens to it in a background
  receiver thread.
* Stays IDLE (no scanning) until the player picks "Sync with Bluetooth" on
  the Player Mode screen, i.e. until the hub tells us mode == "bluetooth".
* Then it scans for nearby Bluetooth Low Energy (BLE) devices in short
  windows and compares them with the registered devices in users.json.
* It reports the result with "bt_status" and "user" messages.

About MAC addresses (important!)
--------------------------------
Modern phones (Android and iPhone) often RANDOMIZE the Bluetooth MAC address
they advertise, and change it every few minutes, for privacy. So a MAC
written in users.json may stop matching later. That's why each user can be
matched by EITHER:
    * "mac"  - the advertised address (works well for fixed devices such as
               smart bands, beacons, headphones), or
    * "name" - the advertised device name (e.g. "Sara-Phone"). Names are
               more stable on phones, but only if the phone actually
               advertises its name (usually only while its Bluetooth
               settings screen is open, or when an app advertises it).
On macOS, bleak does not show real MAC addresses at all: it shows a
per-computer UUID instead. You can put that UUID in the "mac" field.

Run "python bluetooth_id.py --list" to see what your devices advertise.

Threads vs asyncio
------------------
bleak is asyncio-based, but the hub socket uses a normal blocking thread.
The receiver thread never touches asyncio objects directly: it hands each
message to the event loop with loop.call_soon_threadsafe(queue.put_nowait).
"""

import argparse
import asyncio
import json
import os
import socket
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Shared helper (HOST, PORT, send_msg, recv_msgs)
# ---------------------------------------------------------------------------
try:
    from common import HOST, PORT, send_msg, recv_msgs
except ImportError:
    print("[bt] ERROR: common.py not found. Put common.py next to this file.")
    sys.exit(1)

# bleak is imported lazily-safe: if it is missing we still start, report
# "not_found" and explain how to install it.
try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    BleakClient = BleakScanner = None

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
SCAN_WINDOW = 3.0       # seconds per BLE scan window
SCAN_TIMEOUT = 15.0     # give up searching after this many seconds
LOST_AFTER = 20.0       # matched device unseen this long -> user lost
LIST_DURATION = 10.0    # how long --list scans
RETRY_SECONDS = 2.0     # hub reconnect delay
ERROR_PAUSE = 2.0       # small pause after a failed scan (avoid busy loop)

# Hands-free registration (game screen BT_REGISTER): "hold your phone next to
# the laptop" - the nearby devices are listed strongest signal first.
REGISTER_WINDOW = 1.5   # seconds per scan while registering (quick list updates)
REGISTER_TIMEOUT = 120  # stop the registration scan after this many seconds
REGISTER_FORGET = 6.0   # a device not seen for this long leaves the list
MAX_LISTED = 4          # devices offered on screen (fingers 1-4)
NEAR_RSSI = -62         # Tune: signal (dBm) of a phone held next to the laptop
NEAR_MARGIN = 8         # Tune: ...and at least this many dB stronger than the next device

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")


# ===========================================================================
# users.json loading and matching
# ===========================================================================
def normalize_mac(value):
    """'aa-bb-cc-dd-ee-ff' -> 'AA:BB:CC:DD:EE:FF' so comparisons are fair."""
    if not value:
        return None
    return str(value).strip().upper().replace("-", ":")


def normalize_name(value):
    """Names are compared case-insensitively, ignoring outer spaces."""
    if not value:
        return None
    return str(value).strip().lower()


def load_users(path):
    """Read users.json and return a clean list of
    {"id": ..., "mac": ... or None, "name": ... or None}.
    Bad entries are skipped with a warning; a missing file gives []."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[bt] WARNING: {path} not found. No users can be matched.")
        return []
    except (OSError, ValueError) as e:
        print(f"[bt] WARNING: could not read {path}: {e}")
        return []

    users = []
    for entry in data.get("users", []) if isinstance(data, dict) else []:
        try:
            user_id = entry["id"]
            match = entry.get("match", {}) or {}
            mac = normalize_mac(match.get("mac"))
            name = normalize_name(match.get("name"))
        except (KeyError, TypeError, AttributeError):
            print(f"[bt] WARNING: skipping bad users.json entry: {entry}")
            continue
        if not mac and not name:
            print(f"[bt] WARNING: user {user_id} has neither mac nor name, skipped.")
            continue
        users.append({"id": user_id, "mac": mac, "name": name})
    return users


PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content", "profiles")


def register_user(path, address, name):
    """Add a new user to users.json and return its id (user_01, user_02, ...).
    The id is also free in content/profiles, so a new player never inherits
    somebody else's profile. Both the name and the address are stored: the
    name keeps working when a phone changes its address for privacy."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"users": []}
    if not isinstance(data, dict) or not isinstance(data.get("users"), list):
        raise ValueError("users.json has an unexpected format")
    taken = {str(u.get("id")) for u in data["users"] if isinstance(u, dict)}
    n = 1
    while f"user_{n:02d}" in taken or os.path.exists(os.path.join(PROFILES_DIR, f"user_{n:02d}.json")):
        n += 1
    user_id = f"user_{n:02d}"
    match = {"mac": normalize_mac(address), "name": name}
    data["users"].append({"id": user_id, "match": match})
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)                        # never leaves half a file behind
    return user_id


def find_best_match(users, seen):
    """Compare discovered devices with registered users.

    seen = {address: {"name": str or None, "rssi": int}}
    Returns (user_id, rssi, device name) for the registered user with the
    STRONGEST signal (RSSI closest to 0 = nearest), or None if nobody matched.
    """
    best = None
    for address, info in seen.items():
        mac = normalize_mac(address)
        name = normalize_name(info["name"])
        for user in users:
            mac_ok = user["mac"] is not None and user["mac"] == mac
            name_ok = user["name"] is not None and user["name"] == name
            if mac_ok or name_ok:
                if best is None or info["rssi"] > best[1]:
                    best = (user["id"], info["rssi"], info["name"])
    return best


# ===========================================================================
# BLE scanning
# ===========================================================================
class BluetoothUnavailable(Exception):
    """Raised when Bluetooth is off, missing, or bleak is not installed."""


async def scan_window(seconds):
    """Scan for `seconds` and return {address: {"name", "rssi"}}.

    Raises BluetoothUnavailable for any adapter / driver problem."""
    if BleakScanner is None:
        raise BluetoothUnavailable("bleak is not installed (pip install bleak)")
    try:
        # return_adv=True gives us the advertisement data (name + RSSI).
        found = await BleakScanner.discover(timeout=seconds, return_adv=True)
    except asyncio.CancelledError:
        raise                      # normal: someone stopped the scan
    except Exception as e:         # BleakError, OSError, WinRT errors, ...
        raise BluetoothUnavailable(str(e) or type(e).__name__) from e

    seen = {}
    for address, (device, adv) in found.items():
        _devices[address] = device
        seen[address] = {"name": adv.local_name or device.name, "rssi": adv.rssi, "paired": False}
    # Most phones broadcast WITHOUT a name. Windows still knows the name of a
    # phone paired with this laptop (even behind its changing address).
    unnamed = [a for a, info in seen.items() if not info["name"]]
    for address, (name, paired) in zip(unnamed, await asyncio.gather(*(windows_name(a) for a in unnamed))):
        if name:
            seen[address]["name"], seen[address]["paired"] = name, paired
    return seen


# ---------------------------------------------------------------------------
# Getting a device's name when it does not broadcast one
# ---------------------------------------------------------------------------
DEVICE_NAME_UUID = "00002a00-0000-1000-8000-00805f9b34fb"   # GATT "Device Name"
_known_names = {}   # address -> (name Windows knows or None, paired?), cached
_devices = {}       # address -> BLEDevice from the scans (to connect to it)


async def windows_name(address):
    """(name, paired) that Windows has stored for this address, e.g. ("Anas",
    True) for a phone paired with this laptop - Windows recognises a paired
    phone even behind its changing privacy address. (None, False) for unknown
    devices (Windows only has a placeholder like "Bluetooth 3c:8b:...") or
    when not on Windows."""
    if address in _known_names:
        return _known_names[address]
    result = (None, False)
    try:
        from winrt.windows.devices.bluetooth import BluetoothLEDevice
        dev = await BluetoothLEDevice.from_bluetooth_address_async(int(address.replace(":", ""), 16))
        if dev is not None and dev.name and not dev.name.lower().startswith("bluetooth "):
            try:
                paired = bool(dev.device_information.pairing.is_paired)
            except Exception:
                paired = False
            result = (dev.name, paired)
    except Exception:
        pass                        # not Windows, winrt missing, or not a MAC (macOS UUID)
    _known_names[address] = result
    return result


def player_name(device):
    """Device name -> profile name: without technical endings like "_BLE"."""
    if not device:
        return None
    name = device.strip()
    for ending in ("_BLE", " BLE", "-BLE", "_LE", " LE"):
        if name.upper().endswith(ending):
            name = name[: -len(ending)].rstrip()
    return name[:24] or None


async def device_name(address, timeout=6.0):
    """Ask the device itself for its name (GATT "Device Name"). Takes a few
    seconds and only works if the device accepts a connection."""
    if BleakClient is None:
        return None
    try:
        async with BleakClient(_devices.get(address, address), timeout=timeout) as client:
            raw = await client.read_gatt_char(DEVICE_NAME_UUID)
        return raw.decode("utf-8", "replace").strip("\x00 ").strip() or None
    except Exception:
        return None


def explain_bt_problem(error):
    print(f"[bt] Bluetooth is not available: {error}")
    print("[bt]   - Is Bluetooth turned ON in your OS settings?")
    print("[bt]   - Does this computer have a Bluetooth 4.0+ (BLE) adapter?")
    print("[bt]   - Is bleak installed?  pip install bleak")
    print("[bt]   - Linux: is the bluetooth service running? (systemctl status bluetooth)")


# ===========================================================================
# Outputs: real hub connection, or printing (--no-hub)
# ===========================================================================
class HubClient:
    """Keeps a TCP connection to the hub alive in a background thread.

    * Reconnects every RETRY_SECONDS if the hub is missing or drops.
    * Every received message is passed to on_message(msg), called from the
      receiver thread (the caller must make that thread-safe).
    """

    def __init__(self, on_message):
        self.on_message = on_message
        self.sock = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        warned = False
        while True:
            sock = None
            try:
                sock = socket.create_connection((HOST, PORT), timeout=5)
                sock.settimeout(None)               # blocking reads from now on
                send_msg(sock, {"type": "hello", "role": "bluetooth"})
                with self.lock:
                    self.sock = sock
                print(f"[bt] connected to hub at {HOST}:{PORT}")
                warned = False
                for msg in recv_msgs(sock):         # ends on disconnect
                    self.on_message(msg)
                print("[bt] hub disconnected, retrying...")
            except OSError as e:
                if not warned:
                    print(f"[bt] cannot reach hub ({e}); retrying every {RETRY_SECONDS:.0f} s...")
                    warned = True
            except Exception as e:                  # never let the thread die
                print(f"[bt] receiver error: {e}")
            finally:
                with self.lock:
                    self.sock = None
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            time.sleep(RETRY_SECONDS)

    def send(self, obj):
        with self.lock:
            sock = self.sock
        print(f"[bt] -> {json.dumps(obj)}")
        if sock is None:
            print("[bt]    (hub offline, message dropped)")
            return
        try:
            send_msg(sock, obj)
        except OSError as e:
            print(f"[bt]    send failed: {e}")


class PrintOutput:
    """--no-hub: just print what we would send."""

    def send(self, obj):
        print(f"[bt] (no-hub) would send: {json.dumps(obj)}")


# ===========================================================================
# The identification logic
# ===========================================================================
class BluetoothIdentifier:
    """Reacts to hub messages and runs at most ONE scan task at a time."""

    def __init__(self, output, users_path):
        self.out = output
        self.users_path = users_path
        self.users = []
        self.scan_task = None        # asyncio.Task while scanning, else None
        self.current_user = None     # matched user id, or None
        self.last_seen = 0.0         # time.monotonic() of last sighting
        self.last_mode = None        # last mode we heard about
        self.reg_task = None         # asyncio.Task while registering a device
        self.reg_seen = {}           # address -> {"name", "rssi": [last values], "last"}

    # ---------------- messages from the hub ----------------
    def handle(self, msg):
        kind = msg.get("type")
        if kind == "bt_register":
            action = msg.get("action")
            if action == "start":
                self.start_registration()
            elif action == "cancel":
                self.stop_registration()
            elif action == "confirm":
                self.confirm_registration(msg.get("address"))
            elif action == "pick":                   # --no-hub testing: "pick 1"
                listed = self._device_list()["devices"]
                i = int(msg.get("index", 1)) - 1
                if 0 <= i < len(listed):
                    self.confirm_registration(listed[i]["address"])
        elif kind == "mode":
            # An explicit mode message = the player just chose an option.
            # "bluetooth" again after a failure means "retry".
            self._on_mode(msg.get("value"), explicit=True)
        elif kind == "state":
            # State is broadcast on EVERY change (context, user, ...).
            # Only react when the mode actually changed, otherwise we would
            # restart scanning on every screen change.
            self._on_mode(msg.get("mode"), explicit=False)

    def _on_mode(self, mode, explicit):
        changed = mode != self.last_mode
        self.last_mode = mode
        if mode == "bluetooth" and self.is_registering():
            return                       # registering a device: keep that scan
        if mode != "bluetooth":
            self.stop_registration()
        if mode == "bluetooth":
            if (explicit or changed) and not self.is_scanning():
                self.start()
        else:  # "guest" or None
            if self.is_scanning() or self.current_user is not None:
                print(f"[bt] mode is {mode!r}: stopping, forgetting match.")
            # The hub already clears the user, so we send nothing here.
            self.stop()

    # ---------------- start / stop ----------------
    def is_scanning(self):
        return self.scan_task is not None and not self.scan_task.done()

    def start(self):
        self.users = load_users(self.users_path)   # re-read: edits apply on retry
        self.current_user = None
        print(f"[bt] starting search ({len(self.users)} registered users)...")
        self.scan_task = asyncio.create_task(self._scan_loop())

    def stop(self):
        if self.is_scanning():
            self.scan_task.cancel()
        self.scan_task = None
        self.current_user = None

    # ---------------- result helpers ----------------
    def _report_not_found(self):
        self.out.send({"type": "bt_status", "status": "not_found"})
        self.out.send({"type": "user", "id": None})
        self.current_user = None

    # ---------------- hands-free registration ----------------
    def is_registering(self):
        return self.reg_task is not None and not self.reg_task.done()

    def start_registration(self):
        self.stop()                                  # no normal search meanwhile
        self.stop_registration()
        self.users = load_users(self.users_path)
        self.reg_seen = {}
        print("[bt] registration: listing nearby devices...")
        self.reg_task = asyncio.create_task(self._register_loop())

    def stop_registration(self):
        if self.is_registering():
            self.reg_task.cancel()
            print("[bt] registration stopped.")
        self.reg_task = None

    async def _register_loop(self):
        started = time.monotonic()
        try:
            while time.monotonic() - started < REGISTER_TIMEOUT:
                try:
                    seen = await scan_window(REGISTER_WINDOW)
                except BluetoothUnavailable as e:
                    explain_bt_problem(e)
                    self.out.send({"type": "bt_devices", "status": "error", "message": str(e), "devices": []})
                    return
                now = time.monotonic()
                for address, info in seen.items():
                    d = self.reg_seen.setdefault(address, {"name": None, "paired": False, "rssi": [], "last": now})
                    d["name"] = info["name"] or d["name"]
                    d["paired"] = info.get("paired", False) or d["paired"]
                    d["rssi"] = (d["rssi"] + [info["rssi"]])[-3:]     # smooth over 3 scans
                    d["last"] = now
                self.reg_seen = {a: d for a, d in self.reg_seen.items() if now - d["last"] < REGISTER_FORGET}
                self.out.send(self._device_list())
            self.out.send({"type": "bt_devices", "status": "timeout", "devices": []})
        except asyncio.CancelledError:
            pass

    def _device_list(self):
        """The MAX_LISTED strongest devices; the first is "nearest" when it is
        clearly closer than all others (phone held next to the laptop)."""
        ranked = sorted(self.reg_seen.items(), key=lambda kv: -sum(kv[1]["rssi"]) / len(kv[1]["rssi"]))
        devices = []
        for address, d in ranked[:MAX_LISTED]:
            match = find_best_match(self.users, {address: {"name": d["name"], "rssi": 0}})
            devices.append({"address": address, "name": d["name"], "paired": d["paired"],
                            "rssi": round(sum(d["rssi"]) / len(d["rssi"])),
                            "registered": match[0] if match else None})
        nearest = None
        if devices and devices[0]["rssi"] >= NEAR_RSSI and (
                len(devices) == 1 or devices[0]["rssi"] - devices[1]["rssi"] >= NEAR_MARGIN):
            nearest = devices[0]["address"]
        return {"type": "bt_devices", "status": "scanning", "devices": devices, "nearest": nearest}

    def confirm_registration(self, address):
        d = self.reg_seen.get(address)
        if d is None:
            self.out.send({"type": "bt_devices", "status": "error", "devices": [],
                           "message": "that device is not visible any more - try again"})
            return
        self.stop_registration()
        asyncio.create_task(self._confirm(address, d))

    async def _confirm(self, address, d):
        visible_name = (d["name"] or "").strip() or None     # broadcast or known to Windows
        match = find_best_match(load_users(self.users_path), {address: {"name": visible_name, "rssi": 0}})
        name = visible_name
        if name is None and not match:
            print(f"[bt] {address} has no visible name: asking the device...")
            name = await device_name(address)
        new = not match
        if match:
            user_id = match[0]                       # already registered: just log in
            print(f"[bt] {address} is already registered as {user_id}")
        else:
            try:
                user_id = register_user(self.users_path, address, name)
            except (OSError, ValueError) as e:
                print(f"[bt] could not save users.json: {e}")
                self.out.send({"type": "bt_devices", "status": "error", "devices": [],
                               "message": "could not save users.json"})
                return
            print(f"[bt] registered {address} ({name or 'no name'}) as {user_id}")
        number = "".join(ch for ch in user_id if ch.isdigit()).lstrip("0") or "1"
        display = player_name(name) or f"Player {number}"
        # "name": found again by name next time (survives address changes);
        # "mac": only by this address, which phones change for privacy.
        self.out.send({"type": "bt_devices", "status": "registered", "devices": [], "new": new,
                       "id": user_id, "name": display, "matched_by": "name" if visible_name else "mac"})
        # Logged in straight away, then tracked like after a normal sync.
        self.users = load_users(self.users_path)
        self.current_user, self.last_seen = user_id, time.monotonic()
        self.out.send({"type": "bt_status", "status": "found", "name": display})
        self.out.send({"type": "user", "id": user_id})
        self.scan_task = asyncio.create_task(self._scan_loop(resume=True))

    # ---------------- main scan loop ----------------
    async def _scan_loop(self, resume=False):
        """resume=True: the user is already identified (just registered), so
        go straight to tracking them."""
        try:
            if not resume:
                self.out.send({"type": "bt_status", "status": "scanning"})
            started = time.monotonic()

            while True:
                try:
                    seen = await scan_window(SCAN_WINDOW)
                except BluetoothUnavailable as e:
                    explain_bt_problem(e)
                    self._report_not_found()
                    await asyncio.sleep(ERROR_PAUSE)
                    return                       # wait for a new "bluetooth"

                now = time.monotonic()
                match = find_best_match(self.users, seen)

                if self.current_user is None:
                    # ---- Phase 1: SEARCHING ----
                    if match:
                        user_id, rssi, device = match
                        self.current_user = user_id
                        self.last_seen = now
                        print(f"[bt] matched {user_id} (RSSI {rssi} dBm)")
                        # The device's name lets the game name a new profile.
                        self.out.send({"type": "bt_status", "status": "found", "name": device})
                        self.out.send({"type": "user", "id": user_id})
                    elif now - started >= SCAN_TIMEOUT:
                        print(f"[bt] no registered device within {SCAN_TIMEOUT:.0f} s.")
                        self._report_not_found()
                        return                   # idle until player retries
                    else:
                        left = SCAN_TIMEOUT - (now - started)
                        print(f"[bt] {len(seen)} devices seen, no match yet ({left:.0f} s left)")
                else:
                    # ---- Phase 2: TRACKING the matched user ----
                    # We stay with the user we picked, even if another
                    # registered device appears with a stronger signal.
                    if self._user_seen(self.current_user, seen):
                        self.last_seen = now
                    elif now - self.last_seen >= LOST_AFTER:
                        print(f"[bt] {self.current_user} not seen for {LOST_AFTER:.0f} s: lost.")
                        self._report_not_found()
                        return                   # idle until player retries
        except asyncio.CancelledError:
            pass                                 # stopped by guest / null mode

    def _user_seen(self, user_id, seen):
        """True if any device in `seen` matches this specific user."""
        only_this = [u for u in self.users if u["id"] == user_id]
        return find_best_match(only_this, seen) is not None


# ===========================================================================
# --list mode
# ===========================================================================
async def list_devices():
    print(f"[bt] scanning for {LIST_DURATION:.0f} s...")
    try:
        seen = await scan_window(LIST_DURATION)
    except BluetoothUnavailable as e:
        explain_bt_problem(e)
        return
    if not seen:
        print("[bt] no devices found. Open your phone's Bluetooth settings screen and try again.")
        return
    print(f"\n{'NAME':<28} {'ADDRESS (MAC)':<40} RSSI")
    print("-" * 76)
    for address, info in sorted(seen.items(), key=lambda kv: kv[1]["rssi"], reverse=True):
        name = (info["name"] + (" [paired]" if info.get("paired") else "")) if info["name"] else "(no name)"
        print(f"{name:<28} {address:<40} {info['rssi']} dBm")
    print("\nCopy a MAC or a name into users.json. Strongest signal (nearest) is at the top.")


# ===========================================================================
# Startup
# ===========================================================================
def start_stdin_commands(deliver):
    """--no-hub helper: type 'bluetooth', 'guest' or 'null' to fake mode messages,
    'register' to list nearby devices and 'pick N' to register device N."""
    def reader():
        print("[bt] (no-hub) type: bluetooth | guest | null | register | pick N   (Ctrl+C to quit)")
        for line in sys.stdin:
            word = line.strip().lower()
            if word == "register":
                deliver({"type": "bt_register", "action": "start"})
            elif word.startswith("pick") and word[4:].strip().isdigit():
                deliver({"type": "bt_register", "action": "pick", "index": int(word[4:])})
            elif word in ("bluetooth", "guest"):
                deliver({"type": "mode", "value": word})
            elif word in ("null", "none"):
                deliver({"type": "state", "mode": None})
            elif word:
                print("[bt] unknown command:", word)
    threading.Thread(target=reader, daemon=True).start()


async def run(args):
    loop = asyncio.get_running_loop()
    inbox = asyncio.Queue()

    def deliver(msg):
        # Called from OTHER threads -> hand the message to the event loop.
        loop.call_soon_threadsafe(inbox.put_nowait, msg)

    if args.no_hub:
        output = PrintOutput()
        start_stdin_commands(deliver)
    else:
        output = HubClient(deliver)

    identifier = BluetoothIdentifier(output, args.users)

    if args.simulate_mode == "bluetooth":
        print("[bt] simulating: mode = bluetooth")
        identifier.handle({"type": "mode", "value": "bluetooth"})

    while True:
        msg = await inbox.get()
        try:
            identifier.handle(msg)
        except Exception as e:      # a weird message must never kill us
            print(f"[bt] ignored message {msg}: {e}")


def main():
    parser = argparse.ArgumentParser(description="AYATLA Bluetooth identification")
    parser.add_argument("--list", action="store_true",
                        help="scan once and print nearby devices, then exit")
    parser.add_argument("--no-hub", action="store_true",
                        help="print messages instead of sending them to the hub")
    parser.add_argument("--simulate-mode", choices=["bluetooth"],
                        help="act as if this mode message arrived at startup")
    parser.add_argument("--users", default=USERS_FILE,
                        help="path to users.json (default: next to this file)")
    args = parser.parse_args()

    try:
        if args.list:
            asyncio.run(list_devices())
        else:
            asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[bt] bye.")


if __name__ == "__main__":
    main()
