# Vision module

## Files
- `vision.py`: runtime webcam process, MediaPipe Pose + Hands, DollarPy, hub client.
- `record_templates.py` (or double-click `Recorder.bat`): the Gesture Recorder app - one window to record samples for every gesture.
- `templates/*.json`: recorded samples, one file per gesture. The earlier hip-based
  recordings are kept in `templates/old_full_body/` and are not loaded.

## Install
vision.py uses MediaPipe's `mp.solutions` API, which was removed in MediaPipe 0.10.30+,
so the project pins `mediapipe==0.10.21` in its own virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Record samples (Gesture Recorder app)
Close the game first (both need the camera), then double-click `Recorder.bat` or run:
```bash
python record_templates.py              # optional: --camera 1, --person NAME, or a gesture name to preselect
```
Pick a gesture in the list, press **Record** (or SPACE), get ready during the 3-second countdown, then perform it during the recording. The window shows the live camera with your skeleton, sample counts per gesture, what went wrong if a sample is rejected, and a **Live test** (T) that shows what vision recognizes right now (DollarPy scores, rules vs. your samples, distances). **Delete last** (D) / **Delete all** remove bad samples.

| Gesture | How | What the samples do |
|---|---|---|
| `jump` | jump, or raise BOTH arms above your head | DollarPy templates (required: body movements need samples) |
| `thumbs_up` (pauses the game) | a fist with the thumb pointing straight up (the hand may be sideways) | Optional: your examples; a close match wins, otherwise the built-in rule decides |
| `open_palm`, `ok_sign`, `fingers_1`..`fingers_4` | show the sign with a raised hand (fingers up) | Optional: your examples; a close match wins, otherwise the built-in rules decide |
| `hold_x` | cross your forearms in front of your chest | Optional: same as hand signs |
| `object_power_up` | hold the object so it fills the pink box | Learns your object's colour (used with `--enable-object`; default: bright yellow) |

For body movements, **stand still in your normal position during the countdown** (this teaches it your neutral position); only the upper body is needed. Samples are ADDED to the existing ones, so you can record more in later sessions and with other people. A sample is rejected (with the reason shown) if, e.g., the movement is too small or goes the wrong way, the hand is not raised, the wrists are not together for the X, or the box contains several colours. Recommended: 8+ samples per body movement, 5+ per hand sign / X pose (only for the ones that are not detected well), 3+ for the object, ideally from several people. Restart vision / the game afterwards so it loads the new samples.

## Run
```bash
python vision.py
python vision.py --no-hub
python vision.py --camera 1
python vision.py --enable-object
```
Press `q` in the debug window to quit. To run the whole system at once, use `run_ayatla.py` / `AYATLA.bat`.

## Hub contract
Uses the AYATLA shared contract (host/port from `common.py`):
- TCP `127.0.0.1:5000` (override with `VISION_HUB_HOST` / `VISION_HUB_PORT`), JSON Lines, bidirectional.
- First message after connecting: `{"type":"hello","role":"vision"}`.
- Incoming: `{"type":"state","context":"CALIBRATION","settings":{...},...}` and `{"type":"settings","camera_index":0,"gesture_sensitivity":0.72,"hold_time":2.0,"left_handed":false}`.
- Outgoing:
  - `{"type":"gesture","name":"jump","score":0.8,"repeat":false}`
  - `{"type":"calibration","camera":true,"body":true,"hands":false,"progress":0.4,"seconds_left":1.8,"done":false}`
  - `{"type":"tracking","status":"lost"}` / `found`

The wire format is isolated in `HubClient._connect()`, `HubClient._receiver()` and `HubClient.send()`. The camera setting can be overridden with `--camera N`; that override takes precedence over incoming camera settings.

## Squat / jump line
Squats and jumps need no recording. A horizontal line at `SQUAT_LINE` (30%) of the picture height is used for both:
- **Squat:** the head (nose) drops below the line for 0.1 s. Sent once; the head must come back above the line (minus a small margin) before the next squat counts, so holding a crouch is one slide.
- **Jump:** BOTH hands (pose wrist/hand points) go above the line for 0.1 s. Sent once; a hand must come back below the line first. One raised hand never counts, because that is a hand sign (OK, fingers, thumbs up).

Calibration only completes when the head is above the line, and the neutral height is not learned while crouched (so standing back up never looks like a jump). The DollarPy jump classifier (recorded `jump` samples, the course's movement classifier) still runs as a second way to detect a jump; the line and DollarPy share `JUMP_COOLDOWN` (0.8 s) so one movement gives one jump. The vision debug window and the recorder's Live test draw the line (green, red with SQUAT / JUMP). Tune `SQUAT_LINE`, `SQUAT_STABLE_SECONDS`, `SQUAT_REARM_MARGIN` and `JUMP_COOLDOWN` in `vision.py`. The old DollarPy squat and lean recordings are no longer used.

## Lanes (thirds of the camera)
Lanes need no recording. The camera picture is split into three equal thirds (mirrored, so the left third is the player's own left). The third that holds more than half of the player's visible landmarks is their zone; vision sends `zone_left` / `zone_center` / `zone_right` when it changes (repeat:false) and again every second (repeat:true). The game puts the runner in that lane (absolute, so it can never get out of sync), starts each run in the player's current lane, and uses a step into the left/right third on Player Mode (Bluetooth / Guest) and Settings (decrease / increase, back to the middle between steps). The vision debug window draws the thirds with each one's share of the body. Tune with `ZONE_MAJORITY` (0.5) and `ZONE_STABLE_SECONDS` (0.15 s) above `ZoneTracker` in `vision.py`. The old DollarPy lean recordings are no longer used.

## Tuning
- Body movements (constants above `UpperBodyFeatures` in `vision.py`):
  - `MIN_MOTION` (0.25 shoulder widths): head + shoulders must move at least this much. Lower it if small leans/crouches are missed.
  - `ARMS_UP_MIN` (0.4): how far both wrists must be above the nose for "arms up" to count as jump.
  - `NEUTRAL_RADIUS` (0.2): a movement must start near your neutral position (prevents the way back from a crouch counting as a jump).
  - `BODY_COOLDOWN` (0.8 s), `STILL_SPEED`, `BASELINE_SECONDS`.
  - The debug window's `Body:` line shows the live status and DollarPy score.
- DollarPy threshold: `sensitivity_to_threshold()` in `vision.py`. It maps sensitivity 0..1 to score threshold 0.78..0.45.
- Hand debounce: `sensitivity_to_stable_frames()` maps sensitivity 0..1 to 12..5 frames.
- Hand finger extension: adjust the `0.015` y-margin in `hand_pose_name()`.
- Thumb extension: adjust `0.48` palm-width and `0.16` thumb-joint distance thresholds.
- OK sign: adjust `0.32` normalized tip distance.
- Hold X: adjust wrist distance (`0.55` shoulder widths), chest proximity (`0.8`), and crossing rule in `HoldXDetector`. The chest point is derived from the nose and shoulders, so hips are not needed.
- Calibration: needs the nose, both shoulders and a hand; adjust visibility `0.65` and stillness movement `0.006`.
- Tracking: body lost timeout is `1.0s`; found confirmation is `0.5s`.
- Recorded hand signs / X pose: `HAND_MATCH_DIST` (0.22 palm lengths) and `XPOSE_MATCH_DIST` (0.30 shoulder widths) are how close a live hand/pose must be to one of your samples. The recorder's Live test shows the distances.
- Object: recorded colours are loaded from `templates/object_power_up.json`; without samples the default HSV yellow bounds in `ObjectPowerUpDetector` are used. The minimum contour area is there too.

## Notes
- The input image is mirrored for selfie view; MediaPipe handedness labels are interpreted in that mirrored selfie-view convention.
- DollarPy is a 2D point-cloud recognizer. Only x/y are passed; each landmark trajectory receives a distinct stroke ID.
- Features (upper body, "v2"): nose, shoulder midpoint and both wrists, measured from the player's NEUTRAL shoulder midpoint (learned while still) and divided by the neutral shoulder width; x is multiplied by the frame aspect ratio so both axes use the same units. Whole-body moves (jump, crouch, lean/step) stay visible and camera distance does not matter. Template files store this layout in `coordinate_system`; files with another layout are skipped.
- DollarPy only runs after a cheap check that the head and shoulders really moved (or both arms went up), so standing still, a raised hand for a finger sign, or crossed arms never trigger body gestures.
- DollarPy's $P point-cloud matching does not preserve point order, so e.g. a jump (up then down) and a squat (down then up) can look alike. The measured main direction (up = jump, down = squat, left/right = lean) settles this; the movement must still reach the DollarPy score threshold against that gesture's templates.
- After a body gesture is sent the history is cleared and the next movement has to start from the neutral position, so one movement produces one gesture.
- dollarpy's `recognize()` inserts resampled points into the template lists it receives, so `vision.py` passes fresh copies of the templates on every call (otherwise they grow and recognition slows down over time). Coordinates are shifted by +10 because dollarpy's scaling step assumes non-negative values.
- Vision intentionally does not filter by game screen/context except pausing gameplay gestures during calibration. The game remains responsible for deciding the meaning of gesture messages.
