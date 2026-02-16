# Physical Task Guidance System (TGS)

An AR webcam application that guides users through a physical sorting task. Uses ArUco markers to detect a table surface, YOLOv8 to detect objects (cup, bottle, bowl), and draws real-time AR overlays to guide the user to place each object in the correct zone.

## Requirements

- Python 3.9+
- A webcam
- A printer (for ArUco markers)
- Three objects: a **cup**, a **bottle**, and a **bowl**

## Setup

### 1. Clone and enter the project

```bash
cd /path/to/project9_TGS
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs OpenCV, YOLOv8 (ultralytics), MediaPipe, NumPy, and pytest. The YOLOv8 nano model (`yolov8n.pt`) will download automatically on first run (~6MB).

### 3. Generate and print ArUco markers

```bash
python scripts/generate_markers.py
```

This creates a `markers/` folder with individual marker PNGs and a combined `marker_sheet.png`. Print the sheet at **actual size** (no fit-to-page scaling), then cut out the 4 markers.

Each marker is labeled:

| Marker ID | Placement       |
|-----------|-----------------|
| 0         | Top-Left corner |
| 1         | Top-Right corner |
| 2         | Bottom-Right corner |
| 3         | Bottom-Left corner |

### 4. Set up your workspace

1. Place the 4 markers face-up at the corners of a flat surface (desk, table, etc.)
2. Position your webcam so all 4 markers are visible in the frame
3. Place the cup, bottle, and bowl somewhere on or near the table

## Running the App

```bash
source .venv/bin/activate
python -m tgs.main
```

A window will open showing your camera feed. Once all 4 markers are detected, the system draws a grid on the table and shows target zones where each object should be placed.

Follow the instructions in the top banner (e.g., "Place the Cup in Zone A"). Move each object into its highlighted zone and hold it there for ~0.5 seconds to complete each step.

## Keyboard Controls

| Key     | Action                                  |
|---------|-----------------------------------------|
| `q`/ESC | Quit                                    |
| `r`     | Reset the task                          |
| `n`     | New randomized task (different zone positions) |
| `c`     | Recalibrate (re-detect markers)         |
| `d`     | Toggle debug info (FPS counter)         |
| `m`     | Toggle MediaPipe hand tracking          |

## Troubleshooting

**Camera doesn't open** — Edit `tgs/config.py` and change `index: int = 0` to `1` or `2`.

**Markers not detected** — Ensure all 4 are fully visible, avoid glare, and move the camera closer if the markers appear too small.

**Objects not detected** — Ensure good lighting. If detections are inconsistent, lower the confidence threshold in `tgs/config.py` (`confidence_threshold: float = 0.4`).

**Low FPS** — Set `input_size: int = 320` in `tgs/config.py` for faster YOLO inference at the cost of some accuracy.

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Project Structure

```
tgs/
  config.py              # All configuration and calibration persistence
  main.py                # App entry point and main loop
  perception/
    surface_tracker.py   # ArUco marker detection and homography
    object_detector.py   # Threaded YOLOv8 inference
    hand_tracker.py      # Optional MediaPipe hand tracking
  state/
    procedure.py         # Task/step data classes
    engine.py            # Step advancement with debounce and grace periods
  viz/
    renderer.py          # AR overlay drawing (grid, zones, arrows, HUD)
    colors.py            # Color constants
tests/                   # 47 unit tests
scripts/
  generate_markers.py    # ArUco marker generation utility
```
