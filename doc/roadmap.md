# Project 9: Physical Task Guidance System - Development Roadmap

This roadmap outlines the step-by-step development strategy for building the Physical Task Guidance System. This project focuses on a Python/OpenCV application that uses Augmented Reality (AR) to guide users through physical sorting tasks.

## 1. Phase I: Foundation & Perception
**Goal**: Establish the "eyes" of the system. The application must reliably detect the workspace and objects within it.

- [ ] **Environment Setup**
    - Initialize Python project with `opencv-python`, `ultralytics` (YOLOv8), `mediapipe`, and `numpy`.
    - Setup module structure: `perception`, `state`, `viz`, `main.py`.

- [ ] **Surface Tracking (The Workspace)**
    - Implement ArUco marker detection (4 corners).
    - Develop `SurfaceTracker` class to compute Homography Matrix.
    - **Deliverable**: A live feed drawing a virtual grid specifically aligned with the table surface.

- [ ] **Object Detection**
    - Integrate YOLOv8 to detect: Cup, Bottle, Plate/Bowl.
    - Implement logic to map 2D screen bounding boxes to 2D table coordinates $(u, v)$ using the Homography matrix.
    - **Deliverable**: Console output showing accurate table coordinates of objects as they move.

---

## 2. Phase II: Core Logic & State Management
**Goal**: Give the system a "brain". It needs to understand *what* to do and *when* a step is complete.

- [ ] **Procedure Definition**
    - Define a `Procedure` class containing a sequence of `Step` objects.
    - Define target regions for each step (e.g., "Move Cup to Top-Left").

- [ ] **State Engine**
    - Implement the logic loop:
        1. Get current object positions from Perception.
        2. Check if the *active* object is inside the *target* region.
        3. If yes (and stable for $N$ frames), mark step as **Complete** and advance.
    - **Deliverable**: A text-based debug printout that advances through steps 1->2->3 as you move objects correctly.

---

## 3. Phase III: Visualization & AR Feedback
**Goal**: communicate with the User. Replace debug text with intuitive visual overlays.

- [ ] **Perspective Warping**
    - Implement `warp_perspective` rendering to draw target zones (circles/squares) that appear *flat on the table*.
    - Ensure drawings strictly follow the ArUco-defined plane.

- [ ] **Guidance System**
    - Draw dynamic arrows connecting the object's center to the target's center.
    - Change colors based on state: 
        - **Red**: Object detected but in wrong place.
        - **Green**: Object correctly placed.

- [ ] **User Interface (HUD)**
    - Add top-banner text instructions: "Step 1: Move the Red Cup".
    - Add success messages: "Great Job!", "Table Set Successfully".

---

## 4. Phase IV: Testing & Refinement
**Goal**: Polish the prototype for robustness.

- [ ] **Occlusion Handling**
    - Ensure system doesn't crash if markers/objects are briefly covered by hands.
    - Add "lost tracking" warning if markers disappear.
    
- [ ] **Hand Tracking (Bonus)**
    - Integrate MediaPipe Hands to detect user interaction.
    - Pause "Guidance" or highlight items if hand is detected nearby.

- [ ] **Final Verification**
    - Full walkthrough of the sorting task (Cup -> Bottle -> Plate).
    - Validate that calibration works at different camera angles.
