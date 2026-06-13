# Magic Lens AR 

A real-time, gesture-controlled Augmented Reality (AR) interface built with Python, OpenCV, and MediaPipe. 

This project uses advanced hand-tracking, dynamic distance thresholds, and geometric polygon masking (Convex Hulls) to create a flexible, interactive "AR Window" stretched between the user's hands. Different finger combinations trigger unique computer vision filters inside the lens.


##  Features

* **Dynamic Flexible Tracking:** Utilizes `cv2.convexHull` to wrap a perfect, non-collapsing polygon around the user's fingertips, regardless of how the hands are angled.
* **Smart Gesture Recognition:** Combines distance-based thresholds (Pythagorean Euclidean distance) with state-based joint tracking (Y-axis spatial mapping) to prevent false-positive filter triggers.
* **4 Unique AR Lenses:**
  * **Lens 1: Grayscale Contrast** (Trigger: Thumbs + Indexes touching)
  * **Lens 2: Cyber/Matrix** (Trigger: Indexes + Middles touching) - *Isolates and over-amplifies the green color channel.*
  * **Lens 3: Bone-Frost** (Trigger: Middles + Rings touching) - *Applies a high-saturation custom color map.*
  * **Lens 4: Inverted X-Ray** (Trigger: State-based custom gesture - Pinkies UP, Rings DOWN) - *A comfort-focused gesture override that bypasses distance thresholds.*

##  Tech Stack

* **Python 3.11** (Recommended for C++ package stability)
* **OpenCV** (`opencv-python`) - Core image processing, matrix math, and AR masking.
* **Google MediaPipe** (`mediapipe`) - High-speed skeletal hand tracking.
* **NumPy** (`numpy`) - Array manipulation and image masking logic.

##  Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/Sed-J05/hand-tracked-lens-in-AR.git
cd hand-tracked-lens-in-AR
```

**2. For running it, first (in terminal inside the project folder):**
```bash
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate
```

**3. Next install dependencies (inside the same folder's terminal) :**
```bash
pip install opencv-python mediapipe numpy
```

**4. Finally, to run it, simply (inside the same folder's terminal) :**
```bash
python magic_lens.py
```


