# AeroTrack: Aerial CV & Drone Telemetry Engine

AeroTrack is an asynchronous computer vision backend service for detecting, tracking, and analyzing small, fast-moving objects in aerial and drone video footage (e.g. VisDrone). It integrates **YOLOv11** object detection with **ByteTrack** multi-object tracking, spatial tripwire line crossing math, velocity estimation, and a high-performance **FastAPI** telemetry API.

---

## 1. Quickstart & Installation

### Step 1: Create a Python Virtual Environment
We recommend Python 3.10, 3.11, or 3.12:

```bash
# Clone or navigate into the python_backend directory
cd python_backend

# Create virtualenv
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# (Or on Windows)
# .\venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Note: For GPU acceleration with CUDA, install the PyTorch build appropriate for your CUDA driver from [pytorch.org](https://pytorch.org/), then run `pip install -r requirements.txt`.*

---

## 2. Starting the Uvicorn Server

Launch the FastAPI application with Uvicorn:

```bash
# Standard local development (auto-reload on code change)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

For multi-worker production deployment:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Verify the server is up:
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Specification**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 3. Testing the `/api/v1/track` Endpoint

### Method A: Quick Test using `curl`
Upload an `.mp4` video with default tripwire lines:

```bash
curl -X POST "http://localhost:8000/api/v1/track" \
  -H "accept: application/json" \
  -F "file=@/path/to/your_drone_video.mp4;type=video/mp4"
```

### Method B: `curl` with Custom Tripwire Lines and Parameters
Send custom virtual tripwires (e.g., coordinates `(100, 350)` to `(900, 350)`) and GSD (meters per pixel):

```bash
curl -X POST "http://localhost:8000/api/v1/track" \
  -H "accept: application/json" \
  -F "file=@sample_drone.mp4;type=video/mp4" \
  -F 'config_json={
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "ground_sampling_distance_m_per_px": 0.045,
    "drone_fps": 30.0,
    "virtual_lines": [
      {
        "line_id": "corridor_alpha",
        "name": "North-South Expressway Gate",
        "start_x": 120.0,
        "start_y": 420.0,
        "end_x": 980.0,
        "end_y": 420.0,
        "positive_label": "Northbound",
        "negative_label": "Southbound"
      }
    ],
    "generate_annotated_video": false
  }'
```

### Method C: Python Test Script with Synthetic Video Generator
We've included an automated test script (`test_client.py`) that generates a test video with moving vehicles and executes the request:

```bash
python test_client.py
```

### Method D: Testing the RTSP/Stream Endpoint
To ingest a live drone RTSP feed:

```bash
curl -X POST "http://localhost:8000/api/v1/track/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "stream_url": "rtsp://drone-gateway.local:554/live/feed",
    "max_duration_sec": 10,
    "config": {
      "confidence_threshold": 0.3,
      "ground_sampling_distance_m_per_px": 0.05
    }
  }'
```

---

## 4. Pipeline & Spatial Math Details

### A. Small-Object Aerial Detection (YOLOv11)
YOLOv11 offers optimized C3k2 and C2PSA attention blocks designed to preserve small-scale features from high-altitude oblique drone views. ByteTrack associates detections across frames using a Kalman filter.

### B. Virtual Line Intersection (CCW Algorithm)
To test if object motion segment $AB = (P_{t-1}, P_t)$ intersects virtual tripwire $CD = (L_1, L_2)$:
$$\text{ccw}(A, B, C) = (C_y - A_y)(B_x - A_x) > (B_y - A_y)(C_x - A_x)$$
The segments intersect if and only if:
$$(\text{ccw}(A, C, D) \neq \text{ccw}(B, C, D)) \land (\text{ccw}(A, B, C) \neq \text{ccw}(A, B, D))$$

### C. Direction Vector & 2D Cross Product
To determine whether an object is traveling **Northbound** or **Southbound**:
$$\vec{V}_{\text{line}} = (L_{2,x} - L_{1,x}, L_{2,y} - L_{1,y})$$
$$\vec{V}_{\text{motion}} = (P_{t,x} - P_{t-1,x}, P_{t,y} - P_{t-1,y})$$
$$\text{Cross Product} = \vec{V}_{\text{line},x} \cdot \vec{V}_{\text{motion},y} - \vec{V}_{\text{line},y} \cdot \vec{V}_{\text{motion},x}$$
- If $\text{Cross Product} > 0$: Object crossed in the **positive** direction.
- If $\text{Cross Product} < 0$: Object crossed in the **negative** direction.

### D. Velocity Estimation
Using the Ground Sampling Distance (GSD, in meters/pixel) and video framerate:
$$v_{\text{mps}} = \frac{\|\vec{P}_t - \vec{P}_{t-k}\|_2 \cdot \text{GSD}}{\Delta t}, \quad v_{\text{km/h}} = v_{\text{mps}} \times 3.6$$
A moving average window is applied over $k=5$ frames to smooth out centroid jitter caused by perspective skew.
