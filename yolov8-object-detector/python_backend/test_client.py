"""
AeroTrack - Test Client & Synthetic Benchmark Generator
File: test_client.py
Description:
    Generates a synthetic aerial drone video clip with moving vehicles,
    and executes a benchmark test against the running FastAPI /api/v1/track endpoint.
"""

import json
import os
import sys
import time
import cv2
import numpy as np
import requests


def generate_synthetic_drone_video(
    filename: str = "synthetic_drone_test.mp4",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_sec: int = 5,
) -> str:
    """Creates a synthetic drone video featuring simulated vehicles crossing tripwires."""
    print(f"[*] Generating {duration_sec}s synthetic drone video: {filename} ({width}x{height} @ {fps}fps)...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    total_frames = fps * duration_sec

    # Simulated vehicles with starting positions and velocities
    vehicles = [
        {"id": 1, "x": 300, "y": 100, "vx": 0, "vy": 6, "color": (50, 180, 240), "w": 40, "h": 70},  # Southbound
        {"id": 2, "x": 500, "y": 650, "vx": 0, "vy": -7, "color": (80, 220, 120), "w": 42, "h": 75}, # Northbound
        {"id": 3, "x": 700, "y": 120, "vx": 0, "vy": 8, "color": (230, 100, 70), "w": 44, "h": 80},  # Southbound
        {"id": 4, "x": 900, "y": 620, "vx": 0, "vy": -6, "color": (190, 70, 240), "w": 45, "h": 78}, # Northbound
    ]

    for f in range(total_frames):
        # Dark asphalt background with road lanes
        frame = np.full((height, width, 3), (35, 38, 42), dtype=np.uint8)

        # Draw road markings (lane dividers)
        for lx in [200, 400, 600, 800, 1000]:
            cv2.line(frame, (lx, 0), (lx, height), (70, 75, 82), 2)
            # Dashed white markings
            for dash_y in range((f * 4) % 60, height, 60):
                cv2.line(frame, (lx, dash_y), (lx, dash_y + 30), (220, 220, 220), 2)

        # Draw simulated tripwire line (y = 360)
        cv2.line(frame, (100, 360), (1100, 360), (0, 220, 255), 2)

        # Render moving vehicles
        for v in vehicles:
            v["x"] += v["vx"]
            v["y"] += v["vy"]
            x, y, w, h = int(v["x"]), int(v["y"]), v["w"], v["h"]

            # Vehicle body rectangle
            cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), v["color"], -1)
            # Roof / windshield accent
            cv2.rectangle(frame, (x - w // 4, y - h // 4), (x + w // 4, y + h // 4), (20, 20, 20), -1)

        # Timestamp HUD
        cv2.putText(
            frame, f"AEROTRACK SIMULATOR - FRAME {f:04d}", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
        )
        out.write(frame)

    out.release()
    print(f"[✓] Created synthetic video: {filename} ({os.path.getsize(filename) / 1024:.1f} KB)")
    return filename


def test_api_endpoint(
    server_url: str = "http://127.0.0.1:8000",
    video_path: str = "synthetic_drone_test.mp4",
):
    """Executes a real HTTP test against the AeroTrack FastAPI endpoint."""
    health_url = f"{server_url}/api/v1/health"
    track_url = f"{server_url}/api/v1/track"

    print(f"[*] Checking AeroTrack health at {health_url}...")
    try:
        res = requests.get(health_url, timeout=5)
        print(f"[✓] Health status: {res.status_code}")
        print(f"    Payload: {json.dumps(res.json(), indent=2)}")
    except Exception as e:
        print(f"[!] Could not reach health endpoint: {e}")
        print("    Ensure uvicorn is running: uvicorn main:app --reload")
        return

    # Tracking configuration payload
    config_payload = {
        "confidence_threshold": 0.20,
        "iou_threshold": 0.45,
        "ground_sampling_distance_m_per_px": 0.04,
        "drone_fps": 30.0,
        "virtual_lines": [
            {
                "line_id": "mid_crossroad",
                "name": "Midpoint Traffic Tripwire",
                "start_x": 100.0,
                "start_y": 360.0,
                "end_x": 1100.0,
                "end_y": 360.0,
                "positive_label": "Northbound",
                "negative_label": "Southbound",
            }
        ],
        "generate_annotated_video": False,
    }

    print(f"\n[*] Uploading video and calling {track_url}...")
    t0 = time.time()
    with open(video_path, "rb") as f:
        files = {"file": (os.path.basename(video_path), f, "video/mp4")}
        data = {"config_json": json.dumps(config_payload)}
        response = requests.post(track_url, files=files, data=data)

    elapsed = time.time() - t0
    print(f"[✓] HTTP Response: {response.status_code} in {elapsed:.2f}s")

    if response.status_code == 200:
        telemetry = response.json()
        print("\n--- TELEMETRY RESPONSE SUMMARY ---")
        print(f"Status: {telemetry.get('status')}")
        print(f"Video Info: {telemetry.get('video_info')}")
        print(f"Line Summary: {json.dumps(telemetry.get('line_summary'), indent=2)}")
        print(f"Total Crossing Events: {len(telemetry.get('crossing_events', []))}")
        print(f"Performance Metrics: {json.dumps(telemetry.get('metrics'), indent=2)}")
    else:
        print(f"[!] Request failed: {response.text}")


if __name__ == "__main__":
    sample_vid = generate_synthetic_drone_video()
    server_addr = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    test_api_endpoint(server_url=server_addr, video_path=sample_vid)
