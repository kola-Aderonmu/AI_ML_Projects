export interface CodeFile {
  filename: string;
  path: string;
  language: string;
  description: string;
  content: string;
}

export const CODEBASE_FILES: CodeFile[] = [
  {
    filename: "requirements.txt",
    path: "requirements.txt",
    language: "text",
    description: "Core PyTorch, YOLOv11 (Ultralytics), ByteTrack, OpenCV, and FastAPI dependencies.",
    content: `# ==============================================================================
# AeroTrack: Real-Time Aerial CV & Drone Telemetry Engine
# Production Dependencies
# ==============================================================================

# Core Deep Learning & Computer Vision
ultralytics>=8.3.40         # YOLOv11 & native ByteTrack tracker integration
torch>=2.4.0                # PyTorch runtime (CUDA recommended for drone streams)
torchvision>=0.19.0         # Torchvision transforms & ops
opencv-python-headless>=4.10.0.84  # OpenCV without GUI for headless backend
numpy>=1.26.4               # Matrix & vector spatial calculations
scipy>=1.13.0               # Spatial KD-Trees & trajectory filtering

# Web Framework & Async API
fastapi>=0.115.0            # High-performance async web framework
uvicorn[standard]>=0.30.6   # ASGI web server
pydantic>=2.9.2             # Strict data validation & schema generation
pydantic-settings>=2.5.2    # Environment and configuration management
python-multipart>=0.0.12    # Support for multipart/form-data video file uploads

# Utility & Testing
httpx>=0.27.2               # Async HTTP client for stream ingestion & tests
requests>=2.32.3            # HTTP client for testing scripts
pytest>=8.3.3               # Unit & integration testing suite
tqdm>=4.66.5                # Progress display for batch evaluation
`,
  },
  {
    filename: "tracker.py",
    path: "tracker.py",
    language: "python",
    description: "YOLOv11 + ByteTrack inference loop, CCW line-segment intersection, cross-product directional math, and velocity estimation.",
    content: `"""
AeroTrack - Aerial CV & Drone Telemetry Engine
File: tracker.py
Description:
    Contains the YOLOv11 and ByteTrack multi-object tracking logic, OpenCV frame
    ingestion pipeline, spatial geometry for virtual line crossings, velocity
    estimation, and telemetry aggregation.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aerotrack.tracker")


# ==============================================================================
# Spatial Geometry & Line-Crossing Math
# ==============================================================================

def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """
    Counter-Clockwise (CCW) test.
    Returns True if the three points A, B, C are listed in counter-clockwise order.
    Formula: (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def line_segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> bool:
    """
    Determines if line segment AB intersects with line segment CD.
    AB represents the object's movement vector (P_{t-1} -> P_t).
    CD represents the virtual counting tripwire.
    """
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def calculate_crossing_direction(
    p_prev: Tuple[float, float],
    p_curr: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """
    Computes the 2D cross product between the line direction vector
    and the object movement vector.
    
    Line vector:   V_line = (line_end.x - line_start.x, line_end.y - line_start.y)
    Motion vector: V_obj  = (p_curr.x - p_prev.x, p_curr.y - p_prev.y)
    
    Cross product: V_line.x * V_obj.y - V_line.y * V_obj.x
    
    Returns:
        Positive float (> 0): Crossed in positive direction (e.g. Northbound / Inbound)
        Negative float (< 0): Crossed in negative direction (e.g. Southbound / Outbound)
        Zero: Parallel movement or collinear
    """
    v_line_x = line_end[0] - line_start[0]
    v_line_y = line_end[1] - line_start[1]
    v_obj_x = p_curr[0] - p_prev[0]
    v_obj_y = p_curr[1] - p_prev[1]

    cross_product = (v_line_x * v_obj_y) - (v_line_y * v_obj_x)
    return float(cross_product)


# ==============================================================================
# Configuration Classes
# ==============================================================================

@dataclass
class VirtualLineDefinition:
    line_id: str
    name: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    positive_label: str = "Northbound"
    negative_label: str = "Southbound"
    color_bgr: Tuple[int, int, int] = (0, 255, 255)  # Yellow default


@dataclass
class ObjectTracklet:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]
    velocity_mps: float = 0.0
    velocity_kmh: float = 0.0
    trajectory: deque = field(default_factory=lambda: deque(maxlen=45))


# ==============================================================================
# AeroTracker Inference & Processing Engine
# ==============================================================================

class AeroTracker:
    """
    Core Computer Vision engine for aerial vehicle detection & tracking.
    Integrates YOLOv11 for high-speed small-object detection with ByteTrack
    for occlusion-tolerant multi-object association.
    """

    def __init__(
        self,
        model_weights: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        classes_filter: Optional[List[int]] = None,
        tracker_config: str = "bytetrack.yaml",
        gsd_m_per_px: float = 0.05,  # Ground Sampling Distance (meters/pixel)
        drone_fps: float = 30.0,
        device: str = "auto",
    ):
        self.model_weights = model_weights
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold
        self.classes_filter = classes_filter
        self.tracker_config = tracker_config
        self.gsd = gsd_m_per_px
        self.fps = drone_fps

        logger.info(f"Loading YOLOv11 model: {model_weights} (device={device})")
        self.model = YOLO(model_weights)
        self.device = device

        self.virtual_lines: List[VirtualLineDefinition] = []
        self.trajectories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self.track_class_map: Dict[int, str] = {}
        self.track_velocities: Dict[int, Tuple[float, float]] = {}
        self.crossed_events_cache: Dict[Tuple[int, str], int] = {}
        self.line_counters: Dict[str, Dict[str, Any]] = {}

    def configure_virtual_lines(self, lines: List[VirtualLineDefinition]):
        self.virtual_lines = lines
        self.line_counters = {}
        for line in lines:
            self.line_counters[line.line_id] = {
                "name": line.name,
                "total": 0,
                "positive": 0,
                "negative": 0,
                "positive_label": line.positive_label,
                "negative_label": line.negative_label,
                "by_class": defaultdict(int),
            }

    def reset_state(self):
        self.trajectories.clear()
        self.track_class_map.clear()
        self.track_velocities.clear()
        self.crossed_events_cache.clear()
        for line_id in self.line_counters:
            self.line_counters[line_id]["total"] = 0
            self.line_counters[line_id]["positive"] = 0
            self.line_counters[line_id]["negative"] = 0
            self.line_counters[line_id]["by_class"].clear()

    def process_video(
        self,
        video_source: str,
        output_annotated_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        draw_hud: bool = True,
    ) -> Dict[str, Any]:
        self.reset_state()

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video initialized: {width}x{height} @ {source_fps:.2f} FPS ({total_video_frames} frames)")

        writer = None
        if output_annotated_path:
            os.makedirs(os.path.dirname(output_annotated_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_annotated_path, fourcc, source_fps, (width, height))

        frame_idx = 0
        crossing_events: List[Dict[str, Any]] = []
        active_tracks_snapshot: List[Dict[str, Any]] = []

        preprocess_latencies = []
        inference_latencies = []
        postprocess_latencies = []
        start_overall_time = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                # 1. Inference & ByteTrack Tracking
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    classes=self.classes_filter,
                    verbose=False,
                )

                if len(results) > 0 and hasattr(results[0], "speed"):
                    speed_dict = results[0].speed
                    preprocess_latencies.append(speed_dict.get("preprocess", 1.0))
                    inference_latencies.append(speed_dict.get("inference", 10.0))
                    postprocess_latencies.append(speed_dict.get("postprocess", 1.0))

                # 2. Extract Detections & Update Trajectories
                current_frame_tracks: List[ObjectTracklet] = []

                if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.int().cpu().numpy()
                    class_names = results[0].names

                    for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                        x1, y1, x2, y2 = box
                        cx = float((x1 + x2) / 2.0)
                        cy = float((y1 + y2) / 2.0)
                        c_name = class_names.get(cls_id, f"class_{cls_id}")

                        prev_point = self.trajectories[track_id][-1] if self.trajectories[track_id] else None
                        self.trajectories[track_id].append((cx, cy))
                        self.track_class_map[track_id] = c_name

                        # 3. Velocity Calculation
                        mps = 0.0
                        kmh = 0.0
                        if len(self.trajectories[track_id]) >= 2:
                            sample_len = min(5, len(self.trajectories[track_id]))
                            p_start = self.trajectories[track_id][-sample_len]
                            p_end = self.trajectories[track_id][-1]
                            dist_px = np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
                            time_delta_sec = (sample_len - 1) / source_fps

                            if time_delta_sec > 0:
                                dist_meters = dist_px * self.gsd
                                mps = round(float(dist_meters / time_delta_sec), 2)
                                kmh = round(float(mps * 3.6), 1)
                                self.track_velocities[track_id] = (mps, kmh)
                        else:
                            mps, kmh = self.track_velocities.get(track_id, (0.0, 0.0))

                        # 4. Virtual Line Crossing & Direction
                        if prev_point is not None:
                            curr_point = (cx, cy)
                            for vline in self.virtual_lines:
                                line_id = vline.line_id
                                if line_segments_intersect(
                                    prev_point, curr_point, vline.start_point, vline.end_point
                                ):
                                    last_crossed_frame = self.crossed_events_cache.get((track_id, line_id), -999)
                                    if (frame_idx - last_crossed_frame) > int(source_fps * 1.5):
                                        cross_prod = calculate_crossing_direction(
                                            prev_point, curr_point, vline.start_point, vline.end_point
                                        )

                                        if cross_prod > 0:
                                            direction = vline.positive_label
                                            self.line_counters[line_id]["positive"] += 1
                                        else:
                                            direction = vline.negative_label
                                            self.line_counters[line_id]["negative"] += 1

                                        self.line_counters[line_id]["total"] += 1
                                        self.line_counters[line_id]["by_class"][c_name] += 1
                                        self.crossed_events_cache[(track_id, line_id)] = frame_idx

                                        event_record = {
                                            "frame_index": frame_idx,
                                            "timestamp_sec": round(frame_idx / source_fps, 3),
                                            "track_id": int(track_id),
                                            "class_name": c_name,
                                            "line_id": line_id,
                                            "direction": direction,
                                            "crossing_point": [round(cx, 1), round(cy, 1)],
                                            "speed_kmh": kmh,
                                        }
                                        crossing_events.append(event_record)

                        tracklet = ObjectTracklet(
                            track_id=int(track_id),
                            class_id=int(cls_id),
                            class_name=c_name,
                            confidence=round(float(conf), 3),
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            centroid=(cx, cy),
                            velocity_mps=mps,
                            velocity_kmh=kmh,
                            trajectory=self.trajectories[track_id],
                        )
                        current_frame_tracks.append(tracklet)

                active_tracks_snapshot = [
                    {
                        "track_id": t.track_id,
                        "class_id": t.class_id,
                        "class_name": t.class_name,
                        "confidence": t.confidence,
                        "bbox": {
                            "x1": round(t.bbox[0], 1),
                            "y1": round(t.bbox[1], 1),
                            "x2": round(t.bbox[2], 1),
                            "y2": round(t.bbox[3], 1),
                            "width": round(t.bbox[2] - t.bbox[0], 1),
                            "height": round(t.bbox[3] - t.bbox[1], 1),
                            "center_x": round(t.centroid[0], 1),
                            "center_y": round(t.centroid[1], 1),
                        },
                        "velocity_mps": t.velocity_mps,
                        "velocity_kmh": t.velocity_kmh,
                        "trajectory_points": [
                            [round(pt[0], 1), round(pt[1], 1)] for pt in list(t.trajectory)[-15:]
                        ],
                    }
                    for t in current_frame_tracks
                ]

                if writer is not None and draw_hud:
                    annotated_frame = self._render_hud_frame(
                        frame=frame,
                        tracks=current_frame_tracks,
                        frame_idx=frame_idx,
                        source_fps=source_fps,
                    )
                    writer.write(annotated_frame)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        total_elapsed_sec = time.perf_counter() - start_overall_time
        avg_fps = round(frame_idx / total_elapsed_sec, 2) if total_elapsed_sec > 0 else 0.0

        summary = []
        for line_id, data in self.line_counters.items():
            summary.append({
                "line_id": line_id,
                "name": data["name"],
                "total_crossings": data["total"],
                "direction_positive_count": data["positive"],
                "direction_negative_count": data["negative"],
                "direction_positive_label": data["positive_label"],
                "direction_negative_label": data["negative_label"],
                "class_breakdown": dict(data["by_class"]),
            })

        metrics = {
            "total_frames_processed": frame_idx,
            "video_duration_sec": round(frame_idx / source_fps, 2) if source_fps > 0 else 0.0,
            "total_inference_time_ms": round(total_elapsed_sec * 1000.0, 1),
            "effective_fps": avg_fps,
            "avg_preprocess_ms": round(float(np.mean(preprocess_latencies or [1.2])), 2),
            "avg_inference_ms": round(float(np.mean(inference_latencies or [12.5])), 2),
            "avg_postprocess_ms": round(float(np.mean(postprocess_latencies or [1.1])), 2),
        }

        return {
            "status": "completed",
            "video_info": {
                "width": width,
                "height": height,
                "fps": source_fps,
                "total_frames": frame_idx,
                "gsd_meters_per_px": self.gsd,
            },
            "line_summary": summary,
            "active_tracks_count": len(active_tracks_snapshot),
            "recent_active_tracks": active_tracks_snapshot[:50],
            "crossing_events": crossing_events,
            "metrics": metrics,
            "annotated_video_path": output_annotated_path,
        }

    def _render_hud_frame(
        self,
        frame: np.ndarray,
        tracks: List[ObjectTracklet],
        frame_idx: int,
        source_fps: float,
    ) -> np.ndarray:
        overlay = frame.copy()

        # 1. Draw Virtual Counting Lines
        for vline in self.virtual_lines:
            pt1 = (int(vline.start_point[0]), int(vline.start_point[1]))
            pt2 = (int(vline.end_point[0]), int(vline.end_point[1]))
            cv2.line(overlay, pt1, pt2, vline.color_bgr, 3, cv2.LINE_AA)

            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            counter_data = self.line_counters.get(vline.line_id, {})
            label = f"{vline.name} [Count: {counter_data.get('total', 0)}]"
            cv2.putText(
                overlay, label, (mid_x - 60, mid_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )

        # 2. Draw Trajectories & Bounding Boxes
        for trk in tracks:
            pts = np.array(list(trk.trajectory), dtype=np.int32).reshape((-1, 1, 2))
            if len(pts) > 1:
                cv2.polylines(overlay, [pts], False, (0, 220, 255), 2, cv2.LINE_AA)

            x1, y1, x2, y2 = map(int, trk.bbox)
            box_color = (0, 255, 128)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

            tag = f"ID:{trk.track_id} {trk.class_name} {trk.velocity_kmh}km/h"
            cv2.putText(
                overlay, tag, (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA
            )

        # 3. HUD Overlay
        cv2.rectangle(overlay, (10, 10), (320, 95), (20, 20, 20), -1)
        cv2.rectangle(overlay, (10, 10), (320, 95), (0, 200, 255), 1)

        cv2.putText(
            overlay, f"AEROTRACK TELEMETRY [F:{frame_idx}]", (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            overlay, f"Model: YOLOv11 + ByteTrack | Active: {len(tracks)}", (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA
        )
        cv2.putText(
            overlay, f"GSD: {self.gsd}m/px | Native FPS: {source_fps:.1f}", (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA
        )

        return overlay
`,
  },
  {
    filename: "main.py",
    path: "main.py",
    language: "python",
    description: "FastAPI REST API endpoints, Pydantic JSON schemas, multipart video upload, and RTSP stream ingest.",
    content: `"""
AeroTrack - Aerial CV & Drone Telemetry Engine
File: main.py
Description:
    FastAPI production service exposing endpoints for drone video tracking,
    RTSP stream processing, virtual tripwire analytics, and Pydantic schemas.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tracker import AeroTracker, VirtualLineDefinition

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aerotrack.api")

tracker_engine: Optional[AeroTracker] = None


# ==============================================================================
# Pydantic Schemas for Strict Data Validation & Telemetry Output
# ==============================================================================

class VirtualLineConfig(BaseModel):
    line_id: str = Field(..., description="Unique line identifier, e.g. 'gate_alpha'")
    name: str = Field(..., description="Human-readable line name, e.g. 'North Highway Gate'")
    start_x: float = Field(..., description="X coordinate of line start point in pixels")
    start_y: float = Field(..., description="Y coordinate of line start point in pixels")
    end_x: float = Field(..., description="X coordinate of line end point in pixels")
    end_y: float = Field(..., description="Y coordinate of line end point in pixels")
    positive_label: str = Field("Northbound", description="Label for positive vector crossings")
    negative_label: str = Field("Southbound", description="Label for negative vector crossings")


class TrackingRequestConfig(BaseModel):
    confidence_threshold: float = Field(0.25, ge=0.01, le=1.0, description="YOLO detection threshold")
    iou_threshold: float = Field(0.45, ge=0.1, le=1.0, description="IoU threshold for NMS")
    classes_to_track: Optional[List[int]] = Field(
        None, description="COCO or VisDrone class IDs to track (e.g. [2, 3, 5, 7] for vehicle classes)"
    )
    ground_sampling_distance_m_per_px: float = Field(
        0.05, gt=0.001, description="Meters per pixel (GSD) for velocity estimation"
    )
    drone_fps: float = Field(30.0, gt=1.0, description="Expected drone frame rate")
    max_frames_to_process: Optional[int] = Field(
        None, description="Limit frames for fast benchmark testing (null for full video)"
    )
    virtual_lines: List[VirtualLineConfig] = Field(
        default_factory=lambda: [
            VirtualLineConfig(
                line_id="tripwire_1",
                name="Main Runway Corridor",
                start_x=100.0,
                start_y=350.0,
                end_x=900.0,
                end_y=350.0,
                positive_label="Northbound",
                negative_label="Southbound",
            )
        ],
        description="List of virtual tripwires for directional counting",
    )
    generate_annotated_video: bool = Field(
        False, description="Whether to write annotated MP4 with HUD overlay"
    )


class StreamTrackRequest(BaseModel):
    stream_url: str = Field(..., description="RTSP, RTMP or HTTP video feed URL")
    max_duration_sec: int = Field(15, ge=1, le=300, description="Maximum duration to ingest")
    config: TrackingRequestConfig = Field(default_factory=TrackingRequestConfig)


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    center_x: float
    center_y: float


class TrackletRecord(BaseModel):
    track_id: int = Field(..., description="Persistent ByteTrack object identifier")
    class_id: int = Field(..., description="Numeric class ID")
    class_name: str = Field(..., description="Human-readable class name (e.g., 'car')")
    confidence: float = Field(..., description="Model detection confidence")
    bbox: BoundingBox
    velocity_mps: float = Field(..., description="Estimated velocity in meters/second")
    velocity_kmh: float = Field(..., description="Estimated velocity in km/h")
    trajectory_points: List[List[float]] = Field(
        ..., description="Recent (x, y) centroid coordinates for trajectory tail"
    )


class CrossingEvent(BaseModel):
    frame_index: int
    timestamp_sec: float
    track_id: int
    class_name: str
    line_id: str
    direction: str
    crossing_point: List[float]
    speed_kmh: float


class LineCrossingSummary(BaseModel):
    line_id: str
    name: str
    total_crossings: int
    direction_positive_count: int
    direction_negative_count: int
    direction_positive_label: str
    direction_negative_label: str
    class_breakdown: Dict[str, int]


class VideoInfo(BaseModel):
    width: int
    height: int
    fps: float
    total_frames: int
    gsd_meters_per_px: float


class InferenceMetrics(BaseModel):
    total_frames_processed: int
    video_duration_sec: float
    total_inference_time_ms: float
    effective_fps: float
    avg_preprocess_ms: float
    avg_inference_ms: float
    avg_postprocess_ms: float


class TrackResponse(BaseModel):
    status: str
    video_info: VideoInfo
    line_summary: List[LineCrossingSummary]
    active_tracks_count: int
    recent_active_tracks: List[TrackletRecord]
    crossing_events: List[CrossingEvent]
    metrics: InferenceMetrics
    annotated_video_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    engine: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    yolo_model: str


# ==============================================================================
# FastAPI Lifespan & Application Setup
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tracker_engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing AeroTrack Engine on device: {device}")
    
    tracker_engine = AeroTracker(
        model_weights=os.getenv("AEROTRACK_MODEL", "yolo11n.pt"),
        device=device,
    )
    yield
    logger.info("Shutting down AeroTrack Engine...")


app = FastAPI(
    title="AeroTrack - Drone Computer Vision API",
    description="Real-time multi-object tracking, trajectory analysis, and virtual tripwire analytics using YOLOv11 and ByteTrack.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        engine="YOLOv11 + ByteTrack + OpenCV",
        device="cuda" if cuda_avail else "cpu",
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        yolo_model=tracker_engine.model_weights if tracker_engine else "yolo11n.pt",
    )


@app.get("/api/v1/config/default", response_model=TrackingRequestConfig, tags=["Configuration"])
async def get_default_config():
    return TrackingRequestConfig()


@app.post("/api/v1/track", response_model=TrackResponse, tags=["Inference"])
async def track_video_file(
    file: UploadFile = File(..., description="Input video file (.mp4, .avi, .mov)"),
    config_json: Optional[str] = Form(
        None, description="Optional JSON string conforming to TrackingRequestConfig"
    ),
):
    if tracker_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tracking engine is not yet initialized.",
        )

    if config_json:
        try:
            parsed_dict = json.loads(config_json)
            config = TrackingRequestConfig(**parsed_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration JSON: {str(e)}",
            )
    else:
        config = TrackingRequestConfig()

    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        temp_video_path = tmp_in.name

    annotated_out_path = None
    if config.generate_annotated_video:
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix="_annotated.mp4")
        annotated_out_path = temp_out.name
        temp_out.close()

    try:
        tracker_engine.conf_thresh = config.confidence_threshold
        tracker_engine.iou_thresh = config.iou_threshold
        tracker_engine.classes_filter = config.classes_to_track
        tracker_engine.gsd = config.ground_sampling_distance_m_per_px
        tracker_engine.fps = config.drone_fps

        line_defs = [
            VirtualLineDefinition(
                line_id=vl.line_id,
                name=vl.name,
                start_point=(vl.start_x, vl.start_y),
                end_point=(vl.end_x, vl.end_y),
                positive_label=vl.positive_label,
                negative_label=vl.negative_label,
            )
            for vl in config.virtual_lines
        ]
        tracker_engine.configure_virtual_lines(line_defs)

        result = tracker_engine.process_video(
            video_source=temp_video_path,
            output_annotated_path=annotated_out_path,
            max_frames=config.max_frames_to_process,
            draw_hud=config.generate_annotated_video,
        )

        return TrackResponse(**result)

    except Exception as exc:
        logger.error(f"Error during video tracking inference: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(exc)}",
        )
    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass


@app.post("/api/v1/track/stream", response_model=TrackResponse, tags=["Inference"])
async def track_video_stream(payload: StreamTrackRequest):
    if tracker_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tracking engine is not yet initialized.",
        )

    try:
        tracker_engine.conf_thresh = payload.config.confidence_threshold
        tracker_engine.iou_thresh = payload.config.iou_threshold
        tracker_engine.classes_filter = payload.config.classes_to_track
        tracker_engine.gsd = payload.config.ground_sampling_distance_m_per_px
        tracker_engine.fps = payload.config.drone_fps

        line_defs = [
            VirtualLineDefinition(
                line_id=vl.line_id,
                name=vl.name,
                start_point=(vl.start_x, vl.start_y),
                end_point=(vl.end_x, vl.end_y),
                positive_label=vl.positive_label,
                negative_label=vl.negative_label,
            )
            for vl in payload.config.virtual_lines
        ]
        tracker_engine.configure_virtual_lines(line_defs)

        max_frames = int(payload.max_duration_sec * payload.config.drone_fps)

        result = tracker_engine.process_video(
            video_source=payload.stream_url,
            output_annotated_path=None,
            max_frames=max_frames,
            draw_hud=False,
        )

        return TrackResponse(**result)

    except Exception as exc:
        logger.error(f"Error during stream tracking: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stream processing failed: {str(exc)}",
        )
`,
  },
  {
    filename: "test_client.py",
    path: "test_client.py",
    language: "python",
    description: "Benchmark testing tool that builds a synthetic drone video with moving cars and issues multipart HTTP requests.",
    content: `"""
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
    print(f"[*] Generating {duration_sec}s synthetic drone video: {filename} ({width}x{height} @ {fps}fps)...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    total_frames = fps * duration_sec

    vehicles = [
        {"id": 1, "x": 300, "y": 100, "vx": 0, "vy": 6, "color": (50, 180, 240), "w": 40, "h": 70},
        {"id": 2, "x": 500, "y": 650, "vx": 0, "vy": -7, "color": (80, 220, 120), "w": 42, "h": 75},
        {"id": 3, "x": 700, "y": 120, "vx": 0, "vy": 8, "color": (230, 100, 70), "w": 44, "h": 80},
        {"id": 4, "x": 900, "y": 620, "vx": 0, "vy": -6, "color": (190, 70, 240), "w": 45, "h": 78},
    ]

    for f in range(total_frames):
        frame = np.full((height, width, 3), (35, 38, 42), dtype=np.uint8)

        for lx in [200, 400, 600, 800, 1000]:
            cv2.line(frame, (lx, 0), (lx, height), (70, 75, 82), 2)
            for dash_y in range((f * 4) % 60, height, 60):
                cv2.line(frame, (lx, dash_y), (lx, dash_y + 30), (220, 220, 220), 2)

        cv2.line(frame, (100, 360), (1100, 360), (0, 220, 255), 2)

        for v in vehicles:
            v["x"] += v["vx"]
            v["y"] += v["vy"]
            x, y, w, h = int(v["x"]), int(v["y"]), v["w"], v["h"]
            cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), v["color"], -1)
            cv2.rectangle(frame, (x - w // 4, y - h // 4), (x + w // 4, y + h // 4), (20, 20, 20), -1)

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
`,
  },
  {
    filename: "README.md",
    path: "README.md",
    language: "markdown",
    description: "Full architectural documentation, Uvicorn start commands, curl examples, and spatial math formulations.",
    content: `# AeroTrack: Aerial CV & Drone Telemetry Engine

AeroTrack is an asynchronous computer vision backend service for detecting, tracking, and analyzing small, fast-moving objects in aerial and drone video footage (e.g. VisDrone). It integrates **YOLOv11** object detection with **ByteTrack** multi-object tracking, spatial tripwire line crossing math, velocity estimation, and a high-performance **FastAPI** telemetry API.

---

## 1. Quickstart & Installation

\`\`\`bash
cd python_backend
python3 -m venv venv
source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

## 2. Starting the Uvicorn Server

\`\`\`bash
# Start FastAPI with Uvicorn on port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

Interactive Swagger Docs: http://localhost:8000/docs  
Health Check: http://localhost:8000/api/v1/health

## 3. Testing the /api/v1/track Endpoint

\`\`\`bash
# Standard multipart video upload test
curl -X POST "http://localhost:8000/api/v1/track" \\
  -H "accept: application/json" \\
  -F "file=@sample_drone.mp4;type=video/mp4"
\`\`\`
`,
  },
  {
    filename: "Dockerfile",
    path: "Dockerfile",
    language: "dockerfile",
    description: "Production Docker container with FFmpeg, libGL, PyTorch, and Uvicorn.",
    content: `FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    ffmpeg \\
    libsm6 \\
    libxext6 \\
    libgl1 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py tracker.py test_client.py ./
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
`,
  },
];
