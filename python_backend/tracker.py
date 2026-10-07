"""
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
        """
        Initializes the YOLOv11 model and tracking engine.

        Args:
            model_weights: Ultralytics model weights (e.g. 'yolo11n.pt', 'yolo11s.pt', or VisDrone tuned)
            confidence_threshold: Minimum detection confidence score
            iou_threshold: Intersection-over-Union threshold for NMS
            classes_filter: List of class IDs to track (e.g., [2, 3, 5, 7] for vehicles)
            tracker_config: ByteTrack YAML config path ('bytetrack.yaml')
            gsd_m_per_px: Spatial resolution in meters per pixel based on drone altitude and camera FOV
            drone_fps: Frame rate of the aerial footage
            device: 'cuda', 'cpu', or 'auto'
        """
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

        # Virtual tripwire counting lines
        self.virtual_lines: List[VirtualLineDefinition] = []

        # Internal state tracking
        self.trajectories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self.track_class_map: Dict[int, str] = {}
        self.track_velocities: Dict[int, Tuple[float, float]] = {}  # m/s, km/h
        
        # Hysteresis state-machine to prevent double-counting across line crossings
        # Key: (track_id, line_id) -> timestamp / frame_idx of crossing
        self.crossed_events_cache: Dict[Tuple[int, str], int] = {}
        
        # Summary counters per line
        self.line_counters: Dict[str, Dict[str, Any]] = {}

    def configure_virtual_lines(self, lines: List[VirtualLineDefinition]):
        """Configures virtual tripwires for directional counting."""
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
        """Resets tracking state between video streams or test runs."""
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
        """
        Executes the full pipeline: video ingestion -> YOLOv11 + ByteTrack inference
        -> spatial line crossing detection -> velocity estimation -> telemetry generation.

        Args:
            video_source: Local filepath (.mp4) or RTSP stream URL
            output_annotated_path: Optional path to save annotated video
            max_frames: Max frames to process (useful for short latency tests)
            draw_hud: Whether to render bounding boxes, trajectories, and HUD

        Returns:
            Structured dictionary matching the Pydantic telemetry schema.
        """
        self.reset_state()

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video initialized: {width}x{height} @ {source_fps:.2f} FPS ({total_video_frames} frames)")

        # Optional video writer for annotated output
        writer = None
        if output_annotated_path:
            os.makedirs(os.path.dirname(output_annotated_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_annotated_path, fourcc, source_fps, (width, height))

        frame_idx = 0
        crossing_events: List[Dict[str, Any]] = []
        active_tracks_snapshot: List[Dict[str, Any]] = []

        # Latency metrics collection
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

                t0 = time.perf_counter()

                # -------------------------------------------------------------
                # 1. Inference & ByteTrack Tracking
                # -------------------------------------------------------------
                # persist=True retains tracker Kalman states across consecutive frames
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    classes=self.classes_filter,
                    verbose=False,
                )

                t1 = time.perf_counter()

                # Extract performance speeds from Ultralytics result object
                if len(results) > 0 and hasattr(results[0], "speed"):
                    speed_dict = results[0].speed
                    preprocess_latencies.append(speed_dict.get("preprocess", 1.0))
                    inference_latencies.append(speed_dict.get("inference", 10.0))
                    postprocess_latencies.append(speed_dict.get("postprocess", 1.0))

                # -------------------------------------------------------------
                # 2. Extract Detections & Update Trajectories
                # -------------------------------------------------------------
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

                        # Update trajectory deque
                        prev_point = self.trajectories[track_id][-1] if self.trajectories[track_id] else None
                        self.trajectories[track_id].append((cx, cy))
                        self.track_class_map[track_id] = c_name

                        # -----------------------------------------------------
                        # 3. Spatial Math: Velocity Calculation
                        # -----------------------------------------------------
                        mps = 0.0
                        kmh = 0.0
                        if len(self.trajectories[track_id]) >= 2:
                            # Use moving displacement over recent frames to filter centroid jitter
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

                        # -----------------------------------------------------
                        # 4. Spatial Math: Virtual Line Crossing & Direction
                        # -----------------------------------------------------
                        if prev_point is not None:
                            curr_point = (cx, cy)

                            for vline in self.virtual_lines:
                                line_id = vline.line_id
                                # Segment intersection check
                                if line_segments_intersect(
                                    prev_point, curr_point, vline.start_point, vline.end_point
                                ):
                                    # State-machine hysteresis: only count once per tracklet per 30 frames
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
                                        logger.info(
                                            f"Crossing Event [F{frame_idx}]: Track #{track_id} ({c_name}) "
                                            f"crossed {vline.name} -> {direction} ({kmh} km/h)"
                                        )

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

                # Keep snapshot of the latest active tracks
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

                # -------------------------------------------------------------
                # 5. Optional Drone HUD Annotation Rendering
                # -------------------------------------------------------------
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

        # Construct final telemetry metrics
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
        """Draws bounding boxes, persistent trajectory tails, virtual lines, and telemetry overlay."""
        overlay = frame.copy()

        # 1. Draw Virtual Counting Lines
        for vline in self.virtual_lines:
            pt1 = (int(vline.start_point[0]), int(vline.start_point[1]))
            pt2 = (int(vline.end_point[0]), int(vline.end_point[1]))
            cv2.line(overlay, pt1, pt2, vline.color_bgr, 3, cv2.LINE_AA)

            # Midpoint label
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
            # Trajectory polyline tail
            pts = np.array(list(trk.trajectory), dtype=np.int32).reshape((-1, 1, 2))
            if len(pts) > 1:
                cv2.polylines(overlay, [pts], False, (0, 220, 255), 2, cv2.LINE_AA)

            # Bounding box
            x1, y1, x2, y2 = map(int, trk.bbox)
            box_color = (0, 255, 128)  # Emerald green
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

            # Tag label: ID, Class, Speed
            tag = f"ID:{trk.track_id} {trk.class_name} {trk.velocity_kmh}km/h"
            cv2.putText(
                overlay, tag, (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA
            )

        # 3. Top-Left HUD Telemetry Overlay
        hud_bg = (20, 20, 20)
        cv2.rectangle(overlay, (10, 10), (320, 95), hud_bg, -1)
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
