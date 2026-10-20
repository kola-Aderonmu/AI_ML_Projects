"""
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

# Global tracker instance
tracker_engine: Optional[AeroTracker] = None


# ==============================================================================
# Pydantic Schemas for Strict Data Validation & Telemetry Output
# ==============================================================================

class VirtualLineConfig(BaseModel):
    """Configuration for a virtual tripwire line."""
    line_id: str = Field(..., description="Unique line identifier, e.g. 'gate_alpha'")
    name: str = Field(..., description="Human-readable line name, e.g. 'North Highway Gate'")
    start_x: float = Field(..., description="X coordinate of line start point in pixels")
    start_y: float = Field(..., description="Y coordinate of line start point in pixels")
    end_x: float = Field(..., description="X coordinate of line end point in pixels")
    end_y: float = Field(..., description="Y coordinate of line end point in pixels")
    positive_label: str = Field("Northbound", description="Label for positive vector crossings")
    negative_label: str = Field("Southbound", description="Label for negative vector crossings")


class TrackingRequestConfig(BaseModel):
    """Optional tuning parameters for drone video inference."""
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
    """Request payload for processing an RTSP/HTTP drone stream."""
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
    """Initializes models and resources on startup."""
    global tracker_engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing AeroTrack Engine on device: {device}")
    
    # Load default YOLOv11 nano model
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

# Enable CORS for frontend integration
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
    """Returns system status, PyTorch device info, and loaded model weights."""
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
    """Returns default tracking configuration and preset tripwire coordinates."""
    return TrackingRequestConfig()


@app.post("/api/v1/track", response_model=TrackResponse, tags=["Inference"])
async def track_video_file(
    file: UploadFile = File(..., description="Input video file (.mp4, .avi, .mov)"),
    config_json: Optional[str] = Form(
        None, description="Optional JSON string conforming to TrackingRequestConfig"
    ),
):
    """
    Ingests an uploaded aerial video file (.mp4), runs YOLOv11 + ByteTrack inference,
    computes trajectory lines, checks virtual tripwire crossings, and returns full telemetry.
    """
    if tracker_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tracking engine is not yet initialized.",
        )

    # Parse config from form payload
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

    # Save uploaded file to temporary path
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
        # Configure tracker with request parameters
        tracker_engine.conf_thresh = config.confidence_threshold
        tracker_engine.iou_thresh = config.iou_threshold
        tracker_engine.classes_filter = config.classes_to_track
        tracker_engine.gsd = config.ground_sampling_distance_m_per_px
        tracker_engine.fps = config.drone_fps

        # Convert virtual lines to internal definitions
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

        # Execute video processing pipeline
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
        # Cleanup temporary input file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass


@app.post("/api/v1/track/stream", response_model=TrackResponse, tags=["Inference"])
async def track_video_stream(payload: StreamTrackRequest):
    """
    Ingests an RTSP, RTMP, or live HTTP drone video stream, runs real-time tracking,
    and returns cumulative telemetry data.
    """
    if tracker_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tracking engine is not yet initialized.",
        )

    try:
        # Configure tracker
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

        # Estimate max frames based on max_duration_sec
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
