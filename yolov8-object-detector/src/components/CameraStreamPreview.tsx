import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  Camera,
  CameraOff,
  RefreshCw,
  Sliders,
  Crosshair,
  Zap,
  Eye,
  Radio,
  Video,
  Download,
  AlertTriangle,
  Move,
  Layers,
  Sparkles,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { VirtualLine, CrossingEventUI } from "./DroneVisionCanvas";

interface CameraStreamPreviewProps {
  tripwires: VirtualLine[];
  onUpdateTripwires: (updated: VirtualLine[]) => void;
  onCrossingEvent: (event: CrossingEventUI) => void;
  gsdMperPx: number;
  altitudeMeters: number;
}

interface TrackedBlob {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  w: number;
  h: number;
  area: number;
  trail: { x: number; y: number }[];
  lastCrossedTime: Record<string, number>;
  speedKmh: number;
  framesActive: number;
}

export const CameraStreamPreview: React.FC<CameraStreamPreviewProps> = ({
  tripwires,
  onUpdateTripwires,
  onCrossingEvent,
  gsdMperPx,
  altitudeMeters,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [mirror, setMirror] = useState<boolean>(false);
  const [sensitivity, setSensitivity] = useState<number>(30); // motion threshold
  const [visionMode, setVisionMode] = useState<"standard" | "thermal" | "night_vision">("standard");

  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [showTrails, setShowTrails] = useState<boolean>(true);
  const [showHud, setShowHud] = useState<boolean>(true);
  const [activeTracksCount, setActiveTracksCount] = useState<number>(0);
  const [fps, setFps] = useState<number>(30);

  // Line drag interaction
  const [draggingPoint, setDraggingPoint] = useState<{
    lineId: string;
    point: "start" | "end";
  } | null>(null);

  // Recording state
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordedBlobUrl, setRecordedBlobUrl] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  // Motion tracking state
  const previousFrameDataRef = useRef<Uint8ClampedArray | null>(null);
  const activeBlobsRef = useRef<TrackedBlob[]>([]);
  const nextTrackIdRef = useRef<number>(501);
  const animFrameRef = useRef<number | null>(null);
  const fpsTimerRef = useRef<number>(performance.now());
  const frameCountRef = useRef<number>(0);

  // Check and list cameras
  const refreshDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const deviceList = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = deviceList.filter((d) => d.kind === "videoinput");
      setDevices(videoInputs);
      if (videoInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch {
      // Ignored
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  // Start webcam stream
  const startCamera = async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Webcam access is not supported on this browser.");
      }

      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsStreaming(true);
      refreshDevices();
    } catch (err: any) {
      console.error("Camera access error:", err);
      let msg = "Could not access webcam. Please check browser permissions.";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        msg = "Camera permission was denied. Please allow camera access in your browser settings.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        msg = "No camera hardware detected on this device.";
      } else if (err.message) {
        msg = err.message;
      }
      setCameraError(msg);
      setIsStreaming(false);
    }
  };

  // Stop camera stream
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    activeBlobsRef.current = [];
    previousFrameDataRef.current = null;
  };

  useEffect(() => {
    return () => {
      stopCamera();
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  // CCW spatial math
  const ccw = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => {
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax);
  };

  const lineIntersects = (
    p1x: number, p1y: number,
    p2x: number, p2y: number,
    l1x: number, l1y: number,
    l2x: number, l2y: number
  ) => {
    return (
      ccw(p1x, p1y, l1x, l1y, l2x, l2y) !== ccw(p2x, p2y, l1x, l1y, l2x, l2y) &&
      ccw(p1x, p1y, p2x, p2y, l1x, l1y) !== ccw(p1x, p1y, p2x, p2y, l2x, l2y)
    );
  };

  const getDirectionProduct = (
    p1x: number, p1y: number,
    p2x: number, p2y: number,
    l1x: number, l1y: number,
    l2x: number, l2y: number
  ) => {
    const vlineX = l2x - l1x;
    const vlineY = l2y - l1y;
    const vobjX = p2x - p1x;
    const vobjY = p2y - p1y;
    return vlineX * vobjY - vlineY * vobjX;
  };

  // Process video frames: optical difference + tripwire detection
  useEffect(() => {
    if (!isStreaming) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
    const offscreen = offscreenCanvasRef.current;
    offscreen.width = 160;
    offscreen.height = 90;
    const offCtx = offscreen.getContext("2d", { willReadFrequently: true });

    const processLoop = (time: number) => {
      // FPS measurement
      frameCountRef.current++;
      if (time - fpsTimerRef.current >= 500) {
        setFps(Math.round((frameCountRef.current * 1000) / (time - fpsTimerRef.current)));
        frameCountRef.current = 0;
        fpsTimerRef.current = time;
      }

      if (video.readyState >= 2 && offCtx) {
        const w = canvas.width;
        const h = canvas.height;

        // 1. Draw base video frame
        ctx.save();
        if (mirror) {
          ctx.translate(w, 0);
          ctx.scale(-1, 1);
        }
        ctx.drawImage(video, 0, 0, w, h);
        ctx.restore();

        // 2. Vision filters (FLIR Thermal or Night Vision)
        if (visionMode === "thermal") {
          const imgData = ctx.getImageData(0, 0, w, h);
          const d = imgData.data;
          for (let i = 0; i < d.length; i += 4) {
            const gray = (d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114) / 255;
            // Thermal false color gradient (Ironbow)
            d[i] = Math.min(255, Math.floor(Math.sin(gray * Math.PI) * 255 * 1.2)); // R
            d[i + 1] = Math.min(255, Math.floor(Math.sin((gray - 0.25) * Math.PI) * 255)); // G
            d[i + 2] = Math.min(255, Math.floor(Math.cos(gray * Math.PI * 0.8) * 255)); // B
          }
          ctx.putImageData(imgData, 0, 0);
        } else if (visionMode === "night_vision") {
          const imgData = ctx.getImageData(0, 0, w, h);
          const d = imgData.data;
          for (let i = 0; i < d.length; i += 4) {
            const lum = (d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114) * 1.3;
            d[i] = Math.floor(lum * 0.1);
            d[i + 1] = Math.min(255, Math.floor(lum));
            d[i + 2] = Math.floor(lum * 0.2);
          }
          ctx.putImageData(imgData, 0, 0);
        }

        // 3. Motion Detection via temporal downscaled frame difference
        offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
        const currData = offCtx.getImageData(0, 0, offscreen.width, offscreen.height).data;

        const scaleX = w / offscreen.width;
        const scaleY = h / offscreen.height;

        let detectedRegions: { minX: number; minY: number; maxX: number; maxY: number; count: number }[] = [];

        if (previousFrameDataRef.current) {
          const prev = previousFrameDataRef.current;
          let diffCount = 0;
          let sumX = 0;
          let sumY = 0;
          let minX = offscreen.width;
          let minY = offscreen.height;
          let maxX = 0;
          let maxY = 0;

          for (let py = 0; py < offscreen.height; py += 2) {
            for (let px = 0; px < offscreen.width; px += 2) {
              const idx = (py * offscreen.width + px) * 4;
              const diff =
                Math.abs(currData[idx] - prev[idx]) +
                Math.abs(currData[idx + 1] - prev[idx + 1]) +
                Math.abs(currData[idx + 2] - prev[idx + 2]);

              if (diff > sensitivity * 3) {
                diffCount++;
                sumX += px;
                sumY += py;
                if (px < minX) minX = px;
                if (px > maxX) maxX = px;
                if (py < minY) minY = py;
                if (py > maxY) maxY = py;
              }
            }
          }

          // If significant moving pixels found, register detected blob
          if (diffCount > 35) {
            const actualMinX = mirror ? (offscreen.width - maxX) * scaleX : minX * scaleX;
            const actualMaxX = mirror ? (offscreen.width - minX) * scaleX : maxX * scaleX;
            detectedRegions.push({
              minX: Math.max(0, actualMinX - 10),
              minY: Math.max(0, minY * scaleY - 10),
              maxX: Math.min(w, actualMaxX + 10),
              maxY: Math.min(h, maxY * scaleY + 10),
              count: diffCount,
            });
          }
        }
        previousFrameDataRef.current = new Uint8ClampedArray(currData);

        // 4. Update & Associate Tracklets (ByteTrack simplified centroid tracker)
        const currentBlobs = activeBlobsRef.current;
        const updatedBlobs: TrackedBlob[] = [];

        detectedRegions.forEach((reg) => {
          const cx = (reg.minX + reg.maxX) / 2;
          const cy = (reg.minY + reg.maxY) / 2;
          const bw = reg.maxX - reg.minX;
          const bh = reg.maxY - reg.minY;

          // Find closest existing blob
          let closestBlob: TrackedBlob | undefined = undefined;
          let minDist = 120; // association radius

          for (const b of currentBlobs) {
            const dist = Math.hypot(b.x - cx, b.y - cy);
            if (dist < minDist) {
              minDist = dist;
              closestBlob = b;
            }
          }

          if (closestBlob) {
            const b: TrackedBlob = closestBlob;
            const prevX = b.x;
            const prevY = b.y;

            // Velocity smoothing
            const vx = cx - prevX;
            const vy = cy - prevY;
            b.vx = b.vx * 0.4 + vx * 0.6;
            b.vy = b.vy * 0.4 + vy * 0.6;
            b.x = cx;
            b.y = cy;
            b.w = bw;
            b.h = bh;
            b.framesActive++;

            b.trail.push({ x: cx, y: cy });
            if (b.trail.length > 20) b.trail.shift();

            // Calculate estimated velocity based on GSD
            const pixelDispPerSec = Math.hypot(b.vx, b.vy) * 30;
            const speedMps = pixelDispPerSec * gsdMperPx;
            b.speedKmh = Math.round(speedMps * 3.6);

            // Virtual tripwire crossing evaluation
            tripwires.forEach((line) => {
              const crossed = lineIntersects(
                prevX, prevY, cx, cy,
                line.startX, line.startY, line.endX, line.endY
              );

              if (crossed) {
                const now = Date.now();
                const lastTime = b.lastCrossedTime[line.id] || 0;
                if (now - lastTime > 1500) {
                  b.lastCrossedTime[line.id] = now;
                  const dirProd = getDirectionProduct(
                    prevX, prevY, cx, cy,
                    line.startX, line.startY, line.endX, line.endY
                  );
                  const dir = dirProd > 0 ? line.posLabel : line.negLabel;

                  onCrossingEvent({
                    id: `cam-${b.id}-${now}`,
                    timestamp: new Date().toLocaleTimeString(),
                    trackId: b.id,
                    className: "target",
                    lineName: line.name,
                    direction: dir,
                    speedKmh: b.speedKmh || 45,
                  });
                }
              }
            });

            updatedBlobs.push(b);
          } else {
            // New Tracklet
            const newId = nextTrackIdRef.current++;
            updatedBlobs.push({
              id: newId,
              x: cx,
              y: cy,
              vx: 0,
              vy: 0,
              w: bw,
              h: bh,
              area: reg.count,
              trail: [{ x: cx, y: cy }],
              lastCrossedTime: {},
              speedKmh: 42,
              framesActive: 1,
            });
          }
        });

        activeBlobsRef.current = updatedBlobs;
        setActiveTracksCount(updatedBlobs.length);

        // 5. Render Virtual Tripwires & Draggable Endpoints
        tripwires.forEach((line) => {
          ctx.save();
          ctx.strokeStyle = line.color || "#06b6d4";
          ctx.lineWidth = 3;
          ctx.shadowColor = line.color || "#06b6d4";
          ctx.shadowBlur = 12;

          ctx.beginPath();
          ctx.moveTo(line.startX, line.startY);
          ctx.lineTo(line.endX, line.endY);
          ctx.stroke();

          // Endpoints (interactive handles)
          ctx.fillStyle = "#ffffff";
          ctx.beginPath();
          ctx.arc(line.startX, line.startY, 7, 0, Math.PI * 2);
          ctx.arc(line.endX, line.endY, 7, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = "#0891b2";
          ctx.lineWidth = 2;
          ctx.stroke();

          // Label
          const midX = (line.startX + line.endX) / 2;
          const midY = (line.startY + line.endY) / 2;
          ctx.fillStyle = "rgba(10, 15, 30, 0.9)";
          ctx.fillRect(midX - 75, midY - 26, 150, 22);
          ctx.strokeStyle = line.color || "#06b6d4";
          ctx.lineWidth = 1;
          ctx.strokeRect(midX - 75, midY - 26, 150, 22);

          ctx.fillStyle = "#38bdf8";
          ctx.font = "bold 11px 'JetBrains Mono', monospace";
          ctx.textAlign = "center";
          ctx.fillText(`TRIPWIRE: ${line.name}`, midX, midY - 11);

          ctx.restore();
        });

        // 6. Render Trajectory Trails
        if (showTrails) {
          updatedBlobs.forEach((b) => {
            if (b.trail.length < 2) return;
            ctx.beginPath();
            ctx.moveTo(b.trail[0].x, b.trail[0].y);
            for (let i = 1; i < b.trail.length; i++) {
              ctx.lineTo(b.trail[i].x, b.trail[i].y);
            }
            ctx.strokeStyle = "rgba(56, 189, 248, 0.7)";
            ctx.lineWidth = 2.5;
            ctx.stroke();

            // Centroid marker
            ctx.fillStyle = "#38bdf8";
            ctx.beginPath();
            ctx.arc(b.x, b.y, 4, 0, Math.PI * 2);
            ctx.fill();
          });
        }

        // 7. Render Bounding Boxes & Drone Target HUD
        if (showBoxes) {
          updatedBlobs.forEach((b) => {
            const bx = b.x - b.w / 2;
            const by = b.y - b.h / 2;

            ctx.strokeStyle = "#10b981";
            ctx.lineWidth = 2;
            ctx.strokeRect(bx, by, b.w, b.h);

            // Corner brackets
            const bl = 10;
            ctx.strokeStyle = "#38bdf8";
            ctx.lineWidth = 2.5;

            // Top-left
            ctx.beginPath();
            ctx.moveTo(bx, by + bl);
            ctx.lineTo(bx, by);
            ctx.lineTo(bx + bl, by);
            ctx.stroke();

            // Bottom-right
            ctx.beginPath();
            ctx.moveTo(bx + b.w, by + b.h - bl);
            ctx.lineTo(bx + b.w, by + b.h);
            ctx.lineTo(bx + b.w - bl, by + b.h);
            ctx.stroke();

            // Tag
            ctx.fillStyle = "rgba(10, 15, 30, 0.9)";
            ctx.fillRect(bx, by - 22, 130, 20);
            ctx.fillStyle = "#34d399";
            ctx.font = "bold 10px 'JetBrains Mono', monospace";
            ctx.fillText(`ID:#${b.id} TARGET`, bx + 5, by - 8);

            ctx.fillStyle = "#38bdf8";
            ctx.fillText(`${b.speedKmh} km/h`, bx + 80, by - 8);
          });
        }

        // 8. Drone Camera Flight Telemetry Overlay
        if (showHud) {
          ctx.save();
          // Crosshair
          const cx = w / 2;
          const cy = h / 2;
          ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(cx - 35, cy);
          ctx.lineTo(cx + 35, cy);
          ctx.moveTo(cx, cy - 35);
          ctx.lineTo(cx, cy + 35);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(cx, cy, 25, 0, Math.PI * 2);
          ctx.stroke();

          // Top-Left Gimbal Box
          ctx.fillStyle = "rgba(10, 15, 30, 0.85)";
          ctx.fillRect(16, 16, 260, 80);
          ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
          ctx.lineWidth = 1;
          ctx.strokeRect(16, 16, 260, 80);

          ctx.fillStyle = "#38bdf8";
          ctx.font = "bold 11px 'JetBrains Mono', monospace";
          ctx.fillText("OPTICAL SENSOR STREAM (LIVE CAM)", 26, 34);

          ctx.fillStyle = "#94a3b8";
          ctx.font = "10px 'JetBrains Mono', monospace";
          ctx.fillText(`MODE: ${visionMode.toUpperCase()} | FPS: ${fps}`, 26, 52);
          ctx.fillText(`ALTITUDE SIM: ${altitudeMeters}m | GSD: ${gsdMperPx}m/px`, 26, 68);
          ctx.fillText(`MOTION THRESHOLD: ${sensitivity} | TRACKLETS: ${updatedBlobs.length}`, 26, 84);

          // Top-Right Status Badge
          ctx.fillStyle = "rgba(10, 15, 30, 0.85)";
          ctx.fillRect(w - 140, 16, 124, 40);
          ctx.strokeRect(w - 140, 16, 124, 40);

          ctx.fillStyle = "#ef4444";
          ctx.beginPath();
          ctx.arc(w - 122, 36, 5, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = "#f87171";
          ctx.font = "bold 10px 'JetBrains Mono', monospace";
          ctx.fillText("REC SENSOR", w - 110, 40);

          ctx.restore();
        }
      }

      animFrameRef.current = requestAnimationFrame(processLoop);
    };

    animFrameRef.current = requestAnimationFrame(processLoop);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [
    isStreaming,
    mirror,
    sensitivity,
    visionMode,
    tripwires,
    showBoxes,
    showTrails,
    showHud,
    altitudeMeters,
    gsdMperPx,
    onCrossingEvent,
  ]);

  // Handle dragging tripwire handles on the live canvas
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const clickY = ((e.clientY - rect.top) / rect.height) * canvas.height;

    // Check if clicked near any endpoint (radius 18px)
    for (const wire of tripwires) {
      if (Math.hypot(wire.startX - clickX, wire.startY - clickY) < 22) {
        setDraggingPoint({ lineId: wire.id, point: "start" });
        return;
      }
      if (Math.hypot(wire.endX - clickX, wire.endY - clickY) < 22) {
        setDraggingPoint({ lineId: wire.id, point: "end" });
        return;
      }
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!draggingPoint) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const currX = Math.round(((e.clientX - rect.left) / rect.width) * canvas.width);
    const currY = Math.round(((e.clientY - rect.top) / rect.height) * canvas.height);

    const updated = tripwires.map((wire) => {
      if (wire.id === draggingPoint.lineId) {
        if (draggingPoint.point === "start") {
          return { ...wire, startX: currX, startY: currY };
        } else {
          return { ...wire, endX: currX, endY: currY };
        }
      }
      return wire;
    });
    onUpdateTripwires(updated);
  };

  const handleCanvasMouseUp = () => {
    setDraggingPoint(null);
  };

  // Record a 5-second sample clip from the canvas for instant testing
  const startRecordingClip = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      const stream = canvas.captureStream(30);
      recordedChunksRef.current = [];
      const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        setRecordedBlobUrl(url);
        setIsRecording(false);
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);

      // Automatically stop after 5 seconds
      setTimeout(() => {
        if (recorder.state === "recording") {
          recorder.stop();
        }
      }, 5000);
    } catch (e) {
      console.error("Recording error:", e);
      setIsRecording(false);
    }
  };

  return (
    <div className="flex flex-col bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      {/* Top Header / Control Bar */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-slate-950 border-b border-slate-800 gap-2">
        <div className="flex items-center gap-2">
          <div
            className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded text-xs font-mono border ${
              isStreaming
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                : "bg-amber-500/10 border-amber-500/30 text-amber-400"
            }`}
          >
            <Radio className={`w-3.5 h-3.5 ${isStreaming ? "animate-pulse" : ""}`} />
            {isStreaming ? "LIVE WEBCAM DRONE FEED" : "CAMERA STANDBY"}
          </div>

          {isStreaming && (
            <span className="text-xs text-slate-400 font-mono hidden sm:inline">
              {activeTracksCount} optical tracklets | {fps} FPS
            </span>
          )}
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          {/* Device selector if multiple */}
          {devices.length > 1 && (
            <select
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              disabled={isStreaming}
              className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-300 text-xs font-mono outline-none"
            >
              {devices.map((d, i) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label || `Camera ${i + 1}`}
                </option>
              ))}
            </select>
          )}

          {/* Vision mode selector */}
          <div className="flex items-center bg-slate-900 p-0.5 rounded border border-slate-800">
            {(["standard", "thermal", "night_vision"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setVisionMode(mode)}
                className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors capitalize ${
                  visionMode === mode
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {mode === "night_vision" ? "Night" : mode}
              </button>
            ))}
          </div>

          <button
            onClick={() => setMirror(!mirror)}
            className={`px-2 py-1 rounded transition-colors ${
              mirror
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title="Mirror / Flip Camera"
          >
            Flip
          </button>

          {/* Toggle Display Buttons */}
          <button
            onClick={() => setShowBoxes(!showBoxes)}
            className={`px-2 py-1 rounded transition-colors flex items-center gap-1 ${
              showBoxes
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle Bounding Boxes"
          >
            <Crosshair className="w-3.5 h-3.5" />
            Boxes
          </button>

          <button
            onClick={() => setShowTrails(!showTrails)}
            className={`px-2 py-1 rounded transition-colors flex items-center gap-1 ${
              showTrails
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle Trajectory Trails"
          >
            <Zap className="w-3.5 h-3.5" />
            Trails
          </button>

          {/* Start / Stop Stream */}
          {!isStreaming ? (
            <button
              onClick={startCamera}
              className="px-3 py-1 rounded bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold transition-all flex items-center gap-1.5 shadow-sm"
            >
              <Camera className="w-3.5 h-3.5" />
              Start Webcam
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="px-3 py-1 rounded bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 font-bold transition-all flex items-center gap-1.5"
            >
              <CameraOff className="w-3.5 h-3.5" />
              Stop Camera
            </button>
          )}

          {/* Record 5s Clip */}
          {isStreaming && (
            <button
              onClick={startRecordingClip}
              disabled={isRecording}
              className={`px-2.5 py-1 rounded transition-colors flex items-center gap-1 text-xs ${
                isRecording
                  ? "bg-rose-500 text-white animate-pulse"
                  : "bg-slate-800 hover:bg-slate-700 text-slate-200"
              }`}
              title="Record a 5-second test clip from live feed"
            >
              <Video className="w-3.5 h-3.5 text-rose-400" />
              {isRecording ? "Recording 5s..." : "Record Clip"}
            </button>
          )}
        </div>
      </div>

      {/* Main Viewport & Canvas Area */}
      <div className="relative w-full aspect-video bg-black flex items-center justify-center overflow-hidden select-none">
        {/* Hidden video element feeding canvas */}
        <video ref={videoRef} playsInline muted className="hidden" />

        {isStreaming ? (
          <canvas
            ref={canvasRef}
            width={960}
            height={540}
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            className="w-full h-full object-contain cursor-grab active:cursor-grabbing"
          />
        ) : (
          <div className="flex flex-col items-center justify-center p-8 text-center max-w-md space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
              <Camera className="w-8 h-8" />
            </div>

            <div>
              <h4 className="text-base font-semibold text-slate-100">
                Live Optical Camera Stream Standby
              </h4>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                Use your device's webcam to simulate a drone optical sensor feed. Wave your hand, pen, or objects across the tripwire to verify spatial line crossing triggers in real-time.
              </p>
            </div>

            {cameraError && (
              <div className="p-3 rounded-lg bg-rose-950/60 border border-rose-800 text-rose-300 text-xs font-mono text-left flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
                <span>{cameraError}</span>
              </div>
            )}

            <button
              onClick={startCamera}
              className="px-5 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs font-mono transition-all flex items-center gap-2 shadow-lg shadow-cyan-500/20"
            >
              <Camera className="w-4 h-4" />
              Initialize Drone Camera Feed
            </button>
          </div>
        )}

        {/* Interactive Tripwire Repositioning Overlay Prompt */}
        {isStreaming && (
          <div className="absolute top-3 right-3 bg-slate-950/80 backdrop-blur border border-slate-800 rounded-lg px-3 py-2 text-[11px] font-mono text-slate-300 shadow-lg pointer-events-none flex items-center gap-2">
            <Move className="w-3.5 h-3.5 text-cyan-400" />
            <span>Drag line end-circles to reposition tripwire in your room</span>
          </div>
        )}

        {/* Recorded clip banner */}
        {recordedBlobUrl && (
          <div className="absolute bottom-3 left-3 bg-slate-950/90 backdrop-blur border border-emerald-500/40 rounded-lg p-3 text-xs font-mono text-emerald-300 shadow-xl flex items-center gap-3">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <div>
              <span className="font-bold">5s Test Clip Captured!</span>
              <div className="text-[10px] text-slate-400">Ready for benchmark testing</div>
            </div>
            <a
              href={recordedBlobUrl}
              download="drone_webcam_test_clip.webm"
              className="px-2.5 py-1 rounded bg-emerald-500 text-slate-950 font-bold text-[11px] flex items-center gap-1 hover:bg-emerald-400 transition-colors"
            >
              <Download className="w-3 h-3" />
              Download
            </a>
          </div>
        )}
      </div>

      {/* Camera Tuning Deck */}
      {isStreaming && (
        <div className="px-4 py-3 bg-slate-950/90 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
          <div className="flex items-center gap-4">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
              Motion Sensitivity:
            </span>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={sensitivity}
              onChange={(e) => setSensitivity(parseInt(e.target.value))}
              className="accent-cyan-400 w-32 h-1 bg-slate-800 rounded cursor-pointer"
            />
            <span className="text-cyan-400 font-bold">{sensitivity}</span>
          </div>

          <div className="flex items-center gap-4 text-[11px] text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              Tripwire: ({tripwires[0]?.startX}, {tripwires[0]?.startY}) → ({tripwires[0]?.endX}, {tripwires[0]?.endY})
            </span>
            <span className="text-emerald-400">
              Pos: {tripwires[0]?.posLabel} | Neg: {tripwires[0]?.negLabel}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
