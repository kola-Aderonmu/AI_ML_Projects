import React, { useEffect, useRef, useState } from "react";
import { Play, Pause, RotateCcw, Crosshair, Eye, ShieldAlert, Zap, Compass } from "lucide-react";

export interface VirtualLine {
  id: string;
  name: string;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  posLabel: string;
  negLabel: string;
  color: string;
}

export interface CrossingEventUI {
  id: string;
  timestamp: string;
  trackId: number;
  className: string;
  lineName: string;
  direction: string;
  speedKmh: number;
}

interface SimulatedVehicle {
  id: number;
  className: "car" | "van" | "truck" | "bus" | "motorcycle";
  x: number;
  y: number;
  vx: number;
  vy: number;
  targetVy: number;
  width: number;
  height: number;
  color: string;
  confidence: number;
  trail: { x: number; y: number }[];
  lastCrossedLine: Record<string, number>;
  speedKmh: number;
}

interface DroneVisionCanvasProps {
  onCrossingEvent: (event: CrossingEventUI) => void;
  tripwires: VirtualLine[];
  onUpdateTripwire: (updated: VirtualLine[]) => void;
  altitudeMeters: number;
  gsdMperPx: number;
  confThreshold: number;
  selectedClasses: string[];
}

export const DroneVisionCanvas: React.FC<DroneVisionCanvasProps> = ({
  onCrossingEvent,
  tripwires,
  altitudeMeters,
  gsdMperPx,
  confThreshold,
  selectedClasses,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [showTrails, setShowTrails] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [showHudOverlay, setShowHudOverlay] = useState<boolean>(true);
  const [activeVehiclesCount, setActiveVehiclesCount] = useState<number>(0);
  const [fps, setFps] = useState<number>(30);

  // Simulation state refs
  const vehiclesRef = useRef<SimulatedVehicle[]>([]);
  const nextVehicleId = useRef<number>(101);
  const animFrameId = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(performance.now());
  const frameCountRef = useRef<number>(0);
  const fpsTimerRef = useRef<number>(performance.now());

  // CCW test helper
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

  // Cross product direction
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

  // Spawn vehicle helper
  const spawnVehicle = (canvasWidth: number, canvasHeight: number): SimulatedVehicle => {
    const classes: Array<"car" | "van" | "truck" | "bus" | "motorcycle"> = [
      "car", "car", "car", "van", "truck", "bus", "motorcycle"
    ];
    const pickedClass = classes[Math.floor(Math.random() * classes.length)];
    const isSouthbound = Math.random() > 0.5;

    // Road lanes (4 lanes: 2 Southbound, 2 Northbound)
    const lanesSouth = [canvasWidth * 0.28, canvasWidth * 0.40];
    const lanesNorth = [canvasWidth * 0.60, canvasWidth * 0.72];
    const laneX = isSouthbound
      ? lanesSouth[Math.floor(Math.random() * lanesSouth.length)] + (Math.random() * 20 - 10)
      : lanesNorth[Math.floor(Math.random() * lanesNorth.length)] + (Math.random() * 20 - 10);

    const startY = isSouthbound ? -60 : canvasHeight + 60;
    const baseSpeed = isSouthbound ? (2.8 + Math.random() * 2.2) : -(2.8 + Math.random() * 2.2);

    let dims = { w: 32, h: 62 };
    let color = "#38bdf8";

    if (pickedClass === "truck") {
      dims = { w: 38, h: 96 };
      color = "#f97316";
    } else if (pickedClass === "bus") {
      dims = { w: 40, h: 108 };
      color = "#eab308";
    } else if (pickedClass === "van") {
      dims = { w: 34, h: 72 };
      color = "#a855f7";
    } else if (pickedClass === "motorcycle") {
      dims = { w: 18, h: 36 };
      color = "#ec4899";
    }

    const conf = 0.72 + Math.random() * 0.26;
    const id = nextVehicleId.current++;

    return {
      id,
      className: pickedClass,
      x: laneX,
      y: startY,
      vx: (Math.random() - 0.5) * 0.2,
      vy: baseSpeed,
      targetVy: baseSpeed,
      width: dims.w,
      height: dims.h,
      color,
      confidence: conf,
      trail: [],
      lastCrossedLine: {},
      speedKmh: Math.round(Math.abs(baseSpeed) * 22),
    };
  };

  // Reset or initialize vehicles
  const resetVehicles = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const initial: SimulatedVehicle[] = [];
    for (let i = 0; i < 9; i++) {
      const v = spawnVehicle(canvas.width, canvas.height);
      v.y = Math.random() * canvas.height;
      initial.push(v);
    }
    vehiclesRef.current = initial;
  };

  useEffect(() => {
    resetVehicles();
  }, []);

  // Main animation / inference simulation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const render = (time: number) => {
      const dt = (time - lastTimeRef.current) / 1000;
      lastTimeRef.current = time;

      // Calculate FPS
      frameCountRef.current++;
      if (time - fpsTimerRef.current >= 500) {
        setFps(Math.round((frameCountRef.current * 1000) / (time - fpsTimerRef.current)));
        frameCountRef.current = 0;
        fpsTimerRef.current = time;
      }

      const w = canvas.width;
      const h = canvas.height;

      // 1. Draw Aerial Environment (Drone Top-Down Roadway)
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, w, h);

      // Road shoulder & terrain
      ctx.fillStyle = "#1e293b";
      ctx.fillRect(w * 0.18, 0, w * 0.64, h);

      // Asphalt roadway
      ctx.fillStyle = "#182030";
      ctx.fillRect(w * 0.22, 0, w * 0.56, h);

      // Median barrier / grass divider
      ctx.fillStyle = "#334155";
      ctx.fillRect(w * 0.495, 0, w * 0.01, h);

      // Road lane dashes
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 2;
      ctx.setLineDash([20, 24]);

      // Southbound divider
      ctx.beginPath();
      ctx.moveTo(w * 0.355, 0);
      ctx.lineTo(w * 0.355, h);
      ctx.stroke();

      // Northbound divider
      ctx.beginPath();
      ctx.moveTo(w * 0.645, 0);
      ctx.lineTo(w * 0.645, h);
      ctx.stroke();
      ctx.setLineDash([]);

      // Road border solid lines
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(w * 0.23, 0);
      ctx.lineTo(w * 0.23, h);
      ctx.moveTo(w * 0.77, 0);
      ctx.lineTo(w * 0.77, h);
      ctx.stroke();

      // 2. Draw Virtual Tripwires
      tripwires.forEach((line) => {
        ctx.save();
        ctx.strokeStyle = line.color || "#06b6d4";
        ctx.lineWidth = 3;
        ctx.shadowColor = line.color || "#06b6d4";
        ctx.shadowBlur = 10;

        ctx.beginPath();
        ctx.moveTo(line.startX, line.startY);
        ctx.lineTo(line.endX, line.endY);
        ctx.stroke();

        // Draw line endpoints (anchors)
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(line.startX, line.startY, 5, 0, Math.PI * 2);
        ctx.arc(line.endX, line.endY, 5, 0, Math.PI * 2);
        ctx.fill();

        // Tripwire Label
        const midX = (line.startX + line.endX) / 2;
        const midY = (line.startY + line.endY) / 2;
        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.fillRect(midX - 70, midY - 24, 140, 20);
        ctx.strokeStyle = line.color || "#06b6d4";
        ctx.lineWidth = 1;
        ctx.strokeRect(midX - 70, midY - 24, 140, 20);

        ctx.fillStyle = "#38bdf8";
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(`${line.name}`, midX, midY - 10);

        ctx.restore();
      });

      // 3. Update & Render Vehicles
      let currentVehicles = vehiclesRef.current;

      if (isPlaying) {
        // Spawn randomly if count drops
        if (currentVehicles.length < 10 && Math.random() < 0.04) {
          currentVehicles.push(spawnVehicle(w, h));
        }

        currentVehicles = currentVehicles.filter((v) => {
          const prevX = v.x;
          const prevY = v.y;

          // Physics movement
          v.x += v.vx;
          v.y += v.vy;

          // Record trajectory trail for ByteTrack Kalman trail
          v.trail.push({ x: v.x, y: v.y });
          if (v.trail.length > 25) v.trail.shift();

          // Calculate real-time velocity in km/h based on GSD
          const pixelDisplacementPerSec = Math.abs(v.vy) * 30; // 30 fps
          const metersPerSec = pixelDisplacementPerSec * gsdMperPx;
          v.speedKmh = Math.round(metersPerSec * 3.6);

          // Spatial Math: Line crossing detection
          tripwires.forEach((line) => {
            const hasCrossed = lineIntersects(
              prevX, prevY, v.x, v.y,
              line.startX, line.startY, line.endX, line.endY
            );

            if (hasCrossed) {
              const lastTime = v.lastCrossedLine[line.id] || 0;
              const now = Date.now();
              // Hysteresis window: 1.5 seconds
              if (now - lastTime > 1500) {
                v.lastCrossedLine[line.id] = now;
                const prod = getDirectionProduct(
                  prevX, prevY, v.x, v.y,
                  line.startX, line.startY, line.endX, line.endY
                );
                const dirLabel = prod > 0 ? line.posLabel : line.negLabel;

                onCrossingEvent({
                  id: `${v.id}-${Date.now()}`,
                  timestamp: new Date().toLocaleTimeString(),
                  trackId: v.id,
                  className: v.className,
                  lineName: line.name,
                  direction: dirLabel,
                  speedKmh: v.speedKmh,
                });
              }
            }
          });

          // Cull vehicles outside viewport with buffer
          return v.y >= -100 && v.y <= h + 100;
        });

        vehiclesRef.current = currentVehicles;
      }

      setActiveVehiclesCount(currentVehicles.length);

      // Render Trajectory Trails
      if (showTrails) {
        currentVehicles.forEach((v) => {
          if (v.trail.length < 2) return;
          ctx.beginPath();
          ctx.moveTo(v.trail[0].x, v.trail[0].y);
          for (let i = 1; i < v.trail.length; i++) {
            ctx.lineTo(v.trail[i].x, v.trail[i].y);
          }
          ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
          ctx.lineWidth = 2;
          ctx.stroke();

          // Centroid dot
          ctx.fillStyle = "#38bdf8";
          ctx.beginPath();
          ctx.arc(v.x, v.y, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      // Render Vehicles & YOLOv11 Bounding Boxes
      currentVehicles.forEach((v) => {
        // Vehicle Body (Aerial top-down representation)
        ctx.save();
        ctx.translate(v.x, v.y);
        const heading = v.vy > 0 ? Math.PI : 0;
        ctx.rotate(heading);

        // Vehicle Chassis
        ctx.fillStyle = v.color;
        ctx.beginPath();
        ctx.roundRect(-v.width / 2, -v.height / 2, v.width, v.height, 4);
        ctx.fill();

        // Windshield and roof details
        ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
        ctx.fillRect(-v.width * 0.35, -v.height * 0.25, v.width * 0.7, v.height * 0.2);
        ctx.fillRect(-v.width * 0.35, v.height * 0.1, v.width * 0.7, v.height * 0.16);

        // Headlights
        ctx.fillStyle = "#fef08a";
        ctx.fillRect(-v.width * 0.4, -v.height / 2, v.width * 0.25, 3);
        ctx.fillRect(v.width * 0.15, -v.height / 2, v.width * 0.25, 3);

        ctx.restore();

        // YOLOv11 Bounding Box & ByteTrack Identifier
        if (showBoundingBoxes) {
          const bx = v.x - v.width / 2 - 4;
          const by = v.y - v.height / 2 - 4;
          const bw = v.width + 8;
          const bh = v.height + 8;

          ctx.strokeStyle = v.confidence >= confThreshold ? "#10b981" : "#f59e0b";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(bx, by, bw, bh);

          // Corner brackets for high-tech HUD look
          const bracketLen = 6;
          ctx.strokeStyle = "#38bdf8";
          ctx.lineWidth = 2;

          // Top left
          ctx.beginPath();
          ctx.moveTo(bx, by + bracketLen);
          ctx.lineTo(bx, by);
          ctx.lineTo(bx + bracketLen, by);
          ctx.stroke();

          // Bottom right
          ctx.beginPath();
          ctx.moveTo(bx + bw, by + bh - bracketLen);
          ctx.lineTo(bx + bw, by + bh);
          ctx.lineTo(bx + bw - bracketLen, by + bh);
          ctx.stroke();

          if (showLabels) {
            const tag = `#${v.id} ${v.className.toUpperCase()} ${(v.confidence * 100).toFixed(0)}%`;
            const speedTag = `${v.speedKmh} km/h`;

            ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
            ctx.fillRect(bx, by - 18, 120, 16);

            ctx.fillStyle = "#34d399";
            ctx.font = "9px 'JetBrains Mono', monospace";
            ctx.textAlign = "left";
            ctx.fillText(tag, bx + 4, by - 6);

            // Speed badge
            ctx.fillStyle = "#38bdf8";
            ctx.fillText(speedTag, bx + bw + 4, by + 12);
          }
        }
      });

      // 4. Drone HUD Crosshair & Telemetry Overlays
      if (showHudOverlay) {
        ctx.save();
        // Crosshairs in the center
        const cx = w / 2;
        const cy = h / 2;
        ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx - 30, cy);
        ctx.lineTo(cx + 30, cy);
        ctx.moveTo(cx, cy - 30);
        ctx.lineTo(cx, cy + 30);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(cx, cy, 20, 0, Math.PI * 2);
        ctx.stroke();

        // Top-left Gimbal & Drone Camera Specs
        ctx.fillStyle = "rgba(10, 15, 30, 0.85)";
        ctx.fillRect(16, 16, 230, 84);
        ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
        ctx.lineWidth = 1;
        ctx.strokeRect(16, 16, 230, 84);

        ctx.fillStyle = "#38bdf8";
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText("AEROTRACK DRONE GIMBAL 4K", 26, 34);

        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.fillText(`ALTITUDE: ${altitudeMeters}m | GSD: ${gsdMperPx}m/px`, 26, 52);
        ctx.fillText(`MODEL: YOLOv11n + ByteTrack`, 26, 68);
        ctx.fillText(`STATUS: TRACKING ACTIVE (FPS: ${fps})`, 26, 84);

        // Compass Rose Top-Right
        ctx.fillStyle = "rgba(10, 15, 30, 0.85)";
        ctx.fillRect(w - 90, 16, 74, 52);
        ctx.strokeRect(w - 90, 16, 74, 52);
        ctx.fillStyle = "#38bdf8";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText("NORTH ▲", w - 53, 34);
        ctx.fillStyle = "#94a3b8";
        ctx.fillText("HDG 002°", w - 53, 52);

        ctx.restore();
      }

      animFrameId.current = requestAnimationFrame(render);
    };

    animFrameId.current = requestAnimationFrame(render);

    return () => {
      if (animFrameId.current) {
        cancelAnimationFrame(animFrameId.current);
      }
    };
  }, [
    isPlaying,
    tripwires,
    showBoundingBoxes,
    showTrails,
    showLabels,
    showHudOverlay,
    altitudeMeters,
    gsdMperPx,
    confThreshold,
    selectedClasses,
  ]);

  return (
    <div className="flex flex-col bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      {/* Canvas Top Bar */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-slate-950/80 border-b border-slate-800 gap-2">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            LIVE INFERENCE STREAM
          </div>
          <span className="text-xs text-slate-400 font-mono">
            {activeVehiclesCount} tracklets | {fps} FPS
          </span>
        </div>

        {/* Quick Display Toggles */}
        <div className="flex items-center gap-1.5 text-xs font-mono">
          <button
            onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
            className={`px-2 py-1 rounded transition-colors flex items-center gap-1 ${
              showBoundingBoxes
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle YOLOv11 Bounding Boxes"
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
            title="Toggle ByteTrack Kalman Trajectories"
          >
            <Zap className="w-3.5 h-3.5" />
            Trails
          </button>

          <button
            onClick={() => setShowLabels(!showLabels)}
            className={`px-2 py-1 rounded transition-colors flex items-center gap-1 ${
              showLabels
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle ID & Velocity Labels"
          >
            <Eye className="w-3.5 h-3.5" />
            IDs/Speed
          </button>

          <button
            onClick={() => setShowHudOverlay(!showHudOverlay)}
            className={`px-2 py-1 rounded transition-colors flex items-center gap-1 ${
              showHudOverlay
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "bg-slate-800 text-slate-400 hover:text-slate-200"
            }`}
            title="Toggle Drone HUD Overlay"
          >
            <Compass className="w-3.5 h-3.5" />
            HUD
          </button>

          <div className="h-4 w-px bg-slate-800 mx-1"></div>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors flex items-center gap-1"
          >
            {isPlaying ? (
              <>
                <Pause className="w-3.5 h-3.5 text-amber-400" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 text-emerald-400" />
                Resume
              </>
            )}
          </button>

          <button
            onClick={resetVehicles}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Reset Simulation Tracks"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Canvas Viewport */}
      <div className="relative w-full aspect-video bg-black flex items-center justify-center overflow-hidden">
        <canvas
          ref={canvasRef}
          width={960}
          height={540}
          className="w-full h-full object-contain cursor-crosshair"
        />

        {/* Live Tripwire Legend Badge in Bottom-Right */}
        <div className="absolute bottom-3 right-3 bg-slate-950/85 backdrop-blur border border-slate-800 rounded-lg p-2.5 text-xs font-mono space-y-1 shadow-lg pointer-events-none">
          <div className="text-slate-400 font-semibold mb-1 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            ACTIVE TRIPWIRES
          </div>
          {tripwires.map((l) => (
            <div key={l.id} className="flex items-center justify-between gap-4 text-[11px]">
              <span className="text-slate-300">{l.name}</span>
              <span className="text-cyan-400">
                {l.posLabel} ↔ {l.negLabel}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
