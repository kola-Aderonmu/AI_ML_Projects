import React, { useState } from "react";
import {
  DroneVisionCanvas,
  VirtualLine,
  CrossingEventUI,
} from "./components/DroneVisionCanvas";
import { CameraStreamPreview } from "./components/CameraStreamPreview";
import { CodeViewer } from "./components/CodeViewer";
import { ApiConsole } from "./components/ApiConsole";
import { SpatialMathSection } from "./components/SpatialMathSection";
import { CODEBASE_FILES } from "./data/codebase";
import {
  Compass,
  Code2,
  Terminal,
  Activity,
  Sliders,
  Download,
  Flame,
  ArrowUpRight,
  ArrowDownRight,
  Gauge,
  Radio,
  Cpu,
  Eye,
  CheckCircle2,
  FolderArchive,
  BookOpen,
  Camera,
  Video,
  FileSpreadsheet,
} from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState<"mission" | "codebase" | "api" | "math">("mission");
  const [feedMode, setFeedMode] = useState<"simulation" | "camera">("camera");

  // Tripwires state
  const [tripwires, setTripwires] = useState<VirtualLine[]>([
    {
      id: "tripwire_1",
      name: "Highway Expressway Gate",
      startX: 120,
      startY: 270,
      endX: 840,
      endY: 270,
      posLabel: "Northbound",
      negLabel: "Southbound",
      color: "#06b6d4",
    },
  ]);

  // Telemetry metrics
  const [totalCrossings, setTotalCrossings] = useState<number>(0);
  const [northboundCount, setNorthboundCount] = useState<number>(0);
  const [southboundCount, setSouthboundCount] = useState<number>(0);
  const [speedSum, setSpeedSum] = useState<number>(0);
  const [crossingEvents, setCrossingEvents] = useState<CrossingEventUI[]>([]);

  // Drone flight parameters
  const [altitudeMeters, setAltitudeMeters] = useState<number>(75);
  const [gsdMperPx, setGsdMperPx] = useState<number>(0.045);
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [selectedClasses, setSelectedClasses] = useState<string[]>([
    "car",
    "van",
    "truck",
    "bus",
    "motorcycle",
  ]);

  const handleCrossingEvent = (event: CrossingEventUI) => {
    setTotalCrossings((prev) => prev + 1);
    if (event.direction === "Northbound") {
      setNorthboundCount((prev) => prev + 1);
    } else {
      setSouthboundCount((prev) => prev + 1);
    }
    setSpeedSum((prev) => prev + event.speedKmh);
    setCrossingEvents((prev) => [event, ...prev.slice(0, 99)]);
  };

  const handleDownloadTelemetryCsv = () => {
    if (crossingEvents.length === 0) return;

    const headers = [
      "Event_ID",
      "Timestamp",
      "Track_ID",
      "Class",
      "Virtual_Tripwire",
      "Direction",
      "Velocity_KMH",
      "Velocity_MPS",
      "Cross_Product_Classification",
    ];

    const rows = crossingEvents.map((evt) => [
      `"${evt.id}"`,
      `"${evt.timestamp}"`,
      evt.trackId,
      `"${evt.className}"`,
      `"${evt.lineName}"`,
      `"${evt.direction}"`,
      evt.speedKmh,
      (evt.speedKmh / 3.6).toFixed(2),
      `"${evt.direction === "Northbound" ? "Positive (> 0)" : "Negative (< 0)"}"`,
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const timestampStr = new Date().toISOString().replace(/[:.]/g, "-");
    link.href = url;
    link.download = `aerotrack_telemetry_events_${timestampStr}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const avgSpeed = totalCrossings > 0 ? Math.round(speedSum / totalCrossings) : 62;

  const handleDownloadAllFiles = () => {
    CODEBASE_FILES.forEach((f, idx) => {
      setTimeout(() => {
        const blob = new Blob([f.content], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = f.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      }, idx * 250);
    });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/90 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
          {/* Logo & Branding */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-slate-950 font-black">
              <Compass className="w-6 h-6 text-slate-950" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-white">AeroTrack</h1>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono font-medium">
                  YOLOv11 + ByteTrack
                </span>
                <span className="hidden sm:inline-block text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono">
                  FastAPI v1.0.0
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Aerial Computer Vision & Drone Telemetry Engine (VisDrone Optimized)
              </p>
            </div>
          </div>

          {/* Nav Tabs */}
          <nav className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800 text-xs font-mono">
            <button
              onClick={() => setActiveTab("mission")}
              className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === "mission"
                  ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Radio className="w-3.5 h-3.5" />
              <span>Mission Control</span>
            </button>

            <button
              onClick={() => setActiveTab("codebase")}
              className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === "codebase"
                  ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              <span>Python Codebase</span>
            </button>

            <button
              onClick={() => setActiveTab("api")}
              className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === "api"
                  ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Terminal className="w-3.5 h-3.5" />
              <span>API Console</span>
            </button>

            <button
              onClick={() => setActiveTab("math")}
              className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === "math"
                  ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>Spatial Math</span>
            </button>
          </nav>

          {/* Quick Download All Action */}
          <button
            onClick={handleDownloadAllFiles}
            className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono transition-colors border border-slate-700"
            title="Download complete Python backend (.py, requirements.txt, Dockerfile)"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Download All Code</span>
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 w-full space-y-6">
        {/* Tab 1: Live Drone Mission Control */}
        {activeTab === "mission" && (
          <div className="space-y-6">
            {/* Real-time KPI Stats Banner */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
                <div>
                  <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">
                    Total Crossings
                  </span>
                  <span className="text-2xl font-mono font-bold text-cyan-400">{totalCrossings}</span>
                  <span className="text-[10px] text-slate-500 block mt-0.5">Virtual tripwire triggers</span>
                </div>
                <div className="p-2.5 rounded-xl bg-cyan-500/10 text-cyan-400">
                  <Activity className="w-5 h-5" />
                </div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
                <div>
                  <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">
                    Northbound Flow
                  </span>
                  <span className="text-2xl font-mono font-bold text-emerald-400">{northboundCount}</span>
                  <span className="text-[10px] text-emerald-500/80 block mt-0.5">Positive vector (cross &gt; 0)</span>
                </div>
                <div className="p-2.5 rounded-xl bg-emerald-500/10 text-emerald-400">
                  <ArrowUpRight className="w-5 h-5" />
                </div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
                <div>
                  <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">
                    Southbound Flow
                  </span>
                  <span className="text-2xl font-mono font-bold text-amber-400">{southboundCount}</span>
                  <span className="text-[10px] text-amber-500/80 block mt-0.5">Negative vector (cross &lt; 0)</span>
                </div>
                <div className="p-2.5 rounded-xl bg-amber-500/10 text-amber-400">
                  <ArrowDownRight className="w-5 h-5" />
                </div>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
                <div>
                  <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">
                    Avg Velocity
                  </span>
                  <span className="text-2xl font-mono font-bold text-purple-400">{avgSpeed} km/h</span>
                  <span className="text-[10px] text-purple-400/80 block mt-0.5">
                    GSD: {gsdMperPx} m/px ({Math.round(avgSpeed / 3.6)} m/s)
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-purple-500/10 text-purple-400">
                  <Gauge className="w-5 h-5" />
                </div>
              </div>
            </div>

            {/* Video Feed Source Switcher */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-slate-300 font-mono flex items-center gap-1.5">
                  <Radio className="w-3.5 h-3.5 text-cyan-400" />
                  Video Ingestion Source:
                </span>
                <span className="text-[11px] text-slate-500 font-mono hidden md:inline">
                  Test virtual tripwires before uploading video
                </span>
              </div>

              <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800/80 text-xs font-mono">
                <button
                  onClick={() => setFeedMode("camera")}
                  className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                    feedMode === "camera"
                      ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Camera className="w-3.5 h-3.5" />
                  <span>Camera Stream Preview (Webcam)</span>
                </button>

                <button
                  onClick={() => setFeedMode("simulation")}
                  className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                    feedMode === "simulation"
                      ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Radio className="w-3.5 h-3.5" />
                  <span>Simulated Drone Footage (VisDrone)</span>
                </button>
              </div>
            </div>

            {/* Split Grid: Canvas on left, Flight & Tripwire Controls on right */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Drone Canvas / Camera Viewport */}
              <div className="lg:col-span-8">
                {feedMode === "camera" ? (
                  <CameraStreamPreview
                    tripwires={tripwires}
                    onUpdateTripwires={setTripwires}
                    onCrossingEvent={handleCrossingEvent}
                    gsdMperPx={gsdMperPx}
                    altitudeMeters={altitudeMeters}
                  />
                ) : (
                  <DroneVisionCanvas
                    tripwires={tripwires}
                    onUpdateTripwire={setTripwires}
                    onCrossingEvent={handleCrossingEvent}
                    altitudeMeters={altitudeMeters}
                    gsdMperPx={gsdMperPx}
                    confThreshold={confThreshold}
                    selectedClasses={selectedClasses}
                  />
                )}
              </div>

              {/* Drone Config & Tripwire Editor */}
              <div className="lg:col-span-4 space-y-4">
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <span className="text-xs font-semibold text-slate-200 font-mono flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-cyan-400" />
                      Flight & Camera Parameters
                    </span>
                    <span className="text-[10px] text-cyan-400 font-mono">GSD Engine</span>
                  </div>

                  {/* Altitude Slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-slate-400">Drone Altitude:</span>
                      <span className="text-cyan-400 font-bold">{altitudeMeters}m</span>
                    </div>
                    <input
                      type="range"
                      min="30"
                      max="150"
                      step="5"
                      value={altitudeMeters}
                      onChange={(e) => {
                        const alt = parseInt(e.target.value);
                        setAltitudeMeters(alt);
                        // Approximate GSD = altitude * 0.0006
                        setGsdMperPx(parseFloat((alt * 0.0006).toFixed(4)));
                      }}
                      className="w-full accent-cyan-400 h-1 bg-slate-800 rounded-lg cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                      <span>Low AGL (30m)</span>
                      <span>High AGL (150m)</span>
                    </div>
                  </div>

                  {/* GSD Display */}
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80 text-xs font-mono flex items-center justify-between">
                    <span className="text-slate-400">Ground Sampling Distance (GSD):</span>
                    <span className="text-emerald-400 font-bold">{gsdMperPx} m/px</span>
                  </div>

                  {/* Confidence Slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-slate-400">YOLOv11 Conf Cutoff:</span>
                      <span className="text-cyan-400 font-bold">{confThreshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.70"
                      step="0.05"
                      value={confThreshold}
                      onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                      className="w-full accent-cyan-400 h-1 bg-slate-800 rounded-lg cursor-pointer"
                    />
                  </div>

                  {/* Tripwire Settings */}
                  <div className="pt-3 border-t border-slate-800 space-y-2">
                    <div className="text-xs font-semibold text-slate-300 font-mono">
                      Active Tripwire Definition
                    </div>
                    {tripwires.map((wire) => (
                      <div key={wire.id} className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs font-mono space-y-1.5">
                        <div className="flex items-center justify-between">
                          <span className="text-cyan-300 font-medium">{wire.name}</span>
                          <span className="text-[10px] text-slate-500">{wire.id}</span>
                        </div>
                        <div className="text-[11px] text-slate-400">
                          Coords: ({wire.startX}, {wire.startY}) → ({wire.endX}, {wire.endY})
                        </div>
                        <div className="flex items-center gap-2 text-[10px]">
                          <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            + {wire.posLabel}
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                            - {wire.negLabel}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Quick Info card */}
                <div className="bg-cyan-950/20 border border-cyan-800/40 rounded-xl p-4 text-xs font-mono text-cyan-200 space-y-2">
                  <div className="font-bold flex items-center gap-1.5 text-cyan-300">
                    <Cpu className="w-4 h-4" />
                    ByteTrack Occlusion Resilience
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    Vehicles maintain consistent IDs across consecutive frames even during temporary occlusion, using two-stage Hungarian association and Kalman state prediction.
                  </p>
                </div>
              </div>
            </div>

            {/* Real-time Crossing Events Table */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
              <div className="px-4 py-3 bg-slate-950 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Radio className="w-4 h-4 text-cyan-400 animate-pulse" />
                  <span className="text-xs font-semibold text-slate-200 font-mono">
                    Live Spatial Crossing Events Stream
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 font-mono">
                    {crossingEvents.length} logged
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <span className="text-xs text-slate-500 font-mono hidden md:inline">
                    Hysteresis: Active (1.5s lockout)
                  </span>

                  <button
                    onClick={handleDownloadTelemetryCsv}
                    disabled={crossingEvents.length === 0}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-800 disabled:text-slate-500 disabled:border-slate-800 text-slate-950 font-bold text-xs font-mono transition-all shadow-sm"
                    title={
                      crossingEvents.length === 0
                        ? "No crossing events recorded yet to download"
                        : "Download telemetry events as CSV"
                    }
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    <span>Download Telemetry Log (CSV)</span>
                  </button>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-950/60 text-slate-400 text-[11px] border-b border-slate-800/80">
                    <tr>
                      <th className="px-4 py-2.5">Time</th>
                      <th className="px-4 py-2.5">Track ID</th>
                      <th className="px-4 py-2.5">Class</th>
                      <th className="px-4 py-2.5">Tripwire</th>
                      <th className="px-4 py-2.5">Direction</th>
                      <th className="px-4 py-2.5">Velocity</th>
                      <th className="px-4 py-2.5">Cross Product</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {crossingEvents.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="px-4 py-8 text-center text-slate-500 text-xs">
                          Waiting for vehicle crossing triggers... Watch vehicles cross the cyan tripwire above.
                        </td>
                      </tr>
                    ) : (
                      crossingEvents.map((evt) => (
                        <tr key={evt.id} className="hover:bg-slate-800/40 transition-colors">
                          <td className="px-4 py-2.5 text-slate-400">{evt.timestamp}</td>
                          <td className="px-4 py-2.5 text-cyan-400 font-bold">#{evt.trackId}</td>
                          <td className="px-4 py-2.5 capitalize text-slate-200">{evt.className}</td>
                          <td className="px-4 py-2.5 text-slate-300">{evt.lineName}</td>
                          <td className="px-4 py-2.5">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                evt.direction === "Northbound"
                                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                                  : "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                              }`}
                            >
                              {evt.direction}
                            </span>
                          </td>
                          <td className="px-4 py-2.5 text-purple-300 font-semibold">{evt.speedKmh} km/h</td>
                          <td className="px-4 py-2.5 text-slate-400 text-[10px]">
                            {evt.direction === "Northbound" ? "> 0 (ccw positive)" : "< 0 (ccw negative)"}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Python Codebase Explorer */}
        {activeTab === "codebase" && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <Code2 className="w-4 h-4 text-cyan-400" />
                  Complete Python Codebase for AeroTrack
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Production implementation of requirements.txt, main.py (FastAPI), tracker.py (YOLOv11 & ByteTrack), test_client.py, and README.md.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleDownloadAllFiles}
                  className="px-3.5 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs font-mono transition-colors flex items-center gap-1.5 shadow-sm"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download All Files
                </button>
              </div>
            </div>

            <CodeViewer />
          </div>
        )}

        {/* Tab 3: API Console */}
        {activeTab === "api" && (
          <div className="space-y-4">
            <ApiConsole tripwiresCount={tripwires.length} />
          </div>
        )}

        {/* Tab 4: Spatial Math & VisDrone Guide */}
        {activeTab === "math" && (
          <div className="space-y-4">
            <SpatialMathSection />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-slate-500">
          <div>
            AeroTrack Engine — Ultralytics YOLOv11 & ByteTrack Multi-Object Tracker • FastAPI
          </div>
          <div className="flex items-center gap-4">
            <span className="text-emerald-400">● Engine Status: Standby / Ready</span>
            <span>FastAPI Port: 8000</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
