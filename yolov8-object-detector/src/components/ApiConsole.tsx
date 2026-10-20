import React, { useState } from "react";
import {
  Send,
  Terminal,
  FileVideo,
  Copy,
  Check,
  Code2,
  Sliders,
  Sparkles,
  Server,
  Activity,
  Layers,
} from "lucide-react";

interface ApiConsoleProps {
  tripwiresCount: number;
}

export const ApiConsole: React.FC<ApiConsoleProps> = () => {
  const [endpoint, setEndpoint] = useState<"/api/v1/track" | "/api/v1/track/stream">("/api/v1/track");
  const [serverUrl, setServerUrl] = useState<string>("http://127.0.0.1:8000");
  const [streamUrl, setStreamUrl] = useState<string>("rtsp://drone-node-04.mesh:554/live/feed");
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [gsdMeters, setGsdMeters] = useState<number>(0.045);
  const [droneFps, setDroneFps] = useState<number>(30);
  const [selectedSnippetTab, setSelectedSnippetTab] = useState<"curl" | "python-requests" | "python-httpx">("curl");
  const [copiedSnippet, setCopiedSnippet] = useState<boolean>(false);
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  const [responseJson, setResponseJson] = useState<string | null>(null);

  // Generate simulated telemetry response adhering to the exact Pydantic TrackResponse schema
  const handleExecuteRequest = () => {
    setIsExecuting(true);
    setTimeout(() => {
      const simulatedResponse = {
        status: "completed",
        video_info: {
          width: 1920,
          height: 1080,
          fps: droneFps,
          total_frames: 450,
          gsd_meters_per_px: gsdMeters,
        },
        line_summary: [
          {
            line_id: "corridor_alpha",
            name: "Main Arterial Expressway",
            total_crossings: 24,
            direction_positive_count: 14,
            direction_negative_count: 10,
            direction_positive_label: "Northbound",
            direction_negative_label: "Southbound",
            class_breakdown: {
              car: 18,
              van: 3,
              truck: 2,
              motorcycle: 1,
            },
          },
        ],
        active_tracks_count: 12,
        recent_active_tracks: [
          {
            track_id: 104,
            class_id: 2,
            class_name: "car",
            confidence: 0.94,
            bbox: {
              x1: 420.5,
              y1: 310.2,
              x2: 462.8,
              y2: 388.0,
              width: 42.3,
              height: 77.8,
              center_x: 441.6,
              center_y: 349.1,
            },
            velocity_mps: 18.2,
            velocity_kmh: 65.5,
            trajectory_points: [
              [441.6, 290.0],
              [441.6, 310.0],
              [441.6, 330.0],
              [441.6, 349.1],
            ],
          },
          {
            track_id: 107,
            class_id: 7,
            class_name: "truck",
            confidence: 0.91,
            bbox: {
              x1: 612.0,
              y1: 520.4,
              x2: 660.1,
              y2: 622.3,
              width: 48.1,
              height: 101.9,
              center_x: 636.0,
              center_y: 571.3,
            },
            velocity_mps: 14.7,
            velocity_kmh: 53.0,
            trajectory_points: [
              [636.0, 500.0],
              [636.0, 535.0],
              [636.0, 571.3],
            ],
          },
        ],
        crossing_events: [
          {
            frame_index: 84,
            timestamp_sec: 2.8,
            track_id: 104,
            class_name: "car",
            line_id: "corridor_alpha",
            direction: "Northbound",
            crossing_point: [441.6, 350.0],
            speed_kmh: 65.5,
          },
          {
            frame_index: 142,
            timestamp_sec: 4.73,
            track_id: 107,
            class_name: "truck",
            line_id: "corridor_alpha",
            direction: "Southbound",
            crossing_point: [636.0, 350.0],
            speed_kmh: 53.0,
          },
        ],
        metrics: {
          total_frames_processed: 450,
          video_duration_sec: 15.0,
          total_inference_time_ms: 12450.8,
          effective_fps: 36.14,
          avg_preprocess_ms: 1.15,
          avg_inference_ms: 11.2,
          avg_postprocess_ms: 0.95,
        },
        annotated_video_path: null,
      };

      setResponseJson(JSON.stringify(simulatedResponse, null, 2));
      setIsExecuting(false);
    }, 600);
  };

  // Generate curl snippet
  const getCurlCommand = () => {
    if (endpoint === "/api/v1/track") {
      return `curl -X POST "${serverUrl}/api/v1/track" \\
  -H "accept: application/json" \\
  -F "file=@drone_flight_visdrone.mp4;type=video/mp4" \\
  -F 'config_json={
    "confidence_threshold": ${confThreshold},
    "iou_threshold": ${iouThreshold},
    "ground_sampling_distance_m_per_px": ${gsdMeters},
    "drone_fps": ${droneFps},
    "virtual_lines": [
      {
        "line_id": "corridor_alpha",
        "name": "Main Arterial Expressway",
        "start_x": 100.0,
        "start_y": 350.0,
        "end_x": 900.0,
        "end_y": 350.0,
        "positive_label": "Northbound",
        "negative_label": "Southbound"
      }
    ],
    "generate_annotated_video": false
  }'`;
    } else {
      return `curl -X POST "${serverUrl}/api/v1/track/stream" \\
  -H "Content-Type: application/json" \\
  -d '{
    "stream_url": "${streamUrl}",
    "max_duration_sec": 15,
    "config": {
      "confidence_threshold": ${confThreshold},
      "iou_threshold": ${iouThreshold},
      "ground_sampling_distance_m_per_px": ${gsdMeters},
      "drone_fps": ${droneFps}
    }
  }'`;
    }
  };

  // Generate Python requests snippet
  const getPythonRequestsCode = () => {
    if (endpoint === "/api/v1/track") {
      return `import requests
import json

url = "${serverUrl}/api/v1/track"
config = {
    "confidence_threshold": ${confThreshold},
    "iou_threshold": ${iouThreshold},
    "ground_sampling_distance_m_per_px": ${gsdMeters},
    "drone_fps": ${droneFps},
    "virtual_lines": [
        {
            "line_id": "corridor_alpha",
            "name": "Main Arterial Expressway",
            "start_x": 100.0,
            "start_y": 350.0,
            "end_x": 900.0,
            "end_y": 350.0,
            "positive_label": "Northbound",
            "negative_label": "Southbound"
        }
    ]
}

with open("drone_flight_visdrone.mp4", "rb") as f:
    files = {"file": ("drone_flight_visdrone.mp4", f, "video/mp4")}
    data = {"config_json": json.dumps(config)}
    res = requests.post(url, files=files, data=data)

telemetry = res.json()
print("Total Crossings:", telemetry["line_summary"][0]["total_crossings"])
print("Effective FPS:", telemetry["metrics"]["effective_fps"])`;
    } else {
      return `import requests

url = "${serverUrl}/api/v1/track/stream"
payload = {
    "stream_url": "${streamUrl}",
    "max_duration_sec": 15,
    "config": {
        "confidence_threshold": ${confThreshold},
        "ground_sampling_distance_m_per_px": ${gsdMeters}
    }
}

res = requests.post(url, json=payload)
telemetry = res.json()
print("Processed Stream Telemetry:", telemetry["line_summary"])`;
    }
  };

  // Generate Python httpx snippet
  const getPythonHttpxCode = () => {
    return `import asyncio
import httpx
import json

async def track_drone():
    url = "${serverUrl}${endpoint}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Ingest file or stream payload
        response = await client.post(url, ...)
        data = response.json()
        print("Inference Speed:", data["metrics"]["avg_inference_ms"], "ms")

asyncio.run(track_drone())`;
  };

  const currentSnippet =
    selectedSnippetTab === "curl"
      ? getCurlCommand()
      : selectedSnippetTab === "python-requests"
      ? getPythonRequestsCode()
      : getPythonHttpxCode();

  const handleCopySnippet = () => {
    navigator.clipboard.writeText(currentSnippet);
    setCopiedSnippet(true);
    setTimeout(() => setCopiedSnippet(false), 2000);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col space-y-4 p-5">
      {/* Top Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Server className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-100 text-sm flex items-center gap-2">
              FastAPI Interactive Test Console
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono">
                Pydantic v2 Schema
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Test the AeroTrack YOLOv11 + ByteTrack inference endpoint and inspect structured telemetry output.
            </p>
          </div>
        </div>

        {/* Server Target Selector */}
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-slate-400">Server Host:</span>
          <input
            type="text"
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-slate-200 text-xs font-mono focus:border-cyan-500 outline-none w-52"
          />
        </div>
      </div>

      {/* Endpoint & Parameter Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: Parameter controls & Request Builder */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                Endpoint Route
              </span>
              <span className="text-[11px] text-slate-500 font-mono">HTTP POST</span>
            </div>

            {/* Endpoint pills */}
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setEndpoint("/api/v1/track")}
                className={`px-3 py-2 rounded-lg text-xs font-mono font-medium border text-left transition-all ${
                  endpoint === "/api/v1/track"
                    ? "bg-cyan-500/15 border-cyan-500/50 text-cyan-300"
                    : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                <div className="font-bold">/api/v1/track</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Multipart MP4 Video File</div>
              </button>

              <button
                onClick={() => setEndpoint("/api/v1/track/stream")}
                className={`px-3 py-2 rounded-lg text-xs font-mono font-medium border text-left transition-all ${
                  endpoint === "/api/v1/track/stream"
                    ? "bg-cyan-500/15 border-cyan-500/50 text-cyan-300"
                    : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                <div className="font-bold">/api/v1/track/stream</div>
                <div className="text-[10px] text-slate-500 mt-0.5">RTSP / HTTP Live Feed</div>
              </button>
            </div>

            {endpoint === "/api/v1/track/stream" && (
              <div>
                <label className="text-xs text-slate-400 font-mono block mb-1">
                  Stream URL (RTSP / RTMP / HTTP):
                </label>
                <input
                  type="text"
                  value={streamUrl}
                  onChange={(e) => setStreamUrl(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded px-3 py-1.5 text-slate-200 text-xs font-mono focus:border-cyan-500 outline-none"
                />
              </div>
            )}

            {/* Inference Sliders */}
            <div className="space-y-3 pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">YOLOv11 Conf Threshold:</span>
                <span className="text-cyan-400 font-bold">{confThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.80"
                step="0.05"
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 h-1 bg-slate-800 rounded-lg cursor-pointer"
              />

              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">ByteTrack IoU Threshold:</span>
                <span className="text-cyan-400 font-bold">{iouThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.2"
                max="0.8"
                step="0.05"
                value={iouThreshold}
                onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 h-1 bg-slate-800 rounded-lg cursor-pointer"
              />

              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Ground Sampling Distance (GSD):</span>
                <span className="text-cyan-400 font-bold">{gsdMeters.toFixed(3)} m/px</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.15"
                step="0.005"
                value={gsdMeters}
                onChange={(e) => setGsdMeters(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 h-1 bg-slate-800 rounded-lg cursor-pointer"
              />
            </div>

            {/* Run Button */}
            <button
              onClick={handleExecuteRequest}
              disabled={isExecuting}
              className="w-full py-2.5 px-4 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs font-mono transition-all flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20"
            >
              {isExecuting ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  Running YOLOv11 + ByteTrack Inference...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Execute {endpoint} Request
                </>
              )}
            </button>
          </div>

          {/* Snippet Selector */}
          <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Code2 className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-semibold text-slate-300 font-mono">Client Snippet</span>
              </div>
              <div className="flex items-center gap-1">
                {(["curl", "python-requests", "python-httpx"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setSelectedSnippetTab(tab)}
                    className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                      selectedSnippetTab === tab
                        ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                        : "text-slate-500 hover:text-slate-300"
                    }`}
                  >
                    {tab === "python-requests" ? "requests" : tab === "python-httpx" ? "httpx" : "curl"}
                  </button>
                ))}
                <button
                  onClick={handleCopySnippet}
                  className="ml-2 p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
                  title="Copy Snippet"
                >
                  {copiedSnippet ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>
            <pre className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-48 leading-relaxed">
              <code>{currentSnippet}</code>
            </pre>
          </div>
        </div>

        {/* Right Column: Telemetry JSON Inspector */}
        <div className="lg:col-span-7 flex flex-col bg-slate-950/80 rounded-xl border border-slate-800 overflow-hidden min-h-[460px]">
          <div className="px-4 py-2.5 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
              <span className="text-slate-200 font-semibold">Pydantic Telemetry Response Payload</span>
              {responseJson && (
                <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px]">
                  200 OK (application/json)
                </span>
              )}
            </div>

            {responseJson && (
              <button
                onClick={() => {
                  navigator.clipboard.writeText(responseJson);
                }}
                className="flex items-center gap-1 text-[11px] font-mono text-slate-400 hover:text-cyan-300"
              >
                <Copy className="w-3.5 h-3.5" />
                Copy JSON
              </button>
            )}
          </div>

          <div className="flex-1 p-4 overflow-auto font-mono text-xs text-slate-300">
            {responseJson ? (
              <pre className="text-emerald-400/90 leading-relaxed overflow-x-auto">
                <code>{responseJson}</code>
              </pre>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-2 py-16">
                <Terminal className="w-8 h-8 text-slate-600" />
                <p className="text-xs">Click "Execute Request" above to trigger simulated telemetry pipeline</p>
                <p className="text-[11px] text-slate-600">
                  Returns exact JSON parsed by Pydantic models: TrackResponse, LineCrossingSummary, InferenceMetrics
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
