import React from "react";
import { Compass, MoveUpRight, Activity, Gauge, CheckCircle2 } from "lucide-react";

export const SpatialMathSection: React.FC = () => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-2xl space-y-6">
      <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
        <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
          <Compass className="w-5 h-5" />
        </div>
        <div>
          <h3 className="font-semibold text-slate-100 text-sm">
            Spatial Geometry & Computer Vision Mathematics
          </h3>
          <p className="text-xs text-slate-400">
            Formulations implemented in <code className="text-cyan-300">tracker.py</code> for occlusion handling and directional tracking.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Math Card 1: CCW Segment Intersection */}
        <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs font-bold">
            <MoveUpRight className="w-4 h-4" />
            1. Counter-Clockwise (CCW) Segment Intersection
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Detects if an aerial vehicle's displacement vector <code className="text-cyan-300">AB = (P_{`{t-1}`}, P_t)</code> crosses the virtual tripwire <code className="text-emerald-300">CD = (L_1, L_2)</code>:
          </p>
          <div className="bg-slate-900 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-cyan-300">
            ccw(A, B, C) = (C_y - A_y) * (B_x - A_x) &gt; (B_y - A_y) * (C_x - A_x)
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Segments intersect if and only if:
            <br />
            <code className="text-slate-200">
              (ccw(A,C,D) != ccw(B,C,D)) &amp;&amp; (ccw(A,B,C) != ccw(A,B,D))
            </code>
          </p>
        </div>

        {/* Math Card 2: 2D Cross Product Directional Classification */}
        <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-bold">
            <Activity className="w-4 h-4" />
            2. Vector Cross Product for Directionality
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Differentiates <strong>Northbound</strong> vs <strong>Southbound</strong> (or Inbound vs Outbound) flow across any arbitrary angled line:
          </p>
          <div className="bg-slate-900 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-emerald-300">
            V_line = (L2.x - L1.x, L2.y - L1.y)
            <br />
            V_obj  = (P_t.x - P_prev.x, P_t.y - P_prev.y)
            <br />
            cross = V_line.x * V_obj.y - V_line.y * V_obj.x
          </div>
          <div className="flex items-center gap-4 text-[11px] font-mono text-slate-400">
            <span className="flex items-center gap-1 text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" /> cross &gt; 0: Positive (Northbound)
            </span>
            <span className="flex items-center gap-1 text-amber-400">
              <CheckCircle2 className="w-3.5 h-3.5" /> cross &lt; 0: Negative (Southbound)
            </span>
          </div>
        </div>

        {/* Math Card 3: Velocity Estimation via GSD */}
        <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-amber-400 font-mono text-xs font-bold">
            <Gauge className="w-4 h-4" />
            3. Ground Sampling Distance (GSD) Velocity
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Drone altitude and camera optics define the physical resolution:
          </p>
          <div className="bg-slate-900 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-amber-300">
            GSD = (Sensor_Width * Altitude) / (Focal_Length * Image_Width)
            <br />
            v_mps = (Δd_px * GSD) / (k / FPS)
            <br />
            v_kmh = v_mps * 3.6
          </div>
          <p className="text-[11px] text-slate-400">
            A sliding temporal window over 5 frames damps centroid variance caused by drone wind drift.
          </p>
        </div>

        {/* Math Card 4: ByteTrack Low-Confidence Recovery */}
        <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-purple-400 font-mono text-xs font-bold">
            <Activity className="w-4 h-4" />
            4. ByteTrack Occlusion Recovery for VisDrone
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            VisDrone footage features heavy aerial occlusions (trees, overpasses). Traditional trackers discard low-score boxes. ByteTrack retains them:
          </p>
          <ul className="text-[11px] text-slate-300 space-y-1.5 list-disc list-inside">
            <li>
              <strong>Stage 1:</strong> Matches high-confidence detections (<code className="text-purple-300">&gt; 0.5</code>) with active Kalman tracks.
            </li>
            <li>
              <strong>Stage 2:</strong> Recovers remaining unmatched tracks using low-confidence detections (<code className="text-purple-300">0.1 ~ 0.5</code>).
            </li>
            <li>
              Prevents ID switches when small cars pass under bridges or oblique shadows.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
