import { useMemo } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { LineChart as LineIcon } from "lucide-react";
import type { AnalysisResult } from "../types";
import { secondsTick } from "./axis";
import { Card, InfoTip } from "./common";

const Z_HIT = 2.0;   // lipsync_detector.Z_HIT
const Z_MISS = 0.5;  // lipsync_detector.Z_MISS
const HOP_S = 1.0;   // windows are 2 s long, 1 s apart; colour the middle second (as notebook cell D5)

/** Lip–speech agreement per frame at the best offset, with matched / unmatched 2-second speech windows. */
export function AgreementChart({ r }: { r: AnalysisResult }) {
  const tl = r.timeline;
  const data = useMemo(
    () =>
      tl.agreement.map((a, i) => ({
        t: i / tl.fps,
        agreement: a,
        speech: tl.speech_mask[i] ? 1 : 0,
        energy: tl.energy_db[i],
      })),
    [tl],
  );
  const bands = useMemo(
    () =>
      tl.windows_s.map(([s, e], i) => {
        const mid = (s + e) / 2;
        const z = tl.window_z[i];
        const fill = z > Z_HIT ? "#22c55e" : z < Z_MISS ? "#ef4444" : "#94a3b8";
        return { x1: mid - HOP_S / 2, x2: mid + HOP_S / 2, fill, z };
      }),
    [tl],
  );
  const tMax = data.length ? data[data.length - 1].t : 1;
  const energyMin = Math.max(-90, Math.min(...tl.energy_db.filter((x) => x != null && Number.isFinite(x))));

  return (
    <Card
      title={<span className="flex items-center gap-1">Lip–speech agreement over time <InfoTip k="agreement" /></span>}
      icon={<LineIcon size={16} />}
      right={
        <div className="flex flex-wrap gap-3 text-xs text-ink-2">
          <Legend colour="#22c55e" label="lips match the speech" />
          <Legend colour="#ef4444" label="no match" />
          <Legend colour="#94a3b8" label="unclear" />
        </div>
      }
    >
      <p className="mb-2 text-xs text-ink-2">
        Audio offset used: {r.sync_check.best_offset_ms > 0 ? "+" : ""}
        {r.sync_check.best_offset_ms} ms. Coloured bands are the 2-second speech windows the sync check judged
        ({tl.windows_s.length} windows).
      </p>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" strokeOpacity={0.12} />
            {bands.map((b, i) => (
              <ReferenceArea key={i} x1={b.x1} x2={b.x2} fill={b.fill} fillOpacity={0.18} strokeOpacity={0} ifOverflow="hidden" />
            ))}
            <XAxis dataKey="t" type="number" domain={[0, tMax]} tickFormatter={secondsTick} fontSize={11} />
            <YAxis domain={[-0.2, 1]} fontSize={11} tickFormatter={(v: number) => v.toFixed(1)} />
            <ReferenceLine y={0} stroke="currentColor" strokeOpacity={0.3} />
            <Tooltip
              formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
              labelFormatter={(l) => `${Number(l).toFixed(2)} s`}
              contentStyle={{ fontSize: 12 }} labelStyle={{ color: "#94a3b8" }}
            />
            <Line type="monotone" dataKey="agreement" name="cosine agreement" stroke="#60a5fa" dot={false} strokeWidth={1.4} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-3">
        <div className="mb-1 text-xs font-medium text-ink-2">Speech level (dBFS per 40 ms) — shaded frames count as speech</div>
        <div className="h-20">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 0, right: 12, bottom: 0, left: -12 }}>
              <XAxis dataKey="t" type="number" domain={[0, tMax]} hide />
              <YAxis yAxisId="e" domain={[energyMin, 0]} fontSize={10} tickFormatter={(v: number) => v.toFixed(0)} />
              <YAxis yAxisId="s" domain={[0, 1]} hide />
              <Area yAxisId="s" type="stepAfter" dataKey="speech" fill="#a855f7" fillOpacity={0.15} stroke="none" isAnimationActive={false} />
              <Line yAxisId="e" type="monotone" dataKey="energy" stroke="#a78bfa" dot={false} strokeWidth={1} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Card>
  );
}

function Legend({ colour, label }: { colour: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: colour, opacity: 0.5 }} />
      {label}
    </span>
  );
}
