import { useContext, useMemo } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ScanFace } from "lucide-react";
import type { AnalysisResult } from "../types";
import { secondsTick } from "./axis";
import { Card, NewCalcBadge } from "./common";
import { ColourScale, Heatmap } from "./Heatmap";
import { NewDisplaysContext } from "./contexts";

/** The detector's lip–speech agreement for every frame and audio offset, and the best offset over time. */
export function SyncMatrixPanel({ r }: { r: AnalysisResult }) {
  const docs = useContext(NewDisplaysContext);
  const m = r.visuals.sync_matrix;
  const drift = r.visuals.offset_drift ?? [];
  const range = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of m.values ?? []) for (const v of row) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    return Number.isFinite(lo) ? { lo, hi } : { lo: 0, hi: 1 };
  }, [m.values]);
  const driftData = drift.map(([t, ms]) => ({ t, ms }));
  const first = m.offsets_ms[0];
  const last = m.offsets_ms[m.offsets_ms.length - 1];

  return (
    <Card title="Agreement at every audio offset" icon={<ScanFace size={16} />}>
      {m.values && m.values.length > 0 && (
        <div>
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">
              Rows: audio offset from {first} ms (bottom) to +{last} ms (top) · columns: time. A bright horizontal line = the lips follow the audio at that offset.
            </span>
            <NewCalcBadge text={docs.sync_matrix ?? "Display only."} />
          </div>
          <Heatmap values={m.values} min={range.lo} max={range.hi} height={130} label="lip-speech cosine similarity by time and audio offset" />
          <div className="mt-1 flex justify-end">
            <ColourScale min={range.lo} max={range.hi} unit="cosine" />
          </div>
        </div>
      )}
      {driftData.length > 1 && (
        <div className="mt-4">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">Best audio offset in 1-second windows (steady = steady timing)</span>
            <NewCalcBadge text={docs.offset_drift ?? "Display only."} />
          </div>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={driftData} margin={{ top: 4, right: 12, bottom: 0, left: -6 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" strokeOpacity={0.12} />
                <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={secondsTick} fontSize={10} />
                <YAxis domain={[first, last]} fontSize={10} tickFormatter={(v: number) => `${v}`} />
                <ReferenceLine y={0} stroke="currentColor" strokeOpacity={0.4} />
                <Tooltip formatter={(v) => `${v} ms`} labelFormatter={(l) => `${Number(l).toFixed(1)} s`} contentStyle={{ fontSize: 12 }} />
                <Line type="stepAfter" dataKey="ms" name="best offset" stroke="#f59e0b" dot={{ r: 2 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </Card>
  );
}
