import { useContext, useMemo } from "react";
import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { AudioWaveform } from "lucide-react";
import type { AnalysisResult } from "../types";
import { secondsTick } from "./axis";
import { Card, NewCalcBadge } from "./common";
import { ColourScale, Heatmap } from "./Heatmap";
import { NewDisplaysContext } from "./contexts";

export function AudioPanel({ r }: { r: AnalysisResult }) {
  const docs = useContext(NewDisplaysContext);
  const wf = r.visuals.waveform;
  const feats = r.visuals.audio_features;
  const data = useMemo(
    () => (wf ? wf.min.map((lo, i) => ({ t: i / r.timeline.fps, range: [lo, wf.max[i]] as [number, number] })) : []),
    [wf, r.timeline.fps],
  );
  const tMax = data.length ? data[data.length - 1].t : 1;
  if (!wf && !feats) return null;
  return (
    <Card title="Audio the model received" icon={<AudioWaveform size={16} />}>
      {wf && (
        <div>
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">Waveform envelope (16 kHz mono, per 40 ms)</span>
            <NewCalcBadge text={docs.waveform ?? "Display only."} />
          </div>
          <div className="h-24">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: -12 }}>
                <XAxis dataKey="t" type="number" domain={[0, tMax]} tickFormatter={secondsTick} fontSize={10} />
                <YAxis domain={[-1, 1]} fontSize={10} ticks={[-1, 0, 1]} />
                <Area dataKey="range" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.35} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      {feats && feats.values.length > 0 && (
        <div className="mt-4">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">
              Log-mel filterbank energies ({feats.bands} bands, low frequencies at the bottom) — the model's audio input features
            </span>
            <NewCalcBadge text={docs.audio_features ?? "Display only."} />
          </div>
          <Heatmap values={feats.values} min={feats.min} max={feats.max} height={110} label="log-mel filterbank energies over time" />
          <div className="mt-1 flex justify-between">
            <span className="text-[11px] text-slate-500">time →</span>
            <ColourScale min={feats.min} max={feats.max} unit="log energy" />
          </div>
        </div>
      )}
    </Card>
  );
}
