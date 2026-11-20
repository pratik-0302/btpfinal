import { useState, type ReactNode } from "react";
import { ChevronDown, ChevronRight, Cpu } from "lucide-react";
import type { AnalysisResult } from "../types";
import { fmtBytes, fmtNum, fmtSec } from "../format";
import { Card, InfoTip, KV } from "./common";

function Section({ title, children, open = false }: { title: string; children: ReactNode; open?: boolean }) {
  const [o, setO] = useState(open);
  return (
    <div className="border-t border-slate-200 py-2 first:border-t-0 dark:border-slate-800">
      <button type="button" onClick={() => setO(!o)} className="flex w-full items-center gap-1 text-left text-sm font-medium">
        {o ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        {title}
      </button>
      {o && <div className="mt-2 pl-5">{children}</div>}
    </div>
  );
}

export function DetailsPanel({ r }: { r: AnalysisResult }) {
  const f = r.file;
  const q = r.quality;
  const run = r.run;
  const d = r.deepfake_check;
  const th = run.bundle?.thresholds ?? {};
  return (
    <Card title="Technical details" icon={<Cpu size={16} />}>
      <Section title="Deepfake check per 6-second segment" open>
        <p className="mb-2 flex items-center gap-1 text-xs text-slate-500">
          Raw classifier output per segment; the clip's value is their average in logit space. <InfoTip k="segment_p" />
        </p>
        <div className="space-y-1">
          {d.segment_raw_p.map((p, i) => {
            const span = d.segment_spans_s?.[i];
            return (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="w-28 shrink-0 tabular-nums text-slate-500">
                  {span ? `${span[0].toFixed(1)}–${span[1].toFixed(1)} s` : `segment ${i + 1}`}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded bg-slate-200 dark:bg-slate-700">
                  <div className="h-full bg-slate-500" style={{ width: `${p * 100}%` }} />
                </div>
                <span className="w-12 text-right tabular-nums">{p.toFixed(3)}</span>
              </div>
            );
          })}
        </div>
      </Section>
      <Section title="Video file">
        <KV
          rows={[
            ["Name", f.name],
            ["Size", fmtBytes(f.size_bytes)],
            ["Container", f.container ?? "—"],
            ["Duration", fmtSec(f.duration_s ?? null)],
            ["Analysed", `${fmtNum(run.analysed_seconds, 1)} s (limit ${run.max_seconds} s)`],
            ["Resolution (stored)", f.width && f.height ? `${f.width} × ${f.height}` : "—"],
            ["Frame rate (stored)", f.stored_fps ? `${f.stored_fps.toFixed(2)} fps` : "—"],
            ["Video codec", f.video_codec ?? "—"],
            ["Audio", f.audio_codec ? `${f.audio_codec}, ${f.audio_rate ?? "?"} Hz, ${f.audio_channels ?? "?"} ch` : "—"],
          ]}
        />
      </Section>
      <Section title="Input quality and timing">
        <KV
          rows={[
            ["Face found", q.face_detected_frac != null ? `${(q.face_detected_frac * 100).toFixed(0)}% of checked frames` : "—"],
            ["Frames with several faces", q.frames_with_several_faces ?? "—"],
            ["Mouth size", q.mouth_width_px != null ? `${q.mouth_width_px.toFixed(0)} px` : "—"],
            ["Audio gaps filled", q.audio_gap_s != null ? `${q.audio_gaps ?? 0} (${q.audio_gap_s.toFixed(2)} s)` : "—"],
            ["Timing used", q.timing ?? "—"],
            ["Re-timed to", q.retimed_to_fps ? `${q.retimed_to_fps} fps` : "not re-timed"],
          ]}
        />
      </Section>
      <Section title="Run">
        <KV
          rows={[
            ["Processing time", fmtSec(run.processing_seconds)],
            ...Object.entries(run.stage_seconds ?? {}).map(([k, v]) => [`  ${k}`, fmtSec(v)] as [string, string]),
            ["AV-HuBERT device", run.device],
            ["Face detector device", run.cropper_device],
            ["Device choice", run.device_reason],
            ["Checkpoint", run.checkpoint],
            ["PyTorch / scikit-learn", `${run.torch} / ${run.sklearn}`],
          ]}
        />
      </Section>
      <Section title="Model">
        <KV
          rows={[
            ["Encoder", `AV-HuBERT Large (frozen), ${run.avhubert?.layers ?? "?"} layers, dim ${run.avhubert?.dim ?? "?"}`],
            ["Layers used", run.layers_used?.join(", ")],
            ["Sync check", run.bundle?.chosen?.["sync check"] ?? "—"],
            ["Deepfake check", run.bundle?.chosen?.["deepfake check"] ?? "—"],
            ["Sync thresholds (raw)", `REAL below ${fmtNum(th.sync_lo, 3)} · FAKE from ${fmtNum(th.sync_hi, 3)}`],
            ["Deepfake thresholds (raw)", `REAL below ${fmtNum(th.deepfake_lo, 3)} · FAKE from ${fmtNum(th.deepfake_hi, 3)}`],
            ["Bundle trained", run.bundle?.trained ?? "—"],
          ]}
        />
      </Section>
    </Card>
  );
}
