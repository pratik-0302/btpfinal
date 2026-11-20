import { ScanFace } from "lucide-react";
import type { AnalysisResult } from "../types";
import { fmtPct } from "../format";
import { Card, InfoTip } from "./common";

/** The aligned 96×96 grey mouth crops AV-HuBERT actually saw (evenly sampled). */
export function MouthCrops({ r }: { r: AnalysisResult }) {
  const c = r.visuals.mouth_crops;
  const q = r.quality;
  if (!c) return null;
  return (
    <Card title="What the model looked at" icon={<ScanFace size={16} />}>
      <p className="mb-2 text-xs text-ink-2">
        {c.count} of the aligned mouth crops given to AV-HuBERT. If these do not show a mouth, the face was not found
        correctly and the verdict should not be trusted.
      </p>
      <div className="overflow-x-auto rounded-xl border border-white/10 bg-black">
        <img
          src={`data:image/png;base64,${c.png_base64}`}
          alt="sampled mouth crops"
          className="pixelated block h-24 max-w-none"
          style={{ width: `${c.count * 96}px` }}
        />
      </div>
      <div className="mt-1 flex overflow-x-auto text-[10px] tabular-nums text-ink-2">
        {c.times_s.map((t, i) => (
          <span key={i} className="shrink-0 text-center" style={{ width: 96 }}>
            {t.toFixed(1)} s
          </span>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm">
        <span className="flex items-center gap-1">
          Face found: <b>{fmtPct(q.face_detected_frac)}</b> <InfoTip k="face_detected_frac" />
        </span>
        <span className="flex items-center gap-1">
          Mouth size: <b>{q.mouth_width_px != null ? `${q.mouth_width_px.toFixed(0)} px` : "—"}</b> <InfoTip k="mouth_width_px" />
        </span>
        <span>
          Frames with several faces: <b>{q.frames_with_several_faces ?? "—"}</b>
        </span>
      </div>
    </Card>
  );
}
