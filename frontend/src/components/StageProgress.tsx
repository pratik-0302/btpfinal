import { Circle, CircleCheck, CircleX, LoaderCircle } from "lucide-react";
import type { StageView } from "../types";
import { fmtBytes, fmtSec } from "../format";
import { useNow } from "./common";

const LABELS: Record<string, { title: string; desc: string }> = {
  upload: { title: "Upload", desc: "Copying the video to the local server" },
  preflight: { title: "File check", desc: "Reading streams, duration and codecs" },
  decode: { title: "Decode video and audio", desc: "25 fps frames + 16 kHz mono audio (first 30 s)" },
  faces: { title: "Find the face and mouth", desc: "MTCNN on every 2nd frame → aligned 96×96 mouth crops" },
  embeddings: { title: "AV-HuBERT", desc: "Lip and speech embeddings (layers 24, 18, 12)" },
  scoring: { title: "Sync and deepfake checks", desc: "Agreement statistics → two classifiers → verdict" },
};

export interface UploadProgress {
  loaded: number;
  total: number;
  done: boolean;
}

/** Shows only real progress: bytes uploaded, MTCNN batches done, and elapsed time per stage. */
export function StageProgress({ upload, stages, order }: {
  upload: UploadProgress | null;
  stages: Record<string, StageView>;
  order: string[];
}) {
  const anyRunning = Object.values(stages).some((s) => s.state === "running") || (upload !== null && !upload.done);
  const now = useNow(anyRunning);

  const rows: [string, StageView, string | null][] = [];
  if (upload) {
    const pct = upload.total ? Math.floor((upload.loaded / upload.total) * 100) : 0;
    rows.push([
      "upload",
      { state: upload.done ? "done" : "running" },
      upload.done ? fmtBytes(upload.total) : `${fmtBytes(upload.loaded)} of ${fmtBytes(upload.total)} (${pct}%)`,
    ]);
  }
  for (const name of order) {
    const s = stages[name] ?? { state: "pending" };
    let extra: string | null = null;
    if (s.state === "running") {
      const el = s.startedAt ? (now - s.startedAt) / 1000 : null;
      const parts: string[] = [];
      if (s.total) parts.push(`batch ${s.done ?? 0} of ${s.total}`);
      if (s.versions && s.versions > 1) parts.push(`timing version ${s.version} of ${s.versions}`);
      if (el != null && el >= 0) parts.push(`${fmtSec(el)} elapsed`);
      extra = parts.join(" · ");
    } else if (s.state === "done" && s.seconds != null) {
      extra = fmtSec(s.seconds);
    }
    rows.push([name, s, extra]);
  }

  return (
    <ol className="relative">
      {rows.map(([name, s, extra], i) => {
        const l = LABELS[name];
        const last = i === rows.length - 1;
        return (
          <li key={name} className="relative flex items-start gap-3.5 pb-4 last:pb-0">
            {!last && (
              <span
                aria-hidden
                className={`absolute left-[10px] top-6 h-[calc(100%-1.25rem)] w-px ${s.state === "done" ? "bg-good/40" : "bg-hairline"}`}
              />
            )}
            <span className="relative mt-px shrink-0 rounded-full bg-canvas">
              {s.state === "done" && <CircleCheck size={21} className="fill-good/90 text-canvas" />}
              {s.state === "running" && <LoaderCircle size={21} className="animate-spin text-accent-hover" />}
              {s.state === "error" && <CircleX size={21} className="fill-bad text-canvas" />}
              {s.state === "pending" && <Circle size={21} className="text-white/20" />}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline justify-between gap-x-2">
                <span className={`text-[15px] font-medium tracking-tight ${s.state === "pending" ? "text-ink-3" : s.state === "running" ? "text-accent-hover" : ""}`}>
                  {l.title}
                </span>
                {extra && <span className="text-xs tabular-nums text-ink-2">{extra}</span>}
              </div>
              <p className="mt-0.5 text-xs text-ink-3">{l.desc}</p>
              {name === "upload" && upload && !upload.done ? (
                <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                  <div className="progress-shimmer h-full rounded-full bg-gradient-to-r from-accent to-violet transition-all duration-300" style={{ width: `${upload.total ? (upload.loaded / upload.total) * 100 : 0}%` }} />
                </div>
              ) : null}
              {name === "faces" && s.state === "running" && s.total ? (
                <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                  <div className="progress-shimmer h-full rounded-full bg-gradient-to-r from-accent to-violet transition-all duration-500" style={{ width: `${((s.done ?? 0) / s.total) * 100}%` }} />
                </div>
              ) : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
