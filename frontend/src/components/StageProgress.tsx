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
    <ol className="space-y-2">
      {rows.map(([name, s, extra]) => {
        const l = LABELS[name];
        return (
          <li key={name} className="flex items-start gap-3">
            <span className="mt-0.5 shrink-0">
              {s.state === "done" && <CircleCheck size={18} className="text-emerald-600" />}
              {s.state === "running" && <LoaderCircle size={18} className="animate-spin text-sky-600" />}
              {s.state === "error" && <CircleX size={18} className="text-rose-600" />}
              {s.state === "pending" && <Circle size={18} className="text-slate-300 dark:text-slate-600" />}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline justify-between gap-x-2">
                <span className={`text-sm font-medium ${s.state === "pending" ? "text-slate-400" : ""}`}>{l.title}</span>
                {extra && <span className="text-xs tabular-nums text-slate-500">{extra}</span>}
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400">{l.desc}</p>
              {name === "faces" && s.state === "running" && s.total ? (
                <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
                  <div className="h-full bg-sky-500 transition-all" style={{ width: `${((s.done ?? 0) / s.total) * 100}%` }} />
                </div>
              ) : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
