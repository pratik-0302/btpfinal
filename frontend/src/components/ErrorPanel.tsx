import { useState } from "react";
import { CircleX, RefreshCw } from "lucide-react";
import type { ErrorInfo } from "../types";

const HINTS: Record<string, string> = {
  NO_FACE: "Use a video where one speaking face is clearly visible and roughly facing the camera.",
  NO_AUDIO: "The detector compares lips with speech, so the video must contain its audio track.",
  TOO_SHORT: "Use a clip of at least 1 second (ideally several seconds of speech).",
  FILE_UNREADABLE: "Convert the video to MP4 (H.264 + AAC) and try again. Installing ffmpeg lets the detector read more formats.",
  NO_VIDEO_STREAM: "The file contains no video track.",
  UNSUPPORTED_EXTENSION: "Supported: .mp4 .mov .m4v .avi .mkv .webm .3gp .mpeg .mpg .wmv .flv",
  OUT_OF_MEMORY: "Close other applications (especially browser tabs) and restart the analysis worker.",
  WORKER_CRASHED: "Restart the analysis worker, then try again.",
  BUSY: "Wait until the current analysis has finished.",
  WORKER_NOT_READY: "Wait until the model has finished loading (see the status at the top).",
};

export function ErrorPanel({ error, onRestartWorker, onReset }: {
  error: ErrorInfo;
  onRestartWorker?: () => void;
  onReset: () => void;
}) {
  const [show, setShow] = useState(false);
  const canRestart = ["OUT_OF_MEMORY", "WORKER_CRASHED"].includes(error.code);
  return (
    <section className="card border-rose-300 bg-rose-50 dark:border-rose-900 dark:bg-rose-950/30" role="alert">
      <div className="flex gap-3">
        <CircleX className="mt-0.5 shrink-0 text-rose-600" size={22} />
        <div className="min-w-0 flex-1">
          <h2 className="font-semibold text-rose-800 dark:text-rose-300">{error.code === "CANCELLED" ? "Cancelled" : "Analysis failed"}</h2>
          <p className="mt-1 text-sm">{error.message}</p>
          {HINTS[error.code] && <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{HINTS[error.code]}</p>}
          <p className="mt-2 text-xs text-slate-500">
            Code {error.code}
            {error.stage ? ` · stage: ${error.stage}` : ""}
          </p>
          {error.detail && (
            <div className="mt-2">
              <button type="button" className="text-xs underline" onClick={() => setShow(!show)}>
                {show ? "Hide" : "Show"} technical details
              </button>
              {show && (
                <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap rounded bg-white p-2 text-[11px] dark:bg-slate-900">{error.detail}</pre>
              )}
            </div>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            {canRestart && onRestartWorker && (
              <button type="button" onClick={onRestartWorker} className="inline-flex items-center gap-1 rounded-lg bg-rose-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-rose-700">
                <RefreshCw size={14} /> Restart analysis worker
              </button>
            )}
            <button type="button" onClick={onReset} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-white dark:border-slate-700 dark:hover:bg-slate-800">
              Choose another video
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
