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
  const cancelled = error.code === "CANCELLED";
  return (
    <section className={`card animate-rise ${cancelled ? "" : "!border-bad/25"}`} role="alert">
      <div className="flex gap-4">
        <span className={`grid size-11 shrink-0 place-items-center rounded-2xl ${cancelled ? "border border-white/10 bg-white/5 text-ink-2" : "border border-bad/30 bg-bad/10 text-bad"}`}>
          <CircleX size={24} />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-xl font-semibold tracking-tight">{cancelled ? "Analysis cancelled" : "Analysis failed"}</h2>
          <p className="mt-1.5 text-[15px] text-ink">{error.message}</p>
          {HINTS[error.code] && <p className="mt-1 text-[15px] text-ink-2">{HINTS[error.code]}</p>}
          <p className="mt-3 text-xs text-ink-3">
            Code {error.code}
            {error.stage ? ` · stage: ${error.stage}` : ""}
          </p>
          {error.detail && (
            <div className="mt-2">
              <button type="button" className="btn-link !text-xs" onClick={() => setShow(!show)}>
                {show ? "Hide" : "Show"} technical details
              </button>
              {show && (
                <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap rounded-xl border border-white/10 bg-black/30 p-3 text-[11px] text-ink-2">{error.detail}</pre>
              )}
            </div>
          )}
          <div className="mt-5 flex flex-wrap gap-2">
            {canRestart && onRestartWorker && (
              <button type="button" onClick={onRestartWorker} className="btn-primary">
                <RefreshCw size={14} /> Restart analysis worker
              </button>
            )}
            <button type="button" onClick={onReset} className={canRestart ? "btn-secondary" : "btn-primary"}>
              Choose another video
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
