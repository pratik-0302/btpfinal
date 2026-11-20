import { CircleQuestionMark, Cpu, LoaderCircle, RefreshCw, ServerCrash, ShieldCheck } from "lucide-react";
import type { Health } from "../types";

export function Header({ health, unreachable, onAbout, onRestart }: {
  health: Health | null;
  unreachable: boolean;
  onAbout: () => void;
  onRestart: () => void;
}) {
  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="text-sky-600" size={26} />
          <div>
            <h1 className="text-base font-bold leading-tight sm:text-lg">Lip-Sync Deepfake Detector</h1>
            <p className="text-xs text-slate-500">Audiovisual deepfake detection using lip synchronicity · runs locally</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <WorkerStatus health={health} unreachable={unreachable} onRestart={onRestart} />
          <button type="button" onClick={onAbout} className="inline-flex items-center gap-1 rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800">
            <CircleQuestionMark size={16} /> About the model
          </button>
        </div>
      </div>
    </header>
  );
}

function WorkerStatus({ health, unreachable, onRestart }: { health: Health | null; unreachable: boolean; onRestart: () => void }) {
  const pill = "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium";
  if (unreachable)
    return <span className={`${pill} bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300`}><ServerCrash size={14} /> Server not running</span>;
  if (!health) return <span className={`${pill} bg-slate-100 dark:bg-slate-800`}><LoaderCircle size={14} className="animate-spin" /> Connecting…</span>;
  const w = health.worker;
  if (w === "ready" || w === "busy")
    return (
      <span className={`${pill} bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300`} title={health.meta.device_reason}>
        <Cpu size={14} /> Model ready · {health.meta.device?.toUpperCase()}
        {w === "busy" ? " · analysing" : ""}
      </span>
    );
  if (w === "starting")
    return (
      <span className={`${pill} bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300`}>
        <LoaderCircle size={14} className="animate-spin" /> Loading model: {health.loading_message || "starting"}
      </span>
    );
  return (
    <span className="flex items-center gap-2">
      <span className={`${pill} bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300`} title={health.fatal_error?.detail}>
        <ServerCrash size={14} /> {w === "failed" ? health.fatal_error?.message ?? "Model failed to load" : "Worker stopped"}
      </span>
      <button type="button" onClick={onRestart} className="inline-flex items-center gap-1 rounded-lg border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800">
        <RefreshCw size={12} /> Restart
      </button>
    </span>
  );
}
