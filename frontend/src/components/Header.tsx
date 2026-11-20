import { CircleQuestionMark, LoaderCircle, RefreshCw, ShieldCheck } from "lucide-react";
import type { Health } from "../types";

export function Header({ health, unreachable, onAbout, onRestart }: {
  health: Health | null;
  unreachable: boolean;
  onAbout: () => void;
  onRestart: () => void;
}) {
  return (
    <header className="sticky top-0 z-40 px-3 pt-3 sm:px-6 sm:pt-4 lg:px-8">
      <nav className="glass mx-auto flex h-14 max-w-7xl items-center justify-between gap-3 rounded-2xl pl-3 pr-3 shadow-[0_10px_40px_-15px_rgb(0_0_0/0.8)] sm:pl-4 sm:pr-5">
        <a href="/" className="group flex min-w-0 items-center gap-2.5">
          <span className="relative grid size-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-accent to-violet text-white shadow-[inset_0_1px_0_rgb(255_255_255/0.3),0_6px_20px_-6px_rgb(99_102_241/0.9)] transition-transform duration-300 group-hover:scale-105">
            <ShieldCheck size={19} strokeWidth={2.2} />
          </span>
          <span className="truncate text-[15px] font-semibold tracking-tight sm:text-base">
            LipSync <span className="font-normal text-ink-2">Detector</span>
          </span>
        </a>
        <div className="flex items-center gap-2 sm:gap-5">
          <WorkerStatus health={health} unreachable={unreachable} onRestart={onRestart} />
          <button type="button" onClick={onAbout} className="btn-link hidden md:inline-flex">
            About the Model
          </button>
          <button type="button" onClick={onAbout} aria-label="About the model" className="grid size-9 place-items-center rounded-full text-ink-2 transition-colors hover:bg-white/10 hover:text-ink md:hidden">
            <CircleQuestionMark size={19} />
          </button>
        </div>
      </nav>
    </header>
  );
}

function Dot({ color, pulse = false }: { color: string; pulse?: boolean }) {
  return (
    <span className="relative flex size-2">
      {pulse && <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-70 ${color}`} />}
      <span className={`relative inline-flex size-2 rounded-full ${color}`} />
    </span>
  );
}

function WorkerStatus({ health, unreachable, onRestart }: { health: Health | null; unreachable: boolean; onRestart: () => void }) {
  const pill = "inline-flex items-center gap-2 whitespace-nowrap rounded-full border border-white/10 bg-white/[0.05] px-3 py-1.5 text-xs font-medium text-ink-2";
  if (unreachable) return <span className={pill}><Dot color="bg-bad" /> Server not running</span>;
  if (!health) return <span className={pill}><LoaderCircle size={12} className="animate-spin" /> Connecting…</span>;
  const w = health.worker;
  if (w === "ready" || w === "busy")
    return (
      <span className={pill} title={health.meta.device_reason}>
        <Dot color={w === "busy" ? "bg-accent-hover" : "bg-good"} pulse />
        <span className="text-ink">{w === "busy" ? "Analysing" : "Model Ready"}</span>
        <span className="hidden sm:inline">· {health.meta.device?.toUpperCase()}</span>
      </span>
    );
  if (w === "starting")
    return (
      <span className={pill}>
        <LoaderCircle size={12} className="animate-spin text-accent-hover" />
        <span className="max-w-[40vw] truncate">Loading model: {health.loading_message || "starting"}</span>
      </span>
    );
  return (
    <span className="flex items-center gap-2">
      <span className={pill} title={health.fatal_error?.detail}>
        <Dot color="bg-bad" />
        <span className="max-w-[30vw] truncate">{w === "failed" ? health.fatal_error?.message ?? "Model failed to load" : "Worker stopped"}</span>
      </span>
      <button type="button" onClick={onRestart} className="btn-secondary !px-3 !py-1 !text-xs">
        <RefreshCw size={12} /> Restart
      </button>
    </span>
  );
}
