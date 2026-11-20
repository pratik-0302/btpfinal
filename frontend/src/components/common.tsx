import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { Info } from "lucide-react";
import type { Definition } from "../types";

// ---------------------------------------------------------------- definitions (served by GET /api/model)
export const DefinitionsContext = createContext<Record<string, Definition>>({});

/** Small (i) button that shows what a number means and what it is not. */
export function InfoTip({ k, text }: { k?: string; text?: string }) {
  const defs = useContext(DefinitionsContext);
  const d = k ? defs[k] : undefined;
  if (!d && !text) return null;
  return (
    <span className="group relative inline-flex align-middle">
      <button
        type="button"
        aria-label={d ? `What is ${d.label}?` : "More information"}
        className="rounded-full text-slate-400 hover:text-slate-600 focus:text-slate-600 focus:outline-none dark:hover:text-slate-200"
      >
        <Info size={14} />
      </button>
      <span
        role="tooltip"
        className="pointer-events-none invisible absolute left-1/2 top-6 z-30 w-72 -translate-x-1/2 rounded-lg border border-slate-200 bg-white p-3 text-xs font-normal normal-case leading-relaxed tracking-normal text-slate-700 opacity-0 shadow-lg transition group-focus-within:visible group-focus-within:opacity-100 group-hover:visible group-hover:opacity-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
      >
        {d ? (
          <>
            <span className="block font-semibold">{d.label}</span>
            <span className="mt-1 block">{d.meaning}</span>
            {d.not && <span className="mt-1 block font-medium text-amber-700 dark:text-amber-400">{d.not}</span>}
          </>
        ) : (
          text
        )}
      </span>
    </span>
  );
}

export function Card({ title, icon, right, children, className = "" }: {
  title?: ReactNode;
  icon?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          {title && (
            <h2 className="card-title mb-0">
              {icon}
              {title}
            </h2>
          )}
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

/** Marks a display that is computed for the UI and is not a model output. */
export function NewCalcBadge({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex">
      <span className="rounded bg-sky-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-sky-800 dark:bg-sky-900/50 dark:text-sky-300">
        display only
      </span>
      <span className="pointer-events-none invisible absolute right-0 top-6 z-30 w-72 rounded-lg border border-slate-200 bg-white p-3 text-xs normal-case text-slate-700 opacity-0 shadow-lg transition group-hover:visible group-hover:opacity-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
        {text}
      </span>
    </span>
  );
}

/** 0–100 scale with the REAL (≤35) / UNCERTAIN / FAKE (≥65) zones and a marker at the score. */
export function ScoreBar({ score, height = "h-3" }: { score: number; height?: string }) {
  const s = Math.max(0, Math.min(100, score));
  return (
    <div className="relative">
      <div className={`flex ${height} w-full overflow-hidden rounded-full`}>
        <div className="h-full bg-emerald-200 dark:bg-emerald-900" style={{ width: "35%" }} />
        <div className="h-full bg-amber-200 dark:bg-amber-900" style={{ width: "30%" }} />
        <div className="h-full bg-rose-200 dark:bg-rose-900" style={{ width: "35%" }} />
      </div>
      <div
        className="absolute -top-1 h-5 w-1.5 -translate-x-1/2 rounded-full bg-slate-900 ring-2 ring-white dark:bg-white dark:ring-slate-900"
        style={{ left: `${s}%` }}
        aria-hidden
      />
      <div className="mt-1 flex justify-between text-[10px] text-slate-500">
        <span>0 · REAL</span>
        <span style={{ marginLeft: "20%" }}>35</span>
        <span>65</span>
        <span>FAKE · 100</span>
      </div>
    </div>
  );
}

export function Stat({ label, value, tip }: { label: ReactNode; value: ReactNode; tip?: string }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2 dark:bg-slate-800/60">
      <div className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
        {label}
        {tip && <InfoTip k={tip} />}
      </div>
      <div className="mt-0.5 text-base font-semibold tabular-nums">{value}</div>
    </div>
  );
}

export function KV({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)] gap-x-3 gap-y-1 text-sm">
      {rows.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-slate-500 dark:text-slate-400">{k}</dt>
          <dd className="break-words tabular-nums">{v ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Re-renders every `ms` milliseconds while `active` (for elapsed-time displays). */
export function useNow(active: boolean, ms = 500) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setNow(Date.now()), ms);
    return () => window.clearInterval(id);
  }, [active, ms]);
  return now;
}
