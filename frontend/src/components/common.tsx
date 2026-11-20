import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { Info } from "lucide-react";
import type { Definition } from "../types";

// ---------------------------------------------------------------- definitions (served by GET /api/model)
export const DefinitionsContext = createContext<Record<string, Definition>>({});

const popover =
  "pointer-events-none invisible absolute z-30 w-72 rounded-2xl border border-white/10 bg-surface/95 p-4 text-xs font-normal normal-case leading-relaxed tracking-normal text-ink-2 opacity-0 shadow-[0_20px_50px_rgb(0_0_0/0.6)] backdrop-blur-xl transition duration-200";

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
        className="rounded-full text-ink-3 transition-colors hover:text-accent-hover focus:text-accent-hover focus:outline-none"
      >
        <Info size={14} />
      </button>
      <span
        role="tooltip"
        className={`${popover} left-1/2 top-6 -translate-x-1/2 group-focus-within:visible group-focus-within:opacity-100 group-hover:visible group-hover:opacity-100`}
      >
        {d ? (
          <>
            <span className="block text-[13px] font-semibold text-ink">{d.label}</span>
            <span className="mt-1 block">{d.meaning}</span>
            {d.not && <span className="mt-1.5 block font-medium text-warn">{d.not}</span>}
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
    <section className={`card animate-rise ${className}`}>
      {(title || right) && (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
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
      <span className="rounded-full border border-violet/30 bg-violet/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-[#c4b5fd]">
        display only
      </span>
      <span className={`${popover} right-0 top-6 group-hover:visible group-hover:opacity-100`}>{text}</span>
    </span>
  );
}

/** 0–100 scale with the REAL (≤35) / UNCERTAIN / FAKE (≥65) zones and a marker at the score. */
export function ScoreBar({ score, height = "h-2" }: { score: number; height?: string }) {
  const s = Math.max(0, Math.min(100, score));
  return (
    <div>
      <div className={`relative ${height} w-full`}>
        <div className="flex h-full w-full gap-0.5 overflow-hidden rounded-full">
          <div className="h-full bg-gradient-to-r from-good/50 to-good/30" style={{ width: "35%" }} />
          <div className="h-full bg-warn/35" style={{ width: "30%" }} />
          <div className="h-full bg-gradient-to-r from-bad/30 to-bad/55" style={{ width: "35%" }} />
        </div>
        <div
          className="absolute top-1/2 size-[18px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_0_0_3px_rgb(255_255_255/0.15),0_0_16px_rgb(255_255_255/0.5)] transition-[left] duration-700"
          style={{ left: `${s}%` }}
          aria-hidden
        >
          <span className="absolute inset-[5px] rounded-full bg-canvas" />
        </div>
      </div>
      <div className="mt-2 grid grid-cols-[35fr_30fr_35fr] text-[11px] font-medium text-ink-3">
        <span>Real · 0–35</span>
        <span className="text-center">Uncertain</span>
        <span className="text-right">65–100 · Fake</span>
      </div>
    </div>
  );
}

export function Stat({ label, value, tip }: { label: ReactNode; value: ReactNode; tip?: string }) {
  return (
    <div className="tile">
      <div className="flex items-center gap-1 text-xs text-ink-2">
        {label}
        {tip && <InfoTip k={tip} />}
      </div>
      <div className="mt-1 text-base font-semibold tracking-tight tabular-nums sm:text-[17px]">{value}</div>
    </div>
  );
}

export function KV({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <dl className="divide-y divide-hairline text-sm">
      {rows.map(([k, v], i) => (
        <div key={i} className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)] gap-x-3 py-2 first:pt-0 last:pb-0">
          <dt className="text-ink-2">{k}</dt>
          <dd className="break-words text-right tabular-nums">{v ?? "—"}</dd>
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
