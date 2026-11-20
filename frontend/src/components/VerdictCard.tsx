import { ShieldCheck, ShieldQuestionMark, ShieldX, TriangleExclamationPoint } from "lucide-react";
import type { AnalysisResult } from "../types";
import { InfoTip, ScoreBar } from "./common";

const TONE = {
  REAL: { text: "text-good", glow: "from-good/20", badge: "from-emerald-400 to-emerald-600 shadow-emerald-500/40", ring: "border-good/25" },
  FAKE: { text: "text-bad", glow: "from-bad/20", badge: "from-rose-400 to-red-600 shadow-red-500/40", ring: "border-bad/25" },
  UNCERTAIN: { text: "text-warn", glow: "from-warn/20", badge: "from-amber-300 to-amber-500 shadow-amber-500/40", ring: "border-warn/25" },
} as const;

export function VerdictCard({ r }: { r: AnalysisResult }) {
  const v = r.verdict;
  const tone = TONE[v.label as keyof typeof TONE] ?? TONE.UNCERTAIN;
  const Icon = v.label === "REAL" ? ShieldCheck : v.label === "FAKE" ? ShieldX : ShieldQuestionMark;
  const decided =
    v.decided_by === "sync" ? "the sync check" : v.decided_by === "deepfake" ? "the deepfake check" : null;
  return (
    <section className={`card animate-rise relative overflow-hidden !p-0 ${tone.ring}`}>
      <div className={`pointer-events-none absolute inset-x-0 top-0 h-56 bg-gradient-to-b ${tone.glow} to-transparent`} />
      <div className="relative px-6 pb-6 pt-7 sm:px-8 sm:pt-8">
        <div className="flex items-center gap-1.5">
          <span className="eyebrow">Verdict</span>
          <InfoTip k="verdict" />
        </div>
        <div className="mt-3 flex items-center gap-4">
          <span className={`grid size-14 shrink-0 place-items-center rounded-2xl bg-gradient-to-br text-white shadow-lg ${tone.badge}`}>
            <Icon size={30} strokeWidth={2} aria-hidden />
          </span>
          <h2 className={`text-3xl font-semibold leading-tight tracking-tight sm:text-4xl ${tone.text}`}>{v.headline}</h2>
        </div>
        <p className="mt-4 text-base leading-relaxed text-ink/90 sm:text-[17px]">
          <span className="font-semibold">Why: </span>
          {v.reason}
        </p>
        {decided && v.label !== "REAL" && (
          <p className="mt-1 text-sm text-ink-2">Determined by {decided} (the higher of the two scores).</p>
        )}
      </div>

      <div className="relative border-t border-hairline px-6 py-6 sm:px-8">
        <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
          <div>
            <div className="flex items-center gap-1 text-sm font-medium text-ink-2">
              Fake score <InfoTip k="fake_score" />
            </div>
            <div className="mt-1 flex items-baseline gap-1 tabular-nums">
              <span className="bg-gradient-to-b from-white to-white/60 bg-clip-text text-6xl font-semibold tracking-tighter text-transparent sm:text-7xl">{v.fake_score.toFixed(0)}</span>
              <span className="text-xl font-medium text-ink-3">/ 100</span>
            </div>
          </div>
          <div className="pb-2 text-right">
            <div className="text-sm text-ink-2">Real score</div>
            <div className="text-2xl font-semibold tracking-tight tabular-nums">{v.real_score.toFixed(0)}</div>
          </div>
        </div>
        <div className="mt-5">
          <ScoreBar score={v.fake_score} />
        </div>
        <p className="mt-4 text-xs text-ink-3">
          A display scale with the decision thresholds at 35 and 65 — not a probability or a confidence level.
        </p>
      </div>

      {(v.hard_flags.length > 0 || v.notes.length > 0) && (
        <ul className="relative space-y-2 border-t border-hairline bg-white/[0.02] px-6 py-5 sm:px-8">
          {v.hard_flags.map((n) => (
            <li key={n} className="flex gap-2.5 text-sm font-medium text-warn">
              <TriangleExclamationPoint size={16} className="mt-0.5 shrink-0" />
              {n}
            </li>
          ))}
          {v.notes.map((n) => (
            <li key={n} className="flex gap-2.5 text-sm text-ink-2">
              <TriangleExclamationPoint size={16} className="mt-0.5 shrink-0 text-ink-3" />
              {n}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
