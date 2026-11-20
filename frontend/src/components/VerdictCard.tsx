import { ShieldCheck, ShieldQuestionMark, ShieldX, TriangleExclamationPoint } from "lucide-react";
import type { AnalysisResult } from "../types";
import { verdictStyle } from "../format";
import { InfoTip, ScoreBar } from "./common";

export function VerdictCard({ r }: { r: AnalysisResult }) {
  const v = r.verdict;
  const st = verdictStyle(v.label);
  const Icon = v.label === "REAL" ? ShieldCheck : v.label === "FAKE" ? ShieldX : ShieldQuestionMark;
  const decided =
    v.decided_by === "sync" ? "the sync check" : v.decided_by === "deepfake" ? "the deepfake check" : null;
  return (
    <section className={`card ring-1 ${st.ring} ${st.bg}`}>
      <div className="flex items-start gap-4">
        <Icon size={44} className={`${st.text} shrink-0`} aria-hidden />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Verdict <InfoTip k="verdict" />
          </div>
          <h2 className={`text-2xl font-bold sm:text-3xl ${st.text}`}>{v.headline}</h2>
          <p className="mt-1 text-sm text-slate-700 dark:text-slate-300">
            <span className="font-medium">Why: </span>
            {v.reason}
          </p>
          {decided && v.label !== "REAL" && (
            <p className="mt-0.5 text-xs text-slate-500">Determined by {decided} (the higher of the two scores).</p>
          )}
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-1 flex items-center justify-between text-sm">
          <span className="flex items-center gap-1 font-medium">
            Fake score <InfoTip k="fake_score" />
          </span>
          <span className="tabular-nums">
            <span className="text-lg font-bold">{v.fake_score.toFixed(0)}</span>
            <span className="text-slate-500"> / 100 · real score {v.real_score.toFixed(0)}</span>
          </span>
        </div>
        <ScoreBar score={v.fake_score} />
        <p className="mt-2 text-xs text-slate-500">
          A display scale with the decision thresholds at 35 and 65 — not a probability or a confidence level.
        </p>
      </div>

      {(v.hard_flags.length > 0 || v.notes.length > 0) && (
        <ul className="mt-4 space-y-1.5">
          {v.hard_flags.map((n) => (
            <li key={n} className="flex gap-2 text-sm font-medium text-amber-800 dark:text-amber-300">
              <TriangleExclamationPoint size={16} className="mt-0.5 shrink-0" />
              {n}
            </li>
          ))}
          {v.notes.map((n) => (
            <li key={n} className="flex gap-2 text-sm text-slate-600 dark:text-slate-300">
              <TriangleExclamationPoint size={16} className="mt-0.5 shrink-0 text-slate-400" />
              {n}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
