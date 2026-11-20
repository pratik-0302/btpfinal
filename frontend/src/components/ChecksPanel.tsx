import { Activity } from "lucide-react";
import type { AnalysisResult } from "../types";
import { fmtNum, fmtPct, signed } from "../format";
import { Card, InfoTip, ScoreBar, Stat } from "./common";

function CheckTile({ name, tip, score, question }: { name: string; tip: string; score: number; question: string }) {
  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.03] p-5 transition-colors duration-300 hover:border-white/15">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-1 text-[15px] font-semibold tracking-tight">
            {name} <InfoTip k={tip} />
          </div>
          <p className="mt-1 text-xs leading-relaxed text-ink-2">{question}</p>
        </div>
        <span className="text-4xl font-semibold tracking-tighter tabular-nums">{score.toFixed(0)}</span>
      </div>
      <div className="mt-4">
        <ScoreBar score={score} height="h-1.5" />
      </div>
    </div>
  );
}

export function ChecksPanel({ r }: { r: AnalysisResult }) {
  const s = r.sync_check;
  const d = r.deepfake_check;
  return (
    <Card title="The two checks" icon={<Activity size={18} />} className="rise-1">
      <div className="grid gap-3 md:grid-cols-2">
        <CheckTile name="Sync check" tip="sync_score" score={s.score}
          question="Does this speech belong to these lips? (swapped, dubbed or delayed audio)" />
        <CheckTile name="Deepfake check" tip="deepfake_score" score={d.score}
          question="Do lips that follow the speech move the way lip-sync deepfakes do?" />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Matched windows" tip="matched_windows" value={`${fmtPct(s.matched_windows)} of ${s.n_windows}`} />
        <Stat label="Sync strength z" tip="sync_z" value={fmtNum(s.z, 1)} />
        <Stat label="Audio offset" tip="best_offset_ms" value={`${signed(s.best_offset_ms)} ms`} />
        <Stat label="Speech analysed" value={`${fmtNum(s.speech_seconds, 1)} s of ${fmtNum(r.run.analysed_seconds, 1)} s`} />
      </div>
      <p className="mt-4 flex items-center gap-1 text-xs text-ink-3">
        Raw classifier outputs: sync {fmtNum(s.raw_p, 3)} · deepfake {fmtNum(d.raw_p, 3)}
        <InfoTip k="raw_p" />
      </p>
    </Card>
  );
}
