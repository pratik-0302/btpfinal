import { Activity } from "lucide-react";
import type { AnalysisResult } from "../types";
import { fmtNum, fmtPct, signed } from "../format";
import { Card, InfoTip, ScoreBar, Stat } from "./common";

export function ChecksPanel({ r }: { r: AnalysisResult }) {
  const s = r.sync_check;
  const d = r.deepfake_check;
  return (
    <Card title="The two checks" icon={<Activity size={16} />}>
      <div className="grid gap-5 md:grid-cols-2">
        <div>
          <div className="mb-1 flex items-center justify-between text-sm">
            <span className="flex items-center gap-1 font-medium">
              Sync check <InfoTip k="sync_score" />
            </span>
            <span className="font-bold tabular-nums">{s.score.toFixed(0)}</span>
          </div>
          <p className="mb-2 text-xs text-slate-500">Does this speech belong to these lips? (swapped, dubbed or delayed audio)</p>
          <ScoreBar score={s.score} height="h-2" />
        </div>
        <div>
          <div className="mb-1 flex items-center justify-between text-sm">
            <span className="flex items-center gap-1 font-medium">
              Deepfake check <InfoTip k="deepfake_score" />
            </span>
            <span className="font-bold tabular-nums">{d.score.toFixed(0)}</span>
          </div>
          <p className="mb-2 text-xs text-slate-500">Do lips that follow the speech move the way lip-sync deepfakes do?</p>
          <ScoreBar score={d.score} height="h-2" />
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Matched speech windows" tip="matched_windows" value={`${fmtPct(s.matched_windows)} of ${s.n_windows}`} />
        <Stat label="Sync strength z" tip="sync_z" value={fmtNum(s.z, 1)} />
        <Stat label="Audio offset" tip="best_offset_ms" value={`${signed(s.best_offset_ms)} ms`} />
        <Stat label="Speech analysed" value={`${fmtNum(s.speech_seconds, 1)} s of ${fmtNum(r.run.analysed_seconds, 1)} s`} />
      </div>
      <p className="mt-3 flex items-center gap-1 text-xs text-slate-500">
        Raw classifier outputs: sync {fmtNum(s.raw_p, 3)} · deepfake {fmtNum(d.raw_p, 3)}
        <InfoTip k="raw_p" />
      </p>
    </Card>
  );
}
