import { useEffect, useRef, type ComponentType } from "react";
import { Activity, ArrowLeft, AudioWaveform, ChevronRight, Cpu, LineChart, ScanFace } from "lucide-react";
import type { AnalysisResult } from "../types";
import { fmtNum, fmtPct, fmtSec, signed } from "../format";
import { AgreementChart } from "./AgreementChart";
import { MouthCrops } from "./MouthCrops";
import { AudioPanel } from "./AudioPanel";
import { SyncMatrixPanel } from "./SyncMatrixPanel";
import { DetailsPanel } from "./DetailsPanel";

interface Evidence {
  key: string;
  title: string;
  short: string;
  icon: ComponentType<{ size?: number }>;
  desc: string;
  /** one line taken from the actual result */
  summary: (r: AnalysisResult) => string;
  available: (r: AnalysisResult) => boolean;
  Panel: ComponentType<{ r: AnalysisResult }>;
}

export const EVIDENCE: Evidence[] = [
  {
    key: "agreement",
    title: "Lip–speech agreement over time",
    short: "Agreement",
    icon: LineChart,
    desc: "How closely the mouth movements followed the speech, frame by frame, and which speech windows matched.",
    summary: (r) => `${fmtPct(r.sync_check.matched_windows)} of ${r.sync_check.n_windows} speech windows matched`,
    available: () => true,
    Panel: AgreementChart,
  },
  {
    key: "mouth",
    title: "What the model looked at",
    short: "Mouth crops",
    icon: ScanFace,
    desc: "The aligned mouth crops AV-HuBERT actually received, with face-detection quality.",
    summary: (r) => `Face found in ${fmtPct(r.quality.face_detected_frac)} of checked frames`,
    available: (r) => Boolean(r.visuals.mouth_crops),
    Panel: MouthCrops,
  },
  {
    key: "audio",
    title: "Audio the model received",
    short: "Audio",
    icon: AudioWaveform,
    desc: "The 16 kHz waveform and the log-mel features the model used as its audio input.",
    summary: (r) => `${fmtNum(r.sync_check.speech_seconds, 1)} s of speech in ${fmtNum(r.run.analysed_seconds, 1)} s analysed`,
    available: (r) => Boolean(r.visuals.waveform || r.visuals.audio_features),
    Panel: AudioPanel,
  },
  {
    key: "offsets",
    title: "Agreement at every audio offset",
    short: "Audio offset",
    icon: Activity,
    desc: "Whether the sound is shifted against the lips, checked at offsets from −600 to +600 ms.",
    summary: (r) => `Best audio offset ${signed(r.sync_check.best_offset_ms)} ms`,
    available: (r) => Boolean(r.visuals.sync_matrix.values?.length || (r.visuals.offset_drift?.length ?? 0) > 1),
    Panel: SyncMatrixPanel,
  },
  {
    key: "details",
    title: "Technical details",
    short: "Technical",
    icon: Cpu,
    desc: "Per-segment classifier outputs, the video file, input quality, run times, model and thresholds.",
    summary: (r) => `Analysed in ${fmtSec(r.run.processing_seconds)} on ${r.run.device.toUpperCase()}`,
    available: () => true,
    Panel: DetailsPanel,
  },
];

export const isEvidenceKey = (k: string | null): k is string => EVIDENCE.some((e) => e.key === k);

/** Clickable overview cards shown under the verdict. */
export function EvidenceGrid({ r, onOpen }: { r: AnalysisResult; onOpen: (key: string) => void }) {
  const items = EVIDENCE.filter((e) => e.available(r));
  return (
    <section className="mt-5">
      <div className="pt-10">
        <p className="eyebrow">Supporting evidence</p>
        <h2 className="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">What the model saw and heard</h2>
        <p className="mt-2 max-w-2xl text-sm text-ink-2 sm:text-[15px]">
          Open any section to see the evidence behind the verdict in detail.
        </p>
      </div>
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {items.map(({ key, title, icon: Icon, desc, summary }, i) => (
          <button
            key={key}
            type="button"
            onClick={() => onOpen(key)}
            className={`card animate-rise rise-${Math.min(i + 1, 3)} group flex flex-col !p-6 text-left transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.06] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent`}
          >
            <span className="flex items-start justify-between gap-3">
              <span className="icon-chip size-11 transition-transform duration-300 group-hover:scale-110">
                <Icon size={20} />
              </span>
              <span className="grid size-8 place-items-center rounded-full border border-white/10 text-ink-3 transition-all duration-300 group-hover:translate-x-0.5 group-hover:border-accent-hover/50 group-hover:text-accent-hover">
                <ChevronRight size={16} />
              </span>
            </span>
            <span className="mt-5 text-base font-semibold tracking-tight">{title}</span>
            <span className="mt-1.5 flex-1 text-sm leading-relaxed text-ink-2">{desc}</span>
            <span className="mt-5 rounded-xl border border-white/[0.06] bg-white/[0.03] px-3 py-2 text-xs font-medium tabular-nums text-ink">
              {summary(r)}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

/** One evidence section on its own page, with tabs to move between sections. */
export function EvidencePage({ r, view, fileName, onBack, onSelect }: {
  r: AnalysisResult;
  view: string;
  fileName: string | null;
  onBack: () => void;
  onSelect: (key: string) => void;
}) {
  const items = EVIDENCE.filter((e) => e.available(r));
  const cur = items.find((e) => e.key === view) ?? items[0];
  const idx = items.indexOf(cur);
  const next = items[idx + 1];
  const { Panel } = cur;
  const activeTab = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    activeTab.current?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [cur.key]);
  return (
    <div className="pb-4 pt-8 sm:pt-12">
      <button type="button" onClick={onBack} className="btn-secondary animate-rise">
        <ArrowLeft size={15} /> Back to results
      </button>

      <div className="animate-rise mt-6">
        <p className="eyebrow truncate">Supporting evidence{fileName ? ` · ${fileName}` : ""}</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">{cur.title}</h1>
        <p className="mt-2 max-w-2xl text-[15px] text-ink-2">{cur.desc}</p>
      </div>

      <nav aria-label="Evidence sections" className="animate-rise rise-1 -mx-4 mt-6 overflow-x-auto px-4 sm:mx-0 sm:px-0">
        <div className="glass inline-flex gap-1 rounded-full p-1">
          {items.map(({ key, short, icon: Icon }) => {
            const active = key === cur.key;
            return (
              <button
                key={key}
                ref={active ? activeTab : undefined}
                type="button"
                onClick={() => onSelect(key)}
                aria-current={active ? "page" : undefined}
                className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-3.5 py-1.5 text-sm font-medium transition-all duration-200 ${
                  active
                    ? "bg-gradient-to-r from-accent to-violet text-white shadow-[0_6px_20px_-8px_rgb(99_102_241/0.9)]"
                    : "text-ink-2 hover:bg-white/10 hover:text-ink"
                }`}
              >
                <Icon size={14} /> {short}
              </button>
            );
          })}
        </div>
      </nav>

      <div key={cur.key} className="evidence-page mt-6">
        <Panel r={r} />
      </div>

      {next && (
        <div className="mt-6 flex justify-end">
          <button type="button" onClick={() => onSelect(next.key)} className="btn-link">
            Next: {next.title} <ChevronRight size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
