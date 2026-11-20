import { useEffect, useState, type ComponentType, type ReactNode } from "react";
import {
  Activity,
  ArrowLeft,
  AudioWaveform,
  ChevronDown,
  Clock,
  Cpu,
  FileVideoCamera,
  Info,
  LineChart,
  ListChecks,
  ScanFace,
  ShieldCheck,
  TriangleExclamationPoint,
} from "lucide-react";
import type { ModelInfo } from "../types";

/** Visual overview: how the detector works, how the verdict is formed, what each number means, and its limits. */

type Icon = ComponentType<{ size?: number; className?: string; strokeWidth?: number }>;

const STEPS: { icon: Icon; title: string; spec: string }[] = [
  { icon: FileVideoCamera, title: "Decode", spec: "25 fps · 16 kHz · first 30 s" },
  { icon: ScanFace, title: "Find the mouth", spec: "MTCNN · 96×96 mouth crops" },
  { icon: AudioWaveform, title: "AV-HuBERT", spec: "lip and speech embeddings" },
  { icon: LineChart, title: "Compare", spec: "agreement at shifts up to ±600 ms" },
  { icon: Activity, title: "Two checks", spec: "sync · deepfake" },
  { icon: ShieldCheck, title: "Verdict", spec: "REAL · UNCERTAIN · FAKE" },
];

const CHECKS: { icon: Icon; name: string; q: string; catches: string }[] = [
  { icon: AudioWaveform, name: "Sync check", q: "Does this speech belong to these lips?", catches: "swapped, dubbed or delayed audio" },
  { icon: ScanFace, name: "Deepfake check", q: "Do the lips move like a lip-sync deepfake?", catches: "generated lip movements" },
];

const VERDICT_COLOURS = { REAL: "#4ade80", UNCERTAIN: "#fbbf24", FAKE: "#f87171" } as const;

const ZONES: { k: keyof typeof VERDICT_COLOURS; from: number; to: number; range: string }[] = [
  { k: "REAL", from: 0, to: 35, range: "0 – 35" },
  { k: "UNCERTAIN", from: 35, to: 65, range: "35 – 65" },
  { k: "FAKE", from: 65, to: 100, range: "65 – 100" },
];

/** One-line summaries shown on each tile; the full server definition opens on click. */
const TERMS: Record<string, { icon: Icon; short: string }> = {
  verdict: { icon: ShieldCheck, short: "REAL ≤ 35 · UNCERTAIN · FAKE ≥ 65" },
  fake_score: { icon: Activity, short: "The higher of the two check scores, 0–100" },
  sync_score: { icon: AudioWaveform, short: "Does the audio belong to these lips?" },
  deepfake_score: { icon: ScanFace, short: "Do the lips move like a lip-sync deepfake?" },
  raw_p: { icon: Cpu, short: "Each classifier's direct output, 0–1" },
  sync_z: { icon: LineChart, short: "How clearly the lips follow this audio" },
  matched_windows: { icon: ListChecks, short: "Share of 2-second speech windows that match" },
  best_offset_ms: { icon: Clock, short: "How far the audio is shifted against the lips" },
  agreement: { icon: LineChart, short: "Lip–speech similarity for every 40 ms frame" },
  segment_p: { icon: FileVideoCamera, short: "The deepfake check for each 6-second segment" },
  face_detected_frac: { icon: ScanFace, short: "Share of frames in which a face was found" },
  mouth_width_px: { icon: ScanFace, short: "Size of the mouth in the video, in pixels" },
};

function Heading({ eyebrow, title, sub }: { eyebrow: string; title: ReactNode; sub?: string }) {
  return (
    <div className="mb-12 text-center">
      <p className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-accent-hover">
        <span className="h-px w-6 bg-gradient-to-r from-transparent to-accent-hover" />
        {eyebrow}
        <span className="h-px w-6 bg-gradient-to-l from-transparent to-accent-hover" />
      </p>
      <h2 className="mt-3 text-[1.75rem] font-semibold leading-tight tracking-[-0.02em] sm:text-[2.6rem]">{title}</h2>
      {sub && <p className="mx-auto mt-3 max-w-xl text-sm text-ink-2 sm:text-[15px]">{sub}</p>}
    </div>
  );
}

function GlossaryTile({ id, label, meaning, not, open, onToggle }: {
  id: string;
  label: string;
  meaning: string;
  not: string;
  open: boolean;
  onToggle: () => void;
}) {
  const t = TERMS[id];
  const TIcon = t?.icon ?? Activity;
  return (
    <div className={`glass-edge glass-edge-hover ${open ? "!shadow-[0_0_0_1px_rgb(129_140_248/0.22),0_24px_60px_-24px_rgb(99_102_241/0.5)]" : ""}`}>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        aria-controls={`def-${id}`}
        className="flex w-full items-center gap-4 rounded-3xl p-5 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        <span className={`icon-chip size-11 shrink-0 transition-transform duration-300 ${open ? "scale-105" : ""}`}>
          <TIcon size={18} />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-[15px] font-semibold tracking-tight">{label}</span>
          <span className="mt-0.5 block text-xs leading-snug text-ink-2">{t?.short ?? meaning}</span>
        </span>
        <span className={`grid size-7 shrink-0 place-items-center rounded-full border transition-all duration-300 ${open ? "rotate-180 border-accent-hover/40 text-accent-hover" : "border-white/10 text-ink-3"}`}>
          <ChevronDown size={14} />
        </span>
      </button>
      <div id={`def-${id}`} className="expand" data-open={open} role="region" aria-label={`${label} definition`}>
        <div>
          <div className="mx-5 border-t border-white/[0.07] pb-5 pt-4">
            <p className="text-sm leading-relaxed text-ink-2">{meaning}</p>
            {not && (
              <p className="mt-3 inline-flex items-start gap-1.5 rounded-lg bg-warn/[0.08] px-2.5 py-1.5 text-xs font-medium text-warn">
                <TriangleExclamationPoint size={13} className="mt-px shrink-0" />
                {not}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function AboutModel({ info, onClose }: { info: ModelInfo | null; onClose: () => void }) {
  const [openTerms, setOpenTerms] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const meta = info?.meta;
  const badges = [
    "AV-HuBERT Large",
    `${meta?.avhubert?.layers ?? 24} layers · dim ${meta?.avhubert?.dim ?? 1024}`,
    `Layers used ${(meta?.layers_used ?? [24, 18, 12]).join(" · ")}`,
    "2 classifiers",
    "Runs on this computer",
  ];

  return (
    <div className="pb-6 pt-8 sm:pt-12">
      <button type="button" onClick={onClose} className="btn-secondary animate-rise">
        <ArrowLeft size={15} /> Back
      </button>

      {/* ------------------------------------------------ hero */}
      <header className="animate-rise relative mt-8 overflow-hidden rounded-[2.25rem] border border-white/10 px-6 pb-14 pt-16 text-center sm:px-12 sm:pb-20 sm:pt-24">
        {/* lighting */}
        <div aria-hidden className="pointer-events-none absolute inset-0 bg-[radial-gradient(80%_60%_at_50%_0%,rgb(59_130_246/0.22),transparent_70%)]" />
        <div aria-hidden className="pointer-events-none absolute -bottom-32 left-1/2 h-64 w-[70%] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgb(139_92_246/0.28),transparent)]" />
        <div aria-hidden className="ambient-grid pointer-events-none opacity-60" />
        <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent-hover/60 to-transparent" />

        <p className="relative inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.05] px-3.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-2 backdrop-blur-xl sm:text-[11px]">
          <span className="size-1.5 rounded-full bg-gradient-to-r from-accent to-violet shadow-[0_0_10px_rgb(139_92_246)]" />
          About the model
        </p>
        <h1 className="relative mx-auto mt-7 max-w-3xl text-[2.6rem] font-semibold leading-[1.02] tracking-[-0.04em] sm:text-7xl">
          How LipSync Detector <span className="gradient-text">Works</span>
        </h1>
        <p className="relative mx-auto mt-6 max-w-xl text-base leading-relaxed text-ink-2 sm:text-lg">
          It checks one thing: do the lips agree with the speech?
        </p>
        <ul className="relative mx-auto mt-10 flex max-w-3xl flex-wrap justify-center gap-2.5">
          {badges.map((t, i) => (
            <li
              key={t}
              className={`rounded-full border px-4 py-2 text-xs font-medium backdrop-blur-xl transition-colors duration-300 hover:border-white/25 ${
                i === 0
                  ? "border-accent-hover/40 bg-accent/15 text-ink shadow-[0_0_24px_-6px_rgb(59_130_246/0.7)]"
                  : "border-white/10 bg-white/[0.05] text-ink"
              }`}
            >
              {t}
            </li>
          ))}
        </ul>
      </header>

      {/* ------------------------------------------------ pipeline timeline */}
      <section className="animate-rise rise-1 mt-28">
        <Heading eyebrow="How it works" title="From video to verdict" sub="Six steps, all running the original detector code on this computer." />
        <ol className="relative grid gap-4 md:grid-cols-6 md:gap-3">
          {/* connector: horizontal through the nodes on desktop, vertical on phones */}
          <span aria-hidden className="flow-line absolute left-[8.33%] right-[8.33%] top-7 hidden h-px md:block" />
          <span aria-hidden className="flow-line-v absolute bottom-10 left-7 top-7 w-px md:hidden" />
          {STEPS.map(({ icon: StepIcon, title, spec }, i) => (
            <li key={title} className="group relative flex items-center gap-4 md:flex-col md:items-stretch md:gap-4">
              {/* node on the line */}
              <span className="relative z-10 shrink-0 self-start rounded-2xl bg-canvas md:self-center">
                <span className="icon-chip size-14 rounded-2xl transition-transform duration-300 group-hover:scale-110">
                  <StepIcon size={22} />
                </span>
                <span className="absolute -right-2 -top-2 grid size-6 place-items-center rounded-full border border-white/15 bg-surface text-[10px] font-semibold tabular-nums text-ink">
                  {String(i + 1).padStart(2, "0")}
                </span>
              </span>
              {/* step card */}
              <div className="glass-edge glass-edge-hover flex-1 px-4 py-3.5 md:text-center">
                <p className="text-[15px] font-semibold tracking-tight">{title}</p>
                <p className="mt-1 font-mono text-[11px] leading-snug text-ink-3">{spec}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      {/* ------------------------------------------------ two checks */}
      <section className="animate-rise rise-2 mx-auto mt-32 max-w-4xl">
        <Heading eyebrow="Detection checks" title="Two questions, one score" />
        <div className="grid gap-4 sm:grid-cols-2">
          {CHECKS.map(({ icon: CIcon, name, q, catches }) => (
            <div key={name} className="glass-edge glass-edge-hover group overflow-hidden p-7 text-center">
              <div aria-hidden className="pointer-events-none absolute -top-16 left-1/2 size-48 -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgb(99_102_241/0.25),transparent)] opacity-60 transition-opacity duration-500 group-hover:opacity-100" />
              <span className="icon-chip relative mx-auto size-14 rounded-2xl">
                <CIcon size={24} />
              </span>
              <p className="relative mt-5 text-[11px] font-semibold uppercase tracking-[0.16em] text-accent-hover">{name}</p>
              <p className="relative mt-2.5 text-xl font-semibold leading-snug tracking-tight">{q}</p>
              <p className="relative mt-4 inline-flex rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-ink-2">
                Catches {catches}
              </p>
            </div>
          ))}
        </div>

        {/* merge connector */}
        <div aria-hidden className="relative mx-auto h-14">
          <span className="absolute left-1/4 right-1/4 top-0 hidden h-7 rounded-b-3xl border-x border-b border-white/15 sm:block" />
          <span className="absolute left-1/2 top-0 h-14 w-px bg-gradient-to-b from-white/15 to-accent-hover/60 sm:top-7 sm:h-7" />
        </div>
        <div className="flex justify-center">
          <span className="rounded-full bg-gradient-to-r from-accent to-violet px-6 py-2.5 text-sm font-semibold text-white shadow-[inset_0_1px_0_rgb(255_255_255/0.25),0_12px_36px_-10px_rgb(99_102_241/0.95)]">
            Fake score = the higher of the two
          </span>
        </div>
      </section>

      {/* ------------------------------------------------ verdict scale */}
      <section className="animate-rise rise-2 mx-auto mt-28 max-w-4xl">
        <Heading eyebrow="The verdict" title="Where the score lands" />
        <div className="glass-edge p-6 sm:p-10">
          {/* threshold markers */}
          <div className="relative mb-3 h-6 text-xs font-semibold tabular-nums text-ink">
            {[35, 65].map((v) => (
              <span key={v} className="absolute -translate-x-1/2" style={{ left: `${v}%` }}>
                <span className="rounded-md border border-white/15 bg-surface px-1.5 py-0.5">{v}</span>
              </span>
            ))}
          </div>
          <div className="relative flex h-16 w-full gap-[3px]">
            {ZONES.map((z, i) => (
              <div
                key={z.k}
                className={`relative flex items-center justify-center text-[11px] font-bold tracking-[0.08em] text-canvas sm:text-sm ${i === 0 ? "rounded-l-2xl" : ""} ${i === ZONES.length - 1 ? "rounded-r-2xl" : ""}`}
                style={{
                  width: `${z.to - z.from}%`,
                  background: VERDICT_COLOURS[z.k],
                  boxShadow: `0 10px 36px -12px ${VERDICT_COLOURS[z.k]}`,
                }}
              >
                {z.k}
              </div>
            ))}
            {[35, 65].map((v) => (
              <span key={v} aria-hidden className="absolute -top-2 bottom-[-8px] w-px -translate-x-1/2 bg-white/60" style={{ left: `${v}%` }} />
            ))}
          </div>
          <div className="mt-4 flex text-center text-[11px] tabular-nums text-ink-3 sm:text-xs">
            {ZONES.map((z) => (
              <span key={z.k} style={{ width: `${z.to - z.from}%` }}>
                {z.range}
              </span>
            ))}
          </div>
          <div className="mt-8 flex justify-center">
            <span className="inline-flex items-center gap-2 rounded-2xl border border-warn/25 bg-warn/[0.08] px-4 py-2.5 text-xs font-medium text-warn sm:text-sm">
              <TriangleExclamationPoint size={15} className="shrink-0" />
              A display scale — not a probability or a confidence level
            </span>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------ metrics glossary */}
      {info && (
        <section className="animate-rise rise-3 mt-32">
          <Heading eyebrow="Reference" title="Reading the numbers" sub="Open any tile for the full definition and what the number is not." />
          <div className="grid items-start gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(info.definitions).map(([k, d]) => (
              <GlossaryTile
                key={k}
                id={k}
                label={d.label}
                meaning={d.meaning}
                not={d.not}
                open={Boolean(openTerms[k])}
                onToggle={() => setOpenTerms((o) => ({ ...o, [k]: !o[k] }))}
              />
            ))}
          </div>
        </section>
      )}

      {/* ------------------------------------------------ limitations */}
      <section className="animate-rise rise-3 mx-auto mt-32 max-w-4xl">
        <div className="relative overflow-hidden rounded-[2rem] border border-warn/20 bg-gradient-to-br from-warn/[0.07] via-white/[0.02] to-transparent p-6 sm:p-10">
          <div aria-hidden className="pointer-events-none absolute -right-20 -top-20 size-64 rounded-full bg-[radial-gradient(closest-side,rgb(251_191_36/0.14),transparent)]" />
          <div className="relative flex items-start gap-4">
            <span className="grid size-12 shrink-0 place-items-center rounded-2xl border border-warn/30 bg-warn/10 text-warn shadow-[0_0_30px_-8px_rgb(251_191_36/0.6)]">
              <Info size={22} />
            </span>
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-warn">Limitations</p>
              <h2 className="mt-1.5 text-xl font-semibold tracking-tight sm:text-2xl">What it can't tell you</h2>
            </div>
          </div>
          <ul className="relative mt-7 grid gap-3 sm:grid-cols-2">
            {(info?.limitations.length
              ? info.limitations
              : [
                  "The detector only measures lip–speech agreement — it does not inspect the pixels or the sound on their own.",
                  "Only the first 30 seconds of a video are analysed.",
                  "Works best with one clearly visible speaking face.",
                ]
            ).map((l) => (
              <li key={l} className="flex gap-3 rounded-2xl border border-white/[0.06] bg-canvas/40 p-4 text-sm leading-relaxed text-ink-2">
                <TriangleExclamationPoint size={15} className="mt-0.5 shrink-0 text-warn/80" />
                {l}
              </li>
            ))}
          </ul>
          <p className="relative mt-6 text-center text-xs text-ink-3">Results can be wrong — treat a verdict as evidence, not proof.</p>
        </div>
      </section>
    </div>
  );
}
