import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, AudioWaveform, Clock, Cpu, FileVideoCamera, Info, ListChecks, ShieldCheck, X } from "lucide-react";
import { ApiError, deleteJob, getJob, getModelInfo, restartWorker, uploadVideo } from "./api";
import type { AnalysisResult, ErrorInfo, ModelInfo } from "./types";
import { fmtBytes, fmtSec } from "./format";
import { useHealth } from "./hooks/useHealth";
import { PIPELINE_STAGES, useJobEvents } from "./hooks/useJobEvents";
import { Card, DefinitionsContext } from "./components/common";

const FEATURES = [
  {
    icon: Cpu,
    title: "Private Processing",
    text: "Videos are processed locally on your computer. Nothing is sent to the cloud, and uploads are deleted after analysis.",
  },
  {
    icon: AudioWaveform,
    title: "Audiovisual Intelligence",
    text: "AV-HuBERT analyzes the relationship between lip movements and speech, at audio shifts of up to ±600 ms.",
  },
  {
    icon: ShieldCheck,
    title: "Transparent Results",
    text: "Understand every REAL, FAKE or UNCERTAIN verdict through its scores, the reason and the supporting evidence.",
  },
];
import { NewDisplaysContext } from "./components/contexts";
import { Header } from "./components/Header";
import { UploadPanel } from "./components/UploadPanel";
import { AVSignal } from "./components/AVSignal";
import { VideoPreview } from "./components/VideoPreview";
import { StageProgress, type UploadProgress } from "./components/StageProgress";
import { VerdictCard } from "./components/VerdictCard";
import { ChecksPanel } from "./components/ChecksPanel";
import { EvidenceGrid, EvidencePage, isEvidenceKey } from "./components/EvidencePages";

const urlFor = (jobId: string | null, view: string | null, about: boolean) => {
  const q = new URLSearchParams();
  if (jobId) q.set("job", jobId);
  if (jobId && view) q.set("view", view);
  if (about) q.set("page", "about");
  const s = q.toString();
  return window.location.pathname + (s ? `?${s}` : "");
};

const viewFromUrl = () => {
  const v = new URLSearchParams(window.location.search).get("view");
  return isEvidenceKey(v) ? v : null;
};
import { ErrorPanel } from "./components/ErrorPanel";
import { AboutModel } from "./components/AboutModel";

export default function App() {
  const { health, unreachable, refresh } = useHealth();
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [about, setAbout] = useState(() => new URLSearchParams(window.location.search).get("page") === "about");

  const [file, setFile] = useState<File | null>(null);
  const [upload, setUpload] = useState<UploadProgress | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<ErrorInfo | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [restoredName, setRestoredName] = useState<string | null>(null);
  const [view, setView] = useState<string | null>(viewFromUrl);
  const events = useJobEvents(jobId);

  // ?job=<id> in the address bar restores an analysis after a page reload (results live only in server memory)
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("job");
    if (!id) return;
    getJob(id)
      .then((j) => {
        setRestoredName(j.name);
        setJobId(j.id);
      })
      .catch(() => {
        window.history.replaceState(null, "", window.location.pathname);
        setRequestError({
          code: "NOT_FOUND",
          message: "That analysis is no longer available: results are kept only while the server is running.",
          detail: "",
          stage: null,
        });
      });
  }, []);
  useEffect(() => {
    const target = urlFor(jobId, view, about);
    if (window.location.pathname + window.location.search !== target)
      window.history.replaceState(window.history.state, "", target);
  }, [jobId, view, about]);

  // evidence sections open as their own page (?view=<key>); the browser's Back button returns to the results
  const savedScroll = useRef<number | null>(null);
  const openView = useCallback(
    (key: string) => {
      savedScroll.current = window.scrollY;
      window.history.pushState({ lsView: true }, "", urlFor(jobId, key, false));
      setView(key);
      window.scrollTo({ top: 0 });
    },
    [jobId],
  );
  const closeView = useCallback(() => {
    if (window.history.state?.lsView) window.history.back();
    else setView(null);
  }, []);
  useEffect(() => {
    const onPop = () => {
      setView(viewFromUrl());
      setAbout(new URLSearchParams(window.location.search).get("page") === "about");
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  useEffect(() => {
    if (view === null && savedScroll.current !== null) {
      const y = savedScroll.current;
      savedScroll.current = null;
      requestAnimationFrame(() => window.scrollTo({ top: y }));
    } else if (view !== null) {
      window.scrollTo({ top: 0 });
    }
  }, [view]);

  // "About the model" is its own page too (?page=about)
  const aboutScroll = useRef<number | null>(null);
  const openAbout = useCallback(() => {
    if (about) return;
    aboutScroll.current = window.scrollY;
    window.history.pushState({ lsAbout: true }, "", urlFor(jobId, view, true));
    setAbout(true);
  }, [about, jobId, view]);
  const closeAbout = useCallback(() => {
    if (window.history.state?.lsAbout) window.history.back();
    else setAbout(false);
  }, []);
  useEffect(() => {
    if (about) window.scrollTo({ top: 0 });
    else if (aboutScroll.current !== null) {
      const y = aboutScroll.current;
      aboutScroll.current = null;
      requestAnimationFrame(() => window.scrollTo({ top: y }));
    }
  }, [about]);

  const workerReady = health?.worker === "ready";

  // model information (definitions, evaluation): fetched at start, and again once the model has loaded
  const workerUp = health?.worker === "ready" || health?.worker === "busy";
  const haveMeta = Boolean(modelInfo?.meta?.device);
  useEffect(() => {
    if (!health) return;
    if (modelInfo && (haveMeta || !workerUp)) return;
    getModelInfo().then(setModelInfo).catch(() => undefined);
  }, [health, modelInfo, haveMeta, workerUp]);

  // fetch the full result when the job finishes
  useEffect(() => {
    if (events.phase !== "done" || !jobId) return;
    getJob(jobId)
      .then((j) => setResult(j.result ?? null))
      .catch((e) => setRequestError(e instanceof ApiError ? e.info : { code: "INTERNAL", message: String(e), detail: "", stage: null }));
  }, [events.phase, jobId]);

  const reset = useCallback(() => {
    setFile(null);
    setUpload(null);
    setJobId(null);
    setRequestError(null);
    setResult(null);
    setRestoredName(null);
    setView(null);
  }, []);

  const start = useCallback(async (f: File) => {
    setFile(f);
    setResult(null);
    setRequestError(null);
    setJobId(null);
    setUpload({ loaded: 0, total: f.size, done: false });
    try {
      const job = await uploadVideo(f, (loaded, total) => setUpload({ loaded, total, done: false }));
      setUpload({ loaded: f.size, total: f.size, done: true });
      setJobId(job.id);
    } catch (e) {
      setUpload((u) => (u ? { ...u, done: true } : u));
      setRequestError(e instanceof ApiError ? e.info : { code: "INTERNAL", message: String(e), detail: "", stage: "upload" });
    }
  }, []);

  const cancel = useCallback(async () => {
    if (jobId) await deleteJob(jobId).catch(() => undefined);
  }, [jobId]);

  const doRestart = useCallback(async () => {
    await restartWorker().catch(() => undefined);
    setTimeout(() => void refresh(), 300);
  }, [refresh]);

  const running = (upload !== null && !upload.done) || (jobId !== null && events.phase === "running");
  const error = requestError ?? (events.phase === "error" || events.phase === "cancelled" ? events.error : null);
  const busyReason = !health
    ? "Connecting to the local server…"
    : health.worker === "starting"
      ? "Loading the model…"
      : health.worker === "failed" || health.worker === "crashed"
        ? "The analysis worker is not available"
        : running
          ? "Analysing… (one video at a time)"
          : null;
  const warnings = [...new Set([...(events.warnings ?? []), ...(result?.warnings ?? [])])];
  const idle = !file && !restoredName && !upload && !jobId && !result && !error;

  return (
    <DefinitionsContext.Provider value={modelInfo?.definitions ?? {}}>
      <NewDisplaysContext.Provider value={modelInfo?.new_displays ?? {}}>
        <Header health={health} unreachable={unreachable} onAbout={openAbout} onRestart={doRestart} />
        <div className="ambient" aria-hidden>
          <div className="ambient-grid" />
        </div>
        <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          {about ? (
            <AboutModel info={modelInfo} onClose={closeAbout} />
          ) : idle ? (
            /* ---------------- landing: nothing chosen yet */
            <>
              <section className="relative isolate pb-12 pt-16 text-center sm:pb-16 sm:pt-24 lg:pt-32">
                {/* hero lighting */}
                <div aria-hidden className="pointer-events-none absolute left-1/2 top-0 -z-10 h-[34rem] w-[min(68rem,100vw)] -translate-x-1/2 bg-[radial-gradient(50%_55%_at_50%_35%,rgb(59_130_246/0.16),transparent_70%)]" />
                <div aria-hidden className="pointer-events-none absolute left-1/2 top-24 -z-10 h-72 w-[min(44rem,90vw)] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgb(139_92_246/0.13),transparent)] blur-2xl" />

                <span className="animate-rise inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-[9px] font-semibold uppercase tracking-[0.08em] text-ink-2 backdrop-blur-xl sm:text-xs sm:tracking-[0.18em]">
                  <span className="size-1.5 rounded-full bg-gradient-to-r from-accent to-violet shadow-[0_0_8px_rgb(139_92_246)]" />
                  AI-powered audiovisual deepfake detection
                </span>
                <h1 className="animate-rise rise-1 mx-auto mt-8 max-w-5xl text-[2.75rem] font-semibold leading-[1.02] tracking-[-0.045em] sm:text-7xl lg:text-8xl xl:text-[6.75rem]">
                  <span className="bg-gradient-to-b from-white to-white/75 bg-clip-text text-transparent">See Beyond</span> <br />
                  <span className="gradient-text">What You Hear.</span>
                </h1>
                <p className="animate-rise rise-2 mx-auto mt-7 max-w-2xl text-base leading-relaxed text-ink-2 sm:text-xl lg:max-w-3xl lg:text-[1.375rem] lg:leading-relaxed">
                  Advanced AI intelligence that analyzes the relationship between facial movements and speech to
                  uncover potential deepfakes.
                </p>
                <AVSignal className="animate-rise rise-3 mx-auto mt-12 w-full max-w-md opacity-90 sm:max-w-xl lg:mt-14" />
              </section>

              <div className="glow-border animate-rise rise-2 mx-auto max-w-4xl">
               <div className="upload-shell rounded-[calc(1.75rem-1px)] p-2.5 sm:p-3.5">
                <UploadPanel
                  large
                  disabled={!workerReady}
                  busyReason={workerReady ? null : busyReason}
                  extensions={modelInfo?.limits.extensions ?? []}
                  maxMb={modelInfo?.limits.max_upload_mb ?? null}
                  onFile={(f) => void start(f)}
                />
               </div>
              </div>

              <section aria-label="Features" className="mx-auto mt-24 max-w-6xl sm:mt-32">
                <div className="grid gap-4 lg:grid-cols-3 lg:gap-5">
                  {FEATURES.map(({ icon: Icon, title, text }, i) => (
                    <div
                      key={title}
                      onMouseMove={(e) => {
                        const r = e.currentTarget.getBoundingClientRect();
                        e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
                        e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
                      }}
                      className={`card spotlight animate-rise rise-${i + 1} group relative overflow-hidden !p-7 transition-all duration-500 ease-out hover:-translate-y-1.5 hover:border-white/20 sm:!p-8 lg:!p-9`}
                    >
                      <span className="icon-chip relative size-14 rounded-2xl transition-all duration-500 group-hover:scale-110 group-hover:shadow-[inset_0_1px_0_rgb(255_255_255/0.15),0_14px_40px_-10px_rgb(99_102_241/0.9)]">
                        <Icon size={24} />
                      </span>
                      <h3 className="relative mt-7 text-lg font-semibold tracking-tight lg:text-xl">{title}</h3>
                      <p className="relative mt-3 text-[15px] leading-relaxed text-ink-2">{text}</p>
                    </div>
                  ))}
                </div>
              </section>
            </>
          ) : view && result ? (
            /* ---------------- one evidence section on its own page */
            <EvidencePage r={result} view={view} fileName={file?.name ?? restoredName} onBack={closeView} onSelect={setView} />
          ) : (
            /* ---------------- analysis view */
            <>
              <div className="animate-rise flex flex-wrap items-end justify-between gap-4 pb-6 pt-10 sm:pt-14">
                <div className="min-w-0">
                  <p className="eyebrow flex items-center gap-2">
                    <span className={`size-1.5 rounded-full ${running ? "animate-pulse bg-accent-hover" : result ? "bg-good" : error ? "bg-bad" : "bg-ink-3"}`} />
                    {running ? "Analysing" : result ? "Analysis complete" : error ? "Analysis stopped" : "New analysis"}
                  </p>
                  <h1 className="mt-2 truncate text-2xl font-semibold tracking-tight sm:text-4xl" title={file?.name ?? restoredName ?? undefined}>
                    {file?.name ?? restoredName ?? "Choose a video"}
                  </h1>
                  <p className="mt-1 flex flex-wrap items-center gap-x-2 text-sm text-ink-2">
                    {file && <span>{fmtBytes(file.size)}</span>}
                    {!file && restoredName && <span>Restored after a page reload — the preview is not available.</span>}
                    {result && (
                      <span className="inline-flex items-center gap-1">
                        <Clock size={13} /> {fmtSec(result.run.processing_seconds)} on {result.run.device.toUpperCase()}
                      </span>
                    )}
                  </p>
                </div>
                {!running && (file || restoredName) && (
                  <button type="button" onClick={reset} className="btn-primary">
                    <FileVideoCamera size={16} /> New video
                  </button>
                )}
              </div>

              <div className="grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
                {/* left column: input and progress */}
                <div className="space-y-5">
                  {file ? (
                    <div className="card animate-rise !p-2">
                      <VideoPreview file={file} />
                    </div>
                  ) : !restoredName ? (
                    <div className="card animate-rise !p-3">
                      <UploadPanel
                        disabled={!workerReady || running}
                        busyReason={workerReady ? null : busyReason}
                        extensions={modelInfo?.limits.extensions ?? []}
                        maxMb={modelInfo?.limits.max_upload_mb ?? null}
                        onFile={(f) => void start(f)}
                      />
                    </div>
                  ) : null}

                  {(upload || jobId) && (
                    <Card
                      title="Progress"
                      icon={<ListChecks size={18} />}
                      className="rise-1"
                      right={
                        running && jobId ? (
                          <button type="button" onClick={() => void cancel()} className="btn-secondary !px-3 !py-1 !text-xs">
                            <X size={12} /> Cancel
                          </button>
                        ) : null
                      }
                    >
                      <StageProgress upload={upload} stages={events.stages} order={PIPELINE_STAGES} />
                      {events.connectionLost && <p className="mt-3 text-xs text-warn">Lost the connection to the server; retrying…</p>}
                    </Card>
                  )}

                  {warnings.length > 0 && (
                    <div className="animate-rise space-y-1 rounded-3xl border border-warn/25 bg-warn/10 p-5 text-sm text-warn">
                      {warnings.map((w) => <p key={w}>{w}</p>)}
                    </div>
                  )}
                </div>

                {/* right column: verdict */}
                <div className="space-y-5">
                  {error && <ErrorPanel error={error} onReset={reset} onRestartWorker={doRestart} />}
                  {result && <VerdictCard r={result} />}
                  {result && <ChecksPanel r={result} />}
                  {!result && !error && (
                    <div className="card animate-rise flex min-h-80 flex-col items-center justify-center gap-4 text-center">
                      {running ? (
                        <span className="relative grid size-20 place-items-center">
                          <span className="absolute inset-0 animate-ping rounded-3xl bg-violet/20 [animation-duration:2s]" />
                          <span className="icon-chip relative size-20 rounded-3xl">
                            <Activity size={32} />
                          </span>
                        </span>
                      ) : (
                        <span className="grid size-16 place-items-center rounded-2xl border border-white/10 bg-white/5 text-ink-3">
                          <ShieldCheck size={28} />
                        </span>
                      )}
                      <p className="text-xl font-semibold tracking-tight">
                        {running ? "Analysing the video…" : "Choose a video to check whether the lips match the speech."}
                      </p>
                      <p className="max-w-md text-sm leading-relaxed text-ink-2">
                        The detector compares the mouth movements with the speech using AV-HuBERT and reports REAL, FAKE
                        or UNCERTAIN, with the reason and the evidence behind it. Everything runs on this computer.
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {result && <EvidenceGrid r={result} onOpen={openView} />}
            </>
          )}
          <footer className="mt-24 border-t border-hairline sm:mt-32">
            <div className="flex flex-col gap-8 py-10 sm:py-12 md:flex-row md:items-start md:justify-between">
              <div className="flex items-start gap-3">
                <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-accent to-violet text-white shadow-[inset_0_1px_0_rgb(255_255_255/0.3)]">
                  <ShieldCheck size={18} />
                </span>
                <div>
                  <p className="text-[15px] font-semibold tracking-tight">
                    LipSync <span className="font-normal text-ink-2">Detector</span>
                  </p>
                  <p className="mt-1 text-sm text-ink-3">Audiovisual deepfake detection using lip synchronicity</p>
                </div>
              </div>
              <ul className="space-y-3 text-sm leading-relaxed text-ink-2 md:max-w-md">
                <li className="flex items-start gap-2.5">
                  <Cpu size={16} className="mt-0.5 shrink-0 text-ink-3" />
                  Videos are processed locally and deleted after analysis.
                </li>
                <li className="flex items-start gap-2.5">
                  <Info size={16} className="mt-0.5 shrink-0 text-ink-3" />
                  Results describe lip–speech agreement only and can be wrong.
                </li>
              </ul>
            </div>
          </footer>
        </main>
      </NewDisplaysContext.Provider>
    </DefinitionsContext.Provider>
  );
}
