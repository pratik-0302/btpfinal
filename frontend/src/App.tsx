import { useCallback, useEffect, useState } from "react";
import { Clock, FileVideoCamera, ListChecks, X } from "lucide-react";
import { ApiError, deleteJob, getJob, getModelInfo, restartWorker, uploadVideo } from "./api";
import type { AnalysisResult, ErrorInfo, ModelInfo } from "./types";
import { fmtBytes, fmtSec } from "./format";
import { useHealth } from "./hooks/useHealth";
import { PIPELINE_STAGES, useJobEvents } from "./hooks/useJobEvents";
import { Card, DefinitionsContext } from "./components/common";
import { NewDisplaysContext } from "./components/contexts";
import { Header } from "./components/Header";
import { UploadPanel } from "./components/UploadPanel";
import { VideoPreview } from "./components/VideoPreview";
import { StageProgress, type UploadProgress } from "./components/StageProgress";
import { VerdictCard } from "./components/VerdictCard";
import { ChecksPanel } from "./components/ChecksPanel";
import { AgreementChart } from "./components/AgreementChart";
import { MouthCrops } from "./components/MouthCrops";
import { AudioPanel } from "./components/AudioPanel";
import { SyncMatrixPanel } from "./components/SyncMatrixPanel";
import { DetailsPanel } from "./components/DetailsPanel";
import { ErrorPanel } from "./components/ErrorPanel";
import { AboutModel } from "./components/AboutModel";

export default function App() {
  const { health, unreachable, refresh } = useHealth();
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [about, setAbout] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [upload, setUpload] = useState<UploadProgress | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<ErrorInfo | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [restoredName, setRestoredName] = useState<string | null>(null);
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
    const target = jobId ? `${window.location.pathname}?job=${jobId}` : window.location.pathname;
    if (window.location.pathname + window.location.search !== target) window.history.replaceState(null, "", target);
  }, [jobId]);

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

  return (
    <DefinitionsContext.Provider value={modelInfo?.definitions ?? {}}>
      <NewDisplaysContext.Provider value={modelInfo?.new_displays ?? {}}>
        <Header health={health} unreachable={unreachable} onAbout={() => setAbout(true)} onRestart={doRestart} />
        <main className="mx-auto max-w-7xl space-y-4 px-4 py-5">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            {/* ---------------- left column: input and progress */}
            <div className="space-y-4">
              <Card title="Video" icon={<FileVideoCamera size={16} />}>
                {!file && restoredName ? (
                  <div className="flex items-center justify-between gap-2 text-sm">
                    <span className="min-w-0">
                      <b className="break-all">{restoredName}</b>
                      <span className="block text-xs text-slate-500">Restored after a page reload — the preview is not available.</span>
                    </span>
                    {!running && (
                      <button type="button" onClick={reset} className="shrink-0 rounded-lg border border-slate-300 px-3 py-1 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800">
                        New video
                      </button>
                    )}
                  </div>
                ) : file ? (
                  <div className="space-y-3">
                    <VideoPreview file={file} />
                    <div className="flex items-center justify-between gap-2 text-sm">
                      <span className="min-w-0 truncate" title={file.name}>
                        <b>{file.name}</b> <span className="text-slate-500">· {fmtBytes(file.size)}</span>
                      </span>
                      {!running && (
                        <button type="button" onClick={reset} className="shrink-0 rounded-lg border border-slate-300 px-3 py-1 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800">
                          New video
                        </button>
                      )}
                    </div>
                  </div>
                ) : (
                  <UploadPanel
                    disabled={!workerReady || running}
                    busyReason={workerReady ? null : busyReason}
                    extensions={modelInfo?.limits.extensions ?? []}
                    maxMb={modelInfo?.limits.max_upload_mb ?? null}
                    onFile={(f) => void start(f)}
                  />
                )}
              </Card>

              {(upload || jobId) && (
                <Card
                  title="Progress"
                  icon={<ListChecks size={16} />}
                  right={
                    running && jobId ? (
                      <button type="button" onClick={() => void cancel()} className="inline-flex items-center gap-1 rounded-lg border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800">
                        <X size={12} /> Cancel
                      </button>
                    ) : result ? (
                      <span className="inline-flex items-center gap-1 text-xs text-slate-500">
                        <Clock size={12} /> {fmtSec(result.run.processing_seconds)} on {result.run.device.toUpperCase()}
                      </span>
                    ) : null
                  }
                >
                  <StageProgress upload={upload} stages={events.stages} order={PIPELINE_STAGES} />
                  {events.connectionLost && <p className="mt-2 text-xs text-amber-600">Lost the connection to the server; retrying…</p>}
                </Card>
              )}

              {warnings.length > 0 && (
                <div className="rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
                  {warnings.map((w) => <p key={w}>{w}</p>)}
                </div>
              )}
            </div>

            {/* ---------------- right column: verdict */}
            <div className="space-y-4">
              {error && <ErrorPanel error={error} onReset={reset} onRestartWorker={doRestart} />}
              {result && <VerdictCard r={result} />}
              {result && <ChecksPanel r={result} />}
              {!result && !error && (
                <div className="card flex min-h-64 flex-col items-center justify-center gap-2 text-center text-slate-500">
                  <p className="text-base font-medium text-slate-700 dark:text-slate-300">
                    {running ? "Analysing the video…" : "Choose a video to check whether the lips match the speech."}
                  </p>
                  <p className="max-w-md text-sm">
                    The detector compares the mouth movements with the speech using AV-HuBERT and reports REAL, FAKE or
                    UNCERTAIN, with the reason and the evidence behind it. Everything runs on this computer.
                  </p>
                </div>
              )}
            </div>
          </div>

          {result && (
            <>
              <AgreementChart r={result} />
              <div className="grid gap-4 lg:grid-cols-2">
                <MouthCrops r={result} />
                <AudioPanel r={result} />
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <SyncMatrixPanel r={result} />
                <DetailsPanel r={result} />
              </div>
            </>
          )}
          <footer className="pb-6 pt-2 text-center text-xs text-slate-500">
            Results describe lip–speech agreement only and can be wrong. Videos are processed locally and deleted after analysis.
          </footer>
        </main>
        {about && <AboutModel info={modelInfo} onClose={() => setAbout(false)} />}
      </NewDisplaysContext.Provider>
    </DefinitionsContext.Provider>
  );
}
