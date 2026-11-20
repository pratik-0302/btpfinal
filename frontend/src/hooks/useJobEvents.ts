import { useEffect, useRef, useState } from "react";
import type { ErrorInfo, JobEvent, StageEvent, StageName, StageView } from "../types";

export const PIPELINE_STAGES: Exclude<StageName, "upload">[] = ["preflight", "decode", "faces", "embeddings", "scoring"];

export type JobPhase = "running" | "done" | "error" | "cancelled";

export interface JobEventsState {
  stages: Record<string, StageView>;
  warnings: string[];
  phase: JobPhase;
  error: ErrorInfo | null;
  connectionLost: boolean;
}

const initial = (): JobEventsState => ({
  stages: Object.fromEntries(PIPELINE_STAGES.map((s) => [s, { state: "pending" } as StageView])),
  warnings: [],
  phase: "running",
  error: null,
  connectionLost: false,
});

function applyStage(stages: Record<string, StageView>, ev: StageEvent): Record<string, StageView> {
  const prev = stages[ev.stage] ?? { state: "pending" };
  const next: StageView = { ...prev, version: ev.version ?? prev.version, versions: ev.versions ?? prev.versions };
  if (ev.state === "running") {
    next.state = "running";
    if (ev.done === undefined) {
      next.startedAt = ev.t * 1000;
      next.done = undefined;
      next.total = ev.total ?? undefined;
    } else {
      next.done = ev.done;
      next.total = ev.total;
    }
  } else {
    next.state = "done";
    next.seconds = (prev.seconds && ev.version && ev.version > 1 ? prev.seconds : 0) + (ev.seconds ?? 0);
  }
  return { ...stages, [ev.stage]: next };
}

/** Follows /api/jobs/{id}/events (SSE). Progress comes only from real pipeline events. */
export function useJobEvents(jobId: string | null): JobEventsState {
  const [state, setState] = useState<JobEventsState>(initial);
  const closed = useRef(false);

  useEffect(() => {
    setState(initial());
    if (!jobId) return;
    closed.current = false;
    const es = new EventSource(`/api/jobs/${jobId}/events`);

    const onEvent = (msg: MessageEvent) => {
      let ev: JobEvent;
      try {
        ev = JSON.parse(msg.data) as JobEvent;
      } catch {
        return;
      }
      setState((s) => {
        if (ev.type === "stage") return { ...s, stages: applyStage(s.stages, ev), connectionLost: false };
        if (ev.type === "warning") return s.warnings.includes(ev.message) ? s : { ...s, warnings: [...s.warnings, ev.message] };
        const failedStage = ev.error?.stage;
        const stages = { ...s.stages };
        if (ev.type === "error" && failedStage && stages[failedStage]) stages[failedStage] = { ...stages[failedStage], state: "error" };
        if (ev.type === "error") {
          for (const k of Object.keys(stages)) if (stages[k].state === "running") stages[k] = { ...stages[k], state: "error" };
        }
        return {
          ...s,
          stages,
          phase: ev.type === "result" ? "done" : ev.type === "error" ? "error" : "cancelled",
          error: ev.error ?? null,
        };
      });
      if (ev.type === "result" || ev.type === "error" || ev.type === "cancelled") {
        closed.current = true;
        es.close();
      }
    };
    for (const name of ["stage", "warning", "result", "error", "cancelled"]) es.addEventListener(name, onEvent);
    es.onerror = () => {
      if (!closed.current) setState((s) => ({ ...s, connectionLost: es.readyState === EventSource.CLOSED }));
    };
    return () => {
      closed.current = true;
      es.close();
    };
  }, [jobId]);

  return state;
}
