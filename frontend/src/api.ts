import type { ErrorInfo, Health, Job, ModelInfo } from "./types";

// Required by the server on POST/DELETE (see api/app.py require_client_header).
const CLIENT_HEADER = { "X-LipSync-Client": "1" };

export class ApiError extends Error {
  info: ErrorInfo;
  status: number;
  jobId?: string;
  constructor(status: number, info: ErrorInfo, jobId?: string) {
    super(info.message);
    this.info = info;
    this.status = status;
    this.jobId = jobId;
  }
}

function toApiError(status: number, body: unknown): ApiError {
  const b = body as { error?: ErrorInfo; job_id?: string; detail?: string } | null;
  if (b && b.error) return new ApiError(status, b.error, b.job_id);
  return new ApiError(status, {
    code: status === 404 ? "NOT_FOUND" : "INTERNAL",
    message: (b && typeof b.detail === "string" && b.detail) || `Request failed (HTTP ${status}).`,
    detail: "",
    stage: null,
  });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, {
      code: "SERVER_UNREACHABLE",
      message: "The local server is not responding. Is `python server.py` running?",
      detail: "",
      stage: null,
    });
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) throw toApiError(res.status, body);
  return body as T;
}

export const getHealth = () => request<Health>("/api/health");
export const getModelInfo = () => request<ModelInfo>("/api/model");
export const getJob = (id: string) => request<{ job: Job }>(`/api/jobs/${id}`).then((r) => r.job);
export const deleteJob = (id: string) =>
  request<{ state: string }>(`/api/jobs/${id}`, { method: "DELETE", headers: CLIENT_HEADER });
export const restartWorker = () =>
  request<{ worker: string }>("/api/worker/restart", { method: "POST", headers: CLIENT_HEADER });

/** Upload with real byte progress (XMLHttpRequest exposes upload progress; fetch does not). */
export function uploadVideo(file: File, onProgress: (loaded: number, total: number) => void): Promise<Job> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.setRequestHeader("X-LipSync-Client", "1");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded, e.total);
    };
    xhr.onerror = () =>
      reject(
        new ApiError(0, {
          code: "SERVER_UNREACHABLE",
          message: "The upload failed: the local server is not responding.",
          detail: "",
          stage: "upload",
        }),
      );
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve((body as { job: Job }).job);
      else reject(toApiError(xhr.status, body));
    };
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}
