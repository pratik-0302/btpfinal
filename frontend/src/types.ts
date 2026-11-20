// Mirrors worker/result.py (AnalysisResult) and api/app.py responses.

export type VerdictLabel = "REAL" | "FAKE" | "UNCERTAIN";
export type StageName = "upload" | "preflight" | "decode" | "faces" | "embeddings" | "scoring";
export type StageState = "pending" | "running" | "done" | "error";

export interface ErrorInfo {
  code: string;
  message: string;
  detail: string;
  stage: string | null;
}

export interface FileInfo {
  name: string;
  size_bytes: number;
  container?: string | null;
  duration_s?: number | null;
  width?: number | null;
  height?: number | null;
  stored_fps?: number | null;
  frames?: number | null;
  video_codec?: string | null;
  audio_codec?: string | null;
  audio_rate?: number | null;
  audio_channels?: number | null;
  rotate_tag?: string | null;
  has_video?: boolean;
  has_audio?: boolean;
  warnings: string[];
}

export interface RunInfo {
  device: string;
  cropper_device: string;
  device_reason: string;
  checkpoint: string;
  torch: string;
  sklearn: string;
  avhubert: Record<string, number | boolean>;
  layers_used: number[];
  bundle: {
    version: number;
    trained: string;
    chosen: Record<string, string>;
    thresholds: Record<string, number>;
    n_your_videos: number;
    ckpt: string;
  };
  max_seconds: number;
  analysed_seconds: number;
  processing_seconds: number;
  stage_seconds: Record<string, number>;
}

export interface AnalysisResult {
  status: "ok";
  file: FileInfo;
  run: RunInfo;
  verdict: {
    label: VerdictLabel;
    headline: string;
    fake_score: number;
    real_score: number;
    decided_by: "sync" | "deepfake" | null;
    reason: string;
    notes: string[];
    hard_flags: string[];
  };
  sync_check: {
    score: number;
    raw_p: number;
    z: number;
    matched_windows: number;
    n_windows: number;
    best_offset_ms: number;
    speech_seconds: number;
  };
  deepfake_check: {
    score: number;
    raw_p: number;
    segment_raw_p: number[];
    segment_spans_s: [number, number][] | null;
  };
  timeline: {
    fps: number;
    agreement: number[];
    speech_mask: boolean[];
    energy_db: number[];
    windows_s: [number, number][];
    window_z: number[];
  };
  quality: {
    face_detected_frac: number | null;
    frames_with_several_faces: number | null;
    mouth_width_px: number | null;
    audio_gap_s: number | null;
    audio_largest_gap_s: number | null;
    audio_gaps: number | null;
    audio_overlap_s: number | null;
    timing: string | null;
    stored_fps: number | null;
    retimed_to_fps: number | null;
    timing_z_all: number[] | null;
  };
  visuals: {
    mouth_crops: { png_base64: string; count: number; size: number; times_s: number[] } | null;
    waveform: { min: number[]; max: number[] } | null;
    audio_features: { bands: number; values: number[][]; min: number; max: number } | null;
    sync_matrix: { offsets_ms: number[]; values: (number | null)[][] | null };
    offset_drift: [number, number][] | null;
  };
  warnings: string[];
  error: null;
}

export interface Job {
  id: string;
  name: string;
  state: "queued" | "running" | "done" | "error" | "cancelled";
  created: number;
  file: FileInfo | Record<string, never>;
  error: ErrorInfo | null;
  result?: AnalysisResult | null;
}

export interface Health {
  worker: "stopped" | "starting" | "ready" | "busy" | "crashed" | "failed";
  loading_message: string;
  meta: Partial<RunInfo> & { load_seconds?: number };
  fatal_error: ErrorInfo | null;
  restarts: number;
  active_job: string | null;
}

export interface Definition {
  label: string;
  meaning: string;
  not: string;
}

export interface ModelInfo {
  meta: Partial<RunInfo>;
  definitions: Record<string, Definition>;
  new_displays: Record<string, string>;
  evaluation: {
    source: string;
    headline_sets: {
      set: string;
      n_real: number;
      n_fake: number;
      auc: number;
      auc_ci: string;
      eer_pct: number;
      forced_choice_accuracy_pct: number;
      balanced_accuracy_pct: number;
      tpr_at_fpr1_pct: number;
    }[];
    verdict_breakdown: {
      set: string;
      n: number;
      truth: VerdictLabel;
      real_pct: number;
      uncertain_pct: number;
      fake_pct: number;
    }[];
    by_generator: { group: string; n_fake: number; auc: number; fake_pct: number }[];
    notes: string[];
  };
  limitations: string[];
  limits: { max_upload_mb: number; max_seconds: number; extensions: string[]; keep_uploads: boolean };
}

// Server-sent events
export interface StageEvent {
  type: "stage";
  stage: Exclude<StageName, "upload">;
  state: "running" | "done";
  done?: number;
  total?: number;
  version?: number;
  versions?: number;
  seconds?: number;
  t: number;
}
export interface WarningEvent {
  type: "warning";
  message: string;
  t: number;
}
export interface EndEvent {
  type: "result" | "error" | "cancelled";
  error?: ErrorInfo | null;
  t: number;
}
export type JobEvent = StageEvent | WarningEvent | EndEvent;

export interface StageView {
  state: StageState;
  startedAt?: number;
  seconds?: number;
  done?: number;
  total?: number;
  version?: number;
  versions?: number;
}
