"""
Lip-sync deepfake detector (v3).

  video file
    -> 25 fps frames (rotation flag applied) + 16 kHz mono audio aligned to frame 0
    -> face landmarks (MTCNN) -> aligned 96x96 grey mouth crops   (AV-HuBERT preprocessing)
    -> AV-HuBERT, video-only pass  -> lip embeddings   v_t
    -> AV-HuBERT, audio-only pass  -> speech embeddings a_t
    -> sync matrix  C[t, d] = cos(v_t, a_{t+d}),  d = -15..+15 frames (+-0.6 s)
    -> two checks, both using only lip-speech agreement (no pixels, no audio content):
       1. SYNC CHECK: in every 2 s window of speech, is the agreement at the best audio offset
          clearly higher than the same clip's agreement far off-sync? Catches swapped, dubbed
          and delayed audio. Camera, microphone and person cancel out (same clip on both sides).
       2. LIP-SYNC DEEPFAKE CHECK: classifier on offset-invariant agreement features
          (v2 models), for generated lips that do follow the audio.
    -> REAL / UNCERTAIN / FAKE with a reason. Thresholds by grid search with limits on false alarms.

v3: the sync check, edge frames ignored, thresholds that must also catch out-of-sync copies of the
user's own clips, audio placed by its timestamps (gaps filled), UNCERTAIN when the evidence is weak.
v4: files whose footage was stored at the wrong frame rate are re-timed (all LipSyncTIMIT real videos:
25 fps footage stored as 30 fps, so their lips drift out of sync), no forced UNCERTAIN for short clips,
one consistent percentage (fake % + real % = 100).
v1/v2 bundles still load with LipSyncDetector.
"""
from __future__ import annotations

import math
import os
import pickle
import warnings
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from avhubert_lite import load_avhubert

FPS, SR, SPF, TAU = 25, 16000, 640, 15
OFFSETS = np.arange(-TAU, TAU + 1)
RC_TOL = 6                    # best offset searched within +-6 frames (+-240 ms)
RC_R = TAU - RC_TOL           # re-centred window: j = -9..+9 around the best offset
FAR = 5                       # |j| >= 5 frames (200 ms) = "off-sync" reference
SEG_LEN, SEG_MIN = 150, 50    # 6 s segments, at least 2 s

_MEAN_FACE = np.array([[70.924, 97.138], [72.625, 114.904], [75.989, 131.074], [79.27, 146.212], [83.613, 163.26], [91.331, 177.445], [100.272, 187.089], [112.124, 196.004], [130.742, 200.53], [149.855, 195.312], [163.102, 186.441], [173.653, 176.573], [182.436, 162.28], [187.167, 145.094], [190.229, 129.724], [193.021, 113.459], [194.439, 95.58], [81.331, 80.795], [87.759, 75.28], [96.227, 73.839], [104.555, 74.74], [112.232, 76.977], [144.496, 76.424], [152.348, 73.833], [161.131, 72.636], [170.587, 73.848], [178.214, 79.438], [128.734, 95.36], [128.489, 106.925], [128.245, 118.273], [128.266, 127.699], [118.76, 135.194], [122.963, 136.146], [128.87, 137.303], [134.943, 135.997], [139.483, 134.878], [92.522, 94.369], [97.585, 90.96], [105.414, 90.913], [112.772, 94.944], [106.104, 97.085], [98.046, 97.363], [145.525, 94.535], [152.59, 90.215], [160.612, 90.199], [166.677, 93.566], [160.56, 96.481], [152.205, 96.473], [107.168, 157.196], [114.476, 152.12], [123.849, 148.519], [128.976, 149.416], [134.144, 148.426], [144.177, 151.793], [152.193, 156.987], [143.86, 164.003], [136.744, 167.943], [129.153, 168.819], [121.795, 168.023], [115.275, 164.152], [109.231, 157.002], [122.503, 154.407], [129.029, 154.121], [135.836, 154.312], [150.758, 156.795], [135.662, 160.63], [128.952, 161.288], [122.488, 160.509]])
# 20words_mean_face.npy (AV-HuBERT / LRS3 preprocessing). Reference points for the 5 MTCNN landmarks:
_REF_STABLE = np.stack([_MEAN_FACE[36:42].mean(0), _MEAN_FACE[42:48].mean(0), _MEAN_FACE[30]])   # eyes, nose tip
STD_SIZE, CROP, ROI = 256, 96, 88
IMG_MEAN, IMG_STD = 0.421, 0.165


class NoFaceError(RuntimeError):
    pass


# ======================================================================= decoding
def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]


def _frame_rotation(frame, stream):
    """Counter-clockwise display rotation in degrees (phones store portrait video sideways + a flag)."""
    r = getattr(frame, "rotation", None)
    if r is None:
        try:
            r = float(stream.metadata.get("rotate", 0)) * -1.0     # old 'rotate' tag is clockwise
        except Exception:
            r = 0
    return int(round(float(r or 0) / 90.0)) % 4


STANDARD_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0)


def _mismuxed_rate(c):
    """Frame rate the footage really has, if the file was stored at a wrong one.

    Signature (found in all 606 LipSyncTIMIT real videos): 25 fps footage written as 30 fps, so the video
    stream lasts only 0.83x as long as the audio and the lips drift ahead of the speech by 0.2 s per second.
    Rule: video and audio lengths differ by more than 10%, frames / audio length is within 4% of a standard
    frame rate (short clips: audio padding moves it by 1-3%), and the stored rate is more than 8% away from it.
    Normal recordings never match all three (their video and audio have about the same length)."""
    try:
        vs = c.streams.video[0]
        a = c.streams.audio[0] if c.streams.audio else None
        if a is None or not vs.frames or not vs.duration or not a.duration:
            return None
        n = int(vs.frames)
        v_dur = float(vs.duration * vs.time_base)
        a_dur = float(a.duration * a.time_base)
        if a_dur < 1.0 or v_dur <= 0:
            return None
        eff, stored = n / a_dur, n / v_dur
        best = min(STANDARD_FPS, key=lambda r: abs(r - eff))
        if abs(v_dur / a_dur - 1) > 0.10 and abs(best - eff) / best < 0.04 and abs(stored - best) / best > 0.08:
            return float(best)
    except Exception:
        return None
    return None


def _decode(path, max_seconds, max_side):
    import av
    import cv2
    frames, times = [], []
    with av.open(str(path)) as c:
        if not c.streams.video:
            raise ValueError("the file has no video stream")
        vs = c.streams.video[0]
        rate = float(vs.average_rate or 25)
        cand = _mismuxed_rate(c)                             # footage may really be at this rate
        for f in c.decode(video=0):
            t = float(f.time) if f.time is not None else len(times) / rate
            if max_seconds is not None and times and t - times[0] > max_seconds + 0.2:
                break
            img = f.to_ndarray(format="rgb24")
            k = _frame_rotation(f, vs)
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))   # same result as ffmpeg's autorotate
            h, w = img.shape[:2]
            if max(h, w) > max_side:
                s = max_side / max(h, w)
                img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            frames.append(img); times.append(t)
    if not frames:
        raise ValueError("no video frames could be decoded")
    audio, a_start, gaps = _decode_audio(path, None if max_seconds is None else max_seconds + 1.0)
    gaps = {**gaps, "retime_candidate_fps": cand or 0.0, "stored_fps": rate, "retimed_to_fps": 0.0}
    return frames, np.asarray(times), audio, a_start, gaps


def _decode_audio(path, max_seconds=None, tol_s=0.002):
    """-> int16 mono 16 kHz audio, start time (s), gap info.

    Every decoded audio frame is placed at its own timestamp, like a video player does: a gap in
    the audio track becomes silence instead of pulling all later audio earlier (some phone
    recordings and edited/forwarded clips have such gaps, which would otherwise look out of sync)."""
    import av
    from math import gcd
    from scipy import signal
    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise ValueError("the file has no audio track: lip sync cannot be checked")
        st = c.streams.audio[0]
        sr0 = int(st.rate or st.codec_context.sample_rate or SR)
        res = av.AudioResampler(format="flt", layout="mono", rate=sr0)    # same rate: format/layout only
        pieces, a_start, pos = [], None, 0
        gap_total, gap_max, n_gaps, overlap = 0, 0, 0, 0
        tol = int(tol_s * sr0)
        for f in c.decode(audio=0):
            y = [g.to_ndarray().reshape(-1) for g in _as_list(res.resample(f))]
            y = np.concatenate(y).astype(np.float32) if y else np.zeros(0, np.float32)
            if f.time is not None:
                t = float(f.time)
                if a_start is None:
                    a_start = t
                p = int(round((t - a_start) * sr0))
                if p - pos > tol:                                   # gap: fill with silence
                    pieces.append(np.zeros(p - pos, np.float32))
                    gap_total += p - pos; gap_max = max(gap_max, p - pos); n_gaps += (p - pos) > 0.02 * sr0
                    pos = p
                elif pos - p > tol:                                 # overlap: drop the repeated part
                    cut = min(pos - p, len(y)); y = y[cut:]; overlap += cut
            pieces.append(y); pos += len(y)
            if max_seconds is not None and pos > max_seconds * sr0:
                break
        rest = [g.to_ndarray().reshape(-1) for g in _as_list(res.resample(None))]
        if rest:
            pieces.append(np.concatenate(rest).astype(np.float32))
    x = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    if sr0 != SR and len(x):
        g = gcd(SR, sr0)
        x = signal.resample_poly(x, SR // g, sr0 // g)
    audio = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
    gaps = {"audio_gap_s": gap_total / sr0, "audio_largest_gap_s": gap_max / sr0, "audio_gaps": int(n_gaps),
            "audio_overlap_s": overlap / sr0}
    return audio, a_start, gaps


def _ffmpeg_normalise(path):
    """Fallback for files PyAV cannot read: let ffmpeg re-encode (it also applies rotation)."""
    out = Path(tempfile.mkdtemp()) / "normalised.mkv"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0?",
           "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "pcm_s16le", "-ar", str(SR), "-ac", "1", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _assemble(frames, times, audio, a_start, max_seconds):
    """Frames resampled to 25 fps by timestamp; audio aligned to the first frame."""
    v0 = times[0]
    if a_start is not None:                       # align audio to the first video frame
        d = int(round((a_start - v0) * SR))
        audio = np.concatenate([np.zeros(d, np.int16), audio]) if d > 0 else audio[-d:]
    rel = times - v0                               # resample to 25 fps by timestamp
    step = np.median(np.diff(rel)) if len(rel) > 1 else 1 / FPS
    n_out = max(1, int(math.floor((rel[-1] + step) * FPS + 1e-6)))
    idx = np.clip(np.searchsorted(rel, np.arange(n_out) / FPS + 1e-4, side="right") - 1, 0, len(frames) - 1)
    out = np.stack([frames[i] for i in idx])
    if max_seconds is not None:
        out = out[: int(max_seconds * FPS)]
    return out, audio[: len(out) * SPF]


def read_av(path, max_seconds=None, max_side=640, info=None, timing="stored"):
    """-> frames (N, H, W, 3) uint8 RGB at 25 fps (upright), audio int16 mono 16 kHz aligned to frame 0.

    timing: 'stored'  = as the file says;
            'both'    = list of (frames, audio, label): the stored timing, plus a re-timed version when the file
                        has the wrong-frame-rate signature (see _mismuxed_rate). process_video uses the re-timed
                        version unless the stored timing makes lips and speech line up clearly better.
    If a dict is passed as `info`, it receives the audio-gap and frame-rate figures."""
    try:
        frames, times, audio, a_start, dinfo = _decode(path, max_seconds, max_side)
    except ValueError as e:
        if "audio" in str(e):
            raise
        frames, times, audio, a_start, dinfo = _decode(_ffmpeg_normalise(path), max_seconds, max_side)
    except Exception:
        frames, times, audio, a_start, dinfo = _decode(_ffmpeg_normalise(path), max_seconds, max_side)
    if info is not None:
        info.update(dinfo)
    stored = _assemble(frames, times, audio, a_start, max_seconds)
    if timing == "stored":
        return stored
    out = [(*stored, "stored")]
    cand = dinfo.get("retime_candidate_fps") or 0.0
    if cand:
        rt = times[0] + np.arange(len(times)) / cand                 # frame k shown at k / true rate
        out.append((*_assemble(frames, rt, audio, a_start, max_seconds), f"retimed to {cand:g} fps"))
    return out


RETIME_MARGIN = 0.5      # the stored timing is kept only if it beats the re-timed one by this much sync strength (z)


# ======================================================================= mouth crops
def _similarity(src, dst):
    """Umeyama similarity transform src -> dst, as a 2x3 matrix."""
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / ((s0 ** 2).sum() / len(src))
    t = md - scale * R @ ms
    return np.hstack([scale * R, t[:, None]])


def _smooth(x, win):
    """Centred moving average along axis 0 (edge-padded)."""
    if win <= 1 or len(x) < 2:
        return x
    pad = win // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], win - 1 - pad, 0)])
    c = np.cumsum(np.concatenate([np.zeros_like(x[:1]), xp]), 0)
    return (c[win:] - c[:-win]) / win


class MouthCropper:
    """MTCNN 5-point landmarks -> similarity alignment to the AV-HuBERT mean face ->
    96x96 grey crop centred on the mouth (same geometry as the LRS3 preprocessing)."""

    def __init__(self, device="cuda", det_stride=2, batch=32, min_prob=0.90):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
        self.det_stride, self.batch, self.min_prob = det_stride, batch, min_prob

    def landmarks(self, frames, det_stride=None):
        """-> (N, 5, 2) landmarks (interpolated where missing) and an info dict."""
        stride = det_stride or self.det_stride
        n = len(frames)
        det_idx = list(range(0, n, stride))
        if det_idx[-1] != n - 1:
            det_idx.append(n - 1)
        pts = np.full((n, 5, 2), np.nan)
        prev_c, multi = None, 0
        for b in range(0, len(det_idx), self.batch):
            ids = det_idx[b:b + self.batch]
            boxes, probs, points = self.mtcnn.detect(np.stack([frames[i] for i in ids]), landmarks=True)
            for i, bx, pr, pt in zip(ids, boxes, probs, points):
                if bx is None:
                    continue
                ok = [k for k in range(len(bx)) if pr[k] is not None and pr[k] >= self.min_prob]
                if not ok:
                    continue
                multi += len(ok) > 1
                if prev_c is None:      # first face: the largest
                    k = max(ok, key=lambda k: (bx[k][2] - bx[k][0]) * (bx[k][3] - bx[k][1]))
                else:                   # then: follow the same face
                    k = min(ok, key=lambda k: np.hypot(*(((bx[k][:2] + bx[k][2:]) / 2) - prev_c)))
                prev_c = (bx[k][:2] + bx[k][2:]) / 2
                pts[i] = pt[k]
        found = ~np.isnan(pts[:, 0, 0])
        if not found.any():
            raise NoFaceError("no face was found in the video")
        t = np.arange(n)
        for j in range(5):
            for d in range(2):
                pts[:, j, d] = np.interp(t, t[found], pts[found, j, d])
        info = {"face_detected_frac": float(found[det_idx].mean()), "frames_with_several_faces": int(multi),
                "mouth_width_px": float(np.median(np.linalg.norm(pts[:, 3] - pts[:, 4], axis=1)))}
        return pts, info

    def crops(self, frames, pts=None, info=None):
        import cv2
        if pts is None:
            pts, info = self.landmarks(frames)
        stable = _smooth(pts[:, :3], 12)            # eyes + nose: smoothed over ~0.5 s (as the reference)
        mouth = _smooth(pts[:, 3:], 3).mean(1)      # mouth centre from the two corners
        out = np.empty((len(frames), CROP, CROP), np.uint8)
        for i, fr in enumerate(frames):
            M = _similarity(stable[i], _REF_STABLE)
            grey = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
            w = cv2.warpAffine(grey, M, (STD_SIZE, STD_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cx, cy = M @ np.array([mouth[i, 0], mouth[i, 1], 1.0])
            cx = int(round(min(max(cx, CROP / 2), STD_SIZE - CROP / 2)))
            cy = int(round(min(max(cy, CROP / 2), STD_SIZE - CROP / 2)))
            out[i] = w[cy - CROP // 2: cy + CROP // 2, cx - CROP // 2: cx + CROP // 2]
        return out, (info or {"face_detected_frac": 1.0, "frames_with_several_faces": 0,
                              "mouth_width_px": float(np.median(np.linalg.norm(pts[:, 3] - pts[:, 4], axis=1)))})


# ======================================================================= training augmentations
def augment_crops(crops, rng, strength=1.0):
    """Label-preserving camera changes on the 96x96 grey mouth crops: crop/landmark error,
    exposure, contrast, resolution, blur, sensor noise, JPEG compression, low frame rate."""
    import cv2
    x = crops.astype(np.float32)
    T = len(x)
    if rng.random() < 0.8:                                  # alignment error: fixed offset + small per-frame jitter
        ang = rng.uniform(-7, 7) * strength
        sc = 1 + rng.uniform(-0.10, 0.10) * strength
        tx, ty = rng.uniform(-5, 5, 2) * strength
        jit = rng.normal(0, 0.8 * strength, (T, 2))
        out = np.empty_like(x)
        for i in range(T):
            M = cv2.getRotationMatrix2D((CROP / 2, CROP / 2), ang, sc)
            M[:, 2] += (tx + jit[i, 0], ty + jit[i, 1])
            out[i] = cv2.warpAffine(x[i], M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        x = out
    g = float(np.exp(rng.uniform(-0.45, 0.45) * strength))   # exposure / gamma
    x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** g
    m = x.mean()
    x = (x - m) * (1 + rng.uniform(-0.35, 0.25) * strength) + m + rng.uniform(-25, 25) * strength
    if rng.random() < 0.6:                                  # lower resolution
        s = rng.uniform(0.45, 0.95)
        sz = max(16, int(CROP * s))
        x = np.stack([cv2.resize(cv2.resize(f, (sz, sz), interpolation=cv2.INTER_AREA), (CROP, CROP),
                                 interpolation=cv2.INTER_LINEAR) for f in x])
    if rng.random() < 0.4:                                  # defocus / motion blur
        sig = rng.uniform(0.3, 1.4)
        x = np.stack([cv2.GaussianBlur(f, (0, 0), sig) for f in x])
    if rng.random() < 0.6:                                  # sensor noise
        x = x + rng.normal(0, rng.uniform(1, 6), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:                                  # compression
        q = int(rng.uniform(20, 85))
        x = np.stack([cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
                      for f in x])
    if rng.random() < 0.2:                                  # 12.5-15 fps webcam, shown at 25 fps
        rep = 2 if rng.random() < 0.6 else 3
        idx = np.minimum((np.arange(T) // rep) * rep, T - 1)
        x = x[idx]
    return x


def augment_audio(audio_int16, rng, strength=1.0, noise_pool=None):
    """Label-preserving microphone/room changes: reverb, band limits, background noise or
    babble, level changes and clipping, occasional 8 kHz telephone band."""
    from scipy import signal
    a = audio_int16.astype(np.float32) / 32768.0
    n = len(a)
    if n < SR // 4:
        return audio_int16
    rms0 = float(np.sqrt(np.mean(a ** 2)) + 1e-8)
    if rng.random() < 0.5:                                  # room reverb
        rt60 = rng.uniform(0.15, 0.7)
        L = int(rt60 * SR)
        t = np.arange(L) / SR
        rir = rng.normal(0, 1, L) * np.exp(-6.9 * t / rt60)
        rir[0] = 1.0 / max(rng.uniform(0.3, 1.0), 1e-3)
        a = signal.fftconvolve(a, rir / np.sqrt(np.sum(rir ** 2)))[:n]
    if rng.random() < 0.45:                                 # small/cheap microphone band
        lo = rng.uniform(80, 350); hi = rng.uniform(3400, 7500)
        sos = signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
        a = signal.sosfilt(sos, a)
    if rng.random() < 0.7:                                  # background noise at SNR 5-30 dB
        snr = rng.uniform(5, 30) * (1 if strength >= 1 else 1 / max(strength, 1e-3))
        if noise_pool and rng.random() < 0.5:
            src = noise_pool[int(rng.integers(len(noise_pool)))].astype(np.float32) / 32768.0
            if len(src) < n:
                src = np.tile(src, int(np.ceil(n / max(len(src), 1))) + 1)
            o = int(rng.integers(0, len(src) - n + 1))
            noise = src[o:o + n]
        else:
            noise = rng.normal(0, 1, n)
            if rng.random() < 0.5:                          # pink-ish
                noise = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                                       [1, -2.494956002, 2.017265875, -0.522189400], noise)
        p_sig = np.mean(a ** 2) + 1e-10; p_n = np.mean(noise ** 2) + 1e-10
        a = a + noise * np.sqrt(p_sig / (p_n * 10 ** (snr / 10)))
    if rng.random() < 0.15:                                 # telephone band
        a = signal.resample_poly(signal.resample_poly(a, 1, 2), 2, 1)[:n]
    rms = float(np.sqrt(np.mean(a ** 2)) + 1e-8)
    a = a * (rms0 / rms) * 10 ** (rng.uniform(-12, 6) / 20)
    if rng.random() < 0.15:                                 # clipping / automatic gain
        a = np.tanh(a * rng.uniform(1.5, 4.0)) / np.tanh(rng.uniform(1.5, 4.0))
    return np.clip(a * 32768.0, -32768, 32767).astype(np.int16)


# ======================================================================= AV-HuBERT inputs
def video_input(crops):
    """(T, 96, 96) uint8 -> (1, 1, T, 88, 88) normalised, centre-cropped."""
    o = (CROP - ROI) // 2
    x = crops[:, o:o + ROI, o:o + ROI].astype(np.float32) / 255.0
    x = (x - IMG_MEAN) / IMG_STD
    return torch.from_numpy(x)[None, None]


def audio_input(audio_int16):
    """int16 16 kHz -> (1, 104, T) : 26 log-filterbanks at 100 Hz, 4 stacked (25 Hz), per-frame layer norm."""
    from python_speech_features import logfbank
    f = logfbank(audio_int16.astype(np.float64), samplerate=SR).astype(np.float32)   # (Ta, 26)
    r = (-len(f)) % 4
    if r:
        f = np.concatenate([f, np.zeros((r, f.shape[1]), np.float32)])
    f = f.reshape(-1, 4 * f.shape[1])
    x = torch.from_numpy(f)
    x = F.layer_norm(x, x.shape[1:])
    return x.T[None]


def speech_energy(audio_int16, n):
    a = audio_int16[: n * SPF].astype(np.float32) / 32768.0
    if len(a) < n * SPF:
        a = np.pad(a, (0, n * SPF - len(a)))
    e = np.sqrt((a.reshape(n, SPF) ** 2).mean(1) + 1e-10)
    return 20 * np.log10(e + 1e-10)        # dBFS per video frame


class SyncEncoder:
    """Frozen AV-HuBERT; returns lip and speech embeddings for chosen layers."""

    def __init__(self, ckpt_path, device="cuda", layers=None, chunk=500):
        self.model, self.info = load_avhubert(ckpt_path, device)
        L = self.info["layers"]
        self.layers = layers or [L, (3 * L) // 4, L // 2]       # final output first
        self.device, self.chunk = device, chunk

    @torch.no_grad()
    def _run(self, fn, x, T):
        outs = {l: [] for l in self.layers}
        for s in range(0, T, self.chunk):
            xs = x[..., s:s + self.chunk, :, :] if x.dim() == 5 else x[..., s:s + self.chunk]
            hs = fn(xs.to(self.device))
            for l in self.layers:
                outs[l].append(hs[l - 1][0].float())
        return {l: torch.cat(v) for l, v in outs.items()}

    def embed(self, crops, audio_int16):
        v_in, a_in = video_input(crops), audio_input(audio_int16)
        T = min(v_in.shape[2], a_in.shape[-1])
        V = self._run(self.model.forward_video, v_in[:, :, :T], T)
        A = self._run(self.model.forward_audio, a_in[..., :T], T)
        return V, A, T

    def embed_audio(self, audio_int16, T=None):
        """Speech embeddings only (used for partly replaced audio)."""
        a_in = audio_input(audio_int16)
        T = a_in.shape[-1] if T is None else min(int(T), a_in.shape[-1])
        return self._run(self.model.forward_audio, a_in[..., :T], T)


# ======================================================================= sync matrix
def sync_matrix(v, a, tau=TAU):
    """v, a: (T, C) tensors -> (T, 2tau+1) cosine similarity cos(v_t, a_{t+d}); NaN outside the clip."""
    v = F.normalize(v.float(), dim=1)
    a = F.normalize(a.float(), dim=1)
    S = (v @ a.T).cpu().numpy()                     # S[t, s] = cos(v_t, a_s)
    T = len(S)
    C = np.full((T, 2 * tau + 1), np.nan, np.float32)
    for j, d in enumerate(range(-tau, tau + 1)):
        t0, t1 = max(0, -d), min(T, T - d)
        if t1 > t0:
            C[t0:t1, j] = S[np.arange(t0, t1), np.arange(t0, t1) + d]
    return C


def zero_shot_score(C):
    """SpeechForensics score: mean over frames (zero outside the clip), max over offsets. High = real."""
    return float(np.nanmax(np.nan_to_num(C, nan=0.0).mean(0)))


def process_video(path, cropper, encoder, max_seconds=None, prefetched=None, read_info=None):
    """Video file -> sync matrices of every chosen layer + per-frame speech energy.
    prefetched: (frames, audio) or the list returned by read_av(..., timing='both'); read_info: its info dict."""
    ainfo = dict(read_info or {})
    if prefetched is None:
        versions = read_av(path, max_seconds=max_seconds, info=ainfo, timing="both")
    elif isinstance(prefetched, list):
        versions = prefetched
    else:
        versions = [(*prefetched, "stored")]
    res = []
    for frames, audio, label in versions:
        crops, cinfo = cropper.crops(frames)
        V, A, T = encoder.embed(crops, audio)
        Cs = [sync_matrix(V[l], A[l]) for l in encoder.layers]
        E = speech_energy(audio, T)
        z = sync_evidence(Cs[0], E)["z"] if len(versions) > 1 else 0.0
        res.append({"Cs": Cs, "E": E, "T": T, "info": cinfo, "V": V, "A": A, "crops": crops, "audio": audio,
                    "timing": label, "timing_z": z})
    # a file with the wrong-frame-rate signature is played at its true rate, unless the stored timing makes the
    # lips and speech line up clearly better (a wrongly kept drift would look like a fake; a wrongly re-timed fake
    # still looks fake)
    best = res[0] if len(res) == 1 or res[0]["timing_z"] > res[1]["timing_z"] + RETIME_MARGIN else res[1]
    best["timing_z_all"] = [r_["timing_z"] for r_ in res]
    best["info"] = {**best["info"], **ainfo, "timing": best["timing"],
                    "retimed_to_fps": ainfo.get("retime_candidate_fps", 0.0) if best["timing"] != "stored" else 0.0}
    return best


def shifted_matrices(V, A, E, layers, k):
    """Out-of-sync copy of a real clip: video frame t paired with audio frame t+k."""
    T = len(V[layers[0]])
    if T - abs(k) < 25:
        return None
    if k > 0:
        return [sync_matrix(V[l][:T - k], A[l][k:]) for l in layers], E[k:]
    return [sync_matrix(V[l][-k:], A[l][:T + k]) for l in layers], E[:T + k]


def swapped_matrices(V, A2, E2, layers):
    """Out-of-sync copy: the lips of one real clip with the speech of another speaker."""
    n = min(len(V[layers[0]]), len(A2[layers[0]]))
    if n < 25:
        return None
    return [sync_matrix(V[l][:n], A2[l][:n]) for l in layers], E2[:n]


# ======================================================================= v2 features (offset-invariant)
def best_offset(C0, tol=RC_TOL):
    """Global audio offset (frames) with the highest mean agreement, searched within +-tol."""
    prof = np.nanmean(C0, 0)
    sub = prof[TAU - tol: TAU + tol + 1]
    if not np.isfinite(sub).any():
        return 0
    return int(np.nanargmax(sub)) - tol


def recenter(Cs, d):
    """Columns j = -RC_R..+RC_R around offset d (the best offset becomes column RC_R)."""
    lo, hi = TAU + d - RC_R, TAU + d + RC_R + 1
    return [C[:, lo:hi] for C in Cs]


def segment_bounds(T, seg=SEG_LEN, min_len=SEG_MIN):
    """6 s pieces; a short tail (< 2 s) joins the previous piece; clips up to 9 s stay whole."""
    if T <= seg + seg // 2:
        return [(0, T)]
    cuts = list(range(0, T, seg)) + [T]
    b = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    if len(b) > 1 and b[-1][1] - b[-1][0] < min_len:
        b[-2] = (b[-2][0], b[-1][1]); b.pop()
    return b


_FAR_IDX = np.r_[0:RC_R - FAR + 1, RC_R + FAR:2 * RC_R + 1]
_NEAR_IDX = np.r_[RC_R - 2:RC_R, RC_R + 1:RC_R + 3]
_ABS_NAMES = ["peak", "baseline", "peak_minus_base", "sharpness", "c_mean", "c_median", "c_p10", "c_p90",
              "c_std", "local_mean", "c_speech_weighted"]
_REL_NAMES = (["contrast"] + [f"shape{j:+d}" for j in range(-RC_R, RC_R + 1)] +
              ["z_mean", "z_median", "z_p10", "z_p25", "z_std", "z_gt1_speech", "argmax_stability",
               "c_autocorr", "z_speech_weighted", "peak_over_base_ratio"])


def _seg_features(R, e, e_max):
    """R: (T, 2*RC_R+1) re-centred agreement rows of one layer; e: energy dB per frame."""
    j0 = RC_R
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmean(R, 0)
        far = prof[_FAR_IDX]
        peak, base = prof[j0], np.nanmean(far)
        spread = np.nanstd(far) + 1e-4
        c = R[:, j0]
        fr = R[:, _FAR_IDX]
        mu_f, sd_f = np.nanmean(fr, 1), np.nanstd(fr, 1) + 1e-3
        z = (c - mu_f) / sd_f
        loc = np.nanmax(R[:, j0 - 2:j0 + 3], 1)
        am = np.nanargmax(np.nan_to_num(R, nan=-9.0), 1)
        w = np.clip((e - (e_max - 40.0)) / 40.0, 0, 1)
        speech = e > e_max - 30.0
        ok = np.isfinite(c)
        cw = float(np.nansum(np.where(ok, c, 0) * w) / (w[ok].sum() + 1e-6))
        zw = float(np.nansum(np.where(ok, z, 0) * w) / (w[ok].sum() + 1e-6))
        ac = float(np.corrcoef(c[ok][:-1], c[ok][1:])[0, 1]) if ok.sum() > 4 and np.nanstd(c) > 0 else 0.0
        shape = (prof - np.nanmean(prof)) / (np.nanstd(prof) + 1e-4)
        zs = z[speech & ok] if (speech & ok).any() else z[ok]
        absolute = [peak, base, peak - base, peak - np.nanmean(prof[_NEAR_IDX]), np.nanmean(c), np.nanmedian(c),
                    np.nanpercentile(c, 10), np.nanpercentile(c, 90), np.nanstd(c), np.nanmean(loc), cw]
        relative = ([(peak - base) / spread] + list(shape) +
                    [np.nanmean(z), np.nanmedian(z), np.nanpercentile(z, 10), np.nanpercentile(z, 25), np.nanstd(z),
                     float(np.mean(zs > 1.0)) if len(zs) else 0.0, float(np.mean(np.abs(am - j0) <= 1)), ac, zw,
                     (peak - base) / (abs(base) + 0.05)])
    return absolute, relative


def robust_features(Cs, energy, names=False, d=None, bounds=None):
    """Offset-invariant features per segment -> (n_segments, n_features) array.
    Returns (X, rel_mask[, names]) where rel_mask marks the self-normalised features."""
    d = best_offset(Cs[0]) if d is None else d
    Rs = recenter(Cs, d)
    e = np.asarray(energy, np.float32)[: len(Rs[0])]
    e_max = float(np.max(e)) if len(e) else 0.0
    bounds = bounds or segment_bounds(len(Rs[0]))
    rows = []
    for s, t in bounds:
        f_abs, f_rel = [], []
        for R in Rs:
            a_, r_ = _seg_features(R[s:t], e[s:t], e_max)
            f_abs += a_; f_rel += r_
        rows.append(f_abs + f_rel)
    X = np.nan_to_num(np.asarray(rows, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nl = len(Cs)
    rel_mask = np.array([False] * (len(_ABS_NAMES) * nl) + [True] * (len(_REL_NAMES) * nl))
    if names:
        nm = [f"L{i}_{n}" for i in range(nl) for n in _ABS_NAMES] + [f"L{i}_{n}" for i in range(nl) for n in _REL_NAMES]
        return X, rel_mask, nm
    return X, rel_mask


def sequence_input_v2(Cs, d=None, bounds=None):
    """Per-segment (T, 2*(2*RC_R+1)*L) inputs for the temporal CNN: re-centred agreement rows
    and the same rows normalised by each frame's off-sync statistics."""
    d = best_offset(Cs[0]) if d is None else d
    Rs = recenter(Cs, d)
    parts = []
    for R in Rs:
        fr = R[:, _FAR_IDX]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu, sd = np.nanmean(fr, 1, keepdims=True), np.nanstd(fr, 1, keepdims=True) + 1e-3
            Zr = (R - mu) / sd
        parts += [np.nan_to_num(R, nan=0.0), np.clip(np.nan_to_num(Zr, nan=0.0), -10, 10) / 4.0]
    X = np.concatenate(parts, 1).astype(np.float32)
    bounds = bounds or segment_bounds(len(X))
    return [X[s:t] for s, t in bounds]


# ======================================================================= v1 features (kept for old bundles)
def _pct(x, q):
    return np.percentile(x, q) if len(x) else np.nan


def sync_features(Cs, energy, names=False):
    """v1 features (absolute offsets included). Kept so models saved by the first notebook still load."""
    feats, keys = [], []
    e = np.asarray(energy, np.float32)
    speech = e > (np.max(e) - 30.0)
    w = np.clip((e - (np.max(e) - 40.0)) / 40.0, 0, 1)
    for li, C in enumerate(Cs):
        T = len(C)
        prof = np.nanmean(C, 0)
        j0 = TAU
        jbest = int(np.nanargmax(prof))
        c0 = C[:, j0]
        best = np.nanmax(C, 1)
        local = np.nanmax(C[:, j0 - 2:j0 + 3], 1)
        far = np.nanmean(C[:, np.r_[0:j0 - 4, j0 + 5:2 * TAU + 1]], 1)
        am = np.nanargmax(np.nan_to_num(C, nan=-9), 1)
        ws = w[:T]
        f = list(prof) + [
            prof[j0], prof.max(), float(OFFSETS[jbest]), abs(float(OFFSETS[jbest])),
            prof.max() - np.median(prof), prof.max() - prof.mean(), prof.max() - prof[j0],
            prof[j0] - np.nanmean(prof[np.r_[0:j0 - 4, j0 + 5:2 * TAU + 1]]),
            np.nanmean(c0), np.nanstd(c0), _pct(c0, 10), _pct(c0, 50), _pct(c0, 90),
            np.nanmean(best), np.nanstd(best), _pct(best, 10), _pct(best, 50),
            np.nanmean(local), np.nanmean(c0 - far),
            float(np.mean(np.abs(OFFSETS[am] - OFFSETS[jbest]) <= 1)),
            float(np.nansum(c0 * ws) / (ws.sum() + 1e-6)),
            float(np.nanmean(c0[speech[:T]])) if speech[:T].any() else np.nanmean(c0),
            float(np.nanmean(c0[~speech[:T]])) if (~speech[:T]).any() else np.nanmean(c0),
            float(np.corrcoef(c0[:-1], c0[1:])[0, 1]) if T > 3 and np.nanstd(c0) > 0 else 0.0,
        ]
        feats += f
        if names:
            keys += [f"L{li}_prof{d:+d}" for d in OFFSETS] + [f"L{li}_{k}" for k in (
                "p0", "pmax", "argmax", "abs_argmax", "peak_minus_median", "peak_minus_mean", "peak_minus_p0",
                "p0_minus_far", "c0_mean", "c0_std", "c0_p10", "c0_p50", "c0_p90", "best_mean", "best_std",
                "best_p10", "best_p50", "local_mean", "c0_minus_far", "argmax_stability", "c0_energy_weighted",
                "c0_speech", "c0_nonspeech", "c0_autocorr")]
    out = np.nan_to_num(np.asarray(feats, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return (out, keys) if names else out


def sequence_input(Cs, energy=None):
    """v1 per-frame input (raw rows; includes the absolute offset)."""
    return np.concatenate([np.nan_to_num(C, nan=0.0) for C in Cs], 1).astype(np.float32)


# ======================================================================= sequence model
class SyncSeqNet(nn.Module):
    """Small temporal CNN over per-frame sync rows. Input never contains pixels or audio."""

    def __init__(self, d_in, width=64, act="relu", dropout=0.2):
        super().__init__()
        A = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[act]
        self.net = nn.Sequential(
            nn.Conv1d(d_in, width, 5, padding=2), A(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 5, padding=4, dilation=2), A(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 5, padding=8, dilation=4), A())
        self.head = nn.Linear(2 * width, 1)

    def forward(self, x, mask):           # x (B, T, D), mask (B, T) True = valid
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        m = mask[..., None].float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = h.masked_fill(~mask[..., None], -1e4).max(1).values
        return self.head(torch.cat([mean, mx], 1)).squeeze(1)    # logit of P(fake)


def _pad_batch(seqs, max_len=None):
    L = max(len(s) for s in seqs) if max_len is None else max_len
    D = seqs[0].shape[1]
    X = np.zeros((len(seqs), L, D), np.float32)
    M = np.zeros((len(seqs), L), bool)
    for i, s in enumerate(seqs):
        s = s[:L]
        X[i, :len(s)] = s; M[i, :len(s)] = True
    return torch.from_numpy(X), torch.from_numpy(M)


def train_seqnet(seqs, y, val_seqs, val_y, act="relu", epochs=60, lr=1e-3, wd=1e-3, batch=32,
                 seed=0, device="cpu", min_crop=50, patience=12, log=None, w=None, val_groups=None):
    """y: 1 = fake. Class-balanced sampling (within a class, proportional to w), random temporal
    crops; early stop on validation AUC (mean over val_groups if given)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = SyncSeqNet(seqs[0].shape[1], act=act).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    y = np.asarray(y); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    pp, pn = w[pos] / w[pos].sum(), w[neg] / w[neg].sum()
    n_per = max(len(pos), len(neg))
    val_y = np.asarray(val_y)

    def val_auc(pv):
        if val_groups is None:
            return roc_auc_score(val_y, pv) if len(set(val_y)) > 1 else 0.5
        aucs = [roc_auc_score(val_y[g], pv[g]) for g in val_groups if len(set(val_y[g])) > 1]
        return float(np.mean(aucs)) if aucs else 0.5

    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train()
        idx = np.concatenate([rng.choice(pos, n_per, p=pp), rng.choice(neg, n_per, p=pn)])
        rng.shuffle(idx)
        for b in range(0, len(idx), batch):
            bi = idx[b:b + batch]
            crops = []
            for i in bi:
                s = seqs[i]
                if len(s) > min_crop and rng.random() < 0.7:
                    L = int(rng.integers(min_crop, len(s) + 1)); o = int(rng.integers(0, len(s) - L + 1))
                    s = s[o:o + L]
                crops.append(s)
            X, M = _pad_batch(crops)
            loss = F.binary_cross_entropy_with_logits(net(X.to(device), M.to(device)),
                                                      torch.tensor(y[bi], dtype=torch.float32, device=device))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        auc = val_auc(predict_seqnet(net, val_seqs, device))
        if auc > best + 1e-4:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if log and (ep % 10 == 0 or bad >= patience):
            log(f"    seq[{act}] epoch {ep:2d}  val AUC {auc:.4f}  best {best:.4f}")
        if bad >= patience:
            break
    net.load_state_dict(best_state)
    return net.eval(), best


@torch.no_grad()
def predict_seqnet(net, seqs, device="cpu", batch=64):
    net.eval(); out = []
    for b in range(0, len(seqs), batch):
        X, M = _pad_batch(seqs[b:b + batch])
        out.append(torch.sigmoid(net(X.to(device), M.to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ======================================================================= decision helpers
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def mean_prob(ps):
    """Average of probabilities in logit space (used to combine segments)."""
    return float(1 / (1 + np.exp(-np.mean(_logit(ps)))))


def grid_threshold(y_fake, p_fake, metric="balanced_accuracy", groups=None):
    """Grid search over thresholds: 0.01..0.99 plus the 1st..99th percentiles of the scores.
    groups: optional list of index arrays; balanced accuracy then averages the true-negative rate
    over the real part of every group and the true-positive rate over the fake part of every group."""
    p_fake = np.asarray(p_fake, float); y_fake = np.asarray(y_fake)
    ths = np.unique(np.concatenate([np.round(np.arange(0.01, 1.0, 0.01), 2),
                                    np.quantile(p_fake, np.linspace(0.01, 0.99, 99))]))
    groups = groups or [np.arange(len(y_fake))]
    rows = []
    for t in ths:
        pr = p_fake >= t
        tnrs = [np.mean(~pr[g][y_fake[g] == 0]) for g in groups if (y_fake[g] == 0).any()]
        tprs = [np.mean(pr[g][y_fake[g] == 1]) for g in groups if (y_fake[g] == 1).any()]
        tnr = float(np.mean(tnrs)) if tnrs else float("nan")
        tpr = float(np.mean(tprs)) if tprs else float("nan")
        tp = int((pr & (y_fake == 1)).sum()); fp = int((pr & (y_fake == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rows.append({"threshold": float(t), "accuracy": float(np.mean(pr == (y_fake == 1))),
                     "balanced_accuracy": (tnr + tpr) / 2, "tnr": tnr, "tpr": tpr,
                     "f1": 2 * prec * tpr / max(prec + tpr, 1e-9)})
    best = max(rows, key=lambda r: (r[metric], -abs(r["threshold"] - 0.5)))
    return best["threshold"], best[metric], rows


def fit_display_slope(y_fake, p_fake, cap=1.5):
    """Slope for the shown percentage (Platt scaling, balanced classes), capped so the
    percentage stays modest when the model is out of its comfort zone."""
    from sklearn.linear_model import LogisticRegression
    x = _logit(p_fake)[:, None]
    a = 0.0
    try:
        m = LogisticRegression(C=1e4, class_weight="balanced", max_iter=1000).fit(x, np.asarray(y_fake))
        a = float(m.coef_[0, 0])
    except Exception:
        pass
    if not np.isfinite(a) or a <= 1e-3:
        a = 2.0 / (float(np.std(x)) + 1e-6)
    return float(min(a, cap))


def display_percent(p_fake, threshold, slope=1.5):
    """How fake it looks, in %, with the decision threshold at exactly 50%."""
    z = slope * (_logit(p_fake) - _logit(threshold))
    return float(100.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


def verdict(percent_fake, band=15.0):
    if percent_fake >= 50 + band:
        return "FAKE"
    if percent_fake <= 50 - band:
        return "REAL"
    return "UNCERTAIN"


# ======================================================================= the detector
class LipSyncDetector:
    """Bundle = classifier + threshold + feature settings. Encoder and cropper are passed in."""

    def __init__(self, bundle, encoder: SyncEncoder, cropper: MouthCropper):
        self.b, self.enc, self.crop = bundle, encoder, cropper

    @staticmethod
    def _pack(m):
        m = dict(m)
        if m["kind"] == "seq":
            m["model"] = {k: v.cpu() for k, v in m["model"].state_dict().items()}
        if m["kind"] == "ensemble":
            m["members"] = [LipSyncDetector._pack(x) for x in m["members"]]
        return m

    @staticmethod
    def _unpack(m, device):
        if m["kind"] == "seq":
            net = SyncSeqNet(m["seq_dim"], act=m["act"])
            net.load_state_dict(m["model"]); m["model"] = net.to(device).eval()
        if m["kind"] == "ensemble":
            m["members"] = [LipSyncDetector._unpack(x, device) for x in m["members"]]
        return m

    @staticmethod
    def save_bundle(bundle, path):
        with open(path, "wb") as f:
            pickle.dump(LipSyncDetector._pack(bundle), f)

    @staticmethod
    def load_bundle(path, device="cpu"):
        with open(path, "rb") as f:
            return LipSyncDetector._unpack(pickle.load(f), device)

    @staticmethod
    def segment_scores(m, Cs, energy, device="cpu"):
        """P(fake) per 6 s segment (v2 models) or for the whole clip (v1 models)."""
        if m["kind"] == "ensemble":
            per = [LipSyncDetector.segment_scores(x, Cs, energy, device) for x in m["members"]]
            n = min(len(p) for p in per)
            return np.mean([p[:n] for p in per], 0)
        if m.get("v", 1) >= 2:
            if m["kind"] == "seq":
                return predict_seqnet(m["model"], sequence_input_v2(Cs), device)
            if m["features"] == "zs":
                return m["model"].predict_proba(np.array([[zero_shot_score(Cs[0])]], np.float32))[:, 1]
            X, rel = robust_features(Cs, energy)
            if m["features"] == "relative":
                X = X[:, rel]
            return m["model"].predict_proba(X)[:, 1]
        # v1 bundles
        if m["kind"] == "seq":
            return predict_seqnet(m["model"], [sequence_input(Cs, energy)], device)
        x = (np.array([[zero_shot_score(Cs[0])]], np.float32) if m["features"] == "zs"
             else sync_features(Cs, energy)[None])
        return m["model"].predict_proba(x)[:, 1]

    @staticmethod
    def member_score(m, Cs, energy, device="cpu"):
        return mean_prob(LipSyncDetector.segment_scores(m, Cs, energy, device))

    def score_from_matrices(self, Cs, energy):
        return self.member_score(self.b, Cs, energy, self.enc.device)

    def analyze(self, path, max_seconds=30):
        r = process_video(path, self.crop, self.enc, max_seconds=max_seconds)
        if r["T"] < 25:
            raise ValueError("video shorter than 1 second")
        Cs, energy, T, cinfo, crops = r["Cs"], r["E"], r["T"], r["info"], r["crops"]
        seg_p = self.segment_scores(self.b, Cs, energy, self.enc.device)
        p = mean_prob(seg_p)
        thr = self.b["threshold"]
        pct = display_percent(p, thr, self.b.get("display_slope", 1.5))
        d = best_offset(Cs[0])
        R0 = recenter(Cs, d)[0]
        X, rel, names = robust_features(Cs, energy, names=True, d=d, bounds=[(0, T)])
        f = dict(zip(names, X[0]))
        warn = []
        if T < 50:
            warn.append("short clip (< 2 s): less reliable")
        if cinfo["face_detected_frac"] < 0.6:
            warn.append(f"face found in only {cinfo['face_detected_frac']*100:.0f}% of frames")
        if cinfo["mouth_width_px"] < 25:
            warn.append("face is small in the frame: lips are low resolution")
        if float((energy > energy.max() - 30).mean()) < 0.3:
            warn.append("little speech in the clip: lip sync is hard to judge")
        if cinfo["frames_with_several_faces"] > 0:
            warn.append("several faces seen: the largest/most persistent one was analysed")
        ref = self.b.get("reference", {})
        for key, label in (("L0_peak", "lip-speech agreement"), ("L0_contrast", "sync contrast")):
            rr = ref.get("real", {}).get(key)
            if rr is not None and f[key] < rr[0]:
                warn.append(f"{label} {f[key]:.2f} is below what real training videos show "
                            f"(5th percentile {rr[0]:.2f}); unusual for a genuine recording")
        return {"p_fake": p, "threshold": thr, "is_fake": bool(p >= thr), "percent_fake": pct,
                "verdict": verdict(pct, self.b.get("uncertain_band", 15.0)), "segment_p_fake": list(map(float, seg_p)),
                "seconds": T / FPS, "best_offset_ms": int(d * 1000 / FPS),
                "agreement": float(f["L0_peak"]), "contrast": float(f["L0_contrast"]),
                "sync_score": zero_shot_score(Cs[0]),
                "per_frame_sync": np.nan_to_num(R0[:, RC_R]).tolist(), "energy": energy.tolist(),
                "warnings": warn, "mouth_crops": crops, "face_info": cinfo}


# ======================================================================= v3: SYNC CHECK (does the speech match the lips?)
EDGE = 4                       # frames ignored at each end of a clip (edge effects, start/end silence)
WIN, HOP = 50, 25              # 2 s windows, 1 s apart
NULL_GAP = 8                   # off-sync reference: offsets >= 8 frames (320 ms) away from the best offset
Z_HIT, Z_MISS = 2.0, 0.5       # window "matches" if z > 2; "does not match" if z < 0.5
MIN_WIN_FRAMES = 15            # a window needs >= 0.6 s of speech to be judged


def speech_mask(E, T=None):
    """Frames with speech: louder than (loudest - 30 dB) and than (quiet level + 3 dB).
    Digital silence never counts; the first/last EDGE frames are excluded."""
    e = np.asarray(E, np.float32)
    T = len(e) if T is None else int(T)
    e = e[:T]
    m = np.zeros(T, bool)
    live = e > -90.0
    if live.sum() < 5:
        return m
    floor = float(np.percentile(e[live], 10))
    m = live & (e > float(e[live].max()) - 30.0) & (e > floor + 3.0)
    m[:EDGE] = False
    m[max(T - EDGE, 0):] = False
    return m


def _neff(x, max_lag=6):
    """Effective number of independent samples in a series (positive autocorrelation only)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return float(max(n, 1))
    x = x - x.mean()
    v = float((x ** 2).mean())
    if v <= 0:
        return float(n)
    s = 0.0
    for k in range(1, min(max_lag, n // 4) + 1):
        r = float((x[:-k] * x[k:]).mean() / v)
        if r <= 0:
            break
        s += r
    return float(np.clip(n / (1.0 + 2.0 * s), 1.0, n))


def _slog(x):
    """Signed log compression for large z-scores."""
    return float(np.sign(x) * np.log1p(abs(x)))


def sync_evidence(C, E, d=None, tol=RC_TOL):
    """How clearly the lips match the speech, from one layer's band C (T, 31).

    Compares agreement at the best audio offset (within +-tol frames) with agreement of the same
    lips and the same audio >= NULL_GAP frames off-sync. Both sides share person, camera and
    microphone, so those cancel. Returns statistics for the whole clip and per 2 s window."""
    C = np.asarray(C, np.float32)
    T = len(C)
    m = speech_mask(E, T)
    weak_speech = m.sum() < 10
    if weak_speech:                         # too little speech: judge every non-edge frame
        m = np.zeros(T, bool)
        m[EDGE:max(T - EDGE, EDGE)] = True
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmean(C[m], 0) if m.any() else np.nanmean(C, 0)
        if d is None:
            sub = prof[TAU - tol: TAU + tol + 1]
            d = int(np.nanargmax(sub)) - tol if np.isfinite(sub).any() else 0
        jc = TAU + int(d)
        null = np.array([j for j in range(2 * TAU + 1) if abs(j - jc) >= NULL_GAP])
        c = C[:, jc]
        delta = c - np.nanmean(C[:, null], 1)
        ok = m & np.isfinite(delta)
        dv = delta[ok]
        n = int(len(dv))
        mu = float(np.mean(dv)) if n else 0.0
        sd = float(np.std(dv)) + 1e-4 if n > 1 else 1.0
        neff = _neff(dv) if n else 1.0
        tau_ = n / max(neff, 1.0) if n else 1.0
        z = mu / (sd / np.sqrt(max(neff, 1.0))) if n else 0.0
        pn = prof[null]
        prom = float((prof[jc] - np.nanmean(pn)) / (np.nanstd(pn) + 1e-3)) if np.isfinite(prof[jc]) else 0.0
        starts = list(range(0, max(T - WIN, 0) + 1, HOP))
        if T > WIN and (T - WIN) % HOP:
            starts.append(T - WIN)
        wz, wt = [], []
        for s in starts:
            sel = ok[s:s + WIN]
            if sel.sum() < MIN_WIN_FRAMES:
                continue
            dw = delta[s:s + WIN][sel]
            wz.append(float(dw.mean() / (sd / np.sqrt(max(len(dw) / tau_, 1.0)))))
            wt.append((s, min(s + WIN, T)))
    wz = np.asarray(wz, float)
    nw = len(wz)
    miss = wz < Z_MISS
    run, best = 0, 0
    for x in miss:
        run = run + 1 if x else 0
        best = max(best, run)
    return {
        "offset": int(d), "z": float(np.clip(z, -60, 60)), "contrast": mu, "prominence": float(np.clip(prom, -60, 60)),
        "hit_frac": float(np.mean(wz > Z_HIT)) if nw else 0.0,
        "miss_frac": float(np.mean(miss)) if nw else 1.0,
        "miss_run_frac": best / nw if nw else 1.0,
        "wz_p10": float(np.percentile(wz, 10)) if nw else 0.0,
        "wz_median": float(np.median(wz)) if nw else 0.0,
        "n_windows": nw, "speech_seconds": float(m.sum() / FPS) if not weak_speech else 0.0,
        "per_frame": np.nan_to_num(c).astype(np.float32), "delta": np.nan_to_num(delta).astype(np.float32),
        "mask": ok, "windows": wt, "window_z": wz,
    }


GATE_STATS = ["z", "contrast", "prominence", "hit_frac", "miss_frac", "miss_run_frac", "wz_p10", "wz_median"]
_LOGGED = {"z", "prominence", "wz_p10", "wz_median"}


def gate_features(Cs, E, names=False):
    """Sync-check features of every layer (the best offset is found on the final layer and reused).
    Returns x (n_features,), and the final layer's evidence dict (for display)."""
    ev0 = sync_evidence(Cs[0], E)
    evs = [ev0] + [sync_evidence(C, E, d=ev0["offset"]) for C in Cs[1:]]
    x = []
    for ev in evs:
        x += [_slog(ev[k]) if k in _LOGGED else float(ev[k]) for k in GATE_STATS]
    x += [float(np.mean([_slog(ev["z"]) for ev in evs])), float(np.mean([ev["hit_frac"] for ev in evs]))]
    x = np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if names:
        nm = [f"L{i}_{k}" for i in range(len(evs)) for k in GATE_STATS] + ["mean_z", "mean_hit_frac"]
        return x, ev0, nm
    return x, ev0


GATE_VARIANTS = {
    "sync check: logistic regression, all layers": None,                     # every column
    "sync check: final layer z + matched windows": ["L0_z", "L0_hit_frac"],
    "sync check: mean z + matched windows (all layers)": ["mean_z", "mean_hit_frac"],
}


def fit_gate(X, y, w, names, variant):
    """Logistic regression: P(audio does not belong to these lips). y: 1 = out of sync."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cols = GATE_VARIANTS[variant]
    idx = list(range(X.shape[1])) if cols is None else [names.index(c) for c in cols]
    w = np.asarray(w, float).copy()
    for c in (0, 1):
        if (y == c).any():
            w[y == c] *= len(y) / (2 * w[y == c].sum())
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    m.fit(X[:, idx], y, logisticregression__sample_weight=w)
    return {"kind": "gate", "variant": variant, "idx": idx, "model": m}


def gate_predict(g, X):
    X = np.atleast_2d(X)
    return g["model"].predict_proba(X[:, g["idx"]])[:, 1]


def trim_edges(Cs, E):
    """Drop the first/last EDGE frames (used for the deepfake-check features)."""
    T = len(Cs[0])
    if T <= 2 * EDGE + 25:
        return Cs, np.asarray(E)[:T]
    return [C[EDGE:T - EDGE] for C in Cs], np.asarray(E)[EDGE:T - EDGE]


# ======================================================================= v3: audio helpers for training negatives
def read_audio(path, max_seconds=None):
    """Audio only (int16, 16 kHz mono), placed by timestamps and aligned to the first video frame like read_av."""
    import av
    v0 = None
    with av.open(str(path)) as c:
        if c.streams.video:
            for f in c.decode(video=0):
                v0 = float(f.time) if f.time is not None else 0.0
                break
    a, a_start, _ = _decode_audio(path, None if max_seconds is None else max_seconds + 1.0)
    if a_start is not None and v0 is not None:
        k = int(round((a_start - v0) * SR))
        a = np.concatenate([np.zeros(k, np.int16), a]) if k > 0 else a[-k:]
    return a[: int(max_seconds * SR)] if max_seconds else a


def splice_audio(a, b, start, length, b_start, fade=320):
    """Replace a[start:start+length] with b[b_start:b_start+length] (samples), short cross-fades."""
    a = a.astype(np.float32).copy()
    seg = b[b_start:b_start + length].astype(np.float32)
    length = len(seg)
    if length < 2 * fade or start + length > len(a):
        return a.astype(np.int16)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    g = np.ones(length, np.float32)
    g[:fade], g[-fade:] = ramp, ramp[::-1]
    a[start:start + length] = a[start:start + length] * (1 - g) + seg * g
    return np.clip(a, -32768, 32767).astype(np.int16)


# ======================================================================= v3: thresholds and verdict
def _grid(groups):
    s = np.concatenate([np.asarray(g, float) for g in groups if len(g)] + [np.linspace(0, 1, 1001)])
    return np.unique(s)


def fake_threshold(real_groups, limits):
    """Grid search: the lowest threshold t (score >= t -> FAKE) such that, in every group of real
    videos, the share flagged is at most that group's limit."""
    for t in _grid(real_groups):
        if all(np.mean(np.asarray(g) >= t) <= lim for g, lim in zip(real_groups, limits) if len(g)):
            return float(t)
    return 1.0


def real_threshold(fake_groups, limit):
    """Grid search: the highest threshold t (score < t -> may be REAL) such that, in every group of
    fakes, at most `limit` of them fall below t."""
    for t in _grid(fake_groups)[::-1]:
        if all(np.mean(np.asarray(g) < t) <= limit for g in fake_groups if len(g)):
            return float(t)
    return 0.0


def calibrate(real_groups, real_limits, fake_groups, fake_limit):
    """-> (lo, hi): below lo the check says 'real', from hi on it says 'fake', in between 'uncertain'.
    If the classes separate (lo > hi), both meet in the middle (logit scale)."""
    hi = fake_threshold(real_groups, real_limits)
    lo = real_threshold(fake_groups, fake_limit)
    if lo > hi:
        mid = float(1 / (1 + np.exp(-(_logit(lo) + _logit(hi)) / 2)))
        lo = hi = mid
    return float(lo), float(hi)


def band_percent(p, lo, hi, min_gap=0.6):
    """Probability -> 'fake' percentage with lo at 35% and hi at 65% (logit-linear)."""
    x = float(_logit(p))
    a, b = float(_logit(lo)), float(_logit(hi))
    if b - a < min_gap:
        c = (a + b) / 2
        a, b = c - min_gap / 2, c + min_gap / 2
    k = np.log(65 / 35) / ((b - a) / 2)
    return float(100.0 / (1.0 + np.exp(-np.clip(k * (x - (a + b) / 2), -50, 50))))


MIN_SPEECH_S = 0.6               # below this the clip cannot be judged (UNCERTAIN, 50%)


def headline(verdict, percent_fake):
    """The one line shown to the user. fake % + real % = 100. REAL: fake <= 35%; FAKE: fake >= 65%."""
    f = float(np.clip(percent_fake, 0, 100))
    if verdict == "REAL":
        return f"LOOKS REAL — {100 - f:.0f}% real"
    if verdict == "FAKE":
        return f"LOOKS FAKE — {f:.0f}% fake"
    return f"UNCERTAIN — {f:.0f}% fake / {100 - f:.0f}% real"


def window_offsets(C, win=25, hop=12, min_valid=10):
    """Best audio offset (frames) in each short window of one layer's band: shows timing drift."""
    out = []
    T = len(C)
    for s in range(0, max(T - win, 0) + 1, hop):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            blk = C[s:s + win]
            cnt = np.isfinite(blk).sum(0)
            prof = np.where(cnt >= min_valid, np.nanmean(blk, 0), -np.inf)
        if np.isfinite(prof).any():
            out.append(((s + win / 2) / FPS, int(np.argmax(prof)) - TAU))
    return out


def combine(pct_sync, pct_deepfake):
    """Verdict from the two checks. FAKE if either check reaches 65%; REAL only if both are at most 35%."""
    pct = max(pct_sync, pct_deepfake)
    if pct >= 65:
        return "FAKE", pct, ("sync" if pct_sync >= pct_deepfake else "deepfake")
    if pct <= 35:
        return "REAL", pct, None
    return "UNCERTAIN", pct, ("sync" if pct_sync >= pct_deepfake else "deepfake")


def _ranges(windows, z, thr=Z_MISS, merge_gap=0):
    """Time ranges (s) of windows with z < thr, merged."""
    out = []
    for (s, e), v in zip(windows, z):
        if v < thr:
            if out and s <= out[-1][1] + merge_gap:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
    return [(a / FPS, b / FPS) for a, b in out]


class LipSyncDetectorV3:
    """Bundle = sync check (gate) + deepfake check (classifier) + calibrated thresholds."""

    def __init__(self, bundle, encoder: SyncEncoder, cropper: MouthCropper):
        self.b, self.enc, self.crop = bundle, encoder, cropper

    @staticmethod
    def save_bundle(bundle, path):
        b = dict(bundle)
        b["cls"] = LipSyncDetector._pack(b["cls"])
        with open(path, "wb") as f:
            pickle.dump(b, f)

    @staticmethod
    def load_bundle(path, device="cpu"):
        with open(path, "rb") as f:
            b = pickle.load(f)
        b["cls"] = LipSyncDetector._unpack(b["cls"], device)
        return b

    @staticmethod
    def scores(b, Cs, E, device="cpu"):
        """-> P(out of sync), P(lip-sync deepfake), sync evidence (final layer), segment scores."""
        xg, ev = gate_features(Cs, E)
        g = float(gate_predict(b["gate"], xg[None])[0])
        Ct, Et = trim_edges(Cs, E)
        seg = LipSyncDetector.segment_scores(b["cls"], Ct, Et, device)
        return g, mean_prob(seg), ev, seg

    @staticmethod
    def verdict_from_scores(b, g, c, ev, info=None, seconds=None):
        th = b["thresholds"]
        ps = band_percent(g, th["sync_lo"], th["sync_hi"])
        pd_ = band_percent(c, th["deepfake_lo"], th["deepfake_hi"])
        v, pct, why = combine(ps, pd_)
        notes = []
        ref = b.get("reference", {})
        hit = ev["hit_frac"] * 100
        rr = ref.get("real_hit_frac")
        if v == "FAKE" and why == "sync":
            rng = _ranges(ev["windows"], ev["window_z"])
            rs = ", ".join(f"{a:.1f}–{e:.1f} s" for a, e in rng[:4])
            reason = (f"the speech does not match the lips: they matched in only {hit:.0f}% of the 2-second speech windows"
                      + (f" (real videos: usually {rr[0]*100:.0f}% or more)" if rr and rr[0] >= 0.05 else "")
                      + (f"; no match at {rs}" if rs else ""))
        elif v == "FAKE":
            reason = ("the lips follow the speech, but in the way lip-sync deepfakes do "
                      f"(deepfake check {pd_:.0f}%; lips matched the speech in {hit:.0f}% of speech windows)")
        elif v == "REAL":
            reason = (f"lips and speech match in {hit:.0f}% of the 2-second speech windows, "
                      "and nothing looks like a lip-sync deepfake")
        else:
            parts = []
            if 35 < ps < 65:
                parts.append(f"the lip–speech match is weak (matched in {hit:.0f}% of speech windows)")
            if 35 < pd_ < 65:
                parts.append(f"the deepfake check is in its grey zone ({pd_:.0f}%)")
            reason = "; ".join(parts) or "the two checks disagree"
        info = info or {}
        hard = []
        if ev["speech_seconds"] < MIN_SPEECH_S:
            hard.append(f"less than {MIN_SPEECH_S:.1f} s of speech: lip sync cannot be judged")
        if info.get("face_detected_frac", 1.0) < 0.5:
            hard.append(f"face found in only {info['face_detected_frac']*100:.0f}% of frames")
        if hard:
            v, pct = "UNCERTAIN", 50.0          # cannot be judged: no lean either way
            reason = "; ".join(hard)
        if info.get("mouth_width_px", 99) < 25:
            notes.append("the face is small in the frame: lips are low resolution")
        if info.get("frames_with_several_faces", 0) > 0:
            notes.append("several faces seen: the largest/most persistent one was analysed")
        if seconds is not None and seconds < 2.5:
            notes.append("short clip (< 2.5 s): less reliable")
        if info.get("retimed_to_fps"):
            notes.append(f"the file stores {info.get('stored_fps', 0):.0f} fps but the footage is "
                         f"{info['retimed_to_fps']:g} fps (video shorter than audio); frames were re-timed")
        if info.get("audio_gap_s", 0) > 0.1:
            notes.append(f"the audio track had a {info['audio_gap_s']:.2f} s gap; it was filled with silence, "
                         "as a video player does")
        if abs(ev["offset"]) >= RC_TOL:
            notes.append(f"audio is offset by {ev['offset']*40:+d} ms, at the edge of the allowed range")
        return {"verdict": v, "percent_fake": float(pct), "percent_real": 100.0 - float(pct), "why": why,
                "reason": reason, "notes": notes, "pct_sync": ps, "pct_deepfake": pd_, "hard_flags": hard,
                "headline": headline(v, pct)}

    def analyze(self, path, max_seconds=30):
        r = process_video(path, self.crop, self.enc, max_seconds=max_seconds)
        if r["T"] < 25:
            raise ValueError("video shorter than 1 second")
        g, c, ev, seg = self.scores(self.b, r["Cs"], r["E"], self.enc.device)
        out = self.verdict_from_scores(self.b, g, c, ev, r["info"], r["T"] / FPS)
        out.update({"p_out_of_sync": g, "p_deepfake": c, "segment_p_deepfake": list(map(float, seg)),
                    "seconds": r["T"] / FPS, "best_offset_ms": int(ev["offset"] * 1000 / FPS),
                    "matched_windows": ev["hit_frac"], "sync_z": ev["z"], "speech_seconds": ev["speech_seconds"],
                    "per_frame_sync": ev["per_frame"].tolist(), "speech_mask": ev["mask"].tolist(),
                    "windows": ev["windows"], "window_z": ev["window_z"].tolist(),
                    "energy": np.asarray(r["E"]).tolist(), "mouth_crops": r["crops"], "face_info": r["info"]})
        return out
