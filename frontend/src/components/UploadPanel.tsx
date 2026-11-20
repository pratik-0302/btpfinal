import { useRef, useState } from "react";
import { Clock, FileVideoCamera, HardDrive, LoaderCircle, Upload } from "lucide-react";
import { fmtBytes } from "../format";

export function UploadPanel({ disabled, extensions, maxMb, onFile, busyReason, large = false }: {
  disabled: boolean;
  extensions: string[];
  maxMb: number | null;
  onFile: (f: File) => void;
  busyReason: string | null;
  large?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  // dragenter/dragleave also fire for child elements; count them so the highlight doesn't flicker
  const dragDepth = useRef(0);
  const [localError, setLocalError] = useState<string | null>(null);

  const pick = (f: File | undefined) => {
    if (!f) return;
    const ext = f.name.includes(".") ? f.name.slice(f.name.lastIndexOf(".")).toLowerCase() : "";
    if (extensions.length && !extensions.includes(ext)) {
      setLocalError(`"${ext || "no extension"}" is not supported. Use one of: ${extensions.join(" ")}`);
      return;
    }
    if (maxMb && f.size > maxMb * 1024 * 1024) {
      setLocalError(`The file is ${fmtBytes(f.size)}; the limit is ${maxMb} MB.`);
      return;
    }
    setLocalError(null);
    onFile(f);
  };

  const sizeLimit = maxMb ? (maxMb >= 1024 ? `${+(maxMb / 1024).toFixed(1)} GB` : `${maxMb} MB`) : null;
  const PREFERRED = ["MP4", "MOV", "WEBM", "MKV", "AVI"];
  const formats = (extensions.length ? extensions.map((e) => e.replace(".", "").toUpperCase()) : []).sort(
    (a, b) => (PREFERRED.indexOf(a) + 1 || 99) - (PREFERRED.indexOf(b) + 1 || 99),
  );
  const shownFormats = formats.slice(0, 5);
  const iconSize = large ? 38 : 22;

  return (
    <div>
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        onClick={() => !disabled && input.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === "Enter" || e.key === " ")) input.current?.click();
        }}
        onDragEnter={(e) => {
          e.preventDefault();
          dragDepth.current += 1;
          if (!disabled) setDrag(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
        }}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDrag(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          dragDepth.current = 0;
          setDrag(false);
          if (!disabled) pick(e.dataTransfer.files?.[0]);
        }}
        data-drag={drag}
        className={`upload-zone group relative flex flex-col items-center justify-center overflow-hidden rounded-[1.6rem] border border-dashed text-center transition-all duration-500 ease-out focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
          large ? "gap-6 px-6 py-16 sm:gap-7 sm:px-10 sm:py-20 lg:py-24" : "gap-4 px-5 py-10"
        } ${
          disabled
            ? "cursor-not-allowed border-white/10"
            : drag
              ? "scale-[1.015] cursor-copy border-accent-hover/80 bg-accent/[0.09] shadow-[inset_0_0_80px_-20px_rgb(99_102_241/0.55)]"
              : "cursor-pointer border-white/[0.14] hover:border-accent-hover/45 hover:bg-white/[0.025]"
        }`}
      >
        {/* soft light behind the icon */}
        <span aria-hidden className={`pointer-events-none absolute left-1/2 top-[38%] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(closest-side,rgb(99_102_241/0.26),transparent)] transition-all duration-700 group-hover:opacity-100 ${large ? "size-96" : "size-64"} ${drag ? "scale-125 opacity-100" : "opacity-70"}`} />

        <span className={`relative transition-transform duration-500 ease-out ${disabled ? "" : drag ? "-translate-y-1.5 scale-110" : "animate-float"}`}>
          {/* halo rings */}
          {!disabled && (
            <>
              <span aria-hidden className={`absolute -inset-3 rounded-[1.9rem] border border-accent-hover/20 transition-opacity duration-500 ${drag ? "opacity-100" : "opacity-60"}`} />
              <span aria-hidden className={`absolute -inset-3 rounded-[1.9rem] border border-violet/40 ${drag ? "animate-ping" : "hidden"}`} />
            </>
          )}
          <span
            className={`icon-chip relative ${large ? "size-24 rounded-[1.6rem] shadow-[inset_0_1px_0_rgb(255_255_255/0.15),0_18px_50px_-14px_rgb(99_102_241/0.85)]" : "size-14 rounded-2xl"} ${
              disabled ? "opacity-60" : "transition-transform duration-300 group-hover:scale-105"
            }`}
          >
            {disabled && busyReason ? (
              <LoaderCircle size={iconSize} className="animate-spin" />
            ) : drag ? (
              <FileVideoCamera size={iconSize} />
            ) : (
              <Upload size={iconSize} />
            )}
          </span>
        </span>

        <div className="relative">
          <h2 className={`font-semibold tracking-[-0.02em] ${large ? "text-[1.75rem] sm:text-4xl" : "text-xl"}`}>
            {busyReason ?? (drag ? "Drop to analyze" : "Analyze Your Video")}
          </h2>
          <p className={`mt-2.5 text-ink-2 ${large ? "text-[15px] sm:text-lg" : "text-sm sm:text-[15px]"}`}>
            {busyReason ? "The upload opens as soon as the model is ready." : "Drag and drop your video or browse files."}
          </p>
        </div>

        {!disabled && (
          <span className={`btn-primary pointer-events-none relative ${large ? "!px-7 !py-3 !text-[15px]" : ""}`}>
            <Upload size={16} /> Choose Video
          </span>
        )}

        <div className={`relative flex flex-wrap items-center justify-center gap-2 text-ink-2 ${large ? "text-xs sm:text-[13px]" : "text-xs"}`}>
          {shownFormats.length > 0 && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1" title={formats.join(", ")}>
              <FileVideoCamera size={13} className="text-ink-3" />
              {shownFormats.join(" · ")}
              {formats.length > shownFormats.length ? ` +${formats.length - shownFormats.length}` : ""}
            </span>
          )}
          {sizeLimit && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1">
              <HardDrive size={13} className="text-ink-3" /> Up to {sizeLimit}
            </span>
          )}
          <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1">
            <Clock size={13} className="text-ink-3" /> First 30 s analysed
          </span>
        </div>
        <p className="relative text-xs text-ink-3">Needs a clearly visible talking face and its audio track.</p>

        <input
          ref={input}
          type="file"
          accept={extensions.length ? extensions.join(",") : "video/*"}
          className="hidden"
          onChange={(e) => {
            pick(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>
      {localError && (
        <p role="alert" className="mt-3 rounded-xl border border-bad/30 bg-bad/10 px-4 py-2.5 text-center text-sm text-bad">
          {localError}
        </p>
      )}
    </div>
  );
}
