import { useRef, useState } from "react";
import { FileVideoCamera, Upload } from "lucide-react";
import { fmtBytes } from "../format";

export function UploadPanel({ disabled, extensions, maxMb, onFile, busyReason }: {
  disabled: boolean;
  extensions: string[];
  maxMb: number | null;
  onFile: (f: File) => void;
  busyReason: string | null;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
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
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (!disabled) pick(e.dataTransfer.files?.[0]);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center transition ${
          disabled
            ? "cursor-not-allowed border-slate-200 opacity-60 dark:border-slate-800"
            : drag
              ? "border-sky-500 bg-sky-50 dark:bg-sky-950/40"
              : "border-slate-300 hover:border-sky-400 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-900"
        }`}
      >
        {drag ? <FileVideoCamera size={32} className="text-sky-600" /> : <Upload size={32} className="text-slate-400" />}
        <div className="text-sm font-medium">{busyReason ?? "Drop a video here or click to choose"}</div>
        <div className="text-xs text-slate-500">
          A talking face with sound · first 30 s analysed{maxMb ? ` · up to ${maxMb} MB` : ""}
        </div>
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
      {localError && <p className="mt-2 text-sm text-rose-600">{localError}</p>}
    </div>
  );
}
