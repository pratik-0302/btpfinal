import { useEffect, useState } from "react";
import { VideoOff } from "lucide-react";

/**
 * Plays the user's own file straight from the browser (object URL). This is completely separate from the
 * detector, which decodes its own copy with PyAV on the server, so playback never affects the analysis.
 */
export function VideoPreview({ file }: { file: File }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const u = URL.createObjectURL(file);
    setUrl(u);
    setFailed(false);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  if (!url) return null;
  return failed ? (
    <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 rounded-lg bg-slate-100 p-4 text-center text-sm text-slate-500 dark:bg-slate-800">
      <VideoOff size={28} />
      This browser cannot play this video format. The analysis does not depend on the preview and still works.
    </div>
  ) : (
    <video
      key={url}
      src={url}
      controls
      playsInline
      preload="metadata"
      onError={() => setFailed(true)}
      className="aspect-video w-full rounded-lg bg-black object-contain"
    />
  );
}
