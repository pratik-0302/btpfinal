import { useEffect, useRef } from "react";

// Perceptually ordered colour map (viridis anchors), interpolated linearly.
const ANCHORS: [number, number, number][] = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
];

function colour(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (ANCHORS.length - 1);
  const i = Math.min(Math.floor(x), ANCHORS.length - 2);
  const f = x - i;
  const a = ANCHORS[i];
  const b = ANCHORS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/**
 * rows = time steps, columns = bands/offsets. Drawn with time on the x axis and column 0 at the bottom.
 * null cells (outside the clip) are drawn transparent.
 */
export function Heatmap({ values, min, max, height = 140, label }: {
  values: (number | null)[][];
  min: number;
  max: number;
  height?: number;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || values.length === 0) return;
    const T = values.length;
    const K = values[0].length;
    cv.width = T;
    cv.height = K;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(T, K);
    const span = max - min || 1;
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < K; k++) {
        const v = values[t][k];
        const p = ((K - 1 - k) * T + t) * 4;
        if (v == null) {
          img.data[p + 3] = 0;
          continue;
        }
        const [r, g, b] = colour((v - min) / span);
        img.data[p] = r;
        img.data[p + 1] = g;
        img.data[p + 2] = b;
        img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [values, min, max]);
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={label}
      className="pixelated w-full rounded-lg border border-white/10 bg-white/5"
      style={{ height }}
    />
  );
}

export function ColourScale({ min, max, unit }: { min: number; max: number; unit?: string }) {
  const stops = ANCHORS.map((c, i) => `rgb(${c.join(",")}) ${(i / (ANCHORS.length - 1)) * 100}%`).join(", ");
  return (
    <div className="flex items-center gap-2 text-[11px] text-ink-2">
      <span className="tabular-nums">{min.toFixed(2)}</span>
      <div className="h-2 w-32 rounded" style={{ background: `linear-gradient(to right, ${stops})` }} />
      <span className="tabular-nums">{max.toFixed(2)}</span>
      {unit && <span>{unit}</span>}
    </div>
  );
}
