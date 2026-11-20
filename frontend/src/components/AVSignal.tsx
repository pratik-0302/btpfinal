/**
 * Decorative illustration only: a stylised mouth opening and closing in rhythm with a speech-like waveform.
 * It is not connected to any analysis and shows no data (no axes, no numbers); hidden from assistive technology.
 */

const N_BARS = 56;
// fixed, smooth "speech-like" envelope (deterministic so it never looks like live data)
const BARS = Array.from({ length: N_BARS }, (_, i) => {
  const x = i / (N_BARS - 1);
  const env = Math.sin(Math.PI * x) ** 0.6;
  const syll = 0.55 + 0.45 * Math.abs(Math.sin(i * 0.55)) * Math.abs(Math.cos(i * 0.21));
  return Math.max(0.12, env * syll);
});

export function AVSignal({ className = "" }: { className?: string }) {
  const W = 640;
  const H = 96;
  const mid = H / 2;
  const x0 = 170;
  const x1 = W - 24;
  const step = (x1 - x0) / N_BARS;
  return (
    <div className={`av-signal ${className}`} aria-hidden>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full overflow-visible" role="presentation">
        <defs>
          <linearGradient id="av-bar" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stopColor="#60a5fa" />
            <stop offset="1" stopColor="#a78bfa" />
          </linearGradient>
          <linearGradient id="av-link" x1="0" x2="1">
            <stop offset="0" stopColor="#a78bfa" stopOpacity="0" />
            <stop offset="0.5" stopColor="#a78bfa" stopOpacity="0.7" />
            <stop offset="1" stopColor="#60a5fa" stopOpacity="0" />
          </linearGradient>
          <radialGradient id="av-glow">
            <stop offset="0" stopColor="#8b5cf6" stopOpacity="0.35" />
            <stop offset="1" stopColor="#8b5cf6" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* mouth */}
        <circle cx="70" cy={mid} r="46" fill="url(#av-glow)" />
        <g fill="none" stroke="url(#av-bar)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d={`M30 ${mid} C 48 ${mid - 16}, 62 ${mid - 10}, 70 ${mid - 6} C 78 ${mid - 10}, 92 ${mid - 16}, 110 ${mid}`} />
          <path className="av-lip" d={`M30 ${mid} C 50 ${mid + 18}, 90 ${mid + 18}, 110 ${mid}`} />
          <path className="av-gap" d={`M38 ${mid} C 55 ${mid + 6}, 85 ${mid + 6}, 102 ${mid}`} strokeOpacity="0.45" strokeWidth="1.5" />
        </g>

        {/* link between lips and speech */}
        <line x1="122" x2={x0 - 8} y1={mid} y2={mid} stroke="url(#av-link)" strokeWidth="1.5" strokeDasharray="3 4" className="av-link" />

        {/* waveform */}
        <g fill="url(#av-bar)">
          {BARS.map((h, i) => {
            const bh = h * (H - 16);
            return (
              <rect
                key={i}
                className="av-bar"
                x={x0 + i * step + step * 0.2}
                y={mid - bh / 2}
                width={step * 0.6}
                height={bh}
                rx={step * 0.3}
                style={{ animationDelay: `${(i % 14) * -0.09}s`, transformOrigin: `${x0 + i * step + step / 2}px ${mid}px` }}
              />
            );
          })}
        </g>
      </svg>
      <p className="mt-2 text-center text-[10px] font-medium uppercase tracking-[0.2em] text-ink-3/70">Illustration</p>
    </div>
  );
}
