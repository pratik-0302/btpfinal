export const fmtBytes = (n?: number | null) => {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
};

export const fmtNum = (n: number | null | undefined, digits = 2) =>
  n == null || Number.isNaN(n) ? "—" : n.toFixed(digits);

export const fmtPct = (frac: number | null | undefined, digits = 0) =>
  frac == null ? "—" : `${(frac * 100).toFixed(digits)}%`;

export const fmtSec = (s: number | null | undefined) => {
  if (s == null) return "—";
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
};

export const signed = (n: number) => (n > 0 ? `+${n}` : `${n}`);

export const verdictStyle = (label: string) =>
  label === "REAL"
    ? { text: "text-emerald-700 dark:text-emerald-400", bg: "bg-emerald-50 dark:bg-emerald-950/40", ring: "ring-emerald-300 dark:ring-emerald-800", bar: "#059669" }
    : label === "FAKE"
      ? { text: "text-rose-700 dark:text-rose-400", bg: "bg-rose-50 dark:bg-rose-950/40", ring: "ring-rose-300 dark:ring-rose-800", bar: "#e11d48" }
      : { text: "text-amber-700 dark:text-amber-400", bg: "bg-amber-50 dark:bg-amber-950/40", ring: "ring-amber-300 dark:ring-amber-800", bar: "#d97706" };
