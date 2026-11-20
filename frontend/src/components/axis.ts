/** Time-axis label that never repeats on short clips (e.g. 0.5 s steps are shown as 0.5, not rounded to 1). */
export const secondsTick = (v: number) => `${Number.isInteger(v) ? v : v.toFixed(1)} s`;
