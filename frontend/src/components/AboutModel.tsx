import { useEffect } from "react";
import { X } from "lucide-react";
import type { ModelInfo } from "../types";

/** How the detector works, what every number means, and the notebook's evaluation results (with their test sets). */
export function AboutModel({ info, onClose }: { info: ModelInfo | null; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/50 p-4" role="dialog" aria-modal="true" aria-label="About the model">
      <div className="card my-8 w-full max-w-4xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-xl font-bold">About the detector</h2>
          <button type="button" onClick={onClose} aria-label="Close" className="rounded p-1 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X size={20} />
          </button>
        </div>
        {!info ? (
          <p className="text-sm text-slate-500">Loading…</p>
        ) : (
          <div className="space-y-6 text-sm">
            <section>
              <h3 className="mb-2 font-semibold">How it works</h3>
              <p className="text-slate-600 dark:text-slate-300">
                The video is decoded at 25 fps with 16 kHz audio. A face detector (MTCNN) finds the mouth, and a frozen
                AV-HuBERT Large model turns the mouth movements and the speech into separate embeddings. Their cosine
                agreement is measured at audio shifts of up to ±600 ms. Two logistic-regression classifiers use only
                statistics of this agreement: the <b>sync check</b> (does this speech belong to these lips?) and the{" "}
                <b>deepfake check</b> (do the lips follow the speech the way lip-sync deepfakes do?). The verdict uses the
                higher of the two scores.
              </p>
            </section>

            <section>
              <h3 className="mb-2 font-semibold">What the numbers mean</h3>
              <dl className="space-y-2">
                {Object.entries(info.definitions).map(([k, d]) => (
                  <div key={k}>
                    <dt className="font-medium">{d.label}</dt>
                    <dd className="text-slate-600 dark:text-slate-300">
                      {d.meaning} {d.not && <span className="font-medium text-amber-700 dark:text-amber-400">{d.not}</span>}
                    </dd>
                  </div>
                ))}
              </dl>
            </section>

            <section>
              <h3 className="mb-1 font-semibold">Evaluation results (from the training notebook)</h3>
              <p className="mb-2 text-xs text-slate-500">{info.evaluation.source}</p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="text-left text-slate-500">
                    <tr>
                      <th className="py-1 pr-2">Test set</th><th className="pr-2">real</th><th className="pr-2">fake</th>
                      <th className="pr-2">AUC (95% CI)</th><th className="pr-2">EER</th><th className="pr-2">forced-choice acc.</th>
                      <th className="pr-2">balanced acc.</th><th>TPR @ 1% FPR</th>
                    </tr>
                  </thead>
                  <tbody className="tabular-nums">
                    {info.evaluation.headline_sets.map((s) => (
                      <tr key={s.set} className="border-t border-slate-200 dark:border-slate-800">
                        <td className="py-1 pr-2">{s.set}</td><td>{s.n_real}</td><td>{s.n_fake}</td>
                        <td>{s.auc.toFixed(3)} ({s.auc_ci})</td><td>{s.eer_pct.toFixed(1)}%</td>
                        <td>{s.forced_choice_accuracy_pct.toFixed(1)}%</td><td>{s.balanced_accuracy_pct.toFixed(1)}%</td>
                        <td>{s.tpr_at_fpr1_pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <h4 className="mb-1 mt-4 font-medium">Verdicts shown to the user, per set</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="text-left text-slate-500">
                    <tr><th className="py-1 pr-2">Set</th><th className="pr-2">n</th><th className="pr-2">truth</th><th className="pr-2">REAL</th><th className="pr-2">UNCERTAIN</th><th>FAKE</th></tr>
                  </thead>
                  <tbody className="tabular-nums">
                    {info.evaluation.verdict_breakdown.map((s) => (
                      <tr key={s.set} className="border-t border-slate-200 dark:border-slate-800">
                        <td className="py-1 pr-2">{s.set}</td><td>{s.n}</td><td>{s.truth}</td>
                        <td>{s.real_pct.toFixed(1)}%</td><td>{s.uncertain_pct.toFixed(1)}%</td><td>{s.fake_pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <h4 className="mb-1 mt-4 font-medium">LipSyncTIMIT test fakes by group (vs all 96 test real videos)</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="text-left text-slate-500"><tr><th className="py-1 pr-2">Group</th><th className="pr-2">fakes</th><th className="pr-2">AUC</th><th>verdict FAKE</th></tr></thead>
                  <tbody className="tabular-nums">
                    {info.evaluation.by_generator.map((g) => (
                      <tr key={g.group} className="border-t border-slate-200 dark:border-slate-800">
                        <td className="py-1 pr-2">{g.group}</td><td>{g.n_fake}</td><td>{g.auc.toFixed(3)}</td><td>{g.fake_pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-slate-600 dark:text-slate-300">
                {info.evaluation.notes.map((n) => <li key={n}>{n}</li>)}
              </ul>
            </section>

            <section>
              <h3 className="mb-2 font-semibold">Limitations</h3>
              <ul className="list-disc space-y-1 pl-5 text-slate-600 dark:text-slate-300">
                {info.limitations.map((l) => <li key={l}>{l}</li>)}
              </ul>
            </section>

            <section>
              <h3 className="mb-2 font-semibold">Display-only visualisations</h3>
              <ul className="list-disc space-y-1 pl-5 text-slate-600 dark:text-slate-300">
                {Object.entries(info.new_displays).map(([k, t]) => <li key={k}>{t}</li>)}
              </ul>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
