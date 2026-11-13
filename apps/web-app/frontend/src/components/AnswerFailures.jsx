// Answers of a run that failed to generate (the pipeline raised, e.g. for a missing API key).
// They have no verdict, so the results leave them out: say how many, in which configurations,
// and why. `failures` is a results payload's `answer_failures`; nothing renders when none failed.
export const configText = (config) =>
  Object.keys(config || {}).sort().map((k) => `${k}=${config[k] ?? "none"}`).join(" · ") || "all";

export default function AnswerFailures({ failures, style }) {
  if (!failures?.n_failed) return null;
  const configs = failures.by_config || [];
  return (
    <div className="banner" style={style}>
      <b>{failures.n_failed} of {failures.n_answers} answers failed</b> in this run. A failed answer
      has no verdict, so the results leave it out.
      {configs.length > 0 && (
        <div className="hint mono" style={{ marginTop: 6 }}>
          {configs.slice(0, 8).map((c) => `${configText(c.config)}: ${c.n_failed} of ${c.n_answers} failed`).join(" · ")}
          {configs.length > 8 && ` · and ${configs.length - 8} more configurations`}
        </div>
      )}
      {failures.first_error && (
        <div className="mono" style={{ marginTop: 6, fontSize: 12, color: "var(--error)", overflowWrap: "anywhere" }}>
          first error: {failures.first_error}</div>
      )}
    </div>
  );
}
