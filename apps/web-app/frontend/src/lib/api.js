// Thin fetch wrapper around the backend JSON API (proxied to :8000 in dev).
// In DEMO mode it serves a frozen static snapshot instead and blocks mutations (see lib/demo.js).
import { DEMO, demoFile, demoToast } from "./demo.js";

const BASE = "/api";

// POST endpoints that are actually safe reads (no state change) — served from the snapshot in demo.
const SAFE_POST = { "/pipelines/reload": "/pipelines", "/judge/preview": "/judge/preview" };

async function demoRequest(path, method) {
  if (method === "GET" || path in SAFE_POST) {
    if (path.includes("archived=true")) return [];   // demo has no archived studies
    const file = demoFile(SAFE_POST[path] || path);
    const res = await fetch(file);
    if (!res.ok) {
      // sensible fallbacks so pages never crash on a missing snapshot file
      return /\/(studies|rubrics|datasets|pipelines|raters|presets)(\?|$)/.test(path) ? [] : {};
    }
    return res.json();
  }
  demoToast();                       // create/run/delete/archive/rate → blocked
  throw new Error("Read-only demo — changes are disabled here.");
}

async function request(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  if (DEMO) return demoRequest(path, method);

  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (res.status === 204) return null;
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

const get = (p) => request(p);
const post = (p, body) => request(p, { method: "POST", body: JSON.stringify(body) });
const del = (p) => request(p, { method: "DELETE" });

export const api = {
  health: () => get("/health"),
  pipelines: () => get("/pipelines"),
  reloadPipelines: () => post("/pipelines/reload", {}),
  judgePreview: (b) => post("/judge/preview", b),
  judgePresets: () => get("/judge/presets"),
  dataset: (id) => get(`/datasets/${id}`),

  rubrics: () => get("/rubrics"),
  rubric: (id) => get(`/rubrics/${id}`),
  createRubric: (b) => post("/rubrics", b),
  deleteRubric: (id) => del(`/rubrics/${id}`),

  datasets: () => get("/datasets"),
  createDataset: (b) => post("/datasets", b),
  deleteDataset: (id) => del(`/datasets/${id}`),

  studies: (archived = false) => get(`/studies${archived ? "?archived=true" : ""}`),
  study: (id) => get(`/studies/${id}`),
  createStudy: (b) => post("/studies", b),
  importStudy: (bundle, attachTo) =>                          // a cafe.save_evaluation bundle;
    post(`/studies/import${attachTo ? `?attach_to=${attachTo}` : ""}`, bundle),  // attachTo = add as dimension
  archiveStudy: (id) => post(`/studies/${id}/archive`, {}),
  restoreStudy: (id) => post(`/studies/${id}/restore`, {}),
  deleteStudy: (id) => del(`/studies/${id}`),
  runStudy: (id) => post(`/studies/${id}/run`, {}),
  estimateStudy: (id) => post(`/studies/${id}/estimate`, {}),
  results: (id, dimension, filter) => {
    const params = new URLSearchParams();
    if (dimension) params.set("dimension", dimension);
    if (filter?.key) { params.set("fkey", filter.key); params.set("fval", filter.value); }
    const qs = params.toString();
    return get(`/studies/${id}/results${qs ? `?${qs}` : ""}`);
  },
  dimensions: (id) => get(`/studies/${id}/dimensions`),
  studyFilters: (id) => get(`/studies/${id}/filters`),   // question-metadata keys -> values
  streamUrl: (id) => `${BASE}/studies/${id}/stream`,   // for EventSource

  ratingSheet: (id, n = 40) => get(`/studies/${id}/rating-sheet?n=${n}`),
  submitHumanRatings: (id, b) => post(`/studies/${id}/human-ratings`, b),
  studyRaters: (id) => get(`/studies/${id}/raters`),
  reliability: (id) => get(`/studies/${id}/reliability`),
};

// Follow a study's run on its progress stream until the run is over. The connection can drop
// mid-run (a proxy restart or idle timeout, a network blip), and EventSource's own retry gives up
// for good on an HTTP error (say a proxy's 502 while the backend restarts). So on any error the
// stream is closed and the study is read again after a growing wait: still running → subscribe
// again; otherwise the run ended while we were cut off. After a "done"/"failed" event the study
// is read again too, as the run's final state.
//   onProgress(p)        each progress event ({ phase, done, total, status, error }); a run that
//                        ended while cut off gets a last { phase: "idle", status } from the study
//   onEnd(study, error)  once: the run is over (study = the study as read last), or following it
//                        failed MAX_RETRIES times in a row (study = null, error = the last one)
// Returns stop(), which closes the stream and cancels a pending retry.
const RETRY_MS = [1000, 2000, 4000, 8000, 15000];   // wait before the n-th retry in a row
const MAX_RETRIES = 20;                              // ≈ 4 minutes of failures, then give up

export function followRun(id, { onProgress, onEnd = () => {} }) {
  let src = null, timer = null, stopped = false, retries = 0, last = null;
  const emit = (p) => { last = p.status; onProgress(p); };
  const stop = () => { stopped = true; clearTimeout(timer); src?.close(); };
  const subscribe = () => {
    src = new EventSource(api.streamUrl(id));
    src.onmessage = (e) => {
      const p = JSON.parse(e.data);
      emit(p);
      if (p.status === "running") { retries = 0; return; }
      src.close();           // done / failed, or the study has no run (anymore)
      check(true);
    };
    src.onerror = () => { src.close(); retry(); };
  };
  const retry = (error) => {
    if (stopped) return;
    if (retries >= MAX_RETRIES) { stop(); onEnd(null, error); return; }
    timer = setTimeout(() => check(false), RETRY_MS[Math.min(retries, RETRY_MS.length - 1)]);
    retries += 1;
  };
  // Read the study: still running → subscribe again (but right after the stream reported the
  // end, look again in a moment: the backend writes a failed run's status just after the event).
  const check = (afterEnd) => api.study(id).then((s) => {
    if (stopped) return;
    if (s.status === "running") { if (afterEnd) retry(); else subscribe(); return; }
    stop();
    if (last === null || last === "running") emit({ phase: "idle", status: s.status });
    onEnd(s, null);
  }, (e) => retry(e));
  subscribe();
  return stop;
}

// Run phases on the progress stream (backend runner.PROGRESS): "starting", "answers", "judging",
// "judging · <rubric>" (one per extra dimension), "saving", then "done" / "error".
export const isJudgingPhase = (phase) => (phase || "").startsWith("judging");
export const phaseLabel = (phase) => (!phase || phase === "answers" ? "answering" : phase);
