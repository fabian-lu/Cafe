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
  archiveStudy: (id) => post(`/studies/${id}/archive`, {}),
  restoreStudy: (id) => post(`/studies/${id}/restore`, {}),
  deleteStudy: (id) => del(`/studies/${id}`),
  runStudy: (id) => post(`/studies/${id}/run`, {}),
  estimateStudy: (id) => post(`/studies/${id}/estimate`, {}),
  results: (id) => get(`/studies/${id}/results`),
  streamUrl: (id) => `${BASE}/studies/${id}/stream`,   // for EventSource

  ratingSheet: (id, n = 40) => get(`/studies/${id}/rating-sheet?n=${n}`),
  submitHumanRatings: (id, b) => post(`/studies/${id}/human-ratings`, b),
  studyRaters: (id) => get(`/studies/${id}/raters`),
  reliability: (id) => get(`/studies/${id}/reliability`),
};
