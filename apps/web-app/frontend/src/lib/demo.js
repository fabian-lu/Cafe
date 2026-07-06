// Read-only demo mode. Enabled at build time with `VITE_DEMO=1 npm run build`.
// In demo mode the app talks to bundled static JSON (a frozen snapshot of one study) instead of a
// backend, and every mutating action is blocked with a small toast — so it can be hosted as pure
// static files with no server, no database, no API keys, and no cost.

export const DEMO = String(import.meta.env.VITE_DEMO || "") === "1"
  || String(import.meta.env.VITE_DEMO || "").toLowerCase() === "true";

// Map an API path (e.g. "/studies/3/results") to its static file under public/demo-data/.
export function demoFile(path) {
  const clean = path.split("?")[0].replace(/^\/+/, "").replace(/\//g, "-");
  return `${import.meta.env.BASE_URL}demo-data/${clean}.json`;
}

let _timer;
export function demoToast(msg = "Read-only demo — changes are disabled here.") {
  if (typeof document === "undefined") return;
  let el = document.getElementById("demo-toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "demo-toast";
    el.className = "demo-toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  // restart the show animation
  el.classList.remove("show");
  void el.offsetWidth; // reflow
  el.classList.add("show");
  clearTimeout(_timer);
  _timer = setTimeout(() => el.classList.remove("show"), 2600);
}
