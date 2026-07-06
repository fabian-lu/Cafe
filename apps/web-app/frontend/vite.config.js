import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api to the backend. Local dev → localhost:8000; in docker-compose the frontend container
// reaches the backend by its service name (BACKEND_URL=http://backend:8000).
const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

// The read-only demo build is served under /demo on the public site.
const base = process.env.VITE_DEMO ? "/demo/" : "/";

export default defineConfig({
  base,
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
    },
  },
});
