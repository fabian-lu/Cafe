# Builds the public site for cafe-ai.de: the landing page at / and the read-only demo at /demo,
# served by a single nginx. Multi-stage so the final image is just nginx + static files.
#
# Build context is the repo root (see deploy/docker-compose.yml).

# ── build the landing page ───────────────────────────────────────────────────
FROM node:20-alpine AS landing
WORKDIR /landing
COPY apps/landing/package.json apps/landing/package-lock.json* ./
RUN npm install
COPY apps/landing/ ./
RUN npm run build

# ── build the read-only demo (the web-app frontend, VITE_DEMO=1 → static, no backend) ──
FROM node:20-alpine AS demo
WORKDIR /app
COPY apps/web-app/frontend/package.json ./
RUN npm install
COPY apps/web-app/frontend/ ./
RUN VITE_DEMO=1 npm run build

# ── serve ────────────────────────────────────────────────────────────────────
FROM nginx:alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=landing /landing/dist /usr/share/nginx/html
COPY --from=demo /app/dist /usr/share/nginx/html/demo
