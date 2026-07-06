# Deploying CAFE (landing + demo)

Serves the **landing page** at `/` and the **read-only demo** at `/demo` on `cafe-ai.de`, behind the
host's shared nginx proxy (the [`ionos_christoph_server`](https://github.com/fabian-lu/ionos_christoph_server)
repo). Static only — no backend, database, or API keys on the server.

## Prerequisites

- Docker installed on the server, and the proxy stack running (creates the `proxy-net` network).
- DNS `A` records for `cafe-ai.de` and `www.cafe-ai.de` → the server IP.

## Deploy

```bash
git clone https://github.com/fabian-lu/Cafe.git
cd Cafe/deploy
docker compose up -d --build
```

The build compiles the landing page and the demo (`VITE_DEMO=1`) and serves both from one nginx
container. The proxy issues a Let's Encrypt certificate within a minute; then **https://cafe-ai.de** is
live.

## Update after new commits

```bash
cd Cafe && git pull
cd deploy && docker compose up -d --build
```

(Phase 5 automates this from CI on push to `main`.)

## Notes

- The demo data is a frozen snapshot baked into the build (`apps/web-app/frontend/public/demo-data/`);
  it does not change unless regenerated and committed.
- Local preview of the production image: `docker compose up --build` then browse
  `http://localhost` (map a port first, or hit the container directly).
