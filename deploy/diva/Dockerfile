# The diva.cafe-ai.de web container: the web-app frontend built with the DIVA theme baked in
# (VITE_THEME=diva loads src/themes/diva.css on top of the default tokens), served by nginx which
# also proxies /api to the backend and enforces basic auth. Build context is the repo root.

FROM node:20-alpine AS frontend
WORKDIR /app
COPY apps/web-app/frontend/package.json ./
RUN npm install
COPY apps/web-app/frontend/ ./
RUN VITE_THEME=diva npm run build

FROM nginx:alpine
COPY deploy/diva/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend /app/dist /usr/share/nginx/html
