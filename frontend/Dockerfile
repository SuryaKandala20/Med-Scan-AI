# ── Stage 1: Build ──────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --legacy-peer-deps
COPY . .
RUN npm run build

# ── Stage 2: Serve with Nginx ────────────────────────────────────────
FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html

# Proxy /api/* to backend so React routing still works
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
