# Mug Mockup Service

Backend service + Admin UI for generating realistic mug mockups from template assets and design images.

This README is aligned with the current source code in this repository (FastAPI routers under `app/routers` and Admin UI under `admin-ui`).

## 1. Tech Stack

Backend:
- Python 3.11
- FastAPI + Uvicorn
- SQLAlchemy (async) + asyncpg
- Alembic migrations
- OpenCV + NumPy + SciPy + scikit-image + Pillow
- Redis (optional, currently mainly infra dependency)
- httpx (async download pooling for URL analysis import flow)

Frontend:
- React 19
- TypeScript
- Vite

Infra:
- Docker / Docker Compose
- PostgreSQL 16

## 2. Repository Layout

- `app/`: FastAPI application (routers, pipeline, services, DB)
- `admin-ui/`: React/Vite admin frontend
- `templates/`: template assets storage
- `inputs/`: uploaded bases and artworks
- `public/mockups/`: static mockup files
- `output/renders/`: rendered outputs
- `docs/`: project docs
- `tests/`: Python unit tests

## 3. Environment Variables

Use `.env.example` as baseline.

Important variables:
- `DATABASE_URL`: async SQLAlchemy URL, example:
  - `postgresql+asyncpg://postgres:postgres@localhost:5432/mockup_service`
- `ASSET_BASE_DIR`: template root folder, default `./templates`
- `REDIS_URL`: redis URL
- `HOST`: server host
- `PORT`: server port
- `PRINTERVAL_SECTION_URL_TEMPLATES`: optional JSON map for original design source URL templates
- `RENDER_DEVICE`: `cpu` or `gpu` (used in Docker compose)
- `URL_ANALYSIS_PREVIEW_WIDTH`, `URL_ANALYSIS_PREVIEW_HEIGHT`, `URL_ANALYSIS_PREVIEW_QUALITY`, `URL_ANALYSIS_PREVIEW_FIT`
- `URL_ANALYSIS_RENDER_QUALITY`

## 4. Run with Docker (Recommended)

From repository root:

```bash
cp .env.example .env
docker compose up -d --build
```

Service URLs:
- API docs: `http://localhost:8040/docs`
- OpenAPI JSON: `http://localhost:8040/openapi.json`
- Health: `http://localhost:8040/health`

Docker compose services:
- `api` (FastAPI, exposed on `8040 -> 8000`)
- `postgres`
- `redis`

## 5. Run Backend Locally (without Docker)

1) Create virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Prepare database and run migrations:

```bash
alembic upgrade head
```

3) Start API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8040 --reload
```

4) Optional seed:

```bash
python seed.py
```

## 6. Run Admin UI Locally

From `admin-ui`:

```bash
npm install
npm run dev
```

Default API target in `admin-ui/vite.config.ts`:
- `http://127.0.0.1:8040`

Proxy paths:
- `/v1`
- `/admin`
- `/static`

## 7. Authentication Status

Current mode:
- No auth is required for API calls.
- Do not send `Authorization` header.
- Call APIs directly (plain request).

Production note:
- If you later enable auth again, update this README and frontend accordingly.

## 8. API Overview (What each API is for)

Base URL examples:
- Docker: `http://localhost:8040`
- Local direct run: use your configured `PORT`

### 8.1 Health and Static

- `GET /health`: check service status, render device, GPU active state.
- `GET /static/templates/...`: read template files.
- `GET /static/bases/...`: read base images.
- `GET /static/artworks/...`: read downloaded/uploaded artworks.
- `GET /static/mockups/...`: read public mockup files.
- `GET /static/renders/...`: read rendered outputs.

### 8.2 Render APIs (`/v1`)

- `POST /v1/mockup/render`: render from an existing saved template (`template_id`) and a design.
- `POST /v1/mockup/render-preview`: fast template preview render (auto preview mode, JPEG).
- `POST /v1/mockup/render-adhoc`: render without saved template; upload/provide mockup + design directly.
- `POST /v1/mockup/render-async-adhoc`: queue async ad-hoc render job.
- `GET /v1/mockup/render-status/{job_id}`: get async render progress/result URL.
- `POST /v1/mockup/warp-preview`: preview geometric warp using config only.
- `POST /v1/mockup/warp-preview-file`: warp preview with uploaded design.
- `POST /v1/mockup/warp-preview-adhoc`: warp preview using uploaded mockup/design.
- `POST /v1/mockup/warp-prefetch`: warm cache for warp maps.
- `POST /v1/mockup/detect-region`: auto-detect printable region on uploaded mockup.
- `POST /v1/mockup/bake-normal`: generate/bake normal map from mockup.
- `GET /v1/templates`: list active templates for render UI.
- `GET /v1/render/device`: get current render device mode.
- `POST /v1/render/device`: switch render device mode.

Notes:
- Render output is forced to JPEG for lightweight responses.
- `config_json` overrides are supported on render endpoints where defined.
- `POST /v1/mockup/render` supports `is_preview=true` (and optional `preview_max_dim`, default `512`) to speed up template mode by downscaling render assets first.
- `design_url` and `mockup_url` can be either local static path (for example `/static/artworks/...`) or remote HTTP/HTTPS URL.
- Remote URL inputs are auto-downloaded and cached on disk (`inputs/artworks` for design, `inputs/bases` for mockup), so repeated calls are much faster.

### 8.3 Resolve APIs (`/v1`)

- `GET /v1/mockup/resolve`: map source CDN URL -> previously rendered URL (if exists).
- `POST /v1/mockup/resolve-batch`: resolve multiple source URLs in one call.

### 8.4 Admin APIs (`/admin`)

Template management:
- `GET /admin/templates/{slug}/preview-image`: return resized/cropped JPEG preview for template.
- `POST /admin/templates`: create template by uploading mockup/mask/maps.
- `PUT /admin/templates/{slug}/config`: update template config.
- `POST /admin/templates/{slug}/publish`: mark template active.
- `POST /admin/templates/{slug}/archive`: archive template.
- `POST /admin/templates/{slug}/rollback/{history_id}`: rollback config by history id.
- `GET /admin/templates/{slug}/config-history`: list config history.
- `POST /admin/templates/{slug}/generate-maps`: regenerate maps from mockup.
- `POST /admin/templates/save-adhoc`: create template from live preview calibration payload.

Library:
- `GET /admin/library/{lib_type}`: list files in `bases` or `artworks`.
- `POST /admin/library/{lib_type}/upload`: upload file to library.

URL analysis:
- `POST /admin/url-analysis/import`: analyze a product URL and prepare import data fast.
- `POST /admin/url-analysis/template-matches`: find templates matching URL metadata.
- `GET /admin/url-analysis/design-link`: return only design image link (plain text).
- `GET /admin/url-analysis/mockup-link`: return only final mockup image link (plain text).
- `GET /admin/url-analysis/template-preview-link`: alias of mockup-link behavior.

URL analysis behavior highlights:
- JSONB lookup by `design_lookup_key` / `mockup_family_key`.
- Skip re-download when local artwork already exists.
- Async download path with connection pooling.
- `mockup-link` and `template-preview-link` return plain text URL only.

### 8.5 Quick API Examples (No token needed)

Health:

```bash
curl "http://localhost:8040/health"
```

List active templates:

```bash
curl "http://localhost:8040/v1/templates"
```

Render from template:

```bash
curl -X POST "http://localhost:8040/v1/mockup/render" \
  -F "template_id=mug01" \
  -F "design_url=/static/artworks/sample.png"
```

Render directly from remote design URL (single call, auto-cache on backend):

```bash
curl -X POST "http://localhost:8040/v1/mockup/render" \
  -F "template_id=mug01" \
  -F "design_url=https://cdn.printerval.com/image/mugs-11oz,black,print-seller-2025-09-13_funny-baby-huey-cartoon-classic-t-shirt-1-054bcb6a0659c9e13a5f0fd3bfc7c525,2d2d2d.jpeg"
```

Render from template in fast preview mode:

```bash
curl -X POST "http://localhost:8040/v1/mockup/render" \
  -F "template_id=mug01" \
  -F "design_url=/static/artworks/sample.png" \
  -F "is_preview=true" \
  -F "preview_max_dim=512"
```

Render from template via dedicated preview API:

```bash
curl -X POST "http://localhost:8040/v1/mockup/render-preview" \
  -F "template_id=mug01" \
  -F "design_url=/static/artworks/sample.png"
```

Import from source URL:

```bash
curl -X POST "http://localhost:8040/admin/url-analysis/import" \
  -H "Content-Type: application/json" \
  -d "{\"source_url\":\"https://cdn.printerval.com/image/mugs-11oz,black,print-seller-2025-09-13_funny-baby-huey-cartoon-classic-t-shirt-1-054bcb6a0659c9e13a5f0fd3bfc7c525,2d2d2d.jpeg\"}"
```

Find matching templates from URL:

```bash
curl -X POST "http://localhost:8040/admin/url-analysis/template-matches" \
  -H "Content-Type: application/json" \
  -d "{\"source_url\":\"https://cdn.printerval.com/image/mugs-11oz,black,print-seller-2025-09-13_funny-baby-huey-cartoon-classic-t-shirt-1-054bcb6a0659c9e13a5f0fd3bfc7c525,2d2d2d.jpeg\"}"
```

Get design link only:

```bash
curl "http://localhost:8040/admin/url-analysis/design-link?url=https://cdn.printerval.com/image/960x960/mugs-15oz,white,print-seller-2025-09-13_funny-baby-huey-cartoon-classic-t-shirt-1-054bcb6a0659c9e13a5f0fd3bfc7c525,ffffff.jpeg"
```

Get final mockup link only:

```bash
curl "http://localhost:8040/admin/url-analysis/mockup-link?url=https://cdn.printerval.com/image/960x960/mugs-15oz,white,print-seller-2025-09-13_funny-baby-huey-cartoon-classic-t-shirt-1-054bcb6a0659c9e13a5f0fd3bfc7c525,ffffff.jpeg"
```

Resolve existing rendered result:

```bash
curl "http://localhost:8040/v1/mockup/resolve?url=https://cdn.printerval.com/image/mugs-11oz,black,print-seller-2025-09-13_funny-baby-huey-cartoon-classic-t-shirt-1-054bcb6a0659c9e13a5f0fd3bfc7c525,2d2d2d.jpeg"
```

## 9. Typical Workflows

### 9.1 Template-based render

1) Create/publish template via admin APIs.
2) Render with:
   - `POST /v1/mockup/render` using `template_id` + `design_image` or `design_url`.
   - For quick UI preview: add `is_preview=true` or call `POST /v1/mockup/render-preview`.

### 9.2 URL analysis import

1) Call `POST /admin/url-analysis/import` with source URL.
2) If template already matched and local artwork exists, returns quickly.
3) If not matched, ingests artwork, returns presets/meta for editor flow.

### 9.3 Resolve existing render

1) Call `GET /v1/mockup/resolve?url=...`
2) Receive cached/rendered image URL if found.

## 10. Testing

Run tests (if your environment includes `pytest`):

```bash
pytest tests -q
```

Key test areas in `tests/`:
- URL analysis
- design transform
- cylindrical squeeze and mug curve correction
- mesh warp and continuity
- cache behavior

## 11. Troubleshooting

Common checks:
- Ensure PostgreSQL is running and `DATABASE_URL` is valid.
- Run `alembic upgrade head` before starting API.
- Verify required folders exist: `templates`, `inputs`, `public/mockups`, `output/renders`.
- If Admin UI cannot call API, confirm Vite proxy target is correct.
- If URL import is slow on first call, that is expected on cache miss; repeated calls should be much faster.

If remote render/download reports missing `h2` (HTTP/2):

```bash
pip install "httpx[http2]"
```

After install, restart API to apply:

```bash
# local
uvicorn app.main:app --host 0.0.0.0 --port 8040 --reload

# docker
docker compose restart api
```

## 12. Version and Docs

- API app version in code: `1.2.0` (`app/main.py`)
- Live API docs: `/docs`
- Additional project docs: `docs/` and architecture markdown files in repo root.

# Update commit. Trigger build
