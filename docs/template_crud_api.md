# Template CRUD API — Spec & Implementation Guide

Mục tiêu: cung cấp tài liệu hướng dẫn để thêm API CRUD cho `Template` (admin-only), bao gồm schema, endpoints, ví dụ curl, lưu DB và test.

**Tóm tắt chức năng**
- `GET /admin/templates` — danh sách template (pagination/filter)
- `GET /admin/templates/{slug}` — lấy chi tiết template
- `POST /admin/templates` — tạo template mới (upload file mockup đã có hoặc tham chiếu url)
- `PUT /admin/templates/{slug}` — cập nhật config/ảnh/metadata
- `DELETE /admin/templates/{slug}` — xóa template (mềm hoặc cứng)

Tất cả endpoint admin phải bảo vệ bằng header `Authorization: Bearer <ADMIN_TOKEN>` (hiện dùng `verifyAdmin` trong `app/routers/admin.py`).

---

## 1. Pydantic schemas (đề xuất)
File: `app/schemas.py` (mở rộng/điều chỉnh)

- `TemplateCreate` (hiện đã tồn tại) — giữ nguyên, cho phép truyền `mockup_url` hoặc upload riêng qua `/library/bases/upload`.
- `TemplateUpdate` (mới):

```py
class TemplateUpdate(BaseModel):
    name: Optional[str]
    description: Optional[str]
    config: Optional[dict]
    output_width: Optional[int]
    output_height: Optional[int]
```

- `TemplateListItem`, `TemplateDetail` hiện có đủ; dùng lại.

---

## 2. Endpoints (router)
Tạo file: `app/routers/templates.py` hoặc thêm vào `app/routers/admin.py`.

1) GET /admin/templates
- Query params: `page=1&page_size=50` (optional)
- Response: list `TemplateListItem` + pagination

2) GET /admin/templates/{slug}
- Path param: `slug`
- Response: `TemplateDetail`

3) POST /admin/templates
- Body: `TemplateCreate` (JSON)
- Flow:
  - validate slug unique (409 nếu trùng)
  - if `mockup_url` is remote: call `resolve_local_asset_path` or copy to `templates/<slug>/mockup.jpg`
  - create `templates/<slug>/maps` dir and optionally call `generate_all_maps` (async thread)
  - save DB record `Template` with `config` (normalized via existing `_normalize_saved_template_config`)
  - return `{ slug, preview_url, message }` (201)

4) PUT /admin/templates/{slug}
- Body: `TemplateUpdate`
- Flow:
  - fetch template by slug
  - update fields, if `config` passed: call `_normalize_saved_template_config` with existing output size
  - commit
  - return updated `TemplateDetail`

5) DELETE /admin/templates/{slug}
- Optional query param `hard=true` to delete files on disk
- Flow:
  - mark record status='deleted' (soft delete) and/or remove files if hard
  - return 204 no-content

All endpoints should call `verifyAdmin` header dependency.

---

## 3. DB model notes
Model `Template` already exists (check `app/db/models.py`). Ensure fields exist:
- `slug`, `name`, `mockup_path`, `mask_path`, `normal_map_path`, `specular_path`, `config` (JSON), `output_width`, `output_height`, `status`.

When updating `config` that contains `print_area.mesh_control_*` ensure pixel coordinates are used (see existing `_merge_adhoc_user_config` / `_normalize_saved_template_config`).

---

## 4. Example curl requests
Replace `ADMIN_TOKEN` với token thực tế (env `ADMIN_TOKEN`).

Create (POST):

```bash
curl -X POST "http://localhost:8040/admin/templates" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "slug":"my-mug-01",
    "name":"My Mug 01",
    "config": { "print_area": { "top_left": [300,250], "top_right": [1200,250], "bottom_right": [1200,1250], "bottom_left": [300,1250] } },
    "mockup_url":"/static/bases/my-mock.jpg",
    "output_width":1500,
    "output_height":1500
  }'
```

Get list:
```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" "http://localhost:8040/admin/templates"
```

Update (PUT):
```bash
curl -X PUT "http://localhost:8040/admin/templates/my-mug-01" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"My Mug 01 v2","config": {"lighting": {"specular_strength":0.2}}}'
```

Delete (soft):
```bash
curl -X DELETE "http://localhost:8040/admin/templates/my-mug-01" -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

## 5. Implementation checklist (task list)
- [ ] Add router `app/routers/templates.py` implementing endpoints above (use `verifyAdmin` dependency)
- [ ] Add `TemplateUpdate` schema to `app/schemas.py`
- [ ] Add DB migrations if needed (depends on ORM setup; current models likely ok)
- [ ] Add tests in `tests/test_templates_crud.py` (create/update/get/delete)
- [ ] Wire admin-ui (optional): add list page and edit form calling these endpoints

---

## 6. Notes about artwork overwrite flow (your requirement)
Your requirement: when importing a CDN URL that matches a template (by design_lookup_key), if the artwork from the URL is not the same as the template's saved artwork, replace template's artwork with the new artwork and return its URL.

Implementation decisions made in repo:
- `importFromAnalyzedUrl` now calls `analyze_and_ingest_url` and if an existing template is found it ingests the new design and updates `template.config['url_analysis']['design_url']`.
- Ensure `templates/<slug>/mockup.jpg` vs `templates/<slug>/artwork` separation: current repo saves artwork under `inputs/artworks` and stores `design_url` as `/static/artworks/<name>` — acceptable. If you want artwork copied under template dir, add copy step after download.

If you prefer replacing `mockup.jpg` in template folder, implement extra file copy step when new artwork is ingested and update `tpl.mockup_path`.

---

## 7. Testing & smoke steps
1. Start backend:
```bash
docker compose up -d --build api
```
2. Create a template via POST or use existing slug.
3. Call import URL endpoint with the CDN URL you provided:
```bash
curl -X POST "http://localhost:8040/url-analysis/import" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"source_url":"<your-url>"}'
```
4. Check DB `templates` table for updated `config.url_analysis.design_url` or file `templates/<slug>/mockup.jpg` if you opted to copy.

---

## 8. Security & ACL
- All admin CRUD endpoints must require `verifyAdmin` header.
- Validate/normalize user-supplied `config` with `_merge_adhoc_user_config` to avoid malformed print_area.

---

## 9. Rollout plan
- Implement endpoints + tests (2–4h)
- Run integration test with the CDN URL and UI flow (1h)
- Monitor logs for ingestion errors and ensure safe fallback if download fails

---

Nếu bạn muốn, tôi có thể tiếp tục và: 
- (A) tạo file `app/routers/templates.py` với mã mẫu, hoặc
- (B) tạo test skeleton `tests/test_templates_crud.py`.

Chọn A hoặc B hoặc cả hai.