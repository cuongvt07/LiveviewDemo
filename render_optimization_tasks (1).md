# Kế hoạch Tối ưu Render Pipeline & Vá Lỗi

> **Mục tiêu:** Giảm thời gian render ad-hoc từ `5602ms` xuống dưới `800ms`; giảm độ trễ Preview/warp-preview khi tinh chỉnh xuống dưới `500ms` — không ảnh hưởng chất lượng output.

---

## Task 0.1 — Mesh Warp Resolution (Ưu tiên: Khẩn cấp | Ước tính: ~1 ngày)

> `smooth_mesh_warp` đang tiêu tốn 3–5 giây ngay cả khi warp param không hoạt động (`center_focus_width=0`). Nguyên nhân nhiều khả năng là đang tính mesh ở độ phân giải Production (1500×1500) thay vì kích thước Preview thực tế.

### Bối cảnh từ log

```
[PERF]   smooth_mesh_warp: 3062ms   ← center_focus_width=0 (inactive)
[PERF]   smooth_mesh_warp: 5244ms   ← center_focus_width=-0.2 (active)
[PERF]   cylindrical_warp: 91ms     ← nhanh, đã ổn
[PERF]   composite: 36ms            ← nhanh, đã ổn
```

### Checklist

- [ ] Xác định độ phân giải ảnh đầu vào của `smooth_mesh_warp` trong luồng Preview:

  ```python
  logger.info(f"[DEBUG] smooth_mesh_warp input shape: {img.shape}")
  ```

  Nếu shape là `(1500, 1500, ...)` → đây chính là nguyên nhân. Preview không cần hơn `512×512`.

- [ ] Thêm bước downscale **trước khi vào** `smooth_mesh_warp` trong luồng Preview:

  ```python
  PREVIEW_SCALE = 512 / max(img.shape[:2])  # giữ aspect ratio

  def run_preview_pipeline(img, warp_params, ...):
      h, w = img.shape[:2]
      small = cv2.resize(img, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE,
                         interpolation=cv2.INTER_AREA)

      # Tính displacement field trên ảnh nhỏ
      warped_small = smooth_mesh_warp(small, warp_params, ...)

      # Upscale kết quả về kích thước gốc nếu cần trả về full-res
      warped_full = cv2.resize(warped_small, (w, h),
                               interpolation=cv2.INTER_LINEAR)
      return warped_full
  ```

- [ ] Nếu `smooth_mesh_warp` dùng thin-plate spline: kiểm tra số control point — giảm từ N×N dense grid xuống sparse grid (ví dụ 16×16 → 8×8) cho Preview
- [ ] Kiểm tra xem có bước nào bên trong `smooth_mesh_warp` đang dùng NumPy loop thuần (`for row in img`) hay không — thay bằng `cv2.remap` nếu có:

  ```python
  # Nếu đã có displacement map (map_x, map_y), dùng cv2.remap thay loop
  result = cv2.remap(img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)
  ```

- [ ] QA: so sánh Preview output trước/sau bằng mắt (Preview không cần SSIM nghiêm ngặt như Production)
- [ ] Đo và ghi log: `[PERF] smooth_mesh_warp: Xms (input=HxW)`

**Kết quả kỳ vọng:** `smooth_mesh_warp: 3062ms → ~200–400ms`

---

## Task 0.2 — Warp Cache Strategy (Ưu tiên: Khẩn cấp | Ước tính: ~1 ngày)

> Mỗi lần user kéo slider thay đổi `center_focus_width`, `edge_squeeze`,... → cache key thay đổi → CACHE MISS → tính lại toàn bộ warp map (2–10 giây). Đây là nguyên nhân gây cảm giác "đơ" khi tinh chỉnh trực tiếp trên UI.

### Bối cảnh từ log

```
center_focus_width=0    → warp map: CACHE HIT  (0ms)
center_focus_width=-0.2 → warp map: CACHE MISS (9793ms)  ← người dùng kéo slider 1 lần
center_focus_width=-0.2 → warp map: CACHE MISS (1953ms)  ← adhoc cùng params, vẫn miss
```

### Giải pháp A — Quantize params trước khi hash (nhanh, ít rủi ro)

Làm tròn tất cả warp params về bước đủ thô trước khi tạo cache key. Mắt người không phân biệt được `0.18` vs `0.20`, nhưng cache sẽ HIT thay vì MISS.

- [ ] Xác định tất cả param tham gia vào cache key hiện tại (xem hàm tạo key trong warp cache module)
- [ ] Tạo hàm `quantize_warp_params()` chuẩn hóa params trước khi hash:

  ```python
  import math

  def quantize_warp_params(params: dict, step: float = 0.05) -> dict:
      """Làm tròn float params về lưới step để tăng cache hit rate."""
      return {
          k: round(round(v / step) * step, 6) if isinstance(v, float) else v
          for k, v in params.items()
      }

  # Ví dụ:
  # center_focus_width=0.18  → 0.20
  # edge_squeeze=0.032       → 0.05
  # theta_max_deg=52.0       → 50.0  (nếu step=5.0 cho góc)
  ```

- [ ] Áp dụng quantize **trước** khi gọi cache lookup — không thay đổi params truyền vào hàm tính toán thực:

  ```python
  def get_warp_map(params: dict, img_shape: tuple):
      cache_key = make_cache_key(quantize_warp_params(params), img_shape)

      if cache_key in warp_cache:
          return warp_cache[cache_key]  # HIT

      result = compute_warp_map(params, img_shape)  # params gốc, không quantize
      warp_cache[cache_key] = result
      return result
  ```

- [ ] Chọn `step` phù hợp cho từng param — test thực tế trên UI:
  - `center_focus_width`, `edge_squeeze`: step `0.05` (sai số tối đa ~0.025, không nhìn thấy)
  - `theta_max_deg`: step `2.0` (độ)
  - `curve_top`, `curve_bottom`, `pitch`: step `5.0` (px)
- [ ] Log cache hit rate sau deploy: `logger.info(f"warp cache: {len(warp_cache)} entries, hit_rate={hits/(hits+misses):.1%}")`

### Giải pháp B — Background precompute khi slider đang kéo (phức tạp hơn, UX tốt hơn)

Khi user kéo slider → debounce 150ms → trigger tính warp map ngầm trong background → kết quả sẵn sàng trước khi user thả slider.

- [ ] Wrap `compute_warp_map` thành background task (FastAPI BackgroundTasks hoặc asyncio):

  ```python
  import asyncio
  from collections import OrderedDict

  _warp_precompute_cache = OrderedDict()  # LRU manual

  async def precompute_warp_map_bg(params: dict, img_shape: tuple):
      """Gọi từ endpoint preview khi nhận slider event, không await."""
      cache_key = make_cache_key(quantize_warp_params(params), img_shape)
      if cache_key not in _warp_precompute_cache:
          loop = asyncio.get_event_loop()
          result = await loop.run_in_executor(None, compute_warp_map, params, img_shape)
          _warp_precompute_cache[cache_key] = result
          if len(_warp_precompute_cache) > 32:
              _warp_precompute_cache.popitem(last=False)  # evict oldest
  ```

- [ ] Thêm endpoint `/v1/mockup/warp-prefetch` nhận params từ frontend khi slider đang kéo (fire-and-forget)
- [ ] Frontend: gọi `/warp-prefetch` mỗi `debounce(150ms)` khi slider thay đổi, không cần đợi response
- [ ] Điều kiện để làm Task B: Task A đã deploy và ổn định

**Kết quả kỳ vọng (Task A):** CACHE MISS đầu tiên vẫn mất ~2–10s, nhưng sau khi user tinh chỉnh lần 2 trong cùng vùng params → HIT ngay.

**Kết quả kỳ vọng (Task A + B):** Từ lần tinh chỉnh thứ 2 trở đi → dưới `100ms` vì warp map đã precompute xong trước khi user thả slider.

---

## Task 1 — Gamma LUT (Ưu tiên: Cao | Ước tính: ~4 giờ)

> Thay phép tính lũy thừa float `** 2.2` bằng Lookup Table — không thay đổi bất kỳ logic nào.

### Quyết định thiết kế (đã xác nhận)

- **`to_linear`**: LUT 256 phần tử, dtype `float32`, tra bằng NumPy fancy indexing.
  `cv2.LUT` **không dùng được** ở đây vì nó chỉ hỗ trợ `uint8 → uint8`, không ra `float32`.
- **`to_srgb`**: LUT 65536 phần tử (Float Quantization `×65535 → uint16`), tra bằng `np.take()`.
  Giữ nguyên độ chính xác float32, nhanh xấp xỉ cv2.LUT.
- **Output `to_linear` vẫn là `float32 [0.0, 1.0]`** — không thay đổi contract với downstream.

### Checklist

- [ ] Mở `app/pipeline/shared/colorspace.py`
- [ ] Thêm LUT vào module-level (tính 1 lần khi import, không tính lại mỗi request):

  ```python
  import numpy as np

  # LUT cho to_linear: uint8 → float32 [0.0, 1.0]
  # KHÔNG dùng cv2.LUT vì output phải là float32, cv2.LUT chỉ hỗ trợ uint8 output
  _LUT_LINEAR_FLOAT32 = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)

  # LUT cho to_srgb: float32 [0.0, 1.0] → uint8 [0, 255]
  # Kỹ thuật Float Quantization: scale float → uint16 để tra 65536-entry LUT
  _LUT_SRGB_UINT8 = ((np.arange(65536) / 65535.0) ** (1.0 / 2.2) * 255.0).astype(np.uint8)
  ```

- [ ] Thay thân hàm `to_linear()`:

  ```python
  # TRƯỚC
  def to_linear(img: np.ndarray) -> np.ndarray:
      return (img / 255.0) ** 2.2  # chậm: float exponentiation trên 2.25M pixel

  # SAU — NumPy fancy indexing, output giữ đúng float32 [0.0, 1.0]
  def to_linear(img: np.ndarray) -> np.ndarray:
      return _LUT_LINEAR_FLOAT32[img]  # img phải là uint8
  ```

- [ ] Thay thân hàm `to_srgb()`:

  ```python
  # TRƯỚC
  def to_srgb(img: np.ndarray) -> np.ndarray:
      return (img ** (1.0 / 2.2) * 255.0).astype(np.uint8)  # img là float32 [0.0, 1.0]

  # SAU — Float Quantization + np.take LUT 65536 phần tử
  def to_srgb(img: np.ndarray) -> np.ndarray:
      idx = (img * 65535.0).astype(np.uint16)   # scale float → uint16 index
      return np.take(_LUT_SRGB_UINT8, idx)       # tra LUT, output uint8
  ```

- [ ] Viết unit test so sánh giá trị pixel trước/sau (tolerance `≤ 1` trên thang 0–255)
- [ ] Kiểm tra edge case: `img = 0.0`, `img = 1.0`, `img` có giá trị âm do float precision
- [ ] Đo thời gian trước/sau bằng `time.perf_counter()`, ghi vào log
- [ ] Deploy — **không cần feature flag**

**Kết quả kỳ vọng:** `~180ms → ~2–3ms` mỗi lần gọi (×2 = tiết kiệm ~350ms/request)

---

## Task 2 — Fix lỗi URL Parser / Persist (Ưu tiên: Cao | Ước tính: ~2 giờ)

> Lỗi `ValueError` xảy ra mỗi request ad-hoc, khiến cache DB không được ghi.

### Checklist

- [ ] Mở file `app/services/render_persist.py`
- [ ] Tìm hàm `extractSlug()` (hoặc nơi gọi `parse_printerval_liveview_url`)
- [ ] Thêm guard cho local/relative path **và** None/empty:

  ```python
  import hashlib

  def extractSlug(source_url: str) -> str:
      # Guard: None hoặc empty string
      if not source_url:
          return "adhoc_no_url"

      # Guard: Local / relative path (không có protocol)
      if not source_url.startswith(('http://', 'https://')):
          return "adhoc_" + hashlib.md5(source_url.encode('utf-8')).hexdigest()[:12]

      parsed = parse_printerval_liveview_url(source_url)
      return parsed.slug
  ```

- [ ] Viết test case cho 3 trường hợp: URL hợp lệ, relative path, None
- [ ] Kiểm tra log sau deploy — `ValueError` không còn xuất hiện
- [ ] Xác nhận record cache DB được ghi đúng cho request ad-hoc

**Kết quả kỳ vọng:** Không còn exception, persist hoạt động bình thường.

---

## Task 3 — Light Map Downsampling (Ưu tiên: Cao | Ước tính: ~1–2 ngày)

> Chạy pipeline mô phỏng ánh sáng ở `375×375` thay vì `1500×1500`, upscale kết quả về full-res bằng cubic interpolation.

### Quyết định thiết kế (đã xác nhận)

**`extract_masked_highlight_detail` chạy ở full-res — KHÔNG downscale.** Lý do:
- Hàm này dùng Thresholding + Morphology trên `mockup_bgr` thật để tách vùng **hard highlight** (tần số cao, edge sắc nét).
- Morphology kernel có kích thước pixel tuyệt đối → behavior thay đổi hoàn toàn ở độ phân giải khác.
- Upscale mask chi tiết sẽ gây "halo" / viền nhoè trên output cuối.

**Ranh giới scale 0.25 áp dụng đúng cho:**

| Hàm | Scale |
|---|---|
| `build_cylinder_surface_maps` | **0.25** — tần số thấp, safe to downscale |
| `light_field` / `diffuse` / `specular_geometry` | **0.25** — tần số thấp, safe to downscale |
| `extract_masked_lighting` | **Full-res** — mask, cần edge chính xác |
| `extract_masked_highlight_detail` | **Full-res** — hard highlight, cần edge chính xác |
| `cylindrical_warp` | **Full-res** — warp geometry |
| `apply_shadow_overlay` / `apply_mug_lighting` | **Full-res** — composite cuối |

### Checklist

- [ ] Tạo hàm `_compute_light_maps(base_img, ...)` gom toàn bộ nhánh ánh sáng:

  ```python
  import cv2

  LIGHT_SCALE = 0.25  # 1500 → 375

  def _compute_light_maps(base_img, theta_max, ...):
      h, w = base_img.shape[:2]

      # Downscale — INTER_AREA tốt nhất khi shrink
      small = cv2.resize(base_img, None, fx=LIGHT_SCALE, fy=LIGHT_SCALE,
                         interpolation=cv2.INTER_AREA)

      # Toàn bộ pipeline ánh sáng chạy trên small (375×375)
      surface_u, surface_v, normals = build_cylinder_surface_maps(small, theta_max)
      highlight_small = compute_light_field(small, normals, ...)  # diffuse + specular

      # Upscale kết quả về full-res — INTER_CUBIC cho smooth gradient
      highlight_full = cv2.resize(highlight_small, (w, h),
                                  interpolation=cv2.INTER_CUBIC)
      return highlight_full
  ```

- [ ] `extract_masked_highlight_detail` **vẫn nhận `base_img` gốc (full-res)** — không truyền `small` vào
- [ ] `extract_masked_lighting` **vẫn nhận `base_img` gốc (full-res)**
- [ ] QA chất lượng — so sánh output bằng mắt và SSIM score (`≥ 0.97` là đạt):

  ```python
  from skimage.metrics import structural_similarity as ssim
  score = ssim(output_before, output_after, channel_axis=2)
  assert score >= 0.97, f"SSIM quá thấp: {score:.4f}"
  ```

- [ ] Test trên ít nhất 5 mockup đa dạng: cốc trắng, màu đậm, in đậm, góc sáng khác nhau
- [ ] Đo và ghi benchmark từng bước vào log

**Kết quả kỳ vọng:** `build_cylinder ~2172ms → ~130ms`, `light_field ~2780ms → ~170ms`

---

## Task 4 — Geometry Cache cho Ad-hoc (Ưu tiên: Trung bình | Ước tính: ~1 ngày)

> Cache `surface_u`, `surface_v`, `normals` theo key `(small_width, small_height, theta_max)` — dùng `lru_cache` RAM, đủ dùng cho single-process, an toàn hơn Redis.

### Quyết định thiết kế (đã xác nhận)

Dùng `functools.lru_cache` thay Redis vì:
- Pipeline chạy single-process (worker per request, không share state across workers cần thiết)
- Không cần serialize/deserialize NumPy array qua network
- Key cache sau Task 3 chỉ phụ thuộc vào `(small_w, small_h, theta_max)` — không cần `mockup_id` vì geometry cylinder là tham số hình học thuần túy, không phụ thuộc nội dung ảnh

### Checklist

- [ ] Xác nhận `build_cylinder_surface_maps` **không** dùng nội dung pixel của `small_img` để tính geometry (chỉ dùng `shape` và `theta_max`)
- [ ] Tách phần tính geometry ra hàm riêng và wrap `lru_cache`:

  ```python
  from functools import lru_cache

  @lru_cache(maxsize=64)  # 64 combination kích thước × góc là đủ
  def _build_geometry_cached(small_h: int, small_w: int, theta_max: float):
      """Cache key = (height, width, theta_max) — thuần tham số hình học."""
      dummy = np.zeros((small_h, small_w, 3), dtype=np.uint8)
      return build_cylinder_surface_maps(dummy, theta_max)
      # trả về: (surface_u, surface_v, normals)
  ```

- [ ] Gọi từ `_compute_light_maps`:

  ```python
  def _compute_light_maps(base_img, theta_max, ...):
      small = cv2.resize(base_img, None, fx=LIGHT_SCALE, fy=LIGHT_SCALE,
                         interpolation=cv2.INTER_AREA)
      sh, sw = small.shape[:2]

      # Cache hit nếu cùng kích thước và theta_max
      surface_u, surface_v, normals = _build_geometry_cached(sh, sw, round(theta_max, 4))

      highlight_small = compute_light_field(small, normals, ...)
      ...
  ```

- [ ] Thêm log để xác nhận cache hit: `logger.debug(f"geometry cache: {_build_geometry_cached.cache_info()}")`
- [ ] Đặt `maxsize` phù hợp — nếu service phục vụ nhiều kích thước mockup, tăng lên 128
- [ ] Kiểm tra memory usage: mỗi entry ~375×375×3×float32 ≈ ~1.6MB × 64 = ~100MB tối đa

**Kết quả kỳ vọng:** `~2172ms → ~10ms` kể từ lần thứ 2 cùng thông số.

---

## Task 5 — Parallel Execution (Ưu tiên: Thấp | Ước tính: ~4 giờ)

> Chạy `extract_masked_lighting` song song với `build_cylinder_surface_maps` vì hai bước này độc lập nhau.

### Điều kiện tiên quyết

- Task 3 và Task 4 đã hoàn thành và ổn định
- `extract_masked_lighting` đã được xác nhận là thread-safe (không modify state ngoài)

### Checklist

- [ ] Xác nhận `extract_masked_lighting` không có side effect (không ghi file, không dùng global mutable state)
- [ ] Refactor pipeline để tách 2 nhánh độc lập:

  ```python
  from concurrent.futures import ThreadPoolExecutor

  def render_adhoc_pipeline(base_img, ...):
      with ThreadPoolExecutor(max_workers=2) as executor:
          # Nhánh 1: Geometry + Light (đã tối ưu ở Task 3 & 4)
          fut_light = executor.submit(compute_light_maps, base_img, ...)

          # Nhánh 2: Mask extraction (chạy song song)
          fut_mask  = executor.submit(extract_masked_lighting, base_img, ...)

          highlight = fut_light.result()
          mask      = fut_mask.result()

      # Composite kết quả
      return composite(base_img, highlight, mask, ...)
  ```

- [ ] Load test với 10 concurrent requests để kiểm tra GIL contention (NumPy releases GIL khi chạy C extensions)
- [ ] Đo tổng thời gian end-to-end sau khi tích hợp

**Kết quả kỳ vọng:** Tiết kiệm thêm `~400–450ms` nhờ overlap 2 nhánh.

---

## Tổng kết thời gian kỳ vọng

### Luồng Preview / warp-preview (Task 0)

| Task | Hiện tại | Sau tối ưu | Tiết kiệm |
|---|---|---|---|
| smooth_mesh_warp | 3062–5244ms | ~200–400ms | ~2800–4800ms |
| warp map CACHE MISS (lần đầu) | 2000–9793ms | 2000–9793ms | 0ms (không tránh được) |
| warp map CACHE MISS (lần 2+, sau quantize) | 2000–9793ms | ~0ms (HIT) | ~2000–9793ms |
| **Tổng Preview (lần 2+ tinh chỉnh)** | **~15s** | **~500ms** | **~14.5s** |

### Luồng Full Render ad-hoc (Task 1–5)

| Task | Hiện tại | Sau tối ưu | Tiết kiệm |
|---|---|---|---|
| Gamma LUT (to_linear + to_srgb) | ~360ms | ~4ms | ~356ms |
| build_cylinder_surface_maps (cache miss) | 2172ms | ~130ms | ~2040ms |
| build_cylinder_surface_maps (cache hit) | 2172ms | ~10ms | ~2162ms |
| light_field + diffuse + specular | 2780ms | ~170ms | ~2610ms |
| extract_masked_lighting | 500ms | 500ms | 0ms |
| **Tổng full render (worst case)** | **5602ms** | **~804ms** | **~4798ms** |
| **Tổng full render (cache hit)** | **5602ms** | **~684ms** | **~4918ms** |

> **Thứ tự thực hiện đề xuất:** Task 0.1 → Task 0.2A → Task 2 → Task 1 → Task 3 → Task 4 → Task 0.2B → Task 5.
> Task 0 giải quyết UX tức thì (người dùng cảm thấy ngay); Task 1–5 giải quyết render quality và throughput.
