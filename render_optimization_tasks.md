# Kế hoạch Tối ưu Render Pipeline & Vá Lỗi

> **Mục tiêu:** Giảm thời gian render ad-hoc từ `5602ms` xuống dưới `800ms` mà không ảnh hưởng chất lượng output.

---

## Task 1 — Gamma LUT (Ưu tiên: Cao | Ước tính: ~4 giờ)

> Thay phép tính lũy thừa float `** 2.2` bằng Lookup Table — không thay đổi bất kỳ logic nào.

### Checklist

- [ ] Tìm tất cả vị trí gọi hàm `to_linear()` và `to_srgb()` trong codebase
- [ ] Thêm LUT vào module-level (chỉ tính 1 lần khi import):

  ```python
  import numpy as np
  import cv2

  _LUT_LINEAR = (np.arange(256) / 255.0) ** 2.2 * 255.0).astype(np.uint8)
  _LUT_SRGB   = (np.arange(256) / 255.0) ** (1/2.2) * 255.0).astype(np.uint8)
  ```

- [ ] Thay thân hàm `to_linear()`:

  ```python
  # TRƯỚC
  def to_linear(img):
      return (img / 255.0) ** 2.2

  # SAU
  def to_linear(img):
      return cv2.LUT(img, _LUT_LINEAR).astype(np.float32) / 255.0
  ```

- [ ] Thay thân hàm `to_srgb()` tương tự
- [ ] Chạy unit test so sánh giá trị pixel đầu ra (tolerance `< 1/255`)
- [ ] Đo thời gian trước/sau bằng `time.perf_counter()`, ghi vào log
- [ ] Deploy & quan sát production log — **không cần feature flag**

**Kết quả kỳ vọng:** `~180ms → ~2ms` mỗi lần gọi (×2 = tiết kiệm ~350ms/request)

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

> Chạy toàn bộ pipeline mô phỏng ánh sáng ở `375×375` thay vì `1500×1500`, upscale kết quả về full-res bằng cubic interpolation.

### Nguyên lý (quan trọng — đọc trước khi code)

Specular highlight, diffuse shading và shadow là **tín hiệu tần số thấp** — không có edge sắc nét. Bicubic upscale từ `375×375` lên `1500×1500` không gây artifact thị giác trên bề mặt cong đồng nhất (cốc, ly). **Bước `extract_masked_lighting` KHÔNG áp dụng downsampling này** vì mask cần edge chính xác.

### Checklist

- [ ] Xác định hàm entry point của pipeline ánh sáng (thường là hàm gọi `build_cylinder_surface_maps` rồi `highlight_map`)
- [ ] Thêm bước downscale trước khi vào pipeline:

  ```python
  LIGHT_SCALE = 0.25  # 1500 -> 375

  def compute_light_maps(base_img, ...):
      small = cv2.resize(base_img, None, fx=LIGHT_SCALE, fy=LIGHT_SCALE,
                         interpolation=cv2.INTER_AREA)

      # Chạy toàn bộ pipeline trên ảnh nhỏ
      normal_map_small  = build_cylinder_surface_maps(small, ...)
      highlight_small   = compute_highlight_map(small, normal_map_small, ...)

      # Upscale kết quả về kích thước gốc
      h, w = base_img.shape[:2]
      highlight_full = cv2.resize(highlight_small, (w, h),
                                  interpolation=cv2.INTER_CUBIC)
      return highlight_full
  ```

- [ ] Giữ nguyên `extract_masked_lighting` — **vẫn chạy trên ảnh full-res**
- [ ] QA chất lượng: so sánh output trước/sau bằng mắt và SSIM score (`>= 0.97` là đạt)
- [ ] Test trên ít nhất 5 mockup khác nhau (cốc trắng, màu đậm, in đậm, góc sáng khác nhau)
- [ ] Đo thời gian và ghi benchmark

**Kết quả kỳ vọng:** `build_cylinder ~2172ms → ~130ms`, `light_field ~2780ms → ~170ms`

---

## Task 4 — Geometry Cache cho Ad-hoc (Ưu tiên: Trung bình | Ước tính: ~1 ngày)

> Cache Normal Map và Shadow Map theo key = `(width, height, camera_angle, mockup_id)`.

### Checklist

- [ ] Xác định tất cả tham số ảnh hưởng đến output của `build_cylinder_surface_maps()`:
  - Kích thước ảnh `(width, height)`
  - Góc camera / góc nghiêng
  - Template/mockup ID hoặc hash của ảnh base (nếu Normal Map phụ thuộc vào ảnh)
- [ ] Tạo cache key:

  ```python
  import hashlib, json

  def make_geometry_cache_key(width, height, camera_angle, mockup_id=None) -> str:
      parts = {"w": width, "h": height, "angle": round(camera_angle, 2)}
      if mockup_id:
          parts["mid"] = mockup_id
      return "geo:" + hashlib.md5(json.dumps(parts, sort_keys=True).encode()).hexdigest()
  ```

- [ ] Wrap hàm `build_cylinder_surface_maps()` với cache (dùng Redis hoặc `functools.lru_cache` nếu single-process):

  ```python
  def build_cylinder_surface_maps_cached(img, camera_angle, mockup_id=None):
      key = make_geometry_cache_key(*img.shape[:2], camera_angle, mockup_id)

      cached = redis_client.get(key)
      if cached:
          return deserialize(cached)  # CACHE HIT

      result = build_cylinder_surface_maps(img, camera_angle)
      redis_client.setex(key, 3600, serialize(result))  # TTL 1 giờ
      return result
  ```

- [ ] Xác nhận log xuất hiện `geometry: CACHE HIT (0ms)` cho request thứ 2 cùng thông số
- [ ] Thiết lập cache eviction policy (TTL hoặc LRU, tùy memory budget)
- [ ] Kiểm tra memory usage sau 100 request đa dạng

**Kết quả kỳ vọng:** `~2172ms → ~10ms` kể từ lần thứ 2 trở đi.

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

| Task | Hiện tại | Sau tối ưu | Tiết kiệm |
|---|---|---|---|
| Gamma LUT (to_linear + to_srgb) | ~360ms | ~4ms | ~356ms |
| build_cylinder_surface_maps (cache miss) | 2172ms | ~130ms | ~2040ms |
| build_cylinder_surface_maps (cache hit) | 2172ms | ~10ms | ~2162ms |
| light_field + diffuse + specular | 2780ms | ~170ms | ~2610ms |
| extract_masked_lighting | 500ms | 500ms | 0ms |
| **Tổng (worst case / cache miss)** | **5602ms** | **~804ms** | **~4798ms** |
| **Tổng (cache hit)** | **5602ms** | **~684ms** | **~4918ms** |

> **Lưu ý chất lượng:** Tất cả thay đổi chỉ tác động pipeline trung gian. Output cuối vẫn ở `1500×1500px` với đầy đủ specular, diffuse, shadow. `extract_masked_lighting` giữ nguyên full-res để đảm bảo độ chính xác mask.
