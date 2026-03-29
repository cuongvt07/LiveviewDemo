# Mug Technical Algorithms

Tài liệu này tập trung vào phần kỹ thuật cốt lõi của module mug: công nghệ sử dụng, pipeline xử lý, mô hình dữ liệu, thuật toán warp hình trụ, lưới preview và logic compositing ánh sáng. Mục tiêu là mô tả đúng trạng thái code hiện tại, không chỉ ý tưởng.

## 1. Stack công nghệ

### Backend

- Python 3.11
- NumPy cho vectorization và toán ma trận
- OpenCV (`cv2`) cho homography, remap, encode/decode ảnh, Gaussian blur, fill polygon
- FastAPI cho API preview/render
- Pydantic cho request schema
- Docker Compose để chạy runtime đồng nhất

### Frontend

- React + TypeScript
- Vite dev server
- SVG overlay cho lưới và handle chỉnh vùng in
- `fetch` + `FormData` để gọi preview/render API

### Runtime / Acceleration

- OpenCV OpenCL toggle thông qua `gpuRemap` và `/v1/render/device`
- Docker image backend dùng `python:3.11-slim-bookworm`

## 2. Mục tiêu kỹ thuật của module mug

Module mug không dùng TPS như apparel. Bài toán của mug là ánh xạ một artwork 2D phẳng vào một vùng in có:

- phối cảnh ảnh chụp thật
- độ cong ngang do thân cốc là khối trụ
- độ cong dọc ở miệng và đáy cốc
- hiệu ứng ánh sáng bề mặt men sứ

Pipeline cần đảm bảo ba tiêu chí:

1. Preview editor và render cuối dùng cùng một mô hình hình học.
2. Artwork không bị gãy cạnh hoặc méo do forward warp sai hướng.
3. Kết quả cuối có thể ưu tiên color fidelity hoặc realism tùy profile.

## 3. Kiến trúc file hiện tại

### Backend mug

- `app/pipeline/mugs/cylinder_math.py`
  - chứa toán học lõi của cylindrical mapping
  - tính `U`, `V`, horizontal squeeze, center band, edge roll
- `app/pipeline/mugs/cylindrical_warp.py`
  - dựng remap grid thật
  - warp ảnh bằng `cv2`/GPU remap
  - build curved clip mask theo cùng công thức
- `app/pipeline/mugs/mug_pipeline.py`
  - orchestration toàn pipeline mug
- `app/routers/render.py`
  - API preview, preview-file, render-adhoc, config merge, debug logging

### Frontend mug live editor

- `admin-ui/src/components/PrintAreaEditor.tsx`
  - editor state, control UI, payload preview
  - mirror công thức warp để handle/grid khớp backend
- `admin-ui/src/components/MugCurvePreview.tsx`
  - dựng lưới mug cong bằng SVG
  - adaptive grid sizing
- `admin-ui/src/components/LivePreview.tsx`
  - workflow upload mockup, upload artwork, render popup

## 4. Các không gian tọa độ

Hệ thống mug chạy qua nhiều không gian tọa độ khác nhau.

### 4.1 Output image space

Là không gian pixel thật của ảnh mockup đầu ra:

- trục `x`: `0 .. W-1`
- trục `y`: `0 .. H-1`

Đây là không gian mà `cv2.remap` sinh ảnh.

### 4.2 Canonical print-area space

Vùng in được chuẩn hóa về hệ canonical:

- `X_proj ∈ [-1, 1]`
- `Y_proj ∈ [-1, 1]`

Ý nghĩa:

- `X_proj = -1` là mép trái vùng in
- `X_proj = 0` là chính giữa
- `X_proj = 1` là mép phải

Việc chuẩn hóa này tách bài toán hình học mug ra khỏi kích thước ảnh thật.

### 4.3 Design UV space

Artwork sau cùng được sample từ không gian:

- `U ∈ [0, 1]`
- `V ∈ [0, 1]`

Sau đó map về pixel ảnh design:

- `map_x = U * (dw - 1)`
- `map_y = V * (dh - 1)`

## 5. Homography: đưa vùng in thật về không gian chuẩn

Input từ editor là bốn điểm:

- `top_left`
- `top_right`
- `bottom_right`
- `bottom_left`

Trong backend, `cv2.findHomography` được dùng để tạo ma trận biến đổi từ output image space sang canonical space:

```text
dst_corners (pixel thực) -> src_canonical ([-1,1] rect)
```

Trong frontend, editor dùng `solve_homography(...)` và `apply_homography(...)` để mirror logic này, giúp:

- boundary SVG khớp với render thật
- handle và grid preview bám đúng hình học vùng in

## 6. Design transform trước khi warp

Trước khi warp, artwork không đi thẳng vào remap mà qua bước chuẩn hóa:

1. decode ảnh input
2. ước lượng kích thước canvas vùng in bằng `estimate_print_area_canvas_size(...)`
3. áp `scale`, `offset_x`, `offset_y` bằng `apply_design_transform(...)`

Việc này giúp editor có thể:

- phóng to/thu nhỏ artwork
- kéo artwork trong vùng in
- vẫn render cuối đúng như preview

## 7. Cylindrical inverse mapping

Đây là lõi của module mug.

Thay vì forward warp từ input sang output, hệ thống dùng inverse mapping:

1. đi qua từng pixel output
2. quy pixel đó về `X_proj`, `Y_proj`
3. tính vị trí `U`, `V` trên artwork gốc
4. sample artwork bằng interpolation

Đây là hướng đúng vì:

- không tạo lỗ thủng pixel
- giảm aliasing
- giữ biên mượt hơn

## 8. Horizontal cylindrical mapping

### 8.1 Góc quan sát của khối trụ

Cho `theta_max_deg`, hệ dùng:

```math
\theta_{max} = radians(theta\_max\_deg)
```

Từ `X_proj`, tính:

```math
\sin(\theta) = X_{proj} \cdot \sin(\theta_{max})
```

```math
\theta = \arcsin(\sin(\theta))
```

Sau đó chuẩn hóa:

```math
t = \theta / \theta_{max}
```

Nếu không có squeeze, kết quả UV ngang là:

```math
U = (t + 1) / 2
```

### 8.2 Vì sao dùng `arcsin`

`arcsin` mô phỏng đúng quan hệ giữa:

- bề mặt nhìn thấy của hình trụ
- khoảng chiếu lên ảnh phẳng

Nó tạo foreshortening tự nhiên ở hai mép mà linear mapping không làm được.

## 9. Horizontal squeeze system

Phần này hiện nằm trong `cylinder_math.py` và được mirror ở frontend.

Hệ squeeze hiện có hai lớp độc lập:

1. `center_focus_width`
2. `edge_squeeze`

### 9.1 `center_focus_width`

Mục tiêu:

- dương: nới băng giữa, làm phần giữa rộng hơn
- âm: siết băng giữa, dồn độ rộng ra hai mép

Hệ dùng một `center band` gốc:

```text
BASE_CENTER_BAND = 0.30
MIN_CENTER_BAND  = 0.10
MAX_CENTER_BAND  = 0.70
```

Band đích:

```math
target\_band =
\begin{cases}
0.30 + (0.70 - 0.30)\cdot w & w \ge 0 \\
0.30 + (0.30 - 0.10)\cdot w & w < 0
\end{cases}
```

với `w = clamp(center_focus_width, -1, 1)`.

Sau đó remap theo bán kính:

- nếu `radius <= source_band`, scale tuyến tính về `target_band`
- nếu `radius > source_band`, nội suy phần outer band sao cho điểm biên `1.0` vẫn giữ nguyên

Điểm quan trọng:

- trung tâm `0` luôn giữ nguyên
- mép `1` luôn giữ nguyên
- chỉ phân phối lại chiều ngang ở phần giữa và outer band

### 9.2 `edge_squeeze`

Mục tiêu:

- ép dải sát mép vào trong
- giữ vùng giữa gần như ổn định

Biến:

```text
BASE_EDGE_ROLL_START = 0.55
```

Điểm bắt đầu roll thực tế:

```math
edge\_start = max(BASE\_EDGE\_ROLL\_START, protected\_center\_band)
```

Trong đó `protected_center_band` phụ thuộc vào `center_focus_width`, để edge roll không phá hỏng band giữa.

Sau đó với outer band:

```math
progress = (radius - edge\_start) / (1 - edge\_start)
```

```math
rolled = progress^{1 / squeeze\_power}
```

```math
mapped = progress + (rolled - progress)\cdot edge\_squeeze
```

Ý nghĩa:

- `edge_squeeze = 0`: không ép mép
- `edge_squeeze = 1`: ép mép tối đa theo profile power
- `squeeze_power` càng cao thì hiệu ứng càng dồn sát biên

### 9.3 Kết hợp hai lớp

Thứ tự đúng hiện tại:

1. `apply_center_focus_width(radius, center_focus_width)`
2. `apply_edge_roll(radius_after_width, edge_squeeze, squeeze_power, center_focus_width)`

Điều này làm hai control có thể hoạt động độc lập nhưng vẫn ghép được.

## 10. Vertical curve system

Warp dọc có hai mode.

### 10.1 Legacy mode

Nếu không có `curve_top`/`curve_bottom`, hệ dùng:

```math
curve_v = smile\_base + factor \cdot (pitch / 100)\cdot hr\_ratio \cdot 0.15
```

```math
V = V_{canon} - curve_v \cdot (\cos(\theta) - \cos(\theta_{max}))
```

Mode này đơn giản hơn và phụ thuộc vào:

- `smile_base`
- `pitch`
- `hr_ratio`

### 10.2 Editor explicit mode

Khi editor truyền cả `curve_top` và `curve_bottom`, hệ không xấp xỉ nữa mà giải ngược phương trình forward warp dọc.

Forward preview trong editor có dạng:

```math
Y_{out} = S - S \cdot curve(S) \cdot k
```

Trong đó:

- `curve(S)` là nội suy tuyến tính giữa top và bottom
- `k = hr_ratio * 0.15 * (cos(theta) - cos(theta_max))`

Backend giải ngược đúng bằng phương trình bậc hai:

```math
aS^2 + bS + c = 0
```

Rồi chọn nghiệm gần `Y_out` hơn.

Đây là chỗ rất quan trọng để:

- lưới editor
- preview artwork
- render PNG cuối

không lệch nhau ở miệng và đáy cốc.

## 11. Remap grid thật

Trong `cylindrical_warp.py`, backend tạo `map_x`, `map_y` bằng NumPy trên toàn ảnh output:

1. build `X_proj`, `Y_proj` cho mọi pixel output
2. tính `U`, `V`
3. đổi sang pixel artwork
4. invalidate vùng ngoài domain

```math
map_x = clip(U,0,1)\cdot(dw-1)
map_y = clip(V,0,1)\cdot(dh-1)
```

Pixel ngoài miền hợp lệ được đánh dấu:

```text
map_x = -1
map_y = -1
```

Sau đó warp bằng:

- `gpuRemap(...)`
- interpolation `cv2.INTER_LANCZOS4`

## 12. Curved clip mask

Nếu chỉ remap rectangle rồi alpha-mask thô, biên trên/dưới sẽ không khớp đường cong grid đã khóa.

Để giải quyết, backend build một polygon cong bằng cách:

1. sample đường trên `v=0`
2. sample đường dưới `v=1`
3. warp hai đường bằng cùng hàm `_local_warp_point(...)`
4. đưa ngược qua homography
5. `cv2.fillPoly(...)` để tạo mask alpha cong

Kết quả:

- đường viền artwork đúng theo shape đã lock trong editor
- preview và render cuối đồng nhất hơn

## 13. Grid preview của mug

### 13.1 Grid chia theo UV đều

Baseline hiện tại là lưới đều trong UV space.

Mỗi cell được dựng từ:

```text
u0 = c / cols
u1 = (c + 1) / cols
v0 = r / rows
v1 = (r + 1) / rows
```

### 13.2 Adaptive sizing theo kích thước vật lý

Grid không còn fixed `12 x 10`.

Hiện tại:

```math
cols = round(W_px / 90)
rows = round(H_px / 90)
```

với clamp:

- min `4`
- max `24`

Ý nghĩa:

- vùng in càng lớn thì số ô càng nhiều
- ô gần vuông vật lý hơn
- nhìn grid trực quan hơn khi calibrate mug

### 13.3 Vì sao frontend grid phải mirror exact math

Frontend mug grid hiện không dùng Coons patch gần đúng cho hình học chính nữa. Nó dùng cùng hàm warp logic:

- compute `worldAt(u, v)`
- áp horizontal squeeze
- áp vertical curve
- mới đưa qua homography

Nếu frontend dùng math khác backend thì:

- upload artwork sẽ thấy form lưới đổi
- checker trước upload và artwork sau upload không khít

## 14. Preview endpoints

### 14.1 `/v1/mockup/warp-preview`

Dùng khi chưa upload artwork.

Server dựng một checkerboard RGBA adaptive:

- fill cell sáng/tối
- vẽ line trắng
- warp checker theo cùng công thức mug thật

Mục đích:

- cho người dùng nhìn geometry trước
- kiểm tra grid, curve, squeeze

### 14.2 `/v1/mockup/warp-preview-file`

Dùng khi đã có artwork.

Server:

1. decode artwork thật
2. apply design transform
3. warp theo cùng pipeline geometry
4. trả PNG preview

Điểm kỹ thuật quan trọng:

- preview artwork không được thay đổi lưới chuẩn
- artwork chỉ là lớp ảnh đè lên cùng geometry

## 15. Mug pipeline render cuối

Thứ tự trong `run_mug_pipeline(...)` hiện tại:

1. `decode_design(...)`
2. `estimate_print_area_canvas_size(...)`
3. `apply_design_transform(...)`
4. `cylindrical_warp(...)`
5. `apply_color_match(...)` nếu không ở mode preserve color
6. `apply_shadow_overlay(...)` nếu shadow > 0
7. `composite(...)`
8. `apply_specular_gloss(...)` nếu specular > 0

### 15.1 Color fidelity mode hiện tại cho ad-hoc mug

Mặc định `render-adhoc` của mug đang ưu tiên giữ màu gốc:

- `enable_color_match = false`
- `match_strength = 0`
- `shadow_strength = 0`
- `specular_strength = 0`
- `feather_px = 0`
- `preserve_original_color = true`

Điều này tránh hiện tượng artwork bị bạc màu hoặc wash-out khi test geometry.

## 16. Lighting và compositing

### 16.1 Color match

Chỉ chạy khi:

- không bật `preserve_original_color`
- `enable_color_match = true`

Mục tiêu là kéo màu artwork gần ánh sáng môi trường của mockup.

### 16.2 Shadow overlay

Dùng cho realism trên cốc men sứ:

- nhấn khối thân cốc
- làm artwork bám vào vùng tối/sáng của mockup

### 16.3 Specular gloss

Chỉ chạy khi `specular_strength > 0`.

Hiệu ứng:

- phủ highlight bóng lên trên ảnh composite
- giả lập bề mặt men sứ phản quang

### 16.4 Feather edge

`composite(...)` hỗ trợ `feather_px` để làm mềm alpha ở mép.

Hiện trong ad-hoc mug color-faithful mode, giá trị mặc định là `0` để tránh artwork bị nhòe viền ngoài ý muốn khi chỉnh geometry.

## 17. Dữ liệu cấu hình mug

Các tham số chính của mug trong `config["cylinder"]`:

- `theta_max_deg`
- `smile_base`
- `pitch`
- `curve_top`
- `curve_bottom`
- `edge_squeeze`
- `squeeze_power`
- `center_focus_width`

Các range UI hiện hành:

- `curvePct`: `0..100`
- `curveTop`: `-100..100`
- `curveBottom`: `-100..100`
- `edgeSqueeze`: `0..1`
- `squeezePower`: `1..5`
- `centerFocusWidth`: `-1..1`

## 18. Debug và observability

Backend hiện có `summarize_horizontal_squeeze(...)` và `_log_horizontal_squeeze_debug(...)`.

Nó log:

- `sample_in`
- `sample_out`
- `sample_delta`
- `mode`
  - `inactive`
  - `width_only_active`
  - `edge_only_active`
  - `edge_and_width_active`

Frontend editor cũng có log debug tương ứng để so payload preview với render thật.

## 19. Test hiện có

`tests/test_cylindrical_squeeze.py` đang cover:

- no-op khi `edge=0` và `width=0`
- center và edge anchor
- edge roll chỉ tác động outer band
- width dương mở giữa
- width âm siết giữa
- UV mapping chỉ đổi theo trục ngang, không làm lệch `V`

Đây là bộ test quan trọng vì preview mug rất dễ “nhìn có vẻ đúng” nhưng sai profile warp.

## 20. Giới hạn hiện tại

Một số giới hạn kỹ thuật hiện vẫn tồn tại:

1. Hình học mug vẫn là mô hình cylindrical 2.5D, chưa phải mesh 3D thật.
2. `normal_map` trong ad-hoc hiện được synthesize từ profile trụ đơn giản, chưa phải normal map quét vật thể thật.
3. Specular extraction ad-hoc vẫn là heuristic threshold từ ảnh mockup.
4. Preview frontend vẫn là SVG/ảnh 2D, không có physically-based shading realtime.

## 21. Hướng nâng cấp kỹ thuật tiếp theo

### Geometry

- thêm profile cylinder theo template riêng cho từng dòng mug
- hỗ trợ asymmetric mug body nếu mockup không phải trụ đều
- precompute LUT cho `theta -> U`

### Preview

- cache preview PNG theo hash payload
- heatmap trực tiếp cho horizontal squeeze / vertical curvature
- preset control cho mug profile

### Rendering

- profile lighting per template
- specular mask chuẩn hóa theo vật liệu
- optional linear-light full pipeline end-to-end

### Validation

- snapshot test cho preview grid
- image diff test giữa preview và final render
- regression pack cho các template mug chuẩn

## 22. Tóm tắt ngắn

Về bản chất, module mug hiện là:

- một hệ inverse cylindrical mapping theo canonical space
- có thêm hai lớp horizontal redistribution độc lập: `center_focus_width` và `edge_squeeze`
- có vertical curve solved chính xác bằng nghiệm bậc hai khi editor dùng top/bottom curve
- có preview grid adaptive và mirror math giữa frontend/backend
- có pipeline compositing riêng cho vật liệu cốc men sứ

Nếu cần đọc code theo thứ tự đúng, nên đi theo chuỗi:

1. `app/routers/render.py`
2. `app/pipeline/mugs/mug_pipeline.py`
3. `app/pipeline/mugs/cylindrical_warp.py`
4. `app/pipeline/mugs/cylinder_math.py`
5. `admin-ui/src/components/PrintAreaEditor.tsx`
6. `admin-ui/src/components/MugCurvePreview.tsx`
