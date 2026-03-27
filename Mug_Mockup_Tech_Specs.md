# Cấu Trúc & Công Nghệ Hệ Thống Mockup POD (Tập Trung Vào Dòng Sản Phẩm Mugs)

Tài liệu này tổng hợp toàn bộ các công nghệ hiện đang chạy trong ứng dụng, được phân loại thành **Công Nghệ Chung** (cho toàn bộ hệ thống POD Mockup) và **Công Nghệ Riêng** (đặc thù dành riêng cho việc bẻ cong và dựng hình cốc sứ - Mugs), đi kèm với các công thức toán học nội suy.

---

## 1. Công Nghệ Chung Giao Diện & Máy Chủ (Shared Tech Stack)

Hệ thống được thiết kế theo kiến trúc Client-Server, hoạt động qua API và quản trị trên Docker.

### 1.1 Backend & Middleware
*   **Ngôn Ngữ Cơ Sở:** `Python 3.11`.
*   **Web Framework:** `FastAPI` (hiệu năng cao, bất đồng bộ).
*   **Web Server ASGI:** `Uvicorn` (chạy trên port 8000 nội bộ).
*   **Xử Lý Ảnh Chuyên Sâu:** 
    *   `OpenCV` (cv2): Cung cấp các thuật toán ma trận hình học (Homography, WarpPerspective, Remap, DMatch).
    *   `NumPy`: Xử lý mảng và tính toán vector diện rộng, tối ưu tốc độ CPU.
*   **Cơ Sở Dữ Liệu:** `PostgreSQL` (lưu mẫu thiết kế, tọa độ in) truy xuất qua `SQLAlchemy` (với driver asyncpg) và quản lý version bằng `Alembic`.
*   **Cache & Message Broker:** `Redis` (để tracking job hoặc rate-limiting).
*   **Triển Khai (Ops):** `Docker`, `Docker Compose`, `Dockerfile.api`.

### 1.2 Frontend (Admin UI)
*   **Core:** `React` / `TypeScript` / `TailwindCSS` (nếu có).
*   **Module Builder:** `Vite` (môi trường dev nhanh, HMR support).
*   **Đồ Họa Vectơ (Grid Canvas):** Element chuẩn HTML5 `<svg>`, `<polyline>`, `<path>`. Dùng SVG để vẽ lưới định vị biến dạng (Warp Grid) trong thời gian thực mà không làm nghẽn luồng xử lý ảnh.

### 1.3 Quy Tắc Ánh Sáng PBR Chung (Pipeline Lighting)
Chất liệu của Mockup được tái lập qua các lớp MAP:
*   **Shadow Map (Multiply):** Mô phỏng bóng tối tự nhiên đổ trên vật thể.
*   **Normal Map / Displacement Map:** Nhận dạng nếp gấp, bề mặt nhám (chủ yếu dùng cho áo thun, vải vóc - `Thin Plate Spline`).
*   **Linear Color Space:** Mọi tính toán màu được biến đổi từ `sRGB` sang `Linear Space` ($Color^{2.2}$) để trộn ánh sáng thực tế hơn, rồi mới Gamma Correction ($Color^{1/2.2}$) trả về `sRGB`.

---

## 2. Lõi Công Nghệ Dành Riêng Cho Ly Cốc (Mugs Specificity)

Điểm khác biệt lớn nhất của ly cốc (Mugs/Tumblers) so với T-Shirt hay Canvas là bề mặt hình trụ (Cylinder). Do đó, thuật toán ánh xạ (Mapping) chuyển 2D sang phối cảnh biến đổi thành **Cylindrical Inverse Mapping**.

### 2.1 Phương Pháp Chống Vỡ Ảnh (Resampling)
*   Do ảnh ở hai mép viền cốc bị ép chặt lại (Phối cảnh tụ - Foreshortening), hệ thống không dùng phép chiếu thuận mà dùng **Phép Chiếu Ngược (Inverse Mapping)** kết hợp với hàm nội suy `cv2.INTER_LANCZOS4`. Đây là công cụ khử răng cưa và giữ viền chữ/răng cưa trên design tốt nhất thay vì `BILINEAR`.

### 2.2 Công Thức Toán Học Trụ Cốc (Cylindrical Warp Math)

Để ánh xạ một thẻ PNG phẳng dán quanh mặt cốc xoay quanh trục, lưới 2D không bị nén tuyến tính mà thay đổi theo hình Sin.

#### A. Homography (Mặt phẳng tham chiếu tọa độ)
Dùng 4 góc lưới trên cốc $P_{screen}$ để giải ma trận nghịch đảo $H^{-1}$.
Một lưới điểm Pixel `grid(x,y)` trên bức ảnh cuối cùng sẽ được quy về tọa độ chuẩn Canonical: $X_{proj}$ và $Y_{proj}$ $\in [-1, 1]$.

#### B. Ánh Xạ X (Tụ Viền Ngang Trục Trụ)
Giả sử góc xoay ngang của mặt print area là $\theta_{max}$ (Theo mặc định cấu hình là $52^\circ$, tương đương khoảng cách bo mắt người nhìn thấy).
*   Tính góc $\theta$ tại mỗi cột pixel (do mặt trụ bẻ cong):
    $$ \sin(\theta) = X_{proj} \times \sin(\theta_{max}) $$
    $$ \theta = \arcsin(\sin(\theta)) $$
*   Xác định tọa độ ngang ($U$) của điểm ảnh trên thiết kế gốc:
    $$ U_{base} = \frac{\theta}{\theta_{max}} $$
    *Lưu ý: $\cos(\theta)$ sẽ được dùng ở bước Y để tạo độ cong miệng cốc.*

#### C. Ánh Xạ Y (Nụ Cười Mặt Cốc - Smile Curve & Pitch Offset)
Cốc không bao giờ chỉ phẳng thẳng đứng trên màn hình 2D. Góc đặt Camera quay cốc từ trên xuống (hoặc dưới lên) tạo thành đường cong Parabol ở miệng và đáy cốc lớn dần về mép ngoài. 

*   **Tỉ lệ hình học cốc (H/R Ratio):**
    Căn cứ màn hình $H_{px}$ và $W_{px}$ để tính độ vút (tumbler sẽ cong gắt hơn, espresso cong lài hơn).
    $$ HR\_Ratio = \frac{Height}{Width + 1e^{-6}} $$

*   **Độ Cong Biến Thiên (Adaptive Curve):**
    Smile base (viền cung chuẩn) sẽ thay đổi khi Pitch (Góc chúc của Camera) thay đổi.
    $$ factor = \frac{-Y_{proj}}{2.0} $$ (Tại miệng cốc y=-1 -> factor=0.5, Tại đáy cốc y=1 -> factor=-0.5).
    $$ curve\_v = smile_{base} + factor \times \left( \frac{pitch}{100.0} \right) \times HR\_Ratio \times 0.15 $$

*   **Chốt Vị Trí Dọc ($V_{final}$):**
    Độ lệch vệt vòng cung (Arc Displacement) sẽ đạt đỉnh ở giữa cốc ($\theta=0 \rightarrow \cos(0)=1$) và bằng $0$ ở viền mép biên ngoài ($\theta = \theta_{max}$).
    $$ V_{final} = V_{canon} - curve\_v \times (\cos(\theta) - \cos(\theta_{max})) $$

Kết luận biến đổi: Một pixel tọa độ $(x, y)$ trên chiếc cốc render ra 3D sẽ lấy màu tại tọa độ tỉ lệ $(U_{final}, V_{final})$ của bản vẽ thiết kế 2D phẳng truyền lên.

### 2.3 Vật Liệu Riêng Biệt Cho Bề Mặt Sứ (Glossy & Specular)
Với Mugs, vật liệu vải nhám (`Oren-Nayar Diffuse`) bị bỏ qua, hệ thống sẽ kích hoạt tính toán phản xạ trơn bóng (Specular Shininess/Phong Shading):
*   Tính toán sự rực sáng ở cường độ cao $Shininess \approx 20.0$.
*   Thái rọi ánh sáng màn hình (Screen Blend) đè lên Base Warp:
    $$ Color = BaseColor + SpecularMap \times SpecularStrength $$
*   Cắt gọn viền theo hệ thống mặt nạ đa giác (Mask Polygon Points) kết hợp cơ chế Feather Edge tạo mờ rìa (chống răng cưa gắt biên) để Mugs ăn nhập hoàn toàn lơ lửng trong không gian background cảnh thật.
