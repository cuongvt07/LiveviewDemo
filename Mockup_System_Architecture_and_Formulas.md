# Hệ Thống Mockup POD - Cấu Trúc, Công Nghệ & Công Thức Toán Học (Mugs & Clothes)

Tài liệu này trình bày chuyên sâu về các công nghệ, thuật toán và công thức toán học đang được áp dụng trong hệ thống tạo Mockup sản phẩm POD, đặc biệt tập trung vào hai dòng sản phẩm cốt lõi: **Cốc Sứ (Mugs)** và **Quần Áo (Clothes)**.

---

## 1. Công Nghệ Giao Diện & Máy Chủ (Shared Tech Stack)

Hệ thống được thiết kế theo kiến trúc Client-Server với quy trình tự động hóa Mockup hiệu năng cao:

*   **Backend Core:** `Python 3.11`, kết hợp `FastAPI` (xử lý bất đồng bộ API) và ASGI Server `Uvicorn`.
*   **Xử Lý Ảnh Chuyên Quanh (Computer Vision):** Trực tiếp thao tác ma trận pixel bằng `OpenCV (cv2)` và tính toán vector diện rộng với `NumPy`. Tối ưu nội suy để render ảnh đầu ra không bị vỡ.
*   **Cơ Sở Dữ Liệu:** `PostgreSQL` (lưu trữ metadata, tọa độ), sử dụng ORM `SQLAlchemy (asyncpg)` và `Alembic` để quản lý phiên bản database. Caching và Job Tracking bằng `Redis`.
*   **Frontend (Admin UI):** Xây dựng bằng `React`, `TypeScript`, `Vite` và `TailwindCSS`. Đặc biệt sử dụng chuẩn `HTML5 SVG (<polyline>, <path>)` để vẽ lưới định vị Warp Grid trực tiếp trên trình duyệt.
*   **Triển khai:** Hoạt động dưới hệ sinh thái container `Docker` và `docker-compose`.

---

## 2. Đường Ống Xử Lý Ánh Sáng Chung (Pipeline Lighting)

Dù là Cốc hay Áo, để mang lại cảm giác chân thực nhất (Photorealistic), các lớp ảnh cần tuân thủ quy tắc ánh sáng bề mặt:

*   **Linear Color Space (Không Gian Màu Tuyến Tính):**
    Mọi tính toán liên quan đến pha trộn ánh sáng (Blend Normal, Multiply) đều phải chuyển từ hệ màu `sRGB` sang `Linear Space` bằng công thức xấp xỉ $Color^{2.2}$. Lời giải này chặn hiện tượng "viền đen/xám xịt" khi hòa trộn màu. Sau khi hòa xong, mới gamma correction về $Color^{1/2.2}$ để hiển thị.
*   **Von Kries Chromatic Adaptation (Cân Bằng Màu Sắc - Color Match):**
    Dịch chuyển màu thiết kế (Design Color) nhẹ theo "White Point" của môi trường (Mockup Base).
    $$ RGB_{shifted} = RGB \times (1 + (White\_Point - 1.0) \times Strength) $$
    *(Mugs áp dụng cường độ $Strength = 0.40$ do đèn Studio đánh gắt hơn, Clothes lấy $Strength = 0.25$ vì thích hợp tự nhiên hơn).*
*   **Composite Feather Edge:** Dùng kỹ thuật làm mờ rìa (Feather Px) cho mặt nạ $Mask$ giúp vùng in (Print Area) ăn nhập trơn tru với viền.

---

## 3. Lõi Thuật Toán Xử Lý Riêng Biệt Cho Ly Cốc (Mugs)

Bề mặt ly cốc là dạng hình trụ cong (Cylinder). 

### 3.1 Phương Pháp Chống Vỡ Răng Cưa (Resampling)
Do hiệu ứng Phối Cảnh Tụ (Foreshortening) — hai mép cốc ép chặt lại — hệ thống áp dụng **Phép Chiếu Ngược (Inverse Mapping)** thay vì chiếu thuận, dùng bộ lọc nội suy chất lượng cao nhất của OpenCV là `cv2.INTER_LANCZOS4`.

### 3.2 Công Thức Toán Học Trụ Cốc (Cylindrical Warp Math)

Để bẻ cong ảnh phẳng dán quanh mặt cốc xoay quanh trục ngang:

1.  **Homography (Quy chiếu ma trận):** Giải ma trận nghịch đảo $H^{-1}$ từ 4 góc lưới cốc do UI gửi xuống, đưa các Pixel hiện tại về không gian chuẩn Canonical $X_{proj}, Y_{proj} \in [-1, 1]$.
2.  **Ánh Xạ Tụ Viền Chiều Ngang (X-Axis Mapping):**
    Sử dụng góc xoay viền tối đa $\theta_{max}$ (mặc định $52^\circ$). Tính toán $\theta$ theo hàm Sin:
    $$ \theta = \arcsin(X_{proj} \times \sin(\theta_{max})) $$
    Tọa độ U trích xuất từ thiết kế phẳng:
    $$ U_{base} = \frac{\theta}{\theta_{max}} $$
3.  **Tạo Hình Đường Cong "Nụ Cười" & Độ Chúc Camera (Smile Curve & Pitch):**
    Tính độ cong của miệng/đáy cốc bị phụ thuộc vào hình dạng Cốc ($HR\_Ratio$) và góc chúc camera ($Pitch$):
    $$ HR\_Ratio = \frac{Height}{Width + 1e^{-6}} $$
    $$ curve\_v = smile_{base} + \left( \frac{-Y_{proj}}{2.0} \right) \times \left( \frac{Pitch}{100.0} \right) \times HR\_Ratio \times 0.15 $$
    Độ dịch chuyển viền dọc sẽ hình thành vòng cung lớn trũng ở giữa hàm Cos:
    $$ V_{final} = V_{canon} - curve\_v \times (\cos(\theta) - \cos(\theta_{max})) $$

### 3.3 Đánh Bóng Vật Liệu Sứ (Phong Shading - Specular)
Vỏ cốc sứ láng bóng tạo ra điểm sáng hẹp và rực rỡ ($Shininess \approx 20.0$), áp dụng Screen Blend:
$$ Color_{final} = BaseWarped + SpecularMap \times SpecularStrength $$

---

## 4. Lõi Thuật Toán Xử Lý Riêng Biệt Cho Quần Áo (Clothes)

Bề mặt của áo là vải vóc co giãn không theo nguyên tắc đa giác tĩnh, nó phụ thuộc vào nếp nhăn và nếp gấp sinh học.

### 4.1 Biến Dạng Theo Nếp Nhăn Bằng Thin Plate Spline (TPS Warp)
Thin Plate Spline là thuật toán uốn lưới mượt (smooth surface) đi qua một tập điểm cố định.
*   **Dùng OpenCV `createThinPlateSplineShapeTransformer`:** Cấp cho nó danh sách các điểm nguồn (Source Points - lưới thiết kế) và điểm đích (Destination Points - định vị bị nếp nhăn đẩy lệch trên ảnh). Khi áp dụng phép biến dạng, Design sẽ sụt lún theo đúng cấu trúc sóng vải thực tế thay vì biến dạng hình trụ.

### 4.2 Trích Xuất Điểm Biến Dạng Nhăn (Harris Control Points)
Hệ thống tự động phát hiện các nếp nhăn trên áo để sinh chuẩn xác điểm neo cho hàm TPS:
1.  **Harris Corner Detection ($\approx$ Đỉnh Nếp Góc):** 
    Sử dụng thuật toán dò góc Harris (Harris Corner) kết hợp với mask vải để bắt cứng các đỉnh nhăn nhúm gấp khúc sắc nét.
2.  **Dense Grid (Lưới Phủ Vùng Phẳng):** Xen kẽ các điểm mạng lưới trơn tru để tính phủ đều và tránh lỗi TPS méo mó cục bộ ở nơi không có biến động sắc cạnh (dùng np.meshgrid và cv2.findNonZero vùng bao bounding_rect).
3.  **Sobel Gradient Displacement (Tính Độ Dốc Đẩy Điểm):**
    Nhân chập ảnh xám (`gray_f`) bằng bộ lọc Sobel theo $X$ và $Y$, rồi lọc qua Gaussian Blur (lọc nhiễu ảnh thô ráp), mục đích tính **hướng xô** (Gradient Displacement) của các tọa độ đích nếp nhăn bằng cường độ ($Strength \approx 6.0$):
    $$ Dest_x = Src_x + \frac{Gradient\_X}{\max(|Gradient\_X|) + 1e^{-6}} \times Strength $$
    $$ Dest_y = Src_y + \frac{Gradient\_Y}{\max(|Gradient\_Y|) + 1e^{-6}} \times Strength $$

### 4.3 Ánh Sáng Nếp Gấp Cho Vải Cotton (Wrinkle Lighting - Multiply Blend)
Đặc tính vi mô của vải (như cotton) thấm màu design thay vì để ánh bóng, do đó dùng thuật toán đổ bóng vải:
1.  **Multiply Blend Cốt Lõi:** Thiết kế cần tối đúng chiều sâu của sợi vải.
    $$ Multiplied = Design_{Linear} \times Shadow_{Linear} $$
2.  **Hòa Trộn Bù Sáng Tùy Biến (Shadow Strength):** 
    $$ Shadowed = Design_{Linear} + (Multiplied - Design_{Linear}) \times ShadowStrength $$
    *(Vùng sáng đỉnh nếp $\approx 1.0$ tiếp nhận và giữ nguyên màu Design sáng rõ, vùng lõm giặt $\approx 0.5$ khiến mẫu design tối màu đúng theo lõm sóng học vật lý)*.
3.  **Hạn Chế Bắt Sáng Nhẹ (Low Specular):**
    Trái với Cốc Sứ láng bóng, vải cực ít phản ánh sáng đèn rực. Hệ thống giảm Specular ở mức rất rất thấp ($\approx 0.08$), và chỉ chọn riêng dải pixel sáng gắt nhất (Ngưỡng $>240$). Tránh áo in ra biến thành mảng bọc nhựa.
