# Kiến Trúc Chi Tiết: Dịch Vụ Mockup Ly Cốc (Mug Mockup Service)

Tài liệu này cung cấp cái nhìn cực kỳ chi tiết (Deep Dive) vào lõi logic chịu trách nhiệm xử lý Mockup cho dòng sản phẩm Ly Cốc sứ (Mugs/Tumblers). Quá trình render bao gồm nhiều công thức giải tích phi tuyến tính phức tạp nhằm tái tạo chính xác hiệu ứng tụ viền, độ cong của miệng/đáy cốc và chất liệu men sứ bóng theo ánh sáng môi trường. Toàn bộ mã nguồn cốt lõi nằm tại thư mục `app/pipeline/mugs/`.

---

## 1. Lưu Đồ Xử Lý (Pipeline Flow)

Mọi yêu cầu xử lý ảnh Mugs sẽ được điều hướng vào file khởi tạo `mug_pipeline.py` (Hàm `run_mug_pipeline`). Quy trình thực thi gồm **5 Bước Cốt Lõi** tuân thủ nghiêm ngặt thứ tự sau:

1. **Giải mã & Quy chuẩn Design (`decode_design` & `apply_design_transform`)**: Chuyển đổi Offset, canh chỉnh Scale theo cấu hình Print Area Canvas từ Admin UI.
2. **Biến dạng hình trụ Toán học (`cylindrical_warp`)**: Phép chiếu 2D phẳng lên bề mặt 3D cong theo thông số (Mugs đặc thù).
3. **Cân bằng trắng/Màu sắc (`apply_color_match`)**: Chỉnh màu Design ám vào màu môi trường (Áp dụng Von Kries Adaptation nếu không bật `preserve_original_color`).
4. **Đổ bóng bề mặt men sứ (`apply_shadow_overlay`)**: Tạo nổi khố nhờ kĩ thuật Overlay.
5. **Cắt ghép mờ viền & Phủ chói sáng Specular (`composite` & `apply_specular_gloss`)**: Áp dụng Gaussian Mask để thiết kế hòa trộn tự nhiên vào viền không bị gãy răng cưa, cuối cùng phủ vệt sáng đèn Studio đè lên.

Tham số đầu vào cho Pipeline (`MugAssets`):
- `mockup`: Ảnh nền cốc trắng nguyên bản.
- `mask`: Hình dáng đa giác giới hạn khu vực được phép chèn thiết kế in.
- `shadow_map` / `normal_map`: Bản đồ vùng tối bề mặt (Chiết xuất khu vực chắn sáng của Cốc).
- `specular_map`: Bản đồ phản quang sáng bóng, tạo ra vệt chói của bóng đèn hoặc Flash viền.
- `config`: Metadata tinh chỉnh từ người dùng quản trị (độ cong ống kính - pitch, nụ cười - smile, tốc độ chiếu tụ viền ngang).

---

## 2. Đặc Thù Lõi: Toán Học Khối Trụ Cốc (Cylindrical Warp Math)

Logic được viết tại chuỗi thư viện `cylinder_math.py` và `cylindrical_warp.py`. Đối với Cốc, hệ thống không dùng phép dán nhăn (Thin Plate Spline) của Vải mà bắt buộc thực hiện Phép Chiếu Nghịch Đảo Cylindrical (Inverse Mapping) quét từng Pixel tại đầu ra để đối chiếu ngược trục hình trụ thu về không gian 2D gốc, giúp ảnh tránh bể hạt cực độ khi dùng `gpuRemap` xen kẽ ngoại suy `INTER_LANCZOS4`.

### 2.1 Ma Trận Gốc Rễ Đánh Dấu Cốc (Homography Matrix)
Nhận vào 4 tọa độ góc được kéo thủ công từ UI (Top/Bottom - Left/Right) làm `dst_corners`. Sử dụng lệnh `cv2.findHomography` cùng hình chữ nhật gốc 2x2. Nó giải sinh ra ma trận biến đổi phối cảnh 3x3 $H\_mat$.
Quy toàn bộ dải ảnh Output ra một hệ tọa độ lưới ma trận lý tưởng gọi là **Canonical Format** giới hạn trong mảng $[-1, 1]$. Tạm gọi Trục Ngang là $X_{proj}$ và Trục Dọc Cốc là $Y_{proj}$. Nhờ vậy, chiếc Cốc to, nhỏ, méo, chéo cũng trở thành 1 hình dạng chuẩn để chạy toán.

### 2.2 Ánh Xạ Biến Dạng Viền Ngang Trục (Horizontal Squeeze & Arcsin)
Để hai mép viền thiết kế có biểu hiện cuộn tròn bọc ra mặt sau lưng cốc (Tụ phối cảnh - Foreshortening):
* **Cơ bản hình Sin:** Lấy điểm Pixel $X_{proj}$, truy nghịch hàm lượng giác thông qua giới hạn cực đại mà Camera quay Cốc thấy được ($\theta_{max}$, mặc định $\approx 52^\circ\rightarrow 60^\circ$).
  $$ \theta_{Arc} = \arcsin( X_{proj} \times \sin(\theta_{max}) ) $$
* **Bộ nén mở rộng (`squeeze_power` & `edge_squeeze` & `center_focus_width`):** 
  Ngoài độ túm viền tự nhiên, Cốc sẽ áp dụng một thuật toán bóp thêm phi tuyến tính để bảo toàn tỉ lệ logo ở tâm Cốc `center_focus` và ép cong rất gắt ở 2 rìa `edge_squeeze`.
  Công thức lõi nén ngang với số mũ Power (Mặc định 2.0):
  $$ Curved\_Radius = (Radius_{width-adjusted})^{1.0 / Squeeze\_Power} $$
  Kết quả của chiều này trả về là tọa độ lấy nét ngang của Pixel tính trên tấm ảnh Design 2D gốc: $U \in [0, 1]$.

### 2.3 Độ Cong Chiều Dọc "Miệng Nụ Cười" (Smile Curve & Camera Pitch)
Cốc được chụp bởi Camera chiếu chúi xuống (Pitch > 0) hoặc ngước lên (Pitch < 0), miệng cốc và đáy cốc sẽ bị phình vòng cung gọi là nụ cười Smile. Đáy luôn võng mạnh hơn.
* **Tỉ lệ khung hình vật lý thân hình trụ (HR Ratio):** Tỷ lệ Cao / Ngang $Height / Width$.
* **Độ Võng Do Xoay Hình Trụ ($Cos\_Displacement$):** 
  Ở xương sống Cốc ($\theta = 0$), độ sâu võng parabol lớn nhất $\cos(0) - \cos(\theta_{max}) = 1 - 0.x$. Tại mép hai bên hông Cốc ($\theta = \theta_{max}$), lượng xô lệch bù bằng 0 (Hai mép luôn giữ nguyên không bị thụt lùi).
* **Nghiệm Kép (Quadratic Equation - Dành cho Editor tương tác):**
  Hệ thống hỗ trợ 2 mốc cong Top/Bottom độc lập, thay vì tính tương đối, hệ thống giải phương trình bậc hai tuyến tính hóa: $Y_{out} = S - S \times curve(S) \times k$. Từ đó cho ra chuỗi nghiệm $Root_1, Root_2$. Trích lập nhánh cho ra kết quả khớp 100% với mắt người nhìn trên lưới UI đang vặn.
  Toàn bộ độ lệch sẽ cung cấp tọa độ dọc trên ảnh phẳng gốc: $V \in [0, 1]$.
  
*(Từ $U$ và $V$ này, OPENCV sẽ bốc đúng hạt pixel màu đưa ra màn ảnh 3D).*

---

## 3. Quá Trình Đổ Ánh Sáng Tương Tác Men Sứ Bóng (Ceramics Lighting Shader)

Sứ Ceramic (hay Ly Cốc nhựa cách nhiệt Tumbler) có vật lý rất đặc trưng, hệ số phản xạ màu bề mặt cứng rắn làm phản chiếu ánh sáng mạnh mẽ cả tối và chói chang, code quản lý nằm tại `shadow_overlay.py` và `specular_gloss.py`.

### 3.1 Chuyển Đổi Không Gian Màu Định Luật Lý Tưởng (Linear Color Transformation)
Đây là cấu trúc thiết yếu cho Mockup Sứ. Thay vì Blender trên hệ màu nhìn màn hình RGB (sRGB), Pixel phải bị phá chuyển về dạng cường độ sáng dạng năng lượng vật lý là `Linear RGB`.
$$ C_{Linear} = (C_{sRGB})^{2.2} $$
Chỉ quy đổi Linear, các vệt chói sáng mới tỏa đúng dạng Gradient, triệt tiêu tối đa vấn đề bị "nhòe xám bẩn bóng (Dirty Gray Fringes)" gớm ghiếc ở mép chữ dán trên cốc.

### 3.2 Overlay Shadow Blend (Sự Căng Bóng Gốc Lõi)
Ngược biệt với Vải vóc Cotton sùi lỗ nhám dùng `Multiply`, hệ thống Mug Mockup dùng sự nhào nặn sáng/tối hai chiều **Overlay Blend Mode**:
$$ 
Overlay\_Blend(Base, Shadow) = 
\begin{cases} 
2 \times Base \times Shadow & \text{khi } Shadow < 0.5 \\
1 - 2(1 - Base)(1 - Shadow) & \text{khi } Shadow \ge 0.5 
\end{cases} 
$$
Kỹ thuật này khiến lớp màu bám viền của cốc không bị chìm nghỉm vô hồn nhưng đồng thời phần sáng đèn hắt vào thân cốc củng cố thêm sắc trắng chói trang của họa tiết. (`shadow_strength` thường được đặt rất cao ở $0.45$).

### 3.3 Phủ Đệm Chói Sáng Phản Quang (Phong Specular Model Gloss)
Ly cốc sứ phải được phản xạ bóng đèn Flash (Studio Glaze). Code Specular chèn đè lên tất cả.
* Tự động dò ngưỡng từ gốc (`extract_specular_from_mockup`): Hệ thống tự lọc tách các vị trí trắng tinh (Brightness $> 220$), phân tách các hạt sáng nhuyễn thông qua Gaussian Blur (`15x15`). Sinh ra lớp chớp sáng trắng.
* Căn chuẩn mô hình Phong (Phong Shading). Đẩy chỉ số độ chói $Shininess = 20.0$.
* Chế độ hòa trộn phản quang chói gắt ($Screen\_Blend$ trên nền Linear Space):
  $$ Final_{linear} = 1.0 - (1.0 - Composite_{linear}) \times (1.0 - Specular \times Strength_{0.3}) $$

Sau bước này, chất lượng ảnh hiển thị cực kỳ rực rỡ, hạt Pixel được gom thành màu `sRGB` gửi về cho Admin Web Browser ở chuỗi Bytes PNG nguyên khai mượt mà cao cấp và sát với thực tế nhất.
