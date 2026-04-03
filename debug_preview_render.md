# 🐢 Debug chậm preview render trong LiveView

## 📌 Tổng quan vấn đề

Hệ thống preview render (live preview khi kéo mesh / upload design) đang:
- phản hồi chậm
- có delay không ổn định
- đôi khi bị “đứng” hoặc load lâu bất thường

Log backend cho thấy lỗi:

sqlalchemy.exc.ProgrammingError:
UndefinedTableError: relation "rendered_results" does not exist

---

# 🚨 1. Nguyên nhân chính

## ❌ 1.1 Lỗi database làm chậm toàn pipeline

Worker render xong preview nhưng fail khi insert DB → gây delay response.

## ❌ 1.2 Preview dùng pipeline quá nặng

Đang chạy spline, lighting, high-res giống final render → sai kiến trúc.

## ❌ 1.3 Không dùng cache warp map

Rebuild map mỗi lần → cực chậm.

## ❌ 1.4 Ảnh quá lớn

Preview dùng 1024–2048px → remap nặng.

## ❌ 1.5 Interpolation sai

Dùng cubic/lanczos → chậm hơn nhiều.

---

# ⚡ 2. Kiến trúc đúng

Preview = FAST  
Final = HIGH QUALITY

---

## ✅ Preview pipeline chuẩn

- Resize 512px
- Cache warp map
- cv2.INTER_LINEAR
- Không lighting

---

# 🧠 3. Tách Preview vs Final

Preview:
- 512px
- Linear
- No DB
- No lighting

Final:
- 2048px
- Cubic/Lanczos
- Có DB
- Có lighting

---

# 🚀 4. Fix ngay

1. Bỏ DB insert cho preview
2. Resize về 512px
3. Dùng INTER_LINEAR
4. Dùng cache warp map

---

# 📊 5. Benchmark

Preview chuẩn: < 20ms

---

# 🔍 6. Debug checklist

- Log từng bước
- Check cache hit
- Check image size
- Check interpolation

---

# 🏁 7. Kết luận

Vấn đề chính:
- DB error
- Pipeline sai
- Không tối ưu preview

Sau khi fix:
- Preview realtime mượt
- UX cải thiện rõ rệt
