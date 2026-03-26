import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator
import hashlib
import sys

class TPSRemapCache:
    """
    Tối ưu hoá TPS warping bằng cách cache các bản đồ remap (map_x, map_y).
    Sử dụng fixed-point math (CV_16SC2) để đạt tốc độ xử lý CPU cao nhất.
    """
    def __init__(self, max_cache_mb=200):
        self._cache = {}
        self.MAX_CACHE_MB = max_cache_mb

    def _make_key(self, src_pts, dst_pts, shape):
        """ Tạo key duy nhất cho cache dựa trên input points và kích thước ảnh. """
        pts_data = np.concatenate([src_pts, dst_pts]).astype(np.float32).tobytes()
        h_val = hashlib.md5(pts_data).hexdigest()
        return f"{h_val}_{shape[1]}x{shape[0]}"

    def _check_memory(self):
        """ Giới hạn kích thước cache để tránh tràn RAM. """
        total_bytes = 0
        for map_x, map_y in self._cache.values():
            total_bytes += map_x.nbytes + map_y.nbytes
        
        if (total_bytes / (1024 * 1024)) > self.MAX_CACHE_MB:
            # Xoá một nửa cache cũ nhất (FIFO)
            keys = list(self._cache.keys())
            for k in keys[:len(keys)//2]:
                del self._cache[k]

    def get_remap(self, src_pts, dst_pts, output_shape):
        """ 
        Tính toán hoặc lấy từ cache bản đồ remap tối ưu.
        output_shape: (H, W)
        """
        key = self._make_key(src_pts, dst_pts, output_shape)
        if key in self._cache:
            return self._cache[key]

        h, w = output_shape[:2]
        
        # 1. Sử dụng RBFInterpolator (hiệu quả hơn TPS thuần túy cho ít điểm)
        # Smoothness factor chống hiện tượng dao động (ringing) khi ít điểm
        n_pts = len(src_pts)
        smoothing = 0.0 if n_pts >= 12 else max(0.1, (12 - n_pts) * 0.05)
        
        # src_pts -> dst_pts mapping: rbf(query_in_dst) = coord_in_src
        rbf_x = RBFInterpolator(dst_pts, src_pts[:, 0], kernel='thin_plate_spline', smoothing=smoothing)
        rbf_y = RBFInterpolator(dst_pts, src_pts[:, 1], kernel='thin_plate_spline', smoothing=smoothing)
        
        # 2. Tạo grid truy vấn trên toàn bộ ảnh đích
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        query_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        # 3. Tính toán map_x, map_y (float32)
        map_x_f = rbf_x(query_pts).reshape(h, w).astype(np.float32)
        map_y_f = rbf_y(query_pts).reshape(h, w).astype(np.float32)
        
        # 4. Tối ưu hoá: Chuyển sang Fixed-point (CV_16SC2) -> Nhanh gấp 2 lần khi chạy remap()
        map_x_fixed, map_y_fixed = cv2.convertMaps(map_x_f, map_y_f, cv2.CV_16SC2)
        
        self._check_memory()
        self._cache[key] = (map_x_fixed, map_y_fixed)
        
        return self._cache[key]

    def apply(self, design_img, src_pts, dst_pts, output_size=None):
        """
        Áp dụng warping TPS lên thiết kế.
        design_img: ảnh đầu vào
        src_pts, dst_pts: các điểm điều khiển (control points)
        output_size: (W, H) của ảnh kết quả
        """
        if output_size is None:
            output_size = (design_img.shape[1], design_img.shape[0])
            
        shape = (output_size[1], output_size[0]) # (H, W)
        map_x, map_y = self.get_remap(np.array(src_pts), np.array(dst_pts), shape)
        
        return cv2.remap(design_img, map_x, map_y, 
                         interpolation=cv2.INTER_LANCZOS4, 
                         borderMode=cv2.BORDER_CONSTANT, 
                         borderValue=(0,0,0,0))

# Singleton instance
tps_cache = TPSRemapCache()
