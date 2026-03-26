# app/pipeline/warp.py

import cv2
import numpy as np


def cylinder_warp(
    design: np.ndarray,
    print_area: dict,
    output_size: tuple[int, int],
    theta_max_deg: float = 52.0,
    smile: float = 0.15,
    pitch: float = 0.0,
    design_scale: float = 1.0,
    design_offset_x: float = 0.0,
    design_offset_y: float = 0.0
) -> np.ndarray:
    '''
    Unified Cylindrical + Perspective Warp trong 1 bước.
    Đảm bảo 4 điểm góc print_area khớp 100%.
    Tạo smile curve + foreshortening ở viền (design bị tụ nhỏ ở mép cốc).

    Args:
        foreshorten: 0.0 = không tụ viền, 1.0 = tụ viền vật lý đúng cos(theta)
    '''
    W_out, H_out = output_size
    dh, dw = design.shape[:2]

    theta_max = np.deg2rad(theta_max_deg)
    sin_tmax = np.sin(theta_max)
    cos_tmax = np.cos(theta_max)

    # Góc canonical [-1, 1]
    src_canonical = np.float32([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    pa = print_area
    dst_screen = np.float32([pa['top_left'], pa['top_right'], pa['bottom_right'], pa['bottom_left']])

    # Homography [canonical] -> [screen]
    H, _ = cv2.findHomography(src_canonical, dst_screen)
    H_inv = np.linalg.inv(H)

    # Toạ độ canvas
    grid_x, grid_y = np.meshgrid(np.arange(W_out), np.arange(H_out))
    pts_screen = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.float32).reshape(-1, 1, 2)

    # Nghịch đảo về canonical
    pts_canon = cv2.perspectiveTransform(pts_screen, H_inv).reshape(H_out, W_out, 2)
    x_proj = pts_canon[:, :, 0]  # [-1, 1]
    y_proj = pts_canon[:, :, 1]  # [-1, 1]

    # --- Cylinder projection ---
    # Toạ độ canvas để tính height/radius thực tế
    # H/R ratio ảnh hưởng đến độ "smile" của curve:
    # Mug cao (tumbler) cần curve ít hơn mug thấp (espresso) nếu curve cố định theo pixel.
    # Tuy nhiên, smile curve vật lý phụ thuộc vào góc nhìn và tỷ lệ H/R.
    # curve_adaptive = curve * (H / R) - ở đây ta xấp xỉ bằng cách dùng height của print_area.
    
    pa = print_area
    h_pixel = np.linalg.norm(np.array(pa['bottom_left']) - np.array(pa['top_left']))
    w_pixel = np.linalg.norm(np.array(pa['top_right']) - np.array(pa['top_left']))
    hr_ratio = h_pixel / (w_pixel + 1e-6)
    
    # Differential curve logic: 
    # y_proj=-1 (top). factor = (0.5 - 0) = 0.5. 
    # y_proj=1 (bot). factor = (0.5 - 1) = -0.5.
    curve_v = smile + (-y_proj / 2.0) * (pitch / 100.0) * hr_ratio * 0.15
    curve_adaptive = curve_v 

    # Map x_proj (linear) -> theta trên bề mặt trụ (non-linear)
    sin_theta = np.clip(x_proj * sin_tmax, -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    cos_theta = np.cos(theta)

    # 1. Base U coordination mapping (Cylinder θ -> design U [-1, 1])
    u_base = theta / theta_max

    # 2. Design Layer Transformations (X = Wrapping, Y = Translating)
    # design_scale: 1.0 = fit exactly. 
    # Offset is normalized [0, 1] relative to print area.
    
    # Scale & Offset U (X) with 360° wrap
    # We use (u_base + 1)/2 to go to [0,1], transform, then wrap with % 1.0
    u_norm = (u_base + 1.0) * 0.5
    u_transformed = (u_norm - 0.5) / design_scale + 0.5 - (design_offset_x / design_scale)
    u_final = u_transformed % 1.0
    
    # Scale & Offset V (Y) - NO Wrap
    v_norm = (y_proj + 1.0) * 0.5
    v_transformed = (v_norm - 0.5) / design_scale + 0.5 - (design_offset_y / design_scale)
    v_canon = v_transformed * 2.0 - 1.0

    # 3. Final V mapping (including smile curve)
    # The curve depends on the surface geometry (cos_theta), so we apply it to the transformed v
    v_final = v_canon - curve_adaptive * (cos_theta - cos_tmax)

    # 4. Scale (u_final, v_final) -> pixel trên design
    map_x = (u_final * (dw - 1)).astype(np.float32)
    map_y = ((v_final + 1.0) * 0.5 * (dh - 1)).astype(np.float32)

    # Mask vùng hợp lệ
    mask = ((np.abs(x_proj) <= 1.0) & (np.abs(y_proj) <= 1.2)
            & (map_x >= 0) & (map_x <= dw-1)
            & (map_y >= 0) & (map_y <= dh-1))

    warped = np.zeros((H_out, W_out, 4), dtype=design.dtype)
    cv2.remap(design, map_x, map_y, cv2.INTER_LANCZOS4, dst=warped, borderMode=cv2.BORDER_TRANSPARENT)
    warped[~mask] = 0
    return warped


def perspective_warp(
    design: np.ndarray,
    print_area: dict,
    output_size: tuple[int, int]
) -> np.ndarray:
    '''
    Warp design phẳng vào đúng vùng print_area bằng perspective transform.

    Args:
        design:      BGRA ndarray, bất kỳ kích thước nào
        print_area:  dict với top_left, top_right, bottom_right, bottom_left
                     — tọa độ pixel trên ảnh mockup ở output_size
        output_size: (W, H) của ảnh output

    Returns:
        BGRA ndarray shape (H, W, 4) — warped design trên nền trong suốt
    '''
    W, H = output_size
    dh, dw = design.shape[:2]

    # 4 góc của design gốc (src)
    src = np.float32([
        [0,    0   ],   # top-left
        [dw-1, 0   ],   # top-right
        [dw-1, dh-1],   # bottom-right
        [0,    dh-1],   # bottom-left
    ])

    # 4 góc đích trên mockup (dst)
    pa = print_area
    dst = np.float32([
        pa['top_left'],
        pa['top_right'],
        pa['bottom_right'],
        pa['bottom_left'],
    ])

    M = cv2.getPerspectiveTransform(src, dst)

    # Tạo nền trong suốt (BGRA zeros)
    transparent = np.zeros((H, W, 4), dtype=design.dtype)

    # Warp design lên canvas
    warped = cv2.warpPerspective(
        design, M, (W, H),
        dst=transparent,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
def tps_warp(
    design: np.ndarray,
    print_area: dict,
    output_size: tuple[int, int],
    normal_map: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Warp design using Thin Plate Spline (TPS).
    Tính Jacobian xấp xỉ để reproject Normal Map.
    
    Returns: (warped_design, warped_normal)
    '''
    W, H = output_size
    dh, dw = design.shape[:2]
    
    # 1. Các điểm khống chế (anchors)
    src_pts = [[0, 0], [dw-1, 0], [dw-1, dh-1], [0, dh-1]]
    pa = print_area
    dst_corners = np.float32([pa['top_left'], pa['top_right'], pa['bottom_right'], pa['bottom_left']])
    
    M_inv = cv2.getPerspectiveTransform(dst_corners, np.float32(src_pts))
    dst_pts = list(dst_corners)
    
    custom_mask = pa.get('mask_points', [])
    if custom_mask:
        mask_np = np.array(custom_mask, dtype=np.float32).reshape(-1, 1, 2)
        mask_in_design = cv2.perspectiveTransform(mask_np, M_inv).reshape(-1, 2)
        for i in range(len(custom_mask)):
            dst_pts.append(custom_mask[i])
            src_pts.append(mask_in_design[i])

    # 2. Setup TPS: Mapped from DST (mockup) to SRC (design) for remap
    tps = cv2.createThinPlateSplineShapeTransformer()
    s = np.array(src_pts, dtype=np.float32).reshape(1, -1, 2)
    d = np.array(dst_pts, dtype=np.float32).reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    tps.estimateTransformation(d, s, matches)
    
    # 3. Generate Remap Grids
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts_mockup = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).reshape(-1, 1, 2)
    pts_design = tps.applyTransformation(pts_mockup)[1].reshape(H, W, 2)
    
    map_x = pts_design[:, :, 0]
    map_y = pts_design[:, :, 1]
    
    # 4. Warp Design
    warped_design = cv2.remap(design, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # 5. Jacobian & Normal Reprojection
    warped_normal = None
    if normal_map is not None:
        # Resize normal map to output size then warp values? 
        # No, normal map is typically in texture space of the design or object.
        # Here we assume normal_map aligns with design.
        nm_h, nm_w = normal_map.shape[:2]
        warped_normal = cv2.remap(normal_map, map_x * (nm_w/dw), map_y * (nm_h/dh), 
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Finite difference to get Jacobian of the mapping mockup -> design
        # J = [[du/dx, du/dy], [dv/dx, dv/dy]]
        du_dx, du_dy = np.gradient(map_x)
        dv_dx, dv_dy = np.gradient(map_y)
        
        # Chúng ta cần xoay normal vector [nx, ny, nz]
        # n_new = normalize(J^T * n)
        # Vì remap dùng map: mockup -> design, Jacobian này là J = d(Design)/d(Mockup)
        # For normal maps, we usually solve for Jacobian for reprojection
        # but for now we provide a standard TPS warp
        warped_normal = tps_cache.apply(normal_map, src_pts, dst_pts, output_size=(W, H))
        # TODO: Apply Jacobian rotation to normal map

    return warped_design if normal_map is None else (warped_design, warped_normal)


def perspective_from_camera_angle(quad_pts, camera_elevation_deg=15.0):
    """
    Tính perspective correction dựa trên góc camera ước lượng.
    camera_elevation_deg: góc camera so với ngang (0° = ngang, 90° = thẳng từ trên)
    """
    pts = np.array(quad_pts).copy().astype(np.float64)
    
    # Tính tâm
    cx, cy = pts.mean(axis=0)
    
    # Chiều cao vùng in (cạnh trái)
    H = np.linalg.norm(pts[3] - pts[0])
    
    # Foreshortening factor từ góc camera
    theta = np.deg2rad(camera_elevation_deg)
    alpha = np.arctan2(H * 0.3, H) # Ước lượng góc bao phủ
    
    top_scale = np.cos(theta + alpha) / np.cos(theta)
    bottom_scale = np.cos(theta - alpha) / np.cos(theta)
    
    # Áp dụng scale
    mid_top = (pts[0] + pts[1]) / 2
    mid_bot = (pts[3] + pts[2]) / 2
    
    pts[0] = mid_top + (pts[0] - mid_top) * top_scale
    pts[1] = mid_top + (pts[1] - mid_top) * top_scale
    pts[2] = mid_bot + (pts[2] - mid_bot) * bottom_scale
    pts[3] = mid_bot + (pts[3] - mid_bot) * bottom_scale
    
    return pts.tolist()


def estimate_smile_from_contour(mockup_gray, roi_bottom):
    """
    Phát hiện ellipse đáy cốc → tính smile_ratio thực tế.
    roi_bottom: [x, y, w, h] vùng đáy cốc
    """
    x, y, w, h = roi_bottom
    roi = mockup_gray[y:y+h, x:x+w]
    
    edges = cv2.Canny(roi, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_ellipse = None
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (ex, ey), (MA, ma), angle = ellipse
            # Lọc: chỉ lấy ellipse ngang (góc gần 0° hoặc 180°)
            if abs(angle) < 25 or abs(angle - 180) < 25:
                if MA > w * 0.4:
                    best_ellipse = ellipse
    
    if best_ellipse is None:
        return 0.15 # fallback default
    
    (ex, ey), (MA, ma), angle = best_ellipse
    smile_ratio = ma / MA if MA > 0 else 0.15
    return float(np.clip(smile_ratio * 0.5, 0.05, 0.40))


def build_perspective_from_params(quad_pts, tilt_deg, rotate_deg, persp_strength):
    """
    tilt_deg:       nghiêng trái/phải (±45°)  → di chuyển top-edge theo X
    rotate_deg:     xoay trong plane (±30°)   → rotate toàn bộ quad
    persp_strength: hội tụ trên/dưới (±50)   → thu hẹp top hoặc bottom
    """
    pts = np.array(quad_pts, dtype=np.float32)
    cx, cy = pts.mean(axis=0)

    # Rotate
    if rotate_deg != 0:
        angle = np.deg2rad(rotate_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # pts = (pts - [cx, cy]) @ R.T + [cx, cy]
        # Equivalent but more explicit:
        pts = np.dot(pts - np.array([cx, cy]), R.T) + np.array([cx, cy])

    # Tilt: di chuyển top 2 điểm theo X
    # pts order is usually TL, TR, BR, BL from _order_points
    # We assume index 0,1 are TOP, 2,3 are BOTTOM
    h = pts[2,1] - pts[0,1]
    tilt_px = np.tan(np.deg2rad(tilt_deg)) * h * 0.5
    pts[0, 0] += tilt_px
    pts[1, 0] += tilt_px

    # Perspective: thu hẹp top (persp > 0) hoặc bottom (persp < 0)
    scale = 1.0 - abs(persp_strength) / 200.0
    if persp_strength > 0:
        mid_top = (pts[0] + pts[1]) / 2
        pts[0] = mid_top + (pts[0] - mid_top) * scale
        pts[1] = mid_top + (pts[1] - mid_top) * scale
    else:
        mid_bot = (pts[2] + pts[3]) / 2
        pts[2] = mid_bot + (pts[2] - mid_bot) * scale
        pts[3] = mid_bot + (pts[3] - mid_bot) * scale

    return pts.tolist()


def compute_cylinder_params(quad_pts, curve_pct, extra_curve_pct=0):
    """
    curve_pct:       0–100, độ cong ngang (theta_max)
    extra_curve_pct: smile curve strength, adaptive theo H/R
    """
    pts = np.array(quad_pts)
    # Distance TL-TR
    W = np.linalg.norm(pts[1] - pts[0])   # width print area
    # Distance TL-BL
    H = np.linalg.norm(pts[3] - pts[0])   # height print area
    
    # Estimate radius from width and wrap angle
    theta_max_rad = np.deg2rad(curve_pct * 0.9)
    R_estimated = W / (2 * np.sin(theta_max_rad) + 1e-8)

    theta_max = curve_pct * 0.9   # Deg for UI or rad? User said "float(theta_max)" from rad below
    foreshorten = 0.3 + curve_pct / 200.0     # tăng theo độ cong

    # Adaptive smile curve
    h_r_ratio = H / (R_estimated + 1e-8)
    curve_adaptive = (extra_curve_pct / 100.0) * h_r_ratio * 0.15

    return {
        "mode":         "cylinder",
        "theta_max":    float(theta_max * 1.0), # Assuming UI wants deg if it uses theta_max_deg
        "theta_max_rad": float(theta_max_rad),
        "foreshorten":  float(foreshorten),
        "curve":        float(curve_adaptive),
    }


def fold_map_to_tps_points(fold_map: np.ndarray, quad_pts, n_points=16):
    """
    fold_map: H×W float array, giá trị [-1, 1] (admin vẽ brush)
              dương = đẩy ra ngoài (fold lên phía cam)
              âm = kéo vào trong (fold xuống phía sau)
    """
    H, W = fold_map.shape
    
    # Sample N điểm trên grid đều trong print area
    # Chúng ta cần implement _sample_grid_in_quad
    src_pts = _sample_grid_in_quad(quad_pts, n=n_points)

    dst_pts = []
    # Gradient of fold map for displacement direction
    grad_y, grad_x = np.gradient(fold_map.astype(np.float32))

    for (x, y) in src_pts:
        xi, yi = int(x), int(y)
        if 0 <= yi < H and 0 <= xi < W:
            fold_val = fold_map[yi, xi]
            # Fold displacement: theo hướng gradient
            gx = grad_x[yi, xi]
            gy = grad_y[yi, xi]
            
            # Displacement trong image space
            # Phụ thuộc vào gradient (độ dốc nếp gấp)
            dx = -gy * fold_val * 15   # scale factor
            dy = gx * fold_val * 15
            dst_pts.append([x + dx, y + dy])
        else:
            dst_pts.append([x, y])

    return np.array(src_pts).tolist(), np.array(dst_pts).tolist()

def _sample_grid_in_quad(quad_pts, n=16):
    """
    Sample points inside a quadrilateral.
    n=16 -> 4x4 grid.
    """
    pts = np.array(quad_pts, dtype=np.float32)
    p0, p1, p2, p3 = pts # TL, TR, BR, BL
    
    grid_size = int(np.sqrt(n))
    samples = []
    for i in range(grid_size):
        for j in range(grid_size):
            u = i / (grid_size - 1)
            v = j / (grid_size - 1)
            p = (1-u)*(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3
            samples.append(p.tolist())
    return samples
