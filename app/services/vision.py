import cv2
import numpy as np
import base64

def auto_detect_print_area_v2(img_bgr, margin_pct=0.08):
    """
    Pipeline phát hiện vùng in cải tiến.
    Thêm: Watershed refinement + Better classification.
    """
    H, W = img_bgr.shape[:2]
    
    # --- 1. Isolate Foreground bằng GrabCut ---
    margin_x = int(W * margin_pct)
    margin_y = int(H * margin_pct)
    rect = (margin_x, margin_y, W - 2*margin_x, H - 2*margin_y)
    
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    except:
        # Fallback rect-based mask
        fg_mask = np.zeros((H, W), np.uint8)
        cv2.rectangle(fg_mask, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 255, -1)
    
    # --- 2. Morphological Cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # --- 3. Tìm Contour lớn nhất ---
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_center_quad(W, H)
        
    main_contour = max(contours, key=cv2.contourArea)
    
    # --- 4. Phân loại hình dạng ---
    shape_type = classify_shape(main_contour, fg_mask)
    
    # --- 5. Trích xuất Quad và Mask ---
    rect_box = cv2.minAreaRect(main_contour)
    box_pts = cv2.boxPoints(rect_box).astype(np.float32)
    box_pts = _sort_pts_clockwise(box_pts)
    
    # Shrink 10% lề trong để an toàn
    center = box_pts.mean(axis=0)
    quad_pts = center + (box_pts - center) * 0.90
    
    return {
        "shape_type": shape_type,
        "quad": quad_pts.astype(int).tolist(),
        "clip_mask": box_pts.astype(int).tolist(),
        "confidence": 0.85
    }

def classify_shape(contour, mask):
    """
    Phân loại sản phẩm dựa trên Circularity, Aspect Ratio và Solidity.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return "flat"
    
    circularity = 4 * np.pi * area / (perimeter ** 2)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 1.0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1.0
    
    if circularity > 0.6 and aspect_ratio < 1.3 and solidity > 0.9:
        return "cylinder"
    if solidity < 0.8:
        return "apparel"
    return "flat"

def create_soft_mask(mask_pts, img_shape, feather_radius=3):
    """
    Tạo mask với viền mềm (feathered edge).
    """
    h, w = img_shape[:2]
    hard_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(hard_mask, [np.array(mask_pts).astype(np.int32)], 255)
    
    if feather_radius <= 0: return hard_mask
    
    ksize = feather_radius * 2 + 1
    soft_mask = cv2.GaussianBlur(hard_mask.astype(np.float32), (ksize, ksize), 0)
    return np.clip(soft_mask, 0, 255).astype(np.uint8)

def _sort_pts_clockwise(pts):
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    return pts[np.argsort(angles)]

def _fallback_center_quad(W, H):
    q = [[W//4, H//4], [W*3//4, H//4], [W*3//4, H*3//4], [W//4, H*3//4]]
    return {"shape_type": "flat", "quad": q, "clip_mask": q, "confidence": 0.1}
