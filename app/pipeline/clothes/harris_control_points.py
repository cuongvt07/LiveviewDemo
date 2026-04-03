import cv2
import numpy as np

def extract_wrinkle_control_points(
    shirt_img: np.ndarray,
    mask: np.ndarray,
    n_points: int = 80,
    strength: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect wrinkle control points từ ảnh áo.
    Kết hợp Harris Corner (nếp gấp thật) + Dense Grid (coverage đều).

    Harris Corner: bắt điểm gấp cứng — wrinkle peaks
    Dense Grid:    bổ sung vùng phẳng để TPS interpolate smooth

    Returns:
        src_pts: [0,1] normalized coords trên design (Nx2)
        dst_pts: pixel coords trên ảnh áo đã displaced (Nx2)
    """
    H, W = shirt_img.shape[:2]
    gray = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    gray_f = gray_masked.astype(np.float32)

    # Harris corners
    harris = cv2.cornerHarris(gray_f, blockSize=5, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    corner_mask = (harris > harris.max() * 0.02) & (mask > 128)
    corner_ys, corner_xs = np.where(corner_mask)

    if len(corner_xs) > n_points // 2:
        idx = np.random.choice(len(corner_xs), n_points // 2, replace=False)
        corner_xs, corner_ys = corner_xs[idx], corner_ys[idx]

    # Dense grid
    coords = cv2.findNonZero(mask)
    x0, y0, bw, bh = cv2.boundingRect(coords)
    gn = int(np.sqrt(n_points // 2))
    gx = np.linspace(x0 + bw * 0.05, x0 + bw * 0.95, gn)
    gy = np.linspace(y0 + bh * 0.05, y0 + bh * 0.95, gn)
    gxx, gyy = np.meshgrid(gx, gy)
    grid_xs = gxx.ravel().astype(int)
    grid_ys = gyy.ravel().astype(int)

    all_xs = np.clip(np.concatenate([corner_xs, grid_xs]), 0, W - 1)
    all_ys = np.clip(np.concatenate([corner_ys, grid_ys]), 0, H - 1)
    valid = mask[all_ys, all_xs] > 128
    all_xs, all_ys = all_xs[valid], all_ys[valid]

    # Gradient displacement
    gx_g = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=5)
    gy_g = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=5)
    gx_g = cv2.GaussianBlur(gx_g, (21, 21), sigmaX=7)
    gy_g = cv2.GaussianBlur(gy_g, (21, 21), sigmaX=7)
    gx_n = gx_g / (np.abs(gx_g).max() + 1e-6)
    gy_n = gy_g / (np.abs(gy_g).max() + 1e-6)

    dst_xs = np.clip(all_xs + gx_n[all_ys, all_xs] * strength, 0, W - 1)
    dst_ys = np.clip(all_ys + gy_n[all_ys, all_xs] * strength, 0, H - 1)

    src_xs = (all_xs - x0) / bw
    src_ys = (all_ys - y0) / bh

    return (np.stack([src_xs, src_ys], axis=1).astype(np.float32),
            np.stack([dst_xs, dst_ys], axis=1).astype(np.float32))
