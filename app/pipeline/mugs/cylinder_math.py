import numpy as np


BASE_CENTER_BAND = 0.30
MIN_CENTER_BAND = 0.10
MAX_CENTER_BAND = 0.70
BASE_EDGE_ROLL_START = 0.55


def _compute_center_band(center_focus_width: float) -> float:
    width_strength = float(np.clip(center_focus_width, -1.0, 1.0))
    if width_strength >= 0.0:
        return BASE_CENTER_BAND + (MAX_CENTER_BAND - BASE_CENTER_BAND) * width_strength
    return BASE_CENTER_BAND + (BASE_CENTER_BAND - MIN_CENTER_BAND) * width_strength


def apply_center_focus_width(radius: np.ndarray, center_focus_width: float) -> np.ndarray:
    width_strength = float(np.clip(center_focus_width, -1.0, 1.0))
    if abs(width_strength) <= 1e-8:
        return radius

    source_band = BASE_CENTER_BAND
    target_band = _compute_center_band(center_focus_width)

    inner_ratio = np.clip(radius / max(source_band, 1e-8), 0.0, 1.0)
    inner_mapped = target_band * inner_ratio

    outer_ratio = np.clip((radius - source_band) / max(1.0 - source_band, 1e-8), 0.0, 1.0)
    outer_mapped = target_band + (1.0 - target_band) * outer_ratio

    return np.where(radius <= source_band, inner_mapped, outer_mapped)


def apply_edge_roll(
    radius: np.ndarray,
    edge_squeeze: float,
    squeeze_power: float,
    center_focus_width: float,
) -> np.ndarray:
    blend = float(np.clip(edge_squeeze, 0.0, 1.0))
    if blend <= 1e-8:
        return radius

    protected_center_band = _compute_center_band(center_focus_width)
    edge_start = min(max(BASE_EDGE_ROLL_START, protected_center_band), 0.95)
    if edge_start >= 1.0 - 1e-8:
        return radius

    power = max(float(squeeze_power), 1.0)
    progress = np.clip((radius - edge_start) / max(1.0 - edge_start, 1e-8), 0.0, 1.0)
    rolled_progress = np.power(progress, 1.0 / power)
    mapped_progress = progress + (rolled_progress - progress) * blend
    edge_mapped = edge_start + (1.0 - edge_start) * mapped_progress
    return np.where(radius <= edge_start, radius, edge_mapped)


def summarize_horizontal_squeeze(
    theta_max_deg: float,
    edge_squeeze: float = 0.0,
    squeeze_power: float = 2.0,
    center_focus_width: float = 0.0,
) -> dict:
    samples = np.array([0.1, 0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    theta_max = np.radians(theta_max_deg)
    mapped = apply_horizontal_squeeze(
        samples * theta_max,
        theta_max,
        edge_squeeze=edge_squeeze,
        squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
    )
    sample_out = np.round(mapped.astype(np.float32), 4).tolist()
    sample_delta = np.round(mapped - samples, 4).tolist()

    has_edge = edge_squeeze > 1e-8
    has_width = abs(center_focus_width) > 1e-8
    inactive_reason = None
    mode = "edge_and_width_active"
    if not has_edge and not has_width:
        inactive_reason = "edge_and_center_zero"
        mode = "inactive"
    elif not has_edge:
        mode = "width_only_active"
    elif not has_width:
        mode = "edge_only_active"

    return {
        "active": has_edge or has_width,
        "inactive_reason": inactive_reason,
        "mode": mode,
        "sample_in": np.round(samples, 4).tolist(),
        "sample_out": sample_out,
        "sample_delta": sample_delta,
    }


def apply_horizontal_squeeze(
    theta: np.ndarray,
    theta_max: float,
    edge_squeeze: float = 0.0,
    squeeze_power: float = 2.0,
    center_focus_width: float = 0.0,
) -> np.ndarray:
    if abs(theta_max) < 1e-8:
        return np.zeros_like(theta, dtype=np.float32)

    t = np.clip(theta / theta_max, -1.0, 1.0)
    blend = float(np.clip(edge_squeeze, 0.0, 1.0))
    width = float(np.clip(center_focus_width, -1.0, 1.0))
    if blend <= 1e-8 and abs(width) <= 1e-8:
        return t

    radius = np.abs(t)
    sign = np.sign(t)
    width_adjusted_radius = apply_center_focus_width(radius, width)
    mapped_radius = apply_edge_roll(
        width_adjusted_radius,
        edge_squeeze=blend,
        squeeze_power=squeeze_power,
        center_focus_width=width,
    )
    return np.clip(sign * mapped_radius, -1.0, 1.0)


def compute_uv_cylindrical(
    X_proj: np.ndarray,
    Y_proj: np.ndarray,
    theta_max_deg: float = 52.0,
    pitch: float = 0.0,
    hr_ratio: float = 1.0,
    smile_base: float = 0.08,
    curve_top: float | None = None,
    curve_bottom: float | None = None,
    edge_squeeze: float = 0.0,
    squeeze_power: float = 2.0,
    center_focus_width: float = 0.0,
    clamp_v: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cylindrical inverse mapping.

    Supports two vertical curve modes:
    1) Legacy mode via smile_base + pitch.
    2) Explicit editor mode via curve_top/curve_bottom percentages.
    """
    # --- NaN Sanitization: chặn NaN/Inf lan truyền từ homography hoặc input ---
    X_proj = np.nan_to_num(X_proj, nan=0.0, posinf=1.0, neginf=-1.0)
    Y_proj = np.nan_to_num(Y_proj, nan=0.0, posinf=1.0, neginf=-1.0)

    theta_max = np.radians(theta_max_deg)
    if abs(theta_max) < 1e-8:
        U = (X_proj + 1.0) / 2.0
        V_final = (Y_proj + 1.0) / 2.0
        if clamp_v:
            V_final = np.clip(V_final, 0.0, 1.0)
        return np.clip(U, 0.0, 1.0), V_final

    sin_theta = np.clip(X_proj * np.sin(theta_max), -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    t_final = apply_horizontal_squeeze(
        theta,
        theta_max,
        edge_squeeze=edge_squeeze,
        squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
    )
    U = (t_final + 1.0) / 2.0

    V_canon = (Y_proj + 1.0) / 2.0

    if curve_top is not None and curve_bottom is not None:
        cos_displacement = np.cos(theta) - np.cos(theta_max)
        k = hr_ratio * 0.15 * cos_displacement

        top_curve = curve_top / 100.0
        bottom_curve = curve_bottom / 100.0

        # The editor preview uses a forward warp in source canonical space:
        #   Y_out = S - S * curve(S) * k
        # where curve(S) is linearly blended from top/bottom controls.
        # To make render output match the editor grid, invert that quadratic
        # exactly instead of approximating with the output-space Y coordinate.
        curve_mid = 0.5 * (top_curve + bottom_curve)
        curve_slope = 0.5 * (bottom_curve - top_curve)

        quad_a = -k * curve_slope
        quad_b = 1.0 - k * curve_mid

        small_a = np.abs(quad_a) < 1e-8
        source_y = np.empty_like(Y_proj)

        safe_b = np.where(np.abs(quad_b) < 1e-8, 1.0, quad_b)
        source_y[small_a] = Y_proj[small_a] / safe_b[small_a]

        if np.any(~small_a):
            a = quad_a[~small_a]
            b = quad_b[~small_a]
            y = Y_proj[~small_a]
            disc = np.maximum(b * b + 4.0 * a * y, 0.0)
            sqrt_disc = np.sqrt(disc)
            # Bảo vệ chia cho 0: nếu a quá nhỏ sau filter vẫn lọt, fallback về y
            safe_2a = np.where(np.abs(a) < 1e-12, 1.0, 2.0 * a)
            root1 = (-b + sqrt_disc) / safe_2a
            root2 = (-b - sqrt_disc) / safe_2a

            # Choose the branch that stays closest to the output Y coordinate.
            use_root1 = np.abs(root1 - y) <= np.abs(root2 - y)
            source_y[~small_a] = np.where(use_root1, root1, root2)

        # Chặn NaN còn sót sau phép chia/sqrt
        source_y = np.nan_to_num(source_y, nan=0.0, posinf=1.0, neginf=-1.0)
        V_final = (source_y + 1.0) / 2.0
    else:
        factor = -Y_proj / 2.0
        curve_v = smile_base + factor * (pitch / 100.0) * hr_ratio * 0.15
        V_final = V_canon - curve_v * (np.cos(theta) - np.cos(theta_max))

    # Sanitize output cuối cùng
    U = np.nan_to_num(U, nan=0.5, posinf=1.0, neginf=0.0)
    V_final = np.nan_to_num(V_final, nan=0.5, posinf=1.0, neginf=0.0)
    if clamp_v:
        V_final = np.clip(V_final, 0.0, 1.0)
    return np.clip(U, 0.0, 1.0), V_final
