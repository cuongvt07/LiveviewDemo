import numpy as np


def compute_uv_cylindrical(
    X_proj: np.ndarray,
    Y_proj: np.ndarray,
    theta_max_deg: float = 52.0,
    pitch: float = 0.0,
    hr_ratio: float = 1.0,
    smile_base: float = 0.08,
    curve_top: float | None = None,
    curve_bottom: float | None = None,
    clamp_v: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cylindrical inverse mapping.

    Supports two vertical curve modes:
    1) Legacy mode via smile_base + pitch.
    2) Explicit editor mode via curve_top/curve_bottom percentages.
    """
    theta_max = np.radians(theta_max_deg)
    if abs(theta_max) < 1e-8:
        U = (X_proj + 1.0) / 2.0
        V_final = (Y_proj + 1.0) / 2.0
        if clamp_v:
            V_final = np.clip(V_final, 0.0, 1.0)
        return np.clip(U, 0.0, 1.0), V_final

    sin_theta = np.clip(X_proj * np.sin(theta_max), -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    U = (theta / theta_max + 1.0) / 2.0

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
            root1 = (-b + sqrt_disc) / (2.0 * a)
            root2 = (-b - sqrt_disc) / (2.0 * a)

            # Choose the branch that stays closest to the output Y coordinate.
            use_root1 = np.abs(root1 - y) <= np.abs(root2 - y)
            source_y[~small_a] = np.where(use_root1, root1, root2)

        V_final = (source_y + 1.0) / 2.0
    else:
        factor = -Y_proj / 2.0
        curve_v = smile_base + factor * (pitch / 100.0) * hr_ratio * 0.15
        V_final = V_canon - curve_v * (np.cos(theta) - np.cos(theta_max))

    if clamp_v:
        V_final = np.clip(V_final, 0.0, 1.0)
    return np.clip(U, 0.0, 1.0), V_final
