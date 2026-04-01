from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from .cosine_grid import cosine_spacing


@dataclass
class MeshPoint:
    u: float  # [0,1]
    v: float  # [0,1]
    locked: bool = False


@dataclass
class MugMeshConfig:
    cols: int = 12
    rows: int = 8
    spacing: Literal["cosine", "uniform"] = "cosine"
    tension: float = 0.35
    corner_blend_radius: float = 0.08
    symmetry_lock: bool = True
    max_displacement: float = 0.15
    points: list[MeshPoint] = field(default_factory=list)

    def __post_init__(self):
        if not self.points:
            self.points = self._build_default_points()

    def _build_default_points(self) -> list[MeshPoint]:
        u_arr = cosine_spacing(self.cols) if self.spacing == "cosine" else tuple(i / self.cols for i in range(self.cols + 1))
        v_arr = cosine_spacing(self.rows) if self.spacing == "cosine" else tuple(i / self.rows for i in range(self.rows + 1))
        pts: list[MeshPoint] = []
        for vi, v in enumerate(v_arr):
            for ui, u in enumerate(u_arr):
                is_corner = (ui in (0, len(u_arr) - 1) and vi in (0, len(v_arr) - 1))
                pts.append(MeshPoint(u=float(u), v=float(v), locked=is_corner))
        return pts

    def to_numpy(self) -> np.ndarray:
        arr = np.array([[p.u, p.v] for p in self.points], dtype=np.float64)
        return arr.reshape(self.rows + 1, self.cols + 1, 2)

    def move_point(self, idx: int, new_u: float, new_v: float) -> None:
        pt = self.points[idx]
        if pt.locked:
            return

        u_arr = cosine_spacing(self.cols) if self.spacing == "cosine" else tuple(i / self.cols for i in range(self.cols + 1))
        v_arr = cosine_spacing(self.rows) if self.spacing == "cosine" else tuple(i / self.rows for i in range(self.rows + 1))
        orig_u = u_arr[idx % (self.cols + 1)]
        orig_v = v_arr[idx // (self.cols + 1)]

        du = max(-self.max_displacement, min(self.max_displacement, new_u - orig_u))
        dv = max(-self.max_displacement, min(self.max_displacement, new_v - orig_v))

        pt.u = float(min(1.0, max(0.0, orig_u + du)))
        pt.v = float(min(1.0, max(0.0, orig_v + dv)))

        if self.symmetry_lock:
            row = idx // (self.cols + 1)
            col = idx % (self.cols + 1)
            mirror_col = self.cols - col
            mirror_idx = row * (self.cols + 1) + mirror_col
            if mirror_idx != idx and not self.points[mirror_idx].locked:
                self.points[mirror_idx].u = float(1.0 - pt.u)
                self.points[mirror_idx].v = pt.v
