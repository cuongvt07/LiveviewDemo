import unittest

import cv2
import numpy as np

from app.pipeline.mugs.cylindrical_warp import _apply_mockup_curve_correction


class MugCurveCorrectionTests(unittest.TestCase):
    def test_curve_correction_pulls_boundaries_toward_detected_edges(self) -> None:
        mockup = np.zeros((240, 240, 3), dtype=np.uint8)
        x = np.linspace(50, 190, 80, dtype=np.float32)
        true_top = 60.0 + 8.0 * np.square((x - 120.0) / 70.0)
        true_bottom = 180.0 - 6.0 * np.square((x - 120.0) / 70.0)

        top_pts = np.stack([x, true_top], axis=1).astype(np.int32).reshape(-1, 1, 2)
        bottom_pts = np.stack([x, true_bottom], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mockup, [top_pts], False, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(mockup, [bottom_pts], False, (255, 255, 255), 2, cv2.LINE_AA)

        base_top = np.full_like(x, 60.0)
        base_bottom = np.full_like(x, 180.0)
        dst_corners = np.float32([[50, 60], [190, 60], [190, 180], [50, 180]])

        corrected_top, corrected_bottom = _apply_mockup_curve_correction(
            mockup_bgr=mockup,
            dst_corners=dst_corners,
            px_top=x,
            py_top=base_top,
            px_bottom=x,
            py_bottom=base_bottom,
            alpha=0.8,
            snap_threshold_px=2.5,
        )

        base_top_err = float(np.mean(np.abs(base_top - true_top)))
        corrected_top_err = float(np.mean(np.abs(corrected_top - true_top)))
        base_bottom_err = float(np.mean(np.abs(base_bottom - true_bottom)))
        corrected_bottom_err = float(np.mean(np.abs(corrected_bottom - true_bottom)))

        self.assertLess(corrected_top_err, base_top_err)
        self.assertLess(corrected_bottom_err, base_bottom_err)


if __name__ == "__main__":
    unittest.main()
