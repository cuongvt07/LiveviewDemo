"""
Runtime render device config.
Default reads from `RENDER_DEVICE`, but can switch CPU/GPU while service is running.
"""

import os

import cv2
import numpy as np

_DEFAULT_DEVICE = os.getenv("RENDER_DEVICE", "cpu").lower()
RENDER_DEVICE = _DEFAULT_DEVICE if _DEFAULT_DEVICE in ("cpu", "gpu") else "cpu"

def _verify_gpu_context() -> bool:
    """Test GPU thực sự hoạt động — không chỉ check flag."""
    try:
        test = cv2.UMat(np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.GaussianBlur(test, (3, 3), 0).get()
        return True
    except Exception as e:
        print(f"[GPU] Context verification failed: {e}")
        return False


def getRenderDevice() -> str:
    return RENDER_DEVICE


def setRenderDevice(device: str) -> bool:
    """
    Set render device at runtime.
    Returns True when GPU is truly active, otherwise False.
    """
    global RENDER_DEVICE
    target = (device or "cpu").lower()
    if target not in ("cpu", "gpu"):
        raise ValueError("invalid_render_device")

    if target == "gpu":
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL() and _verify_gpu_context():
                RENDER_DEVICE = "gpu"
                print(f"[GPU] OpenCL verified and active - device: {cv2.ocl.Device.getDefault().name()}")
                return True
            cv2.ocl.setUseOpenCL(False)
        print("[GPU] Verification failed - fallback to CPU")
        RENDER_DEVICE = "cpu"
        return False

    cv2.ocl.setUseOpenCL(False)
    RENDER_DEVICE = "cpu"
    print("[CPU] Rendering on CPU")
    return False


def initGpu() -> bool:
    """Init device from env on startup."""
    return setRenderDevice(RENDER_DEVICE)


def isGpuEnabled() -> bool:
    """GPU is enabled only if selected device is gpu and OpenCL is active."""
    return RENDER_DEVICE == "gpu" and cv2.ocl.useOpenCL()


def toGpu(img):
    """Convert numpy array to UMat when GPU is enabled."""
    if isGpuEnabled() and not isinstance(img, cv2.UMat):
        return cv2.UMat(img)
    return img


def toCpu(img):
    """Convert UMat back to numpy array."""
    if isinstance(img, cv2.UMat):
        return img.get()
    return img
