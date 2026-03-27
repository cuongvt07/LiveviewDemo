import React, { useState, useRef, useEffect, useCallback, useLayoutEffect } from 'react';
import { MugCurvePreview } from './MugCurvePreview';
import { WarpGridPreview } from './WarpGridPreview';

interface Point { x: number; y: number; }
interface PrintAreaEditorProps {
  imageUrl: string;
  designFile?: File | null;
  warpConfig: any;
  initialPrintArea?: any;
  onCoordinatesChange: (data: any) => void;
  onConfigChange: (config: any) => void;
  onLockedSnapshotChange?: (snapshot: { printArea: any; warpConfig: any } | null) => void;
  onLiveSnapshotChange?: (snapshot: { printArea: any; warpConfig: any } | null) => void;
  productTypeProp?: string;
  onProductTypeChange?: (val: string) => void;
}

// ─── Math helpers ────────────────────────────────────────────────────────────

function applyCalibration(base: Point[], tilt: number, rotate: number, perspective: number): Point[] {
  let pts = base.map(p => ({ ...p }));
  const cx = pts.reduce((a, p) => a + p.x, 0) / 4;
  const cy = pts.reduce((a, p) => a + p.y, 0) / 4;

  if (rotate !== 0) {
    const a = (rotate * Math.PI) / 180;
    const cos = Math.cos(a), sin = Math.sin(a);
    pts = pts.map(p => ({
      x: cx + (p.x - cx) * cos - (p.y - cy) * sin,
      y: cy + (p.x - cx) * sin + (p.y - cy) * cos,
    }));
  }

  const h = pts[3].y - pts[0].y;
  const tiltOff = Math.tan((tilt * Math.PI) / 180) * h * 0.5;
  pts[0].x += tiltOff; pts[1].x += tiltOff;

  const scale = 1.0 - Math.abs(perspective) / 200.0;
  if (perspective > 0) {
    const mx = (pts[0].x + pts[1].x) / 2;
    pts[0].x = mx + (pts[0].x - mx) * scale;
    pts[1].x = mx + (pts[1].x - mx) * scale;
  } else if (perspective < 0) {
    const mx = (pts[2].x + pts[3].x) / 2;
    pts[2].x = mx + (pts[2].x - mx) * scale;
    pts[3].x = mx + (pts[3].x - mx) * scale;
  }
  return pts;
}

// ─── Homography helpers (Parity with OpenCV) ────────────────────────────────

function solve_homography(src: Point[], dst: Point[]): number[] {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x: sx, y: sy } = src[i];
    const { x: dx, y: dy } = dst[i];
    A.push([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx]);
    b.push(dx);
    A.push([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy]);
    b.push(dy);
  }
  const n = 8;
  for (let i = 0; i < n; i++) {
    let max = i;
    for (let j = i + 1; j < n; j++) if (Math.abs(A[j][i]) > Math.abs(A[max][i])) max = j;
    [A[i], A[max]] = [A[max], A[i]];
    [b[i], b[max]] = [b[max], b[i]];
    for (let j = i + 1; j < n; j++) {
      const f = A[j][i] / A[i][i];
      b[j] -= f * b[i];
      for (let k = i; k < n; k++) A[j][k] -= f * A[i][k];
    }
  }
  const h = new Array(8).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += A[i][j] * h[j];
    h[i] = (b[i] - sum) / A[i][i];
  }
  return [...h, 1];
}

function apply_homography(H: number[], u: number, v: number): Point {
  const x = u * 2 - 1, y = v * 2 - 1;
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w
  };
}

function cylinderWarpPoint(u: number, v: number, curvePct: number, curveTop: number, curveBot: number, hPx: number, wPx: number): Point {
  const thetaMaxDeg = curvePct * 0.9;
  if (thetaMaxDeg <= 0.001) return { x: u, y: v };
  const thetaMax = (thetaMaxDeg * Math.PI) / 180;
  const nx = (u - 0.5) * 2;
  const sinThetaMax = Math.sin(thetaMax);
  const sinTheta = Math.max(-1, Math.min(1, nx * sinThetaMax));
  const theta = Math.asin(sinTheta);
  const wx = (theta / thetaMax + 1) * 0.5;

  const hr_ratio = hPx / (wPx + 1e-8);
  const cosDisplacement = Math.cos(theta) - Math.cos(thetaMax);

  const yProj = (v - 0.5) * 2;
  const curveAtPixel = (curveTop / 100) * (1 - v) + (curveBot / 100) * v;
  const sign = -yProj;
  const wy_canon = yProj + sign * curveAtPixel * hr_ratio * 0.15 * cosDisplacement;
  const wy = (wy_canon + 1) * 0.5;

  return { x: Math.max(0, Math.min(1, wx)), y: Math.max(-0.5, Math.min(1.5, wy)) };
}

function parsePoint(input: any): Point | null {
  if (Array.isArray(input) && input.length >= 2) {
    const x = Number(input[0]);
    const y = Number(input[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) return { x, y };
  }
  if (input && typeof input === 'object') {
    const x = Number(input.x);
    const y = Number(input.y);
    if (Number.isFinite(x) && Number.isFinite(y)) return { x, y };
  }
  return null;
}

function toFiniteNumber(input: any, fallback: number): number {
  return typeof input === 'number' && Number.isFinite(input) ? input : fallback;
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function PrintAreaEditor(props: PrintAreaEditorProps) {
  const { imageUrl, designFile, warpConfig, initialPrintArea, onCoordinatesChange, onConfigChange, onLockedSnapshotChange, onLiveSnapshotChange, productTypeProp, onProductTypeChange } = props;
  
  const [basePoints, setBasePoints] = useState<Point[]>([
    { x: 0.22, y: 0.14 }, { x: 0.78, y: 0.14 },
    { x: 0.78, y: 0.86 }, { x: 0.22, y: 0.86 },
  ]);
  const [tilt, setTilt] = useState(toFiniteNumber(warpConfig?.tilt_deg, 0));
  const [rotate, setRotate] = useState(toFiniteNumber(warpConfig?.rotate_deg, 0));
  const [perspective, setPerspective] = useState(toFiniteNumber(warpConfig?.persp_strength, 0));
  const [curvePct, setCurvePct] = useState(
    toFiniteNumber(
      warpConfig?.curve_pct,
      typeof warpConfig?.theta_max_deg === 'number' ? warpConfig.theta_max_deg / 0.9 : 60
    )
  );
  const [curveTop, setCurveTop] = useState(toFiniteNumber(warpConfig?.curve_top, 0));
  const [curveBot, setCurveBot] = useState(toFiniteNumber(warpConfig?.curve_bottom, 0));
  const [designScale, setDesignScale] = useState(toFiniteNumber(warpConfig?.design_scale, 1.0));
  const [designOffsetX, setDesignOffsetX] = useState(toFiniteNumber(warpConfig?.design_offset_x, 0.0));
  const [designOffsetY, setDesignOffsetY] = useState(toFiniteNumber(warpConfig?.design_offset_y, 0.0));

  const [editorMode, setEditorMode] = useState<'CALIBRATE' | 'DESIGN'>(
    warpConfig?.editor_mode === 'DESIGN' || warpConfig?.is_locked === true ? 'DESIGN' : 'CALIBRATE'
  );
  const [featherRadius, setFeatherRadius] = useState(toFiniteNumber(warpConfig?.feather_radius, 3));
  
  const [localProductType, setLocalProductType] = useState('cylinder_ceramic');
  const productType = productTypeProp || localProductType;
  const setProductType = onProductTypeChange || setLocalProductType;

  const [activeMode, setActiveMode] = useState<'calibrate' | 'mesh' | 'mask'>('calibrate');
  const [activeGroup, setActiveGroup] = useState<'geometry' | 'wrap' | 'blend' | 'advanced'>('geometry');
  const [showGrid, setShowGrid] = useState(true);
  const [maskPoints, setMaskPoints] = useState<Point[]>([]);
  const [meshPoints, setMeshPoints] = useState<{ src: Point; dst: Point }[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [previewOverlayUrl, setPreviewOverlayUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [draggingIdx, setDraggingIdx] = useState<{ type: string; id: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [renderDevice, setRenderDevice] = useState<'cpu' | 'gpu'>('cpu');
  const [gpuAvailable, setGpuAvailable] = useState(true);
  const [isSwitchingDevice, setIsSwitchingDevice] = useState(false);
  const didRestoreRef = useRef(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const overlayAlphaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const calibPts = applyCalibration(basePoints, tilt, rotate, perspective);

  const W_px = Math.abs(calibPts[1].x - calibPts[0].x) * (naturalSize.w || 1000);
  const H_px = Math.abs(calibPts[3].y - calibPts[0].y) * (naturalSize.h || 1000);

  const src_canon_coords = [{ x: -1, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }];
  const H_mat = solve_homography(src_canon_coords, calibPts);

  const localSmile = cylinderWarpPoint(0.5, 0, curvePct, curveTop, curveBot, H_px, W_px);
  const smileHandle = apply_homography(H_mat, localSmile.x, localSmile.y);
  const localPitch = cylinderWarpPoint(0.5, 1, curvePct, curveTop, curveBot, H_px, W_px);
  const pitchHandle = apply_homography(H_mat, localPitch.x, localPitch.y);
  const showBackendOverlay = Boolean(previewOverlayUrl);
  const showEditorGridLayer = showGrid && editorMode === 'CALIBRATE';
  const canDragArtwork = Boolean(designFile) && (editorMode === 'DESIGN' || activeMode === 'calibrate');

  useEffect(() => {
    fetch('/v1/render/device')
      .then(r => r.json())
      .then(data => {
        if (data?.device === 'gpu' || data?.device === 'cpu') setRenderDevice(data.device);
        if (typeof data?.opencl_available === 'boolean') setGpuAvailable(data.opencl_available);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!previewOverlayUrl) {
      overlayAlphaCanvasRef.current = null;
      return;
    }

    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      overlayAlphaCanvasRef.current = canvas;
    };
    img.src = previewOverlayUrl;

    return () => {
      cancelled = true;
      overlayAlphaCanvasRef.current = null;
    };
  }, [previewOverlayUrl]);

  useEffect(() => {
    return () => {
      if (previewOverlayUrl) URL.revokeObjectURL(previewOverlayUrl);
    };
  }, [previewOverlayUrl]);

  useEffect(() => {
    if (didRestoreRef.current || naturalSize.w <= 0 || naturalSize.h <= 0) return;

    const toPct = (p: any): Point | null => {
      const point = parsePoint(p);
      if (!point) return null;
      const looksNormalized = Math.abs(point.x) <= 2 && Math.abs(point.y) <= 2;
      if (looksNormalized) return point;
      return { x: point.x / naturalSize.w, y: point.y / naturalSize.h };
    };

    const toPctList = (arr: any, limit = Infinity): Point[] => {
      if (!Array.isArray(arr)) return [];
      const out: Point[] = [];
      for (const raw of arr) {
        const p = toPct(raw);
        if (p) out.push(p);
        if (out.length >= limit) break;
      }
      return out;
    };

    const restoredTilt = toFiniteNumber(warpConfig?.tilt_deg, 0);
    const restoredRotate = toFiniteNumber(warpConfig?.rotate_deg, 0);
    const restoredPerspective = toFiniteNumber(warpConfig?.persp_strength, 0);
    setTilt(restoredTilt);
    setRotate(restoredRotate);
    setPerspective(restoredPerspective);
    setCurvePct(
      toFiniteNumber(
        warpConfig?.curve_pct,
        typeof warpConfig?.theta_max_deg === 'number' ? warpConfig.theta_max_deg / 0.9 : 60
      )
    );
    setCurveTop(toFiniteNumber(warpConfig?.curve_top, 0));
    setCurveBot(toFiniteNumber(warpConfig?.curve_bottom, 0));
    setFeatherRadius(toFiniteNumber(warpConfig?.feather_radius, 3));
    setDesignScale(toFiniteNumber(warpConfig?.design_scale, 1));
    setDesignOffsetX(toFiniteNumber(warpConfig?.design_offset_x, 0));
    setDesignOffsetY(toFiniteNumber(warpConfig?.design_offset_y, 0));

    const mode = warpConfig?.editor_mode === 'DESIGN' || warpConfig?.is_locked === true ? 'DESIGN' : 'CALIBRATE';
    setEditorMode(mode);

    const rawBase = toPctList(initialPrintArea?.base_points_raw, 4);
    const fallbackQuad = toPctList(initialPrintArea?.quad, 4);
    const fallbackCorners = toPctList([
      initialPrintArea?.top_left,
      initialPrintArea?.top_right,
      initialPrintArea?.bottom_right,
      initialPrintArea?.bottom_left,
    ], 4);
    const restoredBase = rawBase.length === 4 ? rawBase : (fallbackQuad.length === 4 ? fallbackQuad : fallbackCorners);
    if (restoredBase.length === 4) {
      setBasePoints(restoredBase);
      if (rawBase.length !== 4) {
        setTilt(0);
        setRotate(0);
        setPerspective(0);
      }
    }

    const restoredMask = toPctList(initialPrintArea?.mask_points);
    if (restoredMask.length > 0) setMaskPoints(restoredMask);

    const srcList = toPctList(initialPrintArea?.mesh_control_src);
    const dstList = toPctList(initialPrintArea?.mesh_control_dst);
    if (srcList.length > 0 && srcList.length === dstList.length) {
      setMeshPoints(srcList.map((src, i) => ({ src, dst: dstList[i] })));
    }

    if (typeof initialPrintArea?.product_type === 'string' && initialPrintArea.product_type.length > 0) {
      setProductType(initialPrintArea.product_type);
    }

    didRestoreRef.current = true;
  }, [initialPrintArea, naturalSize.h, naturalSize.w, setProductType, warpConfig]);

  useLayoutEffect(() => {
    if (!onLiveSnapshotChange) return;
    if (naturalSize.w === 0) {
      onLiveSnapshotChange(null);
      return;
    }

    const toPx = (p: Point) => [Math.round(p.x * naturalSize.w), Math.round(p.y * naturalSize.h)];
    const hr_ratio = H_px / (W_px + 1e-8);
    const smile_api = (curveTop + curveBot) / 2;
    const pitch_api = (curveTop - curveBot);
    const pa = {
      quad: calibPts.map(toPx),
      base_points_raw: basePoints.map(toPx),
      top_left: toPx(calibPts[0]), top_right: toPx(calibPts[1]),
      bottom_right: toPx(calibPts[2]), bottom_left: toPx(calibPts[3]),
      mask_points: maskPoints.length > 2 ? maskPoints.map(toPx) : null,
      mesh_control_src: meshPoints.map(m => toPx(m.src)),
      mesh_control_dst: meshPoints.map(m => toPx(m.dst)),
      product_type: productType,
      camera_elevation: pitch_api,
    };
    const nextWarpConfig = {
      warp_type: meshPoints.length > 0 ? 'tps' : 'cylinder',
      theta_max_deg: curvePct * 0.9,
      curve_pct: curvePct,
      curve: (smile_api / 100) * hr_ratio * 0.15,
      curve_top: curveTop,
      curve_bottom: curveBot,
      tilt_deg: tilt, rotate_deg: rotate, persp_strength: perspective,
      camera_elevation: pitch_api,
      product_type: productType,
      feather_radius: featherRadius,
      design_scale: designScale,
      design_offset_x: designOffsetX,
      design_offset_y: designOffsetY,
      editor_mode: editorMode,
      is_locked: editorMode === 'DESIGN',
    };

    onLiveSnapshotChange({ printArea: pa, warpConfig: nextWarpConfig });
  }, [basePoints, tilt, rotate, perspective, curvePct, curveTop, curveBot, designScale, designOffsetX, designOffsetY, featherRadius, productType, maskPoints, meshPoints, naturalSize, editorMode]);

  useEffect(() => {
    if (naturalSize.w === 0) return;
    const toPx = (p: Point) => [Math.round(p.x * naturalSize.w), Math.round(p.y * naturalSize.h)];
    const hr_ratio = H_px / (W_px + 1e-8);
    const smile_api = (curveTop + curveBot) / 2;
    const pitch_api = (curveTop - curveBot);
    const pa = {
      quad: calibPts.map(toPx),
      base_points_raw: basePoints.map(toPx),
      top_left: toPx(calibPts[0]), top_right: toPx(calibPts[1]),
      bottom_right: toPx(calibPts[2]), bottom_left: toPx(calibPts[3]),
      mask_points: maskPoints.length > 2 ? maskPoints.map(toPx) : null,
      mesh_control_src: meshPoints.map(m => toPx(m.src)),
      mesh_control_dst: meshPoints.map(m => toPx(m.dst)),
      product_type: productType,
      camera_elevation: pitch_api,
    };
    onCoordinatesChange(pa);
    const nextWarpConfig = {
      warp_type: meshPoints.length > 0 ? 'tps' : 'cylinder',
      theta_max_deg: curvePct * 0.9,
      curve_pct: curvePct,
      curve: (smile_api / 100) * hr_ratio * 0.15,
      curve_top: curveTop,
      curve_bottom: curveBot,
      tilt_deg: tilt, rotate_deg: rotate, persp_strength: perspective,
      camera_elevation: pitch_api,
      product_type: productType,
      feather_radius: featherRadius,
      design_scale: designScale,
      design_offset_x: designOffsetX,
      design_offset_y: designOffsetY,
      editor_mode: editorMode,
      is_locked: editorMode === 'DESIGN',
    };
    onConfigChange(nextWarpConfig);
    if (onLockedSnapshotChange) {
      if (editorMode === 'DESIGN') onLockedSnapshotChange({ printArea: pa, warpConfig: nextWarpConfig });
      else onLockedSnapshotChange(null);
    }

    const timer = setTimeout(() => {
      const templateIdMatch = imageUrl.match(/\/templates\/([^/]+)\//);
      const templateId = templateIdMatch ? templateIdMatch[1] : null;

      const payload = {
        mockup_width: naturalSize.w, mockup_height: naturalSize.h,
        print_area: pa,
        warp_type: meshPoints.length > 0 ? 'tps' : 'cylinder',
        theta_max_deg: curvePct * 0.9,
        curve: (smile_api / 100) * hr_ratio * 0.15,
        curve_top: curveTop,
        curve_bottom: curveBot,
        design_scale: designScale,
        design_offset_x: designOffsetX,
        design_offset_y: designOffsetY,
        mask_points: pa.mask_points,
        template_id: templateId,
      };

      const request = designFile
        ? (() => {
            const formData = new FormData();
            formData.append('config_json', JSON.stringify(payload));
            formData.append('design_image', designFile);
            return fetch('/v1/mockup/warp-preview-file', {
              method: 'POST',
              body: formData,
            });
          })()
        : fetch('/v1/mockup/warp-preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });

      request
        .then(async r => {
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || `Server error ${r.status}`);
          }
          setApiError(null);
          return r.blob();
        })
        .then(blob => setPreviewOverlayUrl(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(blob); }))
        .catch((e) => {
          console.error("Warp Preview Error:", e);
          setApiError(e.message || "Failed to fetch preview");
        });
    }, 400);
    return () => clearTimeout(timer);
  }, [basePoints, tilt, rotate, perspective, curvePct, curveTop, curveBot, designScale, designOffsetX, designOffsetY, featherRadius, productType, maskPoints, meshPoints, naturalSize, editorMode, imageUrl, designFile]);

  const lastPointerPos = useRef({ x: 0, y: 0 });

  const handlePointerDown = (type: string, id: number, e: React.PointerEvent) => {
    e.preventDefault(); e.stopPropagation();
    setDraggingIdx({ type, id });
    lastPointerPos.current = { x: e.clientX, y: e.clientY };
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  };

  const handleArtworkPointerDown = (e: React.PointerEvent<HTMLImageElement>) => {
    if (!canDragArtwork) return;
    const alphaCanvas = overlayAlphaCanvasRef.current;
    if (!alphaCanvas) return;

    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const sampleX = Math.max(0, Math.min(alphaCanvas.width - 1, Math.round(((e.clientX - rect.left) / rect.width) * (alphaCanvas.width - 1))));
    const sampleY = Math.max(0, Math.min(alphaCanvas.height - 1, Math.round(((e.clientY - rect.top) / rect.height) * (alphaCanvas.height - 1))));
    const ctx = alphaCanvas.getContext('2d');
    const alpha = ctx?.getImageData(sampleX, sampleY, 1, 1).data[3] ?? 0;
    if (alpha < 12) return;

    handlePointerDown('design', 0, e);
  };

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!containerRef.current) return;
    const r = containerRef.current.getBoundingClientRect();
    if (!draggingIdx) {
      lastPointerPos.current = { x: e.clientX, y: e.clientY };
      return;
    }
    const rawX = (e.clientX - r.left - pan.x) / (r.width * zoom);
    const rawY = (e.clientY - r.top - pan.y) / (r.height * zoom);
    const x = Math.max(0, Math.min(1, rawX));
    const y = Math.max(0, Math.min(1, rawY));
    if (draggingIdx.type === 'base') {
      const next = [...basePoints]; next[draggingIdx.id] = { x, y }; setBasePoints(next);
    } else if (draggingIdx.type === 'design') {
      const dx = (e.clientX - lastPointerPos.current.x) / (r.width * zoom);
      const dy = (e.clientY - lastPointerPos.current.y) / (r.height * zoom);
      setDesignOffsetX(prev => Math.max(-1, Math.min(1, prev + dx)));
      setDesignOffsetY(prev => Math.max(-1, Math.min(1, prev + dy)));
    } else if (draggingIdx.type === 'mask') {
      const next = [...maskPoints]; next[draggingIdx.id] = { x, y }; setMaskPoints(next);
    } else if (draggingIdx.type === 'mesh') {
      const next = [...meshPoints]; next[draggingIdx.id].dst = { x, y }; setMeshPoints(next);
    } else if (draggingIdx.type === 'smile') {
      const straightMidY = (calibPts[0].y + calibPts[1].y) / 2;
      const dy = (rawY - straightMidY);
      const curveVal = (dy / (H_px / (naturalSize.h || 1000) * 0.12)) * 100;
      setCurveTop(curveVal);
      if (curvePct === 0) setCurvePct(60);
    } else if (draggingIdx.type === 'pitch') {
      const straightMidY = (calibPts[3].y + calibPts[2].y) / 2;
      const dy = (rawY - straightMidY);
      const curveVal = (dy / (H_px / (naturalSize.h || 1000) * 0.12)) * 100;
      // Keep drag direction intuitive for bottom handle:
      // drag down => bottom curve goes down, drag up => bottom curve goes up.
      setCurveBot(-curveVal);
      if (curvePct === 0) setCurvePct(60);
    }
    lastPointerPos.current = { x: e.clientX, y: e.clientY };
  }, [draggingIdx, zoom, pan, basePoints, maskPoints, meshPoints, calibPts, H_px, naturalSize, curvePct]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = -e.deltaY;
    const factor = delta > 0 ? 1.1 : 0.9;
    if (designFile && !e.altKey) {
      setDesignScale(prev => Math.max(0.1, Math.min(5, prev * factor)));
    } else {
      setZoom(prev => Math.max(0.5, Math.min(10, prev * factor)));
    }
  };

  const handleContainerMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && e.altKey)) setIsPanning(true);
  };
  const handleContainerMouseMove = (e: React.MouseEvent) => {
    if (isPanning) setPan(prev => ({ x: prev.x + e.movementX, y: prev.y + e.movementY }));
  };
  const handleContainerMouseUp = () => setIsPanning(false);

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (!containerRef.current || draggingIdx) return;
    const r = containerRef.current.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const y = (e.clientY - r.top) / r.height;
    if (activeMode === 'mask') setMaskPoints([...maskPoints, { x, y }]);
    if (activeMode === 'mesh') setMeshPoints([...meshPoints, { src: { x, y }, dst: { x, y } }]);
  };

  const handleAutoDetect = async () => {
    setIsDetecting(true);
    try {
      const blob = await fetch(imageUrl).then(r => r.blob());
      const fd = new FormData(); fd.append('mockup_image', blob);
      const data = await fetch('/v1/mockup/detect-region', { method: 'POST', body: fd }).then(r => r.json());
      if (data.quad) {
        const toPct = (pts: number[][]) => pts.map(p => ({ x: p[0] / naturalSize.w, y: p[1] / naturalSize.h }));
        setBasePoints(toPct(data.quad));
        if (data.clip_mask) setMaskPoints(toPct(data.clip_mask));
        setTilt(0); setRotate(0); setPerspective(0);
        setCurveTop(25); setCurveBot(5); setCurvePct(60);
      }
    } catch { /* ignore */ } finally { setIsDetecting(false); }
  };

  const make4x4Mesh = () => {
    const rows = 4, cols = 4;
    const pts: { src: Point; dst: Point }[] = [];
    const src_coords = [{ x: -1, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }];
    const H_mat_local = solve_homography(src_coords, calibPts);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const u = c / (cols - 1), v = r / (rows - 1);
      const world = apply_homography(H_mat_local, u, v);
      pts.push({ src: world, dst: { ...world } });
    }
    setMeshPoints(pts);
  };

  const toggleRenderDevice = async () => {
    if (isSwitchingDevice) return;
    const target = renderDevice === 'gpu' ? 'cpu' : 'gpu';
    if (target === 'gpu' && !gpuAvailable) {
      setApiError('GPU/OpenCL không khả dụng trên máy hiện tại.');
      return;
    }
    setIsSwitchingDevice(true);
    try {
      const res = await fetch('/v1/render/device', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device: target }),
      });
      if (!res.ok) throw new Error(`Device switch failed (${res.status})`);
      const data = await res.json();
      if (data?.device === 'gpu' || data?.device === 'cpu') setRenderDevice(data.device);
      if (typeof data?.opencl_available === 'boolean') setGpuAvailable(data.opencl_available);
      setApiError(null);
    } catch (e: any) {
      setApiError(e?.message || 'Không thể chuyển CPU/GPU');
    } finally {
      setIsSwitchingDevice(false);
    }
  };

  const Slider = ({ label, value, min, max, step = 1, unit = '', onChange }: { label: string; value: number; min: number; max: number; step?: number; unit?: string; onChange: (v: number) => void; }) => (
    <div className="ctrl">
      <div className="ctrl-header"><span className="ctrl-label">{label}</span><span className="ctrl-value">{value}{unit}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} className="slider" />
    </div>
  );

  const renderControls = () => {
    const productSelector = !productTypeProp && (
      <div className="product-selector-top glass-panel" style={{ padding: '8px 16px', marginBottom: '12px', border: '1px solid #1e2d44', background: '#111827', borderRadius: '8px' }}>
        <div className="ctrl-header" style={{ marginBottom: '6px' }}><span className="ctrl-label" style={{ color: '#60a5fa', fontSize: '9px' }}>Step 1: Product Category</span></div>
        <select className="select" style={{ width: '100%', fontSize: '0.9rem', padding: '6px 10px' }} value={productType} onChange={e => setProductType(e.target.value)}>
          <optgroup label="🍵 MUGS MODULE"><option value="cylinder_ceramic">Ceramic Mug</option><option value="cylinder_glass">Glass Mug</option><option value="cylinder_travel">Travel Mug</option></optgroup>
          <optgroup label="👕 CLOTHES MODULE"><option value="apparel_cotton">T-Shirt</option><option value="apparel_hoodie">Hoodie</option><option value="apparel_totebag">Tote Bag</option></optgroup>
          <optgroup label="🖼 OTHER"><option value="flat_print">Flat Print</option><option value="plastic_case">Phone Case</option></optgroup>
        </select>
      </div>
    );

    const availableGroups: ('geometry' | 'wrap' | 'blend' | 'advanced')[] = ['geometry'];
    if (productType.startsWith('cylinder')) availableGroups.push('wrap');
    availableGroups.push('blend', 'advanced');

    if (editorMode === 'DESIGN') return (
      <div className="ctrl-section">
        {productSelector}
        {designFile && <p className="mode-hint">Kéo trực tiếp trên artwork 2D để đổi vị trí. Lăn chuột để zoom artwork, `Alt + wheel` để zoom canvas.</p>}
        <div className="ctrl-header"><span className="ctrl-label">Design Transformation</span></div>
        <div className="ctrl-grid">
          <Slider label="Artwork Scale" value={designScale} min={0.1} max={5} step={0.01} unit="x" onChange={setDesignScale} />
          <Slider label="Offset X" value={designOffsetX} min={-1} max={1} step={0.001} onChange={setDesignOffsetX} />
          <Slider label="Offset Y" value={designOffsetY} min={-1} max={1} step={0.001} onChange={setDesignOffsetY} />
        </div>
      </div>
    );

    if (activeMode === 'mesh') return (
      <div className="ctrl-section">{productSelector}<p className="mode-hint">Kéo chấm cam để biến dạng bề mặt.</p><div className="btn-row"><button className="btn-primary" onClick={make4x4Mesh}>⊞ Lưới 4×4</button><button className="btn-ghost" onClick={() => setMeshPoints([])}>↺ Reset</button></div></div>
    );
    if (activeMode === 'mask') return (
      <div className="ctrl-section">{productSelector}<p className="mode-hint">Click để vẽ vùng mask. Đóng vòng để hoàn thành.</p><div className="btn-row"><button className="btn-ghost" onClick={() => setMaskPoints([])}>↺ Xóa Mask</button></div></div>
    );

    return (
      <div className="ctrl-section">
        {productSelector}
        {designFile && activeMode === 'calibrate' && <p className="mode-hint">Artwork là ảnh ngang 2D. Kéo chỉ tác động lên artwork, không kéo mockup gốc.</p>}
        <div className="group-tabs">
          {availableGroups.map(g => (
            <button key={g} className={`group-tab ${activeGroup === g ? 'active' : ''}`} onClick={() => setActiveGroup(g)}>
              {{ geometry: '📐 Geometry', wrap: '🌀 Wrap', blend: '🎨 Blend', advanced: '⚙️ Advanced' }[g]}
            </button>
          ))}
        </div>
        {activeGroup === 'geometry' && (
          <div className="ctrl-grid">
            <Slider label="Tilt" value={tilt} min={-45} max={45} unit="°" onChange={setTilt} />
            <Slider label="Rotate" value={rotate} min={-30} max={30} unit="°" onChange={setRotate} />
            <Slider label="Perspective" value={perspective} min={-50} max={50} onChange={setPerspective} />
          </div>
        )}
        {activeGroup === 'wrap' && (
          <div className="ctrl-grid">
            <Slider label="Cylinder Width" value={curvePct} min={0} max={100} unit="%" onChange={setCurvePct} />
            <Slider label="Curve Top" value={curveTop} min={-100} max={100} unit="%" onChange={setCurveTop} />
            <Slider label="Curve Bottom" value={-curveBot} min={-100} max={100} unit="%" onChange={(v) => setCurveBot(-v)} />
          </div>
        )}
        {activeGroup === 'blend' && (
          <div className="ctrl-grid"><Slider label="Feather Radius" value={featherRadius} min={0} max={12} unit="px" onChange={setFeatherRadius} /></div>
        )}
        {activeGroup === 'advanced' && (
          <div className="ctrl-grid"><button className="btn-ghost" style={{ marginTop: 8 }} onClick={() => { setTilt(0); setRotate(0); setPerspective(0); setCurveTop(0); setCurveBot(0); setDesignScale(1); setDesignOffsetX(0); setDesignOffsetY(0); }}> ↺ Reset All </button></div>
        )}
      </div>
    );
  };

  return (
    <div className="pae-root">
      <div className="toolbar">
        <div className="toolbar-left">
          <div className="logo-badge">Print Area Editor</div>
          <div className="mode-sub-pills" style={{ marginLeft: 20, display: 'flex', gap: 6, visibility: editorMode === 'CALIBRATE' ? 'visible' : 'hidden' }}>
            {(['calibrate', 'mesh', 'mask'] as const).filter(m => (m === 'mesh' ? productType.startsWith('apparel') : true)).map(m => (
              <button key={m} className={`pill pill-sm ${activeMode === m ? 'active' : ''}`} onClick={() => setActiveMode(m)}>
                {m === 'calibrate' && '📐 '} {m === 'mesh' && '⊞ '} {m === 'mask' && '🎭 '} {m.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div className="toolbar-right">
          <button
            className={`icon-btn ${renderDevice === 'gpu' ? 'active' : ''}`}
            onClick={toggleRenderDevice}
            disabled={isSwitchingDevice || (renderDevice === 'cpu' && !gpuAvailable)}
            title={gpuAvailable ? 'Bật/tắt GPU runtime' : 'GPU/OpenCL không khả dụng'}
          >
            {isSwitchingDevice ? 'Switching…' : (renderDevice === 'gpu' ? 'GPU ON' : 'CPU')}
          </button>
          <button className={`icon-btn ${showGrid ? 'active' : ''}`} onClick={() => setShowGrid(v => !v)}>Grid</button>
          <button className="btn-detect" onClick={handleAutoDetect} disabled={isDetecting}>{isDetecting ? 'Detecting…' : '✦ Smart Detect'}</button>
          <div className="workflow-status">
            {editorMode === 'CALIBRATE' ? <button className="btn-confirm-lock" onClick={() => setEditorMode('DESIGN')}>✅ Confirm & Lock</button> : <button className="btn-unlock" onClick={() => setEditorMode('CALIBRATE')}>🔓 Unlock</button>}
          </div>
        </div>
      </div>

      <div className="pae-body">
        <aside className="sidebar">
          {renderControls()}
        </aside>

        <main className="canvas-section">
          <div className="canvas-wrap" ref={containerRef} onPointerMove={handlePointerMove} onPointerUp={() => setDraggingIdx(null)} onWheel={handleWheel} onMouseDown={handleContainerMouseDown} onMouseMove={handleContainerMouseMove} onMouseUp={handleContainerMouseUp} onMouseLeave={handleContainerMouseUp} onClick={handleCanvasClick} style={{ cursor: isPanning || draggingIdx?.type === 'design' ? 'grabbing' : (canDragArtwork ? 'grab' : (activeMode !== 'calibrate' ? 'crosshair' : 'default')), overflow: 'hidden' }}>
            <div className="canvas-container" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: '0 0', width: '100%', height: '100%', position: 'relative', ['--zoom' as any]: zoom }}>
                <img src={imageUrl} alt="Mockup" className="mockup-img" onLoad={e => setNaturalSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
                {showBackendOverlay && previewOverlayUrl && (
                  <img
                    src={previewOverlayUrl}
                    className={`overlay-img ${editorMode === 'CALIBRATE' ? 'overlay-img-calibrate' : 'overlay-img-design'} ${designFile ? 'overlay-img-original-color' : ''} ${canDragArtwork ? 'overlay-img-interactive' : ''}`}
                    alt=""
                    onPointerDown={handleArtworkPointerDown}
                  />
                )}
                <svg className="overlay-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                  {productType.includes('cylinder') ? <MugCurvePreview calibPts={calibPts} curvePct={curvePct} curveTop={curveTop} curveBot={curveBot} hPx={H_px} wPx={W_px} showGrid={showEditorGridLayer} onSmileDrag={setCurveTop} onPitchDrag={setCurveBot} /> : <WarpGridPreview meshPoints={meshPoints} showGrid={showEditorGridLayer} />}
                </svg>
              {editorMode === 'CALIBRATE' && (
                <>
                  {activeMode === 'calibrate' && calibPts.map((p, i) => <div key={i} className="handle handle-corner-dot" style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }} onPointerDown={e => handlePointerDown('base', i, e)} />)}
                  {activeMode === 'calibrate' && productType.startsWith('cylinder') && ( <> <div className="handle handle-smile-dot" style={{ left: `${smileHandle.x * 100}%`, top: `${smileHandle.y * 100}%` }} onPointerDown={e => handlePointerDown('smile', 0, e)} /> <div className="handle handle-pitch-dot" style={{ left: `${pitchHandle.x * 100}%`, top: `${pitchHandle.y * 100}%` }} onPointerDown={e => handlePointerDown('pitch', 0, e)} /> </> )}
                  {activeMode === 'mask' && maskPoints.map((p, i) => <div key={i} className="handle handle-mask" style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }} onPointerDown={e => handlePointerDown('mask', i, e)} />)}
                  {activeMode === 'mesh' && meshPoints.map((m, i) => <div key={i} className="handle handle-mesh" style={{ left: `${m.dst.x * 100}%`, top: `${m.dst.y * 100}%` }} onPointerDown={e => handlePointerDown('mesh', i, e)} />)}
                </>
              )}
            </div>
          </div>
        </main>
      </div>

      {apiError && <div className="error-toast"><span>⚠️ {apiError}</span><button className="btn-ghost" onClick={() => setApiError(null)}>Dismiss</button></div>}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

        .pae-root {
          display: flex; flex-direction: column; height: 100vh;
          background: #0a0c10; color: #e2e8f0; font-family: 'Syne', sans-serif; overflow: hidden;
        }

        .pae-body {
          display: flex; flex: 1; min-height: 0; overflow: hidden;
        }

        /* ── TOOLBAR ── */
        .toolbar {
          display: flex; align-items: center; justify-content: space-between;
          padding: 0 20px; height: 52px; background: #0e1117;
          border-bottom: 1px solid #1e2330; flex-shrink: 0; gap: 12px; z-index: 100;
        }
        .toolbar-left, .toolbar-right { display: flex; align-items: center; gap: 12px; }
        .logo-badge {
          font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
          color: #475569; padding: 4px 10px; border: 1px solid #1e2330; border-radius: 6px;
        }
        .pill {
          padding: 5px 14px; border: none; background: transparent; color: #475569;
          cursor: pointer; border-radius: 6px; font-family: 'Syne', sans-serif;
          font-size: 12px; font-weight: 600; letter-spacing: 0.04em; transition: all 0.15s;
        }
        .pill-sm { font-size: 10px; padding: 4px 10px; }
        .pill:hover { background: #161b27; color: #94a3b8; }
        .pill.active { background: #1a2540; color: #60a5fa; }
        .icon-btn {
          display: flex; align-items: center; gap: 6px; padding: 5px 12px;
          border: 1px solid #1e2330; background: transparent; color: #475569;
          cursor: pointer; border-radius: 6px; font-family: 'Syne', sans-serif;
          font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; transition: all 0.15s;
        }
        .icon-btn:hover { border-color: #2d3a52; color: #94a3b8; }
        .icon-btn.active { border-color: #3b82f6; color: #60a5fa; background: #1a2540; }
        .icon-btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .btn-detect {
          display: flex; align-items: center; gap: 8px; padding: 7px 18px;
          background: linear-gradient(135deg, #1d4ed8, #2563eb); color: white; border: none;
          border-radius: 8px; font-family: 'Syne', sans-serif; font-size: 12px;
          font-weight: 700; cursor: pointer; letter-spacing: 0.04em; transition: opacity 0.15s;
        }
        .btn-detect:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-confirm-lock {
          background: #3b82f6; color: white; border: none; padding: 6px 16px;
          border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 13px; transition: all 0.2s;
        }
        .btn-confirm-lock:hover { background: #2563eb; transform: scale(1.05); }
        .btn-unlock {
          background: #374151; color: #9ca3af; border: 1px solid #4b5563;
          padding: 6px 16px; border-radius: 6px; font-weight: 600; cursor: pointer; font-size: 13px;
        }
        .btn-unlock:hover { color: white; border-color: #6b7280; }

        /* ── SIDEBAR ── */
        .sidebar {
          width: 320px; background: #0e1117; border-right: 1px solid #1e2330;
          display: flex; flex-direction: column; overflow-y: auto; flex-shrink: 0;
        }
        .ctrl-section {
          padding: 24px; display: flex; flex-direction: column; gap: 20px;
        }
        .group-tabs {
          display: flex; flex-direction: column; gap: 4px;
          padding-bottom: 12px; border-bottom: 1px solid #1e2330;
        }
        .group-tab {
          padding: 8px 14px; text-align: left; background: transparent; border: 1px solid transparent;
          color: #475569; cursor: pointer; border-radius: 6px;
          font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 600;
          letter-spacing: 0.05em; transition: all 0.15s;
        }
        .group-tab:hover { color: #94a3b8; background: #141924; }
        .group-tab.active { border-color: #1e2d44; color: #60a5fa; background: #111827; }
        .ctrl-grid {
          display: flex; flex-direction: column; gap: 20px;
        }
        .ctrl { display: flex; flex-direction: column; gap: 8px; }
        .ctrl-header { display: flex; justify-content: space-between; align-items: baseline; }
        .ctrl-label {
          font-size: 10px; font-weight: 700; text-transform: uppercase;
          letter-spacing: 0.1em; color: #475569;
        }
        .ctrl-value { font-family: 'DM Mono', monospace; font-size: 11px; color: #60a5fa; }
        .slider {
          -webkit-appearance: none; appearance: none; width: 100%; height: 3px;
          background: #1e2330; border-radius: 2px; outline: none; cursor: pointer;
        }
        .slider::-webkit-slider-thumb {
          -webkit-appearance: none; width: 14px; height: 14px;
          background: #3b82f6; border-radius: 50%; border: 2px solid #93c5fd;
          cursor: grab; box-shadow: 0 0 6px rgba(59, 130, 246, 0.4); transition: transform 0.1s;
        }
        .slider::-webkit-slider-thumb:active { transform: scale(1.2); cursor: grabbing; }

        /* ── CANVAS ── */
        .canvas-section {
          flex: 1; background: #111; display: flex; align-items: center; justify-content: center;
          position: relative; overflow: hidden;
        }
        .canvas-wrap {
          position: relative; max-width: 100%; max-height: 100%;
          display: flex; align-items: center; justify-content: center;
        }
        .mockup-img {
          display: block; max-width: 100%; max-height: calc(100vh - 52px);
          object-fit: contain; user-select: none;
        }
        .overlay-img {
          position: absolute; top: 0; left: 0; width: 100%; height: 100%;
          object-fit: contain; pointer-events: none;
        }
        .overlay-img-calibrate { opacity: 0.78; }
        .overlay-img-design { opacity: 1; }
        .overlay-img-original-color { opacity: 1; }
        .overlay-img-interactive { pointer-events: auto; cursor: grab; }
        .overlay-img-interactive:active { cursor: grabbing; }
        .overlay-svg {
          position: absolute; top: 0; left: 0; width: 100%; height: 100%;
          pointer-events: none; z-index: 10;
        }

        /* ── GRID LINES ── */
        .grid-boundary-unified {
          fill: none; stroke: #00f2ff; stroke-width: 0.3; opacity: 0.7;
          pointer-events: none; vector-effect: non-scaling-stroke;
        }
        .grid-line {
          fill: none; stroke: #7dd3fc; stroke-width: 0.24; opacity: 0.82;
          vector-effect: non-scaling-stroke;
        }

        /* ── HANDLES ── */
        .handle {
          position: absolute; cursor: pointer; z-index: 100;
          display: flex; align-items: center; justify-content: center;
          transition: transform 0.1s;
          transform: translate(-50%, -50%) scale(calc(1 / var(--zoom, 1)));
          background: transparent;
        }
        .handle:hover { transform: translate(-50%, -50%) scale(calc(1.4 / var(--zoom, 1))); }
        .handle-corner-dot { width: 14px; height: 14px; }
        .handle-corner-dot::after {
          content: ''; width: 5px; height: 5px; background: #00f2ff;
          border-radius: 50%; border: 1px solid rgba(255,255,255,0.6);
          box-shadow: 0 0 3px rgba(0,242,255,0.5);
        }
        .handle-smile-dot { width: 16px; height: 16px; }
        .handle-smile-dot::after {
          content: ''; width: 5px; height: 5px; background: #00f2ff;
          border-radius: 50%; border: 1px solid white;
        }
        .handle-pitch-dot { width: 16px; height: 16px; }
        .handle-pitch-dot::after {
          content: ''; width: 5px; height: 5px; background: #f59e0b;
          border-radius: 50%; border: 1px solid white;
        }
        .handle-mask {
          width: 14px; height: 14px; background: #059669;
          border: 2px solid #6ee7b7; border-radius: 3px;
        }
        .handle-mesh {
          width: 12px; height: 12px; background: #d97706;
          border: 2px solid #fcd34d; border-radius: 50%;
        }

        /* ── MISC ── */
        .select {
          background: #141924; color: #94a3b8; border: 1px solid #1e2330;
          padding: 8px 12px; border-radius: 6px; font-family: 'Syne', sans-serif;
          font-size: 13px; cursor: pointer; width: 100%;
        }
        .select:focus { outline: none; border-color: #3b82f6; }
        .mode-hint { font-size: 12px; color: #64748b; line-height: 1.5; }
        .btn-row { display: flex; gap: 10px; flex-wrap: wrap; }
        .btn-primary {
          padding: 8px 18px; background: #1d4ed8; color: white; border: none;
          border-radius: 7px; font-family: 'Syne', sans-serif; font-size: 12px;
          font-weight: 700; cursor: pointer; transition: background 0.15s;
        }
        .btn-ghost {
          padding: 8px 18px; background: transparent; color: #94a3b8;
          border: 1px solid #1e2330; border-radius: 7px; font-family: 'Syne', sans-serif;
          font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s;
        }
        .btn-ghost:hover { background: #141924; color: #e2e8f0; }
        .error-toast {
          position: fixed; bottom: 20px; right: 20px;
          background: rgba(220, 38, 38, 0.9); color: white; padding: 10px 20px;
          border-radius: 8px; display: flex; gap: 12px; align-items: center; z-index: 1000;
        }
      `}</style>
    </div>
  );
}
