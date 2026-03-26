import React, { useState, useRef, useEffect, useCallback } from 'react';

interface Point { x: number; y: number; }
interface PrintAreaEditorProps {
  imageUrl: string;
  warpConfig: any;
  onCoordinatesChange: (data: any) => void;
  onConfigChange: (config: any) => void;
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
  // Solve Ah = b where A is 8x8 matrix
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
  // Simple Gaussian elimination
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
  const x = u * 2 - 1, y = v * 2 - 1; // Canonical [-1, 1]
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w
  };
}

// Cylinder warp: apply same math as backend unified_cylinder_warp
function cylinderWarpPoint(u: number, v: number, curvePct: number, curveTop: number, curveBot: number, hPx: number, wPx: number): Point {
  const thetaMax = (curvePct * 0.9 * Math.PI) / 180;
  const nx = (u - 0.5) * 2;
  const theta = nx * thetaMax;

  // horizontal cylinder projection
  const wx = Math.sin(theta) / (2 * Math.sin(thetaMax) + 1e-8) + 0.5;

  // vertical cos-based smile (Match Python logic exactly)
  const hr_ratio = hPx / (wPx + 1e-8);

  // Independent Curve Logic:
  // Interpolate between curveTop (v=0) and curveBot (v=1)
  const curve_v = (curveTop / 100) * (1 - v) + (curveBot / 100) * v;
  const curve_adaptive = curve_v * hr_ratio * 0.15;

  const yProj = (v - 0.5) * 2;
  const wy_canon = yProj + curve_adaptive * (Math.cos(theta) - Math.cos(thetaMax));
  const wy = (wy_canon + 1) * 0.5;

  return { x: wx, y: wy };
}

// Build warped grid lines (SVG polyline points) in [0..100] space
function buildCurvedPath(curvePct: number, curveTop: number, curveBot: number, H_px: number, W_px: number, H_mat: number[]): string {
  if (curvePct === 0) {
    const pts = [
      apply_homography(H_mat, -1, -1), apply_homography(H_mat, 1, -1),
      apply_homography(H_mat, 1, 1), apply_homography(H_mat, -1, 1)
    ];
    return `M ${pts[0].x * 100},${pts[0].y * 100} L ${pts[1].x * 100},${pts[1].y * 100} L ${pts[2].x * 100},${pts[2].y * 100} L ${pts[3].x * 100},${pts[3].y * 100} Z`;
  }

  const steps = 24;
  const pathParts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const u = i / steps;
    const l = cylinderWarpPoint(u, 0, curvePct, curveTop, curveBot, H_px, W_px);
    const w = apply_homography(H_mat, l.x, l.y);
    pathParts.push(i === 0 ? `M ${w.x * 100},${w.y * 100}` : `L ${w.x * 100},${w.y * 100}`);
  }
  for (let i = steps; i >= 0; i--) {
    const u = i / steps;
    const l = cylinderWarpPoint(u, 1, curvePct, curveTop, curveBot, H_px, W_px);
    const w = apply_homography(H_mat, l.x, l.y);
    pathParts.push(`L ${w.x * 100},${w.y * 100}`);
  }
  pathParts.push('Z');
  return pathParts.join(' ');
}

function buildWarpedGrid(
  calibPts: Point[],
  steps: number,
  curvePct: number,
  curveTop: number,
  curveBot: number,
  hPx: number,
  wPx: number
): { v: string[]; h: string[] } {
  const v: string[] = [];
  const h: string[] = [];

  const src_canon = [{ x: -1, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }];
  const H_mat = solve_homography(src_canon, calibPts);

  for (let ci = 1; ci < steps; ci++) {
    const u = ci / steps;
    const vPath: string[] = [];
    for (let ri = 0; ri <= steps; ri++) {
      const vv = ri / steps;
      const local = cylinderWarpPoint(u, vv, curvePct, curveTop, curveBot, hPx, wPx);
      const world = apply_homography(H_mat, local.x, local.y);
      vPath.push(`${world.x * 100},${world.y * 100}`);
    }
    v.push(vPath.join(' '));
  }

  for (let ri = 1; ri < steps; ri++) {
    const vv = ri / steps;
    const hPath: string[] = [];
    for (let ci = 0; ci <= steps; ci++) {
      const u = ci / steps;
      const local = cylinderWarpPoint(u, vv, curvePct, curveTop, curveBot, hPx, wPx);
      const world = apply_homography(H_mat, local.x, local.y);
      hPath.push(`${world.x * 100},${world.y * 100}`);
    }
    h.push(hPath.join(' '));
  }

  return { v, h };
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function PrintAreaEditor({ imageUrl, warpConfig, onCoordinatesChange, onConfigChange }: PrintAreaEditorProps) {
  const [basePoints, setBasePoints] = useState<Point[]>([
    { x: 0.22, y: 0.14 }, { x: 0.78, y: 0.14 },
    { x: 0.78, y: 0.86 }, { x: 0.22, y: 0.86 },
  ]);
  const [tilt, setTilt] = useState(warpConfig.tilt_deg || 0);
  const [rotate, setRotate] = useState(warpConfig.rotate_deg || 0);
  const [perspective, setPerspective] = useState(warpConfig.persp_strength || 0);
  const [curvePct, setCurvePct] = useState(60);
  const [curveTop, setCurveTop] = useState(0); // Independent Top curve
  const [curveBot, setCurveBot] = useState(0);  // Independent Bot curve
  const [designScale, setDesignScale] = useState(1.0);
  const [designOffsetX, setDesignOffsetX] = useState(0.0);
  const [designOffsetY, setDesignOffsetY] = useState(0.0);

  const prevImageUrl = useRef(imageUrl);
  useEffect(() => {
    if (imageUrl && imageUrl !== prevImageUrl.current) {
      setDesignScale(1.0);
      setDesignOffsetX(0);
      setDesignOffsetY(0);
      prevImageUrl.current = imageUrl;
    }
  }, [imageUrl]);
  const [editorMode, setEditorMode] = useState<'CALIBRATE' | 'DESIGN'>('CALIBRATE');
  const [featherRadius, setFeatherRadius] = useState(3);
  const [productType, setProductType] = useState('cylinder_ceramic');

  const [activeMode, setActiveMode] = useState<'calibrate' | 'mesh' | 'mask'>('calibrate');
  const [activeGroup, setActiveGroup] = useState<'geometry' | 'wrap' | 'blend' | 'advanced'>('geometry');
  const [showGrid, setShowGrid] = useState(true);
  const [maskPoints, setMaskPoints] = useState<Point[]>([]);
  const [meshPoints, setMeshPoints] = useState<{ src: Point; dst: Point }[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [previewOverlayUrl, setPreviewOverlayUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [draggingIdx, setDraggingIdx] = useState<{ type: string; id: number } | null>(null);

  // Zoom / Pan State
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  const calibPts = applyCalibration(basePoints, tilt, rotate, perspective);

  // Calculate Px sizes for math sync
  const W_px = Math.abs(calibPts[1].x - calibPts[0].x) * (naturalSize.w || 1000);
  const H_px = Math.abs(calibPts[3].y - calibPts[0].y) * (naturalSize.h || 1000);

  const src_canon = [{ x: -1, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }];
  const H_mat = solve_homography(src_canon, calibPts);

  const gridLines = buildWarpedGrid(calibPts, 10, curvePct, curveTop, curveBot, H_px, W_px);

  // Derive Mid Handles — Both on TOP edge
  // Smile handle: midpoint of top edge
  const localSmile = cylinderWarpPoint(0.5, 0, curvePct, curveTop, curveBot, H_px, W_px);
  const smileHandle = apply_homography(H_mat, localSmile.x, localSmile.y);

  // Pitch handle: midpoint of BOTTOM edge
  const localPitch = cylinderWarpPoint(0.5, 1, curvePct, curveTop, curveBot, H_px, W_px);
  const pitchHandle = apply_homography(H_mat, localPitch.x, localPitch.y);

  const curvedBoundaryPath = buildCurvedPath(curvePct, curveTop, curveBot, H_px, W_px, H_mat);

  // ── Sync backend ──
  useEffect(() => {
    if (naturalSize.w === 0) return;
    const toPx = (p: Point) => [Math.round(p.x * naturalSize.w), Math.round(p.y * naturalSize.h)];

    // Calculate hr_ratio for API scaling
    const hr_ratio = H_px / (W_px + 1e-8);
    const smile_api = (curveTop + curveBot) / 2;
    const pitch_api = (curveTop - curveBot);
    const pa = {
      quad: calibPts.map(toPx),
      top_left: toPx(calibPts[0]), top_right: toPx(calibPts[1]),
      bottom_right: toPx(calibPts[2]), bottom_left: toPx(calibPts[3]),
      mask_points: maskPoints.length > 2 ? maskPoints.map(toPx) : null,
      mesh_control_src: meshPoints.map(m => toPx(m.src)),
      mesh_control_dst: meshPoints.map(m => toPx(m.dst)),
      product_type: productType,
      camera_elevation: pitch_api,
    };
    onCoordinatesChange(pa);

    onConfigChange({
      warp_type: meshPoints.length > 0 ? 'tps' : 'cylinder',
      theta_max_deg: curvePct * 0.9,
      curve: (smile_api / 100) * hr_ratio * 0.15,
      tilt_deg: tilt, rotate_deg: rotate, persp_strength: perspective,
      camera_elevation: pitch_api,
      product_type: productType,
      feather_radius: featherRadius,
      design_scale: designScale,
      design_offset_x: designOffsetX,
      design_offset_y: designOffsetY,
    });

    const timer = setTimeout(() => {
      // Extract template_id from URL if possible
      const templateIdMatch = imageUrl.match(/\/templates\/([^/]+)\//);
      const templateId = templateIdMatch ? templateIdMatch[1] : null;

      fetch('/v1/mockup/warp-preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mockup_width: naturalSize.w, mockup_height: naturalSize.h,
          print_area: pa,
          warp_type: meshPoints.length > 0 ? 'tps' : 'cylinder',
          theta_max_deg: curvePct * 0.9,
          curve: smile_api / 100,
          design_scale: designScale,
          design_offset_x: designOffsetX,
          design_offset_y: designOffsetY,
          mask_points: pa.mask_points,
          template_id: templateId,
        }),
      })
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
  }, [basePoints, tilt, rotate, perspective, curvePct, curveTop, curveBot, designScale, designOffsetX, designOffsetY, featherRadius, productType, maskPoints, meshPoints, naturalSize]);

  // ── Pointer handlers ──
  const lastPointerPos = useRef({ x: 0, y: 0 });

  const handlePointerDown = (type: string, id: number, e: React.PointerEvent) => {
    e.preventDefault(); e.stopPropagation();
    setDraggingIdx({ type, id });
    lastPointerPos.current = { x: e.clientX, y: e.clientY };
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!containerRef.current) return;
    const r = containerRef.current.getBoundingClientRect();

    if (!draggingIdx && e.buttons === 1) {
      // Move design in DESIGN mode OR CALIBRATE mode (only if in basic calibrate sub-mode to avoid mesh conflict)
      if (editorMode === 'DESIGN' || activeMode === 'calibrate') {
        const dx = (e.clientX - lastPointerPos.current.x) / (r.width * zoom);
        const dy = (e.clientY - lastPointerPos.current.y) / (r.height * zoom);
        setDesignOffsetX(prev => prev + dx);
        setDesignOffsetY(prev => prev + dy);
      }
      lastPointerPos.current = { x: e.clientX, y: e.clientY };
      return;
    }

    if (!draggingIdx) {
      lastPointerPos.current = { x: e.clientX, y: e.clientY };
      return;
    }

    // Reverse map coordinates taking zoom/pan into account
    const rawX = (e.clientX - r.left - pan.x) / (r.width * zoom);
    const rawY = (e.clientY - r.top - pan.y) / (r.height * zoom);
    const x = Math.max(0, Math.min(1, rawX));
    const y = Math.max(0, Math.min(1, rawY));

    if (draggingIdx.type === 'base') {
      const next = [...basePoints]; next[draggingIdx.id] = { x, y }; setBasePoints(next);
    } else if (draggingIdx.type === 'mask') {
      const next = [...maskPoints]; next[draggingIdx.id] = { x, y }; setMaskPoints(next);
    } else if (draggingIdx.type === 'mesh') {
      const next = [...meshPoints]; next[draggingIdx.id].dst = { x, y }; setMeshPoints(next);
    } else if (draggingIdx.type === 'smile') {
      // Smile: moves curveTop and curveBot together (uniform curvature)
      const straightMidY = (calibPts[0].y + calibPts[1].y) / 2;
      const dy = (y - straightMidY);
      const curveVal = (dy / (H_px / (naturalSize.h || 1000) * 0.12)) * 100;
      setCurveTop(curveVal);
      setCurveBot(curveVal);
      // Auto-enable cylinder if not active
      if (curvePct === 0) setCurvePct(60);
    } else if (draggingIdx.type === 'pitch') {
      // Pitch: controls bottom edge curvature independently
      const straightMidY = (calibPts[3].y + calibPts[2].y) / 2;
      const dy = (y - straightMidY);
      const curveVal = (dy / (H_px / (naturalSize.h || 1000) * 0.12)) * 100;
      setCurveBot(curveVal);
      // Auto-enable cylinder if not active
      if (curvePct === 0) setCurvePct(60);
    }
    lastPointerPos.current = { x: e.clientX, y: e.clientY };
  }, [draggingIdx, editorMode, zoom, pan, basePoints, maskPoints, meshPoints, calibPts, H_px, naturalSize, curvePct]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = -e.deltaY;
    const factor = delta > 0 ? 1.1 : 0.9;
    if (editorMode === 'DESIGN') {
      setDesignScale(prev => Math.max(0.1, Math.min(5, prev * factor)));
    } else {
      setZoom(prev => Math.max(0.5, Math.min(10, prev * factor)));
    }
  };

  const handleContainerMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      setIsPanning(true);
    }
  };

  const handleContainerMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setPan(prev => ({ x: prev.x + e.movementX, y: prev.y + e.movementY }));
    }
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
    const src_canon = [{ x: -1, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }];
    const H_mat = solve_homography(src_canon, calibPts);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const u = c / (cols - 1), v = r / (rows - 1);
      const world = apply_homography(H_mat, u, v);
      pts.push({ src: world, dst: { ...world } });
    }
    setMeshPoints(pts);
  };

  // ── Slider component ──
  const Slider = ({ label, value, min, max, step = 1, unit = '', onChange }: {
    label: string; value: number; min: number; max: number; step?: number; unit?: string; onChange: (v: number) => void;
  }) => (
    <div className="ctrl">
      <div className="ctrl-header">
        <span className="ctrl-label">{label}</span>
        <span className="ctrl-value">{value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))} className="slider" />
    </div>
  );

  // ── Control groups ──
  const renderControls = () => {
    if (editorMode === 'DESIGN') return (
      <div className="ctrl-section">
        <div className="ctrl-header"><span className="ctrl-label">Design Transformation</span></div>
        <div className="ctrl-grid">
          <Slider label="Artwork Scale" value={designScale} min={0.1} max={5} step={0.01} unit="x" onChange={setDesignScale} />
          <Slider label="Offset X (Rotate)" value={designOffsetX} min={-1} max={1} step={0.001} unit="" onChange={setDesignOffsetX} />
          <Slider label="Offset Y" value={designOffsetY} min={-1} max={1} step={0.001} unit="" onChange={setDesignOffsetY} />
        </div>
        <p className="mode-hint">Kéo trực tiếp trên ảnh để di chuyển (360° Wrap). Dùng phím cuộn chuột để Zoom.</p>
        <button className="btn-ghost" style={{ marginTop: 12 }} onClick={() => { setDesignScale(1); setDesignOffsetX(0); setDesignOffsetY(0); }}>
          ↺ Reset Design Position
        </button>
      </div>
    );

    if (activeMode === 'mesh') return (
      <div className="ctrl-section">
        <p className="mode-hint">Click trên canvas để thêm điểm. Kéo chấm cam để biến dạng bề mặt.</p>
        <div className="btn-row">
          <button className="btn-primary" onClick={make4x4Mesh}>⊞ Lưới 4×4</button>
          <button className="btn-ghost" onClick={() => setMeshPoints([])}>↺ Reset Mesh</button>
        </div>
      </div>
    );
    if (activeMode === 'mask') return (
      <div className="ctrl-section">
        <p className="mode-hint">Click để vẽ vùng mask (tay cầm, vật cản). Đóng vòng để hoàn thành.</p>
        <div className="btn-row">
          <button className="btn-ghost" onClick={() => setMaskPoints([])}>↺ Xóa Mask</button>
          {maskPoints.length > 2 && <button className="btn-primary" onClick={() => setMaskPoints([...maskPoints, maskPoints[0]])}>✓ Đóng vòng</button>}
        </div>
      </div>
    );

    // Calibrate mode — grouped
    return (
      <div className="ctrl-section">
        <div className="group-tabs">
          {(['geometry', 'wrap', 'blend', 'advanced'] as const).map(g => (
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
            <Slider label="Curve Bottom" value={curveBot} min={-100} max={100} unit="%" onChange={setCurveBot} />
          </div>
        )}
        {activeGroup === 'blend' && (
          <div className="ctrl-grid">
            <Slider label="Feather Radius" value={featherRadius} min={0} max={12} unit="px" onChange={setFeatherRadius} />
            <div className="ctrl">
              <div className="ctrl-header"><span className="ctrl-label">Blend Mode</span></div>
              <div className="blend-preview">Multiply 85% + Highlight 15%</div>
            </div>
          </div>
        )}
        {activeGroup === 'advanced' && (
          <div className="ctrl-grid">
            <div className="ctrl">
              <div className="ctrl-header"><span className="ctrl-label">Product Type</span></div>
              <select className="select" value={productType} onChange={e => setProductType(e.target.value)}>
                <option value="cylinder_ceramic">🍵 Ceramic Mug</option>
                <option value="apparel_cotton">👕 T-Shirt</option>
                <option value="plastic_case">📱 Plastic Case</option>
                <option value="fabric_bag">👜 Fabric Bag</option>
                <option value="flat_print">🖼 Flat Print</option>
              </select>
            </div>
            <div className="ctrl">
              <div className="ctrl-header"><span className="ctrl-label">Reset All</span></div>
              <button className="btn-ghost" style={{ marginTop: 8 }} onClick={() => {
                setTilt(0); setRotate(0); setPerspective(0);
                setCurvePct(0); setCurveTop(0); setCurveBot(0);
                setDesignScale(1.0); setDesignOffsetX(0); setDesignOffsetY(0);
              }}>
                ↺ Reset Calibration
              </button>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="pae-root">

      {/* ── TOP TOOLBAR ── */}
      <div className="toolbar">
        <div className="toolbar-left">
          <div className="logo-badge">Print Area Editor</div>
          <div className="workflow-status">
            {editorMode === 'CALIBRATE' ? (
              <button className="btn-confirm-lock" onClick={() => setEditorMode('DESIGN')}>
                ✅ Confirm & Lock Print Area
              </button>
            ) : (
              <button className="btn-unlock" onClick={() => setEditorMode('CALIBRATE')}>
                🔓 Unlock to Calibrate
              </button>
            )}
          </div>
          <div className="mode-sub-pills" style={{ marginLeft: 20, display: 'flex', gap: 4, visibility: editorMode === 'CALIBRATE' ? 'visible' : 'hidden' }}>
            {(['calibrate', 'mesh', 'mask'] as const).map(m => (
              <button key={m} className={`pill pill-sm ${activeMode === m ? 'active' : ''}`} onClick={() => setActiveMode(m)}>
                {m.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div className="toolbar-right">
          <div className="zoom-info">
            Zoom: {Math.round(zoom * 100)}%
            <button className="btn-mini" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>Reset</button>
          </div>
          <button className={`icon-btn ${showGrid ? 'active' : ''}`} onClick={() => setShowGrid(v => !v)} title="Toggle Grid">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M1 5h14M1 11h14M5 1v14M11 1v14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
            Grid
          </button>
          <button className="btn-detect" onClick={handleAutoDetect} disabled={isDetecting}>
            {isDetecting ? <span className="spinner" /> : '✦'}
            {isDetecting ? 'Detecting…' : 'Smart Detect'}
          </button>
        </div>
      </div>

      {/* ── MAIN: CANVAS ── */}
      <div className="canvas-section">
        <div className="canvas-wrap" ref={containerRef}
          onPointerMove={handlePointerMove}
          onPointerUp={() => setDraggingIdx(null)}
          onWheel={handleWheel}
          onMouseDown={handleContainerMouseDown}
          onMouseMove={handleContainerMouseMove}
          onMouseUp={handleContainerMouseUp}
          onMouseLeave={handleContainerMouseUp}
          onClick={handleCanvasClick}
          style={{
            cursor: isPanning ? 'grabbing' : (activeMode !== 'calibrate' ? 'crosshair' : 'default'),
            overflow: 'hidden'
          }}
        >
          <div className="canvas-container" style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: '0 0',
            width: '100%', height: '100%',
            position: 'relative',
            ['--zoom' as any]: zoom
          }}>
            <img
              src={imageUrl} alt="Mockup"
              className="mockup-img"
              onLoad={e => setNaturalSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
            />
            {previewOverlayUrl && <img src={previewOverlayUrl} className="overlay-img" alt="" />}

            <svg className="overlay-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
              <defs>
                <clipPath id="qclip">
                  <path d={curvedBoundaryPath} />
                </clipPath>
              </defs>

              {/* Single Unified Curved Grid Boundary */}
              <path d={curvedBoundaryPath} className="grid-boundary-unified" fill="transparent" pointerEvents="none" />

              {/* Warped grid clipped to curved boundary */}
              {showGrid && (
                <g clipPath="url(#qclip)">
                  {gridLines.v.map((pts, i) => (
                    <polyline key={`vf${i}`} points={pts} className="grid-line" />
                  ))}
                  {gridLines.h.map((pts, i) => (
                    <polyline key={`hf${i}`} points={pts} className="grid-line" />
                  ))}
                </g>
              )}

              {/* Curved Boundary Outline */}
              <path d={curvedBoundaryPath} className="grid-boundary-unified" fill="transparent" pointerEvents="none" />
            </svg>

            {/* Calibration Handles (Only in CALIBRATE mode) */}
            {editorMode === 'CALIBRATE' && (
              <>
                {/* Corners — 4 small dot handles */}
                {activeMode === 'calibrate' && calibPts.map((p, i) => (
                  <div key={i} className="handle handle-corner-dot" style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
                    onPointerDown={e => handlePointerDown('base', i, e)} />
                ))}

                {/* Curvature Handles — both on TOP edge */}
                {activeMode === 'calibrate' && (
                  <>
                    {/* Smile handle (cyan) — uniform curvature */}
                    <div className="handle handle-smile-dot" style={{ left: `${smileHandle.x * 100}%`, top: `${smileHandle.y * 100}%` }}
                      onPointerDown={e => handlePointerDown('smile', 0, e)} />
                    {/* Pitch handle (orange) — differential curvature */}
                    <div className="handle handle-pitch-dot" style={{ left: `${pitchHandle.x * 100}%`, top: `${pitchHandle.y * 100}%` }}
                      onPointerDown={e => handlePointerDown('pitch', 0, e)} />
                  </>
                )}

                {/* Mask handles */}
                {activeMode === 'mask' && maskPoints.map((p, i) => (
                  <div key={i} className="handle handle-mask" style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
                    onPointerDown={e => handlePointerDown('mask', i, e)} />
                ))}

                {/* Mesh handles */}
                {activeMode === 'mesh' && meshPoints.map((m, i) => (
                  <div key={i} className="handle handle-mesh" style={{ left: `${m.dst.x * 100}%`, top: `${m.dst.y * 100}%` }}
                    onPointerDown={e => handlePointerDown('mesh', i, e)} />
                ))}

                {/* Corner coordinate readout */}
                {naturalSize.w > 0 && (
                  <div className="coord-readout">
                    {calibPts.map((p, i) => (
                      <span key={i}>{['TL', 'TR', 'BR', 'BL'][i]} ({Math.round(p.x * naturalSize.w)}, {Math.round(p.y * naturalSize.h)})</span>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* ── BOTTOM: CONTROLS ── */}
      <div className="controls-panel">
        {renderControls()}
      </div>

      {apiError && (
        <div className="error-toast">
          <span>⚠️ {apiError}</span>
          <button className="btn-ghost" style={{ padding: '2px 8px', color: 'white' }} onClick={() => setApiError(null)}>Dismiss</button>
        </div>
      )}

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

        .pae-root {
          display: flex;
          flex-direction: column;
          height: 100vh;
          min-height: 700px;
          background: #0a0c10;
          color: #e2e8f0;
          font-family: 'Syne', sans-serif;
          overflow: hidden;
        }

        /* ── TOOLBAR ── */
        .toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 20px;
          height: 52px;
          background: #0e1117;
          border-bottom: 1px solid #1e2330;
          flex-shrink: 0;
          gap: 12px;
        }
        .toolbar-left, .toolbar-right { display: flex; align-items: center; gap: 12px; }
        .logo-badge {
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #475569;
          padding: 4px 10px;
          border: 1px solid #1e2330;
          border-radius: 6px;
        }
        .mode-pills { display: flex; gap: 4px; }
        .pill {
          padding: 5px 14px;
          border: none;
          background: transparent;
          color: #475569;
          cursor: pointer;
          border-radius: 6px;
          font-family: 'Syne', sans-serif;
          font-size: 12px;
          font-weight: 600;
          letter-spacing: 0.04em;
          transition: all 0.15s;
        }
        .pill-sm { font-size: 10px; padding: 4px 10px; }
        .pill:hover { background: #161b27; color: #94a3b8; }
        .pill.active { background: #1a2540; color: #60a5fa; }
        .icon-btn {
          display: flex; align-items: center; gap: 6px;
          padding: 5px 12px;
          border: 1px solid #1e2330;
          background: transparent;
          color: #475569;
          cursor: pointer;
          border-radius: 6px;
          font-family: 'Syne', sans-serif;
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          transition: all 0.15s;
        }
        .icon-btn:hover { border-color: #2d3a52; color: #94a3b8; }
        .icon-btn.active { border-color: #3b82f6; color: #60a5fa; background: #1a2540; }
        .btn-detect {
          display: flex; align-items: center; gap: 8px;
          padding: 7px 18px;
          background: linear-gradient(135deg, #1d4ed8, #2563eb);
          color: white;
          border: none;
          border-radius: 8px;
          font-family: 'Syne', sans-serif;
          font-size: 12px;
          font-weight: 700;
          cursor: pointer;
          letter-spacing: 0.04em;
          transition: opacity 0.15s;
        }
        .btn-detect:disabled { opacity: 0.5; cursor: not-allowed; }
        .spinner {
          width: 12px; height: 12px;
          border: 2px solid rgba(255,255,255,0.3);
          border-top-color: white;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
          display: inline-block;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ── CANVAS ── */
        .canvas-section {
          flex: 1;
          min-height: 0;
          background: #ffffff; /* Use white to be safe, or just transparent */
          display: flex;
          align-items: center;
          justify-content: center;
          position: relative;
          overflow: hidden;
        }
        .canvas-wrap {
          position: relative;
          max-width: 100%;
          max-height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .mockup-img {
          display: block;
          max-width: 100%;
          max-height: calc(100vh - 52px - 160px);
          object-fit: contain;
          user-select: none;
        }
        .overlay-img {
          position: absolute;
          top: 0; left: 0;
          width: 100%; height: 100%;
          object-fit: contain;
          pointer-events: none;
          /* Removed mix-blend-mode: multiply since backend returns full composite */
        }
        .overlay-svg {
          position: absolute;
          top: 0; left: 0;
          width: 100%; height: 100%;
          pointer-events: none;
          z-index: 10;
        }
        .grid-v, .grid-h {
          fill: none;
          stroke: #00f2ff;
          stroke-width: 0.15; /* Ultra thin */
          vector-effect: non-scaling-stroke;
        }
        .grid-v-bg, .grid-h-bg {
          display: none;
        }

        .quad-outline {
          fill: none;
          stroke: #000;
          stroke-width: 0.4;
          stroke-dasharray: 2 4;
          opacity: 0.5;
        }
        
        .handle {
          position: absolute;
          cursor: pointer;
          z-index: 100;
          display: flex; align-items: center; justify-content: center;
          transition: transform 0.1s;
          transform: translate(-50%, -50%) scale(calc(1 / var(--zoom, 1)));
          background: transparent;
        }
        .handle:hover { transform: translate(-50%, -50%) scale(calc(1.4 / var(--zoom, 1))); }

        /* Corner dots — small cyan circles */
        .handle-corner-dot {
          width: 14px; height: 14px; /* Hit area */
        }
        .handle-corner-dot::after {
          content: '';
          width: 5px; height: 5px;
          background: #00f2ff;
          border-radius: 50%;
          border: 1px solid rgba(255,255,255,0.6);
          box-shadow: 0 0 3px rgba(0,242,255,0.5);
        }

        /* Smile handle (cyan) */
        .handle-smile-dot {
          width: 16px; height: 16px; /* Hit area */
        }
        .handle-smile-dot::after {
          content: '';
          width: 5px; height: 5px;
          background: #00f2ff;
          border-radius: 50%;
          border: 1px solid white;
          box-shadow: 0 0 4px rgba(0,242,255,0.6);
        }

        /* Pitch handle (orange) */
        .handle-pitch-dot {
          width: 16px; height: 16px; /* Hit area */
        }
        .handle-pitch-dot::after {
          content: '';
          width: 5px; height: 5px;
          background: #f59e0b;
          border-radius: 50%;
          border: 1px solid white;
          box-shadow: 0 0 4px rgba(245,158,11,0.6);
        }

        .grid-boundary-unified {
          fill: none;
          stroke: #00f2ff;
          stroke-width: 0.3;
          opacity: 0.7;
          pointer-events: none;
        }
        .grid-line {
          fill: none;
          stroke: #00f2ff;
          stroke-width: 0.15;
          opacity: 0.4;
        }

        /* ── HANDLES ── */
        .handle-base {
          width: 14px; height: 14px;
          background: #2563eb;
          border: 1.5px solid #93c5fd;
          border-radius: 50%;
          box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
        }
        .handle-label {
          position: absolute;
          top: -20px;
          font-size: 10px;
          color: #60a5fa;
          white-space: nowrap;
          pointer-events: none;
          opacity: 0.6;
        }
        .handle:active { cursor: grabbing; }
        .handle-base {
          width: 20px; height: 20px;
          background: #2563eb;
          border: 2px solid #93c5fd;
          border-radius: 50%;
          box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2), 0 2px 8px rgba(0,0,0,0.4);
          transition: transform 0.1s;
        }
        .handle-base:hover { transform: translate(-50%, -50%) scale(1.25); }
        .handle-label {
          position: absolute;
          top: -18px;
          font-size: 9px;
          font-family: 'DM Mono', monospace;
          font-weight: 500;
          color: #60a5fa;
          letter-spacing: 0.06em;
          white-space: nowrap;
          pointer-events: none;
        }
        .handle-mask {
          width: 14px; height: 14px;
          background: #059669;
          border: 2px solid #6ee7b7;
          border-radius: 3px;
          box-shadow: 0 0 0 2px rgba(5, 150, 105, 0.25);
        }
        .handle-mesh {
          width: 12px; height: 12px;
          background: #d97706;
          border: 2px solid #fcd34d;
          border-radius: 50%;
          box-shadow: 0 0 6px rgba(245, 158, 11, 0.5);
        }

        .coord-readout {
          position: absolute;
          bottom: 8px;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          gap: 12px;
          background: rgba(0,0,0,0.6);
          padding: 4px 12px;
          border-radius: 6px;
          backdrop-filter: blur(8px);
          pointer-events: none;
        }
        .coord-readout span {
          font-family: 'DM Mono', monospace;
          font-size: 10px;
          color: #64748b;
        }

        /* ── CONTROLS PANEL ── */
        .controls-panel {
          flex-shrink: 0;
          height: 150px;
          background: #0e1117;
          border-top: 1px solid #1e2330;
          display: flex;
          align-items: stretch;
        }
        .ctrl-section {
          flex: 1;
          padding: 16px 24px;
          display: flex;
          flex-direction: column;
          gap: 12px;
          min-width: 0;
        }

        /* Group tabs */
        .group-tabs {
          display: flex;
          gap: 2px;
          border-bottom: 1px solid #1e2330;
          padding-bottom: 12px;
          flex-shrink: 0;
        }
        .group-tab {
          padding: 4px 14px;
          background: transparent;
          border: 1px solid transparent;
          color: #475569;
          cursor: pointer;
          border-radius: 6px;
          font-family: 'Syne', sans-serif;
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.05em;
          transition: all 0.15s;
        }
        .group-tab:hover { color: #94a3b8; background: #141924; }
        .group-tab.active { border-color: #1e2d44; color: #60a5fa; background: #111827; }

        /* Sliders grid */
        .ctrl-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 16px 32px;
          flex: 1;
          align-content: start;
        }
        .ctrl { display: flex; flex-direction: column; gap: 6px; }
        .ctrl-header { display: flex; justify-content: space-between; align-items: baseline; }
        .ctrl-label {
          font-size: 10px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: #475569;
        }
        .ctrl-value {
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          color: #60a5fa;
        }
        .slider {
          -webkit-appearance: none;
          appearance: none;
          width: 100%;
          height: 3px;
          background: #1e2330;
          border-radius: 2px;
          outline: none;
          cursor: pointer;
        }
        .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 14px; height: 14px;
          background: #3b82f6;
          border-radius: 50%;
          border: 2px solid #93c5fd;
          cursor: grab;
          box-shadow: 0 0 6px rgba(59, 130, 246, 0.4);
          transition: transform 0.1s;
        }
        .slider::-webkit-slider-thumb:active { transform: scale(1.3); cursor: grabbing; }
        .slider::-moz-range-thumb {
          width: 14px; height: 14px;
          background: #3b82f6;
          border-radius: 50%;
          border: 2px solid #93c5fd;
          cursor: grab;
        }

        .select {
          background: #141924;
          color: #94a3b8;
          border: 1px solid #1e2330;
          padding: 6px 10px;
          border-radius: 6px;
          font-family: 'Syne', sans-serif;
          font-size: 12px;
          cursor: pointer;
          width: 100%;
        }
        .select:focus { outline: none; border-color: #3b82f6; }

        .blend-preview {
          background: #141924;
          border: 1px solid #1e2330;
          padding: 6px 10px;
          border-radius: 6px;
          font-size: 11px;
          color: #475569;
          font-family: 'DM Mono', monospace;
        }

        /* Mode toolsets */
        .mode-hint {
          font-size: 12px;
          color: #64748b;
          line-height: 1.5;
          max-width: 500px;
        }
        .btn-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
        .btn-primary {
          padding: 8px 18px;
          background: #1d4ed8;
          color: white;
          border: none;
          border-radius: 7px;
          font-family: 'Syne', sans-serif;
          font-size: 12px;
          font-weight: 700;
          cursor: pointer;
          transition: background 0.15s;
        }
        .btn-primary:hover { background: #2563eb; }
        .btn-ghost {
          padding: 8px 18px;
          background: transparent;
          color: #94a3b8;
          border: 1px solid #1e2330;
          border-radius: 7px;
          font-family: 'Syne', sans-serif;
          font-size: 12px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.15s;
        }
        .btn-ghost:hover { background: #141924; color: #e2e8f0; }

        .btn-confirm-lock {
          background: #3b82f6;
          color: white;
          border: none;
          padding: 6px 16px;
          border-radius: 6px;
          font-weight: 700;
          cursor: pointer;
          font-size: 13px;
          transition: all 0.2s;
        }
        .btn-confirm-lock:hover {
          background: #2563eb;
          transform: scale(1.05);
        }
        .btn-unlock {
          background: #374151;
          color: #9ca3af;
          border: 1px solid #4b5563;
          padding: 6px 16px;
          border-radius: 6px;
          font-weight: 600;
          cursor: pointer;
          font-size: 13px;
        }
        .btn-unlock:hover {
          color: white;
          border-color: #6b7280;
        }

        .quad-outline {
          fill: none;
          stroke: #3b82f6;
          stroke-width: 0.25;
          stroke-dasharray: 2 4;
          opacity: 0.4;
        }
      `}</style>
    </div>
  );
}