import React, { useState, useRef, useEffect, useCallback, useLayoutEffect } from 'react';
import { MugCurvePreview } from './MugCurvePreview';
import { WarpGridPreview } from './WarpGridPreview';

interface Point { x: number; y: number; }
interface MeshPoint { src: Point; dst: Point; }
interface PrintAreaEditorProps {
  imageUrl: string;
  designFile?: File | null;
  designUrl?: string | null;
  warpConfig: any;
  initialPrintArea?: any;
  onCoordinatesChange: (data: any) => void;
  onConfigChange: (config: any) => void;
  onLockedSnapshotChange?: (snapshot: { printArea: any; warpConfig: any } | null) => void;
  onLiveSnapshotChange?: (snapshot: { printArea: any; warpConfig: any } | null) => void;
  productTypeProp?: string;
  onProductTypeChange?: (val: string) => void;
}

const BASE_CENTER_BAND = 0.30;
const MIN_CENTER_BAND = 0.10;
const MAX_CENTER_BAND = 0.70;
const BASE_EDGE_ROLL_START = 0.55;
const DEFAULT_MESH_DENSITY_STRENGTH = 2;
const DEFAULT_POINT_NUDGE_STEP = 0.001; // 0.1% in normalized coordinates
const MAX_MESH_DISPLACEMENT = 0.45; // 45% of width/height in normalized coords
const MIN_MESH_TRIANGLE_AREA = 1e-5;

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function signedTriangleArea(a: Point, b: Point, c: Point): number {
  return 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

function getStructuredMeshSide(meshPoints: MeshPoint[]): number {
  const side = Math.round(Math.sqrt(meshPoints.length));
  return side * side === meshPoints.length ? side : 0;
}

function limitMeshDisplacement(meshPoint: MeshPoint, desired: Point): Point {
  const x = clamp01(desired.x);
  const y = clamp01(desired.y);
  const dx = x - meshPoint.src.x;
  const dy = y - meshPoint.src.y;
  const dist = Math.hypot(dx, dy);
  if (dist <= MAX_MESH_DISPLACEMENT) {
    return { x, y };
  }
  const scale = MAX_MESH_DISPLACEMENT / dist;
  return {
    x: clamp01(meshPoint.src.x + dx * scale),
    y: clamp01(meshPoint.src.y + dy * scale),
  };
}

function isValidStructuredMeshCell(meshPoints: MeshPoint[], row: number, col: number, side: number): boolean {
  const i00 = row * side + col;
  const i01 = row * side + col + 1;
  const i11 = (row + 1) * side + col + 1;
  const i10 = (row + 1) * side + col;

  const a = meshPoints[i00];
  const b = meshPoints[i01];
  const c = meshPoints[i11];
  const d = meshPoints[i10];
  if (!a || !b || !c || !d) return false;

  const srcTriA = signedTriangleArea(a.src, b.src, c.src);
  const srcTriB = signedTriangleArea(a.src, c.src, d.src);
  const dstTriA = signedTriangleArea(a.dst, b.dst, c.dst);
  const dstTriB = signedTriangleArea(a.dst, c.dst, d.dst);

  if (Math.abs(dstTriA) <= MIN_MESH_TRIANGLE_AREA || Math.abs(dstTriB) <= MIN_MESH_TRIANGLE_AREA) {
    return false;
  }

  return Math.sign(dstTriA) === Math.sign(srcTriA) && Math.sign(dstTriB) === Math.sign(srcTriB);
}

function isValidStructuredMeshAround(meshPoints: MeshPoint[], movedIndex: number): boolean {
  const side = getStructuredMeshSide(meshPoints);
  if (side < 2 || movedIndex < 0 || movedIndex >= meshPoints.length) return true;

  const row = Math.floor(movedIndex / side);
  const col = movedIndex % side;
  const rowStart = Math.max(0, row - 1);
  const rowEnd = Math.min(side - 2, row);
  const colStart = Math.max(0, col - 1);
  const colEnd = Math.min(side - 2, col);

  for (let r = rowStart; r <= rowEnd; r += 1) {
    for (let c = colStart; c <= colEnd; c += 1) {
      if (!isValidStructuredMeshCell(meshPoints, r, c, side)) {
        return false;
      }
    }
  }
  return true;
}

function constrainStructuredMeshDestination(meshPoints: MeshPoint[], movedIndex: number, desired: Point): Point {
  const candidate = { x: clamp01(desired.x), y: clamp01(desired.y) };
  const side = getStructuredMeshSide(meshPoints);
  if (side < 2 || movedIndex < 0 || movedIndex >= meshPoints.length) {
    return candidate;
  }

  const current = meshPoints[movedIndex]?.dst ?? candidate;
  const isCandidateValid = (point: Point) => {
    const next = meshPoints.map((meshPoint, index) => (
      index === movedIndex ? { ...meshPoint, dst: point } : meshPoint
    ));
    return isValidStructuredMeshAround(next, movedIndex);
  };

  if (isCandidateValid(candidate)) {
    return candidate;
  }

  let low = 0;
  let high = 1;
  let best = current;
  for (let i = 0; i < 12; i += 1) {
    const mid = (low + high) * 0.5;
    const probe = {
      x: current.x + (candidate.x - current.x) * mid,
      y: current.y + (candidate.y - current.y) * mid,
    };
    if (isCandidateValid(probe)) {
      best = probe;
      low = mid;
    } else {
      high = mid;
    }
  }

  return { x: clamp01(best.x), y: clamp01(best.y) };
}

function canNudgeHandle(
  selectedHandle: { type: string; id: number } | null,
  activeMode: 'calibrate' | 'mesh' | 'mask',
  activeGroup: 'geometry' | 'wrap' | 'blend' | 'edge' | 'advanced',
  productType: string,
): boolean {
  if (!selectedHandle) return false;
  if (activeMode === 'mesh') return selectedHandle.type === 'mesh';
  if (activeMode === 'mask') return selectedHandle.type === 'mask';
  if (activeMode !== 'calibrate') return false;
  if (selectedHandle.type === 'base') return true;
  if (selectedHandle.type === 'light') return activeGroup === 'blend' && productType.startsWith('cylinder');
  return false;
}

// Math helpers

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

// Homography helpers (Parity with OpenCV)

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

function apply_projective_homography(H: number[], x: number, y: number): Point {
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w
  };
}

function apply_homography(H: number[], u: number, v: number): Point {
  const x = u * 2 - 1, y = v * 2 - 1;
  return apply_projective_homography(H, x, y);
}

function computeCenterBand(centerFocusWidth: number): number {
  const widthStrength = Math.max(-1, Math.min(1, centerFocusWidth));
  if (widthStrength >= 0) {
    return BASE_CENTER_BAND + (MAX_CENTER_BAND - BASE_CENTER_BAND) * widthStrength;
  }
  return BASE_CENTER_BAND + (BASE_CENTER_BAND - MIN_CENTER_BAND) * widthStrength;
}

function applyCenterFocusWidth(radius: number, centerFocusWidth: number): number {
  const widthStrength = Math.max(-1, Math.min(1, centerFocusWidth));
  if (Math.abs(widthStrength) <= 1e-8) return radius;

  const sourceBand = BASE_CENTER_BAND;
  const targetBand = computeCenterBand(centerFocusWidth);

  if (radius <= sourceBand) {
    const innerRatio = Math.max(0, Math.min(1, radius / Math.max(sourceBand, 1e-8)));
    return targetBand * innerRatio;
  }

  const outerRatio = Math.max(0, Math.min(1, (radius - sourceBand) / Math.max(1 - sourceBand, 1e-8)));
  return targetBand + (1 - targetBand) * outerRatio;
}

function applyEdgeRoll(radius: number, edgeSqueeze: number, squeezePower: number, centerFocusWidth: number): number {
  const blend = Math.max(0, Math.min(1, edgeSqueeze));
  if (blend <= 1e-8) return radius;

  const protectedCenterBand = computeCenterBand(centerFocusWidth);
  const edgeStart = Math.min(Math.max(BASE_EDGE_ROLL_START, protectedCenterBand), 0.95);
  if (edgeStart >= 1 - 1e-8) return radius;

  const power = Math.max(1, squeezePower);
  const progress = Math.max(0, Math.min(1, (radius - edgeStart) / Math.max(1 - edgeStart, 1e-8)));
  const rolledProgress = Math.pow(progress, 1 / power);
  const mappedProgress = progress + (rolledProgress - progress) * blend;
  if (radius <= edgeStart) return radius;
  return edgeStart + (1 - edgeStart) * mappedProgress;
}

function applyHorizontalSqueeze(t: number, edgeSqueeze: number, squeezePower: number, centerFocusWidth: number): number {
  const blend = Math.max(0, Math.min(1, edgeSqueeze));
  const width = Math.max(-1, Math.min(1, centerFocusWidth));
  if (blend <= 1e-8 && Math.abs(width) <= 1e-8) return t;
  const radius = Math.abs(t);
  const widthAdjustedRadius = applyCenterFocusWidth(radius, width);
  const mappedRadius = applyEdgeRoll(widthAdjustedRadius, blend, squeezePower, width);
  return Math.max(-1, Math.min(1, Math.sign(t) * mappedRadius));
}

function cylinderWarpPoint(
  u: number,
  v: number,
  curvePct: number,
  curveTop: number,
  curveBot: number,
  edgeSqueeze: number,
  squeezePower: number,
  centerFocusWidth: number,
  hPx: number,
  wPx: number
): Point {
  const thetaMaxDeg = curvePct * 0.9;
  if (thetaMaxDeg <= 0.001) return { x: u, y: v };
  const thetaMax = (thetaMaxDeg * Math.PI) / 180;
  const nx = (u - 0.5) * 2;
  const sinThetaMax = Math.sin(thetaMax);
  const sinTheta = Math.max(-1, Math.min(1, nx * sinThetaMax));
  const theta = Math.asin(sinTheta);
  const t = theta / thetaMax;
  const tFinal = applyHorizontalSqueeze(t, edgeSqueeze, squeezePower, centerFocusWidth);
  const wx = (tFinal + 1) * 0.5;

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

function edgeLengthPx(a: Point, b: Point, imageW: number, imageH: number): number {
  const dx = (b.x - a.x) * imageW;
  const dy = (b.y - a.y) * imageH;
  return Math.hypot(dx, dy);
}

function toFiniteNumber(input: any, fallback: number): number {
  return typeof input === 'number' && Number.isFinite(input) ? input : fallback;
}

function getStepPrecision(step: number): number {
  const stepText = String(step);
  const dotIndex = stepText.indexOf('.');
  return dotIndex >= 0 ? stepText.length - dotIndex - 1 : 0;
}

function clampControlValue(value: number, min: number, max: number, step: number): number {
  const precision = getStepPrecision(step);
  const clamped = Math.max(min, Math.min(max, value));
  return Number(clamped.toFixed(precision));
}

function buildHorizontalSqueezeDebug(edgeSqueeze: number, squeezePower: number, centerFocusWidth: number) {
  const sampleIn = [0.1, 0.2, 0.4, 0.6, 0.8];
  const sampleOut = sampleIn.map((sample) => Number(applyHorizontalSqueeze(sample, edgeSqueeze, squeezePower, centerFocusWidth).toFixed(4)));
  const sampleDelta = sampleOut.map((mapped, index) => Number((mapped - sampleIn[index]).toFixed(4)));

  const hasEdge = edgeSqueeze > 1e-8;
  const hasWidth = Math.abs(centerFocusWidth) > 1e-8;
  let inactiveReason: string | null = null;
  let mode = 'edge_and_width_active';
  if (!hasEdge && !hasWidth) {
    inactiveReason = 'edge_and_center_zero';
    mode = 'inactive';
  } else if (!hasEdge) {
    mode = 'width_only_active';
  } else if (!hasWidth) {
    mode = 'edge_only_active';
  }

  return {
    active: hasEdge || hasWidth,
    inactiveReason,
    mode,
    sampleIn,
    sampleOut,
    sampleDelta,
  };
}

function getDesignFilenameFromUrl(url: string): string {
  try {
    const parsed = new URL(url, window.location.href);
    const basename = parsed.pathname.split('/').filter(Boolean).pop();
    if (basename) return decodeURIComponent(basename);
  } catch {
    // Fall through to the string fallback below.
  }

  const fallback = url.split('?')[0]?.split('#')[0]?.split('/').filter(Boolean).pop();
  return fallback || 'artwork.png';
}

// Component

interface ControlAdjusterProps {
  label: string;
  description?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (v: number) => void;
}

function ControlAdjuster({
  label,
  description,
  value,
  min,
  max,
  step = 0.1,
  unit = '',
  onChange,
}: ControlAdjusterProps) {
  const precision = getStepPrecision(step);
  const formattedValue = value.toFixed(precision);
  const [draft, setDraft] = useState(formattedValue);

  useEffect(() => {
    setDraft(formattedValue);
  }, [formattedValue]);

  const nudge = (direction: -1 | 1) => {
    onChange(clampControlValue(value + direction * step, min, max, step));
  };

  const commitDraft = () => {
    const normalized = draft.replace(',', '.').trim();
    if (normalized === '' || normalized === '-' || normalized === '+') {
      setDraft(formattedValue);
      return;
    }
    const parsed = Number(normalized);
    if (!Number.isFinite(parsed)) {
      setDraft(formattedValue);
      return;
    }
    const nextValue = clampControlValue(parsed, min, max, step);
    onChange(nextValue);
    setDraft(nextValue.toFixed(precision));
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      e.preventDefault();
      nudge(-1);
      return;
    }
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      e.preventDefault();
      nudge(1);
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      commitDraft();
      e.currentTarget.blur();
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      setDraft(formattedValue);
      e.currentTarget.blur();
    }
  };

  return (
    <div className="ctrl">
      <div className="ctrl-header"><span className="ctrl-label">{label}</span><span className="ctrl-value">{formattedValue}{unit}</span></div>
      <div className="stepper">
        <button
          type="button"
          className="stepper-btn"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => nudge(-1)}
          aria-label={`Giảm ${label}`}
        >
          {'<'}
        </button>
        <input
          className="stepper-input"
          type="text"
          inputMode="decimal"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitDraft}
          onKeyDown={handleInputKeyDown}
          aria-label={label}
        />
        <button
          type="button"
          className="stepper-btn"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => nudge(1)}
          aria-label={`Tăng ${label}`}
        >
          {'>'}
        </button>
      </div>
      <div className="ctrl-step">Bước chỉnh: {step.toFixed(precision)}{unit}</div>
      {description && <div className="ctrl-help">{description}</div>}
    </div>
  );
}

export default function PrintAreaEditor(props: PrintAreaEditorProps) {
  const { imageUrl, designFile, designUrl, warpConfig, initialPrintArea, onCoordinatesChange, onConfigChange, onLockedSnapshotChange, onLiveSnapshotChange, productTypeProp, onProductTypeChange } = props;
  
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
  const [edgeSqueeze, setEdgeSqueeze] = useState(toFiniteNumber(warpConfig?.edge_squeeze, 0));
  const [squeezePower, setSqueezePower] = useState(toFiniteNumber(warpConfig?.squeeze_power, 2));
  const [centerFocusWidth, setCenterFocusWidth] = useState(toFiniteNumber(warpConfig?.center_focus_width, 0));
  const [meshDensityStrength, setMeshDensityStrength] = useState(toFiniteNumber(warpConfig?.mesh_density_strength, DEFAULT_MESH_DENSITY_STRENGTH));
  const [designScale, setDesignScale] = useState(toFiniteNumber(warpConfig?.design_scale, 1.0));
  const [designOffsetX, setDesignOffsetX] = useState(toFiniteNumber(warpConfig?.design_offset_x, 0.0));
  const [designOffsetY, setDesignOffsetY] = useState(toFiniteNumber(warpConfig?.design_offset_y, 0.0));

  const [editorMode, setEditorMode] = useState<'CALIBRATE' | 'DESIGN'>(
    warpConfig?.editor_mode === 'DESIGN' || warpConfig?.is_locked === true ? 'DESIGN' : 'CALIBRATE'
  );
  const [featherRadius, setFeatherRadius] = useState(toFiniteNumber(warpConfig?.feather_radius, 3));
  // Edge (viền) parameters
  const [edgeWidth, setEdgeWidth] = useState(toFiniteNumber(warpConfig?.edge?.edge_width ?? warpConfig?.edge_width ?? 0.18, 0.18));
  const [edgePower, setEdgePower] = useState(toFiniteNumber(warpConfig?.edge?.power ?? warpConfig?.edge_power ?? 1.5, 1.5));
  const [edgeSatLift, setEdgeSatLift] = useState(toFiniteNumber(warpConfig?.edge?.sat_lift ?? warpConfig?.sat_lift ?? 0.55, 0.55));
  const [edgeBlackLift, setEdgeBlackLift] = useState(toFiniteNumber(warpConfig?.edge?.black_lift ?? warpConfig?.black_lift ?? 0.04, 0.04));
  const [edgeShadowStr, setEdgeShadowStr] = useState(toFiniteNumber(warpConfig?.edge?.shadow_str ?? warpConfig?.shadow_str ?? 0.55, 0.55));
  const [edgeShadowFall, setEdgeShadowFall] = useState(toFiniteNumber(warpConfig?.edge?.shadow_fall ?? warpConfig?.shadow_fall ?? 3.0, 3.0));
  // WebGL preview refs
  const edgeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const glStateRef = useRef<any>(null);
  const [normalMapUrl, setNormalMapUrl] = useState<string | null>(null);
  
  const [localProductType, setLocalProductType] = useState('cylinder_ceramic');
  const productType = productTypeProp || localProductType;
  const setProductType = onProductTypeChange || setLocalProductType;

  const [activeMode, setActiveMode] = useState<'calibrate' | 'mesh' | 'mask'>('calibrate');
  const [activeGroup, setActiveGroup] = useState<'geometry' | 'wrap' | 'blend' | 'edge' | 'advanced'>('geometry');
  const [showGrid, setShowGrid] = useState(true);
  const [maskPoints, setMaskPoints] = useState<Point[]>([]);
  const [meshPoints, setMeshPoints] = useState<MeshPoint[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [gridOverlayUrl, setGridOverlayUrl] = useState<string | null>(null);
  const [previewOverlayUrl, setPreviewOverlayUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [draggingIdx, setDraggingIdx] = useState<{ type: string; id: number } | null>(null);
  const [selectedHandle, setSelectedHandle] = useState<{ type: string; id: number } | null>(null);
  const [pointNudgeStep, setPointNudgeStep] = useState(DEFAULT_POINT_NUDGE_STEP);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [renderDevice, setRenderDevice] = useState<'cpu' | 'gpu'>('cpu');
  const [gpuAvailable, setGpuAvailable] = useState(true);
  const [isSwitchingDevice, setIsSwitchingDevice] = useState(false);
  const [resolvedDesignBlob, setResolvedDesignBlob] = useState<Blob | null>(null);
  const didRestoreRef = useRef(false);

  // Lighting controls (allow editing lighting inside the editor)
  const [lightPosX, setLightPosX] = useState(toFiniteNumber(warpConfig?.light_pos_x, 0.62));
  const [lightPosY, setLightPosY] = useState(toFiniteNumber(warpConfig?.light_pos_y, 0.32));
  const [lightHeight, setLightHeight] = useState(toFiniteNumber(warpConfig?.light_height, 55));
  const [lightContrast, setLightContrast] = useState(toFiniteNumber(warpConfig?.light_contrast, 50));
  const [lightHighlight, setLightHighlight] = useState(toFiniteNumber(warpConfig?.light_highlight, 60));
  const [lightSoftness, setLightSoftness] = useState(toFiniteNumber(warpConfig?.light_softness, 55));

  const editorRootRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const controlSectionRef = useRef<HTMLDivElement>(null);
  const controlScrollTopRef = useRef(0);
  const overlayAlphaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const designPreviewRequestSeqRef = useRef(0);
  const gridPreviewRequestSeqRef = useRef(0);
  const calibPts = applyCalibration(basePoints, tilt, rotate, perspective);

  const imageW = naturalSize.w || 1000;
  const imageH = naturalSize.h || 1000;
  const topWidthPx = edgeLengthPx(calibPts[0], calibPts[1], imageW, imageH);
  const bottomWidthPx = edgeLengthPx(calibPts[3], calibPts[2], imageW, imageH);
  const leftHeightPx = edgeLengthPx(calibPts[0], calibPts[3], imageW, imageH);
  const rightHeightPx = edgeLengthPx(calibPts[1], calibPts[2], imageW, imageH);
  const W_px = (topWidthPx + bottomWidthPx) * 0.5;
  const H_px = (leftHeightPx + rightHeightPx) * 0.5;

  const src_canon_coords = [{ x: -1, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }];
  const H_mat = solve_homography(src_canon_coords, calibPts);
  const H_inv_mat = solve_homography(calibPts, src_canon_coords);

  const localSmile = cylinderWarpPoint(0.5, 0, curvePct, curveTop, curveBot, edgeSqueeze, squeezePower, centerFocusWidth, H_px, W_px);
  const smileHandle = apply_homography(H_mat, localSmile.x, localSmile.y);
  const localPitch = cylinderWarpPoint(0.5, 1, curvePct, curveTop, curveBot, edgeSqueeze, squeezePower, centerFocusWidth, H_px, W_px);
  const pitchHandle = apply_homography(H_mat, localPitch.x, localPitch.y);
  const lightHandle = apply_homography(H_mat, lightPosX, lightPosY);
  const designPreviewSource = designFile || resolvedDesignBlob;
  const wantsDesignOverlay = Boolean(designFile || designUrl);
  const isCylinderProduct = productType.includes('cylinder');
  const resolvedWarpType = isCylinderProduct ? 'cylinder' : (meshPoints.length > 0 ? 'tps' : 'perspective');
  const showGridOverlay = showGrid && isCylinderProduct && Boolean(gridOverlayUrl);
  const showDesignOverlay = Boolean(previewOverlayUrl) && wantsDesignOverlay;
  const showEditorGridLayer = showGrid && editorMode === 'CALIBRATE' && !isCylinderProduct;
  const showCylinderMeshOverlay = showGrid && editorMode === 'CALIBRATE' && isCylinderProduct && activeMode === 'mesh' && meshPoints.length > 0;
  const canDragArtwork = wantsDesignOverlay && (editorMode === 'DESIGN' || activeMode === 'calibrate');
  const canNudgeSelectedHandle = canNudgeHandle(selectedHandle, activeMode, activeGroup, productType);
  const selectedPointPosition: Point | null = (() => {
    if (!selectedHandle) return null;
    if (selectedHandle.type === 'base') return basePoints[selectedHandle.id] ?? null;
    if (selectedHandle.type === 'mask') return maskPoints[selectedHandle.id] ?? null;
    if (selectedHandle.type === 'mesh') return meshPoints[selectedHandle.id]?.dst ?? null;
    if (selectedHandle.type === 'light') return lightHandle;
    return null;
  })();

  const captureControlScroll = useCallback(() => {
    if (controlSectionRef.current) {
      controlScrollTopRef.current = controlSectionRef.current.scrollTop;
    }
  }, []);

  useLayoutEffect(() => {
    const controlSection = controlSectionRef.current;
    if (!controlSection) return;
    if (Math.abs(controlSection.scrollTop - controlScrollTopRef.current) > 1) {
      controlSection.scrollTop = controlScrollTopRef.current;
    }
  });

  useEffect(() => {
    if (!selectedHandle) return;
    if (selectedHandle.type === 'base' && selectedHandle.id >= basePoints.length) setSelectedHandle(null);
    if (selectedHandle.type === 'mask' && selectedHandle.id >= maskPoints.length) setSelectedHandle(null);
    if (selectedHandle.type === 'mesh' && selectedHandle.id >= meshPoints.length) setSelectedHandle(null);
  }, [selectedHandle, basePoints.length, maskPoints.length, meshPoints.length]);

  useEffect(() => {
    if (!selectedHandle) return;
    if (activeMode === 'mesh' && selectedHandle.type !== 'mesh') setSelectedHandle(null);
    if (activeMode === 'mask' && selectedHandle.type !== 'mask') setSelectedHandle(null);
    if (activeMode === 'calibrate' && (selectedHandle.type === 'mesh' || selectedHandle.type === 'mask')) setSelectedHandle(null);
  }, [activeMode, selectedHandle]);

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
    if (!isCylinderProduct) return;
    if (Math.abs(centerFocusWidth) <= 1e-8 && edgeSqueeze <= 0) return;

    const debug = buildHorizontalSqueezeDebug(edgeSqueeze, squeezePower, centerFocusWidth);
    console.info('[PrintAreaEditor] horizontal squeeze debug', {
      edge_squeeze: edgeSqueeze,
      squeeze_power: squeezePower,
      center_focus_width: centerFocusWidth,
      ...debug,
    });
    if (!debug.active) {
      console.warn('[PrintAreaEditor] horizontal squeeze inactive', {
        inactive_reason: debug.inactiveReason,
        edge_squeeze: edgeSqueeze,
        center_focus_width: centerFocusWidth,
      });
    }
  }, [productType, edgeSqueeze, squeezePower, centerFocusWidth]);

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
      if (gridOverlayUrl) URL.revokeObjectURL(gridOverlayUrl);
    };
  }, [gridOverlayUrl]);

  useEffect(() => {
    return () => {
      if (previewOverlayUrl) URL.revokeObjectURL(previewOverlayUrl);
    };
  }, [previewOverlayUrl]);

  // WebGL edge preview: initialize only when edge tab active
  useEffect(() => {
    if (activeGroup !== 'edge') {
      const st = glStateRef.current;
      if (st && st.gl) {
        try { st.gl.deleteTexture(st.tex); st.gl.deleteProgram(st.prog); } catch (e) {}
      }
      glStateRef.current = null;
      return;
    }

    const canvas = edgeCanvasRef.current;
    if (!canvas) return;
    const gl = (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null;
    if (!gl) return;
    const glCtx = gl as WebGLRenderingContext;
    const canvasEl = canvas as HTMLCanvasElement;

    const vert = `attribute vec2 a_pos; varying vec2 v_uv; void main(){ v_uv = a_pos*0.5+0.5; gl_Position = vec4(a_pos,0.0,1.0); }`;
    const frag = `precision mediump float; varying vec2 v_uv; uniform sampler2D u_texture; uniform float u_edgeWidth; uniform float u_power; uniform float u_satLift; uniform float u_blackLift; uniform float u_shadowStr; uniform float u_shadowFall; uniform float u_isCylinder; uniform float u_thetaMaxDeg; uniform float u_curveTop; uniform float u_curveBot; uniform float u_edgeSqueeze; uniform float u_squeezePower; uniform float u_centerFocusWidth; uniform float u_hpx; uniform float u_wpx; 
    // helper RGB<->HSV
    vec3 rgb2hsv(vec3 c){vec4 K=vec4(0.,-1./3.,2./3.,-1.);vec4 p=mix(vec4(c.bg,K.wz),vec4(c.gb,K.xy),step(c.b,c.g));vec4 q=mix(vec4(p.xyw,c.r),vec4(c.r,p.yzx),step(p.x,c.r));float d=q.x-min(q.w,q.y);float e=1e-10;return vec3(abs(q.z+(q.w-q.y)/(6.*d+e)),d/(q.x+e),q.x);} 
    vec3 hsv2rgb(vec3 c){vec3 p=abs(fract(c.x+vec3(0.,2./3.,1./3.))*6.-3.)-1.;return c.z*mix(vec3(1.),clamp(p,0.,1.),c.y);} 
    float edgeDist(vec2 uv){float dx=min(uv.x,1.0-uv.x);float dy=min(uv.y,1.0-uv.y);return min(dx,dy);} 
    // center focus/edge roll helpers (approximate JS behavior)
    float computeCenterBand(float widthStrength){float BASE=0.30; float MINB=0.10; float MAXB=0.70; float w=clamp(widthStrength,-1.0,1.0); if(w>=0.0) return BASE + (MAXB-BASE)*w; return BASE + (BASE-MINB)*w;}
    float applyHorizontalSqueeze(float t, float edgeSqueeze, float squeezePower, float centerFocusWidth){float blend = clamp(edgeSqueeze,0.0,1.0); float width = clamp(centerFocusWidth,-1.0,1.0); if(blend<=1e-6 && abs(width)<=1e-6) return t; float radius = abs(t); // map center band
      float sourceBand = 0.30; float targetBand = computeCenterBand(width);
      float widthAdjustedRadius = radius;
      if(radius <= sourceBand){ float innerRatio = clamp(radius / max(sourceBand,1e-8),0.0,1.0); widthAdjustedRadius = targetBand * innerRatio; }
      else { float outerRatio = clamp((radius-sourceBand)/max(1.0-sourceBand,1e-8),0.0,1.0); widthAdjustedRadius = targetBand + (1.0 - targetBand) * outerRatio; }
      float edgeStart = min(max(0.55, targetBand),0.95); if(edgeStart >= 1.0-1e-6) return t; float power = max(1.0, squeezePower); float progress = clamp((widthAdjustedRadius - edgeStart)/max(1.0-edgeStart,1e-8),0.0,1.0); float rolledProgress = pow(progress, 1.0/power); float mappedProgress = progress + (rolledProgress - progress) * blend; if(widthAdjustedRadius <= edgeStart) return t; float mapped = edgeStart + (1.0 - edgeStart) * mappedProgress; return sign(t) * mapped; }
    // cylinder warp using canonical coords
    vec2 cylinderWarp(vec2 uv){ float thetaMax = u_thetaMaxDeg * 0.017453292; if(thetaMax <= 1e-6) return uv; float nx = (uv.x - 0.5) * 2.0; float sinThetaMax = sin(thetaMax); float sinTheta = clamp(nx * sinThetaMax, -1.0, 1.0); float theta = asin(sinTheta); float t = theta / thetaMax; // apply horizontal squeeze
      float tFinal = applyHorizontalSqueeze(t, u_edgeSqueeze, u_squeezePower, u_centerFocusWidth); float wx = (tFinal + 1.0) * 0.5; float hr_ratio = u_hpx / max(u_wpx, 1e-8); float yProj = (uv.y - 0.5) * 2.0; float curveAtPixel = (u_curveTop/100.0)*(1.0-uv.y) + (u_curveBot/100.0)*uv.y; float signY = -yProj; float wy_canon = yProj + signY * curveAtPixel * hr_ratio * 0.15 * (cos(theta) - cos(thetaMax)); float wy = (wy_canon + 1.0) * 0.5; return vec2(clamp(wx,0.0,1.0), clamp(wy,-0.5,1.5)); }
    void main(){ vec2 warpedUV = v_uv; if(u_isCylinder > 0.5) warpedUV = cylinderWarp(v_uv); vec4 col = texture2D(u_texture, warpedUV); float d = edgeDist(warpedUV); float edgeNorm = clamp(d / max(u_edgeWidth, 1e-5), 0.0, 1.0); float fall = pow(edgeNorm, u_power); float alphaFade = 1.0 - fall; vec3 hsv = rgb2hsv(col.rgb); hsv.y = mix(hsv.y, hsv.y*(1.0 - u_satLift), fall); vec3 rgb = hsv2rgb(hsv); rgb = mix(rgb, rgb + vec3(u_blackLift), fall); float shadow = u_shadowStr * exp(-pow(d * u_shadowFall, 2.0)); rgb *= (1.0 - shadow); gl_FragColor = vec4(rgb, col.a * alphaFade); }`;

    function compile(src: string, type: number) { const s = glCtx.createShader(type)!; glCtx.shaderSource(s, src); glCtx.compileShader(s); if (!glCtx.getShaderParameter(s, glCtx.COMPILE_STATUS)) console.error(glCtx.getShaderInfoLog(s)); return s; }
    const vs = compile(vert, glCtx.VERTEX_SHADER); const fs = compile(frag, glCtx.FRAGMENT_SHADER);
    const prog = glCtx.createProgram()!; glCtx.attachShader(prog, vs); glCtx.attachShader(prog, fs); glCtx.linkProgram(prog); if (!glCtx.getProgramParameter(prog, glCtx.LINK_STATUS)) console.error(glCtx.getProgramInfoLog(prog));

    const a_pos = glCtx.getAttribLocation(prog, 'a_pos');
    const loc_tex = glCtx.getUniformLocation(prog, 'u_texture');
    const loc_edgeWidth = glCtx.getUniformLocation(prog, 'u_edgeWidth');
    const loc_power = glCtx.getUniformLocation(prog, 'u_power');
    const loc_satLift = glCtx.getUniformLocation(prog, 'u_satLift');
    const loc_blackLift = glCtx.getUniformLocation(prog, 'u_blackLift');
    const loc_shadowStr = glCtx.getUniformLocation(prog, 'u_shadowStr');
    const loc_shadowFall = glCtx.getUniformLocation(prog, 'u_shadowFall');
    const loc_isCylinder = glCtx.getUniformLocation(prog, 'u_isCylinder');
    const loc_thetaMax = glCtx.getUniformLocation(prog, 'u_thetaMaxDeg');
    const loc_curveTop = glCtx.getUniformLocation(prog, 'u_curveTop');
    const loc_curveBot = glCtx.getUniformLocation(prog, 'u_curveBot');
    const loc_edgeSqueeze = glCtx.getUniformLocation(prog, 'u_edgeSqueeze');
    const loc_squeezePower = glCtx.getUniformLocation(prog, 'u_squeezePower');
    const loc_centerFocusWidth = glCtx.getUniformLocation(prog, 'u_centerFocusWidth');
    const loc_hpx = glCtx.getUniformLocation(prog, 'u_hpx');
    const loc_wpx = glCtx.getUniformLocation(prog, 'u_wpx');
    const loc_normalTex = glCtx.getUniformLocation(prog, 'u_normalTex');
    const loc_useNormal = glCtx.getUniformLocation(prog, 'u_useNormal');
    const loc_lightPos = glCtx.getUniformLocation(prog, 'u_lightPos');

    const vbo = glCtx.createBuffer(); glCtx.bindBuffer(glCtx.ARRAY_BUFFER, vbo); glCtx.bufferData(glCtx.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), glCtx.STATIC_DRAW);
    const tex = glCtx.createTexture(); glCtx.bindTexture(glCtx.TEXTURE_2D, tex); glCtx.texParameteri(glCtx.TEXTURE_2D, glCtx.TEXTURE_WRAP_S, glCtx.CLAMP_TO_EDGE); glCtx.texParameteri(glCtx.TEXTURE_2D, glCtx.TEXTURE_WRAP_T, glCtx.CLAMP_TO_EDGE); glCtx.texParameteri(glCtx.TEXTURE_2D, glCtx.TEXTURE_MIN_FILTER, glCtx.LINEAR); glCtx.texParameteri(glCtx.TEXTURE_2D, glCtx.TEXTURE_MAG_FILTER, glCtx.LINEAR);
    const normalTex = glCtx.createTexture(); glCtx.bindTexture(glCtx.TEXTURE_2D, normalTex); glCtx.texParameteri(glCtx.TEXTURE_2D, glCtx.TEXTURE_WRAP_S, glCtx.CLAMP_TO_EDGE); glCtx.texParameteri(glCtx.TEXTURE_2D, glCtx.TEXTURE_WRAP_T, glCtx.CLAMP_TO_EDGE); glCtx.texParameteri(glCtx.TEXTURE_2D, glCtx.TEXTURE_MIN_FILTER, glCtx.LINEAR); glCtx.texParameteri(glCtx.TEXTURE_2D, glCtx.TEXTURE_MAG_FILTER, glCtx.LINEAR);

    function updateTexture(url: string | null) { if (!url) return; const img = new Image(); img.crossOrigin = 'anonymous'; img.onload = () => { canvasEl.width = img.naturalWidth; canvasEl.height = img.naturalHeight; glCtx.viewport(0,0,canvasEl.width,canvasEl.height); glCtx.bindTexture(glCtx.TEXTURE_2D, tex); glCtx.pixelStorei(glCtx.UNPACK_FLIP_Y_WEBGL, true); glCtx.texImage2D(glCtx.TEXTURE_2D, 0, glCtx.RGBA, glCtx.RGBA, glCtx.UNSIGNED_BYTE, img); render(); }; img.src = url; }
    function updateNormalTexture(url: string | null) { if (!url) return; const img = new Image(); img.crossOrigin = 'anonymous'; img.onload = () => { glCtx.bindTexture(glCtx.TEXTURE_2D, normalTex); glCtx.pixelStorei(glCtx.UNPACK_FLIP_Y_WEBGL, true); glCtx.texImage2D(glCtx.TEXTURE_2D, 0, glCtx.RGBA, glCtx.RGBA, glCtx.UNSIGNED_BYTE, img); render(); }; img.src = url; }

    function render() { glCtx.useProgram(prog); glCtx.bindBuffer(glCtx.ARRAY_BUFFER, vbo); glCtx.enableVertexAttribArray(a_pos); glCtx.vertexAttribPointer(a_pos, 2, glCtx.FLOAT, false, 0, 0); glCtx.activeTexture(glCtx.TEXTURE0); glCtx.bindTexture(glCtx.TEXTURE_2D, tex); glCtx.uniform1i(loc_tex, 0); // normal map at TEX1
      glCtx.activeTexture(glCtx.TEXTURE1); glCtx.bindTexture(glCtx.TEXTURE_2D, normalTex); glCtx.uniform1i(loc_normalTex, 1);
      glCtx.uniform1f(loc_edgeWidth, edgeWidth); glCtx.uniform1f(loc_power, edgePower); glCtx.uniform1f(loc_satLift, edgeSatLift); glCtx.uniform1f(loc_blackLift, edgeBlackLift); glCtx.uniform1f(loc_shadowStr, edgeShadowStr); glCtx.uniform1f(loc_shadowFall, edgeShadowFall); glCtx.uniform1f(loc_useNormal, normalMapUrl ? 1.0 : 0.0);
      // pass light pos as (x,y,heightNormalized)
      const lightZ = lightHeight / Math.max(W_px, H_px, 1);
      glCtx.uniform3f(loc_lightPos, lightPosX, lightPosY, lightZ);
      glCtx.uniform1f(loc_isCylinder, isCylinderProduct ? 1.0 : 0.0); glCtx.uniform1f(loc_thetaMax, curvePct * 0.9); glCtx.uniform1f(loc_curveTop, curveTop); glCtx.uniform1f(loc_curveBot, curveBot); glCtx.uniform1f(loc_edgeSqueeze, edgeSqueeze); glCtx.uniform1f(loc_squeezePower, squeezePower); glCtx.uniform1f(loc_centerFocusWidth, centerFocusWidth); glCtx.uniform1f(loc_hpx, H_px); glCtx.uniform1f(loc_wpx, W_px); glCtx.drawArrays(glCtx.TRIANGLE_STRIP, 0, 4); }

    glStateRef.current = { gl: glCtx, prog, tex, normalTex, updateTexture, updateNormalTexture, render };
    updateTexture(previewOverlayUrl || imageUrl);
    if (normalMapUrl) updateNormalTexture(normalMapUrl);

    return () => { try { glCtx.deleteTexture(tex); glCtx.deleteTexture(normalTex); glCtx.deleteProgram(prog); } catch (e) {} glStateRef.current = null; };
  }, [activeGroup]);

  useEffect(() => {
    const st = glStateRef.current; if (!st) return; st.updateTexture(previewOverlayUrl || imageUrl); st.render();
  }, [previewOverlayUrl, imageUrl, edgeWidth, edgePower, edgeSatLift, edgeBlackLift, edgeShadowStr, edgeShadowFall]);

  useEffect(() => {
    if (!showGrid || !isCylinderProduct) {
      setGridOverlayUrl(prev => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
    }
  }, [showGrid, productType]);

  useEffect(() => {
    if (designFile || !designUrl) {
      setResolvedDesignBlob(null);
      return;
    }

    const controller = new AbortController();
    setResolvedDesignBlob(null);
    setApiError(null);
    (async () => {
      try {
        const response = await fetch(designUrl, {
          signal: controller.signal,
          cache: 'force-cache',
        });
        if (!response.ok) {
          throw new Error(`Artwork source error ${response.status}`);
        }
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        setResolvedDesignBlob(blob);
        setApiError(null);
      } catch (e: any) {
        if (controller.signal.aborted || e?.name === 'AbortError') return;
        console.error('Design source load error:', e);
        setResolvedDesignBlob(null);
        setApiError(e.message || 'Không thể tải nguồn ảnh in');
      }
    })();

    return () => controller.abort();
  }, [designFile, designUrl]);

  useEffect(() => {
    if (designFile || designUrl) {
      return;
    }
    setApiError(null);
    setPreviewOverlayUrl(prev => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
  }, [designFile, designUrl]);

  useEffect(() => {
    setApiError(null);
    setPreviewOverlayUrl(prev => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
  }, [designFile, designUrl, imageUrl]);

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
    setEdgeSqueeze(toFiniteNumber(warpConfig?.edge_squeeze, 0));
    setSqueezePower(toFiniteNumber(warpConfig?.squeeze_power, 2));
    setCenterFocusWidth(toFiniteNumber(warpConfig?.center_focus_width, 0));
    setMeshDensityStrength(toFiniteNumber(warpConfig?.mesh_density_strength, DEFAULT_MESH_DENSITY_STRENGTH));
    setFeatherRadius(toFiniteNumber(warpConfig?.feather_radius, 3));
    setDesignScale(toFiniteNumber(warpConfig?.design_scale, 1));
    setDesignOffsetX(toFiniteNumber(warpConfig?.design_offset_x, 0));
    setDesignOffsetY(toFiniteNumber(warpConfig?.design_offset_y, 0));

    // Restore lighting values if present
    setLightPosX(toFiniteNumber(warpConfig?.light_pos_x, 0.62));
    setLightPosY(toFiniteNumber(warpConfig?.light_pos_y, 0.32));
    setLightHeight(toFiniteNumber(warpConfig?.light_height, 55));
    setLightContrast(toFiniteNumber(warpConfig?.light_contrast, 50));
    setLightHighlight(toFiniteNumber(warpConfig?.light_highlight, 60));
    setLightSoftness(toFiniteNumber(warpConfig?.light_softness, 55));
    // restore edge params
    setEdgeWidth(toFiniteNumber(warpConfig?.edge?.edge_width ?? warpConfig?.edge_width ?? 0.18, 0.18));
    setEdgePower(toFiniteNumber(warpConfig?.edge?.power ?? warpConfig?.edge_power ?? 1.5, 1.5));
    setEdgeSatLift(toFiniteNumber(warpConfig?.edge?.sat_lift ?? warpConfig?.sat_lift ?? 0.55, 0.55));
    setEdgeBlackLift(toFiniteNumber(warpConfig?.edge?.black_lift ?? warpConfig?.black_lift ?? 0.04, 0.04));
    setEdgeShadowStr(toFiniteNumber(warpConfig?.edge?.shadow_str ?? warpConfig?.shadow_str ?? 0.55, 0.55));
    setEdgeShadowFall(toFiniteNumber(warpConfig?.edge?.shadow_fall ?? warpConfig?.shadow_fall ?? 3.0, 3.0));

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
            edge: {
              edge_width: edgeWidth,
              power: edgePower,
              sat_lift: edgeSatLift,
              black_lift: edgeBlackLift,
              shadow_str: edgeShadowStr,
              shadow_fall: edgeShadowFall,
            },
      warp_type: resolvedWarpType,
      theta_max_deg: curvePct * 0.9,
      curve_pct: curvePct,
      curve: (smile_api / 100) * hr_ratio * 0.15,
      curve_top: curveTop,
      curve_bottom: curveBot,
      edge_squeeze: edgeSqueeze,
      squeeze_power: squeezePower,
      center_focus_width: centerFocusWidth,
      mesh_density_strength: meshDensityStrength,
      tilt_deg: tilt, rotate_deg: rotate, persp_strength: perspective,
      camera_elevation: pitch_api,
      product_type: productType,
      feather_radius: featherRadius,
      design_scale: designScale,
      design_offset_x: designOffsetX,
      design_offset_y: designOffsetY,
      editor_mode: editorMode,
      is_locked: editorMode === 'DESIGN',
      // lighting
      light_pos_x: lightPosX,
      light_pos_y: lightPosY,
      light_height: lightHeight,
      light_contrast: lightContrast,
      light_highlight: lightHighlight,
      light_softness: lightSoftness,
    };

    onLiveSnapshotChange({ printArea: pa, warpConfig: nextWarpConfig });
  }, [basePoints, tilt, rotate, perspective, curvePct, curveTop, curveBot, edgeSqueeze, squeezePower, centerFocusWidth, meshDensityStrength, designScale, designOffsetX, designOffsetY, featherRadius, productType, maskPoints, meshPoints, naturalSize, editorMode, lightPosX, lightPosY, lightHeight, lightContrast, lightHighlight, lightSoftness]);

  useEffect(() => {
    if (naturalSize.w === 0) return;
    if (!showGrid || !isCylinderProduct) return;

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
    const templateIdMatch = imageUrl.match(/\/templates\/([^/]+)\//);
    const templateId = templateIdMatch ? templateIdMatch[1] : null;
    const payload = {
      mockup_width: naturalSize.w, mockup_height: naturalSize.h,
      print_area: pa,
      warp_type: resolvedWarpType,
      theta_max_deg: curvePct * 0.9,
      curve: (smile_api / 100) * hr_ratio * 0.15,
      curve_top: curveTop,
      curve_bottom: curveBot,
      edge_squeeze: edgeSqueeze,
      squeeze_power: squeezePower,
      center_focus_width: centerFocusWidth,
      mesh_density_strength: meshDensityStrength,
      design_scale: designScale,
      design_offset_x: designOffsetX,
      design_offset_y: designOffsetY,
      mask_points: pa.mask_points,
      template_id: templateId,
    };

    const requestSeq = ++gridPreviewRequestSeqRef.current;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const response = await fetch('/v1/mockup/warp-preview', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || `Server error ${response.status}`);
        }
        if (controller.signal.aborted || requestSeq !== gridPreviewRequestSeqRef.current) return;
        const blob = await response.blob();
        if (controller.signal.aborted || requestSeq !== gridPreviewRequestSeqRef.current) return;
        setGridOverlayUrl(prev => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(blob);
        });
      } catch (e: any) {
        if (controller.signal.aborted || e?.name === 'AbortError') return;
        console.error('Warp Grid Preview Error:', e);
        if (requestSeq === gridPreviewRequestSeqRef.current) {
          setApiError(e.message || 'Không thể tải xem trước lưới');
        }
      }
    }, 200);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [basePoints, tilt, rotate, perspective, curvePct, curveTop, curveBot, edgeSqueeze, squeezePower, centerFocusWidth, meshDensityStrength, designScale, designOffsetX, designOffsetY, productType, maskPoints, meshPoints, naturalSize, imageUrl, showGrid]);

  useEffect(() => {
    if (naturalSize.w === 0) return;
    if (!designPreviewSource) {
      setApiError(null);
      setPreviewOverlayUrl(prev => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
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
    onCoordinatesChange(pa);
    const nextWarpConfig = {
      warp_type: resolvedWarpType,
      theta_max_deg: curvePct * 0.9,
      curve_pct: curvePct,
      curve: (smile_api / 100) * hr_ratio * 0.15,
      curve_top: curveTop,
      curve_bottom: curveBot,
      edge_squeeze: edgeSqueeze,
      squeeze_power: squeezePower,
      center_focus_width: centerFocusWidth,
      mesh_density_strength: meshDensityStrength,
      tilt_deg: tilt, rotate_deg: rotate, persp_strength: perspective,
      camera_elevation: pitch_api,
      product_type: productType,
      feather_radius: featherRadius,
      design_scale: designScale,
      design_offset_x: designOffsetX,
      design_offset_y: designOffsetY,
      // lighting fields from editor
      light_pos_x: lightPosX,
      light_pos_y: lightPosY,
      light_height: lightHeight,
      light_contrast: lightContrast,
      light_highlight: lightHighlight,
      light_softness: lightSoftness,
      editor_mode: editorMode,
      is_locked: editorMode === 'DESIGN',
    };
    onConfigChange(nextWarpConfig);
    if (onLockedSnapshotChange) {
      if (editorMode === 'DESIGN') onLockedSnapshotChange({ printArea: pa, warpConfig: nextWarpConfig });
      else onLockedSnapshotChange(null);
    }

    const requestSeq = ++designPreviewRequestSeqRef.current;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      const templateIdMatch = imageUrl.match(/\/templates\/([^/]+)\//);
      const templateId = templateIdMatch ? templateIdMatch[1] : null;

      const payload = {
        mockup_width: naturalSize.w, mockup_height: naturalSize.h,
        print_area: pa,
        warp_type: resolvedWarpType,
        theta_max_deg: curvePct * 0.9,
        curve: (smile_api / 100) * hr_ratio * 0.15,
        curve_top: curveTop,
        curve_bottom: curveBot,
        edge_squeeze: edgeSqueeze,
        squeeze_power: squeezePower,
        center_focus_width: centerFocusWidth,
        mesh_density_strength: meshDensityStrength,
        design_scale: designScale,
        design_offset_x: designOffsetX,
        design_offset_y: designOffsetY,
        light_pos_x: lightPosX,
        light_pos_y: lightPosY,
        light_height: lightHeight,
        light_contrast: lightContrast,
        light_highlight: lightHighlight,
        light_softness: lightSoftness,
        mask_points: pa.mask_points,
        template_id: templateId,
      };

      if (payload.warp_type === 'cylinder' && (Math.abs(centerFocusWidth) > 1e-8 || edgeSqueeze > 0)) {
        console.info('[Warp Preview] request payload', {
          theta_max_deg: payload.theta_max_deg,
          edge_squeeze: payload.edge_squeeze,
          squeeze_power: payload.squeeze_power,
          center_focus_width: payload.center_focus_width,
          debug: buildHorizontalSqueezeDebug(edgeSqueeze, squeezePower, centerFocusWidth),
        });
      }

      try {
        const formData = new FormData();
        formData.append('config_json', JSON.stringify(payload));
        formData.append(
          'design_image',
          designPreviewSource,
          designFile?.name || (designUrl ? getDesignFilenameFromUrl(designUrl) : 'artwork.png'),
        );

        const response = await fetch('/v1/mockup/warp-preview-file', {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || `Server error ${response.status}`);
        }
        if (controller.signal.aborted || requestSeq !== designPreviewRequestSeqRef.current) {
          return;
        }
        setApiError(null);
        const blob = await response.blob();
        if (controller.signal.aborted || requestSeq !== designPreviewRequestSeqRef.current) {
          return;
        }
        setPreviewOverlayUrl(prev => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(blob);
        });
      } catch (e: any) {
        if (controller.signal.aborted || e?.name === 'AbortError') {
          return;
        }
        console.error("Warp Preview Error:", e);
        if (requestSeq === designPreviewRequestSeqRef.current) {
          setApiError(e.message || 'Không thể tải xem trước');
        }
      }
    }, 400);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [basePoints, tilt, rotate, perspective, curvePct, curveTop, curveBot, edgeSqueeze, squeezePower, centerFocusWidth, meshDensityStrength, designScale, designOffsetX, designOffsetY, featherRadius, productType, maskPoints, meshPoints, naturalSize, editorMode, imageUrl, designFile, designUrl, designPreviewSource, lightPosX, lightPosY, lightHeight, lightContrast, lightHighlight, lightSoftness]);

  const lastPointerPos = useRef({ x: 0, y: 0 });
  const wasDraggingRef = useRef(false);

  const setSelectedPointPosition = useCallback((targetX: number, targetY: number) => {
    if (!selectedHandle) return;
    const x = clamp01(targetX);
    const y = clamp01(targetY);

    if (selectedHandle.type === 'base') {
      setBasePoints((prev) => {
        if (selectedHandle.id < 0 || selectedHandle.id >= prev.length) return prev;
        const next = [...prev];
        next[selectedHandle.id] = { x, y };
        return next;
      });
      return;
    }

    if (selectedHandle.type === 'mask') {
      setMaskPoints((prev) => {
        if (selectedHandle.id < 0 || selectedHandle.id >= prev.length) return prev;
        const next = [...prev];
        next[selectedHandle.id] = { x, y };
        return next;
      });
      return;
    }

    if (selectedHandle.type === 'mesh') {
      setMeshPoints((prev) => {
        if (selectedHandle.id < 0 || selectedHandle.id >= prev.length) return prev;
        const next = [...prev];
        const current = next[selectedHandle.id];
        const desired = limitMeshDisplacement(current, { x, y });
        const constrained = constrainStructuredMeshDestination(next, selectedHandle.id, desired);
        next[selectedHandle.id] = { ...current, dst: constrained };
        return next;
      });
      return;
    }

    if (selectedHandle.type === 'light') {
      const local = apply_projective_homography(H_inv_mat, x, y);
      setLightPosX(clamp01((local.x + 1) * 0.5));
      setLightPosY(clamp01((local.y + 1) * 0.5));
    }
  }, [selectedHandle, H_inv_mat]);

  const nudgeSelectedPoint = useCallback((dx: number, dy: number) => {
    if (!selectedPointPosition) return;
    setSelectedPointPosition(selectedPointPosition.x + dx, selectedPointPosition.y + dy);
  }, [selectedPointPosition, setSelectedPointPosition]);

  const handleEditorKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement | null;
    if (target) {
      const tag = target.tagName.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable) return;
    }
    const isArrowKey = e.key === 'ArrowUp' || e.key === 'ArrowDown' || e.key === 'ArrowLeft' || e.key === 'ArrowRight';
    if (!isArrowKey) return;
    e.preventDefault();
    if (!canNudgeSelectedHandle) return;
    const step = e.shiftKey ? pointNudgeStep * 10 : pointNudgeStep;
    if (e.key === 'ArrowUp') {
      nudgeSelectedPoint(0, -step);
    } else if (e.key === 'ArrowDown') {
      nudgeSelectedPoint(0, step);
    } else if (e.key === 'ArrowLeft') {
      nudgeSelectedPoint(-step, 0);
    } else if (e.key === 'ArrowRight') {
      nudgeSelectedPoint(step, 0);
    }
  }, [canNudgeSelectedHandle, pointNudgeStep, nudgeSelectedPoint]);

  const handlePointerDown = (type: string, id: number, e: React.PointerEvent) => {
    e.preventDefault(); e.stopPropagation();
    wasDraggingRef.current = false;
    setDraggingIdx({ type, id });
    if (type !== 'design') {
      setSelectedHandle({ type, id });
    }
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
    // mark that a drag occurred so click handler won't add a new point afterwards
    wasDraggingRef.current = true;
    const rawX = (e.clientX - r.left - pan.x) / (r.width * zoom);
    const rawY = (e.clientY - r.top - pan.y) / (r.height * zoom);
    const x = clamp01(rawX);
    const y = clamp01(rawY);
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
      const next = [...meshPoints];
      const srcPt = next[draggingIdx.id]?.src || { x, y };
      const desired = limitMeshDisplacement({ src: srcPt, dst: next[draggingIdx.id].dst }, { x, y });
      next[draggingIdx.id].dst = constrainStructuredMeshDestination(next, draggingIdx.id, desired);
      setMeshPoints(next);
    } else if (draggingIdx.type === 'light') {
      const local = apply_projective_homography(H_inv_mat, rawX, rawY);
      setLightPosX(clamp01((local.x + 1) * 0.5));
      setLightPosY(clamp01((local.y + 1) * 0.5));
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
  }, [draggingIdx, zoom, pan, basePoints, maskPoints, meshPoints, calibPts, H_px, naturalSize, curvePct, H_inv_mat]);

  const handlePointerUp = useCallback(() => {
    if (draggingIdx) {
      wasDraggingRef.current = true;
    }
    setDraggingIdx(null);
  }, [draggingIdx]);

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
    // ignore click events that immediately follow a drag
    if (wasDraggingRef.current) { wasDraggingRef.current = false; return; }
    const r = containerRef.current.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const y = (e.clientY - r.top) / r.height;
    if (activeMode === 'mask') {
      const next = [...maskPoints, { x, y }];
      setMaskPoints(next);
      setSelectedHandle({ type: 'mask', id: next.length - 1 });
    }
    // Cylinder mesh edits existing control points only.
    // Non-cylinder keeps click-to-add mesh point behavior.
    if (activeMode === 'mesh' && !isCylinderProduct) {
      const next = [...meshPoints, { src: { x, y }, dst: { x, y } }];
      setMeshPoints(next);
      setSelectedHandle({ type: 'mesh', id: next.length - 1 });
    }
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

  const buildMeshGrid = (rows: number, cols: number): MeshPoint[] => {
    const srcCoords = [{ x: -1, y: -1 }, { x: 1, y: -1 }, { x: 1, y: 1 }, { x: -1, y: 1 }];
    const hMatLocal = solve_homography(srcCoords, calibPts);
    const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
    const newSrcs: Point[] = [];

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const u = c / (cols - 1);
        const v = r / (rows - 1);
        const warpedLocal = isCylinderProduct
          ? cylinderWarpPoint(u, v, curvePct, curveTop, curveBot, edgeSqueeze, squeezePower, centerFocusWidth, H_px, W_px)
          : { x: u, y: v };
        const world = apply_homography(hMatLocal, warpedLocal.x, warpedLocal.y);
        newSrcs.push({ x: clamp01(world.x), y: clamp01(world.y) });
      }
    }

    // If there are existing meshPoints, try to reuse nearest dst values to preserve edits.
    const existing = meshPoints || [];
    const MAX_MATCH_DIST = 0.25; // normalized distance threshold to consider a match
    return newSrcs.map((src) => {
      let best: { dst: Point; d: number } | null = null;
      for (const ex of existing) {
        const dx = ex.src.x - src.x;
        const dy = ex.src.y - src.y;
        const d = Math.hypot(dx, dy);
        if (!best || d < best.d) best = { dst: ex.dst, d };
      }
      if (best && best.d <= MAX_MATCH_DIST) {
        return { src, dst: { ...best.dst } };
      }
      return { src, dst: { ...src } };
    });
  };

  const make4x4Mesh = () => {
    setMeshPoints(buildMeshGrid(4, 4));
  };

  // Per-tab reset helpers
  const resetGeometry = () => {
    setTilt(0); setRotate(0); setPerspective(0);
  };

  const resetWrap = () => {
    setCurvePct(60); setCurveTop(0); setCurveBot(0); setEdgeSqueeze(0); setSqueezePower(2); setCenterFocusWidth(0); setMeshDensityStrength(DEFAULT_MESH_DENSITY_STRENGTH);
  };

  const resetBlend = () => {
    setFeatherRadius(3); setLightPosX(0.62); setLightPosY(0.32); setLightHeight(55); setLightContrast(50); setLightHighlight(60); setLightSoftness(55);
  };

  const resetEdge = () => {
    setEdgeWidth(0.18); setEdgePower(1.5); setEdgeSatLift(0.55); setEdgeBlackLift(0.04); setEdgeShadowStr(0.55); setEdgeShadowFall(3.0); if (normalMapUrl) { URL.revokeObjectURL(normalMapUrl); setNormalMapUrl(null); }
  };

  useEffect(() => {
    if (editorMode !== 'CALIBRATE' || activeMode !== 'mesh') return;
    if (meshPoints.length > 0) return;
    setMeshPoints(buildMeshGrid(4, 4));
  }, [
    activeMode,
    calibPts,
    centerFocusWidth,
    curveBot,
    curvePct,
    curveTop,
    edgeSqueeze,
    editorMode,
    H_px,
    isCylinderProduct,
    meshPoints.length,
    squeezePower,
    W_px,
  ]);

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
  const renderControls = () => {
    const productSelector = !productTypeProp && (
      <div className="product-selector-top glass-panel" style={{ padding: '5px', marginBottom: '8px', border: '1px solid #1e2d44', background: '#111827', borderRadius: '8px' }}>
        <div className="ctrl-header" style={{ marginBottom: '6px' }}><span className="ctrl-label" style={{ color: '#60a5fa', fontSize: '9px' }}>Bước 1: Chọn loại sản phẩm</span></div>
        <select className="select" style={{ width: '100%', fontSize: '0.9rem', padding: '5px' }} value={productType} onChange={e => setProductType(e.target.value)}>
          <optgroup label="CỐC"><option value="cylinder_ceramic">Cốc sứ</option><option value="cylinder_glass">Cốc thủy tinh</option><option value="cylinder_travel">Ly giữ nhiệt</option></optgroup>
          <optgroup label="QUẦN ÁO"><option value="apparel_cotton">Áo thun</option><option value="apparel_hoodie">Áo hoodie</option><option value="apparel_totebag">Túi tote</option></optgroup>
          <optgroup label="KHÁC"><option value="flat_print">In phẳng</option><option value="plastic_case">Ốp điện thoại</option></optgroup>
        </select>
      </div>
    );

    const availableGroups: ('geometry' | 'wrap' | 'blend' | 'edge' | 'advanced')[] = ['geometry'];
    if (productType.startsWith('cylinder')) availableGroups.push('wrap');
    availableGroups.push('blend', 'edge', 'advanced');

    const selectedTypeLabelMap: Record<string, string> = {
      base: 'Góc vùng in',
      mesh: 'Điểm lưới',
      mask: 'Điểm mặt nạ',
      light: 'Điểm sáng',
      smile: 'Điểm cong trên',
      pitch: 'Điểm cong dưới',
    };
    const selectedPointLabel = selectedHandle
      ? `${selectedTypeLabelMap[selectedHandle.type] || 'Điểm'} #${selectedHandle.id + 1}`
      : 'Chưa chọn điểm';
    const selectedPointXPercent = selectedPointPosition ? selectedPointPosition.x * 100 : 0;
    const selectedPointYPercent = selectedPointPosition ? selectedPointPosition.y * 100 : 0;
    const pointNudgePanel = (
      <div className="point-nudge-panel">
        <div className="ctrl-header">
          <span className="ctrl-label">Di chuyển điểm bằng nút</span>
          <span className="ctrl-value">{selectedPointLabel}</span>
        </div>
        <p className="mode-hint">Chọn một điểm trên vùng in, sau đó bấm các nút hướng hoặc phím mũi tên để căn chính xác. Giữ `Shift` để dịch nhanh hơn.</p>
        {selectedPointPosition && canNudgeSelectedHandle && (
          <div className="ctrl-grid ctrl-grid-two">
            <ControlAdjuster
              label="Tọa độ X"
              description="Nhập trực tiếp tọa độ ngang của điểm (0-100%)."
              value={selectedPointXPercent}
              min={0}
              max={100}
              step={0.01}
              unit="%"
              onChange={(v) => setSelectedPointPosition(v / 100, selectedPointPosition.y)}
            />
            <ControlAdjuster
              label="Tọa độ Y"
              description="Nhập trực tiếp tọa độ dọc của điểm (0-100%)."
              value={selectedPointYPercent}
              min={0}
              max={100}
              step={0.01}
              unit="%"
              onChange={(v) => setSelectedPointPosition(selectedPointPosition.x, v / 100)}
            />
          </div>
        )}
        <ControlAdjuster
          label="Bước nudge điểm"
          description="Bước nhỏ hơn giúp canh mép chính xác hơn."
          value={pointNudgeStep * 100}
          min={0.01}
          max={2}
          step={0.01}
          unit="%"
          onChange={(v) => setPointNudgeStep(v / 100)}
        />
        <div className="nudge-grid">
          <span />
          <button className="btn-ghost nudge-btn" onClick={() => nudgeSelectedPoint(0, -pointNudgeStep)} disabled={!canNudgeSelectedHandle}>Lên</button>
          <span />
          <button className="btn-ghost nudge-btn" onClick={() => nudgeSelectedPoint(-pointNudgeStep, 0)} disabled={!canNudgeSelectedHandle}>Trái</button>
          <button className="btn-ghost nudge-btn" onClick={() => nudgeSelectedPoint(0, pointNudgeStep)} disabled={!canNudgeSelectedHandle}>Xuống</button>
          <button className="btn-ghost nudge-btn" onClick={() => nudgeSelectedPoint(pointNudgeStep, 0)} disabled={!canNudgeSelectedHandle}>Phải</button>
        </div>
      </div>
    );

    if (editorMode === 'DESIGN') return (
      <div className="ctrl-section" ref={controlSectionRef} onScroll={captureControlScroll}>
        {productSelector}
        {designFile && <p className="mode-hint">Kéo trực tiếp trên ảnh in 2D để đổi vị trí. Lăn chuột để zoom ảnh in, `Alt + wheel` để zoom canvas.</p>}
        <div className="ctrl-header"><span className="ctrl-label">Chỉnh ảnh in</span></div>
        <div className="ctrl-grid ctrl-grid-two">
          <ControlAdjuster label="Phóng to ảnh in" description="Tăng để ảnh in phủ rộng hơn trong vùng in." value={designScale} min={0.1} max={5} step={0.1} unit="x" onChange={setDesignScale} />
          <ControlAdjuster label="Dịch ngang" description="Dời ảnh in sang trái hoặc phải bên trong vùng in." value={designOffsetX} min={-1} max={1} step={0.1} onChange={setDesignOffsetX} />
          <ControlAdjuster label="Dịch dọc" description="Dời ảnh in lên hoặc xuống bên trong vùng in." value={designOffsetY} min={-1} max={1} step={0.1} onChange={setDesignOffsetY} />
        </div>
      </div>
    );

    if (activeMode === 'mesh') return (
      <div className="ctrl-section" ref={controlSectionRef} onScroll={captureControlScroll}>
        {productSelector}
        <p className="mode-hint">Kéo trực tiếp các điểm trên lưới cong để tinh chỉnh biến dạng bề mặt.</p>
        <div className="btn-row">
          <button className="btn-primary" onClick={make4x4Mesh}>Lưới 4x4</button>
          <button className="btn-ghost" onClick={make4x4Mesh}>Đặt lại lưới</button>
        </div>
        {pointNudgePanel}
      </div>
    );
    if (activeMode === 'mask') return (
      <div className="ctrl-section" ref={controlSectionRef} onScroll={captureControlScroll}>
        {productSelector}
        <p className="mode-hint">Nhấp để vẽ vùng mặt nạ. Đóng vòng để hoàn thành.</p>
        <div className="btn-row">
          <button className="btn-ghost" onClick={() => setMaskPoints([])}>Xóa mặt nạ</button>
        </div>
        {pointNudgePanel}
      </div>
    );

    return (
      <div className="ctrl-section" ref={controlSectionRef} onScroll={captureControlScroll}>
        {productSelector}
        {designFile && activeMode === 'calibrate' && <p className="mode-hint">Ảnh in là ảnh ngang 2D. Kéo chỉ tác động lên ảnh in, không kéo mockup gốc.</p>}
        <div className="group-tabs">
          {availableGroups.map(g => (
            <button key={g} className={`group-tab ${activeGroup === g ? 'active' : ''}`} onClick={() => setActiveGroup(g)}>
              {{ geometry: 'Hình học', wrap: 'Ôm cong', blend: 'Ánh sáng', edge: 'Viền', advanced: 'Nâng cao' }[g]}
            </button>
          ))}
        </div>
        {activeGroup === 'geometry' && (
          <>
          <div className="ctrl-grid ctrl-grid-two">
            <ControlAdjuster label="Nghiêng" description="Giả góc chụp bằng cách đẩy mép trên theo chiều ngang." value={tilt} min={-45} max={45} unit="°" onChange={setTilt} />
            <ControlAdjuster label="Xoay" description="Xoay toàn bộ vùng in quanh tâm." value={rotate} min={-30} max={30} unit="°" onChange={setRotate} />
            <ControlAdjuster label="Phối cảnh" description="Làm phần trên hoặc dưới hẹp lại để giống ảnh chụp xiên." value={perspective} min={-50} max={50} onChange={setPerspective} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}><button className="btn btn-ghost" onClick={resetGeometry}>Reset Hình học</button></div>
          </>
        )}
        {activeGroup === 'wrap' && (
          <>
          <div className="ctrl-grid ctrl-grid-two">
            <ControlAdjuster label="Độ ôm ngang" description="Tăng để vùng in quấn sang hai bên thân cốc nhiều hơn." value={curvePct} min={0} max={100} unit="%" onChange={setCurvePct} />
            <ControlAdjuster label="Cong mép trên" description="Bẻ đường mép trên lên hoặc xuống để khớp miệng cốc." value={curveTop} min={-100} max={100} unit="%" onChange={setCurveTop} />
            <ControlAdjuster label="Cong mép dưới" description="Bẻ đường mép dưới lên hoặc xuống để khớp đáy cốc." value={-curveBot} min={-100} max={100} unit="%" onChange={(v) => setCurveBot(-v)} />
            <ControlAdjuster label="Cường độ ép mép" description="Cuộn dải gần hai mép vào trong. Vùng giữa gần như giữ nguyên, chỉ phần rìa bị ép mạnh hơn." value={edgeSqueeze} min={0} max={1} step={0.1} onChange={setEdgeSqueeze} />
            <ControlAdjuster label="Độ mạnh chuyển tiếp" description="Điều khiển độ gắt của vùng cuộn mép. Cao hơn thì hiệu ứng dồn sát về mép rõ hơn." value={squeezePower} min={1} max={5} step={0.1} onChange={setSqueezePower} />
            <ControlAdjuster label="Độ rộng vùng giữa" description="Số dương: nới băng giữa, làm các ô ở giữa rộng ra và hai mép hẹp lại. Số âm: đảo chiều, siết băng giữa và dồn độ rộng ra hai mép." value={centerFocusWidth} min={-1} max={1} step={0.1} onChange={setCenterFocusWidth} />
            <ControlAdjuster label="Mật độ mép lưới" description="Tăng để lưới xem trước dày hơn ở hai mép. Tham số này chỉ tăng độ mịn phần mép xem trước, không bóp ảnh in." value={meshDensityStrength} min={1} max={4} step={0.1} onChange={setMeshDensityStrength} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}><button className="btn btn-ghost" onClick={resetWrap}>Reset Ôm cong</button></div>
          </>
        )}
        {activeGroup === 'blend' && (
          <>
            <p className="mode-hint">Kéo chấm vàng trên vùng in để đổi hướng nguồn sáng. Các thanh bên dưới chỉ map về `light_dir`, `light_field` và `specular` nội bộ, không render thêm một lớp giả riêng.</p>
            <div className="ctrl-grid ctrl-grid-two">
              <ControlAdjuster label="Nguồn sáng ngang" description="Dịch điểm sáng từ trái sang phải trên thân cốc." value={lightPosX * 100} min={0} max={100} step={1} unit="%" onChange={(v) => setLightPosX(v / 100)} />
              <ControlAdjuster label="Nguồn sáng dọc" description="Dịch điểm sáng từ trên xuống dưới theo bề mặt vùng in." value={lightPosY * 100} min={0} max={100} step={1} unit="%" onChange={(v) => setLightPosY(v / 100)} />
              <ControlAdjuster label="Chiều cao sáng" description="UI 0-100, nội bộ map sang trục Z để tránh nguồn sáng bị chết." value={lightHeight} min={1} max={100} step={1} onChange={setLightHeight} />
              <ControlAdjuster label="Độ tương phản" description="Tăng chênh sáng tối của light field; thấp hơn thì ánh sáng phẳng hơn." value={lightContrast} min={0} max={100} step={1} onChange={setLightContrast} />
              <ControlAdjuster label="Độ bóng" description="Tăng độ bóng/specular của men sứ trên composite cuối." value={lightHighlight} min={0} max={100} step={1} onChange={setLightHighlight} />
              <ControlAdjuster label="Độ mềm sáng" description="Làm vùng sáng loe rộng và mềm hơn; thấp hơn thì peak gắt hơn." value={lightSoftness} min={0} max={100} step={1} onChange={setLightSoftness} />
              <ControlAdjuster label="Làm mềm viền" description="Làm mượt mép vùng in khi ghép lên mockup." value={featherRadius} min={0} max={12} unit="px" onChange={setFeatherRadius} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}><button className="btn btn-ghost" onClick={resetBlend}>Reset Ánh sáng</button></div>
          </>
        )}
        {activeGroup === 'edge' && (
          <div className="ctrl-grid">
            <ControlAdjuster label="Độ rộng viền" description="Tỉ lệ băng fade ở rìa (fraction of width)" value={edgeWidth} min={0} max={0.5} step={0.01} onChange={setEdgeWidth} />
            <ControlAdjuster label="Độ cong falloff" description="Power curve cho falloff" value={edgePower} min={0.5} max={3} step={0.1} onChange={setEdgePower} />
            <ControlAdjuster label="Giảm saturation" description="Giảm saturation tại viền" value={edgeSatLift} min={0} max={1} step={0.01} onChange={setEdgeSatLift} />
            <ControlAdjuster label="Lift bóng (black)" description="Tăng black level tại viền" value={edgeBlackLift} min={0} max={0.2} step={0.01} onChange={setEdgeBlackLift} />
            <ControlAdjuster label="Cường độ shadow" description="Inner shadow strength" value={edgeShadowStr} min={0} max={1} step={0.01} onChange={setEdgeShadowStr} />
            <ControlAdjuster label="Falloff shadow" description="Shadow falloff" value={edgeShadowFall} min={0.5} max={6} step={0.1} onChange={setEdgeShadowFall} />
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <label className="btn btn-outline">Upload Normal Map<input style={{ display: 'none' }} type="file" accept="image/*" onChange={e => { const f = e.target.files?.[0] || null; setNormalMapUrl(f ? URL.createObjectURL(f) : null); }} /></label>
              {normalMapUrl && <button className="btn btn-ghost" onClick={() => { URL.revokeObjectURL(normalMapUrl); setNormalMapUrl(null); }}>Remove</button>}
              <button className="btn btn-ghost" onClick={resetEdge} style={{ marginLeft: 8 }}>Reset Viền</button>
            </div>
          </div>
        )}
        {activeGroup === 'advanced' && (
          <div className="ctrl-grid">
            <button className="btn-ghost" style={{ marginTop: 8 }} onClick={() => { setTilt(0); setRotate(0); setPerspective(0); setCurveTop(0); setCurveBot(0); setEdgeSqueeze(0); setSqueezePower(2); setCenterFocusWidth(0); setMeshDensityStrength(DEFAULT_MESH_DENSITY_STRENGTH); setDesignScale(1); setDesignOffsetX(0); setDesignOffsetY(0); setLightPosX(0.62); setLightPosY(0.32); setLightHeight(55); setLightContrast(50); setLightHighlight(60); setLightSoftness(55); }}>Đặt lại tất cả</button>
          </div>
        )}
        {pointNudgePanel}
      </div>
    );
  };

  return (
    <div
      className="pae-root"
      ref={editorRootRef}
      tabIndex={0}
      onKeyDown={handleEditorKeyDown}
      onPointerDownCapture={(e) => {
        const target = e.target as HTMLElement;
        const tag = target.tagName.toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
        editorRootRef.current?.focus();
      }}
    >
      <div className="toolbar">
        <div className="toolbar-left">
          <div className="logo-badge">Trình Chỉnh Vùng In</div>
          <div className="mode-sub-pills" style={{ marginLeft: 20, display: 'flex', gap: 6, visibility: editorMode === 'CALIBRATE' ? 'visible' : 'hidden' }}>
            {(['calibrate', 'mesh', 'mask'] as const).map(m => (
              <button key={m} className={`pill pill-sm ${activeMode === m ? 'active' : ''}`} onClick={() => setActiveMode(m)}>
                {{ calibrate: 'Căn chỉnh', mesh: 'Lưới', mask: 'Mặt nạ' }[m]}
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
            {isSwitchingDevice ? 'Đang chuyển...' : (renderDevice === 'gpu' ? 'GPU ON' : 'CPU')}
          </button>
          <button className={`icon-btn ${showGrid ? 'active' : ''}`} onClick={() => setShowGrid(v => !v)}>Lưới</button>
          <button className="btn-detect" onClick={handleAutoDetect} disabled={isDetecting}>{isDetecting ? 'Đang dò...' : 'Dò tự động'}</button>
          <div className="workflow-status">
            {editorMode === 'CALIBRATE' ? <button className="btn-confirm-lock" onClick={() => setEditorMode('DESIGN')}>Chốt vùng in</button> : <button className="btn-unlock" onClick={() => setEditorMode('CALIBRATE')}>Mở khóa</button>}
          </div>
        </div>
      </div>

      <div className="pae-body">
        <aside className="sidebar">
          {renderControls()}
        </aside>

        <main className="canvas-section">
          <div className="canvas-wrap" ref={containerRef} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onWheel={handleWheel} onMouseDown={handleContainerMouseDown} onMouseMove={handleContainerMouseMove} onMouseUp={handleContainerMouseUp} onMouseLeave={handleContainerMouseUp} onClick={handleCanvasClick} style={{ cursor: isPanning || draggingIdx?.type === 'design' ? 'grabbing' : (canDragArtwork ? 'grab' : (activeMode !== 'calibrate' ? 'crosshair' : 'default')), overflow: 'hidden' }}>
            <div className="canvas-container" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: '0 0', width: '100%', height: '100%', position: 'relative', ['--zoom' as any]: zoom }}>
                <img src={imageUrl} alt="Mockup" className="mockup-img" onLoad={e => setNaturalSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
                {showGridOverlay && gridOverlayUrl && !showDesignOverlay && (
                  <img
                    src={gridOverlayUrl}
                    className="overlay-img overlay-img-grid overlay-img-grid-base"
                    alt=""
                  />
                )}
                {showDesignOverlay && previewOverlayUrl && (
                  <img
                    src={previewOverlayUrl}
                    className={`overlay-img ${editorMode === 'CALIBRATE' ? 'overlay-img-calibrate' : 'overlay-img-design'} ${wantsDesignOverlay ? 'overlay-img-original-color' : ''} ${canDragArtwork ? 'overlay-img-interactive' : ''}`}
                    alt=""
                    onPointerDown={handleArtworkPointerDown}
                  />
                )}
                {showGridOverlay && gridOverlayUrl && showDesignOverlay && (
                  <img
                    src={gridOverlayUrl}
                    className="overlay-img overlay-img-grid overlay-img-grid-top"
                    alt=""
                  />
                )}
                <svg className="overlay-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
                  {isCylinderProduct
                    ? <MugCurvePreview calibPts={calibPts} curvePct={curvePct} curveTop={curveTop} curveBot={curveBot} edgeSqueeze={edgeSqueeze} squeezePower={squeezePower} centerFocusWidth={centerFocusWidth} meshDensityStrength={meshDensityStrength} hPx={H_px} wPx={W_px} showGrid={showEditorGridLayer} onSmileDrag={setCurveTop} onPitchDrag={setCurveBot} />
                    : <WarpGridPreview meshPoints={meshPoints} showGrid={showEditorGridLayer} />}
                  {isCylinderProduct && <WarpGridPreview meshPoints={meshPoints} showGrid={showCylinderMeshOverlay} />}
                </svg>
                {activeGroup === 'edge' && (
                  <canvas ref={edgeCanvasRef} className="edge-canvas" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 30 }} />
                )}
              {editorMode === 'CALIBRATE' && (
                <>
                  {activeMode === 'calibrate' && calibPts.map((p, i) => <div key={i} className={`handle handle-corner-dot ${selectedHandle?.type === 'base' && selectedHandle.id === i ? 'selected' : ''}`} style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }} onPointerDown={e => handlePointerDown('base', i, e)} />)}
                  {activeMode === 'calibrate' && productType.startsWith('cylinder') && ( <> <div className={`handle handle-smile-dot ${selectedHandle?.type === 'smile' ? 'selected' : ''}`} style={{ left: `${smileHandle.x * 100}%`, top: `${smileHandle.y * 100}%` }} onPointerDown={e => handlePointerDown('smile', 0, e)} /> <div className={`handle handle-pitch-dot ${selectedHandle?.type === 'pitch' ? 'selected' : ''}`} style={{ left: `${pitchHandle.x * 100}%`, top: `${pitchHandle.y * 100}%` }} onPointerDown={e => handlePointerDown('pitch', 0, e)} /> </> )}
                  {activeMode === 'calibrate' && activeGroup === 'blend' && productType.startsWith('cylinder') && <div className={`handle handle-light-dot ${selectedHandle?.type === 'light' ? 'selected' : ''}`} style={{ left: `${lightHandle.x * 100}%`, top: `${lightHandle.y * 100}%` }} onPointerDown={e => handlePointerDown('light', 0, e)} />}
                  {activeMode === 'mask' && maskPoints.map((p, i) => <div key={i} className={`handle handle-mask ${selectedHandle?.type === 'mask' && selectedHandle.id === i ? 'selected' : ''}`} style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }} onPointerDown={e => handlePointerDown('mask', i, e)} />)}
                  {activeMode === 'mesh' && meshPoints.map((m, i) => <div key={i} className={`handle handle-mesh ${selectedHandle?.type === 'mesh' && selectedHandle.id === i ? 'selected' : ''}`} style={{ left: `${m.dst.x * 100}%`, top: `${m.dst.y * 100}%` }} onPointerDown={e => handlePointerDown('mesh', i, e)} />)}
                </>
              )}
            </div>
          </div>
        </main>
      </div>

      {apiError && <div className="error-toast"><span>Cảnh báo: {apiError}</span><button className="btn-ghost" onClick={() => setApiError(null)}>Đóng</button></div>}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

        .pae-root {
          display: flex; flex-direction: column; height: 100vh;
          background: #0a0c10; color: #e2e8f0; font-family: 'Syne', sans-serif; overflow: hidden;
        }
        .pae-root:focus { outline: none; }

        .pae-body {
          display: flex; flex: 1; min-height: 0; overflow: hidden;
        }

        /* Toolbar */
        .toolbar {
          display: flex; align-items: center; justify-content: space-between;
          padding: 0 5px; height: 48px; background: #0e1117;
          border-bottom: 1px solid #1e2330; flex-shrink: 0; gap: 12px; z-index: 100;
        }
        .toolbar-left, .toolbar-right { display: flex; align-items: center; gap: 12px; }
        .logo-badge {
          font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
          color: #475569; padding: 5px; border: 1px solid #1e2330; border-radius: 6px;
        }
        .pill {
          padding: 5px; border: none; background: transparent; color: #475569;
          cursor: pointer; border-radius: 6px; font-family: 'Syne', sans-serif;
          font-size: 12px; font-weight: 600; letter-spacing: 0.04em; transition: all 0.15s;
        }
        .pill-sm { font-size: 10px; padding: 5px; }
        .pill:hover { background: #161b27; color: #94a3b8; }
        .pill.active { background: #1a2540; color: #60a5fa; }
        .icon-btn {
          display: flex; align-items: center; gap: 6px; padding: 5px;
          border: 1px solid #1e2330; background: transparent; color: #475569;
          cursor: pointer; border-radius: 6px; font-family: 'Syne', sans-serif;
          font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; transition: all 0.15s;
        }
        .icon-btn:hover { border-color: #2d3a52; color: #94a3b8; }
        .icon-btn.active { border-color: #3b82f6; color: #60a5fa; background: #1a2540; }
        .icon-btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .btn-detect {
          display: flex; align-items: center; gap: 8px; padding: 5px;
          background: linear-gradient(135deg, #1d4ed8, #2563eb); color: white; border: none;
          border-radius: 8px; font-family: 'Syne', sans-serif; font-size: 12px;
          font-weight: 700; cursor: pointer; letter-spacing: 0.04em; transition: opacity 0.15s;
        }
        .btn-detect:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-confirm-lock {
          background: #3b82f6; color: white; border: none; padding: 5px;
          border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 13px; transition: all 0.2s;
        }
        .btn-confirm-lock:hover { background: #2563eb; transform: scale(1.05); }
        .btn-unlock {
          background: #374151; color: #9ca3af; border: 1px solid #4b5563;
          padding: 5px; border-radius: 6px; font-weight: 600; cursor: pointer; font-size: 13px;
        }
        .btn-unlock:hover { color: white; border-color: #6b7280; }

        /* Sidebar */
        .sidebar {
          width: 400px; background: #0e1117; border-right: 1px solid #1e2330;
          display: flex; flex-direction: column; min-height: 0; overflow: hidden; flex-shrink: 0;
        }
        .ctrl-section {
          flex: 1; min-height: 0; overflow-y: auto; overscroll-behavior: contain;
          scrollbar-gutter: stable;
          padding: 5px; display: flex; flex-direction: column; gap: 14px;
        }
        .group-tabs {
          display: flex; flex-wrap: wrap; gap: 6px;
          padding-bottom: 5px; border-bottom: 1px solid #1e2330;
        }
        .group-tab {
          padding: 5px; text-align: left; background: transparent; border: 1px solid transparent;
          color: #475569; cursor: pointer; border-radius: 6px;
          font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 600;
          letter-spacing: 0.05em; transition: all 0.15s;
        }
        .group-tab:hover { color: #94a3b8; background: #141924; }
        .group-tab.active { border-color: #1e2d44; color: #60a5fa; background: #111827; }
        .ctrl-grid {
          display: flex; flex-direction: column; gap: 14px;
        }
        .ctrl-grid-two {
          display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-items: start;
        }
        .point-nudge-panel {
          display: flex;
          flex-direction: column;
          gap: 10px;
          margin-top: 12px;
          padding-top: 10px;
          border-top: 1px solid #1e2330;
        }
        .nudge-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 6px;
          align-items: center;
        }
        .nudge-btn {
          min-height: 30px;
          padding: 5px;
        }
        .nudge-btn:disabled {
          opacity: 0.45;
          cursor: not-allowed;
        }
        .ctrl { display: flex; flex-direction: column; gap: 6px; }
        .ctrl-header { display: flex; justify-content: space-between; align-items: baseline; }
        .ctrl-label {
          font-size: 10px; font-weight: 700; text-transform: uppercase;
          letter-spacing: 0.1em; color: #475569;
        }
        .ctrl-value { font-family: 'DM Mono', monospace; font-size: 11px; color: #60a5fa; }
        .stepper {
          display: grid;
          grid-template-columns: 38px minmax(0, 1fr) 38px;
          gap: 6px;
          align-items: center;
          padding: 4px;
          border: 1px solid #1e2330;
          border-radius: 8px;
          background: #111827;
          outline: none;
          transition: border-color 0.15s, box-shadow 0.15s;
        }
        .stepper:focus {
          border-color: #3b82f6;
          box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
        }
        .stepper-btn,
        .stepper-input {
          border: none;
          border-radius: 8px;
          background: #1a2540;
          color: #dbeafe;
          font-family: 'Syne', sans-serif;
        }
        .stepper-btn {
          height: 32px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 700;
          transition: background 0.15s, transform 0.1s;
        }
        .stepper-btn:hover { background: #223153; }
        .stepper-btn:active { transform: scale(0.98); }
        .stepper-input {
          height: 32px;
          width: 100%;
          padding: 0 8px;
          cursor: text;
          text-align: center;
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          font-weight: 500;
          letter-spacing: 0.04em;
        }
        .stepper-input:focus {
          outline: none;
          box-shadow: inset 0 0 0 1px rgba(147, 197, 253, 0.45);
        }
        .ctrl-step {
          font-size: 9px;
          color: #475569;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }
        .ctrl-help {
          font-size: 10px;
          line-height: 1.4;
          color: #64748b;
        }
        .mode-hint { font-size: 11px; color: #64748b; line-height: 1.45; }
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

        /* Canvas */
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
        .overlay-img-grid-base {
          opacity: 1;
        }
        .overlay-img-grid-top {
          opacity: 0.42;
          mix-blend-mode: screen;
          filter: contrast(1.1) brightness(1.08);
          z-index: 6;
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

        /* Grid lines */
        .grid-boundary-unified {
          fill: none; stroke: #00f2ff; stroke-width: 0.3; opacity: 0.7;
          pointer-events: none; vector-effect: non-scaling-stroke;
        }
        .grid-line {
          fill: none; stroke: #7dd3fc; stroke-width: 0.24; opacity: 0.82;
          vector-effect: non-scaling-stroke;
        }

        /* Handles */
        .handle {
          position: absolute; cursor: pointer; z-index: 100;
          display: flex; align-items: center; justify-content: center;
          transition: transform 0.1s;
          transform: translate(-50%, -50%) scale(calc(1 / var(--zoom, 1)));
          background: transparent;
        }
        .handle.selected {
          filter: drop-shadow(0 0 6px rgba(147, 197, 253, 0.85));
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
        .handle-light-dot { width: 20px; height: 20px; }
        .handle-light-dot::before {
          content: '';
          position: absolute;
          inset: 2px;
          border-radius: 50%;
          border: 1px solid rgba(253, 224, 71, 0.95);
          box-shadow: 0 0 12px rgba(250, 204, 21, 0.45);
        }
        .handle-light-dot::after {
          content: '';
          width: 8px;
          height: 8px;
          background: #facc15;
          border-radius: 50%;
          border: 1px solid rgba(255,255,255,0.9);
        }
        .handle-mask {
          width: 14px; height: 14px; background: #059669;
          border: 2px solid #6ee7b7; border-radius: 3px;
        }
        .handle-mesh {
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: transparent;
        }
        .handle-mesh::after {
          content: '';
          width: 5px;
          height: 5px;
          background: #050505;
          border-radius: 50%;
          border: 1px solid rgba(255, 255, 255, 0.72);
          box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35);
        }

        /* Misc */
        .select {
          background: #141924; color: #94a3b8; border: 1px solid #1e2330;
          padding: 5px; border-radius: 6px; font-family: 'Syne', sans-serif;
          font-size: 12px; cursor: pointer; width: 100%;
        }
        .select:focus { outline: none; border-color: #3b82f6; }
        .mode-hint { font-size: 11px; color: #64748b; line-height: 1.45; }
        .btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
        .btn-primary {
          padding: 5px; background: #1d4ed8; color: white; border: none;
          border-radius: 7px; font-family: 'Syne', sans-serif; font-size: 11px;
          font-weight: 700; cursor: pointer; transition: background 0.15s;
        }
        .btn-ghost {
          padding: 5px; background: transparent; color: #94a3b8;
          border: 1px solid #1e2330; border-radius: 7px; font-family: 'Syne', sans-serif;
          font-size: 11px; font-weight: 600; cursor: pointer; transition: all 0.15s;
        }
        .btn-ghost:hover { background: #141924; color: #e2e8f0; }
        .error-toast {
          position: fixed; bottom: 20px; right: 20px;
          background: rgba(220, 38, 38, 0.9); color: white; padding: 5px;
          border-radius: 8px; display: flex; gap: 12px; align-items: center; z-index: 1000;
        }
      `}</style>
    </div>
  );
}


