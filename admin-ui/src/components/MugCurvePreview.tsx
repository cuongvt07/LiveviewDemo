import React from 'react';

interface Point {
  x: number;
  y: number;
}

interface MugCurvePreviewProps {
  calibPts: Point[];
  curvePct: number;
  curveTop: number;
  curveBot: number;
  edgeSqueeze: number;
  squeezePower: number;
  centerFocusWidth: number;
  meshDensityStrength: number;
  hPx: number;
  wPx: number;
  showGrid: boolean;
  onSmileDrag: (y: number) => void;
  onPitchDrag: (y: number) => void;
}

const BASE_CENTER_BAND = 0.30;
const MIN_CENTER_BAND = 0.10;
const MAX_CENTER_BAND = 0.70;
const BASE_EDGE_ROLL_START = 0.55;
const GRID_CELL_PX = 90;
const MIN_GRID_DIVS = 4;
const MAX_GRID_DIVS = 24;

function clampGridDivs(value: number): number {
  return Math.max(MIN_GRID_DIVS, Math.min(MAX_GRID_DIVS, value));
}

function computeAdaptiveGrid(wPx: number, hPx: number): { cols: number; rows: number } {
  if (!Number.isFinite(wPx) || !Number.isFinite(hPx) || wPx <= 0 || hPx <= 0) {
    return { cols: 12, rows: 10 };
  }
  return {
    cols: clampGridDivs(Math.round(wPx / GRID_CELL_PX)),
    rows: clampGridDivs(Math.round(hPx / GRID_CELL_PX)),
  };
}

function computeAdaptiveAxisEdges(divisions: number, meshDensityStrength: number): number[] {
  const divs = Math.max(1, Math.floor(divisions));
  const power = Math.max(1, meshDensityStrength);
  return Array.from({ length: divs + 1 }, (_, index) => {
    const t = index / divs;
    if (power <= 1 + 1e-8) return t;
    if (t <= 0.5) return 0.5 * Math.pow(t / 0.5, power);
    return 1 - 0.5 * Math.pow((1 - t) / 0.5, power);
  }).map((value, index, arr) => {
    if (index === 0) return 0;
    if (index === arr.length - 1) return 1;
    return Math.max(0, Math.min(1, value));
  });
}

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
  const x = u * 2 - 1;
  const y = v * 2 - 1;
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w,
  };
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

export const MugCurvePreview: React.FC<MugCurvePreviewProps> = ({
  calibPts,
  curvePct,
  curveTop,
  curveBot,
  edgeSqueeze,
  squeezePower,
  centerFocusWidth,
  meshDensityStrength,
  hPx,
  wPx,
  showGrid,
}) => {
  const srcCanon = [
    { x: -1, y: -1 },
    { x: 1, y: -1 },
    { x: 1, y: 1 },
    { x: -1, y: 1 },
  ];
  const hMat = solve_homography(srcCanon, calibPts);

  const worldAt = (u: number, v: number): Point => {
    const wLocal = cylinderWarpPoint(u, v, curvePct, curveTop, curveBot, edgeSqueeze, squeezePower, centerFocusWidth, hPx, wPx);
    return apply_homography(hMat, wLocal.x, wLocal.y);
  };
  const patchAt = (u: number, v: number) => worldAt(u, v);

  const pathSamples = 36;
  const pathParts: string[] = [];
  for (let i = 0; i <= pathSamples; i++) {
    const u = i / pathSamples;
    const p = patchAt(u, 0);
    pathParts.push(i === 0 ? `M ${p.x * 100},${p.y * 100}` : `L ${p.x * 100},${p.y * 100}`);
  }
  for (let i = pathSamples; i >= 0; i--) {
    const u = i / pathSamples;
    const p = patchAt(u, 1);
    pathParts.push(`L ${p.x * 100},${p.y * 100}`);
  }
  pathParts.push('Z');
  const boundaryPath = pathParts.join(' ');

  const { cols, rows } = computeAdaptiveGrid(wPx, hPx);
  const uEdges = computeAdaptiveAxisEdges(cols, meshDensityStrength);
  const lineSamples = Math.min(72, Math.max(24, Math.ceil(Math.max(cols, rows) * 2)));

  return (
    <g className="mug-curve-preview">
      <defs>
        <clipPath id="mug-clip-unified">
          <path d={boundaryPath} />
        </clipPath>
      </defs>

      {showGrid && (
        <g clipPath="url(#mug-clip-unified)">
          {Array.from({ length: rows }).flatMap((_, r) =>
            Array.from({ length: cols }).map((__, c) => {
              const u0 = uEdges[c];
              const u1 = uEdges[c + 1];
              const v0 = r / rows;
              const v1 = (r + 1) / rows;
              const p00 = patchAt(u0, v0);
              const p10 = patchAt(u1, v0);
              const p11 = patchAt(u1, v1);
              const p01 = patchAt(u0, v1);
              const pts = [
                `${p00.x * 100},${p00.y * 100}`,
                `${p10.x * 100},${p10.y * 100}`,
                `${p11.x * 100},${p11.y * 100}`,
                `${p01.x * 100},${p01.y * 100}`,
              ].join(' ');
              const fill = (r + c) % 2 === 0 ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.07)';
              return <polygon key={`cell-${r}-${c}`} points={pts} fill={fill} stroke="none" />;
            })
          )}

          {Array.from({ length: cols + 1 }).map((_, ci) => {
            const u = uEdges[ci];
            const line: string[] = [];
            for (let s = 0; s <= lineSamples; s++) {
              const v = s / lineSamples;
              const p = patchAt(u, v);
              line.push(`${s === 0 ? 'M' : 'L'} ${p.x * 100},${p.y * 100}`);
            }
            return <path key={`gv-${ci}`} d={line.join(' ')} className="grid-line" />;
          })}

          {Array.from({ length: rows + 1 }).map((_, ri) => {
            const v = ri / rows;
            const line: string[] = [];
            for (let s = 0; s <= lineSamples; s++) {
              const u = s / lineSamples;
              const p = patchAt(u, v);
              line.push(`${s === 0 ? 'M' : 'L'} ${p.x * 100},${p.y * 100}`);
            }
            return <path key={`gh-${ri}`} d={line.join(' ')} className="grid-line" />;
          })}
        </g>
      )}

      <path d={boundaryPath} className="grid-boundary-unified" fill="transparent" />
    </g>
  );
};
