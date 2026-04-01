import React from 'react';

interface Point {
  x: number;
  y: number;
}

interface WarpGridPreviewProps {
  meshPoints: { src: Point; dst: Point }[];
  showGrid: boolean;
}

const quadArea = (a: Point, b: Point, c: Point, d: Point) => {
  const area =
    a.x * b.y - a.y * b.x +
    b.x * c.y - b.y * c.x +
    c.x * d.y - c.y * d.x +
    d.x * a.y - d.y * a.x;
  return Math.abs(area) * 0.5;
};

const toEdgeKey = (a: number, b: number) => (a < b ? `${a}-${b}` : `${b}-${a}`);

export const WarpGridPreview: React.FC<WarpGridPreviewProps> = ({ meshPoints, showGrid }) => {
  if (meshPoints.length === 0 || !showGrid) return null;

  const n = meshPoints.length;
  const side = Math.round(Math.sqrt(n));
  const isStructuredGrid = side * side === n;

  const fallbackEdges = new Set<string>();
  if (!isStructuredGrid) {
    meshPoints.forEach((p, i) => {
      const nearest = meshPoints
        .map((q, j) => ({ j, d2: (q.src.x - p.src.x) ** 2 + (q.src.y - p.src.y) ** 2 }))
        .filter(item => item.j !== i)
        .sort((a, b) => a.d2 - b.d2)
        .slice(0, 3);
      nearest.forEach(item => fallbackEdges.add(toEdgeKey(i, item.j)));
    });
  }

  return (
    <g className="warp-grid-preview">
      {isStructuredGrid && (
        <g className="tps-mesh-cells">
          {Array.from({ length: side - 1 }).flatMap((_, r) =>
            Array.from({ length: side - 1 }).map((__, c) => {
              const i00 = r * side + c;
              const i01 = r * side + c + 1;
              const i11 = (r + 1) * side + c + 1;
              const i10 = (r + 1) * side + c;
              const srcA = meshPoints[i00].src;
              const srcB = meshPoints[i01].src;
              const srcC = meshPoints[i11].src;
              const srcD = meshPoints[i10].src;
              const dstA = meshPoints[i00].dst;
              const dstB = meshPoints[i01].dst;
              const dstC = meshPoints[i11].dst;
              const dstD = meshPoints[i10].dst;

              const srcArea = quadArea(srcA, srcB, srcC, srcD);
              const dstArea = quadArea(dstA, dstB, dstC, dstD);
              const ratio = srcArea > 1e-8 ? dstArea / srcArea : 1.0;

              const clamped = Math.max(0.7, Math.min(1.4, ratio));
              const expand = clamped >= 1;
              const intensity = Math.round((Math.abs(clamped - 1) / 0.4) * 140 + 35);
              const fill = expand
                ? `rgba(59, 130, 246, ${Math.min(0.35, intensity / 255)})`
                : `rgba(251, 146, 60, ${Math.min(0.35, intensity / 255)})`;

              const points = [
                `${dstA.x * 100},${dstA.y * 100}`,
                `${dstB.x * 100},${dstB.y * 100}`,
                `${dstC.x * 100},${dstC.y * 100}`,
                `${dstD.x * 100},${dstD.y * 100}`,
              ].join(' ');

              return <polygon key={`cell-${r}-${c}`} points={points} fill={fill} stroke="none" />;
            })
          )}
        </g>
      )}

      {isStructuredGrid ? (
        <g className="tps-mesh-lines">
          {Array.from({ length: side }).map((_, c) => {
            const points = Array.from({ length: side })
              .map((__, r) => {
                const p = meshPoints[r * side + c].dst;
                return `${p.x * 100},${p.y * 100}`;
              })
              .join(' ');
            return <polyline key={`v-${c}`} points={points} className="grid-line tps-grid" />;
          })}
          {Array.from({ length: side }).map((_, r) => {
            const points = Array.from({ length: side })
              .map((__, c) => {
                const p = meshPoints[r * side + c].dst;
                return `${p.x * 100},${p.y * 100}`;
              })
              .join(' ');
            return <polyline key={`h-${r}`} points={points} className="grid-line tps-grid" />;
          })}
        </g>
      ) : (
        <g className="tps-mesh-lines">
          {[...fallbackEdges].map(edge => {
            const [aStr, bStr] = edge.split('-');
            const a = Number(aStr);
            const b = Number(bStr);
            const p0 = meshPoints[a]?.dst;
            const p1 = meshPoints[b]?.dst;
            if (!p0 || !p1) return null;
            return (
              <line
                key={`edge-${edge}`}
                x1={`${p0.x * 100}`}
                y1={`${p0.y * 100}`}
                x2={`${p1.x * 100}`}
                y2={`${p1.y * 100}`}
                className="grid-line tps-grid"
              />
            );
          })}
        </g>
      )}

      <g className="tps-vectors">
        {meshPoints.map((m, i) => (
          <line
            key={`vec-${i}`}
            x1={`${m.src.x * 100}%`}
            y1={`${m.src.y * 100}%`}
            x2={`${m.dst.x * 100}%`}
            y2={`${m.dst.y * 100}%`}
            stroke="rgba(251, 191, 36, 0.35)"
            strokeWidth="0.22"
            strokeDasharray="0.7 0.7"
            vectorEffect="non-scaling-stroke"
            pointerEvents="none"
          />
        ))}
      </g>
    </g>
  );
};
