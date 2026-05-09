"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Player } from "@/app/context/SlateContext";

interface Props {
  players: Player[];
}

function dotColor(lev: number): string {
  if (lev >= 1.3) return "#00ff88";
  if (lev >= 0.8) return "#f59e0b";
  return "#475569";
}

export default function LeverageScatter({ players }: Props) {
  const data = players.map((p) => ({
    x: p.ownership,
    y: p.proj_pts,
    name: p.name,
    leverage: p.leverage,
    salary: p.salary,
    color: dotColor(p.leverage),
    is_pitcher: p.is_pitcher,
  }));

  const avgOwn = parseFloat(
    (players.reduce((s, p) => s + p.ownership, 0) / players.length).toFixed(1)
  );
  const avgPts = parseFloat(
    (players.reduce((s, p) => s + p.proj_pts, 0) / players.length).toFixed(1)
  );

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: { payload: (typeof data)[0] }[];
  }) => {
    if (active && payload && payload[0]) {
      const d = payload[0].payload;
      return (
        <div
          className="rounded px-3 py-2 text-xs"
          style={{
            backgroundColor: "#0f1629",
            border: "1px solid #1e2d4a",
          }}
        >
          <div className="font-semibold text-slate-200 mb-1">{d.name}</div>
          <div className="font-data space-y-0.5">
            <div style={{ color: "#00ff88" }}>
              Proj: {d.y.toFixed(1)} pts
            </div>
            <div className="text-slate-400">Own: {d.x.toFixed(1)}%</div>
            <div className="text-slate-400">
              ${d.salary.toLocaleString()}
            </div>
            <div
              style={{ color: dotColor(d.leverage) }}
            >
              Lev: {d.leverage.toFixed(2)}
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  const CustomDot = (props: {
    cx?: number;
    cy?: number;
    payload?: (typeof data)[0];
  }) => {
    const { cx, cy, payload } = props;
    if (cx === undefined || cy === undefined || !payload) return null;
    return (
      <circle
        cx={cx}
        cy={cy}
        r={payload.is_pitcher ? 5 : 4}
        fill={payload.color}
        fillOpacity={0.75}
        stroke={payload.is_pitcher ? payload.color : "none"}
        strokeWidth={payload.is_pitcher ? 1.5 : 0}
      />
    );
  };

  return (
    <div className="relative" style={{ height: 320 }}>
      {/* Quadrant labels */}
      <div
        className="absolute text-xs font-semibold tracking-wider pointer-events-none z-10"
        style={{ top: 8, left: 80, color: "#00ff88", opacity: 0.6 }}
      >
        ★ GPP GOLD
      </div>
      <div
        className="absolute text-xs text-slate-600 tracking-wider pointer-events-none z-10"
        style={{ top: 8, right: 16 }}
      >
        POPULAR
      </div>
      <div
        className="absolute text-xs text-slate-600 tracking-wider pointer-events-none z-10"
        style={{ bottom: 8, left: 80 }}
      >
        LOW UPSIDE
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 24, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid stroke="#1e2d4a" strokeDasharray="2 4" />
          <XAxis
            dataKey="x"
            name="Ownership"
            type="number"
            tick={{
              fill: "#475569",
              fontSize: 11,
              fontFamily: "JetBrains Mono",
            }}
            tickFormatter={(v) => `${v}%`}
            axisLine={{ stroke: "#1e2d4a" }}
            tickLine={false}
            label={{
              value: "Projected Ownership %",
              position: "insideBottom",
              offset: -12,
              fill: "#475569",
              fontSize: 11,
            }}
          />
          <YAxis
            dataKey="y"
            name="Proj Pts"
            type="number"
            tick={{
              fill: "#475569",
              fontSize: 11,
              fontFamily: "JetBrains Mono",
            }}
            axisLine={{ stroke: "#1e2d4a" }}
            tickLine={false}
            label={{
              value: "Proj Pts (q50)",
              angle: -90,
              position: "insideLeft",
              fill: "#475569",
              fontSize: 11,
            }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: "3 3", stroke: "#1e2d4a" }} />
          <ReferenceLine x={avgOwn} stroke="#1e2d4a" strokeDasharray="4 2" />
          <ReferenceLine y={avgPts} stroke="#1e2d4a" strokeDasharray="4 2" />
          <Scatter
            data={data}
            shape={<CustomDot />}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
