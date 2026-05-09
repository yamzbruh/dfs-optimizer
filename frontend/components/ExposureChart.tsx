"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Lineup, Player } from "@/app/context/SlateContext";

interface Props {
  lineups: Lineup[];
  players: Player[];
}

export default function ExposureChart({ lineups, players }: Props) {
  if (!lineups.length) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-600 text-sm">
        No lineups generated yet.
      </div>
    );
  }

  // Calculate exposure for each player
  const exposureMap: Record<string, number> = {};
  for (const lineup of lineups) {
    for (const lp of lineup.players) {
      exposureMap[lp.player.dk_id] =
        (exposureMap[lp.player.dk_id] || 0) + 1;
    }
  }

  const data = Object.entries(exposureMap)
    .map(([dk_id, count]) => {
      const player = players.find((p) => p.dk_id === dk_id);
      return {
        name: player ? player.name.split(" ").slice(-1)[0] : dk_id,
        fullName: player?.name ?? dk_id,
        exposure: parseFloat(((count / lineups.length) * 100).toFixed(1)),
        is_pitcher: player?.is_pitcher ?? false,
      };
    })
    .sort((a, b) => b.exposure - a.exposure)
    .slice(0, 20);

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
          className="rounded px-3 py-2 text-xs font-data"
          style={{
            backgroundColor: "#0f1629",
            border: "1px solid #1e2d4a",
            color: "#f8fafc",
          }}
        >
          <div className="font-semibold text-slate-200">{d.fullName}</div>
          <div style={{ color: "#00ff88" }}>{d.exposure}% exposure</div>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ height: Math.max(200, data.length * 28 + 60) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={data}
          margin={{ top: 4, right: 48, left: 80, bottom: 4 }}
        >
          <CartesianGrid
            horizontal={false}
            stroke="#1e2d4a"
            strokeDasharray="2 4"
          />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fill: "#475569", fontSize: 11, fontFamily: "JetBrains Mono" }}
            tickFormatter={(v) => `${v}%`}
            axisLine={{ stroke: "#1e2d4a" }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "Inter" }}
            axisLine={false}
            tickLine={false}
            width={76}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "#ffffff06" }} />
          <ReferenceLine
            x={70}
            stroke="#ef4444"
            strokeDasharray="4 2"
            label={{
              value: "70%",
              fill: "#ef4444",
              fontSize: 10,
              fontFamily: "JetBrains Mono",
            }}
          />
          <Bar dataKey="exposure" radius={[0, 2, 2, 0]} maxBarSize={18}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={
                  entry.exposure > 70
                    ? "#ef4444"
                    : entry.is_pitcher
                    ? "#60a5fa"
                    : "#00ff88"
                }
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
