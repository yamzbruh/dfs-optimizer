"use client";

import { useState } from "react";
import { Lineup } from "@/app/context/SlateContext";

const SLOT_ORDER = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"];

interface Props {
  lineup: Lineup;
}

export default function LineupCard({ lineup }: Props) {
  const [expanded, setExpanded] = useState(false);
  const salaryPct = (lineup.total_salary / 50000) * 100;

  const sortedPlayers = [...lineup.players].sort((a, b) => {
    const ai = SLOT_ORDER.indexOf(a.slot);
    const bi = SLOT_ORDER.indexOf(b.slot);
    return ai - bi;
  });

  return (
    <div
      className="rounded-lg overflow-hidden cursor-pointer transition-all duration-150"
      style={{
        backgroundColor: "#0f1629",
        border: `1px solid ${lineup.is_valid ? "#1e2d4a" : "#ef444440"}`,
      }}
      onClick={() => setExpanded((v) => !v)}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3"
        style={{ borderBottom: "1px solid #1e2d4a" }}
      >
        <div className="flex items-center gap-3">
          <span
            className="font-data text-xs font-bold w-8 h-8 rounded flex items-center justify-center"
            style={{ backgroundColor: "#1e2d4a", color: "#00ff88" }}
          >
            #{lineup.id}
          </span>
          <div>
            <div className="flex items-center gap-2">
              <span
                className="font-data font-bold text-lg"
                style={{ color: "#00ff88" }}
              >
                {lineup.projected_pts.toFixed(1)}
              </span>
              <span className="text-xs text-slate-500">pts</span>
              <span
                className="text-xs font-semibold px-1.5 py-0.5 rounded font-data ml-1"
                style={{
                  backgroundColor: lineup.is_valid
                    ? "rgba(0,255,136,0.1)"
                    : "rgba(239,68,68,0.1)",
                  color: lineup.is_valid ? "#00ff88" : "#ef4444",
                }}
              >
                {lineup.is_valid ? "VALID" : "INVALID"}
              </span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div
            className="font-data text-sm font-semibold"
            style={{ color: "#f8fafc" }}
          >
            ${lineup.total_salary.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500">
            Lev: <span className="font-data" style={{ color: "#f59e0b" }}>{lineup.leverage_score.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Salary bar */}
      <div
        className="h-1 w-full"
        style={{ backgroundColor: "#1e2d4a" }}
      >
        <div
          className="h-full transition-all duration-300"
          style={{
            width: `${salaryPct}%`,
            backgroundColor:
              salaryPct > 99 ? "#00ff88" : salaryPct > 94 ? "#f59e0b" : "#1e4d6b",
          }}
        />
      </div>

      {/* Compact player list (always visible) */}
      <div className="px-4 py-2">
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {sortedPlayers.slice(0, expanded ? 10 : 4).map((lp, i) => (
            <div key={i} className="flex items-center gap-1">
              <span
                className="font-data text-xs"
                style={{ color: lp.player.is_pitcher ? "#60a5fa" : "#475569" }}
              >
                {lp.slot}
              </span>
              <span
                className="text-xs font-medium"
                style={{ color: lp.player.is_pitcher ? "#93c5fd" : "#e2e8f0" }}
              >
                {lp.player.name.split(" ").slice(-1)[0]}
              </span>
            </div>
          ))}
          {!expanded && sortedPlayers.length > 4 && (
            <span className="text-xs text-slate-600">
              +{sortedPlayers.length - 4} more
            </span>
          )}
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div style={{ borderTop: "1px solid #1e2d4a" }}>
          {sortedPlayers.map((lp, i) => (
            <div
              key={i}
              className="flex items-center justify-between px-4 py-1.5 hover:bg-[#ffffff05]"
              style={{ borderBottom: "1px solid #0a0e1a" }}
            >
              <div className="flex items-center gap-3 flex-1">
                <span
                  className="font-data text-xs w-8 text-center"
                  style={{ color: lp.player.is_pitcher ? "#60a5fa" : "#475569" }}
                >
                  {lp.slot}
                </span>
                <span
                  className="text-sm font-medium"
                  style={{ color: lp.player.is_pitcher ? "#93c5fd" : "#e2e8f0" }}
                >
                  {lp.player.name}
                </span>
                <span className="text-xs text-slate-500">{lp.player.team}</span>
              </div>
              <div className="flex items-center gap-4">
                <span
                  className="font-data text-xs"
                  style={{ color: "#00ff88" }}
                >
                  ${lp.player.salary.toLocaleString()}
                </span>
                <span className="font-data text-xs text-slate-400">
                  {lp.player.proj_pts.toFixed(1)} pts
                </span>
              </div>
            </div>
          ))}
          <div
            className="flex justify-between px-4 py-2 text-xs font-semibold"
            style={{ backgroundColor: "#080c18" }}
          >
            <span className="text-slate-400">
              ${lineup.total_salary.toLocaleString()} / $50,000
              &nbsp;&nbsp;({salaryPct.toFixed(1)}% used)
            </span>
            <span className="font-data" style={{ color: "#00ff88" }}>
              {lineup.projected_pts.toFixed(1)} proj pts
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
