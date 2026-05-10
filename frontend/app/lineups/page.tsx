"use client";

import { useSlate } from "@/app/context/SlateContext";
import LineupCard from "@/components/LineupCard";
import ExposureChart from "@/components/ExposureChart";
import StatBadge from "@/components/StatBadge";

export default function LineupsPage() {
  const { lineups, playerPool, exportLineups, loading, error } = useSlate();

  const valid = lineups.filter((l) => l.is_valid);
  const invalid = lineups.filter((l) => !l.is_valid);
  const avgSalary =
    lineups.length > 0
      ? Math.round(
          lineups.reduce((s, l) => s + l.total_salary, 0) / lineups.length
        )
      : 0;
  const avgPts =
    lineups.length > 0
      ? (
          lineups.reduce((s, l) => s + l.projected_pts, 0) / lineups.length
        ).toFixed(1)
      : "0.0";
  const avgLev =
    lineups.length > 0
      ? (
          lineups.reduce((s, l) => s + l.leverage_score, 0) / lineups.length
        ).toFixed(2)
      : "0.00";

  async function handleExport() {
    await exportLineups();
  }

  if (!lineups.length) {
    return (
      <div className="max-w-screen-2xl mx-auto px-6 py-16 text-center">
        <div className="text-4xl mb-4">📋</div>
        <h2 className="text-xl font-semibold text-slate-400 mb-2">
          No lineups yet
        </h2>
        <p className="text-slate-600 text-sm">
          Go to the OPTIMIZER tab and click Generate Lineups.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-screen-2xl mx-auto px-6 py-6 flex flex-col gap-6 relative">
      {loading && (
        <div
          className="absolute inset-0 z-30 flex items-center justify-center rounded-lg"
          style={{ backgroundColor: "rgba(10,14,26,0.65)" }}
        >
          <div className="flex flex-col items-center gap-3">
            <span className="inline-block w-10 h-10 border-2 border-slate-600 border-t-[#00ff88] rounded-full animate-spin" />
            <span className="text-xs uppercase tracking-widest text-slate-400">
              Working…
            </span>
          </div>
        </div>
      )}

      {error && (
        <div
          className="px-4 py-3 rounded text-sm font-medium"
          style={{
            backgroundColor: "rgba(239,68,68,0.08)",
            border: "1px solid #ef444440",
            color: "#ef4444",
          }}
        >
          {error}
        </div>
      )}

      {/* Summary row */}
      <section>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            {
              value: `${valid.length}/${lineups.length}`,
              label: "Valid Lineups",
              color:
                valid.length === lineups.length
                  ? ("green" as const)
                  : ("amber" as const),
            },
            {
              value: `$${avgSalary.toLocaleString()}`,
              label: "Avg Salary",
              color: "green" as const,
            },
            { value: avgPts, label: "Avg Proj Pts", color: "green" as const },
            {
              value: avgLev,
              label: "Avg Leverage",
              color: "amber" as const,
            },
          ].map(({ value, label, color }) => (
            <div
              key={label}
              className="rounded-lg px-5 py-4"
              style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
            >
              <StatBadge value={value} label={label} color={color} size="lg" />
            </div>
          ))}
        </div>

        {invalid.length > 0 && (
          <div
            className="mt-3 px-4 py-3 rounded text-sm"
            style={{
              backgroundColor: "rgba(239,68,68,0.06)",
              border: "1px solid #ef444430",
              color: "#ef4444",
            }}
          >
            ⚠ {invalid.length} lineup{invalid.length !== 1 ? "s" : ""} invalid —
            review before export
          </div>
        )}
      </section>

      <div className="flex gap-6 items-start">
        {/* Lineup grid */}
        <div className="flex-1 min-w-0">
          <h2
            className="text-xs font-semibold uppercase tracking-widest mb-4"
            style={{ color: "#00ff88" }}
          >
            Generated Lineups — click to expand
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3">
            {lineups.map((lineup) => (
              <LineupCard key={lineup.id} lineup={lineup} />
            ))}
          </div>
        </div>

        {/* Sidebar: exposure + export */}
        <div className="w-80 shrink-0 flex flex-col gap-5">
          {/* Export */}
          <div
            className="rounded-lg p-5"
            style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
          >
            <h3
              className="text-xs font-semibold uppercase tracking-widest mb-3"
              style={{ color: "#00ff88" }}
            >
              Export
            </h3>
            <div className="text-xs text-slate-500 mb-4 space-y-1">
              <div>
                <span
                  className="font-data"
                  style={{
                    color:
                      valid.length === lineups.length ? "#00ff88" : "#f59e0b",
                  }}
                >
                  {valid.length}/{lineups.length}
                </span>{" "}
                lineups valid
              </div>
              <div>
                Format:{" "}
                <span className="font-data text-slate-400">
                  P,P,C,1B,2B,3B,SS,OF,OF,OF
                </span>
              </div>
            </div>

            {invalid.length > 0 && (
              <div
                className="text-xs px-3 py-2 rounded mb-3"
                style={{
                  backgroundColor: "rgba(239,68,68,0.08)",
                  border: "1px solid #ef444430",
                  color: "#ef4444",
                }}
              >
                ⚠ {invalid.length} invalid lineup{invalid.length !== 1 ? "s" : ""}
              </div>
            )}

            <button
              onClick={() => void handleExport()}
              disabled={loading}
              className="w-full py-3 rounded-lg font-semibold text-sm tracking-wide transition-all duration-150"
              style={{
                backgroundColor: loading ? "#1e2d4a" : "#00ff88",
                color: loading ? "#475569" : "#0a0e1a",
                cursor: loading ? "not-allowed" : "pointer",
                boxShadow: loading ? "none" : "0 0 16px rgba(0,255,136,0.25)",
              }}
            >
              ↓ Export DK Upload CSV
            </button>
          </div>

          {/* Exposure chart */}
          <div
            className="rounded-lg p-5"
            style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
          >
            <h3
              className="text-xs font-semibold uppercase tracking-widest mb-1"
              style={{ color: "#00ff88" }}
            >
              Player Exposure
            </h3>
            <p className="text-xs text-slate-600 mb-4">
              Red line = 70% max exposure
            </p>
            <ExposureChart lineups={lineups} players={playerPool} />
          </div>
        </div>
      </div>
    </div>
  );
}
