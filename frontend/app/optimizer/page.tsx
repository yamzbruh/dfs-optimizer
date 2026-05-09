"use client";

import { useState } from "react";
import { useSlate, MOCK_LINEUPS } from "@/app/context/SlateContext";
import LockBanPanel from "@/components/LockBanPanel";

export default function OptimizerPage() {
  const {
    players,
    lockedIds,
    bannedIds,
    addLock,
    removeLock,
    addBan,
    removeBan,
    setLineups,
    isGenerating,
    setIsGenerating,
  } = useSlate();

  const [maxExposure, setMaxExposure] = useState(70);
  const [nLineups, setNLineups] = useState(20);
  const [generated, setGenerated] = useState(false);

  // Top leverage plays
  const leveragePlays = [...players]
    .filter((p) => !p.is_pitcher)
    .sort((a, b) => b.leverage - a.leverage)
    .slice(0, 10);

  // Popoff candidates: high q85 relative to q50
  const popoffs = [...players]
    .sort((a, b) => (b.proj_pts_q85 - b.proj_pts) - (a.proj_pts_q85 - a.proj_pts))
    .slice(0, 8);

  function handleGenerate() {
    setIsGenerating(true);
    // Simulate generation delay
    setTimeout(() => {
      setLineups(MOCK_LINEUPS);
      setIsGenerating(false);
      setGenerated(true);
    }, 1800);
  }

  return (
    <div className="max-w-screen-2xl mx-auto px-6 py-6">
      <div className="flex gap-6">

        {/* Left panel: Controls */}
        <div
          className="w-80 shrink-0 flex flex-col gap-6 rounded-lg p-5"
          style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
        >
          <h2
            className="text-xs font-semibold uppercase tracking-widest"
            style={{ color: "#00ff88" }}
          >
            Optimizer Controls
          </h2>

          <LockBanPanel
            players={players}
            lockedIds={lockedIds}
            bannedIds={bannedIds}
            onLock={addLock}
            onUnlock={removeLock}
            onBan={addBan}
            onUnban={removeBan}
          />

          <div style={{ borderTop: "1px solid #1e2d4a" }} />

          {/* Settings */}
          <div className="flex flex-col gap-4">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500">
              Settings
            </h3>

            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Max Exposure</span>
                <span
                  className="font-data font-semibold"
                  style={{ color: "#f59e0b" }}
                >
                  {maxExposure}%
                </span>
              </div>
              <input
                type="range"
                min={10}
                max={100}
                step={5}
                value={maxExposure}
                onChange={(e) => setMaxExposure(Number(e.target.value))}
                className="w-full accent-amber-400"
              />
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Lineups to Generate</span>
                <span
                  className="font-data font-semibold"
                  style={{ color: "#00ff88" }}
                >
                  {nLineups}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={20}
                step={1}
                value={nLineups}
                onChange={(e) => setNLineups(Number(e.target.value))}
                className="w-full accent-green-400"
              />
            </div>

            {lockedIds.length > 0 && (
              <div
                className="text-xs px-3 py-2 rounded"
                style={{
                  backgroundColor: "rgba(0,255,136,0.06)",
                  border: "1px solid #00ff8820",
                  color: "#00ff88",
                }}
              >
                {lockedIds.length} player{lockedIds.length !== 1 ? "s" : ""} locked · will appear in all {nLineups} lineups
              </div>
            )}
            {bannedIds.length > 0 && (
              <div
                className="text-xs px-3 py-2 rounded"
                style={{
                  backgroundColor: "rgba(239,68,68,0.06)",
                  border: "1px solid #ef444420",
                  color: "#ef4444",
                }}
              >
                {bannedIds.length} player{bannedIds.length !== 1 ? "s" : ""} banned
              </div>
            )}
          </div>

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="w-full py-3 rounded-lg font-semibold text-sm tracking-wide transition-all duration-150 flex items-center justify-center gap-2"
            style={{
              backgroundColor: isGenerating ? "#1e2d4a" : "#00ff88",
              color: isGenerating ? "#475569" : "#0a0e1a",
              cursor: isGenerating ? "not-allowed" : "pointer",
              boxShadow: isGenerating ? "none" : "0 0 20px rgba(0,255,136,0.3)",
            }}
          >
            {isGenerating ? (
              <>
                <span className="inline-block w-4 h-4 border-2 border-slate-500 border-t-slate-300 rounded-full animate-spin" />
                Generating…
              </>
            ) : (
              `⚡ Generate ${nLineups} Lineups`
            )}
          </button>

          {generated && !isGenerating && (
            <div
              className="text-xs text-center font-data"
              style={{ color: "#00ff88" }}
            >
              ✓ {nLineups} lineups ready → view in LINEUPS tab
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="flex-1 flex flex-col gap-6 min-w-0">

          {/* Top Leverage Plays */}
          <section>
            <div className="flex items-baseline justify-between mb-3">
              <h2
                className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: "#00ff88" }}
              >
                Top 10 Leverage Plays
              </h2>
              <span className="text-xs text-slate-500">
                High projection · low ownership = GPP edge
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {leveragePlays.map((p, i) => {
                const isTop = i < 3;
                return (
                  <div
                    key={p.dk_id}
                    className={`rounded-lg p-4 transition-all duration-150 ${isTop ? "glow-green-sm" : ""}`}
                    style={{
                      backgroundColor: "#0f1629",
                      border: `1px solid ${isTop ? "#00ff8840" : "#1e2d4a"}`,
                    }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <div className="flex items-center gap-2">
                          {isTop && (
                            <span
                              className="font-data text-xs font-bold"
                              style={{ color: "#00ff88" }}
                            >
                              #{i + 1}
                            </span>
                          )}
                          <span className="font-semibold text-slate-200">
                            {p.name}
                          </span>
                        </div>
                        <span className="text-xs text-slate-500">
                          {p.team} · {p.dk_position}
                        </span>
                      </div>
                      <span
                        className="font-data text-sm font-bold"
                        style={{ color: "#00ff88" }}
                      >
                        {p.leverage.toFixed(2)}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <div>
                        <div
                          className="font-data text-sm font-semibold"
                          style={{ color: "#f8fafc" }}
                        >
                          {p.proj_pts.toFixed(1)}
                        </div>
                        <div className="text-xs text-slate-600">Proj pts</div>
                      </div>
                      <div>
                        <div className="font-data text-sm text-slate-300">
                          {p.ownership.toFixed(1)}%
                        </div>
                        <div className="text-xs text-slate-600">Own%</div>
                      </div>
                      <div>
                        <div
                          className="font-data text-sm font-semibold"
                          style={{ color: "#00ff88" }}
                        >
                          ${p.salary.toLocaleString()}
                        </div>
                        <div className="text-xs text-slate-600">Salary</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Popoff Players */}
          <section>
            <div className="flex items-baseline justify-between mb-3">
              <h2
                className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: "#f59e0b" }}
              >
                Potential Popoff Players
              </h2>
              <span className="text-xs text-slate-500">
                Wide q15→q85 interval = high ceiling upside
              </span>
            </div>
            <div
              className="overflow-auto rounded-lg"
              style={{ border: "1px solid #1e2d4a" }}
            >
              <table className="w-full text-sm" style={{ borderCollapse: "collapse" }}>
                <thead
                  style={{ backgroundColor: "#080c18", borderBottom: "1px solid #1e2d4a" }}
                >
                  <tr>
                    {["Name", "Team", "Salary", "Median (q50)", "Ceiling (q85)", "Interval", "Own%"].map(
                      (h) => (
                        <th
                          key={h}
                          className="px-4 py-2 text-xs font-semibold uppercase tracking-widest text-slate-500 text-left"
                        >
                          {h}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {popoffs.map((p) => {
                    const interval = p.proj_pts_q85 - p.proj_pts_q15;
                    return (
                      <tr
                        key={p.dk_id}
                        style={{ borderBottom: "1px solid #111827" }}
                        className="hover:bg-[#ffffff06]"
                      >
                        <td className="px-4 py-2.5 font-medium text-slate-200">
                          {p.name}
                        </td>
                        <td className="px-4 py-2.5 font-data text-xs text-slate-400">
                          {p.team}
                        </td>
                        <td
                          className="px-4 py-2.5 font-data font-semibold"
                          style={{ color: "#00ff88" }}
                        >
                          ${p.salary.toLocaleString()}
                        </td>
                        <td className="px-4 py-2.5 font-data text-slate-300">
                          {p.proj_pts.toFixed(1)}
                        </td>
                        <td
                          className="px-4 py-2.5 font-data font-semibold"
                          style={{ color: "#f59e0b" }}
                        >
                          {p.proj_pts_q85.toFixed(1)}
                        </td>
                        <td className="px-4 py-2.5 font-data text-slate-300">
                          <span
                            className="px-2 py-0.5 rounded text-xs"
                            style={{
                              backgroundColor: "rgba(245,158,11,0.1)",
                              color: "#f59e0b",
                            }}
                          >
                            ±{interval.toFixed(1)}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 font-data text-slate-400">
                          {p.ownership.toFixed(1)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
