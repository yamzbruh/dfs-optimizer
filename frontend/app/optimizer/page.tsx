"use client";

import { useState } from "react";
import { useSlate } from "@/app/context/SlateContext";
import LockBanPanel from "@/components/LockBanPanel";

function formatSimsK(n: number): string {
  if (n >= 1000) {
    const k = n / 1000;
    const s =
      n % 1000 === 0 ? String(Math.round(k)) : k.toFixed(1).replace(/\.0$/, "");
    return `${s}k`;
  }
  return String(n);
}

export default function OptimizerPage() {
  const {
    playerPool,
    lockedIds,
    bannedIds,
    lockPlayer,
    unlockPlayer,
    banPlayer,
    unbanPlayer,
    generateLineups,
    projectOwnership,
    loading,
    error,
    ownershipSimsApplied,
    lineupStatus,
  } = useSlate();

  const [maxExposure, setMaxExposure] = useState(70);
  const [nLineups, setNLineups] = useState(20);
  const [generated, setGenerated] = useState(false);
  const [ownershipSims, setOwnershipSims] = useState(10000);
  const [ownershipRunActive, setOwnershipRunActive] = useState(false);
  const [ownershipNotice, setOwnershipNotice] = useState<string | null>(null);
  const [showAutoBanList, setShowAutoBanList] = useState(false);
  const [showDtdList, setShowDtdList] = useState(false);

  const lockedArr = Array.from(lockedIds);
  const bannedArr = Array.from(bannedIds);

  const autoBanRows =
    lineupStatus?.report.filter((r) => r.status === "unavailable") ?? [];
  const dtdRows = lineupStatus?.report.filter((r) => r.status === "dtd") ?? [];

  const autoBannedSet = new Set(lineupStatus?.auto_banned_ids ?? []);

  const leveragePlays = [...playerPool]
    .filter((p) => !p.is_pitcher && !autoBannedSet.has(p.dk_id))
    .sort((a, b) => b.leverage - a.leverage)
    .slice(0, 10);

  const popoffs = [...playerPool]
    .filter((p) => !autoBannedSet.has(p.dk_id))
    .sort(
      (a, b) =>
        b.proj_pts_q85 - b.proj_pts - (a.proj_pts_q85 - a.proj_pts)
    )
    .slice(0, 8);

  async function handleOwnership() {
    setOwnershipNotice(null);
    setOwnershipRunActive(true);
    try {
      const n = await projectOwnership(ownershipSims);
      if (n !== undefined) {
        setOwnershipNotice(`Ownership updated — ${n} players repriced`);
      }
    } finally {
      setOwnershipRunActive(false);
    }
  }

  async function handleGenerate() {
    setGenerated(false);
    const ok = await generateLineups(nLineups, maxExposure);
    if (ok) setGenerated(true);
  }

  return (
    <div className="max-w-screen-2xl mx-auto px-6 py-6 relative">
      {loading && (
        <div
          className="absolute inset-0 z-30 flex items-center justify-center rounded-lg"
          style={{ backgroundColor: "rgba(10,14,26,0.65)" }}
        >
          <div className="flex flex-col items-center gap-3">
            <span className="inline-block w-10 h-10 border-2 border-slate-600 border-t-[#00ff88] rounded-full animate-spin" />
            <span className="text-xs uppercase tracking-widest text-slate-400">
              {loading && ownershipRunActive
                ? `Running ${ownershipSims.toLocaleString()} Monte Carlo ownership sims…`
                : "Optimizer running…"}
            </span>
          </div>
        </div>
      )}

      {error && (
        <div
          className="mb-4 px-4 py-3 rounded text-sm font-medium"
          style={{
            backgroundColor: "rgba(239,68,68,0.08)",
            border: "1px solid #ef444440",
            color: "#ef4444",
          }}
        >
          {error}
        </div>
      )}

      {lineupStatus && (autoBanRows.length > 0 || dtdRows.length > 0) && (
        <div className="mb-4 flex flex-col gap-2">
          {autoBanRows.length > 0 && (
            <div
              className="rounded-lg overflow-hidden text-sm"
              style={{
                backgroundColor: "rgba(239,68,68,0.08)",
                border: "1px solid #ef444440",
              }}
            >
              <button
                type="button"
                onClick={() => setShowAutoBanList((v) => !v)}
                className="w-full px-4 py-2.5 flex items-center justify-between text-left font-medium"
                style={{ color: "#ef4444" }}
              >
                <span>
                  {autoBanRows.length} player
                  {autoBanRows.length !== 1 ? "s" : ""} auto-banned (IL / OUT /
                  SUSP)
                </span>
                <span className="text-xs opacity-80">
                  {showAutoBanList ? "▾" : "▸"}
                </span>
              </button>
              {showAutoBanList && (
                <ul
                  className="px-4 pb-3 pt-0 space-y-1 text-xs text-slate-300 border-t"
                  style={{ borderColor: "#ef444420" }}
                >
                  {autoBanRows.map((r) => (
                    <li key={r.dk_id}>
                      <span className="font-medium text-slate-200">{r.name}</span>
                      <span className="text-slate-500"> · {r.team}</span>
                      {r.reason ? (
                        <span className="text-slate-500"> — {r.reason}</span>
                      ) : null}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
          {dtdRows.length > 0 && (
            <div
              className="rounded-lg overflow-hidden text-sm"
              style={{
                backgroundColor: "rgba(234,179,8,0.08)",
                border: "1px solid #eab30840",
              }}
            >
              <button
                type="button"
                onClick={() => setShowDtdList((v) => !v)}
                className="w-full px-4 py-2.5 flex items-center justify-between text-left font-medium"
                style={{ color: "#eab308" }}
              >
                <span>
                  {dtdRows.length} player{dtdRows.length !== 1 ? "s" : ""} DTD
                  (day-to-day — not auto-banned)
                </span>
                <span className="text-xs opacity-80">
                  {showDtdList ? "▾" : "▸"}
                </span>
              </button>
              {showDtdList && (
                <ul
                  className="px-4 pb-3 pt-0 space-y-1 text-xs text-slate-300 border-t"
                  style={{ borderColor: "#eab30830" }}
                >
                  {dtdRows.map((r) => (
                    <li key={r.dk_id}>
                      <span className="font-medium text-slate-200">{r.name}</span>
                      <span className="text-slate-500"> · {r.team}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}

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
            players={playerPool}
            lockedIds={lockedArr}
            bannedIds={bannedArr}
            onLock={lockPlayer}
            onUnlock={unlockPlayer}
            onBan={banPlayer}
            onUnban={unbanPlayer}
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
                max={150}
                step={1}
                value={nLineups}
                onChange={(e) => setNLineups(Number(e.target.value))}
                className="w-full accent-green-400"
              />
            </div>

            {lockedArr.length > 0 && (
              <div
                className="text-xs px-3 py-2 rounded"
                style={{
                  backgroundColor: "rgba(0,255,136,0.06)",
                  border: "1px solid #00ff8820",
                  color: "#00ff88",
                }}
              >
                {lockedArr.length} player{lockedArr.length !== 1 ? "s" : ""}{" "}
                locked · will appear in all {nLineups} lineups
              </div>
            )}
            {bannedArr.length > 0 && (
              <div
                className="text-xs px-3 py-2 rounded"
                style={{
                  backgroundColor: "rgba(239,68,68,0.06)",
                  border: "1px solid #ef444420",
                  color: "#ef4444",
                }}
              >
                {bannedArr.length} player{bannedArr.length !== 1 ? "s" : ""}{" "}
                banned
              </div>
            )}
          </div>

          <div style={{ borderTop: "1px solid #1e2d4a" }} />

          {/* Ownership proxy */}
          <div className="flex flex-col gap-3">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500">
              Ownership
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              {ownershipSimsApplied === null
                ? "Using flat ownership estimates"
                : `Using simulated ownership (${formatSimsK(
                    ownershipSimsApplied
                  )} sims)`}
            </p>
            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Simulation draws</span>
                <span
                  className="font-data font-semibold"
                  style={{ color: "#00ff88" }}
                >
                  {ownershipSims.toLocaleString()}
                </span>
              </div>
              <input
                type="range"
                min={1000}
                max={10000}
                step={1000}
                value={ownershipSims}
                onChange={(e) => setOwnershipSims(Number(e.target.value))}
                disabled={loading || playerPool.length === 0}
                className="w-full accent-green-400"
              />
              <span className="text-[10px] text-slate-600">
                1k fast · 10k accurate (2–3 min)
              </span>
            </div>
            <button
              type="button"
              onClick={() => void handleOwnership()}
              disabled={
                loading || playerPool.length === 0 || ownershipRunActive
              }
              className="w-full py-3 rounded-lg font-semibold text-sm tracking-wide transition-all duration-150 flex items-center justify-center gap-2"
              style={{
                backgroundColor:
                  loading || playerPool.length === 0 ? "#1e2d4a" : "#1e4d6b",
                color:
                  loading || playerPool.length === 0 ? "#475569" : "#e2e8f0",
                cursor:
                  loading || playerPool.length === 0
                    ? "not-allowed"
                    : "pointer",
                border: "1px solid #1e2d4a",
              }}
            >
              Run Ownership Sims (2–3 min)
            </button>
            {ownershipNotice && !loading && (
              <div
                className="text-xs text-center font-data"
                style={{ color: "#00ff88" }}
              >
                {ownershipNotice}
              </div>
            )}
          </div>

          <div style={{ borderTop: "1px solid #1e2d4a" }} />

          {/* Generate button */}
          <button
            onClick={() => void handleGenerate()}
            disabled={loading || playerPool.length === 0}
            className="w-full py-3 rounded-lg font-semibold text-sm tracking-wide transition-all duration-150 flex items-center justify-center gap-2"
            style={{
              backgroundColor:
                loading || playerPool.length === 0 ? "#1e2d4a" : "#00ff88",
              color:
                loading || playerPool.length === 0 ? "#475569" : "#0a0e1a",
              cursor:
                loading || playerPool.length === 0
                  ? "not-allowed"
                  : "pointer",
              boxShadow:
                loading || playerPool.length === 0
                  ? "none"
                  : "0 0 20px rgba(0,255,136,0.3)",
            }}
          >
            {loading ? (
              <>
                <span className="inline-block w-4 h-4 border-2 border-slate-500 border-t-slate-300 rounded-full animate-spin" />
                Generating…
              </>
            ) : (
              `⚡ Generate ${nLineups} Lineups`
            )}
          </button>

          {generated && !loading && (
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
          {playerPool.length === 0 && (
            <p className="text-sm text-slate-500">
              Load a slate and projections on the SLATE tab before running the
              optimizer.
            </p>
          )}

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
                  style={{
                    backgroundColor: "#080c18",
                    borderBottom: "1px solid #1e2d4a",
                  }}
                >
                  <tr>
                    {[
                      "Name",
                      "Team",
                      "Salary",
                      "Median (q50)",
                      "Ceiling (q85)",
                      "Interval",
                      "Own%",
                    ].map((h) => (
                      <th
                        key={h}
                        className="px-4 py-2 text-xs font-semibold uppercase tracking-widest text-slate-500 text-left"
                      >
                        {h}
                      </th>
                    ))}
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
