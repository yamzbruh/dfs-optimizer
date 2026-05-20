"use client";

import { useMemo, useState } from "react";
import { useSlate, type Player } from "@/app/context/SlateContext";

type PosFilter =
  | "ALL"
  | "HITTERS"
  | "SP"
  | "RP"
  | "C"
  | "1B"
  | "2B"
  | "3B"
  | "SS"
  | "OF";

type SortCol =
  | "pos"
  | "name"
  | "salary"
  | "team"
  | "opp"
  | "q15"
  | "q50"
  | "q85"
  | "own"
  | "leverage";

type SortDir = "asc" | "desc";

const POS_TABS: PosFilter[] = [
  "ALL",
  "HITTERS",
  "SP",
  "RP",
  "C",
  "1B",
  "2B",
  "3B",
  "SS",
  "OF",
];

function matchesPosFilter(p: Player, filter: PosFilter): boolean {
  if (filter === "ALL") return true;
  if (filter === "HITTERS") return !p.is_pitcher;
  if (filter === "SP") return (p.dk_position || "").toUpperCase() === "SP";
  if (filter === "RP") return (p.dk_position || "").toUpperCase() === "RP";
  return p.position_eligibility.includes(filter);
}

function q50ColorClass(q50: number, sortedDesc: number[]): string {
  const n = sortedDesc.length;
  if (n === 0) return "text-slate-400";
  const rank = sortedDesc.findIndex((v) => v <= q50);
  const idx = rank === -1 ? n - 1 : rank;
  const pct = idx / n;
  if (pct < 0.2) return "text-[#00ff88] font-semibold";
  if (pct < 0.6) return "text-[#f59e0b]";
  return "text-slate-400";
}

export default function ProjectionsPage() {
  const {
    playerPool,
    projections,
    lockedIds,
    bannedIds,
    lineupStatus,
    lockPlayer,
    unlockPlayer,
    banPlayer,
    unbanPlayer,
    loading,
    error,
  } = useSlate();

  const [search, setSearch] = useState("");
  const [posFilter, setPosFilter] = useState<PosFilter>("ALL");
  const [sort, setSort] = useState<{ col: SortCol; dir: SortDir }>({
    col: "q50",
    dir: "desc",
  });

  const autoBannedSet = useMemo(
    () => new Set(lineupStatus?.auto_banned_ids ?? []),
    [lineupStatus]
  );

  const q50SortedDesc = useMemo(
    () =>
      [...playerPool.map((p) => p.proj_pts)].sort((a, b) => b - a),
    [playerPool]
  );

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    let out = playerPool.filter((p) => matchesPosFilter(p, posFilter));
    if (q) {
      out = out.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.team.toLowerCase().includes(q)
      );
    }

    const getVal = (p: Player, col: SortCol): number | string => {
      switch (col) {
        case "pos":
          return p.dk_position;
        case "name":
          return p.name;
        case "salary":
          return p.salary;
        case "team":
          return p.team;
        case "opp":
          return p.opp;
        case "q15":
          return p.proj_pts_q15;
        case "q50":
          return p.proj_pts;
        case "q85":
          return p.proj_pts_q85;
        case "own":
          return p.ownership;
        case "leverage":
          return p.leverage;
        default:
          return 0;
      }
    };

    out = [...out].sort((a, b) => {
      const av = getVal(a, sort.col);
      const bv = getVal(b, sort.col);
      if (typeof av === "string" && typeof bv === "string") {
        return sort.dir === "asc"
          ? av.localeCompare(bv)
          : bv.localeCompare(av);
      }
      return sort.dir === "asc"
        ? (av as number) - (bv as number)
        : (bv as number) - (av as number);
    });

    return out;
  }, [playerPool, search, posFilter, sort]);

  function toggleSort(col: SortCol) {
    setSort((s) =>
      s.col === col
        ? { col, dir: s.dir === "asc" ? "desc" : "asc" }
        : { col, dir: col === "name" || col === "team" || col === "opp" || col === "pos" ? "asc" : "desc" }
    );
  }

  function SortIcon({ col }: { col: SortCol }) {
    if (sort.col !== col) return <span className="text-slate-600 ml-1">↕</span>;
    return (
      <span style={{ color: "#00ff88" }} className="ml-1">
        {sort.dir === "asc" ? "↑" : "↓"}
      </span>
    );
  }

  const TH = ({
    label,
    col,
    right,
  }: {
    label: string;
    col: SortCol;
    right?: boolean;
  }) => (
    <th
      className={`px-3 py-2.5 text-xs font-semibold uppercase tracking-widest text-slate-500 whitespace-nowrap cursor-pointer hover:text-slate-300 select-none ${
        right ? "text-right" : "text-left"
      }`}
      onClick={() => toggleSort(col)}
    >
      {label}
      <SortIcon col={col} />
    </th>
  );

  if (projections.length === 0) {
    return (
      <div className="max-w-screen-2xl mx-auto px-6 py-16 text-center">
        <div className="text-4xl mb-4">📊</div>
        <h2 className="text-xl font-semibold text-slate-400 mb-2">
          No projections loaded
        </h2>
        <p className="text-slate-600 text-sm">
          Load a slate on the SLATE tab to build projections, then return here.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-screen-2xl mx-auto px-6 py-6 flex flex-col gap-4 relative">
      {loading && (
        <div
          className="absolute inset-0 z-30 flex items-center justify-center rounded-lg"
          style={{ backgroundColor: "rgba(10,14,26,0.65)" }}
        >
          <span className="inline-block w-10 h-10 border-2 border-slate-600 border-t-[#00ff88] rounded-full animate-spin" />
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

      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1
          className="text-xs font-semibold uppercase tracking-widest"
          style={{ color: "#00ff88" }}
        >
          Projections
        </h1>
        <span className="text-xs text-slate-500 font-data">
          {filtered.length} / {playerPool.length} players
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search name or team…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-1.5 text-sm rounded bg-[#0f1629] border border-[#1e2d4a] text-slate-200 placeholder-slate-600 focus:outline-none focus:border-[#00ff88] w-52"
        />
        <div className="flex gap-1 flex-wrap">
          {POS_TABS.map((pos) => (
            <button
              key={pos}
              type="button"
              onClick={() => setPosFilter(pos)}
              className="px-2.5 py-1 text-xs font-semibold rounded font-data transition-all"
              style={{
                backgroundColor: posFilter === pos ? "#00ff88" : "#0f1629",
                color: posFilter === pos ? "#0a0e1a" : "#64748b",
                border: `1px solid ${posFilter === pos ? "#00ff88" : "#1e2d4a"}`,
              }}
            >
              {pos}
            </button>
          ))}
        </div>
      </div>

      <div
        className="overflow-auto rounded-lg"
        style={{ border: "1px solid #1e2d4a", maxHeight: "calc(100vh - 220px)" }}
      >
        <table className="w-full text-sm" style={{ borderCollapse: "collapse" }}>
          <thead
            style={{
              backgroundColor: "#080c18",
              position: "sticky",
              top: 0,
              zIndex: 2,
              borderBottom: "1px solid #1e2d4a",
            }}
          >
            <tr>
              <TH label="POS" col="pos" />
              <TH label="NAME" col="name" />
              <TH label="SALARY" col="salary" right />
              <TH label="TEAM" col="team" />
              <TH label="OPP" col="opp" />
              <TH label="Q15" col="q15" right />
              <TH label="Q50" col="q50" right />
              <TH label="Q85" col="q85" right />
              <TH label="OWN%" col="own" right />
              <TH label="LEVERAGE" col="leverage" right />
              <th className="px-3 py-2.5 text-xs font-semibold uppercase tracking-widest text-slate-500 text-center">
                LOCK
              </th>
              <th className="px-3 py-2.5 text-xs font-semibold uppercase tracking-widest text-slate-500 text-center">
                BAN
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((p) => {
              const isLocked = lockedIds.has(p.dk_id);
              const isUserBanned = bannedIds.has(p.dk_id);
              const isAutoBanned = autoBannedSet.has(p.dk_id);
              const isBanned = isUserBanned || isAutoBanned;
              const isSp = (p.dk_position || "").toUpperCase() === "SP";

              const rowBg = isBanned
                ? "rgba(239,68,68,0.06)"
                : isLocked
                  ? "rgba(0,255,136,0.06)"
                  : "transparent";

              return (
                <tr
                  key={p.dk_id}
                  style={{
                    borderBottom: "1px solid #111827",
                    backgroundColor: rowBg,
                  }}
                  className="hover:bg-[#ffffff06]"
                >
                  <td className="px-3 py-2 font-data text-xs text-slate-400">
                    {p.dk_position}
                    {isSp && (
                      <span className="ml-1.5">
                        {p.proj_pts === 0 ? (
                          <span
                            className="text-[10px] px-1 py-0.5 rounded"
                            style={{
                              backgroundColor: "rgba(245,158,11,0.15)",
                              color: "#f59e0b",
                            }}
                            title="Not confirmed starter"
                          >
                            ⚠ unconfirmed
                          </span>
                        ) : (
                          <span
                            className="text-[10px] px-1 py-0.5 rounded"
                            style={{
                              backgroundColor: "rgba(0,255,136,0.12)",
                              color: "#00ff88",
                            }}
                            title="Confirmed starter"
                          >
                            ✓ confirmed
                          </span>
                        )}
                      </span>
                    )}
                  </td>
                  <td
                    className={`px-3 py-2 font-medium whitespace-nowrap ${
                      isBanned
                        ? "text-red-400 line-through"
                        : isLocked
                          ? "text-[#00ff88]"
                          : "text-slate-200"
                    }`}
                  >
                    {p.name}
                  </td>
                  <td
                    className={`px-3 py-2 font-data text-right ${
                      isBanned ? "text-red-400/70 line-through" : "text-[#00ff88]"
                    }`}
                  >
                    ${p.salary.toLocaleString()}
                  </td>
                  <td
                    className={`px-3 py-2 font-data text-xs ${
                      isBanned ? "text-red-400/70 line-through" : "text-slate-400"
                    }`}
                  >
                    {p.team}
                  </td>
                  <td
                    className={`px-3 py-2 font-data text-xs ${
                      isBanned ? "text-red-400/70 line-through" : "text-slate-500"
                    }`}
                  >
                    {p.opp || "—"}
                  </td>
                  <td
                    className={`px-3 py-2 font-data text-right ${
                      isBanned ? "text-red-400/70 line-through" : "text-slate-300"
                    }`}
                  >
                    {p.proj_pts_q15.toFixed(1)}
                  </td>
                  <td
                    className={`px-3 py-2 font-data text-right ${
                      isBanned
                        ? "text-red-400/70 line-through"
                        : q50ColorClass(p.proj_pts, q50SortedDesc)
                    }`}
                  >
                    {p.proj_pts.toFixed(1)}
                  </td>
                  <td
                    className={`px-3 py-2 font-data text-right ${
                      isBanned ? "text-red-400/70 line-through" : "text-slate-300"
                    }`}
                  >
                    {p.proj_pts_q85.toFixed(1)}
                  </td>
                  <td
                    className={`px-3 py-2 font-data text-right ${
                      isBanned ? "text-red-400/70 line-through" : "text-slate-400"
                    }`}
                  >
                    {p.ownership.toFixed(1)}%
                  </td>
                  <td
                    className={`px-3 py-2 font-data text-right ${
                      isBanned ? "text-red-400/70 line-through" : "text-slate-300"
                    }`}
                  >
                    {p.leverage.toFixed(2)}
                  </td>
                  <td className="px-2 py-2 text-center">
                    <button
                      type="button"
                      onClick={() =>
                        isLocked ? unlockPlayer(p.dk_id) : lockPlayer(p.dk_id)
                      }
                      className="px-2 py-0.5 text-[10px] font-semibold rounded uppercase tracking-wide transition-colors"
                      style={{
                        backgroundColor: isLocked
                          ? "rgba(0,255,136,0.2)"
                          : "#0f1629",
                        color: isLocked ? "#00ff88" : "#64748b",
                        border: `1px solid ${isLocked ? "#00ff8840" : "#1e2d4a"}`,
                      }}
                    >
                      {isLocked ? "UNLOCK" : "LOCK"}
                    </button>
                  </td>
                  <td className="px-2 py-2 text-center">
                    <button
                      type="button"
                      onClick={() =>
                        isUserBanned
                          ? unbanPlayer(p.dk_id)
                          : banPlayer(p.dk_id)
                      }
                      disabled={isAutoBanned && !isUserBanned}
                      title={
                        isAutoBanned && !isUserBanned
                          ? "Auto-banned (IL/OUT) — cannot unban"
                          : undefined
                      }
                      className="px-2 py-0.5 text-[10px] font-semibold rounded uppercase tracking-wide transition-colors"
                      style={{
                        backgroundColor: isBanned
                          ? "rgba(239,68,68,0.2)"
                          : "#0f1629",
                        color: isBanned ? "#ef4444" : "#64748b",
                        border: `1px solid ${isBanned ? "#ef444440" : "#1e2d4a"}`,
                        cursor:
                          isAutoBanned && !isUserBanned
                            ? "not-allowed"
                            : "pointer",
                        opacity: isAutoBanned && !isUserBanned ? 0.6 : 1,
                      }}
                    >
                      {isBanned ? "UNBAN" : "BAN"}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[10px] text-slate-600">
        Q50 colors: top 20% green · middle 40% amber · bottom 40% gray (slate-wide)
      </p>
    </div>
  );
}
