"use client";

import { useState, useMemo } from "react";
import { Player, LineupStatus } from "@/app/context/SlateContext";

const STATUS_COLORS: Record<LineupStatus, { bg: string; text: string }> = {
  confirmed_starting: { bg: "rgba(0,255,136,0.12)", text: "#00ff88" },
  projected_starting: { bg: "rgba(245,158,11,0.12)", text: "#f59e0b" },
  unknown: { bg: "rgba(100,116,139,0.12)", text: "#94a3b8" },
  scratched: { bg: "rgba(239,68,68,0.12)", text: "#ef4444" },
};

const STATUS_LABELS: Record<LineupStatus, string> = {
  confirmed_starting: "✓ CONF",
  projected_starting: "~ PROJ",
  unknown: "? UNK",
  scratched: "✗ SCRTCH",
};

function leverageColor(lev: number): string {
  if (lev >= 1.3) return "#00ff88";
  if (lev >= 0.9) return "#f59e0b";
  return "#ef4444";
}

type SortDir = "asc" | "desc";
type SortCol =
  | "name"
  | "team"
  | "salary"
  | "avg_pts"
  | "proj_pts"
  | "ownership"
  | "leverage";

const POSITIONS = ["ALL", "P", "C", "1B", "2B", "3B", "SS", "OF"];

interface Props {
  players: Player[];
  onLock?: (p: Player) => void;
  onBan?: (p: Player) => void;
  lockedIds?: string[];
  bannedIds?: string[];
}

export default function PlayerTable({
  players,
  onLock,
  onBan,
  lockedIds = [],
  bannedIds = [],
}: Props) {
  const [search, setSearch] = useState("");
  const [posFilter, setPosFilter] = useState("ALL");
  const [teamFilter, setTeamFilter] = useState("ALL");
  const [salaryMin, setSalaryMin] = useState(2000);
  const [salaryMax, setSalaryMax] = useState(15000);
  const [sort, setSort] = useState<{ col: SortCol; dir: SortDir }>({
    col: "proj_pts",
    dir: "desc",
  });

  const teams = useMemo(
    () => ["ALL", ...Array.from(new Set(players.map((p) => p.team))).sort()],
    [players]
  );

  const filtered = useMemo(() => {
    let out = players;
    if (search)
      out = out.filter((p) =>
        p.name.toLowerCase().includes(search.toLowerCase())
      );
    if (posFilter !== "ALL")
      out = out.filter((p) => p.position_eligibility.includes(posFilter));
    if (teamFilter !== "ALL") out = out.filter((p) => p.team === teamFilter);
    out = out.filter(
      (p) => p.salary >= salaryMin && p.salary <= salaryMax
    );
    out = [...out].sort((a, b) => {
      const v = (p: Player) => p[sort.col as keyof Player] as number | string;
      const av = v(a);
      const bv = v(b);
      if (typeof av === "string")
        return sort.dir === "asc"
          ? (av as string).localeCompare(bv as string)
          : (bv as string).localeCompare(av as string);
      return sort.dir === "asc"
        ? (av as number) - (bv as number)
        : (bv as number) - (av as number);
    });
    return out;
  }, [players, search, posFilter, teamFilter, salaryMin, salaryMax, sort]);

  function toggleSort(col: SortCol) {
    setSort((s) =>
      s.col === col ? { col, dir: s.dir === "asc" ? "desc" : "asc" } : { col, dir: "desc" }
    );
  }

  function SortIcon({ col }: { col: SortCol }) {
    if (sort.col !== col)
      return <span className="text-slate-600 ml-1">↕</span>;
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
    col?: SortCol;
    right?: boolean;
  }) => (
    <th
      className={`px-3 py-2 text-xs font-semibold uppercase tracking-widest text-slate-500 whitespace-nowrap ${right ? "text-right" : "text-left"} ${col ? "cursor-pointer hover:text-slate-300 transition-colors duration-150 select-none" : ""}`}
      onClick={col ? () => toggleSort(col) : undefined}
    >
      {label}
      {col && <SortIcon col={col} />}
    </th>
  );

  return (
    <div className="flex flex-col gap-3">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search player…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-1.5 text-sm rounded bg-[#0f1629] border border-[#1e2d4a] text-slate-200 placeholder-slate-600 focus:outline-none focus:border-[#00ff88] transition-colors w-44"
        />

        {/* Position chips */}
        <div className="flex gap-1 flex-wrap">
          {POSITIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => setPosFilter(pos)}
              className="px-2.5 py-1 text-xs font-semibold rounded transition-all duration-150 font-data"
              style={{
                backgroundColor:
                  posFilter === pos ? "#00ff88" : "#0f1629",
                color: posFilter === pos ? "#0a0e1a" : "#64748b",
                border: `1px solid ${posFilter === pos ? "#00ff88" : "#1e2d4a"}`,
              }}
            >
              {pos}
            </button>
          ))}
        </div>

        {/* Team dropdown */}
        <select
          value={teamFilter}
          onChange={(e) => setTeamFilter(e.target.value)}
          className="px-2 py-1.5 text-xs rounded bg-[#0f1629] border border-[#1e2d4a] text-slate-300 focus:outline-none focus:border-[#00ff88]"
        >
          {teams.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>

        {/* Salary range */}
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>$</span>
          <input
            type="number"
            value={salaryMin}
            onChange={(e) => setSalaryMin(Number(e.target.value))}
            className="w-16 px-2 py-1 rounded bg-[#0f1629] border border-[#1e2d4a] text-slate-300 focus:outline-none font-data"
            step={100}
          />
          <span>–</span>
          <input
            type="number"
            value={salaryMax}
            onChange={(e) => setSalaryMax(Number(e.target.value))}
            className="w-16 px-2 py-1 rounded bg-[#0f1629] border border-[#1e2d4a] text-slate-300 focus:outline-none font-data"
            step={100}
          />
        </div>

        <span className="text-xs text-slate-500 ml-auto">
          {filtered.length} players
        </span>
      </div>

      {/* Table */}
      <div
        className="overflow-auto rounded-lg"
        style={{ border: "1px solid #1e2d4a", maxHeight: 520 }}
      >
        <table className="w-full text-sm" style={{ borderCollapse: "collapse" }}>
          <thead
            style={{
              backgroundColor: "#080c18",
              position: "sticky",
              top: 0,
              zIndex: 1,
              borderBottom: "1px solid #1e2d4a",
            }}
          >
            <tr>
              <TH label="Name" col="name" />
              <TH label="Team" col="team" />
              <TH label="Opp" />
              <TH label="Pos" />
              <TH label="Salary" col="salary" right />
              <TH label="Avg Pts" col="avg_pts" right />
              <TH label="Proj" col="proj_pts" right />
              <TH label="Own%" col="ownership" right />
              <TH label="Lev" col="leverage" right />
              <TH label="Status" />
              {(onLock || onBan) && <TH label="" />}
            </tr>
          </thead>
          <tbody>
            {filtered.map((p) => {
              const isLocked = lockedIds.includes(p.dk_id);
              const isBanned = bannedIds.includes(p.dk_id);
              const rowBg = isBanned
                ? "rgba(239,68,68,0.05)"
                : isLocked
                ? "rgba(0,255,136,0.05)"
                : "transparent";
              return (
                <tr
                  key={p.dk_id}
                  style={{
                    borderBottom: "1px solid #111827",
                    backgroundColor: rowBg,
                    transition: "background 150ms ease",
                  }}
                  className="hover:bg-[#ffffff08]"
                >
                  <td className="px-3 py-2 font-medium text-slate-200 whitespace-nowrap">
                    {isLocked && (
                      <span style={{ color: "#00ff88" }} className="mr-1 text-xs">
                        🔒
                      </span>
                    )}
                    {isBanned && (
                      <span style={{ color: "#ef4444" }} className="mr-1 text-xs">
                        🚫
                      </span>
                    )}
                    {p.name}
                  </td>
                  <td className="px-3 py-2 text-slate-400 font-data text-xs">
                    {p.team}
                  </td>
                  <td className="px-3 py-2 text-slate-500 text-xs">{p.opp}</td>
                  <td className="px-3 py-2">
                    <span
                      className="font-data text-xs px-1.5 py-0.5 rounded"
                      style={{
                        backgroundColor: "#1e2d4a",
                        color: p.is_pitcher ? "#60a5fa" : "#94a3b8",
                      }}
                    >
                      {p.dk_position}
                    </span>
                  </td>
                  <td
                    className="px-3 py-2 text-right font-data font-semibold"
                    style={{ color: "#00ff88" }}
                  >
                    ${p.salary.toLocaleString()}
                  </td>
                  <td className="px-3 py-2 text-right font-data text-slate-400">
                    {p.avg_pts.toFixed(1)}
                  </td>
                  <td
                    className="px-3 py-2 text-right font-data font-semibold text-slate-200"
                  >
                    {p.proj_pts.toFixed(1)}
                  </td>
                  <td className="px-3 py-2 text-right font-data text-slate-400">
                    {p.ownership.toFixed(1)}%
                  </td>
                  <td
                    className="px-3 py-2 text-right font-data font-semibold"
                    style={{ color: leverageColor(p.leverage) }}
                  >
                    {p.leverage.toFixed(2)}
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className="text-xs font-semibold px-2 py-0.5 rounded font-data whitespace-nowrap"
                      style={{
                        backgroundColor: STATUS_COLORS[p.status].bg,
                        color: STATUS_COLORS[p.status].text,
                      }}
                    >
                      {STATUS_LABELS[p.status]}
                    </span>
                  </td>
                  {(onLock || onBan) && (
                    <td className="px-3 py-2">
                      <div className="flex gap-1">
                        {onLock && (
                          <button
                            onClick={() => onLock(p)}
                            className="text-xs px-2 py-0.5 rounded transition-all duration-150"
                            style={{
                              backgroundColor: isLocked
                                ? "rgba(0,255,136,0.15)"
                                : "#0f1629",
                              color: isLocked ? "#00ff88" : "#64748b",
                              border: `1px solid ${isLocked ? "#00ff88" : "#1e2d4a"}`,
                            }}
                          >
                            {isLocked ? "LOCKED" : "LOCK"}
                          </button>
                        )}
                        {onBan && (
                          <button
                            onClick={() => onBan(p)}
                            className="text-xs px-2 py-0.5 rounded transition-all duration-150"
                            style={{
                              backgroundColor: isBanned
                                ? "rgba(239,68,68,0.15)"
                                : "#0f1629",
                              color: isBanned ? "#ef4444" : "#64748b",
                              border: `1px solid ${isBanned ? "#ef4444" : "#1e2d4a"}`,
                            }}
                          >
                            {isBanned ? "BANNED" : "BAN"}
                          </button>
                        )}
                      </div>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>

        {filtered.length === 0 && (
          <div className="py-12 text-center text-slate-600 text-sm">
            No players match the current filters.
          </div>
        )}
      </div>
    </div>
  );
}
