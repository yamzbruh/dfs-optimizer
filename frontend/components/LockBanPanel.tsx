"use client";

import { useState, useRef, useEffect } from "react";
import { Player } from "@/app/context/SlateContext";

interface Props {
  players: Player[];
  lockedIds: string[];
  bannedIds: string[];
  onLock: (id: string) => void;
  onUnlock: (id: string) => void;
  onBan: (id: string) => void;
  onUnban: (id: string) => void;
}

function SearchInput({
  players,
  excludeIds,
  placeholder,
  onSelect,
  accentColor,
}: {
  players: Player[];
  excludeIds: string[];
  placeholder: string;
  onSelect: (id: string) => void;
  accentColor: string;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const results = query
    ? players
        .filter(
          (p) =>
            !excludeIds.includes(p.dk_id) &&
            p.name.toLowerCase().includes(query.toLowerCase())
        )
        .slice(0, 8)
    : [];

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div className="relative" ref={containerRef}>
      <input
        type="text"
        placeholder={placeholder}
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        className="w-full px-3 py-2 text-sm rounded bg-[#080c18] text-slate-200 placeholder-slate-600 focus:outline-none transition-colors"
        style={{ border: `1px solid ${open && results.length ? accentColor + "60" : "#1e2d4a"}` }}
      />
      {open && results.length > 0 && (
        <div
          className="absolute top-full left-0 right-0 rounded-b overflow-hidden z-20"
          style={{
            backgroundColor: "#0a0e1a",
            border: `1px solid ${accentColor}40`,
            borderTop: "none",
          }}
        >
          {results.map((p) => (
            <div
              key={p.dk_id}
              className="flex items-center justify-between px-3 py-2 cursor-pointer transition-colors duration-100 hover:bg-[#ffffff08]"
              onMouseDown={(e) => {
                e.preventDefault();
                onSelect(p.dk_id);
                setQuery("");
                setOpen(false);
              }}
            >
              <div>
                <span className="text-sm text-slate-200">{p.name}</span>
                <span className="text-xs text-slate-500 ml-2">
                  {p.team} · {p.dk_position}
                </span>
              </div>
              <span
                className="font-data text-xs font-semibold"
                style={{ color: "#00ff88" }}
              >
                ${p.salary.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PanelList({
  ids,
  players,
  onRemove,
  label,
  accentColor,
}: {
  ids: string[];
  players: Player[];
  onRemove: (id: string) => void;
  label: string;
  accentColor: string;
}) {
  const items = ids
    .map((id) => players.find((p) => p.dk_id === id))
    .filter(Boolean) as Player[];

  if (items.length === 0) {
    return (
      <div
        className="text-xs text-slate-600 px-3 py-3 rounded"
        style={{ border: "1px dashed #1e2d4a" }}
      >
        No {label.toLowerCase()} players
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      {items.map((p) => (
        <div
          key={p.dk_id}
          className="flex items-center justify-between px-3 py-2 rounded"
          style={{
            backgroundColor: `${accentColor}10`,
            border: `1px solid ${accentColor}30`,
          }}
        >
          <div>
            <span className="text-sm font-medium" style={{ color: accentColor }}>
              {p.name}
            </span>
            <span className="text-xs text-slate-500 ml-2">
              {p.dk_position} · ${p.salary.toLocaleString()}
            </span>
          </div>
          <button
            onClick={() => onRemove(p.dk_id)}
            className="text-slate-500 hover:text-slate-200 text-lg leading-none transition-colors ml-2"
            title="Remove"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

export default function LockBanPanel({
  players,
  lockedIds,
  bannedIds,
  onLock,
  onUnlock,
  onBan,
  onUnban,
}: Props) {
  return (
    <div className="flex flex-col gap-6">
      {/* Lock */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "#00ff88" }}>
            🔒 Lock Players
          </span>
          <span
            className="font-data text-xs px-1.5 rounded"
            style={{ backgroundColor: "#00ff8820", color: "#00ff88" }}
          >
            {lockedIds.length}
          </span>
        </div>
        <SearchInput
          players={players}
          excludeIds={lockedIds}
          placeholder="Search to lock a player…"
          onSelect={onLock}
          accentColor="#00ff88"
        />
        <PanelList
          ids={lockedIds}
          players={players}
          onRemove={onUnlock}
          label="Locked"
          accentColor="#00ff88"
        />
      </div>

      <div style={{ borderTop: "1px solid #1e2d4a" }} />

      {/* Ban */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "#ef4444" }}>
            🚫 Ban Players
          </span>
          <span
            className="font-data text-xs px-1.5 rounded"
            style={{ backgroundColor: "#ef444420", color: "#ef4444" }}
          >
            {bannedIds.length}
          </span>
        </div>
        <SearchInput
          players={players}
          excludeIds={bannedIds}
          placeholder="Search to ban a player…"
          onSelect={onBan}
          accentColor="#ef4444"
        />
        <PanelList
          ids={bannedIds}
          players={players}
          onRemove={onUnban}
          label="Banned"
          accentColor="#ef4444"
        />
      </div>
    </div>
  );
}
