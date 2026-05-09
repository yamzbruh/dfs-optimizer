"use client";

import { useState, useRef } from "react";
import { useSlate, MOCK_SLATE_INFO } from "@/app/context/SlateContext";
import PlayerTable from "@/components/PlayerTable";
import StatBadge from "@/components/StatBadge";

export default function SlatePage() {
  const { players, slateInfo, setSlateInfo, lockedIds, bannedIds, addLock, removeLock, addBan, removeBan } = useSlate();
  const [dragOver, setDragOver] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "success" | "error">("idle");
  const [uploadMsg, setUploadMsg] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFile(file: File) {
    if (!file.name.endsWith(".csv")) {
      setUploadStatus("error");
      setUploadMsg("Invalid file type. Please upload a DraftKings salary CSV.");
      return;
    }
    // Simulate parse — in production this calls the backend
    setSlateInfo({
      ...MOCK_SLATE_INFO,
      file_name: file.name,
      file_hash: Math.random().toString(36).slice(2) + "a0b1c2d3e4f5",
    });
    setUploadStatus("success");
    setUploadMsg(`Parsed ${players.length} players from ${MOCK_SLATE_INFO.game_count} games`);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  const pitchers = players.filter((p) => p.is_pitcher);
  const hitters = players.filter((p) => !p.is_pitcher);
  const games = Array.from(new Set(players.map((p) => p.game_info)));

  const handleLock = (p: { dk_id: string }) => {
    if (lockedIds.includes(p.dk_id)) removeLock(p.dk_id);
    else addLock(p.dk_id);
  };
  const handleBan = (p: { dk_id: string }) => {
    if (bannedIds.includes(p.dk_id)) removeBan(p.dk_id);
    else addBan(p.dk_id);
  };

  return (
    <div className="max-w-screen-2xl mx-auto px-6 py-6 flex flex-col gap-6">

      {/* Upload Zone */}
      <section>
        <h2
          className="text-xs font-semibold uppercase tracking-widest mb-4"
          style={{ color: "#00ff88" }}
        >
          01 — Upload Salary CSV
        </h2>

        <div
          className="relative flex flex-col items-center justify-center gap-3 rounded-lg p-10 transition-all duration-150 cursor-pointer"
          style={{
            border: `2px dashed ${dragOver ? "#00ff88" : "#1e2d4a"}`,
            backgroundColor: dragOver ? "rgba(0,255,136,0.04)" : "#0f1629",
          }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
          />
          <div
            className="text-4xl"
            style={{ filter: "grayscale(0.4)" }}
          >
            📂
          </div>
          <div className="text-center">
            <p className="font-semibold text-slate-300">
              Drop DraftKings salary CSV here
            </p>
            <p className="text-xs text-slate-500 mt-1">
              or click to browse · accepts .csv
            </p>
          </div>

          {slateInfo && (
            <div
              className="mt-2 px-4 py-2 rounded text-xs font-data"
              style={{ backgroundColor: "#080c18", border: "1px solid #1e2d4a" }}
            >
              <span className="text-slate-400">File: </span>
              <span className="text-slate-200">{slateInfo.file_name}</span>
              <span className="text-slate-500 mx-2">·</span>
              <span className="text-slate-400">SHA256: </span>
              <span style={{ color: "#00ff88" }}>{slateInfo.file_hash.slice(0, 16)}…</span>
            </div>
          )}
        </div>

        {uploadStatus !== "idle" && (
          <div
            className="mt-3 px-4 py-3 rounded text-sm font-medium"
            style={{
              backgroundColor:
                uploadStatus === "success"
                  ? "rgba(0,255,136,0.08)"
                  : "rgba(239,68,68,0.08)",
              border: `1px solid ${uploadStatus === "success" ? "#00ff8840" : "#ef444440"}`,
              color: uploadStatus === "success" ? "#00ff88" : "#ef4444",
            }}
          >
            {uploadStatus === "success" ? "✓ " : "✗ "}
            {uploadMsg}
          </div>
        )}
      </section>

      {/* Slate Stats */}
      <section>
        <h2
          className="text-xs font-semibold uppercase tracking-widest mb-4"
          style={{ color: "#00ff88" }}
        >
          02 — Slate Overview
        </h2>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-4">
          {[
            { value: players.length.toLocaleString(), label: "Players" },
            { value: pitchers.length.toString(), label: "Pitchers", color: "blue" as const },
            { value: hitters.length.toString(), label: "Hitters" },
            { value: games.length.toString(), label: "Games" },
            { value: Array.from(new Set(players.map((p) => p.team))).length.toString(), label: "Teams" },
            { value: slateInfo ? new Date(slateInfo.lock_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "--:--", label: "Lock Time", color: "amber" as const },
          ].map(({ value, label, color }) => (
            <div
              key={label}
              className="rounded-lg px-4 py-4"
              style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
            >
              <StatBadge value={value} label={label} color={color ?? "green"} size="lg" />
            </div>
          ))}
        </div>

        {/* Games list */}
        <div className="mt-4 flex flex-wrap gap-2">
          {games.map((g) => (
            <span
              key={g}
              className="font-data text-xs px-3 py-1 rounded"
              style={{
                backgroundColor: "#0f1629",
                border: "1px solid #1e2d4a",
                color: "#94a3b8",
              }}
            >
              {g}
            </span>
          ))}
        </div>
      </section>

      {/* Player Pool Table */}
      <section>
        <h2
          className="text-xs font-semibold uppercase tracking-widest mb-4"
          style={{ color: "#00ff88" }}
        >
          03 — Player Pool
        </h2>
        <PlayerTable
          players={players}
          onLock={handleLock}
          onBan={handleBan}
          lockedIds={lockedIds}
          bannedIds={bannedIds}
        />
      </section>
    </div>
  );
}
