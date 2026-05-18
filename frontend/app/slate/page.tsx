"use client";

import { useState, useRef, useEffect } from "react";
import { useSlate } from "@/app/context/SlateContext";
import type { DraftGroupSlateRow } from "@/app/context/SlateContext";
import PlayerTable from "@/components/PlayerTable";
import StatBadge from "@/components/StatBadge";

export default function SlatePage() {
  const {
    slateInfo,
    playerPool,
    projections,
    uploadCSV,
    fetchTodaysSlates,
    selectSlateByDg,
    loading,
    error,
    lockedIds,
    bannedIds,
    lockPlayer,
    unlockPlayer,
    banPlayer,
    unbanPlayer,
  } = useSlate();
  const [dragOver, setDragOver] = useState(false);
  const [fileBanner, setFileBanner] = useState<{
    ok: boolean;
    text: string;
  } | null>(null);
  const [todaysSlates, setTodaysSlates] = useState<DraftGroupSlateRow[]>([]);
  const [slatesLoaded, setSlatesLoaded] = useState(false);
  const [showManualUpload, setShowManualUpload] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const rows = await fetchTodaysSlates();
      if (!cancelled) {
        setTodaysSlates(rows);
        setSlatesLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [fetchTodaysSlates]);

  async function handleFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setFileBanner({
        ok: false,
        text: "Invalid file type. Please upload a DraftKings salary CSV.",
      });
      return;
    }
    setFileBanner(null);
    const ok = await uploadCSV(file);
    if (ok) {
      setShowManualUpload(false);
      setFileBanner({
        ok: true,
        text: "Salary file ingested and projections refreshed.",
      });
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) void handleFile(file);
  }

  const lockedArr = Array.from(lockedIds);
  const bannedArr = Array.from(bannedIds);

  const handleLock = (p: { dk_id: string }) => {
    if (lockedIds.has(p.dk_id)) unlockPlayer(p.dk_id);
    else lockPlayer(p.dk_id);
  };
  const handleBan = (p: { dk_id: string }) => {
    if (bannedIds.has(p.dk_id)) unbanPlayer(p.dk_id);
    else banPlayer(p.dk_id);
  };

  async function handleSelectSlate(dg: number) {
    setFileBanner(null);
    const ok = await selectSlateByDg(dg);
    if (ok) {
      setShowManualUpload(false);
      setFileBanner({
        ok: true,
        text: `Slate dg=${dg} loaded — projections refreshed.`,
      });
    }
  }

  const slateFromApi = Boolean(slateInfo?.draft_group_id);
  const hideUploadZone = slateFromApi && !showManualUpload;

  const emptySlate = !slateInfo;

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

      {emptySlate && !loading && slatesLoaded && todaysSlates.length === 0 && (
        <div
          className="rounded-lg px-6 py-10 text-center"
          style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
        >
          <div className="text-3xl mb-3">📋</div>
          <p className="text-slate-300 font-semibold mb-1">
            No automated slates found for today
          </p>
          <p className="text-xs text-slate-500">
            Use manual CSV upload below, or wait for the morning automation job.
          </p>
        </div>
      )}

      {emptySlate && !loading && (!slatesLoaded || todaysSlates.length > 0) && (
        <div
          className="rounded-lg px-6 py-10 text-center"
          style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
        >
          <div className="text-3xl mb-3">📋</div>
          <p className="text-slate-300 font-semibold mb-1">
            Select a slate or upload a DK salary CSV
          </p>
          <p className="text-xs text-slate-500">
            Choose a draft group above, or drop a file in the upload zone.
          </p>
        </div>
      )}

      {/* Today's DK slates (draft groups) */}
      <section className="mb-2">
        <h2
          className="text-xs font-semibold uppercase tracking-widest mb-3"
          style={{ color: "#00ff88" }}
        >
          00 — Today&apos;s slates
        </h2>
        {!slatesLoaded ? (
          <p className="text-xs text-slate-500">Loading slates…</p>
        ) : todaysSlates.length === 0 ? (
          <p className="text-xs text-slate-500">
            No slates from the API. You can still upload a salary CSV manually.
          </p>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {todaysSlates.map((s) => (
              <button
                key={s.dg}
                type="button"
                disabled={loading}
                onClick={() => void handleSelectSlate(s.dg)}
                className="text-left rounded-lg p-4 transition-all duration-150 border"
                style={{
                  backgroundColor:
                    slateInfo?.draft_group_id === s.dg ? "#0a2818" : "#0f1629",
                  borderColor:
                    slateInfo?.draft_group_id === s.dg ? "#00ff8860" : "#1e2d4a",
                  cursor: loading ? "not-allowed" : "pointer",
                  opacity: loading ? 0.6 : 1,
                }}
              >
                <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
                  Draft group
                </div>
                <div
                  className="font-data text-lg font-semibold mb-2"
                  style={{ color: "#00ff88" }}
                >
                  dg={s.dg}
                </div>
                <div className="text-xs text-slate-300 mb-1 line-clamp-2">
                  {s.name}
                </div>
                <div className="text-xs text-slate-500 mb-1">
                  Lock {s.lock_time_et} · {s.contest_count} contests
                </div>
                <div className="text-xs text-slate-500 mb-1">
                  {s.max_entries.toLocaleString()} max entries
                </div>
                <div className="text-xs text-slate-500 mb-1">
                  Top prize: ${(s.max_prize_pool / 1000).toFixed(0)}K
                </div>
                <div className="text-xs text-slate-500 mb-2">
                  Current fill: {s.total_current_entries.toLocaleString()} entered
                </div>
                <div className="text-xs font-medium">
                  {s.csv_exists ? (
                    <span style={{ color: "#00ff88" }}>✅ CSV ready</span>
                  ) : (
                    <span style={{ color: "#f59e0b" }}>❌ CSV not on disk</span>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}
      </section>

      {slateFromApi && (
        <div
          className="mb-4 rounded-lg px-4 py-3 text-sm flex flex-wrap items-center justify-between gap-2"
          style={{
            backgroundColor: "rgba(0,255,136,0.08)",
            border: "1px solid #00ff8840",
            color: "#86efac",
          }}
        >
          <span>
            <strong>Selected slate</strong> · dg={slateInfo!.draft_group_id} · Lock{" "}
            {slateInfo!.lock_time_et ?? "—"} ·{" "}
            {slateInfo!.csv_path ? (
              <span className="text-slate-400 font-mono text-xs">
                {slateInfo!.csv_path}
              </span>
            ) : null}
          </span>
          {!showManualUpload && (
            <button
              type="button"
              className="text-xs underline text-slate-400 hover:text-slate-200"
              onClick={() => setShowManualUpload(true)}
            >
              Upload CSV manually instead
            </button>
          )}
        </div>
      )}

      {/* Upload Zone */}
      {!hideUploadZone && (
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
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleFile(f);
            }}
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
              <span className="text-slate-200">
                {slateInfo.file_name ?? "—"}
              </span>
              <span className="text-slate-500 mx-2">·</span>
              <span className="text-slate-400">SHA256: </span>
              <span style={{ color: "#00ff88" }}>
                {slateInfo.sha256.slice(0, 16)}…
              </span>
            </div>
          )}
        </div>

        {fileBanner && !error && (
          <div
            className="mt-3 px-4 py-3 rounded text-sm font-medium"
            style={{
              backgroundColor: fileBanner.ok
                ? "rgba(0,255,136,0.08)"
                : "rgba(239,68,68,0.08)",
              border: `1px solid ${fileBanner.ok ? "#00ff8840" : "#ef444440"}`,
              color: fileBanner.ok ? "#00ff88" : "#ef4444",
            }}
          >
            {fileBanner.ok ? "✓ " : "✗ "}
            {fileBanner.text}
          </div>
        )}
      </section>
      )}

      {!emptySlate && (
        <>
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
                {
                  value: slateInfo!.player_count.toLocaleString(),
                  label: "Players",
                },
                {
                  value: slateInfo!.pitcher_count.toString(),
                  label: "Pitchers",
                  color: "blue" as const,
                },
                {
                  value: slateInfo!.hitter_count.toString(),
                  label: "Hitters",
                },
                {
                  value: slateInfo!.game_count.toString(),
                  label: "Games",
                },
                {
                  value: slateInfo!.team_count.toString(),
                  label: "Teams",
                },
                {
                  value:
                    slateInfo!.lock_time_et ??
                    (slateInfo!.lock_time &&
                    !Number.isNaN(new Date(slateInfo!.lock_time).getTime())
                      ? new Date(slateInfo!.lock_time).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })
                      : "--:--"),
                  label: "Lock Time",
                  color: "amber" as const,
                },
              ].map(({ value, label, color }) => (
                <div
                  key={label}
                  className="rounded-lg px-4 py-4"
                  style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
                >
                  <StatBadge
                    value={value}
                    label={label}
                    color={color ?? "green"}
                    size="lg"
                  />
                </div>
              ))}
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {slateInfo!.games.map((g) => (
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
            {projections.length === 0 ? (
              <p className="text-sm text-slate-500">
                No projection rows yet. Upload again or wait for the projection
                step to finish.
              </p>
            ) : (
              <PlayerTable
                players={playerPool}
                onLock={handleLock}
                onBan={handleBan}
                lockedIds={lockedArr}
                bannedIds={bannedArr}
              />
            )}
          </section>
        </>
      )}
    </div>
  );
}
