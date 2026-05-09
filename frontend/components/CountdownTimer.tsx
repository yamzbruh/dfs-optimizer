"use client";

import { useEffect, useState } from "react";

interface CountdownTimerProps {
  lockTime: string; // ISO string
  compact?: boolean;
}

function formatCountdown(ms: number): string {
  if (ms <= 0) return "LOCKED";
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}h ${m.toString().padStart(2, "0")}m ${s.toString().padStart(2, "0")}s`;
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

function getColor(ms: number): string {
  if (ms <= 0) return "#ef4444";
  if (ms < 30 * 60 * 1000) return "#ef4444";
  if (ms < 2 * 60 * 60 * 1000) return "#f59e0b";
  return "#00ff88";
}

export default function CountdownTimer({ lockTime, compact = false }: CountdownTimerProps) {
  const [remaining, setRemaining] = useState<number>(0);

  useEffect(() => {
    const target = new Date(lockTime).getTime();
    const tick = () => setRemaining(Math.max(0, target - Date.now()));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [lockTime]);

  const color = getColor(remaining);
  const label = formatCountdown(remaining);

  if (compact) {
    return (
      <span className="font-data text-sm font-semibold" style={{ color }}>
        {label}
      </span>
    );
  }

  return (
    <div className="flex flex-col items-end gap-0">
      <span className="text-xs uppercase tracking-widest text-slate-500">
        Locks in
      </span>
      <span
        className="font-data text-lg font-bold leading-tight"
        style={{ color }}
      >
        {label}
      </span>
    </div>
  );
}
