"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSlate } from "@/app/context/SlateContext";
import CountdownTimer from "./CountdownTimer";

const LINKS = [
  { href: "/slate", label: "SLATE" },
  { href: "/optimizer", label: "OPTIMIZER" },
  { href: "/lineups", label: "LINEUPS" },
  { href: "/model", label: "MODEL" },
];

export default function Nav() {
  const pathname = usePathname();
  const { slateInfo, lineups } = useSlate();

  const validCount = lineups.filter((l) => l.is_valid).length;
  const denom = lineups.length > 0 ? lineups.length : 20;

  return (
    <header
      className="sticky top-0 z-50 flex items-center justify-between px-6 h-14"
      style={{
        backgroundColor: "#0a0e1a",
        borderBottom: "1px solid #1e2d4a",
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-6">
        <span
          className="font-data text-lg font-bold tracking-wider"
          style={{ color: "#00ff88" }}
        >
          DFS WAR ROOM
        </span>

        {/* Nav links */}
        <nav className="flex items-center gap-1">
          {LINKS.map(({ href, label }) => {
            const active = pathname === href || pathname?.startsWith(href + "/");
            return (
              <Link
                key={href}
                href={href}
                className="relative px-4 py-4 text-xs font-semibold tracking-widest transition-colors duration-150"
                style={{
                  color: active ? "#00ff88" : "#64748b",
                }}
              >
                {label}
                {active && (
                  <span
                    className="absolute bottom-0 left-0 right-0 h-0.5"
                    style={{ backgroundColor: "#00ff88" }}
                  />
                )}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Right side: slate date + lineups + countdown */}
      <div className="flex items-center gap-6">
        {slateInfo && (
          <div className="flex items-center gap-4">
            <span className="font-data text-xs text-slate-500 tracking-wide">
              {slateInfo.display_date ?? slateInfo.games[0] ?? "—"}
            </span>
            <span
              className="font-data text-xs font-semibold px-2 py-0.5 rounded"
              style={{
                backgroundColor: "#0f1629",
                border: "1px solid #1e2d4a",
                color: validCount === denom ? "#00ff88" : "#f59e0b",
              }}
            >
              {validCount}/{denom} LINEUPS
            </span>
            <CountdownTimer lockTime={slateInfo.lock_time} />
          </div>
        )}
      </div>
    </header>
  );
}
