"use client";

interface StatBadgeProps {
  value: string | number;
  label: string;
  color?: "green" | "amber" | "red" | "gray" | "blue";
  size?: "sm" | "md" | "lg";
}

const colorMap = {
  green: "#00ff88",
  amber: "#f59e0b",
  red: "#ef4444",
  gray: "#94a3b8",
  blue: "#60a5fa",
};

export default function StatBadge({
  value,
  label,
  color = "green",
  size = "md",
}: StatBadgeProps) {
  const valueSize =
    size === "lg" ? "text-3xl" : size === "md" ? "text-2xl" : "text-lg";
  const labelSize = size === "lg" ? "text-sm" : "text-xs";

  return (
    <div
      className="flex flex-col items-start gap-0.5"
      style={{ minWidth: 0 }}
    >
      <span
        className={`${valueSize} font-bold tracking-tight leading-none font-data`}
        style={{ color: colorMap[color] }}
      >
        {value}
      </span>
      <span className={`${labelSize} font-medium uppercase tracking-widest text-slate-500`}>
        {label}
      </span>
    </div>
  );
}
