"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useSlate } from "@/app/context/SlateContext";
import StatBadge from "@/components/StatBadge";
import LeverageScatter from "@/components/LeverageScatter";

export default function ModelPage() {
  const { modelMetrics, players } = useSlate();
  const m = modelMetrics;

  const metrics = [
    {
      value: m.q50_rmse.toFixed(2),
      label: "q50 RMSE",
      color: "green" as const,
      desc: "Root mean squared error on holdout",
    },
    {
      value: m.q50_mae.toFixed(2),
      label: "q50 MAE",
      color: "green" as const,
      desc: "Mean absolute error on holdout",
    },
    {
      value: (m.q85_coverage * 100).toFixed(1) + "%",
      label: "q85 Coverage",
      color: m.q85_coverage >= 0.8 ? "green" as const : "amber" as const,
      desc: "% actuals below q85 prediction (target 85%)",
    },
    {
      value: (m.q15_coverage * 100).toFixed(1) + "%",
      label: "q15 Coverage",
      color: "amber" as const,
      desc: "% actuals below q15 — MLB scoring is zero-heavy",
    },
    {
      value: m.interval_width.toFixed(1),
      label: "Interval Width",
      color: "amber" as const,
      desc: "Mean q85−q15 width (higher = more volatility)",
    },
    {
      value: m.train_rows.toLocaleString(),
      label: "Train Rows",
      color: "gray" as const,
      desc: "Pitch-by-pitch Statcast plate appearances",
    },
  ];

  return (
    <div className="max-w-screen-2xl mx-auto px-6 py-6 flex flex-col gap-8">

      {/* Metrics row */}
      <section>
        <h2
          className="text-xs font-semibold uppercase tracking-widest mb-4"
          style={{ color: "#00ff88" }}
        >
          01 — XGBoost Quantile Model Metrics
        </h2>
        <p className="text-xs text-slate-500 mb-5">
          Three quantile regression models (q15 / q50 / q85) trained on{" "}
          {m.train_rows.toLocaleString()} Statcast PAs ·{" "}
          {m.feature_count} features · holdout: May 2025
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {metrics.map(({ value, label, color, desc }) => (
            <div
              key={label}
              className="rounded-lg px-4 py-4 flex flex-col gap-2"
              style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
            >
              <StatBadge value={value} label={label} color={color} size="md" />
              <p className="text-xs text-slate-600 leading-snug">{desc}</p>
            </div>
          ))}
        </div>

        {/* Coverage note */}
        {m.q15_coverage > 0.25 && (
          <div
            className="mt-3 px-4 py-3 rounded text-xs"
            style={{
              backgroundColor: "rgba(245,158,11,0.06)",
              border: "1px solid #f59e0b30",
              color: "#f59e0b",
            }}
          >
            ℹ q15 coverage {(m.q15_coverage * 100).toFixed(1)}% {">"} 25% — expected behavior.
            MLB DK scoring is zero-heavy (many 0-point games), so the floor reads high. The q85
            ceiling metric is more meaningful for GPP targeting.
          </div>
        )}
      </section>

      {/* Scatter + Feature Importance */}
      <div className="flex gap-6">

        {/* Leverage Scatter */}
        <div className="flex-1 min-w-0">
          <div
            className="rounded-lg p-5 h-full"
            style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
          >
            <h2
              className="text-xs font-semibold uppercase tracking-widest mb-1"
              style={{ color: "#00ff88" }}
            >
              02 — Ownership vs Projected Points
            </h2>
            <p className="text-xs text-slate-500 mb-5">
              <span style={{ color: "#00ff88" }}>●</span> High leverage (≥1.3) &nbsp;
              <span style={{ color: "#f59e0b" }}>●</span> Medium &nbsp;
              <span className="text-slate-600">●</span> Low
              &nbsp;·&nbsp; Top-left quadrant = GPP gold
            </p>
            <LeverageScatter players={players} />
          </div>
        </div>

        {/* Feature Importance */}
        <div className="w-96 shrink-0">
          <div
            className="rounded-lg p-5"
            style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
          >
            <h2
              className="text-xs font-semibold uppercase tracking-widest mb-1"
              style={{ color: "#00ff88" }}
            >
              03 — SHAP Feature Importance
            </h2>
            <p className="text-xs text-slate-500 mb-5">
              Top 10 features · q50 model · mean |SHAP|
            </p>
            <div style={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={m.feature_importance}
                  margin={{ top: 0, right: 24, left: 130, bottom: 0 }}
                >
                  <CartesianGrid horizontal={false} stroke="#1e2d4a" strokeDasharray="2 4" />
                  <XAxis
                    type="number"
                    tick={{ fill: "#475569", fontSize: 10, fontFamily: "JetBrains Mono" }}
                    tickFormatter={(v) => v.toFixed(2)}
                    axisLine={{ stroke: "#1e2d4a" }}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fill: "#94a3b8", fontSize: 10, fontFamily: "JetBrains Mono" }}
                    axisLine={false}
                    tickLine={false}
                    width={126}
                  />
                  <Tooltip
                    formatter={(v) => [typeof v === 'number' ? v.toFixed(3) : v, "SHAP"]}
                    contentStyle={{
                      backgroundColor: "#0f1629",
                      border: "1px solid #1e2d4a",
                      borderRadius: 4,
                      fontSize: 11,
                      fontFamily: "JetBrains Mono",
                      color: "#f8fafc",
                    }}
                    cursor={{ fill: "#ffffff06" }}
                  />
                  <Bar dataKey="importance" radius={[0, 2, 2, 0]} maxBarSize={16}>
                    {m.feature_importance.map((_, i) => (
                      <Cell
                        key={i}
                        fill="#00ff88"
                        fillOpacity={1 - i * 0.07}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Model architecture note */}
      <section>
        <div
          className="rounded-lg p-5"
          style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
        >
          <h2
            className="text-xs font-semibold uppercase tracking-widest mb-4"
            style={{ color: "#00ff88" }}
          >
            04 — Architecture Notes
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-slate-400">
            <div>
              <div className="font-semibold text-slate-200 mb-2">Training Data</div>
              <ul className="space-y-1 text-slate-500">
                <li>• Statcast pitch-by-pitch, 2023–2026</li>
                <li>• {m.train_rows.toLocaleString()} plate appearances</li>
                <li>• Terminal events only (events ≠ null)</li>
                <li>• Holdout: May {new Date().getFullYear() - 1} ({m.test_rows.toLocaleString()} rows)</li>
              </ul>
            </div>
            <div>
              <div className="font-semibold text-slate-200 mb-2">Model Config</div>
              <ul className="space-y-1 text-slate-500 font-data">
                <li>objective: reg:quantileerror</li>
                <li>n_estimators: 500</li>
                <li>max_depth: 6 · lr: 0.05</li>
                <li>subsample: 0.8 · colsample: 0.8</li>
              </ul>
            </div>
            <div>
              <div className="font-semibold text-slate-200 mb-2">Features ({m.feature_count})</div>
              <ul className="space-y-1 text-slate-500">
                <li>• Rolling exit velo / xwOBA (7/14/30d)</li>
                <li>• Barrel rate, hard-hit rate windows</li>
                <li>• Platoon advantage, batting order</li>
                <li>• Run differential, high-leverage flag</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
