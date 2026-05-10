"use client";

import { useEffect } from "react";
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
  const { modelInfo, playerPool, fetchModelInfo, loading, error } = useSlate();

  useEffect(() => {
    void fetchModelInfo();
  }, [fetchModelInfo]);

  const hitterLoaded = modelInfo?.hitter_metrics?.loaded === true;
  const pitcherLoaded = modelInfo?.pitcher_metrics?.loaded === true;
  const hitterFc = modelInfo?.hitter_metrics?.feature_count ?? 0;
  const pitcherFc = modelInfo?.pitcher_metrics?.feature_count ?? 0;

  const hitterBarData =
    modelInfo?.hitter_features.slice(0, 12).map((feature) => ({
      feature,
      /** API does not expose SHAP; bars are uniform to show column order only. */
      presence: 1,
    })) ?? [];

  const hasHitterFeatures = hitterBarData.length > 0;

  return (
    <div className="max-w-screen-2xl mx-auto px-6 py-6 flex flex-col gap-8 relative">
      {loading && (
        <div
          className="absolute inset-0 z-30 flex items-center justify-center rounded-lg"
          style={{ backgroundColor: "rgba(10,14,26,0.65)" }}
        >
          <div className="flex flex-col items-center gap-3">
            <span className="inline-block w-10 h-10 border-2 border-slate-600 border-t-[#00ff88] rounded-full animate-spin" />
            <span className="text-xs uppercase tracking-widest text-slate-400">
              Loading model info…
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

      {/* Model load status */}
      <section>
        <h2
          className="text-xs font-semibold uppercase tracking-widest mb-4"
          style={{ color: "#00ff88" }}
        >
          01 — Loaded models
        </h2>
        <p className="text-xs text-slate-500 mb-5">
          Status from <span className="font-data text-slate-400">GET /api/model-info</span>{" "}
          · quantile bundles load on API startup when joblib artifacts exist.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            {
              value: hitterLoaded ? "LOADED" : "NOT LOADED",
              label: "Hitter q bundle",
              color: hitterLoaded ? ("green" as const) : ("amber" as const),
            },
            {
              value: pitcherLoaded ? "LOADED" : "NOT LOADED",
              label: "Pitcher q bundle",
              color: pitcherLoaded ? ("green" as const) : ("amber" as const),
            },
            {
              value: String(hitterFc),
              label: "Hitter feature cols",
              color: "gray" as const,
            },
            {
              value: String(pitcherFc),
              label: "Pitcher feature cols",
              color: "gray" as const,
            },
          ].map(({ value, label, color }) => (
            <div
              key={label}
              className="rounded-lg px-4 py-4 flex flex-col gap-2"
              style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
            >
              <StatBadge value={value} label={label} color={color} size="md" />
            </div>
          ))}
        </div>
      </section>

      {/* Scatter + Feature column chart (order-only when SHAP not provided) */}
      <div className="flex gap-6">
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
            {playerPool.length > 0 ? (
              <LeverageScatter players={playerPool} />
            ) : (
              <div
                className="flex items-center justify-center text-sm text-slate-600"
                style={{ height: 320 }}
              >
                Load a slate and projections to plot the current pool.
              </div>
            )}
          </div>
        </div>

        <div className="w-96 shrink-0">
          <div
            className="rounded-lg p-5"
            style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
          >
            <h2
              className="text-xs font-semibold uppercase tracking-widest mb-1"
              style={{ color: "#00ff88" }}
            >
              03 — Hitter feature columns
            </h2>
            <p className="text-xs text-slate-500 mb-5">
              First 12 hitter model input columns (order as returned by the API).
              Magnitudes are not available from{" "}
              <span className="font-data text-slate-400">/api/model-info</span>.
            </p>
            {hasHitterFeatures ? (
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={hitterBarData.slice(0, 12)}
                    margin={{ top: 0, right: 24, left: 130, bottom: 0 }}
                  >
                    <CartesianGrid
                      horizontal={false}
                      stroke="#1e2d4a"
                      strokeDasharray="2 4"
                    />
                    <XAxis
                      type="number"
                      domain={[0, 1.5]}
                      tick={{
                        fill: "#475569",
                        fontSize: 10,
                        fontFamily: "JetBrains Mono",
                      }}
                      tickFormatter={() => ""}
                      axisLine={{ stroke: "#1e2d4a" }}
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="feature"
                      tick={{
                        fill: "#94a3b8",
                        fontSize: 10,
                        fontFamily: "JetBrains Mono",
                      }}
                      axisLine={false}
                      tickLine={false}
                      width={126}
                    />
                    <Tooltip
                      formatter={(_, _n, item) => {
                        const feat = (item as { payload?: { feature: string } })
                          ?.payload?.feature;
                        return [feat ?? "", "column"];
                      }}
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
                    <Bar dataKey="presence" radius={[0, 2, 2, 0]} maxBarSize={16}>
                      {hitterBarData.slice(0, 12).map((_, i) => (
                        <Cell key={i} fill="#00ff88" fillOpacity={0.85} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div
                className="flex items-center justify-center text-xs text-slate-600 text-center px-2"
                style={{ height: 300 }}
              >
                No hitter feature list returned yet — train/load a hitter model on
                the API host.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Feature lists */}
      <section>
        <div
          className="rounded-lg p-5"
          style={{ backgroundColor: "#0f1629", border: "1px solid #1e2d4a" }}
        >
          <h2
            className="text-xs font-semibold uppercase tracking-widest mb-4"
            style={{ color: "#00ff88" }}
          >
            04 — Full feature column lists
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
            <div>
              <div className="font-semibold text-slate-200 mb-2">Hitter</div>
              <ul
                className="space-y-1 text-slate-500 font-data max-h-48 overflow-y-auto"
                style={{ border: "1px solid #1e2d4a", borderRadius: 6, padding: 8 }}
              >
                {(modelInfo?.hitter_features ?? []).length ? (
                  modelInfo!.hitter_features.map((f) => (
                    <li key={f}>{f}</li>
                  ))
                ) : (
                  <li className="text-slate-600">—</li>
                )}
              </ul>
            </div>
            <div>
              <div className="font-semibold text-slate-200 mb-2">Pitcher</div>
              <ul
                className="space-y-1 text-slate-500 font-data max-h-48 overflow-y-auto"
                style={{ border: "1px solid #1e2d4a", borderRadius: 6, padding: 8 }}
              >
                {(modelInfo?.pitcher_features ?? []).length ? (
                  modelInfo!.pitcher_features.map((f) => (
                    <li key={f}>{f}</li>
                  ))
                ) : (
                  <li className="text-slate-600">—</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
