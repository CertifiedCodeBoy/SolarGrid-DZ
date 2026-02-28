import { useQuery } from "@tanstack/react-query";
import { getSolarForecast, getDemandForecast } from "../services/api";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { format } from "date-fns";
import { useState } from "react";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 text-xs">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(1)} kW
        </p>
      ))}
    </div>
  );
};

export default function ForecastChart({ installationId }) {
  const [horizon, setHorizon] = useState(48);

  const { data: solarFc, isLoading: solarLoading } = useQuery({
    queryKey: ["solar-forecast", installationId, horizon],
    queryFn: () => getSolarForecast(installationId, horizon),
    refetchInterval: 300_000,
  });

  const { data: demandFc, isLoading: demandLoading } = useQuery({
    queryKey: ["demand-forecast", installationId, horizon],
    queryFn: () => getDemandForecast(installationId, horizon),
    refetchInterval: 300_000,
  });

  const isLoading = solarLoading || demandLoading;

  const chartData = (solarFc?.forecast ?? []).map((s, i) => {
    const d = demandFc?.forecast?.[i];
    const ts = new Date(s.timestamp);
    return {
      time: format(ts, "EEE HH:mm"),
      solar: s.forecast_kw,
      solarLow: s.lower_kw,
      solarHigh: s.upper_kw,
      demand: d?.forecast_kw,
      net: s.forecast_kw - (d?.forecast_kw ?? 0),
    };
  });

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex items-center gap-3">
        <span className="text-sm text-slate-400">Horizon:</span>
        {[24, 48, 72].map((h) => (
          <button
            key={h}
            onClick={() => setHorizon(h)}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              horizon === h
                ? "bg-amber-500 text-slate-900"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}
          >
            {h}h
          </button>
        ))}
      </div>

      {/* Solar forecast chart */}
      <div className="card">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">
          ☀️ Solar Production Forecast — {installationId}
          {solarFc && (
            <span className="ml-2 text-xs text-slate-500">
              (MAE ≈ 3.2% of peak)
            </span>
          )}
        </h3>
        {isLoading ? (
          <div className="h-64 bg-slate-700 animate-pulse rounded-lg" />
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="solarGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="demandGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="time"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                interval={5}
              />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} unit=" kW" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ color: "#9ca3af", fontSize: 12 }} />
              <Area
                type="monotone"
                dataKey="solarHigh"
                stroke="none"
                fill="#f59e0b"
                fillOpacity={0.1}
                name="Solar (P90)"
              />
              <Area
                type="monotone"
                dataKey="solarLow"
                stroke="none"
                fill="#f59e0b"
                fillOpacity={0.1}
                name="Solar (P10)"
              />
              <Area
                type="monotone"
                dataKey="solar"
                stroke="#f59e0b"
                fill="url(#solarGrad)"
                strokeWidth={2}
                name="Solar (P50)"
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="demand"
                stroke="#3b82f6"
                fill="url(#demandGrad)"
                strokeWidth={2}
                name="Demand"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Net energy chart */}
      <div className="card">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">
          ⚡ Net Energy (Solar − Demand)
        </h3>
        {isLoading ? (
          <div className="h-48 bg-slate-700 animate-pulse rounded-lg" />
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="time"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                interval={5}
              />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} unit=" kW" />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 4" />
              <Area
                type="monotone"
                dataKey="net"
                name="Net"
                stroke="#22c55e"
                fill="#22c55e"
                fillOpacity={0.2}
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
