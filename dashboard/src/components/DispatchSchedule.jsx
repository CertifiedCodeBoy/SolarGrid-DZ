import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getDispatchSchedule, runDispatch } from "../services/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Zap, Play, TrendingUp, TrendingDown } from "lucide-react";

export default function DispatchSchedule({ installationId }) {
  const qc = useQueryClient();

  const { data: schedule, isLoading } = useQuery({
    queryKey: ["dispatch-schedule", installationId],
    queryFn: () => getDispatchSchedule(installationId, 24),
    refetchInterval: 300_000,
  });

  const dispatchMutation = useMutation({
    mutationFn: () => runDispatch(installationId),
    onSuccess: () =>
      qc.invalidateQueries(["dispatch-schedule", installationId]),
  });

  const rows = schedule?.schedule ?? [];
  const summary = schedule?.summary ?? {};

  // Prepare chart data
  const chartData = rows.slice(0, 24).map((r, i) => ({
    hour: `H+${i + 1}`,
    charge: r.p_charge_kw,
    discharge: -r.p_discharge_kw,
    import: r.p_grid_import_kw,
    export: r.p_grid_export_kw,
    soc: (r.soc ?? 0.5) * 100,
  }));

  return (
    <div className="space-y-5">
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card">
          <p className="stat-label">Self-Consumption</p>
          <p className="stat-value text-green-400">
            {summary.self_consumption_rate_pct?.toFixed(1) ?? "—"}%
          </p>
        </div>
        <div className="card">
          <p className="stat-label">Grid Import Cost</p>
          <p className="stat-value text-red-400">
            €{summary.total_import_cost_eur?.toFixed(2) ?? "—"}
          </p>
        </div>
        <div className="card">
          <p className="stat-label">Export Revenue</p>
          <p className="stat-value text-amber-400">
            €{summary.total_export_revenue_eur?.toFixed(2) ?? "—"}
          </p>
        </div>
        <div className="card">
          <p className="stat-label">CO₂ Avoided</p>
          <p className="stat-value text-emerald-400">
            {summary.co2_avoided_kg?.toFixed(1) ?? "—"} kg
          </p>
        </div>
      </div>

      {/* Dispatch action button */}
      <div className="card flex items-center justify-between">
        <div>
          <p className="text-white font-semibold">Execute Dispatch Step</p>
          <p className="text-slate-400 text-xs mt-0.5">
            Run MPC optimiser for the current hour timestep
          </p>
        </div>
        <button
          onClick={() => dispatchMutation.mutate()}
          disabled={dispatchMutation.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-amber-500 hover:bg-amber-400 text-slate-900 font-semibold rounded-lg transition-colors disabled:opacity-50"
        >
          <Play size={16} />
          {dispatchMutation.isPending ? "Running…" : "Run Dispatch"}
        </button>
      </div>

      {/* Last action result */}
      {dispatchMutation.data && (
        <div className="card border border-amber-500/30 bg-amber-900/10">
          <p className="text-sm font-semibold text-amber-400 mb-2">
            Latest Dispatch Action
          </p>
          <div className="grid grid-cols-3 gap-3 text-sm">
            <div>
              <p className="text-slate-400 text-xs">Action</p>
              <p
                className={`font-bold ${dispatchMutation.data.action === "charge" ? "text-green-400" : dispatchMutation.data.action === "discharge" ? "text-orange-400" : "text-slate-300"}`}
              >
                {dispatchMutation.data.action?.toUpperCase()}
              </p>
            </div>
            <div>
              <p className="text-slate-400 text-xs">Power</p>
              <p className="text-white font-bold">
                {dispatchMutation.data.p_charge_kw > 0
                  ? `+${dispatchMutation.data.p_charge_kw}`
                  : dispatchMutation.data.p_discharge_kw > 0
                    ? `-${dispatchMutation.data.p_discharge_kw}`
                    : "0"}{" "}
                kW
              </p>
            </div>
            <div>
              <p className="text-slate-400 text-xs">Battery SoC After</p>
              <p className="text-white font-bold">
                {((dispatchMutation.data.soc_after ?? 0) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="card">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">
          <Zap size={15} className="inline mr-2 text-amber-400" />
          24-Hour MPC Dispatch Schedule
        </h3>
        {isLoading ? (
          <div className="h-64 bg-slate-700 animate-pulse rounded-lg" />
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData} barGap={2}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="hour"
                tick={{ fill: "#9ca3af", fontSize: 10 }}
                interval={2}
              />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} unit=" kW" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #475569",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ color: "#9ca3af", fontSize: 12 }} />
              <Bar
                dataKey="charge"
                name="Charge"
                fill="#22c55e"
                radius={[2, 2, 0, 0]}
              />
              <Bar
                dataKey="discharge"
                name="Discharge"
                fill="#ef4444"
                radius={[2, 2, 0, 0]}
              />
              <Bar
                dataKey="import"
                name="Grid Import"
                fill="#3b82f6"
                radius={[2, 2, 0, 0]}
              />
              <Bar
                dataKey="export"
                name="Grid Export"
                fill="#f59e0b"
                radius={[2, 2, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
