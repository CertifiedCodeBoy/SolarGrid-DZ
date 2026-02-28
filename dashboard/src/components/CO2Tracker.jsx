import { useQuery } from "@tanstack/react-query";
import {
  getCarbonReport,
  getMonthlyCarbon,
  getNationalTarget,
} from "../services/api";
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
import { Leaf, TreeDeciduous, Car, Target } from "lucide-react";

export default function CO2Tracker() {
  const { data: report } = useQuery({
    queryKey: ["carbon-report"],
    queryFn: () => getCarbonReport(30),
    refetchInterval: 300_000,
  });

  const { data: monthly } = useQuery({
    queryKey: ["monthly-carbon"],
    queryFn: getMonthlyCarbon,
    staleTime: 3600_000,
  });

  const { data: target } = useQuery({
    queryKey: ["national-target"],
    queryFn: getNationalTarget,
    staleTime: 3600_000,
  });

  return (
    <div className="space-y-5">
      {/* Key stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <Leaf size={28} className="text-emerald-400 mx-auto mb-2" />
          <p className="stat-value text-emerald-400">
            {report?.co2_avoided_tons?.toFixed(1) ?? "—"}
          </p>
          <p className="stat-label">Tons CO₂ Avoided (30d)</p>
        </div>
        <div className="card text-center">
          <TreeDeciduous size={28} className="text-green-400 mx-auto mb-2" />
          <p className="stat-value text-green-400">
            {report?.equivalent_trees?.toFixed(0) ?? "—"}
          </p>
          <p className="stat-label">Equivalent Trees</p>
        </div>
        <div className="card text-center">
          <Car size={28} className="text-blue-400 mx-auto mb-2" />
          <p className="stat-value text-blue-400">
            {report?.equivalent_cars_off_road?.toFixed(1) ?? "—"}
          </p>
          <p className="stat-label">Cars Off Road (eq.)</p>
        </div>
        <div className="card text-center">
          <Target size={28} className="text-amber-400 mx-auto mb-2" />
          <p className="stat-value text-amber-400">
            {report?.self_consumption_rate_pct?.toFixed(1) ?? "—"}%
          </p>
          <p className="stat-label">Self-Consumption Rate</p>
        </div>
      </div>

      {/* Monthly bar chart */}
      {monthly?.monthly && (
        <div className="card">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">
            Monthly CO₂ Avoidance (kg)
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={monthly.monthly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="month" tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} unit=" kg" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #475569",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(v) => [`${v.toLocaleString()} kg`, "CO₂ Avoided"]}
              />
              <Bar
                dataKey="co2_avoided_kg"
                name="CO₂ Avoided"
                radius={[4, 4, 0, 0]}
              >
                {monthly.monthly.map((_, i) => (
                  <Cell key={i} fill={`hsl(${140 + i * 3}, 60%, 45%)`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* National target progress */}
      {target && (
        <div className="card">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">
            🇩🇿 Algeria 2030 Renewable Target
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">
                Target: {target.target_capacity_gw} GW by {target.target_year}
              </span>
              <span className="text-white font-semibold">
                {target.progress_pct}% achieved
              </span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-amber-500 to-green-500 h-3 rounded-full transition-all duration-1000"
                style={{ width: `${Math.min(target.progress_pct, 100)}%` }}
              />
            </div>
            <div className="grid grid-cols-3 gap-4 pt-2">
              <div>
                <p className="text-xs text-slate-500">Current</p>
                <p className="text-white font-semibold">
                  {target.current_capacity_gw} GW
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Years Remaining</p>
                <p className="text-white font-semibold">
                  {target.years_remaining}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Needed/Year</p>
                <p className="text-white font-semibold">
                  {target.annual_addition_needed_gw} GW
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
