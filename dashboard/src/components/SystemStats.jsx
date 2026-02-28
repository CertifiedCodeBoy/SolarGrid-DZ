import { Sun, Zap, Battery, TrendingDown, Leaf, Activity } from "lucide-react";

function StatCard({
  icon: Icon,
  label,
  value,
  unit,
  color = "text-white",
  sublabel,
}) {
  return (
    <div className="card flex items-start gap-4">
      <div className={`p-3 rounded-lg bg-slate-700 ${color}`}>
        <Icon size={22} />
      </div>
      <div className="flex-1">
        <p className="stat-label">{label}</p>
        <p className="stat-value">
          {value ?? "—"}
          {unit && (
            <span className="text-sm font-normal text-slate-400 ml-1">
              {unit}
            </span>
          )}
        </p>
        {sublabel && <p className="text-xs text-slate-400 mt-1">{sublabel}</p>}
      </div>
    </div>
  );
}

export default function SystemStats({ overview, loading }) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="card animate-pulse h-28 bg-slate-700" />
        ))}
      </div>
    );
  }

  if (!overview) {
    return (
      <div className="card text-center text-slate-400 py-12">
        <Activity size={40} className="mx-auto mb-3 opacity-40" />
        <p>Connecting to SolarGrid API…</p>
        <p className="text-xs mt-1">
          Make sure the backend is running on port 8000
        </p>
      </div>
    );
  }

  const stats = [
    {
      icon: Sun,
      label: "Solar Output",
      value: overview.total_solar_kw?.toFixed(1),
      unit: "kW",
      color: "text-amber-400",
      sublabel: `${overview.n_installations} installations`,
    },
    {
      icon: Activity,
      label: "Total Demand",
      value: overview.total_demand_kw?.toFixed(1),
      unit: "kW",
      color: "text-blue-400",
      sublabel: "District consumption",
    },
    {
      icon: Battery,
      label: "Avg Battery SoC",
      value: ((overview.total_battery_soc_avg ?? 0) * 100).toFixed(1),
      unit: "%",
      color: "text-green-400",
      sublabel: "Across all BESS",
    },
    {
      icon: Zap,
      label: "Grid Import",
      value: overview.total_grid_import_kw?.toFixed(1),
      unit: "kW",
      color: "text-orange-400",
      sublabel: `Export: ${overview.total_grid_export_kw?.toFixed(1)} kW`,
    },
    {
      icon: TrendingDown,
      label: "Self-Consumption",
      value: overview.system_self_consumption_pct?.toFixed(1),
      unit: "%",
      color: "text-purple-400",
      sublabel: "Solar used locally",
    },
    {
      icon: Leaf,
      label: "CO₂ Avoided",
      value: overview.co2_avoided_today_kg?.toFixed(0),
      unit: "kg",
      color: "text-emerald-400",
      sublabel: "Today",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        {stats.map((s) => (
          <StatCard key={s.label} {...s} />
        ))}
      </div>

      {/* District table */}
      {overview.district_balances?.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">
            District Energy Balance
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-500 uppercase">
                  <th className="text-left pb-2">District</th>
                  <th className="text-left pb-2">Type</th>
                  <th className="text-right pb-2">Solar kW</th>
                  <th className="text-right pb-2">Demand kW</th>
                  <th className="text-right pb-2">Net kW</th>
                  <th className="text-right pb-2">Battery SoC</th>
                </tr>
              </thead>
              <tbody>
                {overview.district_balances.map((d) => (
                  <tr key={d.district_id} className="border-t border-slate-700">
                    <td className="py-2 text-white font-mono text-xs">
                      {d.district_id}
                    </td>
                    <td className="py-2 text-slate-300">{d.district_type}</td>
                    <td className="py-2 text-right text-amber-400">
                      {d.solar_kw.toFixed(1)}
                    </td>
                    <td className="py-2 text-right text-blue-400">
                      {d.demand_kw.toFixed(1)}
                    </td>
                    <td
                      className={`py-2 text-right font-semibold ${d.net_kw >= 0 ? "text-green-400" : "text-red-400"}`}
                    >
                      {d.net_kw >= 0 ? "+" : ""}
                      {d.net_kw.toFixed(1)}
                    </td>
                    <td className="py-2 text-right text-slate-300">
                      {(d.battery_soc * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
