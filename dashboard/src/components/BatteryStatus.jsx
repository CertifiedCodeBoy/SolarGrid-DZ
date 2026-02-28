import { Battery, Zap, TrendingUp, TrendingDown } from "lucide-react";

function SoCGauge({ soc }) {
  const pct = Math.round((soc ?? 0) * 100);
  const color = pct > 60 ? "#22c55e" : pct > 30 ? "#f59e0b" : "#ef4444";
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - pct / 100);

  return (
    <div className="flex flex-col items-center">
      <svg width="180" height="180" viewBox="0 0 180 180">
        <circle
          cx="90"
          cy="90"
          r={radius}
          fill="none"
          stroke="#334155"
          strokeWidth="14"
        />
        <circle
          cx="90"
          cy="90"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          transform="rotate(-90 90 90)"
          style={{ transition: "stroke-dashoffset 1s ease" }}
        />
        <text
          x="90"
          y="85"
          textAnchor="middle"
          fill="white"
          fontSize="28"
          fontWeight="700"
        >
          {pct}%
        </text>
        <text x="90" y="108" textAnchor="middle" fill="#94a3b8" fontSize="13">
          SoC
        </text>
      </svg>
    </div>
  );
}

export default function BatteryStatus({ installationId, overview }) {
  const districtBalance = overview?.district_balances?.find(
    (d) => d.district_id === installationId || d.district_id.includes("01"),
  );
  const soc = districtBalance?.battery_soc ?? 0.5;
  const netKw = districtBalance?.net_kw ?? 0;
  const action = netKw > 10 ? "Charging" : netKw < -10 ? "Discharging" : "Idle";
  const actionColor =
    action === "Charging"
      ? "text-green-400"
      : action === "Discharging"
        ? "text-orange-400"
        : "text-slate-400";

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Gauge */}
        <div className="card flex flex-col items-center justify-center py-8">
          <SoCGauge soc={soc} />
          <div
            className={`flex items-center gap-2 mt-3 font-semibold ${actionColor}`}
          >
            {action === "Charging" && <TrendingUp size={18} />}
            {action === "Discharging" && <TrendingDown size={18} />}
            {action === "Idle" && <Battery size={18} />}
            <span>{action}</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">{installationId}</p>
        </div>

        {/* Stats */}
        <div className="space-y-3">
          <div className="card">
            <div className="flex items-center gap-3">
              <Battery size={20} className="text-green-400" />
              <div>
                <p className="stat-label">State of Charge</p>
                <p className="text-lg font-bold text-white">
                  {(soc * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
          <div className="card">
            <div className="flex items-center gap-3">
              <Zap size={20} className="text-amber-400" />
              <div>
                <p className="stat-label">Net Power Flow</p>
                <p
                  className={`text-lg font-bold ${netKw >= 0 ? "text-green-400" : "text-red-400"}`}
                >
                  {netKw >= 0 ? "+" : ""}
                  {netKw?.toFixed(1)} kW
                </p>
              </div>
            </div>
          </div>
          <div className="card">
            <p className="stat-label mb-3">System Health Guidelines</p>
            <ul className="space-y-1 text-xs text-slate-400">
              <li className="flex gap-2">
                <span className="text-green-400">●</span> SoC &gt; 60% — optimal
                buffer zone
              </li>
              <li className="flex gap-2">
                <span className="text-amber-400">●</span> SoC 30–60% —
                acceptable operating range
              </li>
              <li className="flex gap-2">
                <span className="text-red-400">●</span> SoC &lt; 30% — priority
                recharge from grid
              </li>
              <li className="flex gap-2">
                <span className="text-blue-400">●</span> SoC &gt; 95% — suspend
                charging, export excess
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* All district batteries */}
      {overview?.district_balances && (
        <div className="card">
          <h3 className="text-sm font-semibold text-slate-300 mb-4">
            All District Batteries
          </h3>
          <div className="space-y-3">
            {overview.district_balances.map((d) => {
              const pct = (d.battery_soc * 100).toFixed(1);
              const barColor =
                d.battery_soc > 0.6
                  ? "bg-green-500"
                  : d.battery_soc > 0.3
                    ? "bg-amber-500"
                    : "bg-red-500";
              return (
                <div key={d.district_id} className="flex items-center gap-3">
                  <span className="text-xs text-slate-400 w-28 shrink-0">
                    {d.district_id}
                  </span>
                  <div className="flex-1 bg-slate-700 rounded-full h-2.5">
                    <div
                      className={`${barColor} h-2.5 rounded-full`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-white w-12 text-right">
                    {pct}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
