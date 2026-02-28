import { MapPin, ArrowRight } from "lucide-react";

// Simplified district map without Mapbox dependency (works without API key)
// Shows district cards with power flows between them.

const DISTRICT_COORDS = {
  "DZ-ALG-01": { x: 48, y: 45, name: "El Harrach" },
  "DZ-ALG-02": { x: 64, y: 35, name: "Hussein Dey" },
  "DZ-ALG-03": { x: 35, y: 55, name: "Bir Mourad Raïs" },
  "DZ-ALG-04": { x: 58, y: 58, name: "Kouba" },
  "DZ-ALG-05": { x: 42, y: 32, name: "Bab Ezzouar" },
};

function DistrictNode({ district, x, y, name }) {
  const soc = district?.battery_soc ?? 0.5;
  const net = district?.net_kw ?? 0;
  const isExporting = net > 0;
  const nodeColor = isExporting ? "#22c55e" : "#ef4444";

  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Pulse ring */}
      <circle r="18" fill={nodeColor} fillOpacity="0.1">
        <animate
          attributeName="r"
          values="16;22;16"
          dur="3s"
          repeatCount="indefinite"
        />
        <animate
          attributeName="fill-opacity"
          values="0.1;0;0.1"
          dur="3s"
          repeatCount="indefinite"
        />
      </circle>
      {/* Node */}
      <circle r="14" fill="#1e293b" stroke={nodeColor} strokeWidth="2" />
      <text
        textAnchor="middle"
        fill="white"
        fontSize="8"
        dy="3"
        fontWeight="700"
      >
        {(soc * 100).toFixed(0)}%
      </text>
      {/* Label */}
      <text textAnchor="middle" fill="#94a3b8" fontSize="7" dy="26">
        {name}
      </text>
      <text textAnchor="middle" fill={nodeColor} fontSize="7" dy="36">
        {net >= 0 ? "+" : ""}
        {net?.toFixed(0)} kW
      </text>
    </g>
  );
}

export default function DistrictMap({ overview }) {
  const districts = overview?.district_balances ?? [];

  return (
    <div className="space-y-5">
      <div className="card">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">
          🗺️ District Energy Map — Algiers
        </h3>
        <div
          className="relative bg-slate-900 rounded-lg overflow-hidden border border-slate-700"
          style={{ height: 400 }}
        >
          <svg
            width="100%"
            height="100%"
            viewBox="0 0 100 100"
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Background grid */}
            <defs>
              <pattern
                id="grid"
                width="5"
                height="5"
                patternUnits="userSpaceOnUse"
              >
                <path
                  d="M 5 0 L 0 0 0 5"
                  fill="none"
                  stroke="#1e293b"
                  strokeWidth="0.2"
                />
              </pattern>
            </defs>
            <rect width="100" height="100" fill="url(#grid)" />

            {/* Algeria coastline simulation */}
            <path
              d="M 20,20 Q 50,10 80,18 L 85,25 Q 60,20 30,25 Z"
              fill="#1e3a5f"
              fillOpacity="0.4"
            />

            {/* Connection lines between districts */}
            {districts
              .filter((d) => DISTRICT_COORDS[d.district_id])
              .map((d, i) => {
                const from = DISTRICT_COORDS[d.district_id];
                return districts
                  .slice(i + 1)
                  .filter((d2) => DISTRICT_COORDS[d2.district_id])
                  .map((d2) => {
                    const to = DISTRICT_COORDS[d2.district_id];
                    return (
                      <line
                        key={`${d.district_id}-${d2.district_id}`}
                        x1={from.x}
                        y1={from.y}
                        x2={to.x}
                        y2={to.y}
                        stroke="#334155"
                        strokeWidth="0.5"
                        strokeDasharray="2 2"
                      />
                    );
                  });
              })}

            {/* District nodes */}
            {districts.map((d) => {
              const coords = DISTRICT_COORDS[d.district_id];
              if (!coords) return null;
              return (
                <DistrictNode
                  key={d.district_id}
                  district={d}
                  x={coords.x}
                  y={coords.y}
                  name={coords.name}
                />
              );
            })}
          </svg>

          {/* Legend */}
          <div className="absolute bottom-3 left-3 flex gap-4 text-xs">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-green-500 inline-block" />
              Surplus
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-red-500 inline-block" />
              Deficit
            </span>
            <span className="text-slate-500">Node % = Battery SoC</span>
          </div>
        </div>
      </div>

      {/* District list */}
      <div className="card">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">
          District Details
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {districts.map((d) => (
            <div key={d.district_id} className="bg-slate-700 rounded-lg p-3">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-white text-sm font-semibold">
                    {d.district_id}
                  </p>
                  <p className="text-slate-400 text-xs">
                    {d.district_type} · {DISTRICT_COORDS[d.district_id]?.name}
                  </p>
                </div>
                <span
                  className={`badge ${d.net_kw >= 0 ? "badge-green" : "badge-red"}`}
                >
                  {d.net_kw >= 0 ? "Surplus" : "Deficit"}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 mt-3 text-xs text-center">
                <div>
                  <p className="text-amber-400 font-semibold">
                    {d.solar_kw.toFixed(0)}
                  </p>
                  <p className="text-slate-500">Solar kW</p>
                </div>
                <div>
                  <p className="text-blue-400 font-semibold">
                    {d.demand_kw.toFixed(0)}
                  </p>
                  <p className="text-slate-500">Demand kW</p>
                </div>
                <div>
                  <p className="text-green-400 font-semibold">
                    {(d.battery_soc * 100).toFixed(0)}%
                  </p>
                  <p className="text-slate-500">Battery</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
