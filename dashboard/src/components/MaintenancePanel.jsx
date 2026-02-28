import { useQuery } from "@tanstack/react-query";
import { getMaintenanceReport } from "../services/api";
import {
  Wrench,
  AlertTriangle,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { useState } from "react";

const PRIORITY_CONFIG = {
  OK: {
    color: "text-green-400",
    bg: "bg-green-900/30",
    icon: CheckCircle,
    label: "OK",
  },
  MEDIUM: {
    color: "text-yellow-400",
    bg: "bg-yellow-900/30",
    icon: AlertCircle,
    label: "Medium",
  },
  HIGH: {
    color: "text-orange-400",
    bg: "bg-orange-900/30",
    icon: AlertTriangle,
    label: "High",
  },
  CRITICAL: {
    color: "text-red-400",
    bg: "bg-red-900/30",
    icon: XCircle,
    label: "Critical",
  },
};

const FAULT_LABELS = {
  normal: "Normal operation",
  shading: "Partial shading",
  soiling: "Dust/soiling",
  hardware: "Hardware fault",
  mismatch: "String mismatch",
  degradation: "Efficiency degradation",
};

export default function MaintenancePanel({ installationId }) {
  const [filter, setFilter] = useState("ALL");

  const { data: report, isLoading } = useQuery({
    queryKey: ["maintenance", installationId],
    queryFn: () => getMaintenanceReport(installationId, 50),
    refetchInterval: 600_000,
  });

  const panels = report?.report ?? [];
  const filtered =
    filter === "ALL"
      ? panels
      : panels.filter((p) => p.maintenance_priority === filter);

  const counts = {
    CRITICAL: panels.filter((p) => p.maintenance_priority === "CRITICAL")
      .length,
    HIGH: panels.filter((p) => p.maintenance_priority === "HIGH").length,
    MEDIUM: panels.filter((p) => p.maintenance_priority === "MEDIUM").length,
    OK: panels.filter((p) => p.maintenance_priority === "OK").length,
  };

  return (
    <div className="space-y-5">
      {/* Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(PRIORITY_CONFIG).map(([key, cfg]) => {
          const Icon = cfg.icon;
          return (
            <button
              key={key}
              onClick={() => setFilter(filter === key ? "ALL" : key)}
              className={`card text-center transition-all ${filter === key ? "ring-2 ring-amber-500" : ""} hover:bg-slate-700`}
            >
              <Icon size={24} className={`${cfg.color} mx-auto mb-2`} />
              <p className={`text-2xl font-bold ${cfg.color}`}>
                {counts[key] ?? 0}
              </p>
              <p className="stat-label">{cfg.label} Priority</p>
            </button>
          );
        })}
      </div>

      {/* Panel table */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-300">
            <Wrench size={16} className="inline mr-2 text-amber-400" />
            Panel Health Report — {installationId}
            {report && (
              <span className="ml-2 text-xs text-slate-500">
                ({report.n_panels_flagged}/{report.n_panels_checked} flagged)
              </span>
            )}
          </h3>
          {filter !== "ALL" && (
            <button
              onClick={() => setFilter("ALL")}
              className="text-xs text-amber-400 hover:text-amber-300"
            >
              Clear filter
            </button>
          )}
        </div>

        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 10 }).map((_, i) => (
              <div
                key={i}
                className="h-10 bg-slate-700 animate-pulse rounded"
              />
            ))}
          </div>
        ) : (
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-slate-800">
                <tr className="text-xs text-slate-500 uppercase">
                  <th className="text-left pb-2 pr-4">Panel ID</th>
                  <th className="text-left pb-2 pr-4">Priority</th>
                  <th className="text-left pb-2 pr-4">Fault Type</th>
                  <th className="text-right pb-2 pr-4">Anomaly Rate</th>
                  <th className="text-right pb-2">Score</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 100).map((p) => {
                  const cfg =
                    PRIORITY_CONFIG[p.maintenance_priority] ??
                    PRIORITY_CONFIG.OK;
                  const Icon = cfg.icon;
                  return (
                    <tr
                      key={p.panel_id}
                      className="border-t border-slate-700 hover:bg-slate-700/30"
                    >
                      <td className="py-2 pr-4 font-mono text-white text-xs">
                        {p.panel_id}
                      </td>
                      <td className="py-2 pr-4">
                        <span
                          className={`flex items-center gap-1 text-xs ${cfg.color}`}
                        >
                          <Icon size={12} />
                          {cfg.label}
                        </span>
                      </td>
                      <td className="py-2 pr-4 text-slate-300 text-xs">
                        {FAULT_LABELS[p.dominant_fault] ?? p.dominant_fault}
                      </td>
                      <td className="py-2 pr-4 text-right text-xs">
                        <span
                          className={
                            p.anomaly_rate > 0.1
                              ? "text-red-400"
                              : "text-slate-300"
                          }
                        >
                          {(p.anomaly_rate * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-2 text-right text-xs text-slate-400">
                        {p.avg_anomaly_score?.toFixed(3)}
                      </td>
                    </tr>
                  );
                })}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-slate-500">
                      No panels matching this filter
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
