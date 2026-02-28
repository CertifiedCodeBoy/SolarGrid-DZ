import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getSystemOverview, getInstallations } from "./services/api";
import SystemStats from "./components/SystemStats";
import ForecastChart from "./components/ForecastChart";
import BatteryStatus from "./components/BatteryStatus";
import CO2Tracker from "./components/CO2Tracker";
import DistrictMap from "./components/DistrictMap";
import MaintenancePanel from "./components/MaintenancePanel";
import DispatchSchedule from "./components/DispatchSchedule";
import {
  Sun,
  Battery,
  Zap,
  Leaf,
  Wrench,
  Map,
  BarChart2,
  RefreshCw,
} from "lucide-react";

const TABS = [
  { id: "overview", label: "Overview", icon: BarChart2 },
  { id: "forecast", label: "Forecasts", icon: Sun },
  { id: "dispatch", label: "Dispatch", icon: Zap },
  { id: "battery", label: "Battery", icon: Battery },
  { id: "carbon", label: "Carbon", icon: Leaf },
  { id: "map", label: "District Map", icon: Map },
  { id: "maintenance", label: "Maintenance", icon: Wrench },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedInstallation, setSelectedInstallation] = useState(null);

  const {
    data: overview,
    isLoading: overviewLoading,
    refetch: refetchOverview,
  } = useQuery({
    queryKey: ["system-overview"],
    queryFn: getSystemOverview,
    refetchInterval: 30_000,
  });

  const { data: installations = [] } = useQuery({
    queryKey: ["installations"],
    queryFn: getInstallations,
    onSuccess: (data) => {
      if (data.length > 0 && !selectedInstallation) {
        setSelectedInstallation(data[0].installation_id);
      }
    },
  });

  const selected = selectedInstallation || installations[0]?.installation_id;

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div className="flex items-center justify-between max-w-screen-2xl mx-auto">
          <div className="flex items-center gap-3">
            <span className="text-3xl">☀️</span>
            <div>
              <h1 className="text-xl font-bold text-white">SolarGrid DZ</h1>
              <p className="text-xs text-slate-400">
                Smart Solar Energy Management · Algeria
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {overview && (
              <div className="flex items-center gap-2 text-sm">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-green-400 font-medium">
                  {overview.total_solar_kw?.toFixed(1)} kW live
                </span>
              </div>
            )}
            <button
              onClick={() => refetchOverview()}
              className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-screen-2xl mx-auto px-6 py-6">
        {/* Installation selector */}
        <div className="flex items-center gap-3 mb-6">
          <span className="text-sm text-slate-400">Installation:</span>
          <select
            value={selected || ""}
            onChange={(e) => setSelectedInstallation(e.target.value)}
            className="bg-slate-700 border border-slate-600 text-white text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-amber-500"
          >
            {installations.map((inst) => (
              <option key={inst.installation_id} value={inst.installation_id}>
                {inst.installation_id} — {inst.district_id} (
                {inst.district_type})
              </option>
            ))}
          </select>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-6 bg-slate-800 p-1 rounded-xl border border-slate-700 w-fit">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === id
                  ? "bg-amber-500 text-slate-900"
                  : "text-slate-400 hover:text-white hover:bg-slate-700"
              }`}
            >
              <Icon size={15} />
              {label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            <SystemStats overview={overview} loading={overviewLoading} />
          </div>
        )}
        {activeTab === "forecast" && selected && (
          <ForecastChart installationId={selected} />
        )}
        {activeTab === "dispatch" && selected && (
          <DispatchSchedule installationId={selected} />
        )}
        {activeTab === "battery" && selected && (
          <BatteryStatus installationId={selected} overview={overview} />
        )}
        {activeTab === "carbon" && <CO2Tracker />}
        {activeTab === "map" && <DistrictMap overview={overview} />}
        {activeTab === "maintenance" && selected && (
          <MaintenancePanel installationId={selected} />
        )}
      </div>
    </div>
  );
}
