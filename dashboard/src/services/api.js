import axios from "axios";

const api = axios.create({ baseURL: "/api/v1" });

export const getInstallations = () =>
  api.get("/installations/").then((r) => r.data);
export const getSystemOverview = () =>
  api.get("/installations/system/overview").then((r) => r.data);
export const getSolarForecast = (id, horizon = 48) =>
  api.get(`/forecasts/solar/${id}?horizon=${horizon}`).then((r) => r.data);
export const getDemandForecast = (id, horizon = 48) =>
  api.get(`/forecasts/demand/${id}?horizon=${horizon}`).then((r) => r.data);
export const getDispatchSchedule = (id, horizon = 24) =>
  api.get(`/dispatch/${id}/schedule?horizon=${horizon}`).then((r) => r.data);
export const getCarbonReport = (days = 30) =>
  api.get(`/carbon/report?period_days=${days}`).then((r) => r.data);
export const getMonthlyCarbon = () =>
  api.get("/carbon/monthly").then((r) => r.data);
export const getNationalTarget = () =>
  api.get("/carbon/national-target").then((r) => r.data);
export const getMaintenanceReport = (id, nPanels = 50) =>
  api.get(`/maintenance/${id}/report?n_panels=${nPanels}`).then((r) => r.data);
export const getMaintenanceAlerts = (id) =>
  api.get(`/maintenance/${id}/alerts`).then((r) => r.data);
export const runDispatch = (id) =>
  api.post(`/dispatch/${id}/action`).then((r) => r.data);

export default api;
