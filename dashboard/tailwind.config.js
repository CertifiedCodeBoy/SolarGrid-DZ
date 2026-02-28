/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        solar: {
          50: "#fffbeb",
          100: "#fef3c7",
          500: "#f59e0b",
          600: "#d97706",
          700: "#b45309",
        },
        grid: { 50: "#eff6ff", 500: "#3b82f6", 700: "#1d4ed8" },
        carbon: { 50: "#f0fdf4", 500: "#22c55e", 700: "#15803d" },
      },
    },
  },
  plugins: [],
};
