import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "system-ui", "sans-serif"],
      },
      colors: {
        marine: {
          50: "#eef8ff",
          100: "#d9efff",
          200: "#bce3ff",
          300: "#8ed3ff",
          400: "#59b9ff",
          500: "#3399ff",
          600: "#1d7af5",
          700: "#1661e2",
          800: "#194fb7",
          900: "#1b4590",
          950: "#0b1e47",
        },
        ocean: {
          400: "#38bdf8",
          500: "#0ea5e9",
          600: "#0284c7",
        },
        cyan: {
          300: "#67e8f9",
          400: "#22d3ee",
          500: "#06b6d4",
        },
        abyss: {
          950: "#02080f",
          900: "#051018",
          850: "#071422",
          800: "#0a1a2b",
          700: "#0e2338",
        },
      },
      boxShadow: {
        glass: "0 8px 32px rgba(0, 0, 0, 0.35)", // softer, more diffuse than drop
        "glass-sm": "0 2px 12px rgba(0, 0, 0, 0.25)",
        glow: "0 0 0 1px rgba(56, 189, 248, 0.15), 0 0 24px rgba(6, 182, 212, 0.08)",
        "glow-strong":
          "0 0 0 1px rgba(56, 189, 248, 0.25), 0 0 32px rgba(6, 182, 212, 0.15)",
      },
      backgroundImage: {
        "deep-radial":
          "radial-gradient(1200px 600px at 70% -10%, rgba(14, 116, 144, 0.18), transparent 60%), radial-gradient(900px 600px at 20% 110%, rgba(37, 99, 235, 0.12), transparent 55%)",
      },
    },
  },
  plugins: [],
};

export default config;
