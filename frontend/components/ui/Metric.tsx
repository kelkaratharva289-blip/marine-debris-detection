interface MetricProps {
  label: string;
  value: string | number;
  accent?: "cyan" | "emerald" | "amber" | "red" | "slate";
  hint?: string;
  progress?: number;
}

const VALUE_ACCENT: Record<string, string> = {
  cyan: "text-cyan-300",
  emerald: "text-emerald-300",
  amber: "text-amber-300",
  red: "text-red-300",
  slate: "text-slate-100",
};

const BAR_ACCENT: Record<string, string> = {
  cyan: "bg-cyan-400",
  emerald: "bg-emerald-400",
  amber: "bg-amber-400",
  red: "bg-red-400",
  slate: "bg-slate-400",
};

export default function Metric({
  label,
  value,
  accent = "cyan",
  hint,
  progress,
}: MetricProps) {
  return (
    <div className="glass-subtle rounded-md p-4">
      <div className="flex items-baseline justify-between">
        <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
          {label}
        </p>
      </div>
      <p className={`mt-1.5 text-2xl font-bold tabular-nums ${VALUE_ACCENT[accent]}`}>
        {value}
      </p>
      {hint && <p className="mt-0.5 text-xs text-slate-500">{hint}</p>}
      {typeof progress === "number" && (
        <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
          <div
            className={`h-full rounded-full ${BAR_ACCENT[accent]} transition-all duration-300`}
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>
      )}
    </div>
  );
}
