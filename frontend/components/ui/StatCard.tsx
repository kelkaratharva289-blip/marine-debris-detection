interface StatCardProps {
  label: string;
  value: number | string;
  accent?: string;
  icon?: React.ReactNode;
  detail?: string;
}

export default function StatCard({
  label,
  value,
  accent = "text-cyan-300",
  icon,
  detail,
}: StatCardProps) {
  return (
    <div className="glass rounded-lg p-5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          {label}
        </p>
        {icon && <span className="text-slate-500">{icon}</span>}
      </div>
      <p className={`mt-2 text-3xl font-bold tabular-nums ${accent}`}>{value}</p>
      {detail && <p className="mt-1 text-xs text-slate-500">{detail}</p>}
    </div>
  );
}
