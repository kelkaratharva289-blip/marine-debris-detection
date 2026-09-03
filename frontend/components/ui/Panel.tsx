interface PanelProps {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}

export default function Panel({
  title,
  subtitle,
  icon,
  actions,
  children,
  className = "",
  bodyClassName = "",
}: PanelProps) {
  return (
    <section className={`glass flex flex-col overflow-hidden rounded-lg ${className}`}>
      <header className="flex items-center justify-between border-b border-white/[0.06] px-5 py-3">
        <div className="flex items-center gap-2.5">
          {icon && <span className="text-cyan-400">{icon}</span>}
          <div>
            <h2 className="text-sm font-semibold text-slate-100">{title}</h2>
            {subtitle && (
              <p className="text-xs text-slate-500">{subtitle}</p>
            )}
          </div>
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </header>
      <div className={`p-5 ${bodyClassName}`}>{children}</div>
    </section>
  );
}
