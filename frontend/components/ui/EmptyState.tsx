interface EmptyStateProps {
  title: string;
  description?: string;
  action?: React.ReactNode;
}

export default function EmptyState({
  title,
  description,
  action,
}: EmptyStateProps) {
  return (
    <div className="glass rounded-lg px-6 py-14 text-center">
      <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-md border border-cyan-400/20 bg-cyan-400/5 text-cyan-400">
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M20 12V8a6 6 0 0 0-8-5.3A6 6 0 0 0 4 8v4" />
          <rect x="2" y="12" width="20" height="8" rx="2" />
        </svg>
      </div>
      <p className="font-semibold text-slate-100">{title}</p>
      {description && (
        <p className="mx-auto mt-1 max-w-sm text-sm text-slate-400">
          {description}
        </p>
      )}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
