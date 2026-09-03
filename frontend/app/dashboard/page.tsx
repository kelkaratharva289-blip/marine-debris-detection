import PageShell from "@/components/layout/PageShell";
import CommandCenter from "@/components/dashboard/CommandCenter";

export default function Dashboard() {
  return (
    <PageShell>
      <div className="mb-6">
        <div className="flex items-center gap-2.5">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
            Marine Intelligence &middot; Command Center
          </p>
        </div>
        <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-50 sm:text-3xl">
          Operational Dashboard
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Live sonar analysis, detection overlay, and geospatial operations.
        </p>
      </div>
      <CommandCenter />
    </PageShell>
  );
}
