import PageShell from "@/components/layout/PageShell";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <nav className="sticky top-0 z-50 border-b border-white/[0.06] bg-abyss-950/70 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md border border-cyan-400/25 bg-cyan-400/10 text-sm font-extrabold text-cyan-300">
              SS
            </div>
            <div className="leading-tight">
              <span className="block text-lg font-bold tracking-tight text-slate-50">
                SonicSweep
              </span>
              <span className="block text-[10px] font-medium uppercase tracking-[0.2em] text-slate-500">
                Marine Intelligence
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <a href="/dashboard" className="btn-ghost">
              Dashboard
            </a>
            <a href="/upload" className="btn-primary">
              Upload
            </a>
          </div>
        </div>
      </nav>

      <main className="flex flex-1 flex-col items-center justify-center px-6 py-24 text-center">
        <p className="mb-4 inline-flex items-center gap-2 rounded-md border border-cyan-400/20 bg-cyan-400/[0.06] px-3 py-1 text-xs font-medium uppercase tracking-[0.2em] text-cyan-300">
          <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
          Deep-Sea Intelligence Platform
        </p>

        <h1 className="text-balance text-4xl font-bold tracking-tight text-slate-50 sm:text-5xl md:text-6xl">
          Scan the sea.
          <br />
          <span className="bg-gradient-to-r from-cyan-300 to-marine-400 bg-clip-text text-transparent">
            Protect the deep.
          </span>
        </h1>

        <p className="mt-6 max-w-2xl text-balance text-base text-slate-400 sm:text-lg">
          SonicSweep combines side-scan sonar imagery with AI to map marine debris
          and detect underwater anomalies — bringing clarity to the depths.
        </p>

        <div className="mt-10 flex flex-col gap-3 sm:flex-row">
          <a href="/dashboard" className="btn-primary px-6 py-3 text-base">
            Open Dashboard
          </a>
          <a href="/upload" className="btn-ghost px-6 py-3 text-base">
            Upload Imagery
          </a>
        </div>

        <div className="mt-16 grid w-full max-w-3xl grid-cols-1 gap-4 sm:grid-cols-3">
          {[
            {
              title: "AI Detection",
              body: "YOLO-powered debris and anomaly classification.",
            },
            {
              title: "Geospatial",
              body: "PostGIS-backed location intelligence on live maps.",
            },
            {
              title: "Operational",
              body: "Streamlined uploads, processing, and review.",
            },
          ].map((f) => (
            <div
              key={f.title}
              className="glass rounded-lg p-5 text-left"
            >
              <p className="text-sm font-semibold text-slate-100">{f.title}</p>
              <p className="mt-1 text-sm text-slate-400">{f.body}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
