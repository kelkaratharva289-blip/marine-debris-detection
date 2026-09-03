"use client";

interface PipelineStep {
  label: string;
  state: "pending" | "active" | "done" | "error";
}

const STEP_LABELS = [
  "Uploading sonar",
  "Preprocessing",
  "AI inference",
  "Anomaly analysis",
  "Risk scoring",
  "Geotagging",
  "Saving to database",
];

interface PipelineProgressProps {
  step?: number; // current active step index (0-based), -1 = done
  error?: boolean;
}

export default function PipelineProgress({
  step = 0,
  error = false,
}: PipelineProgressProps) {
  const activeIdx = Math.max(0, Math.min(step, STEP_LABELS.length - 1));
  const done = step >= STEP_LABELS.length;
  const steps: PipelineStep[] = STEP_LABELS.map((label, i) => {
    let state: PipelineStep["state"] = "pending";
    if (error) {
      if (i === activeIdx) state = "error";
      else if (i < activeIdx) state = "done";
    } else if (done) {
      state = "done";
    } else if (i < activeIdx) {
      state = "done";
    } else if (i === activeIdx) {
      state = "active";
    }
    return { label, state };
  });

  return (
    <div className="rounded-lg border border-white/[0.07] bg-abyss-900/60 px-4 py-4">
      <p className="text-xs font-semibold uppercase tracking-widest text-cyan-400">
        {error
          ? "Analysis interrupted"
          : done
            ? "Analysis complete"
            : "Analyzing sonar scan"}
      </p>
      <ol className="mt-3 space-y-2">
        {steps.map((s) => (
          <li key={s.label} className="flex items-center gap-2.5">
            <StepIcon state={s.state} />
            <span
              className={`text-sm transition-colors ${
                s.state === "active"
                  ? "font-medium text-slate-100"
                  : s.state === "done"
                    ? "text-slate-300"
                    : "text-slate-600"
              }`}
            >
              {s.label}
            </span>
            {s.state === "active" && !error && (
              <span className="ml-auto h-3.5 w-3.5 animate-spin rounded-full border-2 border-cyan-400/30 border-t-cyan-400" />
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

function StepIcon({ state }: { state: PipelineStep["state"] }) {
  if (state === "done") {
    return (
      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-400/15 text-emerald-300">
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M20 6 9 17l-5-5" />
        </svg>
      </span>
    );
  }
  if (state === "error") {
    return (
      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-red-400/15 text-red-300">
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
        >
          <path d="M18 6 6 18M6 6l12 12" />
        </svg>
      </span>
    );
  }
  if (state === "active") {
    return (
      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-cyan-400/15 text-cyan-300">
        <svg
          width="10"
          height="10"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <circle cx="12" cy="12" r="6" />
        </svg>
      </span>
    );
  }
  return (
    <span className="flex h-5 w-5 items-center justify-center rounded-full border border-white/10 text-slate-600">
      <span className="h-1.5 w-1.5 rounded-full bg-slate-600" />
    </span>
  );
}
