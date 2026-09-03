export interface BadgeProps {
  label: string;
  tone?: "neutral" | "cyan" | "emerald" | "amber" | "red";
  dot?: boolean;
}

const TONES: Record<string, string> = {
  neutral: "text-slate-400 bg-white/[0.04] border-white/[0.06]",
  cyan: "text-cyan-300 bg-cyan-400/10 border-cyan-400/20",
  emerald: "text-emerald-300 bg-emerald-400/10 border-emerald-400/20",
  amber: "text-amber-300 bg-amber-400/10 border-amber-400/20",
  red: "text-red-300 bg-red-400/10 border-red-400/20",
};

const DOTS: Record<string, string> = {
  neutral: "bg-slate-400",
  cyan: "bg-cyan-400",
  emerald: "bg-emerald-400",
  amber: "bg-amber-400",
  red: "bg-red-400",
};

export default function Badge({ label, tone = "neutral", dot = false }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium ${TONES[tone] ?? TONES.neutral}`}
    >
      {dot && (
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${
            DOTS[tone] ?? DOTS.neutral
          }`}
        />
      )}
      {label}
    </span>
  );
}
