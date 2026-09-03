import { Detection } from "@/lib/types";
import Badge from "@/components/ui/Badge";

const TONE: Record<
  "natural" | "artificial" | "uncertain",
  "emerald" | "cyan" | "amber"
> = {
  natural: "emerald",
  artificial: "cyan",
  uncertain: "amber",
};

const LABEL: Record<
  "natural" | "artificial" | "uncertain",
  string
> = {
  natural: "Natural",
  artificial: "Artificial",
  uncertain: "Uncertain",
};

export default function AnomalyBadge({
  detection,
  showConfidence = true,
}: {
  detection: Detection;
  showConfidence?: boolean;
}) {
  const cls = detection.anomaly_class;
  if (!cls) {
    return <span className="text-xs text-slate-600">—</span>;
  }
  const label = showConfidence
    ? `${LABEL[cls]}${
        detection.anomaly_confidence != null
          ? ` · ${Math.round(detection.anomaly_confidence * 100)}%`
          : ""
      }`
    : LABEL[cls];

  return <Badge label={label} tone={TONE[cls]} dot />;
}
