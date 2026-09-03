"use client";

import { Detection } from "@/lib/types";
import {
  averageConfidence,
  averageRiskScore,
  classLabel,
} from "@/lib/utils";
import Metric from "@/components/ui/Metric";
import Badge from "@/components/ui/Badge";

interface DetectionStatsProps {
  detections: Detection[];
}

export default function DetectionStats({ detections }: DetectionStatsProps) {
  const avgConfidence = averageConfidence(detections);
  const avgRisk = averageRiskScore(detections);

  const severityCount = {
    high: detections.filter((d) => d.severity === "high").length,
    medium: detections.filter((d) => d.severity === "medium").length,
    low: detections.filter((d) => d.severity === "low").length,
  };

  const byClass = countByClass(detections);
  const totalByClass = Object.values(byClass).reduce((n, c) => n + c, 0);
  const maxClassCount = Math.max(1, ...Object.values(byClass));

  const riskTone =
    avgRisk >= 75
      ? "red"
      : avgRisk >= 50
        ? "amber"
        : "emerald";

  if (detections.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-md border border-dashed border-white/[0.08] bg-white/[0.02] px-4 py-8 text-center">
        <p className="text-xs text-slate-500">
          No detection data to summarize yet.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <Metric
          label="Confidence"
          value={`${(avgConfidence * 100).toFixed(1)}%`}
          accent="cyan"
          hint={`Across ${detections.length} detection(s)`}
          progress={avgConfidence * 100}
        />
        <Metric
          label="Risk Score"
          value={`${avgRisk}/100`}
          accent={riskTone}
          hint="AI + artificiality + confidence"
          progress={avgRisk}
        />
      </div>

      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
          Severity Breakdown
        </p>
        <div className="flex gap-2">
          <Badge label={`High ${severityCount.high}`} tone="red" dot />
          <Badge label={`Medium ${severityCount.medium}`} tone="amber" dot />
          <Badge label={`Low ${severityCount.low}`} tone="emerald" dot />
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
          Detection Classes
        </p>
        {totalByClass === 0 ? (
          <p className="text-xs text-slate-500">No detections to summarize.</p>
        ) : (
          <div className="space-y-2">
            {Object.entries(byClass)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, count]) => (
                <div key={cls} className="flex items-center gap-2">
                  <span className="w-24 shrink-0 truncate text-xs capitalize text-slate-300">
                    {classLabel(cls)}
                  </span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                    <div
                      className="h-full rounded-full bg-cyan-400 transition-all duration-300"
                      style={{ width: `${(count / maxClassCount) * 100}%` }}
                    />
                  </div>
                  <span className="w-6 shrink-0 text-right text-xs tabular-nums text-slate-400">
                    {count}
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}

function countByClass(detections: Detection[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const d of detections) {
    counts[d.class_label] = (counts[d.class_label] ?? 0) + 1;
  }
  return counts;
}
