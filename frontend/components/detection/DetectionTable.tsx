"use client";

import { Detection } from "@/lib/types";
import {
  classLabel,
  detectionRiskScore,
  formatDate,
  hasLocation,
  riskLevelFor,
  severityLabel,
} from "@/lib/utils";
import Badge from "@/components/ui/Badge";
import AnomalyBadge from "@/components/detection/AnomalyBadge";

const RISK_TEXT: Record<string, string> = {
  low: "text-emerald-300",
  medium: "text-amber-300",
  high: "text-red-300",
  critical: "text-red-300",
};

interface DetectionTableProps {
  detections: Detection[];
  selectedId?: string | null;
  onSelect?: (id: string) => void;
}

export default function DetectionTable({
  detections,
  selectedId,
  onSelect,
}: DetectionTableProps) {
  if (detections.length === 0) {
    return null;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-left text-sm">
        <thead className="text-xs uppercase tracking-wide text-slate-500">
          <tr className="border-b border-white/[0.06]">
            <th className="py-2.5 pr-3 font-medium">Object</th>
            <th className="px-3 py-2.5 font-medium">Class</th>
            <th className="px-3 py-2.5 font-medium">Mask</th>
            <th className="px-3 py-2.5 font-medium">Confidence</th>
            <th className="px-3 py-2.5 font-medium">Anomaly</th>
            <th className="px-3 py-2.5 font-medium">Risk</th>
            <th className="px-3 py-2.5 font-medium">Severity</th>
            <th className="px-3 py-2.5 font-medium">Location</th>
            <th className="px-3 py-2.5 text-right font-medium">Detected</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.04]">
          {detections.map((det, idx) => {
            const risk = detectionRiskScore(det);
            const riskLevel = riskLevelFor(risk);
            return (
              <tr
                key={det.id}
                onClick={() => onSelect?.(det.id)}
                className={`cursor-pointer transition-colors duration-150 ${
                  selectedId === det.id
                    ? "bg-cyan-400/[0.08]"
                    : "hover:bg-white/[0.03]"
                }`}
              >
                <td className="py-2.5 pr-3 text-xs tabular-nums text-slate-500">
                  #{idx + 1}
                </td>
                <td className="px-3 py-2.5 font-medium capitalize text-slate-100">
                  {classLabel(det.class_label)}
                </td>
                <td className="px-3 py-2.5">
                  {det.mask_polygon && det.mask_polygon.length > 0 ? (
                    <span
                      title={`Masked area ${(det.mask_area ?? 0) * 100}%`}
                      className="inline-flex items-center gap-1 rounded-sm bg-cyan-400/10 px-1.5 py-0.5 text-[10px] font-medium text-cyan-300"
                    >
                      <svg
                        width="10"
                        height="10"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        className="opacity-80"
                      >
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
                      </svg>
                      Segmented
                    </span>
                  ) : (
                    <span className="text-[10px] text-slate-600">—</span>
                  )}
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 overflow-hidden rounded-full bg-white/[0.06]">
                      <div
                        className="h-full rounded-full bg-cyan-400"
                        style={{ width: `${det.confidence * 100}%` }}
                      />
                    </div>
                    <span className="tabular-nums text-slate-300">
                      {(det.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="px-3 py-2.5">
                  <AnomalyBadge detection={det} />
                </td>
                <td className="px-3 py-2.5 text-xs tabular-nums">
                  <span
                    className={RISK_TEXT[riskLevel]}
                    title={riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)}
                  >
                    {risk} · {riskLevel}
                  </span>
                </td>
                <td className="px-3 py-2.5">
                  <Badge
                    label={severityLabel(det.severity)}
                    tone={
                      det.severity === "high"
                        ? "red"
                        : det.severity === "medium"
                          ? "amber"
                          : "emerald"
                    }
                    dot
                  />
                </td>
                <td className="px-3 py-2.5 text-xs">
                  {hasLocation(det) ? (
                    <span className="tabular-nums text-slate-300">
                      {det.latitude!.toFixed(4)}, {det.longitude!.toFixed(4)}
                    </span>
                  ) : (
                    <span className="text-slate-500 italic">
                      Location unavailable
                    </span>
                  )}
                </td>
                <td className="py-2.5 pl-3 text-right text-xs text-slate-500">
                  {formatDate(det.created_at)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
