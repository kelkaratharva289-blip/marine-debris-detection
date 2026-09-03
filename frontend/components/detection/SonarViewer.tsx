"use client";

import { useState } from "react";
import { Detection } from "@/lib/types";
import { classLabel, severityColor } from "@/lib/utils";

interface SonarViewerProps {
  imageUrl?: string | null;
  scanName?: string;
  detections: Detection[];
  selectedDetectionId?: string | null;
  onSelectDetection?: (id: string) => void;
}

const FALLBACK_IMAGE = "/sonar-placeholder.svg";

function polygonPoints(polygon: [number, number][]): string {
  return polygon.map(([x, y]) => `${x * 100}%,${y * 100}%`).join(" ");
}

export default function SonarViewer({
  imageUrl,
  scanName,
  detections,
  selectedDetectionId,
  onSelectDetection,
}: SonarViewerProps) {
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgError, setImgError] = useState(false);
  const [showMasks, setShowMasks] = useState(true);

  const useFallback = !imageUrl || imgError;

  const maskCount = detections.filter(
    (d) => d.mask_polygon && d.mask_polygon.length > 0
  ).length;

  const anomalyTone: Record<
    string,
    { label: string; color: string }
  > = {
    natural: { label: "Natural", color: "#34d399" },
    artificial: { label: "Artificial", color: "#22d3ee" },
    uncertain: { label: "Uncertain", color: "#fbbf24" },
  };

  return (
    <div className="relative aspect-[16/9] w-full overflow-hidden rounded-md border border-white/[0.06] bg-abyss-900">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={useFallback ? FALLBACK_IMAGE : imageUrl!}
        alt={scanName ? `${scanName} sonar scan` : "Sonar scan"}
        className="absolute inset-0 h-full w-full object-cover"
        onLoad={() => setImgLoaded(true)}
        onError={() => setImgError(true)}
      />

      {/* Segmented mask overlay layer */}
      {showMasks && (
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          {detections.map((det) => {
            if (!det.mask_polygon || det.mask_polygon.length < 3) return null;
            const color = severityColor(det.severity);
            const isSelected = selectedDetectionId === det.id;
            return (
              <polygon
                key={`mask-${det.id}`}
                points={polygonPoints(det.mask_polygon)}
                fill={color}
                fillOpacity={isSelected ? 0.45 : 0.22}
                stroke={color}
                strokeWidth={isSelected ? 0.5 : 0.25}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
      )}

      {/* Scan lines / graticule overlay for scientific feel */}
      <div
        className="pointer-events-none absolute inset-0 opacity-25"
        style={{
          backgroundImage:
            "linear-gradient(rgba(103,232,249,0.12) 1px, transparent 1px), linear-gradient(90deg, rgba(103,232,249,0.12) 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />

      {/* Mask overlay toggle */}
      {maskCount > 0 && (
        <button
          type="button"
          onClick={() => setShowMasks((v) => !v)}
          className="absolute right-2 top-2 z-10 flex items-center gap-1.5 rounded-sm bg-abyss-950/80 px-2 py-1 text-[11px] font-medium text-slate-200 shadow-sm backdrop-blur-sm transition-colors hover:bg-abyss-950"
        >
          <span
            className={`inline-block h-2 w-2 rounded-sm ${
              showMasks ? "bg-cyan-400" : "bg-slate-600"
            }`}
          />
          {showMasks ? "Masks on" : "Masks off"}
        </button>
      )}

      {/* AI detection overlay */}
        {detections.map((det) => {
          const color = severityColor(det.severity);
          const isSelected = selectedDetectionId === det.id;
          const anomaly =
            det.anomaly_class != null ? anomalyTone[det.anomaly_class] : null;
          return (
            <button
              key={det.id}
              type="button"
              onClick={() => onSelectDetection?.(det.id)}
              className="group absolute cursor-pointer bg-transparent p-0 focus:outline-none"
              style={{
                left: `${det.bbox_x * 100}%`,
                top: `${det.bbox_y * 100}%`,
                width: `${det.bbox_width * 100}%`,
                height: `${det.bbox_height * 100}%`,
              }}
            >
              <span
                className={`absolute inset-0 border transition-colors duration-150 ${
                  isSelected ? "border-2" : "border"
                }`}
                style={{
                  borderColor: color,
                  boxShadow: isSelected
                    ? `0 0 0 1px rgba(3,8,15,0.9), 0 0 14px ${color}66`
                    : "0 0 0 1px rgba(3,8,15,0.8)",
                  backgroundColor: `${color}14`,
                }}
              />
              <span
                className="absolute -top-6 left-0 flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[10px] font-semibold leading-none text-abyss-950 shadow-sm"
                style={{ backgroundColor: color }}
              >
                {classLabel(det.class_label)}
                <span
                  className="rounded-sm px-1 py-[1px] text-[9px] font-bold uppercase leading-none text-white"
                  style={{
                    backgroundColor:
                      anomaly?.color ?? "rgba(0,0,0,0.45)",
                  }}
                >
                  {anomaly?.label ?? "· " + (det.confidence * 100).toFixed(0) + "%"}
                </span>
              </span>
            </button>
          );
        })}

      {detections.length === 0 && !useFallback && (
        <div className="absolute inset-x-0 bottom-0 flex justify-center pb-3">
          <span className="rounded-sm bg-abyss-950/70 px-2.5 py-1 text-[11px] font-medium text-slate-300 backdrop-blur-sm">
            No detections — run AI analysis
          </span>
        </div>
      )}

      {useFallback && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-xs text-slate-500">
            Sonar imagery preview will appear after upload
          </p>
        </div>
      )}
    </div>
  );
}
