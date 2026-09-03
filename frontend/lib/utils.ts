import { API_BASE_URL } from "./api";

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function severityLabel(severity: string): string {
  const labels: Record<string, string> = {
    high: "High",
    medium: "Medium",
    low: "Low",
  };
  return labels[severity] ?? "Unknown";
}

export function severityColor(severity: string): string {
  const colors: Record<string, string> = {
    high: "#ef4444",
    medium: "#f59e0b",
    low: "#22c55e",
  };
  return colors[severity] ?? "#6b7280";
}

export function classLabel(label: string): string {
  return label.replace(/_/g, " ");
}

export function averageConfidence(detections: { confidence: number }[]): number {
  if (detections.length === 0) return 0;
  const sum = detections.reduce((n, d) => n + d.confidence, 0);
  return sum / detections.length;
}

export function riskScore(confidence: number, severity: string): number {
  const severityWeight =
    severity === "high" ? 1 : severity === "medium" ? 0.6 : 0.25;
  // Confidence (0-1) combined with severity weight, scaled 0-100
  return Math.round((confidence * 0.7 + severityWeight * 0.3) * 100);
}

export type RiskLevel = "low" | "medium" | "high" | "critical";

export function riskLevelFor(score: number): RiskLevel {
  if (score >= 90) return "critical";
  if (score >= 75) return "high";
  if (score >= 50) return "medium";
  return "low";
}

export const RISK_LEVEL_COLOR: Record<RiskLevel, string> = {
  low: "emerald",
  medium: "amber",
  high: "red",
  critical: "red",
};

// Hex colors used to style map markers by risk level (Low → Critical).
export const RISK_LEVEL_HEX: Record<RiskLevel, string> = {
  low: "#22c55e",
  medium: "#f59e0b",
  high: "#fb923c",
  critical: "#ef4444",
};

// Map marker color for a detection, based on its (backend or client) risk.
export function riskHexFor(d: DetectionRiskLike): string {
  return RISK_LEVEL_HEX[riskLevelFor(detectionRiskScore(d))];
}

export interface DetectionRiskLike {
  confidence: number;
  severity: string;
  risk_score?: number | null;
}

// Prefer the backend-computed risk score when available; fall back to the
// client heuristic so the UI still works for legacy/foreign data.
export function detectionRiskScore(d: {
  risk_score?: number | null;
  confidence: number;
  severity: string;
}): number {
  if (d.risk_score != null && isFinite(d.risk_score)) {
    return d.risk_score;
  }
  return riskScore(d.confidence, d.severity);
}

export function averageRiskScore(
  detections: { confidence: number; severity: string; risk_score?: number | null }[]
): number {
  if (detections.length === 0) return 0;
  const total = detections.reduce(
    (n, d) => n + detectionRiskScore(d),
    0
  );
  return Math.round(total / detections.length);
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function downloadReport(
  format: "json" | "csv",
  scan: { id: string; name: string; created_at: string },
  detections: DetectionLike[]
) {
  if (format === "json") {
    const payload = {
      scan: {
        id: scan.id,
        name: scan.name,
        created_at: scan.created_at,
        detection_count: detections.length,
      },
      detections: detections.map((d) => ({
        object_type: d.class_label,
        size: d.mask_area ?? sizeFromBbox(d),
        confidence: d.confidence,
        risk_score: detectionRiskScore(d),
        risk_level: riskLevelFor(detectionRiskScore(d)),
        latitude: d.latitude,
        longitude: d.longitude,
        location: hasLocation(d)
          ? `${d.latitude!.toFixed(6)}, ${d.longitude!.toFixed(6)}`
          : "Location unavailable",
        detected_at: d.detected_at ?? d.created_at ?? null,
        bbox: {
          x: d.bbox_x,
          y: d.bbox_y,
          width: d.bbox_width,
          height: d.bbox_height,
        },
      })),
      summary: {
        average_confidence: Number(averageConfidence(detections).toFixed(3)),
        average_risk_score: averageRiskScore(detections),
      },
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    downloadBlob(blob, `${safeName(scan.name)}.report.json`);
    return;
  }

  const header = [
    "object_type",
    "size",
    "confidence",
    "risk_score",
    "risk_level",
    "latitude",
    "longitude",
    "location",
    "detected_at",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
  ];
  const rows = detections.map((d) => [
    csvCell(classLabel(d.class_label)),
    csvCell(d.mask_area != null ? d.mask_area.toFixed(4) : sizeFromBbox(d).toFixed(4)),
    d.confidence.toFixed(3),
    String(detectionRiskScore(d)),
    riskLevelFor(detectionRiskScore(d)),
    d.latitude != null ? d.latitude.toFixed(6) : "",
    d.longitude != null ? d.longitude.toFixed(6) : "",
    csvCell(
      hasLocation(d)
        ? `${d.latitude!.toFixed(6)}, ${d.longitude!.toFixed(6)}`
        : "Location unavailable"
    ),
    csvCell(d.detected_at ?? d.created_at ?? ""),
    d.bbox_x.toFixed(4),
    d.bbox_y.toFixed(4),
    d.bbox_width.toFixed(4),
    d.bbox_height.toFixed(4),
  ]);
  const csv = [header.join(","), ...rows.map((r) => r.join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  downloadBlob(blob, `${safeName(scan.name)}.report.csv`);
}

// Quote and escape a CSV cell when it contains delimiters, quotes, or newlines.
function csvCell(value: string): string {
  if (/[",\n\r]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

export function safeName(name: string): string {
  return (name || "scan").replace(/[^a-z0-9]+/gi, "_").toLowerCase();
}

export function imageUrlForScan(scanId: string): string | undefined {
  return `${API_BASE_URL}/scans/${scanId}/image`;
}

// Return true only when a real, valid coordinate pair exists.
export function hasLocation(d: {
  latitude?: number | null;
  longitude?: number | null;
}): boolean {
  return (
    d.latitude != null &&
    d.longitude != null &&
    isFinite(d.latitude) &&
    isFinite(d.longitude)
  );
}

// Format a coordinate pair, or "Location unavailable" when no GPS exists.
export function formatLocation(d: {
  latitude?: number | null;
  longitude?: number | null;
}): string {
  if (!hasLocation(d)) return "Location unavailable";
  return `${d.latitude!.toFixed(5)}, ${d.longitude!.toFixed(5)}`;
}

export function formatCoordinate(value: number | null | undefined, axis: "N" | "S" | "E" | "W"): string {
  if (value == null || !isFinite(value)) return "—";
  return `${value.toFixed(5)}° ${axis}`;
}

export interface DetectionLike {
  class_label: string;
  confidence: number;
  severity: string;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  mask_area?: number | null;
  latitude?: number | null;
  longitude?: number | null;
  detected_at?: string | null;
  created_at?: string | null;
}

// Normalised bbox area as a proxy "object size" when no mask area is set.
export function sizeFromBbox(d: {
  bbox_width: number;
  bbox_height: number;
}): number {
  return d.bbox_width * d.bbox_height;
}
