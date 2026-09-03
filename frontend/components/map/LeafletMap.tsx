"use client";

import { useEffect, useMemo, useState } from "react";
import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Detection, ScanListItem } from "@/lib/types";
import { fetchAllDetections, fetchScans } from "@/lib/api";
import {
  RISK_LEVEL_HEX,
  classLabel,
  detectionRiskScore,
  formatDate,
  hasLocation,
  riskHexFor,
  riskLevelFor,
} from "@/lib/utils";
import Link from "next/link";

const DEFAULT_CENTER: [number, number] = [20, -60];
const DEFAULT_ZOOM = 2.5;

const MARKER_COLORS: Record<string, string> = {
  uploaded: "#38bdf8",
  processing: "#fbbf24",
  completed: "#34d399",
};

function createMarkerIcon(riskLevel: string, color: string, selected = false) {
  const inner =
    riskLevel === "critical"
      ? `<div class="anomaly-marker anomaly-marker-critical" style="--marker:${color}"></div>`
      : riskLevel === "high"
        ? `<div class="anomaly-marker anomaly-marker-high" style="--marker:${color}"></div>`
        : riskLevel === "medium"
          ? `<div class="anomaly-marker anomaly-marker-medium" style="--marker:${color}"></div>`
          : `<div class="anomaly-marker anomaly-marker-low" style="--marker:${color}"></div>`;
  const size = selected ? 24 : 20;
  const html = `${inner}${
    selected
      ? `<div class="anomaly-marker-ring" style="--marker:${color}"></div>`
      : ""
  }`;
  return L.divIcon({
    className: "",
    html: `<div style="position:relative;width:${size}px;height:${size}px">${html}</div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

function createScanMarkerIcon(color: string) {
  return L.divIcon({
    className: "",
    html: `<div style="width:12px;height:12px;background:${color};border:2px solid rgba(255,255,255,0.9);border-radius:50%;box-shadow:0 0 0 3px rgba(56,189,248,0.15), 0 0 8px rgba(0,0,0,0.4)"></div>`,
    iconSize: [12, 12],
    iconAnchor: [6, 6],
  });
}

interface DetectionMapProps {
  detections?: Detection[];
  scans?: ScanListItem[];
  onSelectScan?: (scan: ScanListItem) => void;
  onSelectDetection?: (id: string) => void;
  selectedDetectionId?: string | null;
}

export default function LeafletMap({
  detections: detectionsProp,
  scans: scansProp,
  onSelectScan,
  onSelectDetection,
  selectedDetectionId,
}: DetectionMapProps) {
  const [scans, setScans] = useState<ScanListItem[]>([]);
  const [selected, setSelected] = useState<ScanListItem | null>(null);
  const [remoteDetections, setRemoteDetections] = useState<Detection[]>([]);
  const [mapError, setMapError] = useState("");

  const scansList = scansProp ?? scans;

  const detections = detectionsProp ?? remoteDetections;

  useEffect(() => {
    if (scansProp !== undefined) return;
    let cancelled = false;
    fetchScans()
      .then((data) => !cancelled && setScans(data))
      .catch(() => !cancelled && setMapError("Could not load scan locations."));
    return () => {
      cancelled = true;
    };
  }, [scansProp]);

  useEffect(() => {
    if (detectionsProp !== undefined) return;
    let cancelled = false;
    fetchAllDetections()
      .then((data) => !cancelled && setRemoteDetections(data))
      .catch(() => !cancelled && setMapError("Could not load detections."));
    return () => {
      cancelled = true;
    };
  }, [detectionsProp]);

  const geoDetections = useMemo(
    () => detections.filter((d) => hasLocation(d)),
    [detections]
  );

  const geoScans = scansList.filter(
    (s) => s.latitude != null && s.longitude != null
  );

  const hasGeoData = geoDetections.length > 0 || geoScans.length > 0;

  const legend = (
    <div className="bg-abyss-900/85 px-3 py-2 backdrop-blur-sm">
      <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wide text-slate-400">
        Anomaly Risk
      </p>
      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {(["low", "medium", "high", "critical"] as const).map((level) => (
          <span
            key={level}
            className="inline-flex items-center gap-1.5 text-[11px] capitalize text-slate-300"
          >
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: RISK_LEVEL_HEX[level] }}
            />
            {level}
          </span>
        ))}
      </div>
    </div>
  );

  return (
    <div className="relative flex h-full flex-col overflow-hidden rounded-lg">
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={DEFAULT_ZOOM}
        scrollWheelZoom
        style={{ height: "100%", width: "100%", minHeight: 480, background: "#071422" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {geoDetections.map((det) => {
          const risk = detectionRiskScore(det);
          const riskLevel = riskLevelFor(risk);
          const color = riskHexFor(det);
          const isSelected = det.id === selectedDetectionId;
          return (
            <Marker
              key={det.id}
              position={[det.latitude!, det.longitude!]}
              icon={createMarkerIcon(riskLevel, color, isSelected)}
              eventHandlers={{
                click: () => onSelectDetection?.(det.id),
              }}
            >
              <Popup>
                <div className="min-w-40">
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-semibold capitalize text-slate-50">
                      {classLabel(det.class_label)}
                    </p>
                    <span
                      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase"
                      style={{
                        color: color,
                        background: `${color}1a`,
                        border: `1px solid ${color}40`,
                      }}
                    >
                      <span
                        className="h-1.5 w-1.5 rounded-full"
                        style={{ background: color }}
                      />
                      {riskLevel}
                    </span>
                  </div>
                  <dl className="mt-2 space-y-1 text-xs text-slate-300">
                    <div className="flex justify-between gap-4">
                      <dt className="text-slate-500">Confidence</dt>
                      <dd className="tabular-nums">
                        {(det.confidence * 100).toFixed(1)}%
                      </dd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <dt className="text-slate-500">Risk</dt>
                      <dd className="tabular-nums font-medium">{risk}/100</dd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <dt className="text-slate-500">Artificiality</dt>
                      <dd className="tabular-nums">
                        {det.artificial_probability != null
                          ? `${(det.artificial_probability * 100).toFixed(0)}%`
                          : "—"}
                      </dd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <dt className="text-slate-500">Latitude</dt>
                      <dd className="tabular-nums">{det.latitude!.toFixed(5)}</dd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <dt className="text-slate-500">Longitude</dt>
                      <dd className="tabular-nums">{det.longitude!.toFixed(5)}</dd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <dt className="text-slate-500">Detected</dt>
                      <dd className="tabular-nums">
                        {formatDate(det.detected_at ?? det.created_at)}
                      </dd>
                    </div>
                  </dl>
                </div>
              </Popup>
            </Marker>
          );
        })}
        {geoScans.map((scan) => {
          const color =
            MARKER_COLORS[scan.status] ?? MARKER_COLORS.uploaded;
          return (
            <Marker
              key={`scan-${scan.id}`}
              position={[scan.latitude!, scan.longitude!]}
              icon={createScanMarkerIcon(color)}
              eventHandlers={{
                click: () => {
                  setSelected(scan);
                  onSelectScan?.(scan);
                },
              }}
            >
              <Popup>
                <div>
                  <p className="font-semibold text-slate-50">{scan.name}</p>
                  <p className="mt-0.5 text-slate-400">
                    {formatDate(scan.created_at)}
                  </p>
                  <p className="mt-1 flex items-center gap-1.5 text-slate-300">
                    <span
                      className="inline-block h-1.5 w-1.5 rounded-full"
                      style={{ background: color }}
                    />
                    {scan.detection_count} detection(s)
                  </p>
                  <Link
                    href={`/scans/${scan.id}`}
                    className="mt-2 inline-block text-xs font-medium text-cyan-300 hover:underline"
                  >
                    View details →
                  </Link>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>

      {!hasGeoData && !mapError && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <p className="rounded-md bg-abyss-950/80 px-3 py-2 text-xs text-slate-400 backdrop-blur-sm">
            No geotagged scans or detections to display yet
          </p>
        </div>
      )}
      {mapError && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <p className="rounded-md border border-red-400/25 bg-red-400/10 px-3 py-2 text-xs text-red-300 backdrop-blur-sm">
            {mapError}
          </p>
        </div>
      )}

      {legend}

      {selected && (
        <div className="border-t border-white/[0.06] bg-abyss-800/80 px-4 py-3 backdrop-blur-md">
          <div className="flex items-center justify-between">
            <div className="min-w-0">
              <p className="truncate font-semibold text-slate-100">
                {selected.name}
              </p>
              <p className="truncate text-xs text-slate-400">
                {selected.location_name ?? "Unknown location"} ·{" "}
                {formatDate(selected.created_at)}
              </p>
            </div>
            <Link
              href={`/scans/${selected.id}`}
              className="btn-ghost ml-4 shrink-0 px-3 py-1.5 text-xs"
            >
              View
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
