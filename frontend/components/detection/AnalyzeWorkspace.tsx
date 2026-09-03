"use client";

import { useEffect, useState } from "react";
import { ApiError, analyzeImage } from "@/lib/api";
import {
  DetectionAnalysisResult,
  AnalyzeResponse,
  Detection,
} from "@/lib/types";
import { riskLevelFor, hasLocation, formatLocation } from "@/lib/utils";
import Badge, { BadgeProps } from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import SonarViewer from "@/components/detection/SonarViewer";
import ScanMap from "@/components/map/ScanMap";

type Phase = "idle" | "running" | "done" | "error";

// Per-object severity used only as a display accent (Low/Medium/High) inferred
// from the backend-provided risk level so the overlay keeps its tone coding.
function severityFromRisk(level?: string | null): string {
  if (level === "critical" || level === "high") return "high";
  if (level === "medium") return "medium";
  return "low";
}

// Map the stateless analysis result onto the shape SonarViewer already renders
// (bbox + class label + confidence + anomaly class). Only real backend values
// are carried through.
function toViewerDetection(d: DetectionAnalysisResult, index: number): Detection {
  return {
    id: `analyze-${index}`,
    scan_id: "",
    class_label: d.object_name || d.object,
    confidence: d.confidence,
    bbox_x: d.bbox_x,
    bbox_y: d.bbox_y,
    bbox_width: d.bbox_width,
    bbox_height: d.bbox_height,
    severity: severityFromRisk(d.risk_level),
    anomaly_class: d.anomaly_type,
    artificial_probability: d.artificial_probability,
    risk_score: d.risk_score,
    risk_level: d.risk_level,
    latitude: d.latitude,
    longitude: d.longitude,
    geo_source: d.geo_source,
    detected_at: d.geo_timestamp,
    created_at: new Date().toISOString(),
  };
}

function riskTone(level?: string | null): BadgeProps["tone"] {
  if (level === "critical" || level === "high") return "red";
  if (level === "medium") return "amber";
  return "emerald";
}

export default function AnalyzeWorkspace() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [response, setResponse] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function handleFile(next: File | null) {
    setFile(next);
    setResponse(null);
    setError("");
    setPhase("idle");
  }

  async function run() {
    if (!file) return;
    setError("");
    setPhase("running");
    try {
      const res = await analyzeImage(file);
      setResponse(res);
      setPhase("done");
    } catch (err) {
      setPhase("error");
      setError(
        err instanceof ApiError && err.status !== 502
          ? err.message
          : "Analysis failed. Check the backend connection and try again."
      );
    }
  }

  const busy = phase === "running";
  const detections = (response?.detections ?? []).map(toViewerDetection);
  const geoCount = detections.filter((d) => hasLocation(d)).length;

  return (
    <div className="mx-auto w-full max-w-5xl space-y-6">
      <div className="mb-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
          Direct Analysis
        </p>
        <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-50 sm:text-3xl">
          Analyze Sonar Image
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Run inference, anomaly classification and risk scoring on a single
          image — no scan saved to the database.
        </p>
      </div>

      <div className="glass rounded-lg p-6">
        <label className="field-label">Sonar image</label>
        <input
          type="file"
          accept="image/*,.tif,.tiff,.png,.jpg,.jpeg"
          disabled={busy}
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
          className="block w-full cursor-pointer rounded-md border border-dashed border-white/15 bg-abyss-900/40 px-4 py-6 text-center text-sm text-slate-300 transition-colors duration-150 file:mr-3 file:rounded-md file:border-0 file:bg-cyan-400/15 file:px-3 file:py-1.5 file:text-sm file:font-semibold file:text-cyan-300 hover:border-cyan-400/40 disabled:cursor-not-allowed disabled:opacity-60"
        />
        {file && (
          <p className="mt-2 text-xs text-slate-500">
            <span className="font-medium text-slate-300">{file.name}</span>
            {" · "}
            {(file.size / 1024 / 1024).toFixed(1)} MB
          </p>
        )}

        {error && (
          <p
            role="alert"
            className="mt-4 rounded-md border border-red-400/25 bg-red-400/10 px-3.5 py-2.5 text-sm text-red-300"
          >
            {error}
          </p>
        )}

        <div className="mt-4 flex items-center gap-3">
          <Button onClick={run} disabled={!file || busy} loading={busy}>
            {busy ? "Analyzing" : file ? "Analyze Image" : "Select an image"}
          </Button>
          {response && (
            <Badge
              label={`${response.count} object${response.count === 1 ? "" : "s"}`}
              tone={response.count > 0 ? "cyan" : "neutral"}
              dot
            />
          )}
        </div>
      </div>

      {phase === "error" || phase === "done" ? (
        <SonarViewer imageUrl={previewUrl} scanName={file?.name} detections={detections} />
      ) : (
        <SonarViewer imageUrl={previewUrl} scanName={file?.name} detections={[]} />
      )}

      {response && response.detections.length > 0 && (
        <div>
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-100">Geolocation</h2>
            <span className="text-xs text-slate-500">
              {geoCount > 0
                ? `${geoCount} of ${detections.length} object(s) geotagged`
                : "No GPS metadata in this image"}
            </span>
          </div>
          <div className="glass overflow-hidden rounded-lg p-2">
            <div className="overflow-hidden rounded-md">
              <ScanMap className="min-h-[320px]" detections={detections} />
            </div>
          </div>
        </div>
      )}

      {response && response.detections.length > 0 && (
        <div className="glass overflow-hidden rounded-lg">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead className="border-b border-white/[0.06] text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-5 py-3 font-medium">Object</th>
                  <th className="px-5 py-3 font-medium">Class</th>
                  <th className="px-5 py-3 font-medium">Confidence</th>
                  <th className="px-5 py-3 font-medium">Anomaly</th>
                  <th className="px-5 py-3 font-medium">Risk</th>
                  <th className="px-5 py-3 font-medium">Artificial</th>
                  <th className="px-5 py-3 font-medium">Location</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {response.detections.map((d, i) => (
                  <tr
                    key={i}
                    className="transition-colors duration-150 hover:bg-white/[0.03]"
                  >
                    <td className="px-5 py-3.5 capitalize text-slate-100">
                      {d.object_name || d.object}
                    </td>
                    <td className="px-5 py-3.5 text-slate-400">{d.object}</td>
                    <td className="px-5 py-3.5 tabular-nums text-slate-300">
                      {(d.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="px-5 py-3.5">
                      <Badge
                        label={
                          d.anomaly_type === "artificial"
                            ? "Artificial"
                            : d.anomaly_type === "natural"
                              ? "Natural"
                              : "Uncertain"
                        }
                        tone={
                          d.anomaly_type === "artificial"
                            ? "cyan"
                            : d.anomaly_type === "natural"
                              ? "emerald"
                              : "amber"
                        }
                        dot
                      />
                    </td>
                    <td className="px-5 py-3.5">
                      {d.risk_score != null ? (
                        <Badge
                          label={`${Math.round(d.risk_score)} · ${
                            d.risk_level ?? riskLevelFor(d.risk_score)
                          }`}
                          tone={riskTone(d.risk_level)}
                        />
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="px-5 py-3.5 tabular-nums text-slate-400">
                      {d.artificial_probability != null
                        ? `${(d.artificial_probability * 100).toFixed(0)}%`
                        : "—"}
                    </td>
                    <td className="px-5 py-3.5 text-xs">
                      {hasLocation(d) ? (
                        <span className="tabular-nums text-slate-300">
                          {formatLocation(d)}
                        </span>
                      ) : (
                        <span className="italic text-slate-600">
                          Location unavailable
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
