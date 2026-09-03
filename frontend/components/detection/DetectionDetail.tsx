"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Scan, Detection } from "@/lib/types";
import { ApiError, fetchScan, fetchDetections, runDetection } from "@/lib/api";
import {
  classLabel,
  detectionRiskScore,
  downloadReport,
  formatDate,
  formatFileSize,
  hasLocation,
  riskLevelFor,
  severityColor,
  severityLabel,
} from "@/lib/utils";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import AnomalyBadge from "@/components/detection/AnomalyBadge";
import PipelineProgress from "@/components/ui/PipelineProgress";

export default function DetectionDetail({ scanId }: { scanId: string }) {
  const searchParams = useSearchParams();
  const [scan, setScan] = useState<Scan | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [running, setRunning] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(0);
  const [runError, setRunError] = useState("");
  const [successNotice, setSuccessNotice] = useState("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // Surface a one-time success notice when arriving from the upload flow
  // (e.g. `?detected=3`).
  useEffect(() => {
    const detected = searchParams.get("detected");
    if (detected !== null && Number(detected) > 0) {
      setSuccessNotice(`Analysis complete — ${detected} object(s) detected.`);
    }
  }, [searchParams]);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setLoadError("");
      setRunError("");
      try {
        const [scanRes, detRes] = await Promise.all([
          fetchScan(scanId),
          fetchDetections(scanId),
        ]);
        setScan(scanRes);
        setDetections(detRes);
      } catch {
        setLoadError(
          "Failed to reach the backend. Check the connection and retry."
        );
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [scanId]);

  const stopPipeline = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  async function runDetectionNow() {
    setRunning(true);
    setRunError("");
    setSuccessNotice("");
    setPipelineStep(0);
    const steps = ["preprocess", "infer", "anomaly", "risk", "geotag", "save"];
    let i = 0;
    const chunk = 500;
    timerRef.current = setInterval(() => {
      i += 1;
      setPipelineStep(Math.min(i, steps.length));
      if (i >= steps.length && timerRef.current) clearInterval(timerRef.current);
    }, chunk);
    try {
      const data = await runDetection(scanId);
      stopPipeline();
      setPipelineStep(steps.length);
      setDetections(data);
      setSuccessNotice(`${data.length} object(s) detected.`);
    } catch (err) {
      stopPipeline();
      setRunError(
        err instanceof ApiError
          ? err.message
          : "The AI detection run failed. Check the backend and try again."
      );
      setPipelineStep(0);
    } finally {
      setRunning(false);
    }
  }

  if (loading) {
    return (
      <div className="mx-auto w-full max-w-4xl">
        <div className="space-y-6">
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-9 w-72" />
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-24" />
            ))}
          </div>
          <Skeleton className="h-72" />
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="py-16 text-center">
        <p className="text-red-300">{loadError}</p>
        <Link href="/scans" className="btn-ghost mt-4">
          Back to scans
        </Link>
      </div>
    );
  }

  if (!scan) {
    return (
      <div className="py-16 text-center">
        <p className="text-slate-400">Scan not found.</p>
        <Link href="/scans" className="btn-ghost mt-4">
          Back to scans
        </Link>
      </div>
    );
  }

  const severitySummary = countSeverity(detections);

  return (
    <div className="mx-auto w-full max-w-4xl">
      <Link
        href="/scans"
        className="inline-flex items-center gap-1 text-sm text-slate-400 transition-colors hover:text-cyan-300"
      >
        <span aria-hidden>←</span> Back to scans
      </Link>

      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-50 sm:text-3xl">
            {scan.name}
          </h1>
          {scan.description && (
            <p className="mt-1 max-w-xl text-sm text-slate-400">
              {scan.description}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {detections.length > 0 && (
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => downloadReport("json", scan, detections)}
              >
                JSON
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => downloadReport("csv", scan, detections)}
              >
                CSV
              </Button>
            </div>
          )}
        </div>
        <Button
          onClick={runDetectionNow}
          disabled={running}
          loading={running}
        >
          {running ? (
            "Analyzing"
          ) : (
            <>
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="11" cy="11" r="7" />
                <path d="m21 21-4.3-4.3" />
              </svg>
              Run AI Detection
            </>
          )}
        </Button>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <DetailCard label="Location" value={scan.location_name ?? "—"} />
        <DetailCard
          label="Coordinates"
          value={
            scan.latitude != null && scan.longitude != null
              ? `${scan.latitude.toFixed(3)}, ${scan.longitude.toFixed(3)}`
              : "—"
          }
        />
        <DetailCard label="Depth" value={scan.depth ? `${scan.depth} m` : "—"} />
        <DetailCard label="File" value={scan.file_size ? formatFileSize(scan.file_size) : scan.filename} />
      </div>

      {successNotice && (
        <p className="mt-6 rounded-md border border-emerald-400/25 bg-emerald-400/10 px-3.5 py-2.5 text-sm text-emerald-300">
          {successNotice}
        </p>
      )}

      {runError && (
        <p
          role="alert"
          className="mt-6 rounded-md border border-red-400/25 bg-red-400/10 px-3.5 py-2.5 text-sm text-red-300"
        >
          {runError}
        </p>
      )}

      <div className="mt-8 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100">
          Detections{" "}
          <span className="ml-1 rounded-md bg-white/[0.05] px-2 py-0.5 text-sm text-slate-400">
            {detections.length}
          </span>
        </h2>
        {detections.length > 0 && (
          <div className="flex gap-2">
            {(["high", "medium", "low"] as const).map((sev) => {
              const count = severitySummary[sev] ?? 0;
              if (count === 0) return null;
              return (
                <Badge
                  key={sev}
                  label={`${severityLabel(sev)} ${count}`}
                  tone={
                    sev === "high" ? "red" : sev === "medium" ? "amber" : "emerald"
                  }
                  dot
                />
              );
            })}
          </div>
        )}
      </div>

      {running && (
        <div className="mt-6">
          <PipelineProgress step={pipelineStep} error={false} />
        </div>
      )}

      {detections.length === 0 ? (
        <div className="glass mt-4 rounded-lg px-6 py-12 text-center">
          <p className="text-slate-400">
            {running
              ? "Analyzing — results will appear here."
              : "No detections yet. Run the AI model to analyze this scan."}
          </p>
          {!running && (
            <div className="mt-4">
              <Button onClick={runDetectionNow} loading={running}>
                Run AI Detection
              </Button>
            </div>
          )}
        </div>
      ) : (
        <div className="glass mt-4 overflow-hidden rounded-lg">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead className="border-b border-white/[0.06] text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-5 py-3 font-medium">Class</th>
                  <th className="px-5 py-3 font-medium">Confidence</th>
                  <th className="px-5 py-3 font-medium">Anomaly</th>
                  <th className="px-5 py-3 font-medium">Risk</th>
                  <th className="px-5 py-3 font-medium">Severity</th>
                  <th className="px-5 py-3 font-medium">Location</th>
                  <th className="px-5 py-3 font-medium">Segmentation</th>
                  <th className="px-5 py-3 font-medium">Detected</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {detections.map((det) => {
                  const color = severityColor(det.severity);
                  const sevTone =
                    det.severity === "high"
                      ? ("red" as const)
                      : det.severity === "medium"
                        ? ("amber" as const)
                        : ("emerald" as const);
                  return (
                    <tr
                      key={det.id}
                      className="transition-colors duration-150 hover:bg-white/[0.03]"
                    >
                      <td className="px-5 py-3.5 font-medium capitalize text-slate-100">
                        {classLabel(det.class_label)}
                      </td>
                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-3">
                          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-white/[0.06]">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${Math.round(det.confidence * 100)}%`,
                                backgroundColor: color,
                              }}
                            />
                          </div>
                          <span className="w-11 tabular-nums text-slate-300">
                            {(det.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-5 py-3.5">
                        <AnomalyBadge detection={det} />
                      </td>
                      <td className="px-5 py-3.5 text-xs tabular-nums">
                        {(() => {
                          const risk = detectionRiskScore(det);
                          const level = riskLevelFor(risk);
                          const color =
                            level === "low"
                              ? "text-emerald-300"
                              : level === "medium"
                                ? "text-amber-300"
                                : "text-red-300";
                          return (
                            <span className={color}>
                              {risk}
                              <span className="text-slate-500"> · {level}</span>
                            </span>
                          );
                        })()}
                      </td>
                      <td className="px-5 py-3.5">
                        <Badge
                          label={severityLabel(det.severity)}
                          tone={sevTone}
                          dot
                        />
                      </td>
                      <td className="px-5 py-3.5 text-xs">
                        {hasLocation(det) ? (
                          <span className="tabular-nums text-slate-300">
                            {det.latitude!.toFixed(5)}, {det.longitude!.toFixed(5)}
                          </span>
                        ) : (
                          <span className="text-slate-500 italic">
                            Location unavailable
                          </span>
                        )}
                      </td>
                      <td className="px-5 py-3.5">
                        {det.mask_polygon && det.mask_polygon.length > 0 ? (
                          <span className="text-xs font-medium text-cyan-300">
                            Masked
                            {det.mask_area != null
                              ? ` (${(det.mask_area * 100).toFixed(0)}%)`
                              : ""}
                          </span>
                        ) : (
                          <span className="text-xs text-slate-600">—</span>
                        )}
                      </td>
                      <td className="px-5 py-3.5 text-slate-400">
                        {formatDate(det.created_at)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function DetailCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass-subtle rounded-md p-3.5">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className="mt-1 truncate text-sm font-medium text-slate-200">{value}</p>
    </div>
  );
}

function countSeverity(detections: Detection[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const d of detections) {
    counts[d.severity] = (counts[d.severity] ?? 0) + 1;
  }
  return counts;
}

function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse rounded-md bg-white/[0.05] ${className}`}
      aria-hidden
    />
  );
}
