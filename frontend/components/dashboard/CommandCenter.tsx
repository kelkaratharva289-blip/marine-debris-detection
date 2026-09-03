"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ScanListItem, Detection } from "@/lib/types";
import { fetchDetections, fetchScans, runDetection, ApiError } from "@/lib/api";
import {
  averageConfidence,
  averageRiskScore,
  downloadReport,
  imageUrlForScan,
} from "@/lib/utils";
import Panel from "@/components/ui/Panel";
import Metric from "@/components/ui/Metric";
import Button from "@/components/ui/Button";
import EmptyState from "@/components/ui/EmptyState";
import SonarViewer from "@/components/detection/SonarViewer";
import DetectionStats from "@/components/detection/DetectionStats";
import DetectionTable from "@/components/detection/DetectionTable";
import UploadPanel from "@/components/detection/UploadPanel";
import ScanMap from "@/components/map/ScanMap";

const REFRESH_MS = 4000;

export default function CommandCenter() {
  const [scans, setScans] = useState<ScanListItem[]>([]);
  const [selected, setSelected] = useState<ScanListItem | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedDetection, setSelectedDetection] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [noticeTone, setNoticeTone] = useState<"default" | "error" | "success">(
    "default"
  );
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadScans = useCallback(async () => {
    try {
      const data = await fetchScans();
      setScans(data);
      setLoadError("");
    } catch {
      setLoadError("Could not reach the backend.");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadDetections = useCallback(async (scan: ScanListItem) => {
    try {
      const data = await fetchDetections(scan.id);
      setDetections(data);
      return data;
    } catch {
      setNotice("Could not load detections for this scan.");
      setNoticeTone("error");
      return [];
    }
  }, []);

  useEffect(() => {
    loadScans();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-select the most recent scan once the list loads.
  useEffect(() => {
    if (loading) return;
    if (scans.length === 0) return;
    if (!selected) {
      selectScan(scans[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scans, loading]);

  // Poll scans while any are processing so statuses stay live.
  useEffect(() => {
    if (scans.some((s) => s.status === "processing")) {
      pollRef.current = setInterval(() => {
        loadScans();
        if (selected) loadDetections(selected);
      }, REFRESH_MS);
    } else if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scans, selected]);

  function selectScan(scan: ScanListItem) {
    setSelected(scan);
    setSelectedDetection(null);
    setNotice("");
    loadDetections(scan);
  }

  async function handleRunDetection() {
    if (!selected) return;
    setAnalyzing(true);
    setNotice("");
    try {
      const data = await runDetection(selected.id);
      setDetections(data);
      setNotice(`${data.length} object(s) detected.`);
      setNoticeTone(data.length > 0 ? "success" : "default");
      await loadScans();
    } catch (err) {
      setNotice(
        err instanceof ApiError ? err.message : "Detection run failed."
      );
      setNoticeTone("error");
    } finally {
      setAnalyzing(false);
    }
  }

  const avgConfidence = averageConfidence(detections);
  const avgRisk = averageRiskScore(detections);
  const riskTone = avgRisk >= 75 ? "red" : avgRisk >= 50 ? "amber" : "emerald";

  const noticeClasses =
    noticeTone === "error"
      ? "border-red-400/25 bg-red-400/10 text-red-300"
      : noticeTone === "success"
        ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-300"
        : "border-white/[0.06] bg-white/[0.03] text-slate-300";

  if (loading) {
    return (
      <div className="space-y-5">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="glass h-24 animate-pulse rounded-md" aria-hidden />
          ))}
        </div>
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          <div className="space-y-5 lg:col-span-2">
            <div className="glass h-72 animate-pulse rounded-lg" aria-hidden />
            <div className="glass h-56 animate-pulse rounded-lg" aria-hidden />
          </div>
          <div className="glass h-72 animate-pulse rounded-lg" aria-hidden />
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-red-400/25 bg-red-400/10 px-6 py-16 text-center">
        <p className="text-red-300">{loadError}</p>
        <Button variant="ghost" className="mt-4" onClick={() => { setLoading(true); loadScans(); }}>
          Retry
        </Button>
      </div>
    );
  }

  if (scans.length === 0) {
    return (
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <EmptyState
            title="No scans yet"
            description="Upload your first side-scan sonar image to begin debris and anomaly detection."
            action={
              <a href="/upload" className="btn-primary">
                Upload a scan
              </a>
            }
          />
        </div>
        <div>
          <Panel
            title="Ingest"
            subtitle="Upload sonar imagery"
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 16V4" />
                <path d="m7 9 5-5 5 5" />
                <path d="M4 20h16" />
              </svg>
            }
          >
            <UploadPanel onUploaded={loadScans} />
          </Panel>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric
          label="Total Scans"
          value={scans.length}
          accent="cyan"
          hint="Sonar surveys ingested"
        />
        <Metric
          label="Detections"
          value={detections.length}
          accent="emerald"
          hint={selected ? selected.name : "Select a scan"}
        />
        <Metric
          label="Confidence"
          value={`${(avgConfidence * 100).toFixed(0)}%`}
          accent="cyan"
          progress={avgConfidence * 100}
        />
        <Metric
          label="Risk Score"
          value={`${avgRisk}/100`}
          accent={riskTone}
          progress={avgRisk}
        />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          <Panel
            title="Sonar Acquisition"
            subtitle={selected?.name ?? "No scan selected"}
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M8 3H5a2 2 0 0 0-2 2v3" />
                <path d="M16 3h3a2 2 0 0 1 2 2v3" />
                <path d="M8 21H5a2 2 0 0 1-2-2v-3" />
                <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            }
            actions={
              <Button
                onClick={handleRunDetection}
                disabled={!selected || analyzing}
                loading={analyzing}
                size="sm"
              >
                Run Detection
              </Button>
            }
          >
            <SonarViewer
              imageUrl={selected ? imageUrlForScan(selected.id) : undefined}
              scanName={selected?.name}
              detections={detections}
              selectedDetectionId={selectedDetection}
              onSelectDetection={setSelectedDetection}
            />
            {notice && (
              <p className={`mt-3 rounded-md border px-3 py-2 text-xs ${noticeClasses}`}>
                {notice}
              </p>
            )}
          </Panel>

          <Panel
            title="Detection Statistics"
            subtitle="AI analysis summary"
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 3v16a2 2 0 0 0 2 2h16" />
                <path d="m19 9-5 5-4-4-3 3" />
              </svg>
            }
          >
            <DetectionStats detections={detections} />
          </Panel>

          <Panel
            title="Detected Objects"
            subtitle={`${detections.length} object(s) in current scan`}
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="4" width="18" height="16" rx="2" />
                <path d="M3 10h18" />
              </svg>
            }
            bodyClassName="p-0"
          >
            {detections.length === 0 ? (
              <div className="p-5">
                <EmptyState
                  title="No detections in this scan"
                  description="Run the AI model to identify debris and anomalies."
                  action={
                    <Button
                      onClick={handleRunDetection}
                      disabled={analyzing}
                      loading={analyzing}
                      size="sm"
                    >
                      Run Detection
                    </Button>
                  }
                />
              </div>
            ) : (
              <DetectionTable
                detections={detections}
                selectedId={selectedDetection}
                onSelect={setSelectedDetection}
              />
            )}
          </Panel>
        </div>

        <div className="space-y-5">
          <Panel
            title="Operational Map"
            subtitle="Scan locations"
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z" />
                <circle cx="12" cy="10" r="3" />
              </svg>
            }
            bodyClassName="p-2"
          >
            <div className="overflow-hidden rounded-md">
              <ScanMap
                className="min-h-[300px]"
                detections={detections}
                scans={scans}
                selectedDetectionId={selectedDetection}
                onSelectDetection={setSelectedDetection}
                onSelectScan={(scan) => selectScan(scan)}
              />
            </div>
          </Panel>

          <Panel
            title="Ingest"
            subtitle="Upload sonar imagery"
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 16V4" />
                <path d="m7 9 5-5 5 5" />
                <path d="M4 20h16" />
              </svg>
            }
          >
            <UploadPanel onUploaded={loadScans} />
          </Panel>

          <Panel
            title="Export Report"
            subtitle="Download analysis data"
            icon={
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <path d="M7 10l5 5 5-5" />
                <path d="M12 15V3" />
              </svg>
            }
          >
            <div className="space-y-3">
              <div className="rounded-md bg-white/[0.03] px-3 py-2 text-xs text-slate-400">
                {selected
                  ? `Scan: ${selected.name} · ${detections.length} objects`
                  : "Select a scan to enable export."}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!selected}
                  onClick={() =>
                    selected && downloadReport("json", selected, detections)
                  }
                >
                  JSON
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={!selected}
                  onClick={() =>
                    selected && downloadReport("csv", selected, detections)
                  }
                >
                  CSV
                </Button>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
