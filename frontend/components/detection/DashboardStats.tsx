"use client";

import { useEffect, useState } from "react";
import { ScanListItem } from "@/lib/types";
import { fetchScans } from "@/lib/api";
import Link from "next/link";
import Button from "@/components/ui/Button";
import StatCard from "@/components/ui/StatCard";
import PageHeader from "@/components/layout/PageHeader";
import EmptyState from "@/components/ui/EmptyState";

interface Stats {
  totalScans: number;
  completedScans: number;
  processingScans: number;
  uploadedScans: number;
  totalDetections: number;
}

const EMPTY_STATS: Stats = {
  totalScans: 0,
  completedScans: 0,
  processingScans: 0,
  uploadedScans: 0,
  totalDetections: 0,
};

export default function DashboardStats() {
  const [scans, setScans] = useState<ScanListItem[] | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");

  async function fetchStats() {
    setLoaded(false);
    setError("");
    try {
      const data = await fetchScans();
      setScans(data);
    } catch {
      setError("Could not connect to the backend.");
      setScans([]);
    } finally {
      setLoaded(true);
    }
  }

  useEffect(() => {
    fetchStats();
  }, []);

  if (!loaded) {
    return (
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="glass h-24 animate-pulse rounded-lg" aria-hidden />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-red-400/25 bg-red-400/10 px-6 py-12 text-center">
        <p className="text-red-300">{error}</p>
        <Button variant="ghost" className="mt-4" onClick={fetchStats}>
          Retry
        </Button>
      </div>
    );
  }

  const list = scans ?? [];
  const stats: Stats = {
    ...EMPTY_STATS,
    totalScans: list.length,
    completedScans: list.filter((s) => s.status === "completed").length,
    processingScans: list.filter((s) => s.status === "processing").length,
    uploadedScans: list.filter((s) => s.status === "uploaded").length,
    totalDetections: list.reduce((n, s) => n + s.detection_count, 0),
  };

  return (
    <div>
      <PageHeader
        eyebrow="Operations Overview"
        title="Dashboard"
        description="Real-time state of your sonar survey archive and detections."
      />

      {stats.totalScans === 0 ? (
        <EmptyState
          title="No scans yet"
          description="Upload your first side-scan sonar image to begin debris detection."
          action={
            <Link href="/upload" className="btn-primary">
              Upload a scan
            </Link>
          }
        />
      ) : (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <StatCard label="Total Scans" value={stats.totalScans} accent="text-cyan-300" />
          <StatCard
            label="Completed"
            value={stats.completedScans}
            accent="text-emerald-300"
            detail="Scans fully analyzed"
          />
          <StatCard
            label="Processing"
            value={stats.processingScans}
            accent="text-amber-300"
            detail="In the AI pipeline"
          />
          <StatCard
            label="Detections"
            value={stats.totalDetections}
            accent="text-marine-300"
            detail="Debris & anomalies found"
          />
        </div>
      )}
    </div>
  );
}