"use client";

import { useEffect, useState } from "react";
import { ScanListItem } from "@/lib/types";
import { fetchScans } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import Link from "next/link";
import Badge from "@/components/ui/Badge";
import Button from "@/components/ui/Button";
import EmptyState from "@/components/ui/EmptyState";

type BadgeTone = "neutral" | "cyan" | "emerald" | "amber" | "red";

const STATUS_BADGE: Record<string, { label: string; tone: BadgeTone }> = {
  uploaded: { label: "Uploaded", tone: "cyan" },
  processing: { label: "Processing", tone: "amber" },
  completed: { label: "Completed", tone: "emerald" },
  failed: { label: "Failed", tone: "red" },
};

export default function ScansTable() {
  const [scans, setScans] = useState<ScanListItem[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    loadScans();
  }, []);

  async function loadScans() {
    try {
      const data = await fetchScans();
      setScans(data);
      setError("");
    } catch {
      setError("Could not connect to the backend.");
      setScans([]);
    }
  }

  if (scans === null) {
    return (
      <div className="space-y-3">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="glass h-16 animate-pulse rounded-lg"
            aria-hidden
          />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-red-400/25 bg-red-400/10 p-8 text-center">
        <p className="text-red-300">{error}</p>
        <Button variant="ghost" className="mt-4" onClick={loadScans}>
          Retry
        </Button>
      </div>
    );
  }

  if (scans.length === 0) {
    return (
      <EmptyState
        title="No scans yet"
        description="Upload your first side-scan sonar image to build the archive."
        action={
          <Link href="/upload" className="btn-primary">
            Upload a scan
          </Link>
        }
      />
    );
  }

  return (
    <div className="glass overflow-hidden rounded-lg">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="border-b border-white/[0.06] text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-5 py-3 font-medium">Name</th>
              <th className="px-5 py-3 font-medium">Location</th>
              <th className="px-5 py-3 font-medium">Status</th>
              <th className="px-5 py-3 text-right font-medium">Detections</th>
              <th className="px-5 py-3 font-medium">Uploaded</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/[0.04]">
            {scans.map((scan) => {
              const badge = STATUS_BADGE[scan.status] ?? STATUS_BADGE.uploaded;
              return (
                <tr
                  key={scan.id}
                  className="transition-colors duration-150 hover:bg-white/[0.03]"
                >
                  <td className="px-5 py-3.5">
                    <Link
                      href={`/scans/${scan.id}`}
                      className="font-medium text-cyan-300 hover:text-cyan-200"
                    >
                      {scan.name}
                    </Link>
                  </td>
                  <td className="px-5 py-3.5 text-slate-400">
                    {scan.location_name ?? "—"}
                  </td>
                  <td className="px-5 py-3.5">
                    <Badge
                      label={badge.label}
                      tone={badge.tone}
                      dot
                    />
                  </td>
                  <td className="px-5 py-3.5 text-right tabular-nums text-slate-300">
                    {scan.detection_count}
                  </td>
                  <td className="px-5 py-3.5 text-slate-400">
                    {formatDate(scan.created_at)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
