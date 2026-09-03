"use client";

import dynamic from "next/dynamic";
import { Detection, ScanListItem } from "@/lib/types";

const LeafletMap = dynamic(() => import("./LeafletMap"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full min-h-[480px] items-center justify-center text-sm text-slate-500">
      Loading map...
    </div>
  ),
});

interface ScanMapProps {
  className?: string;
  detections?: Detection[];
  scans?: ScanListItem[];
  onSelectScan?: (scan: ScanListItem) => void;
  onSelectDetection?: (id: string) => void;
  selectedDetectionId?: string | null;
}

export default function ScanMap({
  className = "",
  detections,
  scans,
  onSelectScan,
  onSelectDetection,
  selectedDetectionId,
}: ScanMapProps) {
  return (
    <div className={`h-full w-full ${className}`}>
      <LeafletMap
        detections={detections}
        scans={scans}
        onSelectScan={onSelectScan}
        onSelectDetection={onSelectDetection}
        selectedDetectionId={selectedDetectionId}
      />
    </div>
  );
}
