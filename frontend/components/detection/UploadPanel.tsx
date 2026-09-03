"use client";

import { useRef, useState } from "react";
import { ApiError, uploadScan } from "@/lib/api";
import { formatFileSize } from "@/lib/utils";

const MAX_FILE_MB = 50;

interface UploadPanelProps {
  onUploaded?: () => void;
}

export default function UploadPanel({ onUploaded }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setError("");
    setSuccess("");

    if (f.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`File too large — maximum is ${MAX_FILE_MB} MB.`);
      if (inputRef.current) inputRef.current.value = "";
      setFile(null);
      return;
    }

    setUploading(true);
    try {
      await uploadScan({ file: f });
      setSuccess("Uploaded. Select the scan to analyze.");
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
      onUploaded?.();
    } catch (err) {
      setError(
        err instanceof ApiError && err.status !== 500
          ? err.message
          : "Upload failed — check backend connection."
      );
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="space-y-4">
      <label className="flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-white/15 bg-abyss-900/40 px-4 py-8 text-center transition-colors duration-150 hover:border-cyan-400/40">
        <input
          ref={inputRef}
          type="file"
          accept="image/*,.tif,.tiff,.png,.jpg,.jpeg"
          className="hidden"
          onChange={handleFile}
          disabled={uploading}
        />
        <svg
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-cyan-400"
        >
          <path d="M12 16V4" />
          <path d="m7 9 5-5 5 5" />
          <path d="M4 20h16" />
        </svg>
        <p className="mt-3 text-sm font-medium text-slate-200">
          {uploading ? "Uploading..." : "Drop sonar imagery to upload"}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          TIF, PNG, JPG · up to 50 MB
        </p>
      </label>

      {uploading && (
        <div className="h-1 overflow-hidden rounded-full bg-white/[0.06]">
          <div className="h-full w-1/2 animate-pulse rounded-full bg-cyan-400" />
        </div>
      )}

      {file && (
        <div className="flex items-center justify-between rounded-md bg-white/[0.03] px-3 py-2 text-xs text-slate-300">
          <span className="truncate">{file.name}</span>
          <span className="ml-2 shrink-0 text-slate-500">
            {formatFileSize(file.size)}
          </span>
        </div>
      )}

      {success && (
        <p className="rounded-md border border-emerald-400/25 bg-emerald-400/10 px-3 py-2 text-xs text-emerald-300">
          {success}
        </p>
      )}
      {error && (
        <p
          role="alert"
          className="rounded-md border border-red-400/25 bg-red-400/10 px-3 py-2 text-xs text-red-300"
        >
          {error}
        </p>
      )}

      <p className="text-[11px] leading-relaxed text-slate-500">
        Imagery is stored and queued for AI debris &amp; anomaly detection. Maps
        to the active sonar scan for overlay review.
      </p>
    </div>
  );
}