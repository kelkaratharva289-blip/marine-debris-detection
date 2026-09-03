"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, uploadScan, runDetection } from "@/lib/api";
import { useRouter } from "next/navigation";
import PipelineProgress from "@/components/ui/PipelineProgress";

const MAX_FILE_MB = 50;

type Phase = "idle" | "uploading" | "analyzing";

// Pipeline stages that map to the real backend flow.
const STAGES = [
  "Preprocessing",
  "AI inference",
  "Anomaly analysis",
  "Risk scoring",
  "Geotagging",
  "Saving to database",
];

function validateCoordinates(lat: string, lon: string): string | null {
  if (!lat && !lon) return null;
  if (lat === "" && lon !== "") return "Latitude is required if longitude is set.";
  if (lon === "" && lat !== "") return "Longitude is required if latitude is set.";
  const la = Number(lat);
  const lo = Number(lon);
  if (!Number.isFinite(la) || la < -90 || la > 90)
    return "Latitude must be between -90 and 90.";
  if (!Number.isFinite(lo) || lo < -180 || lo > 180)
    return "Longitude must be between -180 and 180.";
  return null;
}

export default function UploadForm() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [locationName, setLocationName] = useState("");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [depth, setDepth] = useState("");
  const [analyze, setAnalyze] = useState(true);
  const [phase, setPhase] = useState<Phase>("idle");
  const [stage, setStage] = useState(0);
  const [error, setError] = useState("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  function validate(): string | null {
    if (!file) return "Please select a sonar image file.";
    if (file.size > MAX_FILE_MB * 1024 * 1024)
      return `File is too large. Maximum size is ${MAX_FILE_MB} MB.`;
    const coordErr = validateCoordinates(latitude, longitude);
    if (coordErr) return coordErr;
    if (depth && (Number(depth) < 0 || !Number.isFinite(Number(depth))))
      return "Depth must be a non-negative number.";
    return null;
  }

  // Progressively advance the visual pipeline stage while the backend runs
  // the detection job. The backend reports completion via the run response.
  function startPipeline() {
    setPhase("analyzing");
    setStage(0);
    let i = 0;
    const total = 4200;
    const chunk = total / STAGES.length;
    timerRef.current = setInterval(() => {
      i += 1;
      setStage(Math.min(i, STAGES.length));
      if (i >= STAGES.length && timerRef.current) clearInterval(timerRef.current);
    }, chunk);
  }

  function stopPipeline() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function resetToIdle() {
    stopPipeline();
    setPhase("idle");
    setStage(0);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }

    setError("");
    setPhase("uploading");

    try {
      const scan = await uploadScan({
        file: file!,
        name: name || undefined,
        description: description || undefined,
        location_name: locationName || undefined,
        latitude: latitude || undefined,
        longitude: longitude || undefined,
        depth: depth || undefined,
      });

      if (analyze) {
        startPipeline();
        try {
          const detections = await runDetection(scan.id);
          stopPipeline();
          setStage(STAGES.length);
          const count = detections.length;
          setError("");
          router.push(
            `/scans/${scan.id}?detected=${count}`
          );
        } catch (runErr) {
          stopPipeline();
          resetToIdle();
          setError(
            runErr instanceof ApiError
              ? runErr.message
              : "Analysis pipeline failed after upload. You can run it again from the scan page."
          );
        }
      } else {
        router.push(`/scans/${scan.id}`);
      }
    } catch (uploadErr) {
      resetToIdle();
      setError(
        uploadErr instanceof ApiError && uploadErr.status !== 500
          ? uploadErr.message
          : "Upload failed. Please check the backend connection and try again."
      );
    }
  }

  const busy = phase !== "idle";

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-xl space-y-6">
      <div className="mb-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
          New Survey
        </p>
        <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-50 sm:text-3xl">
          Upload Sonar Imagery
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Add side-scan sonar images for automated debris and anomaly detection.
        </p>
      </div>

      <div className="glass rounded-lg p-6">
        <div className="space-y-5">
          <div>
            <label className="field-label">Image file</label>
            <input
              type="file"
              accept="image/*,.tif,.tiff,.png,.jpg,.jpeg"
              disabled={busy}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full cursor-pointer rounded-md border border-dashed border-white/15 bg-abyss-900/40 px-4 py-6 text-center text-sm text-slate-300 transition-colors duration-150 file:mr-3 file:rounded-md file:border-0 file:bg-cyan-400/15 file:px-3 file:py-1.5 file:text-sm file:font-semibold file:text-cyan-300 hover:border-cyan-400/40 disabled:cursor-not-allowed disabled:opacity-60"
            />
            {file && (
              <p className="mt-2 text-xs text-slate-500">
                <span className="font-medium text-slate-300">{file.name}</span>
                {" · "}
                {(file.size / 1024 / 1024).toFixed(1)} MB
                {file.size > MAX_FILE_MB * 1024 * 1024 && (
                  <span className="ml-1 text-red-400">
                    — exceeds {MAX_FILE_MB} MB limit
                  </span>
                )}
              </p>
            )}
          </div>

          <div>
            <label className="field-label">Scan name</label>
            <input
              type="text"
              value={name}
              disabled={busy}
              onChange={(e) => setName(e.target.value)}
              className="field disabled:opacity-60"
              placeholder="e.g. Bimini Reef Transect 4"
            />
          </div>

          <div>
            <label className="field-label">Description</label>
            <textarea
              value={description}
              disabled={busy}
              onChange={(e) => setDescription(e.target.value)}
              className="field min-h-[84px] resize-y disabled:opacity-60"
              placeholder="Notes about this survey leg..."
            />
          </div>

          <div>
            <label className="field-label">Location name</label>
            <input
              type="text"
              value={locationName}
              disabled={busy}
              onChange={(e) => setLocationName(e.target.value)}
              className="field disabled:opacity-60"
              placeholder="e.g. Bimini, Bahamas"
            />
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div>
              <label className="field-label">Latitude</label>
              <input
                type="number"
                step="any"
                value={latitude}
                disabled={busy}
                onChange={(e) => setLatitude(e.target.value)}
                className="field disabled:opacity-60"
                placeholder="25.7"
              />
            </div>
            <div>
              <label className="field-label">Longitude</label>
              <input
                type="number"
                step="any"
                value={longitude}
                disabled={busy}
                onChange={(e) => setLongitude(e.target.value)}
                className="field disabled:opacity-60"
                placeholder="-79.3"
              />
            </div>
            <div>
              <label className="field-label">Depth (m)</label>
              <input
                type="number"
                step="any"
                value={depth}
                disabled={busy}
                onChange={(e) => setDepth(e.target.value)}
                className="field disabled:opacity-60"
                placeholder="15"
              />
            </div>
          </div>
        </div>
      </div>

      <label className="flex cursor-pointer items-center gap-2.5 rounded-md border border-white/[0.07] bg-abyss-900/40 px-4 py-3">
        <input
          type="checkbox"
          checked={analyze}
          onChange={(e) => setAnalyze(e.target.checked)}
          disabled={busy}
          className="h-4 w-4 accent-cyan-400"
        />
        <span className="text-sm text-slate-200">
          Analyze immediately
        </span>
        <span className="text-xs text-slate-500">
          Run the full pipeline (preprocess → AI → anomaly → risk → geotag → DB)
        </span>
      </label>

      {error && (
        <p
          role="alert"
          className="rounded-md border border-red-400/25 bg-red-400/10 px-3.5 py-2.5 text-sm text-red-300"
        >
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={busy}
        className="btn-primary w-full px-4 py-3 text-base disabled:cursor-not-allowed disabled:opacity-60"
      >
        {phase === "uploading"
          ? "Uploading..."
          : phase === "analyzing"
            ? "Analyzing..."
            : "Upload & Analyze"}
      </button>

      {phase === "analyzing" && (
        <div className="mt-2">
          <PipelineProgress step={stage} error={false} />
        </div>
      )}
    </form>
  );
}
