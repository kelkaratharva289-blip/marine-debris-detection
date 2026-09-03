import { Scan, ScanListItem, Detection, AnalyzeResponse } from "@/lib/types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail, res.status);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function fetchHealth(): Promise<{
  status: string;
  app: string;
  version: string;
}> {
  return request(`${API_BASE_URL}/health`);
}

// ---------------------------------------------------------------------------
// Scans
// ---------------------------------------------------------------------------

export async function fetchScans(): Promise<ScanListItem[]> {
  return request<ScanListItem[]>(`${API_BASE_URL}/scans/`);
}

export async function fetchScan(id: string): Promise<Scan> {
  return request<Scan>(`${API_BASE_URL}/scans/${id}`);
}

export interface UploadScanParams {
  file: File;
  name?: string;
  description?: string;
  location_name?: string;
  latitude?: string;
  longitude?: string;
  depth?: string;
}

export async function uploadScan(params: UploadScanParams): Promise<Scan> {
  const formData = new FormData();
  formData.append("file", params.file);
  if (params.name) formData.append("name", params.name);
  if (params.description) formData.append("description", params.description);
  if (params.location_name) formData.append("location_name", params.location_name);
  if (params.latitude) formData.append("latitude", params.latitude);
  if (params.longitude) formData.append("longitude", params.longitude);
  if (params.depth) formData.append("depth", params.depth);

  return request<Scan>(`${API_BASE_URL}/scans/upload`, {
    method: "POST",
    body: formData,
  });
}

export async function deleteScan(id: string): Promise<void> {
  return request<void>(`${API_BASE_URL}/scans/${id}`, { method: "DELETE" });
}

// ---------------------------------------------------------------------------
// Detections
// ---------------------------------------------------------------------------

export async function fetchDetections(scanId: string): Promise<Detection[]> {
  return request<Detection[]>(`${API_BASE_URL}/detections/scan/${scanId}`);
}

export async function fetchAllDetections(
  opts?: { scanId?: string; riskLevel?: string }
): Promise<Detection[]> {
  const params = new URLSearchParams();
  if (opts?.scanId) params.set("scan_id", opts.scanId);
  if (opts?.riskLevel) params.set("risk_level", opts.riskLevel);
  const qs = params.toString();
  return request<Detection[]>(
    `${API_BASE_URL}/detections/${qs ? `?${qs}` : ""}`
  );
}

export async function runDetection(scanId: string): Promise<Detection[]> {
  return request<Detection[]>(`${API_BASE_URL}/detections/run/${scanId}`, {
    method: "POST",
  });
}

export async function deleteDetection(id: string): Promise<void> {
  return request<void>(`${API_BASE_URL}/detections/${id}`, { method: "DELETE" });
}

// ---------------------------------------------------------------------------
// Stateless image analysis
// ---------------------------------------------------------------------------

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<AnalyzeResponse>(`${API_BASE_URL}/detections/analyze`, {
    method: "POST",
    body: formData,
  });
}
