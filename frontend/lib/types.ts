export interface Scan {
  id: string;
  name: string;
  description?: string;
  location_name?: string;
  latitude?: number;
  longitude?: number;
  depth?: number;
  scan_area_sqm?: number;
  filename: string;
  file_path: string;
  file_size?: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface ScanListItem {
  id: string;
  name: string;
  location_name?: string;
  latitude?: number;
  longitude?: number;
  status: string;
  created_at: string;
  detection_count: number;
}

export interface Detection {
  id: string;
  scan_id: string;
  class_label: string;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  severity: string;
  annotated_image_path?: string;
  mask_polygon?: [number, number][] | null;
  mask_area?: number | null;
  mask_image_path?: string | null;
  anomaly_class?: 'natural' | 'artificial' | 'uncertain' | null;
  natural_probability?: number | null;
  artificial_probability?: number | null;
  anomaly_confidence?: number | null;
  anomaly_features?: Record<string, number | string | null> | null;
  ai_confidence?: number | null;
  final_confidence?: number | null;
  risk_score?: number | null;
  risk_level?: 'low' | 'medium' | 'high' | 'critical' | null;
  latitude?: number | null;
  longitude?: number | null;
  geo_source?: string | null;
  detected_at?: string | null;
  created_at: string;
}

// Direct, stateless image analysis (POST /detections/analyze). One entry per
// detected object, including the downstream anomaly classification + risk.
export interface DetectionAnalysisResult {
  object: string;
  object_name: string;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  anomaly_type: "natural" | "artificial" | "uncertain";
  artificial_probability?: number | null;
  risk_score?: number | null;
  risk_level?: "low" | "medium" | "high" | "critical" | null;
  latitude?: number | null;
  longitude?: number | null;
  geo_source?: string | null;
  geo_timestamp?: string | null;
}

export interface AnalyzeResponse {
  detections: DetectionAnalysisResult[];
  count: number;
  model: string;
  processed: boolean;
}
