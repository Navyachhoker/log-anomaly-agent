export interface Report {
  label: "Normal" | "Anomaly";
  confidence: number;
  p_anomaly: number;
  threshold: number;
  cause: string | null;
  suggested_fix: string[] | null;
  diagnosis_error: string | null;
  timestamp: string;
  latency_ms: { classify: number; diagnose: number | null; total: number };
  llm_usage: { calls: number; input_tokens: number; output_tokens: number; latency_s: number } | null;
  snippet: string;
  truncated: boolean;
  advisory: string;
}

export interface Sample {
  id: string;
  title: string;
  true_label: string;
  log: string;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion: { tp: number; fp: number; fn: number; tn: number };
  ci95: Record<string, [number, number]>;
}

export interface ModelResult {
  key: string;
  name: string;
  n: number;
  hardware: string | null;
  precision: string | null;
  unparseable: { count: number; rate: number };
  as_generated: Metrics;
  "scored_at_0.5"?: Metrics;
  scored_at_tuned_threshold?: Metrics & { threshold: number };
  latency: { n?: number; mean_s?: number; median_s?: number; p95_s?: number };
  cost: {
    estimated_commercial_usd_per_1000: number | null;
    actual_spend_usd: number | null;
    assumptions: string;
  };
}

export interface Evaluation {
  generated_at: string;
  test_examples: { n: number; anomalies: number; anomaly_rate: number };
  threshold: { threshold: number; tuned_on: string; val_f1_at_0_5?: number } | null;
  models: ModelResult[];
  end_to_end: Record<string, unknown> | null;
  notes: string[];
}