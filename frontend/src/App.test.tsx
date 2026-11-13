import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { Report } from "./types";

const base: Report = {
  label: "Anomaly", confidence: 0.93, p_anomaly: 0.93, threshold: 0.5, cause: "A DataNode dropped a block.",
  suggested_fix: ["Check DataNode status", "Re-replicate the block"], diagnosis_error: null, timestamp: "2026-01-01T00:00:00+00:00",
  latency_ms: { classify: 1200, diagnose: 900, total: 2100 }, llm_usage: null, snippet: "INFO a: b", truncated: false,
  advisory: "Advisory only. Nothing was executed.",
};

function mockFetch(analyze: () => Response | Promise<Response>, evaluation?: () => Response) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.includes("/api/samples")) return new Response("[]", { status: 200 });
    if (url.includes("/api/evaluation")) return evaluation ? evaluation() : new Response(JSON.stringify({ detail: "No evaluation results yet." }), { status: 404 });
    return analyze();
  }));
}

describe("Analyze page", () => {
  beforeEach(() => vi.unstubAllGlobals());
  afterEach(() => vi.unstubAllGlobals());

  it("disables the button until there is text", () => {
    mockFetch(() => new Response(JSON.stringify(base)));
    render(<App />);
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Log text"), { target: { value: "some log" } });
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeEnabled();
  });

  it("shows label, cause, remediation, advisory notice and JSON for an anomaly", async () => {
    mockFetch(() => new Response(JSON.stringify(base)));
    render(<App />);
    fireEvent.change(screen.getByLabelText("Log text"), { target: { value: "some log" } });
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    expect(await screen.findByText("A DataNode dropped a block.")).toBeInTheDocument();
    expect(screen.getAllByText("Anomaly").length).toBeGreaterThan(0);
    expect(screen.getByText("Re-replicate the block")).toBeInTheDocument();
    expect(screen.getByText("Advisory only. Nothing was executed.")).toBeInTheDocument();
    expect(screen.getByText(/93.0%/)).toBeInTheDocument();
  });

  it("shows no diagnosis section for a normal log", async () => {
    mockFetch(() => new Response(JSON.stringify({ ...base, label: "Normal", cause: null, suggested_fix: null, latency_ms: { classify: 900, diagnose: null, total: 900 } })));
    render(<App />);
    fireEvent.change(screen.getByLabelText("Log text"), { target: { value: "some log" } });
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    await waitFor(() => expect(screen.getAllByText("Normal").length).toBeGreaterThan(0));
    expect(screen.queryByText("Probable root cause")).not.toBeInTheDocument();
  });

  it("shows the diagnosis error when the LLM failed but keeps the classification", async () => {
    mockFetch(() => new Response(JSON.stringify({ ...base, cause: null, suggested_fix: null, diagnosis_error: "HTTP 429" })));
    render(<App />);
    fireEvent.change(screen.getByLabelText("Log text"), { target: { value: "some log" } });
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    expect(await screen.findByText(/Diagnosis unavailable: HTTP 429/)).toBeInTheDocument();
  });

  it("shows the backend's error message", async () => {
    mockFetch(() => new Response(JSON.stringify({ detail: "No model found in models/merged" }), { status: 503 }));
    render(<App />);
    fireEvent.change(screen.getByLabelText("Log text"), { target: { value: "some log" } });
    fireEvent.click(screen.getByRole("button", { name: "Run analysis" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No model found in models/merged");
  });
});

describe("Evaluation page", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("explains when no results exist yet", async () => {
    mockFetch(() => new Response("{}"));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Evaluation" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No evaluation results yet.");
  });

  it("renders the comparison table from the API data", async () => {
    const metrics = { accuracy: 0.9, precision: 0.5, recall: 0.8, f1: 0.6154, confusion: { tp: 8, fp: 8, fn: 2, tn: 82 }, ci95: {} };
    const model = (key: string, name: string) => ({ key, name, n: 100, hardware: "T4", precision: null, unparseable: { count: 0, rate: 0 },
      as_generated: metrics, latency: { median_s: 0.1, mean_s: 0.1, p95_s: 0.2 }, cost: { estimated_commercial_usd_per_1000: null, actual_spend_usd: 0, assumptions: "test" } });
    const data = { generated_at: "now", test_examples: { n: 100, anomalies: 10, anomaly_rate: 0.1 }, threshold: null, end_to_end: null, notes: ["a note"],
      models: [model("finetuned", "Fine-tuned"), model("untuned", "Untuned"), model("zeroshot", "Zero-shot")] };
    mockFetch(() => new Response("{}"), () => new Response(JSON.stringify(data)));
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Evaluation" }));
    expect(await screen.findByText("Model comparison")).toBeInTheDocument();
    expect(screen.getAllByText(/90\.0%/).length).toBeGreaterThan(0);
    expect(screen.getByText("a note")).toBeInTheDocument();
  });
});