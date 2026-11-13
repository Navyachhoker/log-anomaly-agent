import type { Evaluation, Report, Sample } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error("Cannot reach the backend. Is it running on port 8000?");
  }
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      const detail = body.detail;
      message = Array.isArray(detail) ? detail.map((d: { msg: string }) => d.msg).join("; ") : detail ?? message;
    } catch {
      /* keep the generic message */
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export const analyzeLog = (log: string) =>
  request<Report>("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ log }),
  });

export const getSamples = () => request<Sample[]>("/api/samples");
export const getEvaluation = () => request<Evaluation>("/api/evaluation");