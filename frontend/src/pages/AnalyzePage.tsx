import { useEffect, useState } from "react";
import { analyzeLog, getSamples } from "../api";
import type { Report, Sample } from "../types";

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="bar" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className="bar-fill" style={{ width: `${value * 100}%` }} />
    </div>
  );
}

export default function AnalyzePage() {
  const [log, setLog] = useState("");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    getSamples().then(setSamples).catch(() => setSamples([]));
  }, []);

  async function run() {
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      setReport(await analyzeLog(log));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  async function copyJson() {
    await navigator.clipboard?.writeText(JSON.stringify(report, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  const isAnomaly = report?.label === "Anomaly";

  return (
    <div className="grid">
      <section className="card">
        <h2>Log input</h2>
        <p className="hint">Paste the log lines of one HDFS block (raw lines are fine; timestamps and block IDs are cleaned automatically).</p>
        {samples.length > 0 && (
          <select
            aria-label="Load a sample"
            defaultValue=""
            onChange={(e) => {
              const sample = samples.find((s) => s.id === e.target.value);
              if (sample) setLog(sample.log);
            }}
          >
            <option value="" disabled>Load a sample log…</option>
            {samples.map((s) => (
              <option key={s.id} value={s.id}>{s.title}</option>
            ))}
          </select>
        )}
        <textarea
          aria-label="Log text"
          value={log}
          onChange={(e) => setLog(e.target.value)}
          placeholder="081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating"
          rows={12}
        />
        <button className="primary" onClick={run} disabled={loading || !log.trim()}>
          {loading ? "Analyzing…" : "Run analysis"}
        </button>
        {error && <div className="alert error" role="alert">{error}</div>}
      </section>

      <section className="card">
        <h2>Result</h2>
        {!report && !loading && !error && <p className="hint">Run an analysis to see the classification here.</p>}
        {loading && <p className="hint">Running the classifier…</p>}
        {report && (
          <>
            <div className="row">
              <span className={`badge ${isAnomaly ? "bad" : "good"}`}>{report.label}</span>
              <span className="hint">
                confidence {(report.confidence * 100).toFixed(1)}% · P(anomaly) {report.p_anomaly.toFixed(3)} · threshold {report.threshold}
              </span>
            </div>
            <ConfidenceBar value={report.confidence} />
            <p className="hint">Confidence is the model's score for its predicted label, not a calibrated probability.</p>

            {isAnomaly && (
              <div className="diagnosis">
                <h3>Probable root cause</h3>
                {report.cause ? <p>{report.cause}</p> : <p className="alert warn">Diagnosis unavailable: {report.diagnosis_error}</p>}
                {report.suggested_fix && (
                  <>
                    <h3>Suggested remediation</h3>
                    <ol>{report.suggested_fix.map((step, i) => <li key={i}>{step}</li>)}</ol>
                  </>
                )}
              </div>
            )}

            <div className="alert info">{report.advisory}</div>
            <div className="chips">
              <span>classify {report.latency_ms.classify} ms</span>
              {report.latency_ms.diagnose !== null && <span>diagnose {report.latency_ms.diagnose} ms</span>}
              <span>total {report.latency_ms.total} ms</span>
              {report.truncated && <span>snippet truncated</span>}
            </div>

            <div className="row spread">
              <h3>Structured JSON</h3>
              <button onClick={copyJson}>{copied ? "Copied" : "Copy"}</button>
            </div>
            <pre className="json">{JSON.stringify(report, null, 2)}</pre>
          </>
        )}
      </section>
    </div>
  );
}