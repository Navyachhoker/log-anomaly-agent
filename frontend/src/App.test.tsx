import { useState } from "react";
import AnalyzePage from "./pages/AnalyzePage";
import EvaluationPage from "./pages/EvaluationPage";

export default function App() {
  const [tab, setTab] = useState<"analyze" | "evaluation">("analyze");
  return (
    <div className="app">
      <header>
        <div>
          <h1>Log Anomaly Classifier</h1>
          <p className="subtitle">Fine-tuned Qwen2.5-1.5B classifies HDFS logs; an LLM diagnoses anomalies. Advisory only.</p>
        </div>
        <nav>
          <button className={tab === "analyze" ? "active" : ""} onClick={() => setTab("analyze")}>Analyze</button>
          <button className={tab === "evaluation" ? "active" : ""} onClick={() => setTab("evaluation")}>Evaluation</button>
        </nav>
      </header>
      <main>{tab === "analyze" ? <AnalyzePage /> : <EvaluationPage />}</main>
    </div>
  );
}