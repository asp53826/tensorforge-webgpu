import { CircleAlert, Gauge, LoaderCircle, Play, ShieldCheck } from "lucide-react";
import type { GpuResult } from "../runtime/webgpu";

interface BenchmarkPanelProps {
  state: "idle" | "running" | "done" | "error";
  cpuMs: number | null;
  gpu: GpuResult | null;
  correctness: { maxAbsolute: number; maxRelative: number; passed: boolean } | null;
  error: string | null;
  onRun: () => void;
}

const time = (value: number | null) => value === null ? "—" : `${value.toFixed(value < 1 ? 3 : 2)} ms`;

export function BenchmarkPanel({ state, cpuMs, gpu, correctness, error, onRun }: BenchmarkPanelProps) {
  const gpuMs = gpu?.supported ? gpu.average : null;
  return (
    <section className="instrument benchmark-instrument" aria-labelledby="benchmark-title">
      <header className="instrument-head">
        <div><span className="eyebrow">Local execution</span><h2 id="benchmark-title">Hardware probe</h2></div>
        <span className={`status-led ${gpu?.supported ? "is-live" : ""}`}><i />{gpu?.supported ? "GPU online" : "Awaiting run"}</span>
      </header>
      <p className="benchmark-copy">Measures this browser only. Five timed runs follow one warm-up; results are not a cross-device speed claim.</p>
      <div className="benchmark-grid">
        <div><span>CPU reference</span><strong>{time(cpuMs)}</strong><small>single-thread JS</small></div>
        <div><span>WebGPU pipeline</span><strong>{time(gpuMs)}</strong><small>3 compute dispatches</small></div>
      </div>
      {correctness && (
        <div className={`correctness ${correctness.passed ? "is-pass" : "is-fail"}`}>
          {correctness.passed ? <ShieldCheck size={19} /> : <CircleAlert size={19} />}
          <span><strong>{correctness.passed ? "Output verified" : "Tolerance exceeded"}</strong><small>max |Δ| {correctness.maxAbsolute.toExponential(2)}</small></span>
        </div>
      )}
      {!correctness && gpu && !gpu.supported && <div className="correctness is-warn"><CircleAlert size={19} /><span><strong>CPU mode</strong><small>{gpu.reason}</small></span></div>}
      {error && <div className="correctness is-fail"><CircleAlert size={19} /><span><strong>Run failed</strong><small>{error}</small></span></div>}
      <button className="run-button" onClick={onRun} disabled={state === "running"}>
        {state === "running" ? <LoaderCircle className="spin" size={18} /> : state === "done" ? <Gauge size={18} /> : <Play size={18} fill="currentColor" />}
        {state === "running" ? "Profiling pipeline…" : state === "done" ? "Run again" : "Compile + run"}
      </button>
      {gpu?.supported && <p className="adapter-name">Adapter: {gpu.adapter}</p>}
    </section>
  );
}
