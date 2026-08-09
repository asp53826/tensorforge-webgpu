import "@fontsource-variable/instrument-sans";
import "@fontsource-variable/syne";
import "@fontsource/ibm-plex-mono/400.css";
import { ArrowUpRight, Boxes, Cpu, Github, MemoryStick, RadioTower, RotateCcw, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { BenchmarkPanel } from "./components/BenchmarkPanel";
import { GraphStage } from "./components/GraphStage";
import { KernelInspector } from "./components/KernelInspector";
import { MemoryMap } from "./components/MemoryMap";
import { PassRail } from "./components/PassRail";
import { benchmarkCpu, compareOutputs } from "./compiler/cpu";
import { compileGraph } from "./compiler/compiler";
import { createDemoGraph, PRESETS } from "./compiler/models";
import type { TensorNode } from "./compiler/types";
import { benchmarkWebGpu, createKernelSources, type GpuResult } from "./runtime/webgpu";

const formatBytes = (bytes: number) => bytes >= 1024 * 1024
  ? `${(bytes / 1024 / 1024).toFixed(2)} MB`
  : `${(bytes / 1024).toFixed(1)} KB`;

export default function App() {
  const [presetId, setPresetId] = useState("standard");
  const [fusion, setFusion] = useState(true);
  const [memoryReuse, setMemoryReuse] = useState(true);
  const [activePass, setActivePass] = useState(5);
  const [selectedId, setSelectedId] = useState("probabilities");
  const [runState, setRunState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [cpuMs, setCpuMs] = useState<number | null>(null);
  const [gpu, setGpu] = useState<GpuResult | null>(null);
  const [correctness, setCorrectness] = useState<ReturnType<typeof compareOutputs> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const preset = PRESETS.find((item) => item.id === presetId) ?? PRESETS[1]!;
  const source = useMemo(() => createDemoGraph(preset), [preset]);
  const compilation = useMemo(() => compileGraph(source, { fusion, memoryReuse }), [source, fusion, memoryReuse]);
  const snapshot = compilation.snapshots[Math.min(activePass, compilation.snapshots.length - 1)]!;
  const selectedNode = snapshot.graph.nodes.find((node) => node.id === selectedId) ?? snapshot.graph.nodes.at(-1)!;
  const sourceStats = compilation.snapshots[0]!.stats;
  const finalStats = compilation.snapshots.at(-1)!.stats;
  const naiveTransient = compilation.memory.naiveBytes - compilation.memory.persistentBytes;
  const memorySaving = memoryReuse && naiveTransient > 0
    ? 100 - (compilation.memory.peakTransientBytes / naiveTransient) * 100
    : 0;
  const kernels = gpu?.kernels ?? createKernelSources(preset);

  useEffect(() => {
    if (!snapshot.graph.nodes.some((node) => node.id === selectedId)) setSelectedId(snapshot.graph.nodes.at(-1)?.id ?? "input");
  }, [snapshot.graph.nodes, selectedId]);

  useEffect(() => {
    setRunState("idle");
    setCpuMs(null);
    setGpu(null);
    setCorrectness(null);
    setError(null);
  }, [presetId]);

  const animatePasses = () => {
    setActivePass(0);
    compilation.snapshots.slice(1).forEach((_, index) => {
      window.setTimeout(() => setActivePass(index + 1), (index + 1) * 420);
    });
  };

  const runBenchmark = async () => {
    setRunState("running");
    setError(null);
    setCorrectness(null);
    try {
      await new Promise((resolve) => window.setTimeout(resolve, 80));
      const cpu = benchmarkCpu(preset, 5);
      setCpuMs(cpu.average);
      const result = await benchmarkWebGpu(preset, 5);
      setGpu(result);
      if (result.supported) setCorrectness(compareOutputs(cpu.output, result.output));
      setRunState("done");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unknown runtime error");
      setRunState("error");
    }
  };

  const selectNode = (node: TensorNode) => setSelectedId(node.id);

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="TensorForge home"><span className="brand-mark"><i /><i /><i /></span><strong>TensorForge</strong><small>WEBGPU LAB</small></a>
        <nav aria-label="Primary navigation">
          <a href="#compiler-workbench">Workbench</a>
          <a href="#generated-code">WGSL</a>
          <a href="#architecture">Architecture</a>
        </nav>
        <a className="source-link" href="https://github.com/asp53826/tensorforge-webgpu" target="_blank" rel="noreferrer"><Github size={16} />Source<ArrowUpRight size={14} /></a>
      </header>

      <main id="top">
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="kicker"><span />A browser-native compiler experiment</p>
            <h1 id="hero-title">Compile the graph.<br /><em>Watch the silicon change.</em></h1>
            <p className="hero-lede">TensorForge lowers a typed tensor graph through canonicalization, fusion, dead-code elimination, and liveness-aware allocation—then executes the generated WGSL on your GPU.</p>
            <div className="hero-actions">
              <a className="primary-action" href="#compiler-workbench">Open workbench<ArrowUpRight size={17} /></a>
              <button className="text-action" onClick={runBenchmark}><RadioTower size={16} />Probe my GPU</button>
            </div>
          </div>
          <div className="hero-schematic" aria-hidden="true">
            <div className="chip-label">TF·01</div>
            <div className="chip-core"><span>TENSOR</span><strong>IR</strong><small>F32 / WGSL</small></div>
            <div className="signal signal-a" /><div className="signal signal-b" /><div className="signal signal-c" />
            <div className="pin pin-a">FUSE</div><div className="pin pin-b">ALLOC</div><div className="pin pin-c">DISPATCH</div>
            <div className="scope-lines"><i /><i /><i /><i /><i /></div>
          </div>
        </section>

        <section className="proof-strip" aria-label="Compiler output summary">
          <div><Boxes size={18} /><span>Graph nodes</span><strong>{sourceStats.nodes} → {finalStats.nodes}</strong></div>
          <div><Cpu size={18} /><span>GPU dispatches</span><strong>{sourceStats.kernels} → {finalStats.kernels}</strong></div>
          <div><MemoryStick size={18} /><span>Transient reuse</span><strong>−{Math.max(0, memorySaving).toFixed(0)}%</strong></div>
          <div><Sparkles size={18} /><span>Target</span><strong>WGSL / f32</strong></div>
        </section>

        <section className="workbench" id="compiler-workbench" aria-labelledby="workbench-title">
          <header className="section-heading">
            <div><p className="section-number">01 / COMPILER WORKBENCH</p><h2 id="workbench-title">Interrogate every rewrite.</h2></div>
            <p>Select a pass to inspect the intermediate representation. Every node, byte count, and generated kernel comes from the running compiler.</p>
          </header>

          <div className="control-deck">
            <div className="preset-control">
              <span className="control-label">MODEL SHAPE</span>
              <div className="segmented-control">
                {PRESETS.map((item) => <button key={item.id} aria-pressed={preset.id === item.id} onClick={() => setPresetId(item.id)}><strong>{item.label}</strong><small>{item.batch}×{item.input}×{item.hidden}×{item.output}</small></button>)}
              </div>
            </div>
            <div className="toggle-rack">
              <button className="compiler-toggle" aria-pressed={fusion} onClick={() => setFusion((value) => !value)}><span><i /></span><strong>Operator fusion</strong><small>{fusion ? "enabled" : "bypassed"}</small></button>
              <button className="compiler-toggle" aria-pressed={memoryReuse} onClick={() => setMemoryReuse((value) => !value)}><span><i /></span><strong>Buffer reuse</strong><small>{memoryReuse ? "enabled" : "independent"}</small></button>
              <button className="replay-button" onClick={animatePasses}><RotateCcw size={15} />Replay passes</button>
            </div>
          </div>

          <div className="compiler-frame">
            <PassRail snapshots={compilation.snapshots} active={activePass} onChange={setActivePass} />
            <div className="graph-panel">
              <header className="graph-toolbar">
                <div><span className="live-dot" />{snapshot.action.toUpperCase()} / <strong>{snapshot.name}</strong></div>
                <div><span>{snapshot.stats.nodes} nodes</span><span>{snapshot.stats.kernels} kernels</span><span>{formatBytes(snapshot.stats.transientBytes)}</span></div>
              </header>
              <GraphStage graph={snapshot.graph} selected={selectedNode.id} onSelect={selectNode} />
              <footer className="graph-caption"><span>{snapshot.detail}</span><strong>{snapshot.stats.removed > 0 ? `−${snapshot.stats.removed} nodes` : snapshot.stats.fused > 0 ? `${snapshot.stats.fused} patterns` : "IR valid"}</strong></footer>
            </div>
            <aside className="node-inspector" aria-label="Selected tensor details">
              <span className="eyebrow">Selected tensor</span>
              <p className="node-id">%{selectedNode.id}</p>
              <h3>{selectedNode.label}</h3>
              <dl>
                <div><dt>Operation</dt><dd>{selectedNode.op}</dd></div>
                <div><dt>Shape</dt><dd>[{selectedNode.shape.join(", ")}]</dd></div>
                <div><dt>Storage</dt><dd>{formatBytes(selectedNode.bytes)}</dd></div>
                <div><dt>Type</dt><dd>{selectedNode.dtype}</dd></div>
              </dl>
              <div className="inspector-inputs"><span>INPUTS</span>{selectedNode.inputs.length ? selectedNode.inputs.map((input) => <code key={input}>%{input}</code>) : <em>source tensor</em>}</div>
            </aside>
          </div>
        </section>

        <section className="instrument-grid" id="generated-code">
          <MemoryMap memory={compilation.memory} graph={compilation.finalGraph} />
          <KernelInspector kernels={kernels} />
          <BenchmarkPanel state={runState} cpuMs={cpuMs} gpu={gpu} correctness={correctness} error={error} onRun={runBenchmark} />
        </section>

        <section className="architecture" id="architecture" aria-labelledby="architecture-title">
          <header className="section-heading">
            <div><p className="section-number">02 / SYSTEM NOTES</p><h2 id="architecture-title">Small enough to inspect. Real enough to execute.</h2></div>
            <p>The scope is intentionally narrow: a two-layer projection, a typed IR, three optimizations, a linear-scan allocator, and an actual WebGPU backend.</p>
          </header>
          <div className="architecture-grid">
            <article><span>FRONTEND</span><strong>Typed graph IR</strong><p>Explicit tensor ranks and f32 storage make shape errors visible before GPU work begins.</p></article>
            <article><span>MIDDLE-END</span><strong>Rewrite pipeline</strong><p>Canonicalization, pattern fusion, output-rooted liveness, and buffer slot reuse are independently switchable.</p></article>
            <article><span>BACKEND</span><strong>WGSL compute</strong><p>Three generated kernels dispatch through WebGPU and read results back for numerical comparison.</p></article>
            <article><span>VALIDATION</span><strong>CPU oracle</strong><p>Seeded parameters and an equivalent JavaScript path keep every run reproducible and testable.</p></article>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div><span className="brand-mark"><i /><i /><i /></span><strong>TensorForge</strong></div>
        <p>Built as an inspectable compiler experiment by <a href="https://asp53826.github.io/">Aaryan Patel</a>.</p>
        <a href="https://github.com/asp53826/tensorforge-webgpu" target="_blank" rel="noreferrer">Read the source<ArrowUpRight size={14} /></a>
      </footer>
    </div>
  );
}
