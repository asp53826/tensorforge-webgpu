import { ArrowDown, CheckCircle2, ExternalLink, ShieldAlert } from "lucide-react";
import { compileGraph } from "../compiler/compiler";
import { createDemoGraph, PRESETS } from "../compiler/models";

const formatBytes = (bytes: number) => bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(2)} MB` : `${(bytes / 1024).toFixed(1)} KB`;

export function ArenaMatrix() {
  const rows = PRESETS.map((preset) => {
    const source = createDemoGraph(preset);
    const baseline = compileGraph(source, { fusion: false, memoryReuse: false });
    const optimized = compileGraph(source, { fusion: true, memoryReuse: true });
    return {
      preset,
      nodesBefore: baseline.finalGraph.nodes.length,
      nodesAfter: optimized.finalGraph.nodes.length,
      kernelsBefore: baseline.snapshots.at(-1)!.stats.kernels,
      kernelsAfter: optimized.snapshots.at(-1)!.stats.kernels,
      bytesBefore: baseline.memory.plannedBytes,
      bytesAfter: optimized.memory.plannedBytes,
      slotsBefore: baseline.memory.slots,
      slotsAfter: optimized.memory.slots
    };
  });

  return <section className="arena" id="kernel-arena" aria-labelledby="arena-title">
    <header className="arena-heading"><div><p className="section-number">00 / KERNELARENA COMPARISON MODE</p><h2 id="arena-title">Three shapes. Same compiler. Every delta inspectable.</h2></div><p>This matrix invokes the compiler twice for every model shape. It is not decorative project metadata: the node counts, dispatch counts, allocation slots, and bytes below are produced from the live TypeScript IR.</p></header>
    <div className="arena-matrix" role="table" aria-label="Compiler optimization comparison">
      <div className="arena-row arena-header" role="row"><span role="columnheader">Shape</span><span role="columnheader">IR nodes</span><span role="columnheader">GPU dispatches</span><span role="columnheader">Allocated bytes</span><span role="columnheader">Buffer slots</span></div>
      {rows.map((row) => <div className="arena-row" role="row" key={row.preset.id}>
        <div role="cell"><strong>{row.preset.label}</strong><small>{row.preset.batch}×{row.preset.input}×{row.preset.hidden}×{row.preset.output}</small></div>
        <div role="cell"><span>{row.nodesBefore}</span><ArrowDown aria-hidden="true" /><strong>{row.nodesAfter}</strong></div>
        <div role="cell"><span>{row.kernelsBefore}</span><ArrowDown aria-hidden="true" /><strong>{row.kernelsAfter}</strong></div>
        <div role="cell"><span>{formatBytes(row.bytesBefore)}</span><ArrowDown aria-hidden="true" /><strong>{formatBytes(row.bytesAfter)}</strong></div>
        <div role="cell"><span>{row.slotsBefore}</span><ArrowDown aria-hidden="true" /><strong>{row.slotsAfter}</strong></div>
      </div>)}
    </div>
    <div className="arena-contract"><p><CheckCircle2 aria-hidden="true" /><span><strong>Fresh compiler output</strong>Values are recalculated in your browser from the source graph and selected compiler passes.</span></p><p><ShieldAlert aria-hidden="true" /><span><strong>No universal speedup claim</strong>Graph simplification does not guarantee lower wall time on every GPU; use the local runtime probe below.</span></p><a href="https://asp53826.github.io/labs/" target="_blank" rel="noreferrer">Open the full Proof Laboratory<ExternalLink aria-hidden="true" /></a></div>
  </section>;
}
