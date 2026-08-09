import type { MemoryPlan, TensorGraph } from "../compiler/types";

const formatBytes = (bytes: number) => {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
};

export function MemoryMap({ memory, graph }: { memory: MemoryPlan; graph: TensorGraph }) {
  const max = Math.max(...memory.allocations.map((allocation) => allocation.bytes), 1);
  const names = new Map(graph.nodes.map((node) => [node.id, node.label]));
  const naiveTransient = memory.naiveBytes - memory.persistentBytes;
  const saving = memory.plannedBytes !== memory.naiveBytes && naiveTransient > 0
    ? 100 - (memory.peakTransientBytes / naiveTransient) * 100
    : 0;
  return (
    <section className="instrument memory-instrument" aria-labelledby="memory-title">
      <header className="instrument-head">
        <div><span className="eyebrow">Liveness allocator</span><h2 id="memory-title">Memory plan</h2></div>
        <span className="saving-chip">−{Math.max(0, saving).toFixed(0)}% transient</span>
      </header>
      <div className="memory-summary">
        <div><span>Naive</span><strong>{formatBytes(memory.naiveBytes)}</strong></div>
        <div><span>Planned</span><strong>{formatBytes(memory.plannedBytes)}</strong></div>
        <div><span>Slots</span><strong>{memory.slots}</strong></div>
      </div>
      <div className="slot-stack" aria-label="Allocated tensor buffers">
        {memory.allocations.map((allocation) => (
          <div className={`slot-row ${allocation.persistent ? "is-persistent" : ""}`} key={allocation.nodeId}>
            <span className="slot-number">S{String(allocation.slot).padStart(2, "0")}</span>
            <div className="slot-track">
              <span style={{ width: `${Math.max(8, (allocation.bytes / max) * 100)}%` }} />
            </div>
            <span className="slot-name">{names.get(allocation.nodeId)}</span>
            <span className="slot-size">{formatBytes(allocation.bytes)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
