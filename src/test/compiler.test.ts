import { describe, expect, it } from "vitest";
import { canonicalize, compileGraph, validateAndInfer } from "../compiler/compiler";
import { createDemoGraph, PRESETS } from "../compiler/models";

const preset = PRESETS[1]!;

describe("TensorForge compiler pipeline", () => {
  it("lowers the demo graph from 13 nodes and 8 kernels to 8 nodes and 3 kernels", () => {
    const compilation = compileGraph(createDemoGraph(preset), { fusion: true, memoryReuse: true });
    expect(compilation.snapshots[0]?.stats).toMatchObject({ nodes: 13, kernels: 8 });
    expect(compilation.snapshots.at(-1)?.stats).toMatchObject({ nodes: 8, kernels: 3 });
    expect(compilation.finalGraph.nodes.map((node) => node.op)).toContain("fusedLinearGelu");
    expect(compilation.finalGraph.nodes.map((node) => node.op)).toContain("fusedLinear");
  });

  it("rewires identity consumers during canonicalization", () => {
    const graph = canonicalize(createDemoGraph(preset));
    expect(graph.nodes.some((node) => node.id === "identity")).toBe(false);
    expect(graph.nodes.find((node) => node.id === "matmul2")?.inputs[0]).toBe("gelu1");
  });

  it("removes the output-disconnected debug tap", () => {
    const compilation = compileGraph(createDemoGraph(preset), { fusion: true, memoryReuse: true });
    expect(compilation.finalGraph.nodes.some((node) => node.id === "debug")).toBe(false);
  });

  it("never reuses persistent parameter slots for transient tensors", () => {
    const compilation = compileGraph(createDemoGraph(preset), { fusion: true, memoryReuse: true });
    const persistent = new Set(compilation.memory.allocations.filter((item) => item.persistent).map((item) => item.slot));
    const transient = new Set(compilation.memory.allocations.filter((item) => !item.persistent).map((item) => item.slot));
    expect([...persistent].some((slot) => transient.has(slot))).toBe(false);
  });

  it("reduces allocated bytes when buffer reuse is enabled", () => {
    const reused = compileGraph(createDemoGraph(preset), { fusion: true, memoryReuse: true }).memory;
    const independent = compileGraph(createDemoGraph(preset), { fusion: true, memoryReuse: false }).memory;
    expect(reused.plannedBytes).toBeLessThan(independent.plannedBytes);
  });

  it("rejects incompatible matrix dimensions", () => {
    const graph = createDemoGraph(preset);
    graph.nodes.find((node) => node.id === "weight1")!.shape = [127, preset.hidden];
    expect(() => validateAndInfer(graph)).toThrow(/cannot multiply/);
  });
});
