import { describe, expect, it } from "vitest";
import { compareOutputs, executeCpu } from "../compiler/cpu";
import { PRESETS } from "../compiler/models";
import { createKernelSources } from "../runtime/webgpu";

describe("TensorForge execution paths", () => {
  it("produces deterministic row-normalized CPU probabilities", () => {
    const preset = PRESETS[0]!;
    const first = executeCpu(preset).output;
    const second = executeCpu(preset).output;
    expect([...first]).toEqual([...second]);
    for (let row = 0; row < preset.batch; row += 1) {
      let total = 0;
      for (let column = 0; column < preset.output; column += 1) total += first[row * preset.output + column] ?? 0;
      expect(total).toBeCloseTo(1, 5);
    }
  });

  it("reports numerical differences against the configured tolerance", () => {
    const reference = new Float32Array([0.25, 0.75]);
    expect(compareOutputs(reference, new Float32Array([0.25001, 0.74999])).passed).toBe(true);
    expect(compareOutputs(reference, new Float32Array([0.3, 0.7])).passed).toBe(false);
  });

  it("emits fused linear, linear, and softmax WGSL entry points", () => {
    const kernels = createKernelSources(PRESETS[1]!);
    expect(kernels).toHaveLength(3);
    expect(kernels.every((kernel) => kernel.source.includes("@compute"))).toBe(true);
    expect(kernels[0]?.source).toContain("tanh");
    expect(kernels[2]?.source).toContain("denominator");
  });
});
