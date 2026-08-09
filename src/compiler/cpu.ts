import { createModelData } from "./models";
import type { ExecutionResult, ModelData, ModelPreset } from "./types";

const gelu = (value: number) =>
  0.5 * value * (1 + Math.tanh(0.7978845608 * (value + 0.044715 * value * value * value)));

function linear(
  input: Float32Array,
  weights: Float32Array,
  bias: Float32Array,
  rows: number,
  inner: number,
  columns: number,
  activation: boolean
) {
  const output = new Float32Array(rows * columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      let sum = bias[column] ?? 0;
      for (let index = 0; index < inner; index += 1) {
        sum += (input[row * inner + index] ?? 0) * (weights[index * columns + column] ?? 0);
      }
      output[row * columns + column] = activation ? gelu(sum) : sum;
    }
  }
  return output;
}

function softmax(input: Float32Array, rows: number, columns: number) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    let maximum = -Infinity;
    for (let column = 0; column < columns; column += 1) {
      maximum = Math.max(maximum, input[row * columns + column] ?? -Infinity);
    }
    let denominator = 0;
    for (let column = 0; column < columns; column += 1) {
      denominator += Math.exp((input[row * columns + column] ?? 0) - maximum);
    }
    for (let column = 0; column < columns; column += 1) {
      output[row * columns + column] = Math.exp((input[row * columns + column] ?? 0) - maximum) / denominator;
    }
  }
  return output;
}

export function executeCpuWithData(preset: ModelPreset, data: ModelData): Float32Array {
  const hidden = linear(data.input, data.weight1, data.bias1, preset.batch, preset.input, preset.hidden, true);
  const logits = linear(hidden, data.weight2, data.bias2, preset.batch, preset.hidden, preset.output, false);
  return softmax(logits, preset.batch, preset.output);
}

export function executeCpu(preset: ModelPreset): ExecutionResult {
  const data = createModelData(preset);
  const started = performance.now();
  const output = executeCpuWithData(preset, data);
  return { output, milliseconds: performance.now() - started };
}

export function benchmarkCpu(preset: ModelPreset, iterations = 5) {
  executeCpu(preset);
  const samples: number[] = [];
  let output: Float32Array = new Float32Array(0);
  for (let index = 0; index < iterations; index += 1) {
    const result = executeCpu(preset);
    samples.push(result.milliseconds);
    output = result.output;
  }
  const average = samples.reduce((total, value) => total + value, 0) / samples.length;
  return { average, minimum: Math.min(...samples), samples, output };
}

export function compareOutputs(reference: Float32Array, candidate: Float32Array) {
  if (reference.length !== candidate.length) throw new Error("Output lengths differ");
  let maxAbsolute = 0;
  let maxRelative = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const expected = reference[index] ?? 0;
    const actual = candidate[index] ?? 0;
    const absolute = Math.abs(expected - actual);
    const relative = absolute / Math.max(Math.abs(expected), 1e-7);
    maxAbsolute = Math.max(maxAbsolute, absolute);
    maxRelative = Math.max(maxRelative, relative);
  }
  return { maxAbsolute, maxRelative, passed: maxAbsolute <= 2e-4 };
}
