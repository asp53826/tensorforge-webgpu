import { createModelData } from "../compiler/models";
import type { ModelPreset } from "../compiler/types";

export interface KernelSource {
  id: "linear-gelu" | "linear" | "softmax";
  label: string;
  dispatch: string;
  source: string;
}

export interface GpuBenchmark {
  supported: true;
  output: Float32Array;
  average: number;
  minimum: number;
  samples: number[];
  adapter: string;
  kernels: KernelSource[];
}

export interface GpuUnavailable {
  supported: false;
  reason: string;
  kernels: KernelSource[];
}

export type GpuResult = GpuBenchmark | GpuUnavailable;

function linearKernel(rows: number, inner: number, columns: number, withGelu: boolean) {
  const activation = withGelu
    ? "value = 0.5 * value * (1.0 + tanh(0.7978845608 * (value + 0.044715 * value * value * value)));"
    : "";
  return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let column = gid.x;
  let row = gid.y;
  if (row >= ${rows}u || column >= ${columns}u) { return; }

  var value = bias[column];
  for (var k = 0u; k < ${inner}u; k = k + 1u) {
    value = value + input[row * ${inner}u + k] * weight[k * ${columns}u + column];
  }
  ${activation}
  output[row * ${columns}u + column] = value;
}`.trim();
}

function softmaxKernel(rows: number, columns: number) {
  return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= ${rows}u) { return; }

  var maximum = -3.402823e+38;
  for (var column = 0u; column < ${columns}u; column = column + 1u) {
    maximum = max(maximum, input[row * ${columns}u + column]);
  }

  var denominator = 0.0;
  for (var column = 0u; column < ${columns}u; column = column + 1u) {
    denominator = denominator + exp(input[row * ${columns}u + column] - maximum);
  }
  for (var column = 0u; column < ${columns}u; column = column + 1u) {
    output[row * ${columns}u + column] = exp(input[row * ${columns}u + column] - maximum) / denominator;
  }
}`.trim();
}

export function createKernelSources(preset: ModelPreset): KernelSource[] {
  return [
    {
      id: "linear-gelu",
      label: "fused_linear_gelu.wgsl",
      dispatch: `${Math.ceil(preset.hidden / 8)} × ${Math.ceil(preset.batch / 8)} workgroups`,
      source: linearKernel(preset.batch, preset.input, preset.hidden, true)
    },
    {
      id: "linear",
      label: "fused_linear.wgsl",
      dispatch: `${Math.ceil(preset.output / 8)} × ${Math.ceil(preset.batch / 8)} workgroups`,
      source: linearKernel(preset.batch, preset.hidden, preset.output, false)
    },
    {
      id: "softmax",
      label: "row_softmax.wgsl",
      dispatch: `${preset.batch} workgroups`,
      source: softmaxKernel(preset.batch, preset.output)
    }
  ];
}

function storageBuffer(device: GPUDevice, data: Float32Array) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function emptyStorage(device: GPUDevice, floats: number) {
  return device.createBuffer({
    size: floats * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
}

function createPipeline(device: GPUDevice, source: string) {
  return device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: source }), entryPoint: "main" }
  });
}

function encodePass(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, x: number, y = 1) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y);
  pass.end();
}

export async function benchmarkWebGpu(preset: ModelPreset, iterations = 5): Promise<GpuResult> {
  const kernels = createKernelSources(preset);
  if (!navigator.gpu) {
    return { supported: false, reason: "This browser does not expose WebGPU. The CPU reference path remains fully interactive.", kernels };
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return { supported: false, reason: "No compatible GPU adapter was returned by the browser.", kernels };

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice();
  } catch {
    return { supported: false, reason: "The GPU adapter declined a compute device.", kernels };
  }

  const data = createModelData(preset);
  const input = storageBuffer(device, data.input);
  const weight1 = storageBuffer(device, data.weight1);
  const bias1 = storageBuffer(device, data.bias1);
  const weight2 = storageBuffer(device, data.weight2);
  const bias2 = storageBuffer(device, data.bias2);
  const hidden = emptyStorage(device, preset.batch * preset.hidden);
  const logits = emptyStorage(device, preset.batch * preset.output);
  const output = emptyStorage(device, preset.batch * preset.output);

  const pipeline1 = createPipeline(device, kernels[0]?.source ?? "");
  const pipeline2 = createPipeline(device, kernels[1]?.source ?? "");
  const pipeline3 = createPipeline(device, kernels[2]?.source ?? "");
  const group1 = device.createBindGroup({
    layout: pipeline1.getBindGroupLayout(0),
    entries: [input, weight1, bias1, hidden].map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const group2 = device.createBindGroup({
    layout: pipeline2.getBindGroupLayout(0),
    entries: [hidden, weight2, bias2, logits].map((buffer, binding) => ({ binding, resource: { buffer } }))
  });
  const group3 = device.createBindGroup({
    layout: pipeline3.getBindGroupLayout(0),
    entries: [logits, output].map((buffer, binding) => ({ binding, resource: { buffer } }))
  });

  const submit = async () => {
    const encoder = device.createCommandEncoder();
    encodePass(encoder, pipeline1, group1, Math.ceil(preset.hidden / 8), Math.ceil(preset.batch / 8));
    encodePass(encoder, pipeline2, group2, Math.ceil(preset.output / 8), Math.ceil(preset.batch / 8));
    encodePass(encoder, pipeline3, group3, preset.batch);
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - started;
  };

  await submit();
  const samples: number[] = [];
  for (let index = 0; index < iterations; index += 1) samples.push(await submit());

  const readback = device.createBuffer({
    size: preset.batch * preset.output * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const readEncoder = device.createCommandEncoder();
  readEncoder.copyBufferToBuffer(output, 0, readback, 0, readback.size);
  device.queue.submit([readEncoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const values = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  const info = adapter.info;
  const adapterName = info.description || info.device || info.architecture || info.vendor || "Browser GPU adapter";
  for (const buffer of [input, weight1, bias1, weight2, bias2, hidden, logits, output, readback]) buffer.destroy();
  device.destroy();

  return {
    supported: true,
    output: values,
    average: samples.reduce((total, value) => total + value, 0) / samples.length,
    minimum: Math.min(...samples),
    samples,
    adapter: adapterName,
    kernels
  };
}
