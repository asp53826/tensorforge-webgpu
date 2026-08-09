# TensorForge

TensorForge is a visual tensor compiler that runs entirely in the browser. It lowers a typed two-layer projection graph through inspectable optimization passes, generates WGSL compute kernels, executes them with WebGPU, and checks the GPU output against a deterministic CPU reference.

**[Launch the live compiler workbench](https://asp53826.github.io/tensorforge-webgpu/)**

## What it demonstrates

- A small typed tensor intermediate representation with explicit shapes and storage sizes
- Shape inference and matrix compatibility validation
- Identity canonicalization and consumer rewiring
- Pattern-based `MatMul + Bias + GELU` and `MatMul + Bias` fusion
- Output-rooted dead-code elimination
- Liveness analysis and reusable transient buffer slots
- Generated WGSL compute shaders dispatched through the browser's real WebGPU API
- Seeded inputs and parameters with CPU/GPU numerical comparison
- Hardware-local profiling that avoids treating one machine as a universal benchmark

The default graph compiles from **13 nodes / 8 primitive kernels** to **8 nodes / 3 GPU dispatches**. Those counts are structural properties tested in CI, not hand-entered marketing numbers.

## Compiler pipeline

```text
Tensor graph
   │
   ├─ shape inference + validation
   ├─ identity canonicalization
   ├─ linear / bias / activation fusion
   ├─ dead-code elimination
   └─ liveness-aware memory planning
            │
            ├─ fused linear + GELU.wgsl
            ├─ fused linear.wgsl
            └─ row softmax.wgsl
                         │
                     WebGPU queue
```

## Run locally

Requirements: Node.js 22+ and a current browser with WebGPU support for GPU execution.

```bash
npm install
npm run dev
```

Then open the local URL printed by Vite. Browsers without WebGPU still expose the complete compiler UI, generated WGSL, memory planner, and CPU reference path.

## Verify

```bash
npm run check
npm run build
```

The Vitest suite covers graph lowering, fusion, dead-code elimination, persistent/transient slot separation, memory reuse, shape errors, deterministic CPU output, probability normalization, tolerance checks, and WGSL generation.

## Scope and limitations

TensorForge is an educational compiler experiment, not a general-purpose ML framework. The IR currently supports a narrow two-layer MLP-style graph, tensors are dense `f32`, kernels are deliberately readable rather than autotuned, GPU timing uses queue wall time rather than timestamp queries, and the softmax kernel favors clarity over parallel reduction. The interface reports measurements from the visitor's current browser and hardware only.

## Stack

React 19 · TypeScript · Vite · WebGPU/WGSL · Vitest · GitHub Actions/Pages

## License

[MIT](./LICENSE)
