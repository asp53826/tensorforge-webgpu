import { Check, Copy } from "lucide-react";
import { useState } from "react";
import type { KernelSource } from "../runtime/webgpu";

export function KernelInspector({ kernels }: { kernels: KernelSource[] }) {
  const [active, setActive] = useState(0);
  const [copied, setCopied] = useState(false);
  const kernel = kernels[active] ?? kernels[0];
  if (!kernel) return null;

  const copy = async () => {
    await navigator.clipboard.writeText(kernel.source);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <section className="instrument kernel-instrument" aria-labelledby="kernel-title">
      <header className="instrument-head kernel-head">
        <div><span className="eyebrow">Generated target</span><h2 id="kernel-title">WGSL kernels</h2></div>
        <button className="copy-button" onClick={copy}>{copied ? <Check size={14} /> : <Copy size={14} />}{copied ? "Copied" : "Copy"}</button>
      </header>
      <div className="kernel-tabs" role="tablist" aria-label="Generated kernels">
        {kernels.map((item, index) => (
          <button key={item.id} role="tab" aria-selected={index === active} onClick={() => setActive(index)}>{item.label}</button>
        ))}
      </div>
      <div className="dispatch-line"><span>DISPATCH</span>{kernel.dispatch}</div>
      <pre className="kernel-code" tabIndex={0}><code>{kernel.source}</code></pre>
    </section>
  );
}
