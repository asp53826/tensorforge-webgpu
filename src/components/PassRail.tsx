import { Braces, CheckCheck, Combine, FileInput, MemoryStick, Scissors, Workflow } from "lucide-react";
import type { PassSnapshot } from "../compiler/types";

const icons = [FileInput, CheckCheck, Braces, Combine, Scissors, MemoryStick];

interface PassRailProps {
  snapshots: PassSnapshot[];
  active: number;
  onChange: (index: number) => void;
}

export function PassRail({ snapshots, active, onChange }: PassRailProps) {
  return (
    <div className="pass-rail" role="tablist" aria-label="Compiler passes">
      {snapshots.map((snapshot, index) => {
        const Icon = icons[index] ?? Workflow;
        return (
          <button
            key={snapshot.id}
            className={`pass-tab ${active === index ? "is-active" : ""} ${active > index ? "is-complete" : ""}`}
            role="tab"
            aria-selected={active === index}
            onClick={() => onChange(index)}
          >
            <span className="pass-index">0{index + 1}</span>
            <Icon size={16} strokeWidth={1.8} aria-hidden="true" />
            <span><strong>{snapshot.name}</strong><small>{snapshot.action}</small></span>
          </button>
        );
      })}
    </div>
  );
}
