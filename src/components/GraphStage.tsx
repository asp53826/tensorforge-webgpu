import type { TensorGraph, TensorNode } from "../compiler/types";

interface GraphStageProps {
  graph: TensorGraph;
  selected: string;
  onSelect: (node: TensorNode) => void;
}

const WIDTH = 980;
const NODE_WIDTH = 132;
const NODE_HEIGHT = 58;

function layoutGraph(graph: TensorGraph) {
  const depth = new Map<string, number>();
  for (const node of graph.nodes) {
    const inputDepths = node.inputs.map((input) => depth.get(input) ?? 0);
    depth.set(node.id, node.inputs.length ? Math.max(...inputDepths) + 1 : 0);
  }
  const columns = new Map<number, TensorNode[]>();
  for (const node of graph.nodes) {
    const level = depth.get(node.id) ?? 0;
    columns.set(level, [...(columns.get(level) ?? []), node]);
  }
  const maxDepth = Math.max(...depth.values(), 1);
  const positions = new Map<string, { x: number; y: number }>();
  for (const [level, nodes] of columns) {
    const x = 28 + (level / maxDepth) * (WIDTH - NODE_WIDTH - 56);
    const span = Math.min(280, (nodes.length - 1) * 72);
    nodes.forEach((node, index) => {
      const y = 184 - span / 2 + index * (nodes.length > 1 ? span / (nodes.length - 1) : 0);
      positions.set(node.id, { x, y });
    });
  }
  return positions;
}

const shortOp = (op: TensorNode["op"]) => {
  const labels: Partial<Record<TensorNode["op"], string>> = {
    fusedLinearGelu: "FUSED + GELU",
    fusedLinear: "FUSED LINEAR",
    debugTap: "DEBUG TAP",
    parameter: "PARAMETER"
  };
  return labels[op] ?? op.toUpperCase();
};

export function GraphStage({ graph, selected, onSelect }: GraphStageProps) {
  const positions = layoutGraph(graph);
  const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
  return (
    <div className="graph-scroller" aria-label={`Interactive ${graph.name} compiler graph`}>
      <svg className="graph-stage" viewBox={`0 0 ${WIDTH} 430`} role="group">
        <defs>
          <pattern id="graph-grid" width="24" height="24" patternUnits="userSpaceOnUse">
            <path d="M 24 0 L 0 0 0 24" fill="none" stroke="currentColor" strokeOpacity=".08" />
          </pattern>
          <filter id="soft-glow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <rect width={WIDTH} height="430" fill="url(#graph-grid)" />
        <g className="graph-edges">
          {graph.nodes.flatMap((node) =>
            node.inputs.map((input) => {
              const from = positions.get(input);
              const to = positions.get(node.id);
              if (!from || !to || !nodeMap.has(input)) return null;
              const x1 = from.x + NODE_WIDTH;
              const y1 = from.y + NODE_HEIGHT / 2;
              const x2 = to.x;
              const y2 = to.y + NODE_HEIGHT / 2;
              const bend = Math.max(28, (x2 - x1) * 0.48);
              return (
                <path
                  key={`${input}-${node.id}`}
                  className={`graph-edge ${node.op.startsWith("fused") ? "is-fused" : ""}`}
                  d={`M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`}
                />
              );
            })
          )}
        </g>
        {graph.nodes.map((node, index) => {
          const position = positions.get(node.id) ?? { x: 0, y: 0 };
          const active = selected === node.id;
          return (
            <g
              key={node.id}
              className={`graph-node op-${node.op} ${active ? "is-selected" : ""}`}
              transform={`translate(${position.x} ${position.y})`}
              role="button"
              tabIndex={0}
              aria-label={`${node.label}, ${shortOp(node.op)}, shape ${node.shape.join(" by ")}`}
              onClick={() => onSelect(node)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(node);
                }
              }}
              style={{ "--node-delay": `${index * 28}ms` } as React.CSSProperties}
            >
              <rect className="node-shadow" x="3" y="4" width={NODE_WIDTH} height={NODE_HEIGHT} rx="7" />
              <rect className="node-body" width={NODE_WIDTH} height={NODE_HEIGHT} rx="7" />
              <rect className="node-signal" width="4" height={NODE_HEIGHT} rx="2" />
              <text className="node-op" x="14" y="20">{shortOp(node.op)}</text>
              <text className="node-label" x="14" y="40">{node.label}</text>
              <circle className="node-port" cx={NODE_WIDTH} cy={NODE_HEIGHT / 2} r="3" />
            </g>
          );
        })}
      </svg>
    </div>
  );
}
