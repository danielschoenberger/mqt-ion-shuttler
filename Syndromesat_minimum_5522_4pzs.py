from __future__ import annotations

import json
import os
from itertools import pairwise, permutations
from typing import TYPE_CHECKING

import networkx as nx
from z3 import And, AtLeast, AtMost, Bool, Not, Or, Solver, sat, If, Implies, Sum

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if TYPE_CHECKING:
    Edge = tuple[tuple[int, int], tuple[int, int]]
    Node = tuple[int, int]
    Graph = nx.Graph

# ==========================================
# Graph & Helpers
# ==========================================

def _canon(edge: Edge) -> Edge:
    return tuple(sorted(edge, key=lambda x: (x[0], x[1])))

def create_graph(m: int, n: int, v_size: int, h_size: int) -> Graph:
    m_ext = m + (v_size - 1) * (m - 1)
    n_ext = n + (h_size - 1) * (n - 1)
    g = nx.grid_2d_graph(m_ext, n_ext)
    for i in range(0, m_ext - v_size, v_size):
        for k in range(1, v_size):
            for j in range(n_ext - 1):
                if ((i + k, j), (i + k, j + 1)) in g.edges():
                    g.remove_edge((i + k, j), (i + k, j + 1))
    for i in range(0, n_ext - h_size, h_size):
        for k in range(1, h_size):
            for j in range(m_ext - 1):
                if ((j, i + k), (j + 1, i + k)) in g.edges():
                    g.remove_edge((j, i + k), (j + 1, i + k))
    for i in range(0, m_ext - v_size, v_size):
        for k in range(1, v_size):
            for j in range(0, n_ext - h_size, h_size):
                for p in range(1, h_size):
                    if (i + k, j + p) in g.nodes():
                        g.remove_node((i + k, j + p))
    nx.set_edge_attributes(g, "trap", "edge_type")
    for n_ in g.nodes():
        g.nodes[n_]["node_type"] = "trap_node"
    for i in range(0, m_ext, v_size):
        for j in range(0, n_ext, h_size):
            g.add_node((i, j), node_type="junction_node")
    return g

def map_logical_to_physical_split(edge: Edge, step: int = 2) -> tuple[Edge, Edge]:
    u, v = sorted(edge)
    y1, x1 = u[0] * step, u[1] * step
    y2, x2 = v[0] * step, v[1] * step
    if y1 == y2:  # Horizontal
        e1 = ((y1, x1), (y1, x1 + 1))
        e2 = ((y1, x1 + 1), (y1, x1 + 2))
        return (e1, e2)
    # Vertical
    e1 = ((y1, x1), (y1 + 1, x1))
    e2 = ((y1 + 1, x1), (y1 + 2, x1))
    return (e1, e2)

def create_idc_dict(g: Graph) -> dict[int, Edge]:
    return {i: _canon(e) for i, e in enumerate(g.edges())}

def get_idx(d: dict[int, Edge], e: Edge) -> int:
    target = _canon(e)
    for idx, edge in d.items():
        if edge == target: return idx
    raise ValueError(f"Edge {e} not found")

def get_idc(d: dict[int, Edge], i: int) -> Edge:
    return d[i]

def get_path_between(g: Graph, e1: Edge, e2: Edge) -> list[Edge]:
    path_nodes = nx.shortest_path(g, e1[0], e2[0])
    path_edges = list(pairwise(path_nodes))
    return [e for e in path_edges if _canon(e) != _canon(e1) and _canon(e) != _canon(e2)]

def get_moves_through_node(g: Graph, d: dict[int, Edge], node: Node) -> list[tuple[int, int]]:
    conn_edges = g.edges(node)
    conn_indices = [get_idx(d, e) for e in conn_edges]
    return list(permutations(conn_indices, 2))

def get_junctions(g: Graph, n1: Node, n2: Node, h: int, v: int) -> list[Node]:
    if g.nodes[n1]["node_type"] == "junction_node" and g.nodes[n2]["node_type"] == "junction_node":
        return [n1, n2]
    js: list[Node] = []
    limit = max(h, v)
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for k in range(1, limit + 1):
            nxt = (n1[0] + dy * k, n1[1] + dx * k)
            if nxt not in g: break
            if g.nodes[nxt]["node_type"] == "junction_node":
                js.append(nxt)
                break
    return js

def get_possible_moves_over_junction(nx_g: Graph, edge: Edge, h_size: int, v_size: int) -> list[Edge]:
    n1, n2 = edge
    if nx_g.nodes[n1]["node_type"] != "junction_node":
        jcts = get_junctions(nx_g, n1, n2, h_size, v_size)
    else:
        jcts = get_junctions(nx_g, n2, n1, h_size, v_size)
    poss: list[Edge] = []
    for j in jcts:
        for e in nx_g.edges(j): poss.append(e)
    for j in jcts:
        path_nodes = nx.shortest_path(nx_g, n1, j)
        path_edges = list(pairwise(path_nodes))
        poss_canon = [_canon(p) for p in poss]
        for e_betw in path_edges:
            if _canon(e_betw) in poss_canon:
                poss = [p for p in poss if _canon(p) != _canon(e_betw)]
    return poss

def get_possible_previous_edges_from_junction_move(nx_g: Graph, edge: Edge, h_size: int, v_size: int) -> list[Edge]:
    n1, n2 = edge
    target_jct = n1 if nx_g.nodes[n1]["node_type"] == "junction_node" else n2
    junction_neighbors = list(nx_g.neighbors(target_jct))
    current_arm_node = n2 if n1 == target_jct else n1
    possible_previous_edges: list[Edge] = []
    for neighbor in junction_neighbors:
        if neighbor == current_arm_node: continue
        chain_edges: list[Edge] = []
        curr, prev = neighbor, target_jct
        chain_edges.append(_canon((prev, curr)))
        while nx_g.nodes[curr]["node_type"] != "junction_node":
            neighbors = list(nx_g.neighbors(curr))
            if len(neighbors) == 1: break
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            chain_edges.append(_canon((curr, next_node)))
            prev, curr = curr, next_node
        possible_previous_edges.extend(chain_edges)
    return possible_previous_edges

def create_graph_dict(nx_g: Graph, func, h_size: int, v_size: int, *, edges: str | list[Edge] = "all") -> dict[Edge, list[Edge]]:
    d: dict[Edge, list[Edge]] = {}
    if edges == "all": edges = list(nx_g.edges())
    for e in edges:
        d[e] = func(nx_g, e, h_size, v_size)
        d[tuple(reversed(e))] = func(nx_g, tuple(reversed(e)), h_size, v_size)
    return d

def _fmt_coord(node: Node) -> str:
    return f"({node[0]}, {node[1]})"

def write_viz_json(
    *,
    solver: SyndromeScheduler,
    model,
    filename: str,
    m: int,
    n: int,
    v_size: int,
    h_size: int,
    gate_duration: int = 2,
) -> None:
    idx_to_edge = {i: e for i, e in enumerate(solver.edge_list)}
    inner_pz_edges_block: list[list[str]] = []
    
    # Full PZ Visualization (Both edges of every valid pair)
    for eL, eR in solver.valid_pz_pairs:
        for edge in [eL, eR]:
            u, v = sorted(edge)
            inner_pz_edges_block.append([_fmt_coord(u), _fmt_coord(v)])

    timeline_data: list[dict[str, object]] = []
    
    gate_starts: dict[int, int] = {}
    for k in range(len(solver.gates)):
        for t in range(solver.timesteps):
            if bool(model.evaluate(solver.gate_start[t][k])):
                gate_starts[k] = t
                break

    for t in range(solver.timesteps):
        frame_ions: list[dict[str, object]] = []
        frame_gates: list[dict[str, object]] = []

        for ion in solver.ions:
            for edge_idx in range(len(solver.edge_list)):
                if bool(model.evaluate(solver.states[(t, edge_idx, ion)])):
                    edge = idx_to_edge[edge_idx]
                    u, v = sorted(edge)
                    frame_ions.append({"id": f"$q_{{{ion}}}$", "edge": [_fmt_coord(u), _fmt_coord(v)]})
                    break

        for k, start_t in gate_starts.items():
            if t >= start_t and t < start_t + gate_duration:
                ionA, ionB = solver.gates[k]
                chosen_pair = None
                for eL, eR in solver.valid_pz_pairs:
                    idxL, idxR = get_idx(solver.idc, eL), get_idx(solver.idc, eR)
                    on_lr = bool(model.evaluate(solver.states[(t, idxL, ionA)])) and bool(model.evaluate(solver.states[(t, idxR, ionB)]))
                    on_rl = bool(model.evaluate(solver.states[(t, idxL, ionB)])) and bool(model.evaluate(solver.states[(t, idxR, ionA)]))
                    if on_lr or on_rl:
                        chosen_pair = (eL, eR)
                        break
                
                if chosen_pair:
                    for e in chosen_pair:
                        u, v = sorted(e)
                        frame_gates.append({
                            "id": f"gate_{k}",
                            "type": "OP",
                            "qubits": [f"$q_{{{ionA}}}$", f"$q_{{{ionB}}}$"],
                            "edge": [_fmt_coord(u), _fmt_coord(v)],
                            "duration": gate_duration,
                            "pz": "gate",
                        })

        timeline_data.append({"t": t, "ions": frame_ions, "gates": frame_gates})

    # --- CHANGED: Disable strict border flags ---
    pz_flags = {"top": False, "right": False, "bottom": False, "left": False}

    payload = {
        "architecture": {
            "grid": {"rows": m, "cols": n},
            "sites": {"vertical": v_size, "horizontal": h_size},
            "pzs": pz_flags,
            "innerPZEdges": inner_pz_edges_block,
        },
        "timeline": timeline_data,
    }

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

# ==========================================
# Syndrome Scheduler (Geometric Sort + V=2)
# ==========================================

class SyndromeScheduler:
    def __init__(
        self, 
        m: int = 5, 
        n: int = 4, 
        v_size: int = 2, 
        h_size: int = 2, 
        gate_duration: int = 2,
        # --- CHANGED: Accepts string OR list of edges ---
        pz_config: str | list[tuple[tuple[int, int], tuple[int, int]]] = "all"  
    ) -> None:
        assert v_size == h_size, (
            "map_logical_to_physical_split uses a single step; require v_size == h_size"
        )
        self.m = m
        self.n = n
        self.v_size = v_size
        self.h_size = h_size
        self.gate_duration = gate_duration
        self.pz_config = pz_config

        config_str = "Custom List" if isinstance(self.pz_config, list) else self.pz_config
        print(f"Initializing Scheduler on {self.m}x{self.n} Grid (PZ Config: {config_str})...")

        # 1. Physical Layer
        self.graph: Graph = create_graph(self.m, self.n, v_size=v_size, h_size=h_size)
        self.idc = create_idc_dict(self.graph)
        self.edge_list = [self.idc[i] for i in range(len(self.idc))]

        # 2. Configure PZs
        self.valid_pz_pairs: list[tuple[Edge, Edge]] = []

        def add_pz_if_exists(u_log: tuple[int, int], v_log: tuple[int, int]):
            """Maps logical edge to physical split and adds if valid."""
            e1, e2 = map_logical_to_physical_split((u_log, v_log), step=self.v_size)
            # Ensure e1, e2 are canonical before checking graph
            e1, e2 = _canon(e1), _canon(e2)
            if e1 in self.graph.edges() and e2 in self.graph.edges():
                self.valid_pz_pairs.append((e1, e2))
                print(f"  > Added PZ Pair: {_fmt_coord(e1[0])}-{_fmt_coord(e1[1])} & {_fmt_coord(e2[0])}-{_fmt_coord(e2[1])}")

        # --- LOGIC: Handle List vs String ---
        if isinstance(self.pz_config, list):
            # User provided a custom list of logical edges
            for edge in self.pz_config:
                u, v = sorted(edge) # normalize order
                add_pz_if_exists(u, v)
        else:
            # Presets
            if self.pz_config in ["horizontal", "all"]:
                for y in range(self.m):
                    for x in range(self.n - 1):
                        add_pz_if_exists((y, x), (y, x + 1))

            if self.pz_config in ["vertical", "all"]:
                for x in range(self.n):
                    for y in range(self.m - 1):
                        add_pz_if_exists((y, x), (y + 1, x))
        
        print(f"  > Configured {len(self.valid_pz_pairs)} valid PZ locations.")

        self.ions: list[int] = []
        self.start_pos: dict[int, Edge] = {}
        self.ion_kind: dict[int, str] = {}
        self.gates: list[tuple[int, int]] = []

        # Per-plaquette structural info.
        # Weight-4: {"ancilla_id", "type": "Z"/"X", "weight": 4, "pair1": [a,b], "pair2": [c,d]}
        #   pair indexing convention:
        #     Z-shape (row sweep): pair1=[TL,TR] (top row), pair2=[BL,BR] (bottom row); internal = [left, right]
        #     N-shape (col sweep): pair1=[TL,BL] (left col), pair2=[TR,BR] (right col); internal = [top, bottom]
        # Weight-2: {"ancilla_id", "type", "weight": 2, "gates": [a, b]}
        self.plaquettes: list[dict] = []
        # ancilla_ion_id -> {data_ion_id: gate_idx}
        self.ancilla_neighbor_gate: dict[int, dict[int, int]] = {}
        # ancilla_ion_id -> "Z" or "X"
        self.ancilla_type: dict[int, str] = {}
        # Mixed-type adjacent ancilla pairs that share 2 data qubits: (A, B, q1, q2).
        # Drives the AABB/ABAB/BABA/BBAA consistency rule (forbids ABBA/BAAB).
        self.mixed_adjacent_pairs: list[tuple[int, int, int, int]] = []

        self.s: Solver | None = None
        self.states: dict[tuple[int, int, int], Bool] = {}
        self.gate_start: list[list[Bool]] = []
        self.gate_time: dict[int, object] = {}
        self.timesteps: int = 0
        self.model = None

    def generate_workload(self) -> None:
        """
        Procedurally generates layout using the Exact Geometric Scan logic
        from the user's provided snippet.
        """
        m, n = self.m, self.n
        
        # --- 1. Edge Generation Logic (COPIED FROM SNIPPET) ---
        x0, x1, y0, y1 = 1, n - 1, 1, m - 2

        # Ancilla edges (Vertical)
        ancilla_edges = [((y, x), (y + 1, x)) for x in range(x0, x1) for y in range(y0, y1)]
        ancilla_edges.extend([((0, x), (1, x)) for x in range(x0, x1) if x % 2 == 0])
        ancilla_edges.extend([((y1, x - 1), (y1 + 1, x - 1)) for x in range(x0, x1) if x % 2 == 0])
        ancilla_edges.extend([((y - 1, 0), (y, 0)) for y in range(y0, y1) if y % 2 == 0])
        ancilla_edges.extend([((y, n - 1), (y + 1, n - 1)) for y in range(y0, y1) if y % 2 == 0])

        # Data edges (Horizontal)
        data_edges = [((y, x), (y, x + 1)) for y in range(1, m - 1) for x in range(n - 1)]

        # --- 2. Geometric Sort & ID Assignment ---
        all_ion_edges = []
        for e in data_edges:
            all_ion_edges.append((_canon(e), "data"))
        for e in ancilla_edges:
            all_ion_edges.append((_canon(e), "ancilla"))

        def geometric_sort_key(item):
            edge = item[0]
            (y_a, x_a), (y_b, x_b) = sorted(edge)
            mid_y = (y_a + y_b) / 2
            mid_x = (x_a + x_b) / 2
            return (mid_y, mid_x)

        all_ion_edges.sort(key=geometric_sort_key)

        # Assign IDs and Physical Map
        self.ions = []
        logical_pos_map = {} # ID -> Logical Edge
        
        for idx, (logical_edge, ktype) in enumerate(all_ion_edges):
            ion_id = idx
            self.ions.append(ion_id)
            self.ion_kind[ion_id] = "data" if ktype == "data" else "anc"
            logical_pos_map[ion_id] = logical_edge
            
            # PHYSICAL MAP: Map logical edge to e1 (top/left physical edge)
            e1, _e2 = map_logical_to_physical_split(logical_edge, step=self.v_size)
            self.start_pos[ion_id] = _canon(e1)

        print(f"Generated {len(self.ions)} ions (Expected 17 for m=5,n=4).")

        # --- 3. Dynamic Gate Generation ---
        # Since IDs are procedural, we must find neighbors geometrically.
        
        # Helper to find Data neighbors of an Ancilla
        # An Ancilla at ((y, x), (y+1, x)) touches:
        # TL: ((y, x-1), (y, x))
        # TR: ((y, x), (y, x+1))
        # BL: ((y+1, x-1), (y+1, x))
        # BR: ((y+1, x), (y+1, x+1))
        
        # Build lookup for Data ions: Logical Edge -> Ion ID
        edge_to_data_id = {edge: i for i, edge in logical_pos_map.items() if self.ion_kind[i] == "data"}
        
        for i, edge in logical_pos_map.items():
            if self.ion_kind[i] == "anc":
                # Determine neighbors
                u, v = sorted(edge)
                y, x = u # Top node of vertical edge
                
                # Logical coords of neighbors
                tl = _canon(((y, x - 1), (y, x)))
                tr = _canon(((y, x), (y, x + 1)))
                bl = _canon(((y + 1, x - 1), (y + 1, x)))
                br = _canon(((y + 1, x), (y + 1, x + 1)))
                
                neighbors = []
                if tl in edge_to_data_id: neighbors.append((edge_to_data_id[tl], "TL", tl))
                if tr in edge_to_data_id: neighbors.append((edge_to_data_id[tr], "TR", tr))
                if bl in edge_to_data_id: neighbors.append((edge_to_data_id[bl], "BL", bl))
                if br in edge_to_data_id: neighbors.append((edge_to_data_id[br], "BR", br))
                
                if len(neighbors) < 2: continue
                
                # Determine Type (Z vs X)
                # Standard convention (d=3): (r+c) odd -> Z.
                # Here we have physical coords y, x.
                # Ancilla top node is (y, x).
                # y is row index (0..4). x is col index (0..3).
                # Logic used previously: (y + x) % 2 == 1 -> Z.
                # Let's align with the JSON: a0 at (0,1)-(1,1) was Z. 0+1=1 (Odd).
                # a1 at (1,0)-(2,0) was X. 1+0=1 (Odd). Wait.
                # Let's check neighbors.
                # Z needs N-shape (TL, TR, BL, BR).
                # X needs Z-shape (TL, BL, TR, BR).
                
                # Heuristic: Check if neighbors form a vertical pair (Left/Right) or Horizontal?
                # Actually, simpler:
                # Top boundary -> Z. Left boundary -> X.
                # y=0 -> Z. x=0 -> X.
                # Checkerboard: (y + x) % 2 == 1 -> Z.
                stype = "X" if (y + x) % 2 == 1 else "Z"
                
                # Corrections for boundaries based on standard rotated code:
                # If it has TL, TR but no BL, BR -> Top Boundary -> Z.
                # If it has TL, BL but no TR, BR -> Left Boundary -> X.
                has = set(n[1] for n in neighbors)
                if "TL" in has and "TR" in has and "BL" not in has: stype = "Z" # Top
                elif "BL" in has and "BR" in has and "TL" not in has: stype = "Z" # Bot
                elif "TL" in has and "BL" in has and "TR" not in has: stype = "X" # Left
                elif "TR" in has and "BR" in has and "TL" not in has: stype = "X" # Right
                
                # Record ancilla type
                self.ancilla_type[i] = stype

                # Emit gates by corner label (no sorting — order is chosen by SAT)
                label_to_gate: dict[str, int] = {}
                neighbor_map: dict[int, int] = {}
                for (did, label, _) in neighbors:
                    gate_idx = len(self.gates)
                    self.gates.append((i, did))
                    label_to_gate[label] = gate_idx
                    neighbor_map[did] = gate_idx
                self.ancilla_neighbor_gate[i] = neighbor_map

                # Plaquette record with pair structure for weight-4
                if len(neighbors) == 4:
                    if stype == "Z":
                        # Z-shape: row sweep. pair1 = top row, pair2 = bottom row.
                        # Internal indexing [left, right].
                        pair1 = [label_to_gate["TL"], label_to_gate["TR"]]
                        pair2 = [label_to_gate["BL"], label_to_gate["BR"]]
                    else:  # stype == "X" -> N-shape: column sweep
                        # pair1 = left col, pair2 = right col. Internal [top, bottom].
                        pair1 = [label_to_gate["TL"], label_to_gate["BL"]]
                        pair2 = [label_to_gate["TR"], label_to_gate["BR"]]
                    self.plaquettes.append({
                        "ancilla_id": i,
                        "type": stype,
                        "weight": 4,
                        "pair1": pair1,
                        "pair2": pair2,
                    })
                else:
                    # Weight-2 (or any other weight): emit gates, no intra-plaquette constraint.
                    self.plaquettes.append({
                        "ancilla_id": i,
                        "type": stype,
                        "weight": len(neighbors),
                        "gates": list(label_to_gate.values()),
                    })

        # Build mixed-type adjacent consistency pairs
        self._build_mixed_adjacency()

    def _build_mixed_adjacency(self) -> None:
        """Enumerate opposite-type ancilla pairs that share >= 2 data qubits.
        Each recorded tuple (A, B, q1, q2) drives the consistency constraint
        (t_{A,q1} < t_{B,q1}) iff (t_{A,q2} < t_{B,q2}).
        """
        self.mixed_adjacent_pairs = []
        ancilla_ids = list(self.ancilla_neighbor_gate.keys())
        for a_i, A in enumerate(ancilla_ids):
            for B in ancilla_ids[a_i + 1:]:
                if self.ancilla_type[A] == self.ancilla_type[B]:
                    continue  # same-type: CNOTs commute, no constraint
                shared = sorted(
                    set(self.ancilla_neighbor_gate[A].keys())
                    & set(self.ancilla_neighbor_gate[B].keys())
                )
                if len(shared) < 2:
                    continue
                for qi in range(len(shared)):
                    for qj in range(qi + 1, len(shared)):
                        self.mixed_adjacent_pairs.append((A, B, shared[qi], shared[qj]))
        print(f"  > Found {len(self.mixed_adjacent_pairs)} mixed-type adjacency consistency pairs.")

    def _init_sat(self, timesteps: int) -> None:
        self.timesteps = timesteps
        self.s = Solver()
        self.states = {}
        for t in range(timesteps):
            for e in range(len(self.edge_list)):
                for i in self.ions: self.states[(t, e, i)] = Bool(f"s_{t}_{e}_{i}")
        num_gates = len(self.gates)
        self.gate_start = [[Bool(f"start_gate_{t}_{k}") for k in range(num_gates)] for t in range(timesteps)]
        # Symbolic integer time of each gate (gate_start is one-hot over valid starts).
        self.gate_time = {
            k: Sum([If(self.gate_start[t][k], t, 0) for t in range(timesteps)])
            for k in range(num_gates)
        }

    def _add_movement_constraints(self) -> None:
        junction_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]["node_type"] == "junction_node"]
        junction_edges = [list(self.graph.edges(n)) for n in junction_nodes]
        junction_edges_flat = [(sorted(e)[0], sorted(e)[1]) for sub in junction_edges for e in sub]
        junction_move_dict = create_graph_dict(self.graph, get_possible_moves_over_junction, self.h_size, self.v_size)
        prev_junction_move_dict = create_graph_dict(self.graph, get_possible_previous_edges_from_junction_move, self.h_size, self.v_size, edges=junction_edges_flat)

        # 1. Start
        for e_idx in range(len(self.edge_list)):
            edge = self.idc[e_idx]
            ions_here = [i for i, pos in self.start_pos.items() if _canon(pos) == edge]
            for i in self.ions: self.s.add(self.states[(0, e_idx, i)] if i in ions_here else Not(self.states[(0, e_idx, i)]))
        
        # 2. Conservation
        for t in range(1, self.timesteps):
            for i in self.ions:
                self.s.add(AtMost(*[self.states[(t, e, i)] for e in range(len(self.edge_list))], 1))
                self.s.add(AtLeast(*[self.states[(t, e, i)] for e in range(len(self.edge_list))], 1))

        # 3. Move
        for t in range(self.timesteps - 1):
            for i in self.ions:
                for e_idx in range(len(self.edge_list)):
                    edge = self.idc[e_idx]
                    possible_next = list(junction_move_dict[edge])
                    for neighbor in self.graph.edges(edge): possible_next.append(neighbor)
                    next_conds = []
                    for n_edge in possible_next:
                        n_idx = get_idx(self.idc, n_edge)
                        path_edges = get_path_between(self.graph, edge, n_edge)
                        path_clear = And(*[Not(self.states[(t, get_idx(self.idc, pe), oi)]) for pe in path_edges for oi in self.ions])
                        next_conds.append(And(self.states[(t + 1, n_idx, i)], path_clear))
                    self.s.add(Or(Not(self.states[(t, e_idx, i)]), And(self.states[(t, e_idx, i)], Or(*next_conds))))

        # 4. Junction Cap
        for t in range(1, self.timesteps):
            for node in junction_nodes:
                self.s.add(AtMost(*[And(self.states[(t, get_idx(self.idc, je), i)], Or(*[self.states[(t-1, get_idx(self.idc, prev), i)] for prev in prev_junction_move_dict[je]])) for je in self.graph.edges(node) for i in self.ions], 1))

        # 5. Anti-swap
        for t in range(1, self.timesteps):
            for n in self.graph.nodes():
                moves = get_moves_through_node(self.graph, self.idc, n)
                if moves: self.s.add(AtMost(*[And(self.states[(t, m[1], i)], self.states[(t - 1, m[0], i)]) for i in self.ions for m in moves], 1))

        # 6. Edge Cap
        for t in range(1, self.timesteps):
            for e_idx in range(len(self.edge_list)): self.s.add(AtMost(*[self.states[(t, e_idx, i)] for i in self.ions], 1))

    def _add_gate_constraints(self) -> None:
        valid_start = range(1, self.timesteps - self.gate_duration + 1)
        for k in range(len(self.gates)):
            self.s.add(AtMost(*[self.gate_start[t][k] for t in range(self.timesteps)], 1))
            self.s.add(AtLeast(*[self.gate_start[t][k] for t in valid_start], 1))
            for t in range(self.timesteps):
                if t not in valid_start: self.s.add(Not(self.gate_start[t][k]))

        pz_pair_idx = [(get_idx(self.idc, eL), get_idx(self.idc, eR)) for (eL, eR) in self.valid_pz_pairs]
        for t in range(self.timesteps):
            for k, (ionA, ionB) in enumerate(self.gates):
                if t > self.timesteps - self.gate_duration: continue
                or_clauses = []
                for (idxL, idxR) in pz_pair_idx:
                    stay_LR = And(*[And(self.states[(t+dt, idxL, ionA)], self.states[(t+dt, idxR, ionB)]) for dt in range(self.gate_duration)])
                    stay_RL = And(*[And(self.states[(t+dt, idxL, ionB)], self.states[(t+dt, idxR, ionA)]) for dt in range(self.gate_duration)])
                    or_clauses.append(stay_LR); or_clauses.append(stay_RL)
                self.s.add(Implies(self.gate_start[t][k], Or(*or_clauses)))

        ion_to_gate = {}
        for k, (a, b) in enumerate(self.gates):
            ion_to_gate.setdefault(a, []).append(k)
            ion_to_gate.setdefault(b, []).append(k)
        for t in range(self.timesteps):
            for ion, gates in ion_to_gate.items():
                if len(gates) < 2: continue
                active = []
                for k in gates:
                    relevant = [self.gate_start[t-dt][k] for dt in range(self.gate_duration) if t-dt >= 0]
                    if relevant: active.append(Or(*relevant))
                if active: self.s.add(AtMost(*active, 1))

        # --- Intra-plaquette ordering for weight-4 ancillas ---
        # Replaces the old strict chain (0->1->2->3). Encodes exactly the 4 aligned
        # orderings per plaquette: pair1-before-pair2 OR pair2-before-pair1, AND the
        # two pairs' internal directions agree (no "kinked" orderings).
        for plaq in self.plaquettes:
            if plaq["weight"] != 4:
                continue
            pair1 = plaq["pair1"]
            pair2 = plaq["pair2"]
            pair1_first = And(*[
                self.gate_time[g2] >= self.gate_time[g1] + self.gate_duration
                for g1 in pair1 for g2 in pair2
            ])
            pair2_first = And(*[
                self.gate_time[g1] >= self.gate_time[g2] + self.gate_duration
                for g1 in pair1 for g2 in pair2
            ])
            self.s.add(Or(pair1_first, pair2_first))
            a, b = pair1
            c, d = pair2
            self.s.add(
                (self.gate_time[a] < self.gate_time[b]) ==
                (self.gate_time[c] < self.gate_time[d])
            )

        # --- Cross-plaquette consistency on shared qubits ---
        # For each pair of opposite-type ancillas sharing qubits q1, q2, the relative
        # order on q1 and q2 must match: (t_{A,q1} < t_{B,q1}) iff (t_{A,q2} < t_{B,q2}).
        # Forbids ABBA / BAAB interleavings that would produce incorrect syndromes.
        for (A, B, q1, q2) in self.mixed_adjacent_pairs:
            kA1 = self.ancilla_neighbor_gate[A][q1]
            kA2 = self.ancilla_neighbor_gate[A][q2]
            kB1 = self.ancilla_neighbor_gate[B][q1]
            kB2 = self.ancilla_neighbor_gate[B][q2]
            self.s.add(
                (self.gate_time[kA1] < self.gate_time[kB1]) ==
                (self.gate_time[kA2] < self.gate_time[kB2])
            )

    def validate_solution(self) -> bool:
        """Post-solve sanity check for the new pair/alignment/consistency constraints."""
        if self.model is None:
            return False
        # Extract integer time of each gate
        gt: dict[int, int] = {}
        for k in range(len(self.gates)):
            for t in range(self.timesteps):
                if bool(self.model.evaluate(self.gate_start[t][k])):
                    gt[k] = t
                    break
        ok = True
        # Intra-plaquette checks
        for plaq in self.plaquettes:
            if plaq["weight"] != 4:
                continue
            pair1, pair2 = plaq["pair1"], plaq["pair2"]
            p1_max = max(gt[g] for g in pair1)
            p2_max = max(gt[g] for g in pair2)
            p1_min = min(gt[g] for g in pair1)
            p2_min = min(gt[g] for g in pair2)
            if not (p1_max + self.gate_duration <= p2_min or
                    p2_max + self.gate_duration <= p1_min):
                print(f"  [validate] pair-order violation on ancilla {plaq['ancilla_id']}")
                ok = False
            a, b = pair1
            c, d = pair2
            if (gt[a] < gt[b]) != (gt[c] < gt[d]):
                print(f"  [validate] alignment violation on ancilla {plaq['ancilla_id']}")
                ok = False
        # Cross-plaquette consistency checks
        for (A, B, q1, q2) in self.mixed_adjacent_pairs:
            kA1 = self.ancilla_neighbor_gate[A][q1]
            kA2 = self.ancilla_neighbor_gate[A][q2]
            kB1 = self.ancilla_neighbor_gate[B][q1]
            kB2 = self.ancilla_neighbor_gate[B][q2]
            if (gt[kA1] < gt[kB1]) != (gt[kA2] < gt[kB2]):
                print(f"  [validate] consistency violation on ancillas {A},{B} qubits {q1},{q2}")
                ok = False
        if ok:
            print("  [validate] all intra-plaquette and cross-plaquette constraints hold.")
        return ok

    def solve(self, *, t_min: int = 12, t_max: int = 80, step: int = 2) -> bool:
        if not self.gates: self.generate_workload()
        for T in range(t_min, t_max + 1, step):
            self._init_sat(T)
            self._add_movement_constraints()
            self._add_gate_constraints()
            print(f"Checking T={T}...", end="\r")
            if self.s.check() == sat:
                print(f"SAT found at T={T}      ")
                self.model = self.s.model()
                self.validate_solution()
                return True
        return False

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    d = 3 
    
    if d == 3:
        m, n = 5, 4
    elif d == 5:
        m, n = 7, 6
    else:
        raise ValueError("Unsupported d value.")
    
    # ----------------------------------------------------
    # CONFIGURATION
    # ----------------------------------------------------
    # Option 1: Preset String ("all", "vertical", "horizontal")
    # selected_pz_config = "horizontal"
    
    # Option 2: Custom List of Logic Edges
    # Example: A few specific horizontal and vertical zones
    custom_pzs = [
        ((1, 0), (2, 0)), # Left
        ((1, 3), (2, 3)), # Right
        ((1, 1),(1, 2)), # Top
        ((3, 1), (3, 2)), # Bottom
    ]
    
    # Set this variable to either the string or the list
    selected_pz_config = custom_pzs
    
    sched = SyndromeScheduler(
        m=m, n=n, v_size=2, h_size=2, gate_duration=2, 
        pz_config=selected_pz_config
    )
    
    # step=2 ensures even timesteps (12, 14...)
    # step=1 finds the absolute minimum (e.g. 11)
    ok = sched.solve(t_min=11, t_max=100, step=1)
    
    print(f"Solution Found: {ok}")
    if ok:
        suffix = "custom" if isinstance(selected_pz_config, list) else selected_pz_config
        out_json = f"benchmarks/solution_d{d}_{suffix}.json"
        write_viz_json(solver=sched, model=sched.model, filename=out_json, m=sched.m, n=sched.n, v_size=2, h_size=2)
        print(f"Saved to {out_json}")