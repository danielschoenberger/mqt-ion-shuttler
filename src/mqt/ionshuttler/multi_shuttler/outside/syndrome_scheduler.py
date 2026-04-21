from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing_extensions import Literal
from collections import defaultdict
from itertools import combinations
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge


class BoundaryOrientation(str, Enum):
    UNDECIDED = "undecided"
    A_FIRST = "a_first"
    B_FIRST = "b_first"


@dataclass(frozen=True)
class SyndromeGate:
    ancilla: int
    data: int
    plaquette: int
    step_idx: int

    @property
    def gate_tuple(self) -> tuple[int, int]:
        return (self.ancilla, self.data)


@dataclass(frozen=True)
class SharedBoundary:
    boundary_str: str
    plaquette_a: int
    plaquette_b: int
    shared_data_qubits: tuple[int, int]


@dataclass
class SyndromeSchedulerState:
    plaquette_programs: dict[int, list[SyndromeGate]]
    plaquette_ptr: dict[int, int]
    shared_boundaries: dict[int, SharedBoundary]
    qubit_to_boundary: dict[tuple[int, int], int]  # (plaquette, data_qubit) -> boundary id
    boundary_orientation: dict[int, BoundaryOrientation]  # boundary id -> orientation
    executed_gates: set[tuple[int, int]] = field(default_factory=set)    # executed gates as (plaquette, data_qubit) - creates empty set for each new instance
    predecessors: dict[int, dict[SyndromeGate, set[tuple[int, int]]]] = field(default_factory=dict)  # for each plaquette: gate to set of predecessor gates

    def copy(self) -> "SyndromeSchedulerState":
        return deepcopy(self)
    

def init_syndrome_scheduler_state(
        plaquette_programs: dict[int, list[SyndromeGate]],
        shared_boundaries: dict[int, SharedBoundary], 
        predecessors: dict[int, dict[SyndromeGate, set[tuple[int, int]]]]
    ) -> SyndromeSchedulerState:
    
    plaquette_ptr = {}
    qubit_to_boundary = {}
    boundary_orientation = {}

    for plaquette in plaquette_programs:
        plaquette_ptr[plaquette] = 0
    for boundary_str, boundary in shared_boundaries.items():
        for data_qubit in boundary.shared_data_qubits:
            qubit_to_boundary[(boundary.plaquette_a, data_qubit)] = boundary_str
            qubit_to_boundary[(boundary.plaquette_b, data_qubit)] = boundary_str
        boundary_orientation[boundary_str] = BoundaryOrientation.UNDECIDED


    state = SyndromeSchedulerState(
        plaquette_programs=plaquette_programs,
        plaquette_ptr=plaquette_ptr,
        shared_boundaries=shared_boundaries,
        qubit_to_boundary=qubit_to_boundary,
        boundary_orientation=boundary_orientation,
        predecessors=predecessors,
    )

    return state


Basis = Literal["X", "Z"]
@dataclass(frozen=True, slots=True)
class Plaquette:
    plaquette_id: int
    row: int
    col: int
    ancilla: int
    data_qubits: tuple[int, ...]
    basis: Basis


def build_surface_code_pattern(
    distance: int
):
    """Build a single round of surface code syndrome extraction."""
    data_qubit_id = 0
    plaquette_counter = 0
    data_qubits = {}
    ancilla_qubits = {}
    ancilla_offset = distance * distance
    plaquettes: dict[int, Plaquette] = {}

    def bulk_basis(i: int, j: int) -> Basis:
        return "X" if (i + j) % 2 == 0 else "Z"

    def opposite_basis(basis: Basis) -> Basis:
        return "Z" if basis == "X" else "X"

    for i in range(distance):
        for j in range(distance):
            data_qubits[data_qubit_id] = (i, j)

            if i != distance - 1 and j != distance - 1:
                plaquette_id = plaquette_counter
                plaquette_counter += 1

                ancilla_id = ancilla_offset + i * (distance - 1) + j
                ancilla_qubits[ancilla_id] = (i + 0.5, j + 0.5)

                plaquettes[plaquette_id] = Plaquette(
                    plaquette_id=plaquette_id,
                    row=i,
                    col=j,
                    ancilla=ancilla_id,
                    data_qubits=(
                        data_qubit_id,
                        data_qubit_id + 1,
                        data_qubit_id + distance,
                        data_qubit_id + distance + 1,
                    ),
                    basis=bulk_basis(i, j),
                )

            data_qubit_id += 1

    next_ancilla_id = ancilla_offset + (distance - 1) * (distance - 1)

    def add_boundary_plaquette(
        ancilla_pos: tuple[float, float],
        data_ids: tuple[int, int],
        neighboring_bulk_basis: Basis,
        row: float,
        col: float,
    ) -> None:
        nonlocal plaquette_counter, next_ancilla_id

        ancilla_id = next_ancilla_id
        next_ancilla_id += 1
        ancilla_qubits[ancilla_id] = ancilla_pos

        plaquettes[plaquette_counter] = Plaquette(
            plaquette_id=plaquette_counter,
            row=row,
            col=col,
            ancilla=ancilla_id,
            data_qubits=data_ids,
            basis=opposite_basis(neighboring_bulk_basis),
        )
        plaquette_counter += 1

    # top boundary
    # touches bulk plaquette at (0, j)
    for j in range(1, distance - 1, 2):
        add_boundary_plaquette(
            ancilla_pos=(-0.5, j + 0.5),
            data_ids=(j, j + 1),
            neighboring_bulk_basis=bulk_basis(0, j),
            row=-0.5,
            col=j + 0.5,
        )

    # bottom boundary
    # touches bulk plaquette at (distance-2, j)
    for j in range(0, distance - 1, 2):
        base = (distance - 1) * distance + j
        add_boundary_plaquette(
            ancilla_pos=(distance - 0.5, j + 0.5),
            data_ids=(base, base + 1),
            neighboring_bulk_basis=bulk_basis(distance - 2, j),
            row=distance - 0.5,
            col=j + 0.5,
        )

    # left boundary
    # touches bulk plaquette at (i, 0)
    for i in range(0, distance - 1, 2):
        top = i * distance
        bottom = (i + 1) * distance
        add_boundary_plaquette(
            ancilla_pos=(i + 0.5, -0.5),
            data_ids=(top, bottom),
            neighboring_bulk_basis=bulk_basis(i, 0),
            row=i + 0.5,
            col=-0.5,
        )

    # right boundary
    # touches bulk plaquette at (i, distance-2)
    for i in range(1, distance - 1, 2):
        top = i * distance + (distance - 1)
        bottom = (i + 1) * distance + (distance - 1)
        add_boundary_plaquette(
            ancilla_pos=(i + 0.5, distance - 0.5),
            data_ids=(top, bottom),
            neighboring_bulk_basis=bulk_basis(i, distance - 2),
            row=i + 0.5,
            col=distance - 0.5,
        )

    return data_qubits, ancilla_qubits, plaquettes

def build_surface_code_programs(
        plaquettes: dict[int, Plaquette],
        x_hook_schedule: str = "N",
        z_hook_schedule: str = "Z",
    ) -> tuple[dict[int, list[SyndromeGate]], dict[int, dict[SyndromeGate, set[tuple[int, int]]]]]:
        
    programs = {}
    n_pattern = [1, 3, 0, 2]
    z_pattern = [1, 0, 3, 2]

    x_schedule = n_pattern if x_hook_schedule == "N" else z_pattern
    z_schedule = n_pattern if z_hook_schedule == "N" else z_pattern

    predecessors: dict[int, dict[SyndromeGate, set[tuple[int, int]]]] = {i: {} for i in plaquettes.keys()}

    for plaquette in plaquettes.values():
        program = []
        if len(plaquette.data_qubits) == 4:
            pattern = plaquette.basis
            if pattern == "X":
                reordered_data_qubits = [plaquette.data_qubits[i] for i in x_schedule]
            else:
                reordered_data_qubits = [plaquette.data_qubits[i] for i in z_schedule]
            
            executed_gate_tuples: set[tuple[int, int]] = set()
            for step_idx, data_qubit in enumerate(reordered_data_qubits):
                program.append(
                    SyndromeGate(
                        ancilla=plaquette.ancilla,
                        data=data_qubit,
                        plaquette=plaquette.plaquette_id,
                        step_idx=step_idx,
                    )
                )
                predecessors[plaquette.plaquette_id][program[-1]] = executed_gate_tuples.copy()
                executed_gate_tuples.add(program[-1].gate_tuple)
        else:
            # boundary plaquette with only 2 data qubits, no hooks, both gates can be executed in either order
            for step_idx, data_qubit in enumerate(plaquette.data_qubits):
                boundary_gate = SyndromeGate(
                    ancilla=plaquette.ancilla,
                    data=data_qubit,
                    plaquette=plaquette.plaquette_id,
                    step_idx=step_idx,
                )
                program.append(boundary_gate)
                predecessors[plaquette.plaquette_id][boundary_gate] = set()

        programs[plaquette.plaquette_id] = program
    return programs, predecessors


def build_shared_boundaries(
    plaquettes: dict[int, Plaquette],
) -> dict[int, SharedBoundary]:

    # map each data qubit to the plaquettes touching it
    data_to_plaquettes: dict[int, list[int]] = defaultdict(list)

    for plaquette in plaquettes.values():
        for dq in plaquette.data_qubits:
            data_to_plaquettes[dq].append(plaquette.plaquette_id)

    # temporary structure:
    # pair of plaquettes -> shared data qubits
    shared_map: dict[tuple[int, int], list[int]] = defaultdict(list)

    for dq, touching_plaquettes in data_to_plaquettes.items():

        # all plaquette pairs touching this qubit
        for p1, p2 in combinations(sorted(touching_plaquettes), 2):
            shared_map[(p1, p2)].append(dq)

    # construct SharedBoundary objects
    shared_boundaries: dict[int, SharedBoundary] = {}

    boundary_counter = 0

    for (p1, p2), shared_dqs in shared_map.items():

        # only keep true edges (two shared data qubits)
        if len(shared_dqs) != 2:
            continue

        shared_boundaries[boundary_counter] = SharedBoundary(
            boundary_str=f"B{boundary_counter}",
            plaquette_a=p1,
            plaquette_b=p2,
            shared_data_qubits=tuple(sorted(shared_dqs)),
        )

        boundary_counter += 1

    return shared_boundaries


def is_gate_allowed(state: SyndromeSchedulerState, gate: SyndromeGate) -> bool:
    prog = state.plaquette_programs[gate.plaquette]

    # the gate belongs to this plaquette and has not already been executed.
    if gate not in prog or gate.gate_tuple in state.executed_gates:
        return False

    # For a fully ordered plaquette, this set contains the previous gates.
    # For an edge plaquette with two unordered gates, sets are empty.
    preds = state.predecessors.get(gate.plaquette, {}).get(gate, set())

    if not preds.issubset(state.executed_gates):
        return False

    boundary_id = state.qubit_to_boundary.get((gate.plaquette, gate.data))
    if boundary_id is None:
        return True

    boundary = state.shared_boundaries[boundary_id]
    orientation = state.boundary_orientation[boundary_id]

    if orientation == BoundaryOrientation.UNDECIDED:
        return True

    if gate.plaquette == boundary.plaquette_a:
        my_side = BoundaryOrientation.A_FIRST
        other_plaquette = boundary.plaquette_b
    else:
        my_side = BoundaryOrientation.B_FIRST
        other_plaquette = boundary.plaquette_a

    if orientation == my_side:
        return True

    # check if the other gate on the other plaquette has been executed (since here it is already decided that the other goes first)
    other_gate_tuple = None
    for g in state.plaquette_programs[other_plaquette]:
        if g.data == gate.data:
            other_gate_tuple = g.gate_tuple
            break

    if other_gate_tuple is None:
        return False

    return other_gate_tuple in state.executed_gates


def get_front_layer(state: SyndromeSchedulerState) -> list[SyndromeGate]:
    front = []
    for plaquette, prog in state.plaquette_programs.items():
        ptr = state.plaquette_ptr[plaquette]
        if ptr < len(prog):
            gate = prog[ptr]
            if is_gate_allowed(state, gate):
                front.append(gate)
    return front


def execute_gate(state: SyndromeSchedulerState, gate: SyndromeGate) -> None:
    edge_id = state.qubit_to_boundary.get((gate.plaquette, gate.data))
    if edge_id is not None and state.boundary_orientation[edge_id] == BoundaryOrientation.UNDECIDED:
        boundary = state.shared_boundaries[edge_id]
        if gate.plaquette == boundary.plaquette_a:
            state.boundary_orientation[edge_id] = BoundaryOrientation.A_FIRST
        else:
            state.boundary_orientation[edge_id] = BoundaryOrientation.B_FIRST

    ptr = state.plaquette_ptr[gate.plaquette]
    prog = state.plaquette_programs[gate.plaquette]
    if ptr >= len(prog) or prog[ptr] != gate:
        msg = f"Wrong next gate for plaquette {gate.plaquette}"
        raise ValueError(msg)

    state.plaquette_ptr[gate.plaquette] += 1
    state.executed_gates.add(gate.gate_tuple)



def assign_plaquettes_to_pzs(
    plaquettes: dict[int, Plaquette],
    shared_boundaries: dict[int, SharedBoundary],
    graph,
    strategy: str = "geometric",
) -> dict[int, str]:
    """Assign each plaquette to a processing zone.

    Parameters
    ----------
    plaquettes : mapping of plaquette id → Plaquette
    shared_boundaries : mapping of boundary id → SharedBoundary
    graph : Graph with ``.pzs``, ``.state``, and path-finding support
    strategy : ``"geometric"`` (distance-based) or ``"round_robin"``

    Returns
    -------
    dict mapping plaquette_id → pz_name
    """
    from .cycles import find_path_edge_to_edge
    from .scheduling import get_edge_idc_by_pz_name

    if strategy == "round_robin":
        plaq_ids = sorted(plaquettes.keys())
        return {
            plaq_id: graph.pzs[i % len(graph.pzs)].name
            for i, plaq_id in enumerate(plaq_ids)
        }

    # --- geometric strategy ---

    # 1. Separate bulk (4 data qubits) from boundary (2 data qubits)
    bulk_plaqs = {pid: p for pid, p in plaquettes.items() if len(p.data_qubits) == 4}
    boundary_plaqs = {pid: p for pid, p in plaquettes.items() if len(p.data_qubits) == 2}

    # 2. Compute distance from each bulk plaquette's ancilla to each PZ
    plaq_pz_dist: dict[int, dict[str, int]] = {}
    for pid, plaq in bulk_plaqs.items():
        ancilla_edge = graph.state[plaq.ancilla]
        dists: dict[str, int] = {}
        for pz in graph.pzs:
            pz_parking = get_edge_idc_by_pz_name(graph, pz.name)
            path = find_path_edge_to_edge(graph, ancilla_edge, pz_parking)
            dists[pz.name] = len(path) if path is not None else 10**6
        plaq_pz_dist[pid] = dists

    # 3. Assign each bulk plaquette to nearest PZ
    plaquette_to_pz: dict[int, str] = {}
    for pid, dists in plaq_pz_dist.items():
        plaquette_to_pz[pid] = min(dists, key=dists.get)

    # 4. Rebalance: ensure no PZ is overloaded (>120% of fair share)
    def _gate_count(pid: int) -> int:
        return len(plaquettes[pid].data_qubits)  # 4 for bulk, 2 for boundary

    total_bulk_gates = sum(_gate_count(pid) for pid in bulk_plaqs)
    fair_share = total_bulk_gates / len(graph.pzs)
    threshold = fair_share * 1.2

    # Build adjacency for rebalance cost estimation
    adjacency: dict[int, set[int]] = defaultdict(set)
    for _bid, boundary in shared_boundaries.items():
        if boundary.plaquette_a in bulk_plaqs and boundary.plaquette_b in bulk_plaqs:
            adjacency[boundary.plaquette_a].add(boundary.plaquette_b)
            adjacency[boundary.plaquette_b].add(boundary.plaquette_a)

    for _iteration in range(len(bulk_plaqs)):
        # Compute load per PZ
        pz_load: dict[str, int] = {pz.name: 0 for pz in graph.pzs}
        for pid in bulk_plaqs:
            pz_load[plaquette_to_pz[pid]] += _gate_count(pid)

        overloaded = [pz_name for pz_name, load in pz_load.items() if load > threshold]
        if not overloaded:
            break

        moved = False
        for pz_name in overloaded:
            # Plaquettes on this PZ, sorted by fewest same-PZ neighbors (cheapest to move)
            candidates = [pid for pid, pn in plaquette_to_pz.items() if pn == pz_name]
            candidates.sort(
                key=lambda pid: sum(
                    1 for nb in adjacency.get(pid, set()) if plaquette_to_pz.get(nb) == pz_name
                )
            )
            for pid in candidates:
                # Find nearest underloaded PZ
                underloaded = [
                    pn for pn, load in pz_load.items()
                    if pn != pz_name and load + _gate_count(pid) <= threshold
                ]
                if not underloaded:
                    continue
                best_pz = min(underloaded, key=lambda pn: plaq_pz_dist[pid][pn])
                plaquette_to_pz[pid] = best_pz
                pz_load[pz_name] -= _gate_count(pid)
                pz_load[best_pz] += _gate_count(pid)
                moved = True
                if pz_load[pz_name] <= threshold:
                    break
        if not moved:
            break

    # 5. Assign boundary plaquettes to the same PZ as their adjacent bulk plaquette
    data_to_bulk: dict[int, int] = {}
    for pid, plaq in bulk_plaqs.items():
        for dq in plaq.data_qubits:
            if dq not in data_to_bulk:
                data_to_bulk[dq] = pid

    for pid, plaq in boundary_plaqs.items():
        # Find which bulk plaquette shares data qubits with this boundary plaquette
        neighbor_bulk = None
        for dq in plaq.data_qubits:
            if dq in data_to_bulk:
                neighbor_bulk = data_to_bulk[dq]
                break
        if neighbor_bulk is not None and neighbor_bulk in plaquette_to_pz:
            plaquette_to_pz[pid] = plaquette_to_pz[neighbor_bulk]
        else:
            # Fallback: assign to nearest PZ by ancilla distance
            ancilla_edge = graph.state[plaq.ancilla]
            best_pz = graph.pzs[0].name
            best_dist = 10**6
            for pz in graph.pzs:
                pz_parking = get_edge_idc_by_pz_name(graph, pz.name)
                path = find_path_edge_to_edge(graph, ancilla_edge, pz_parking)
                d = len(path) if path is not None else 10**6
                if d < best_dist:
                    best_dist = d
                    best_pz = pz.name
            plaquette_to_pz[pid] = best_pz

    return plaquette_to_pz


def build_syndrome_map_to_pz(
    plaquettes: dict[int, Plaquette],
    plaquette_to_pz: dict[int, str],
) -> dict[int, str]:
    """Derive qubit-level ``map_to_pz`` from plaquette assignment.

    Each ancilla → its plaquette's PZ.
    Shared data qubits → first plaquette's PZ that claims them.
    """
    map_to_pz: dict[int, str] = {}
    for pid in sorted(plaquette_to_pz.keys()):
        plaq = plaquettes[pid]
        pz_name = plaquette_to_pz[pid]
        # Ancilla always goes to its plaquette's PZ
        map_to_pz[plaq.ancilla] = pz_name
        # Data qubits: first-claim wins
        for dq in plaq.data_qubits:
            if dq not in map_to_pz:
                map_to_pz[dq] = pz_name
    return map_to_pz


if __name__ == "__main__":
    print("Building surface code pattern...")
    distance = 3
    data_qubits, ancilla_qubits, plaquettes = build_surface_code_pattern(distance)
    print("Building surface code programs...")
    plaquette_programs, predecessors = build_surface_code_programs(plaquettes)
    print(plaquette_programs, '\n')
    print(predecessors)
    print("Building shared boundaries...")
    shared_boundaries = build_shared_boundaries(plaquettes)
    print(shared_boundaries)

    state = init_syndrome_scheduler_state(plaquette_programs, shared_boundaries, predecessors)

    for p in plaquette_programs:
        print(f"Plaquette {p}:")
        for i in range(4):
            print(is_gate_allowed(state, plaquette_programs[p][i]))




def plot_basic_grid(
    data_qubits: dict[int, tuple[float, float]],
    ancilla_qubits: dict[int, tuple[float, float]],
    plaquettes: dict[int, Plaquette],
    distance: int,
):
    fig, ax = plt.subplots(figsize=(distance * 1.5, distance * 1.5))

    def plot_coords(row: float, col: float) -> tuple[float, float]:
        return col, -row

    # 1. Draw plaquettes first
    for plaquette in plaquettes.values():
        facecolor = "lightblue" if plaquette.basis == "X" else "lightpink"

        if len(plaquette.data_qubits) == 4:
            dq_coords = [data_qubits[q] for q in plaquette.data_qubits]
            pts = [plot_coords(row, col) for row, col in dq_coords]

            cx = sum(x for x, _ in pts) / 4
            cy = sum(y for _, y in pts) / 4
            pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

            poly = Polygon(
                pts,
                closed=True,
                facecolor=facecolor,
                edgecolor="none",
                alpha=0.5,
                zorder=1,
            )
            ax.add_patch(poly)

            ax.text(
                cx,
                cy - 0.18,
                str(plaquette.plaquette_id),
                fontsize=8,
                ha="center",
                va="center",
                zorder=2,
            )

        elif len(plaquette.data_qubits) == 2:
            # semicircle with the two data qubits as diameter endpoints
            (r1, c1) = data_qubits[plaquette.data_qubits[0]]
            (r2, c2) = data_qubits[plaquette.data_qubits[1]]
            x1, y1 = plot_coords(r1, c1)
            x2, y2 = plot_coords(r2, c2)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            radius = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2

            if plaquette.row == -0.5:
                theta1, theta2 = 0, 180
                label_pos = (cx, cy + radius * 0.55)
            elif plaquette.row == distance - 0.5:
                theta1, theta2 = 180, 360
                label_pos = (cx, cy - radius * 0.55)
            elif plaquette.col == -0.5:
                theta1, theta2 = 90, 270
                label_pos = (cx - radius * 0.55, cy)
            elif plaquette.col == distance - 0.5:
                theta1, theta2 = -90, 90
                label_pos = (cx + radius * 0.55, cy)
            else:
                raise ValueError("Boundary plaquette not on recognized boundary.")

            wedge = Wedge(
                center=(cx, cy),
                r=radius,
                theta1=theta1,
                theta2=theta2,
                facecolor=facecolor,
                edgecolor="none",
                alpha=0.5,
                zorder=1,
            )
            ax.add_patch(wedge)

            ax.text(
                label_pos[0],
                label_pos[1],
                str(plaquette.plaquette_id),
                fontsize=8,
                ha="center",
                va="center",
                zorder=2,
            )

    # 2. Plot data qubits
    for q_id, (row, col) in data_qubits.items():
        x, y = plot_coords(row, col)
        ax.plot(x, y, "ko", markersize=8, zorder=3)
        ax.text(x + 0.08, y + 0.08, f"D{q_id}", fontsize=10, fontweight="bold", zorder=4)

    # 3. Plot ancilla qubits
    for a_id, (row, col) in ancilla_qubits.items():
        x, y = plot_coords(row, col)
        ax.plot(x, y, "wo", markeredgecolor="red", markersize=8, zorder=3)
        ax.text(x + 0.08, y + 0.08, f"A{a_id}", fontsize=10, color="red", zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.0, distance)
    ax.set_ylim(-distance, 1.0)

    plt.title(f"Surface Code Layout (d={distance})", pad=20)
    plt.tight_layout()
    plt.show()
    
distance = 3
data_q, ancilla_q, plaquettes = build_surface_code_pattern(distance)
#plot_basic_grid(data_q, ancilla_q, plaquettes, distance)