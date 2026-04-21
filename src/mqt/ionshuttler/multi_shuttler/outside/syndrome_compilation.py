"""Syndrome-scheduler compilation helpers.

Parallel to the DAG-based functions in ``compilation.py``, but driven by
:class:`SyndromeSchedulerState` instead of a Qiskit ``DAGDependency``.

Every public function produces the **same output types** that the downstream
shuttling engine (priority queues, move lists, cycles/paths) already consumes,
so no changes are needed below this layer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .graph_utils import update_distance_map
from .cycles import get_state_idxs
from .scheduling import assign_gate_to_pz
from .syndrome_scheduler import (
    SyndromeGate,
    SyndromeSchedulerState,
    execute_gate,
    get_front_layer,
)

# Default load penalty weight for load-aware PZ assignment.
# Roughly one 2Q gate execution time — ensures distance dominates for far PZs.
_LOAD_PENALTY = 3

if TYPE_CHECKING:
    from .graph import Graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate_qubits(gate: SyndromeGate) -> tuple[int, int]:
    """Return the qubit-index tuple for a SyndromeGate (ancilla, data)."""
    return gate.gate_tuple


def _map_syndrome_front_layer_to_pzs(
    graph: Graph,
    front_layer: list[SyndromeGate],
    *,
    load_aware: bool = False,
    dist_map: dict[int, dict[str, int]] | None = None,
) -> dict[str, list[SyndromeGate]]:
    """Map each front-layer syndrome gate to a processing zone.

    Equivalent of ``compilation.map_front_gates_to_pzs`` but for
    :class:`SyndromeGate` objects.

    When *load_aware* is True, evaluates ALL PZs for each gate with a cost
    function that penalises already-loaded PZs, promoting parallelism.
    A pre-computed *dist_map* (from ``update_distance_map``) can be passed
    to avoid redundant work.

    Returns a dict ``{pz_name: [gate, ...]}``.
    """
    gates_by_pz: dict[str, list[SyndromeGate]] = {pz.name: [] for pz in graph.pzs}

    if not load_aware:
        for gate in front_layer:
            pz_name = assign_gate_to_pz(graph, _gate_qubits(gate))
            gates_by_pz[pz_name].append(gate)
        return gates_by_pz

    # --- load-aware path: evaluate all PZs per gate ---
    if dist_map is None:
        ion_state = get_state_idxs(graph)
        dist_map = update_distance_map(graph, ion_state)

    pz_names = [pz.name for pz in graph.pzs]

    # Compute minimum achievable cost per gate (ignoring load) for sorting
    def _min_cost(gate: SyndromeGate) -> int:
        a, d = gate.ancilla, gate.data
        return min(dist_map[a][pn] + dist_map[d][pn] for pn in pz_names)

    sorted_gates = sorted(front_layer, key=_min_cost)

    # Track load as we assign greedily
    pz_load: dict[str, int] = {pn: 0 for pn in pz_names}

    for gate in sorted_gates:
        a, d = gate.ancilla, gate.data
        best_pz = pz_names[0]
        best_cost = math.inf
        for pn in pz_names:
            cost = dist_map[a][pn] + dist_map[d][pn] + _LOAD_PENALTY * pz_load[pn]
            if cost < best_cost:
                best_cost = cost
                best_pz = pn
        gates_by_pz[best_pz].append(gate)
        pz_load[best_pz] += 1

    return gates_by_pz


def _find_best_syndrome_gate(
    graph: Graph,
    candidates: list[SyndromeGate],
    dist_map: dict[int, dict[str, int]],
    gate_to_pz: dict[SyndromeGate, str],
) -> SyndromeGate:
    """Pick the candidate whose ions are closest to their target PZ.

    Equivalent of ``compilation.find_best_gate`` but for
    :class:`SyndromeGate` objects.
    """
    best_gate: SyndromeGate | None = None
    min_cost = math.inf
    for gate in candidates:
        pz_name = gate_to_pz[gate]
        qubits = _gate_qubits(gate)
        cost = max(dist_map[q][pz_name] for q in qubits)
        if len(qubits) == 2 and cost == 0:
            return gate
        if cost < min_cost:
            min_cost = cost
            best_gate = gate
    assert best_gate is not None
    return best_gate


# ---------------------------------------------------------------------------
# Public API — drop-in replacements called from shuttle.py
# ---------------------------------------------------------------------------

def create_initial_syndrome_sequence(
    state: SyndromeSchedulerState,
) -> list[tuple[int, ...]]:
    """Build the initial ``graph.sequence`` from plaquette programs.

    Simply flattens all plaquette programs into a list of ``(ancilla, data)``
    tuples in plaquette order.  This is the syndrome equivalent of
    ``compilation.create_initial_sequence`` (which reads a QASM file).
    """
    seq: list[tuple[int, ...]] = []
    for _plaq_id, program in state.plaquette_programs.items():
        for gate in program:
            seq.append(_gate_qubits(gate))
    return seq


def create_updated_syndrome_sequence(
    graph: Graph,
    state: SyndromeSchedulerState,
) -> list[tuple[int, ...]]:
    """Reorder the sequence using the syndrome front layer + distances.

    Equivalent of ``compilation.create_updated_sequence_destructive`` — works
    on a *copy* of the state so the original is preserved.
    """
    load_aware = getattr(graph, "syndrome_load_aware", False)
    working_state = state.copy()
    seq: list[tuple[int, ...]] = []

    ion_state = get_state_idxs(graph)
    dist_map = update_distance_map(graph, ion_state)

    while True:
        front = get_front_layer(working_state)
        if not front:
            break

        gates_by_pz = _map_syndrome_front_layer_to_pzs(
            graph, front, load_aware=load_aware, dist_map=dist_map,
        )
        gate_to_pz = {g: pz for pz, gs in gates_by_pz.items() for g in gs}

        for pz_name, pz_gates in gates_by_pz.items():
            if pz_gates:
                best = _find_best_syndrome_gate(graph, pz_gates, dist_map, gate_to_pz)
                execute_gate(working_state, best)
                seq.append(_gate_qubits(best))

    return seq


def get_syndrome_front_gates_and_update_sequence(
    graph: Graph,
    state: SyndromeSchedulerState,
    max_rounds: int = 5,
) -> dict[str, SyndromeGate]:
    """Get the first executable gate per PZ and reorder ``graph.sequence``.

    This is the syndrome equivalent of
    ``compilation.get_all_first_gates_and_update_sequence_non_destructive``.

    It does **not** mutate ``state`` — it only reads the front layer.
    Actual gate execution (state mutation) happens later in
    :func:`execute_syndrome_gates`.

    Returns ``{pz_name: SyndromeGate}`` for the first round's gates only.
    """
    ordered_sequence: list[tuple[int, ...]] = []
    virtually_executed: set[tuple[int, int]] = set()
    first_gates_by_pz: dict[str, SyndromeGate] = {}

    ion_state = get_state_idxs(graph)
    dist_map = update_distance_map(graph, ion_state)

    # We work on a temporary copy so the real state stays unchanged.
    working_state = state.copy()

    for round_idx in range(max_rounds):
        front = get_front_layer(working_state)
        if not front:
            break

        load_aware = getattr(graph, "syndrome_load_aware", False)
        gates_by_pz = _map_syndrome_front_layer_to_pzs(
            graph, front, load_aware=load_aware, dist_map=dist_map,
        )
        gate_to_pz = {g: pz for pz, gs in gates_by_pz.items() for g in gs}

        round_gates: list[SyndromeGate] = []
        for pz_name, pz_gates in gates_by_pz.items():
            if not pz_gates:
                continue
            best = _find_best_syndrome_gate(graph, pz_gates, dist_map, gate_to_pz)

            if round_idx == 0 and pz_name not in first_gates_by_pz:
                first_gates_by_pz[pz_name] = best

            round_gates.append(best)
            ordered_sequence.append(_gate_qubits(best))
            virtually_executed.add(_gate_qubits(best))

            # Advance the working copy so the next round sees new front layer
            execute_gate(working_state, best)

        # Remove found gates from existing sequence
        for gate in round_gates:
            gt = _gate_qubits(gate)
            if gt in graph.sequence:
                graph.sequence.remove(gt)

    graph.sequence = ordered_sequence + graph.sequence
    return first_gates_by_pz


def execute_syndrome_gates(
    state: SyndromeSchedulerState,
    graph: Graph,
    completed_gates: dict[str, SyndromeGate],
) -> None:
    """Mark gates as executed in the syndrome state and remove from sequence.

    Equivalent of ``compilation.remove_processed_gates`` but mutates the
    :class:`SyndromeSchedulerState` instead of a Qiskit DAG.
    """
    for _pz_name, gate in completed_gates.items():
        gt = _gate_qubits(gate)
        execute_gate(state, gate)
        if gt in graph.sequence:
            graph.sequence.remove(gt)
