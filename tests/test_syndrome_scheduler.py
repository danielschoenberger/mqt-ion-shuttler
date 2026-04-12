from __future__ import annotations

import pytest

from mqt.ionshuttler.multi_shuttler.outside.syndrome_scheduler import (
    BoundaryOrientation,
    SharedBoundary,
    SyndromeGate,
    SyndromeSchedulerState,
    execute_gate,
    get_front_layer,
    is_gate_allowed,
)


def _make_two_plaquette_state(
    boundary_orientation: BoundaryOrientation = BoundaryOrientation.UNDECIDED,
) -> tuple[SyndromeSchedulerState, list[SyndromeGate], list[SyndromeGate]]:
    """
    Two plaquettes sharing data qubits 3 and 4.

    Plaquette A (id=0, ancilla=100): D1 -> D4 -> D0 -> D3  (steps 0..3)
    Plaquette B (id=1, ancilla=101): D3 -> D2 -> D5 -> D4  (steps 0..3)

    Shared boundary "AB": plaquette A vs B on data qubits {3, 4}.
      - A touches D4 at step 1 and D3 at step 3
      - B touches D3 at step 0 and D4 at step 3
    """
    gates_A = [
        SyndromeGate(ancilla=100, data=1, plaquette=0, step_idx=0),
        SyndromeGate(ancilla=100, data=4, plaquette=0, step_idx=1),
        SyndromeGate(ancilla=100, data=0, plaquette=0, step_idx=2),
        SyndromeGate(ancilla=100, data=3, plaquette=0, step_idx=3),
    ]
    gates_B = [
        SyndromeGate(ancilla=101, data=3, plaquette=1, step_idx=0),
        SyndromeGate(ancilla=101, data=2, plaquette=1, step_idx=1),
        SyndromeGate(ancilla=101, data=5, plaquette=1, step_idx=2),
        SyndromeGate(ancilla=101, data=4, plaquette=1, step_idx=3),
    ]

    # Each gate maps to the set of gate_tuples that must appear in executed_gates first.
    predecessors = {
        0: {
            gates_A[0]: set(),
            gates_A[1]: {gates_A[0].gate_tuple},
            gates_A[2]: {gates_A[0].gate_tuple, gates_A[1].gate_tuple},
            gates_A[3]: {gates_A[0].gate_tuple, gates_A[1].gate_tuple, gates_A[2].gate_tuple},
        },
        1: {
            gates_B[0]: set(),
            gates_B[1]: {gates_B[0].gate_tuple},
            gates_B[2]: {gates_B[0].gate_tuple, gates_B[1].gate_tuple},
            gates_B[3]: {gates_B[0].gate_tuple, gates_B[1].gate_tuple, gates_B[2].gate_tuple},
        },
    }

    boundary = SharedBoundary(
        boundary_str="AB",
        plaquette_a=0,
        plaquette_b=1,
        shared_data_qubits=(3, 4),
    )

    state = SyndromeSchedulerState(
        plaquette_programs={0: gates_A, 1: gates_B},
        plaquette_ptr={0: 0, 1: 0},
        shared_boundaries={"AB": boundary},
        qubit_to_boundary={
            (0, 3): "AB",
            (0, 4): "AB",
            (1, 3): "AB",
            (1, 4): "AB",
        },
        boundary_orientation={"AB": boundary_orientation},
        predecessors=predecessors,
    )
    return state, gates_A, gates_B



# Tests for is_gate_allowed

def test_initial_only_first_step_allowed():
    """Only step-0 gates are ready in a fresh state (no executed gates)."""
    state, gates_A, gates_B = _make_two_plaquette_state()

    assert is_gate_allowed(state, gates_A[0]) is True
    assert is_gate_allowed(state, gates_B[0]) is True

    assert is_gate_allowed(state, gates_A[1]) is False
    assert is_gate_allowed(state, gates_A[2]) is False
    assert is_gate_allowed(state, gates_A[3]) is False
    assert is_gate_allowed(state, gates_B[1]) is False
    assert is_gate_allowed(state, gates_B[2]) is False
    assert is_gate_allowed(state, gates_B[3]) is False


def test_executing_predecessor_unlocks_next_gate():
    """After marking A[0] as executed, A[1] becomes allowed; A[2] stays blocked."""
    state, gates_A, gates_B = _make_two_plaquette_state()

    state.executed_gates.add(gates_A[0].gate_tuple)

    assert is_gate_allowed(state, gates_A[1]) is True
    assert is_gate_allowed(state, gates_A[2]) is False   # still needs A[1]
    assert is_gate_allowed(state, gates_B[0]) is True    # independent of A


def test_sequential_execution_unlocks_all_steps():
    """Executing A's gates one by one unlocks each subsequent step."""
    state, gates_A, _ = _make_two_plaquette_state()

    for i in range(3):
        assert is_gate_allowed(state, gates_A[i + 1]) is False  # blocked before
        state.executed_gates.add(gates_A[i].gate_tuple)
        assert is_gate_allowed(state, gates_A[i + 1]) is True   # unblocked after


def test_boundary_undecided_both_sides_allowed():
    """With UNDECIDED boundary both plaquettes may proceed to shared-qubit gates independently."""
    state, gates_A, gates_B = _make_two_plaquette_state(BoundaryOrientation.UNDECIDED)

    # B[0] touches shared D3 with no predecessors — allowed immediately
    assert is_gate_allowed(state, gates_B[0]) is True

    # A[1] touches shared D4 but requires A[0] first
    assert is_gate_allowed(state, gates_A[1]) is False
    state.executed_gates.add(gates_A[0].gate_tuple)
    assert is_gate_allowed(state, gates_A[1]) is True


def test_boundary_a_first_blocks_b_on_shared_qubits():
    """A_FIRST: B must wait for A to execute its gate on each shared qubit before B can proceed."""
    state, gates_A, gates_B = _make_two_plaquette_state(BoundaryOrientation.A_FIRST)

    # B[0] wants D3; A's D3 gate is A[3] and hasn't run yet
    assert is_gate_allowed(state, gates_B[0]) is False

    # Execute all of A's program, including A[3] (D3)
    for g in gates_A:
        state.executed_gates.add(g.gate_tuple)

    assert is_gate_allowed(state, gates_B[0]) is True


def test_boundary_b_first_blocks_a_on_shared_qubits():
    """B_FIRST: A must wait for B to execute its gate on each shared qubit before A can proceed."""
    state, gates_A, gates_B = _make_two_plaquette_state(BoundaryOrientation.B_FIRST)

    # Satisfy A[1]'s predecessor so the only remaining obstacle is the boundary
    state.executed_gates.add(gates_A[0].gate_tuple)

    # A[1] wants D4; B's D4 gate is B[3] and hasn't run yet
    assert is_gate_allowed(state, gates_A[1]) is False

    # Execute all of B's program, including B[3] (D4)
    for g in gates_B:
        state.executed_gates.add(g.gate_tuple)

    assert is_gate_allowed(state, gates_A[1]) is True


def test_boundary_a_first_does_not_block_a_itself():
    """A_FIRST never blocks A's own gates — only A's predecessors matter."""
    state, gates_A, _ = _make_two_plaquette_state(BoundaryOrientation.A_FIRST)

    # A[1] (D4, shared): only needs A[0] as predecessor
    state.executed_gates.add(gates_A[0].gate_tuple)
    assert is_gate_allowed(state, gates_A[1]) is True

    # A[3] (D3, shared): needs A[0..2]
    for g in gates_A[:3]:
        state.executed_gates.add(g.gate_tuple)
    assert is_gate_allowed(state, gates_A[3]) is True



# Tests for get_front_layer

def test_front_layer_initial_contains_both_step0_gates():
    """Fresh state: one ready gate per plaquette, both at step 0."""
    state, gates_A, gates_B = _make_two_plaquette_state()

    front = get_front_layer(state)

    assert set(front) == {gates_A[0], gates_B[0]}


def test_front_layer_advances_after_execution():
    """After executing A[0], the front layer replaces it with A[1]."""
    state, gates_A, gates_B = _make_two_plaquette_state()

    execute_gate(state, gates_A[0])
    front = get_front_layer(state)

    assert gates_A[0] not in front
    assert gates_A[1] in front
    assert gates_B[0] in front  # B is unaffected


def test_front_layer_respects_boundary_constraint():
    """A_FIRST: B[0] is blocked by the boundary and must not appear in the front layer."""
    state, gates_A, gates_B = _make_two_plaquette_state(BoundaryOrientation.A_FIRST)

    front = get_front_layer(state)

    assert gates_A[0] in front
    assert gates_B[0] not in front  # blocked: A must execute D3 first



# Tests for execute_gate

def test_execute_gate_updates_ptr_and_executed_set():
    """execute_gate advances the plaquette pointer and records the gate tuple."""
    state, gates_A, _ = _make_two_plaquette_state()

    execute_gate(state, gates_A[0])

    assert state.plaquette_ptr[0] == 1
    assert gates_A[0].gate_tuple in state.executed_gates


def test_execute_gate_sets_boundary_orientation():
    """Executing the first gate on a shared qubit locks the boundary orientation."""
    state, _, gates_B = _make_two_plaquette_state()

    assert state.boundary_orientation["AB"] == BoundaryOrientation.UNDECIDED
    execute_gate(state, gates_B[0])  # B goes first on D3
    assert state.boundary_orientation["AB"] == BoundaryOrientation.B_FIRST


def test_execute_gate_raises_for_out_of_order_gate():
    """Executing a gate that is not next in the program raises ValueError."""
    state, gates_A, _ = _make_two_plaquette_state()

    with pytest.raises(ValueError):
        execute_gate(state, gates_A[1])  # A[0] must go first
