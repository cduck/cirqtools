
from collections import defaultdict

import cirq
import numpy as np


class ClassicalSimulator(cirq.SimulatesSamples,
                         cirq.SimulatesIntermediateState):
    """An efficient simulator that can only simulate unitary gates that are
    permutations of the basis states."""

    def _run(self, circuit, param_resolver, repetitions):
        param_resolver = param_resolver or cirq.ParamResolver({})
        resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
        assert not cirq.is_parameterized(resolved_circuit)
        return self._run_sweep_repeat(resolved_circuit, repetitions)

    def _run_sweep_repeat(self, circuit, repetitions):
        measurements = defaultdict(list)
        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                circuit,
                qubit_order=cirq.QubitOrder.DEFAULT,
                initial_state=0)
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k].append(np.array(v, dtype=np.uint8))
        return {k: np.array(v, dtype=np.uint8) for k, v in measurements.items()}

    def _simulator_iterator(self, circuit, param_resolver, qubit_order,
                            initial_state):
        param_resolver = param_resolver or cirq.ParamResolver({})
        resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
        assert not cirq.is_parameterized(resolved_circuit)
        actual_initial_state = 0 if initial_state is None else initial_state
        return self._base_iterator(resolved_circuit, qubit_order,
                                   actual_initial_state,
                                   perform_measurements=True)

    def _base_iterator(self, circuit, qubit_order, initial_state,
                       perform_measurements=True):
        qubits = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(
            circuit.all_qubits())
        num_qubits = len(qubits)
        qid_shape = cirq.qid_shape(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        if isinstance(initial_state, int):
            state_val = initial_state
        else:
            state_val = cirq.big_endian_digits_to_int(initial_state,
                                                      base=qid_shape)
        state = np.array(list(cirq.big_endian_int_to_digits(state_val,
                                                            base=qid_shape)),
                         dtype=np.uint8)
        if len(circuit) == 0:
            yield ClassicalSimulatorStep(state, {}, qubit_map)

        def on_stuck(bad_op):
            return TypeError(
                "Can't simulate unknown operations that don't specify a "
                "unitary or a decomposition. {!r}".format(bad_op))

        def keep(op):
            return ((cirq.num_qubits(op) <= 32 and (cirq.has_unitary(op) or
                                                    cirq.has_mixture(op))) or
                    cirq.is_measurement(op) or
                    isinstance(op.gate, cirq.ResetChannel))

        def simulate_op(op, temp_state):
            indices = [qubit_map[q] for q in op.qubits]
            if isinstance(op.gate, cirq.ResetChannel):
                self._simulate_reset(op, temp_state, indices)
            elif cirq.is_measurement(op):
                if perform_measurements:
                    self._simulate_measurement(
                        op, temp_state, indices, measurements)
            else:
                decomp_ops = cirq.decompose_once(op, default=None)
                if decomp_ops is None:
                    self._simulate_from_matrix(op, temp_state, indices)
                else:
                    try:
                        temp2_state = temp_state.copy()
                        for sub_op in cirq.flatten_op_tree(decomp_ops):
                            simulate_op(sub_op, temp2_state)
                        temp_state[...] = temp2_state
                    except ValueError:
                        # Non-classical unitary in the decomposition
                        self._simulate_from_matrix(op, temp_state, indices)


        for moment in circuit:
            measurements = defaultdict(list)
            known_ops = cirq.decompose(moment, keep=keep,
                                       on_stuck_raise=on_stuck)
            for op in known_ops:
                simulate_op(op, state)
            yield ClassicalSimulatorStep(state.copy(), measurements, qubit_map)

    def _simulate_reset(self, op, state, indices):
        is_reset = isinstance(op.gate, cirq.ResetChannel)
        if is_reset:
            for i in indices:
                state[i] = 0

    def _simulate_measurement(self, op, state, indices, measurements):
        is_meas = isinstance(op.gate, cirq.MeasurementGate)
        if is_meas:
            invert_mask = op.gate.full_invert_mask()
            bits = [state[i] for i in indices]
            corrected = [bit ^ (bit < 2 and mask)
                         for bit, mask in zip(bits, invert_mask)]
            key = cirq.measurement_key(op.gate)
            measurements[key].extend(corrected)

    def _simulate_from_matrix(self, op, state, indices):
        if cirq.has_unitary(op):
            self._simulate_unitary(op, state, indices)
        else:
            self._simulate_mixture(op, state, indices)

    def _simulate_mixture(self, op, state, indices):
        probs, unitaries = zip(*cirq.mixture(op))
        index = np.random.choice(range(len(unitaries)), p=probs)
        qid_shape = cirq.qid_shape(op)
        unitary = unitaries[index]
        self._apply_unitary(op, unitary, qid_shape, state, indices)

    def _simulate_unitary(self, op, state, indices):
        qid_shape = cirq.qid_shape(op)
        unitary = cirq.unitary(op)
        self._apply_unitary(op, unitary, qid_shape, state, indices)

    def _apply_unitary(self, op, unitary, op_shape, state, indices):
        indices = list(indices)
        target_state = state[indices]
        target_val = cirq.big_endian_digits_to_int(target_state, base=op_shape)
        result_wavefunction = unitary[:, target_val]
        result_val = np.argmax(np.abs(result_wavefunction))
        if not (np.isclose(np.abs(result_wavefunction[result_val]), 1) and
                np.sum(1-np.isclose(result_wavefunction, 0)) == 1):
            # The output state vector does not represent a single basis state
            raise ValueError(
                "Can't simulate non-classical operations. "
                "The operation's unitary is not a permutation matrix: "
                "{!r}\n{!r}".format(op, unitary))
        result_state = cirq.big_endian_int_to_digits(result_val, base=op_shape)
        state[indices] = result_state


class ClassicalSimulatorStep(cirq.StepResult):
    def __init__(self, state, measurements, qubit_map):
        super().__init__(measurements)
        self.state = state
        self.qubit_map = qubit_map

    def _simulator_state(self):
        return self.state

    def sample(self, qubits, repetitions=1):
        values = [self.state[self.qubit_map[q]] for q in qubits]
        results = np.empty((repetitions, len(qubits)), dtype=np.uint8)
        results[:] = values  # Every repetition is the same
        return results
