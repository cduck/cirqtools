
from collections import defaultdict

import cirq
import numpy as np


class ClassicalSimulator(cirq.SimulatesSamples,
                         cirq.SimulatesIntermediateState):
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
                "Can't simulate unknown operations that don't specify a"
                "unitary or a decomposition. {!r}".format(bad_op))

        def keep(potential_op):
            return (cirq.has_unitary(potential_op) or
                    cirq.has_mixture(potential_op) or
                    cirq.is_measurement(potential_op) or
                    cirq.op_gate_isinstance(potential_op, cirq.ResetChannel))

        for moment in circuit:
            measurements = defaultdict(list)
            known_ops = cirq.decompose(moment, keep=keep,
                                       on_stuck_raise=on_stuck)
            for op in known_ops:
                indices = [qubit_map[q] for q in op.qubits]
                if cirq.op_gate_isinstance(op, cirq.ResetChannel):
                    self._simulate_reset(op, state, indices)
                elif cirq.has_unitary(op):
                    self._simulate_unitary(op, state, indices)
                elif cirq.is_measurement(op):
                    if perform_measurements:
                        self._simulate_measurement(op, state, indices,
                                                   measurements)
                elif cirq.has_mixture(op):
                    self._simulate_mixture(op, state, indices)
            yield ClassicalSimulatorStep(list(state), measurements, qubit_map)

    def _simulate_reset(op, state, indices):
        reset = cirq.op_gate_of_type(op, cirq.ResetChannel)
        if reset:
            for i in indices:
                state[i] = 0

    def _simulate_measurement(self, op, state, indices, measurements):
        meas = cirq.op_gate_of_type(op, cirq.MeasurementGate)
        if meas:
            invert_mask = meas.full_invert_mask()
            bits = [state[i] for i in indices]
            corrected = [bit ^ (bit < 2 and mask)
                         for bit, mask in zip(bits, invert_mask)]
            key = cirq.measurement_key(meas)
            measurements[key].extend(corrected)

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
                "The operation's unitary is not a permuation matrix: {!r}\n{!r}"
                "".format(op, unitary))
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
