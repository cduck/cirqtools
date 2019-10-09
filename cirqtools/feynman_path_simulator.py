
from collections import Counter, defaultdict
import itertools

import cirq
import numpy as np

from . import qudit


class PathLimitError(Exception): pass


class _State:
    def __init__(self, qid_shape, path_limit, tolerance, state=None):
        self.qid_shape = qid_shape
        self.path_limit = path_limit
        if state is None:
            state = Counter({(0,) * len(qid_shape): 1})
        self.state = state
        self.buffer_state = Counter()
        self.tolerance = tolerance

    @staticmethod
    def from_basis_state(int_state, qid_shape, *args, **kwargs):
        digits = cirq.big_endian_int_to_digits(int_state, base=qid_shape)
        state = Counter({tuple(digits): 1})
        return _State(qid_shape, *args, **kwargs, state=state)

    def apply_unitary(self, indices, unitary):
        sub_shape = [self.qid_shape[i] for i in indices]
        self.buffer_state.clear()
        for digits, amplitude in self.state.items():
            new_digits = list(digits)
            if np.isclose(amplitude, 0, rtol=1, atol=self.tolerance):
                continue
            target_val = cirq.big_endian_digits_to_int(
                (digits[i] for i in indices), base=sub_shape)
            result_wavefunction = unitary[:, target_val]
            for result_amp, new_targ_digits in zip(
                    result_wavefunction,
                    itertools.product(*(range(d) for d in sub_shape))):
                if np.isclose(result_amp, 0, rtol=1, atol=self.tolerance):
                    continue
                for i, dig in zip(indices, new_targ_digits):
                    new_digits[i] = dig
                self.buffer_state[tuple(new_digits)] += amplitude * result_amp
        self.state, self.buffer_state = self.buffer_state, self.state
        if len(self.state) > self.path_limit:
            raise PathLimitError

    def _sample_measurement(self, indices, repetitions):
        indices = list(indices)
        amps = np.array(list(self.state.values()), dtype=complex)
        states = np.array(list(self.state.keys()), dtype=np.uint8)
        probs = np.abs(amps) ** 2
        choice_i = np.random.choice(np.arange(len(probs)),
                                    size=repetitions,
                                    replace=True,
                                    p=probs)
        choice = states[choice_i,:][:,indices]
        return choice, indices, states, amps

    def sample_measurement(self, indices, repetitions=1):
        return _sample_measurement(indices, repetitions)[0]

    def apply_measurement(self, indices):
        choice, indices, states, amps = self._sample_measurement(indices, 1)
        choice = choice[0]
        mask = np.all(states[:,indices] == choice, axis=1)
        amps[mask] /= np.linalg.norm(amps[mask])
        self.state.clear()
        self.state.update(dict(zip(map(tuple, states[mask]), amps[mask])))
        return choice


class FeynmanPathSimulator(cirq.SimulatesSamples,
                           cirq.SimulatesIntermediateState):
    def __init__(self, path_limit=2**8, tolerance=1e-12):
        self.path_limit = path_limit
        self.tolerance = tolerance

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
        state = _State.from_basis_state(state_val,
                                        qid_shape,
                                        path_limit=self.path_limit,
                                        tolerance=self.tolerance)
        if len(circuit) == 0:
            yield FeynmanPathSimulatorStep(state, {}, qubit_map)

        def on_stuck(bad_op):
            return TypeError(
                "Can't simulate unknown operations that don't specify a"
                "unitary or a decomposition. {!r}".format(bad_op))

        def keep(potential_op):
            return (cirq.has_unitary(potential_op) or
                    cirq.has_mixture(potential_op) or
                    cirq.is_measurement(potential_op) or
                    cirq.op_gate_isinstance(potential_op, cirq.ResetChannel))

        def simulate_op(op, temp_state):
            indices = [qubit_map[q] for q in op.qubits]
            if cirq.op_gate_isinstance(op, cirq.ResetChannel):
                self._simulate_reset(op, cirq.ResetChannel)
            elif cirq.is_measurement(op):
                if perform_measurements:
                    self._simulate_measurement(
                        op, temp_state, indices, measurements)
            elif cirq.has_mixture(op):
                self._simulate_mixture(op, temp_state, indices)
            else:
                if cirq.num_qubits(op) <= 3:
                    self._simulate_unitary(op, temp_state, indices)
                else:
                    decomp_ops = cirq.decompose_once(op, default=None)
                    if decomp_ops is None:
                        self._simulate_unitary(op, temp_state, indices)
                    else:
                        for sub_op in cirq.flatten_op_tree(decomp_ops):
                            simulate_op(sub_op, temp_state)

        for moment in circuit:
            measurements = defaultdict(list)
            known_ops = cirq.decompose(moment, keep=keep,
                                       on_stuck_raise=on_stuck)
            for op in known_ops:
                simulate_op(op, state)
            yield FeynmanPathSimulatorStep(state, measurements, qubit_map)

    def _simulate_reset(op, state, indices):
        reset = cirq.op_gate_of_type(op, cirq.ResetChannel)
        if reset:
            meas = state.apply_measurement(indices)[0]
            if meas != 0:
                # Reset to 0
                unitary = cirq.unitary(qudit.FlipGate(meas+1, 0, meas))
                state.apply_unitary(indices, unitary)

    def _simulate_measurement(self, op, state, indices, measurements):
        meas = cirq.op_gate_of_type(op, cirq.MeasurementGate)
        if meas:
            invert_mask = meas.full_invert_mask()
            bits = state.apply_measurement(indices)
            corrected = [bit ^ (bit < 2 and mask)
                         for bit, mask in zip(bits, invert_mask)]
            key = cirq.measurement_key(meas)
            measurements[key].extend(corrected)

    def _simulate_mixture(self, op, state, indices):
        probs, unitaries = zip(*cirq.mixture(op))
        index = np.random.choice(range(len(unitaries)), p=probs)
        qid_shape = cirq.qid_shape(op)
        unitary = unitaries[index]
        state.apply_unitary(indices, unitary)

    def _simulate_unitary(self, op, state, indices):
        qid_shape = cirq.qid_shape(op)
        unitary = cirq.unitary(op)
        state.apply_unitary(indices, unitary)


class FeynmanPathSimulatorStep(cirq.StepResult):
    def __init__(self, state, measurements, qubit_map):
        super().__init__(measurements)
        self._state = state
        self.qubit_map = qubit_map

    def _simulator_state(self):
        return self._state.state

    def sample(self, qubits, repetitions=1):
        indices = [self.qubit_map[q] for q in qubits]
        return self._state.sample_measurement(indices, repetitions)
