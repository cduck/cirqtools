import cirq
import numpy as np

from . import common_gates


__all__ = (
    'TwoControlPlusOneGate',
    'TwoControlMinusOneGate',
    'C1C1PlusOne',
    'C1C1MinusOne',
)


class TwoControlPlusOneGate(cirq.Gate):
    """Gate sequence from https://arxiv.org/abs/1105.5485."""
    def __init__(self, c0_val, c1_val):
        self.c0_val = c0_val
        self.c1_val = c1_val

    def _qid_shape_(self):
        return (3, 3, 3)

    def _decompose_(self, qids):
        q0, q1, q2 = qids
        yield common_gates.Ry12(-np.pi/4)(q2)
        yield common_gates.F12(q2).controlled_by(q1,
                                                 control_values=[self.c1_val])
        yield common_gates.Ry12(-np.pi/4)(q2)
        yield common_gates.F12(q2).controlled_by(q0,
                                                 control_values=[self.c0_val])
        yield common_gates.Ry12(np.pi/4)(q2)
        yield common_gates.F12(q2).controlled_by(q1,
                                                 control_values=[self.c1_val])
        yield common_gates.Ry12(np.pi/4)(q2)
        yield common_gates.Ry01(np.pi/4)(q2)
        yield common_gates.F01(q2).controlled_by(q1,
                                                 control_values=[self.c1_val])
        yield common_gates.Ry01(np.pi/4)(q2)
        yield common_gates.F01(q2).controlled_by(q0,
                                                 control_values=[self.c0_val])
        yield common_gates.Ry01(-np.pi/4)(q2)
        yield common_gates.F01(q2).controlled_by(q1,
                                                 control_values=[self.c1_val])
        yield common_gates.Ry01(-np.pi/4)(q2)

    def __pow__(self, exponent):
        if exponent == 1:
            return self
        if exponent == -1:
            return TwoControlMinusOneGate(self.c0_val, self.c1_val)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        q0, q1 = cirq.LineQid.for_qid_shape((3, 3))
        return cirq.circuit_diagram_info(common_gates.PlusOne.controlled(
            control_values=(self.c0_val, self.c1_val),
            control_qid_shape=(3, 3)))


class TwoControlMinusOneGate(cirq.Gate):
    def __init__(self, c0_val, c1_val):
        self.c0_val = c0_val
        self.c1_val = c1_val

    def _qid_shape_(self):
        return (3, 3, 3)

    def _decompose_(self, qids):
        return cirq.inverse(cirq.decompose_once_with_qubits(self ** -1, qids))

    def __pow__(self, exponent):
        if exponent == 1:
            return self
        if exponent == -1:
            return TwoControlPlusOneGate(self.c0_val, self.c1_val)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        q0, q1 = cirq.LineQid.for_qid_shape((3, 3))
        return cirq.circuit_diagram_info(common_gates.MinusOne.controlled(
            control_values=(self.c0_val, self.c1_val),
            control_qid_shape=(3, 3)))


C1C1PlusOne = TwoControlPlusOneGate(1, 1)
C1C1MinusOne = TwoControlMinusOneGate(1, 1)
