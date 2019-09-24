import cirq
import numpy as np


__all__ = (
    'PlusGate',
    'SubspaceSwapGate',
    'SingleQuditSubspaceGate',
    'FlipGate',

    'PlusOne',
    'MinusOne',
    'F01',
    'F02',
    'F12',
    'SWAP2',
    'SWAP3',
    'Ry01',
    'Ry12',
    'C1F01',
    'C2F01',
    'C1F12',
    'C2F12',
    'C1PlusOne',
    'C1MinusOne',
)


class PlusGate(cirq.SingleQubitGate):
    def __init__(self, dimension, increment=1):
        self.dimension = dimension
        self.increment = increment % dimension

    def _qid_shape_(self):
        return (self.dimension,)

    def num_qubits(self):
        return 1

    def _unitary_(self):
        inc = (self.increment - 1) % self.dimension + 1
        u = np.empty((self.dimension, self.dimension))
        u[inc:] = np.eye(self.dimension)[:-inc]
        u[:inc] = np.eye(self.dimension)[-inc:]
        return u

    def __pow__(self, exponent):
        if exponent % 1 == 0:
            return PlusGate(self.dimension, self.increment * exponent)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo((
            '[{:+d}%{}]'.format(self.increment, self.dimension),))


class SubspaceSwapGate(cirq.Gate):
    def __init__(self, dim0, dim1, swap_dim0=2, swap_dim1=2):
        assert 0 < swap_dim0 <= dim0
        assert 0 < swap_dim1 <= dim1
        assert swap_dim0 <= dim1
        assert swap_dim1 <= dim0
        self.dim0 = dim0
        self.dim1 = dim1
        self.swap_dim0 = swap_dim0
        self.swap_dim1 = swap_dim1

    def _qid_shape_(self):
        return (self.dim0, self.dim1)

    def _unitary_(self):
        u = np.zeros((self.dim0 * self.dim1,) * 2, dtype=np.complex128)
        max_swap_dim = max(self.swap_dim0, self.swap_dim1)
        for i0 in range(self.dim0):
            for i1 in range(self.dim1):
                # TODO: swap_dim0 != swap_dim1
                if i0 < max_swap_dim and i1 < max_swap_dim:
                #if i0 < self.swap_dim0 and i1 < self.swap_dim1:
                    u[i1*self.dim0 + i0, i0*self.dim0 + i1] = 1
                else:
                    u[i1*self.dim0 + i0, i1*self.dim0 + i0] = 1
        return u

    def __pow__(self, exponent):
        if exponent % 2 == 0:
            return self
        elif exponent % 2 == 1:
            return SubspaceSwapGate(self.dim0, self.dim1, self.swap_dim1,
                                    self.swap_dim0)
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo((
            'SWAP(d={})'.format(self.swap_dim0),
            'SWAP(d={})'.format(self.swap_dim1),
        ))


class SingleQuditSubspaceGate(cirq.SingleQubitGate):
    def __init__(self, sub_gate, dimension, sub_levels=None):
        self.sub_gate = sub_gate
        self.dimension = dimension
        if sub_levels is None:
            sub_levels = range(cirq.qid_shape(sub_gate)[0])
        assert len(set(sub_levels)) == len(sub_levels), 'Duplicate levels given'
        self.sub_levels = list(sub_levels)

    def _qid_shape_(self):
        return (self.dimension,)

    def _unitary_(self):
        sub_shape = cirq.qid_shape(self.sub_gate)
        assert sub_shape[0] == len(self.sub_levels), (
            'Wrong number of levels given')
        sub_u = cirq.unitary(self.sub_gate).reshape(sub_shape * 2)
        u = cirq.eye_tensor((self.dimension,), dtype=sub_u.dtype)
        temp = u[self.sub_levels, :, ...].copy()
        temp[:, self.sub_levels, ...] = sub_u
        u[self.sub_levels, :, ...] = temp
        return u

    def __pow__(self, exponent):
        return SingleQuditSubspaceGate(
            self.sub_gate ** exponent, self.dimension, self.sub_levels)

    def _circuit_diagram_info_(self, args):
        sub_info = cirq.circuit_diagram_info(self.sub_gate)
        return cirq.CircuitDiagramInfo(
            ('{}:{}'.format(sub_info.wire_symbols[0],
                            ''.join(map(str, self.sub_levels))),),
            exponent=sub_info.exponent,
        )

class FlipGate(SingleQuditSubspaceGate):
    def __init__(self, dimension, flip_a=0, flip_b=1):
        assert 0 <= flip_a < dimension
        assert 0 <= flip_b < dimension
        assert flip_a != flip_b
        if flip_a > flip_b:
            flip_a, flip_b = flip_b, flip_a
        super().__init__(cirq.X, dimension, (flip_a, flip_b))
        self.flip_a = flip_a
        self.flip_b = flip_b

    def __pow__(self, exponent):
        if exponent % 1 == 0:
            return self
        return NotImplemented

    def _circuit_diagram_info_(self, args):
        if args.use_unicode_characters:
            return '[{}⇄{}]'.format(self.flip_a, self.flip_b)
        return cirq.CircuitDiagramInfo((
            '[{}<->{}]'.format(self.flip_a, self.flip_b),))


# Qutrit gates
PlusOne = PlusGate(3)
MinusOne = PlusGate(3, -1)
F01 = FlipGate(3, 0, 1)  # SingleQuditSubspaceGate(cirq.X, 3, (0, 1))
F02 = FlipGate(3, 0, 2)  # SingleQuditSubspaceGate(cirq.X, 3, (0, 2))
F12 = FlipGate(3, 1, 2)  # SingleQuditSubspaceGate(cirq.X, 3, (1, 2))
SWAP2 = SubspaceSwapGate(3, 3, 2, 2)
SWAP3 = SubspaceSwapGate(3, 3, 3, 3)
Ry01 = lambda theta: SingleQuditSubspaceGate(cirq.Ry(theta), 3, (0, 1))
Ry12 = lambda theta: SingleQuditSubspaceGate(cirq.Ry(theta), 3, (1, 2))
C1F01 = cirq.ControlledGate(F01, control_values=(1,), control_qid_shape=(3,))
C2F01 = cirq.ControlledGate(F01, control_values=(2,), control_qid_shape=(3,))
C1F12 = cirq.ControlledGate(F12, control_values=(1,), control_qid_shape=(3,))
C2F12 = cirq.ControlledGate(F12, control_values=(2,), control_qid_shape=(3,))
C1PlusOne = cirq.ControlledGate(PlusOne, control_values=(1,),
                                control_qid_shape=(3,))
C1MinusOne = cirq.ControlledGate(MinusOne, control_values=(1,),
                                 control_qid_shape=(3,))
