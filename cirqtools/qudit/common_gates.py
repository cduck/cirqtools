import cirq
import numpy as np


__all__ = (
    'PlusGate',
    'SubspaceSwapGate',
    'SingleQuditSubspaceGate',
    'FlipGate',
    'Chrestenson',
    'ZGate',
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
            return '[{}<->{}]'.format(self.flip_a, self.flip_b)
        return cirq.CircuitDiagramInfo((
            '[{}<->{}]'.format(self.flip_a, self.flip_b),))

class Chrestenson(cirq.SingleQubitGate):
    def __init__(self,dimension):
        self.dimension = dimension

    def _qid_shape_(self):
        return (self.dimension,)

    def _unitary_(self):
        idx = np.arange(self.dimension, dtype=np.complex128)
        u = idx[:, np.newaxis] * idx[np.newaxis, :]
        u *= 2j*np.pi/self.dimension
        np.exp(u, out=u)
        u /= np.sqrt(self.dimension)
        return u

    def _circuit_diagram_info_(self,args):
        return cirq.CircuitDiagramInfo(('[C_r]',))
    

class ZGate(cirq.SingleQubitGate):
    def __init__(self,dimension,increment=1):
        self.dimension = dimension
        self.increment = increment

    def _qid_shape_(self):
        return (self.dimension,)

    def _unitary_(self):
        u = np.diag(np.exp(np.linspace(0, self.increment*2j*np.pi, self.dimension,endpoint=False, dtype=np.complex128)))
        
        return u

    def __pow__(self, exponent):
        if exponent % 1 == 0:
            return ZGate(self.dimension, self.increment * exponent)
        return NotImplemented

    def _circuit_diagram_info_(self,args):
        return cirq.CircuitDiagramInfo(('[Z {:+d}]'.format(self.increment),))

