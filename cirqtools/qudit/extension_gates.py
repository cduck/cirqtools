import cirq


class SimpleExtensionGate(cirq.Gate):
    def __init__(self, gate, new_qid_shape):
        sub_qid_shape = cirq.qid_shape(gate)
        if len(new_qid_shape) != len(sub_qid_shape):
            raise ValueError(
                f'Cannot extend gate <{gate}> to qid shape with a different '
                f'length: <{new_qid_shape}>.')
        if any(d1 > d2 for d1, d2 in zip(sub_qid_shape, new_qid_shape)):
            raise ValueError(
                f'Cannot extend gate <{gate}> to smaller qid shape: '
                f'<{new_qid_shape}>.')
        self.gate = gate
        self.new_qid_shape = new_qid_shape

    def _qid_shape_(self):
        return self.new_qid_shape

    def _apply_unitary_(self, args):
        return cirq.apply_unitary(self.gate, args, default=NotImplemented)

    # TODO: Implement more efficient _unitary_ and other methods

    def _circuit_diagram_info_(self, args):
        info = cirq.circuit_diagram_info(
            self.gate, args, default=NotImplemented)
        sub_qid_shape = cirq.qid_shape(self.gate)
        sep = ' ' if any(d > 10 for d in sub_qid_shape) else ''
        syms = tuple(
            f'{sym}:{sep.join(map(str, range(d)))}'
            for sym, d in zip(info.wire_symbols, cirq.qid_shape(self.gate))
        )

    def __repr__(self):
        return f'SimpleExtensionGate({self.gate!r}, {self.new_qid_shape!r})'


def simple_extention(gate, *qids):
    return SimpleExtensionGate(gate, cirq.qid_shape(qids)).on(*qids)
