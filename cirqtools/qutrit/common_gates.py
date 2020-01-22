import cirq

from .. import qudit


__all__ = (
    'PlusOne',
    'MinusOne',
    'Chrest_3',
    'Z1',
    'Z2',
    'F01',
    'F02',
    'F12',
    'SWAP2',
    'SWAP3',
    'Ry01',
    'Ry02',
    'Ry12',
    'C0F01',
    'C1F01',
    'C2F01',
    'C0F02',
    'C1F02',
    'C2F02',
    'C0F12',
    'C1F12',
    'C2F12',
    'C0PlusOne',
    'C0MinusOne',
    'C1PlusOne',
    'C1MinusOne',
    'C2PlusOne',
    'C2MinusOne',
)


PlusOne = qudit.PlusGate(3)
MinusOne = qudit.PlusGate(3, -1)
Chrest_3 = qudit.Chrestenson(3)
Z1 = qudit.ZGate(3,increment=1)
Z2 = qudit.ZGate(3,increment=2)
F01 = qudit.FlipGate(3, 0, 1)  # or SingleQuditSubspaceGate(cirq.X, 3, (0, 1))
F02 = qudit.FlipGate(3, 0, 2)  # or SingleQuditSubspaceGate(cirq.X, 3, (0, 2))
F12 = qudit.FlipGate(3, 1, 2)  # or SingleQuditSubspaceGate(cirq.X, 3, (1, 2))
SWAP2 = qudit.SubspaceSwapGate(3, 3, 2, 2)
SWAP3 = qudit.SubspaceSwapGate(3, 3, 3, 3)
Ry01 = lambda theta: qudit.SingleQuditSubspaceGate(cirq.Ry(theta), 3, (0, 1))
Ry02 = lambda theta: qudit.SingleQuditSubspaceGate(cirq.Ry(theta), 3, (0, 2))
Ry12 = lambda theta: qudit.SingleQuditSubspaceGate(cirq.Ry(theta), 3, (1, 2))
C0F01 = cirq.ControlledGate(F01, control_values=(0,), control_qid_shape=(3,))
C1F01 = cirq.ControlledGate(F01, control_values=(1,), control_qid_shape=(3,))
C2F01 = cirq.ControlledGate(F01, control_values=(2,), control_qid_shape=(3,))
C0F02 = cirq.ControlledGate(F02, control_values=(0,), control_qid_shape=(3,))
C1F02 = cirq.ControlledGate(F02, control_values=(1,), control_qid_shape=(3,))
C2F02 = cirq.ControlledGate(F02, control_values=(2,), control_qid_shape=(3,))
C0F12 = cirq.ControlledGate(F12, control_values=(0,), control_qid_shape=(3,))
C1F12 = cirq.ControlledGate(F12, control_values=(1,), control_qid_shape=(3,))
C2F12 = cirq.ControlledGate(F12, control_values=(2,), control_qid_shape=(3,))
C0PlusOne = cirq.ControlledGate(PlusOne, control_values=(0,),
                                control_qid_shape=(3,))
C0MinusOne = cirq.ControlledGate(MinusOne, control_values=(0,),
                                 control_qid_shape=(3,))
C1PlusOne = cirq.ControlledGate(PlusOne, control_values=(1,),
                                control_qid_shape=(3,))
C1MinusOne = cirq.ControlledGate(MinusOne, control_values=(1,),
                                 control_qid_shape=(3,))
C2PlusOne = cirq.ControlledGate(PlusOne, control_values=(2,),
                                control_qid_shape=(3,))
C2MinusOne = cirq.ControlledGate(MinusOne, control_values=(2,),
                                 control_qid_shape=(3,))
