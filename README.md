# Cirq Tools

cirqtools is a Python package where I keep common code between my various quantum projects.

## Qudit

`cirqtools.qudit` contains various gate definitions useful when dealing with qudits.

## Qutrit

`cirqtools.qutrit` contains qutrit-specific instances of types in `cirqtools.qudit`.

## Simulators

`cirqtools.ClassicalSimulator` will efficiently simulate any classical reversible circuit (any circuit containing gates whose unitaries are permutation matrices).

`cirqtools.FeynmanPathSimulator` is an inefficient attempt attempt at a Feynman Path-based simulator.
