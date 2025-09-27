import numpy as np
from .gates import Rx, Ry, Rz, controlled_x, single_qubit_operator, I

class QNNLayer:
    """One parameterized layer: single-qubit rotations + entangling chain."""
    def __init__(self, n_qubits, random_state=None):
        self.n_qubits = n_qubits
        rng = np.random.default_rng(random_state)
        # parameters: for each qubit, three angles (Rx, Ry, Rz)
        self.params = rng.uniform(0, 2*np.pi, size=(n_qubits, 3))

    def num_params(self):
        return int(self.params.size)

    def forward(self, state):
        # Apply Rx, Ry, Rz on each qubit sequentially
        for q in range(self.n_qubits):
            rx = single_qubit_operator(self.n_qubits, Rx(self.params[q,0]), q)
            state = rx @ state
            ry = single_qubit_operator(self.n_qubits, Ry(self.params[q,1]), q)
            state = ry @ state
            rz = single_qubit_operator(self.n_qubits, Rz(self.params[q,2]), q)
            state = rz @ state
        # Entangle with a simple chain of CNOTs: (0->1), (1->2), ...
        for c in range(self.n_qubits - 1):
            cnot = controlled_x(self.n_qubits, control=c, target=c+1)
            state = cnot @ state
        return state
