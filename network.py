import numpy as np
from .state import zero_state, normalize
from .layers import QNNLayer
from .gates import single_qubit_operator, Z, I

class SimpleQNN:
    """A simple stacked QNN using QNNLayer objects."""
    def __init__(self, n_qubits=2, n_layers=1, random_state=None):
        assert n_qubits >= 1, "n_qubits must be >= 1"
        self.n_qubits = n_qubits
        self.layers = [QNNLayer(n_qubits, random_state=random_state) for _ in range(n_layers)]

    def encode_angles(self, x):
        """Encode classical input x into angles per qubit.
        If x is scalar: encode on qubit 0; if iterable, map to first len(x) qubits."""
        angles = np.zeros(self.n_qubits)
        try:
            # iterable
            for i, val in enumerate(x):
                if i >= self.n_qubits:
                    break
                angles[i] = float(val)
        except TypeError:
            # scalar
            angles[0] = float(x)
        return angles

    def forward(self, x):
        state = zero_state(self.n_qubits)
        angles = self.encode_angles(x)
        # Apply Ry(angle) encoding on each qubit
        for q, a in enumerate(angles):
            U = single_qubit_operator(self.n_qubits, __import__('qnnlib').gates.Ry(a), q)
            state = U @ state
        # pass through layers
        for layer in self.layers:
            state = layer.forward(state)
        return normalize(state)

    def expectation(self, state, qubit=0):
        """Return expectation value of Z on `qubit` (range 0..n_qubits-1)."""
        assert 0 <= qubit < self.n_qubits
        Z_op = single_qubit_operator(self.n_qubits, Z, qubit)
        return float(np.real(np.vdot(state, Z_op @ state)))

    def get_parameters(self):
        # flatten parameters from all layers into 1D array
        return np.concatenate([layer.params.flatten() for layer in self.layers])

    def set_parameters(self, flat_params):
        # set params back to layers (expects correct size)
        idx = 0
        for layer in self.layers:
            n = layer.params.size
            layer.params = flat_params[idx:idx+n].reshape(layer.params.shape)
            idx += n

    def num_parameters(self):
        return sum(layer.num_params() for layer in self.layers)
