import numpy as np
from .gates import kron, I

def zero_state(n_qubits):
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    return state

def normalize(state):
    norm = np.linalg.norm(state)
    if norm == 0:
        return state
    return state / norm

def apply_unitary(state, U):
    return U @ state
