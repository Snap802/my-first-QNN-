import numpy as np

# Basic single-qubit matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)

def Rx(theta):
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def Ry(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def Rz(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)

def kron(*mats):
    """Tensor product of matrices (left-to-right)."""
    out = np.array([[1]], dtype=complex)
    for m in mats:
        out = np.kron(out, m)
    return out

def single_qubit_operator(n_qubits, gate, target):
    """Construct the full 2^n x 2^n operator that applies `gate` to `target` qubit."""
    ops = [I] * n_qubits
    ops[target] = gate
    return kron(*ops)

def controlled_x(n_qubits, control, target):
    """Construct CNOT (control -> target) for `n_qubits` using projectors."""
    P0 = np.array([[1,0],[0,0]], dtype=complex)
    P1 = np.array([[0,0],[0,1]], dtype=complex)
    # Build operator acting on all qubits for the two branches
    ops0 = [I] * n_qubits
    ops1 = [I] * n_qubits
    ops0[control] = P0
    ops1[control] = P1
    # target: I in branch 0, X in branch 1
    ops0[target] = I
    ops1[target] = X
    return kron(*ops0) + kron(*ops1)
