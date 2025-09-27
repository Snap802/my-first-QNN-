"""qnnlib - Minimal, educational Quantum Neural Network simulation library.
This library simulates quantum states with dense statevectors and is intended
for small numbers of qubits (typically n_qubits <= 12) on classical hardware.
"""

from .network import SimpleQNN
from .training import train, evaluate
__all__ = ["SimpleQNN", "train", "evaluate"]
