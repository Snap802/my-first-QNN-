# QNNLIB!

Minimal educational Quantum Neural Network simulation library.

## About
- Uses dense statevector simulation (numpy).
- Intended for small numbers of qubits (memory/time scale as 2^n).

## Quickstart
1. Unzip the package and `cd` into the directory containing the `qnnlib/` folder.
2. Create virtualenv: `python -m venv venv` then `source venv/bin/activate` (Linux/macOS) or `venv\\Scripts\\activate` (Windows).
3. Install requirements: `pip install -r requirements.txt`.
4. Run demo: `python -m examples.xor_demo` or `python examples/xor_demo.py`.

## Warning
Simulation memory grows as 2**n_qubits. Keep n_qubits small (<=12 recommended).
