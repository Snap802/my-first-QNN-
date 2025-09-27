import numpy as np

def mean_squared_loss(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    return float(np.mean((preds - targets)**2))

def evaluate(qnn, X, Y):
    preds = []
    for x in X:
        state = qnn.forward(x)
        preds.append(qnn.expectation(state))
    loss = mean_squared_loss(preds, Y)
    return loss, preds

def numerical_gradient(qnn, X, Y, eps=1e-3):
    """Compute numerical gradients of loss wrt parameters (finite difference)."""
    orig_params = qnn.get_parameters()
    grads = np.zeros_like(orig_params)
    base_loss, _ = evaluate(qnn, X, Y)
    for i in range(len(orig_params)):
        p0 = orig_params.copy()
        p0[i] += eps
        qnn.set_parameters(p0)
        l_plus, _ = evaluate(qnn, X, Y)

        p1 = orig_params.copy()
        p1[i] -= eps
        qnn.set_parameters(p1)
        l_minus, _ = evaluate(qnn, X, Y)

        grads[i] = (l_plus - l_minus) / (2 * eps)
    # restore params
    qnn.set_parameters(orig_params)
    return grads

def train(qnn, X, Y, epochs=50, lr=0.1, eps=1e-3, verbose=True):
    """Simple gradient-descent training with numerical gradients."""
    for epoch in range(1, epochs+1):
        loss, _ = evaluate(qnn, X, Y)
        grads = numerical_gradient(qnn, X, Y, eps=eps)
        params = qnn.get_parameters()
        params = params - lr * grads
        qnn.set_parameters(params)
        if verbose and (epoch % max(1, epochs//10) == 0 or epoch<=5):
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")

