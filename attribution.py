
import torch
import numpy as np

def color_marker_sensitivity(model, X):
    X_t = torch.tensor(X, requires_grad=True)
    pred = model(X_t)
    pred.sum().backward()
    grads = X_t.grad.numpy()
    return grads[:, 6]  # index 6 = color_marker

def color_flip_test(model, X):
    X_orig = X.copy()
    X_flip = X.copy()
    X_flip[:, 6] = 1.0 - X_flip[:, 6]  # flip the binary marker

    with torch.no_grad():
        pred_orig = model(torch.tensor(X_orig)).numpy()
        pred_flip = model(torch.tensor(X_flip)).numpy()

    delta = np.abs(pred_flip - pred_orig)
    return delta  # angle change (radians) purely from flipping color