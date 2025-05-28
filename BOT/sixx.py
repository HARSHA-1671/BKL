import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)
def add_bias(X): return np.hstack([np.ones((X.shape[0], 1)), X])
def compute_W(X, x0, tau):
    d = X - x0
    W = np.exp(-np.sum(d**2, axis=1) / (2 * tau**2))
    return np.diag(W)
def lwlr(x0, X, y, tau):
    X_b = add_bias(X)
    x0_b = add_bias(x0.reshape(1, -1))
    W = compute_W(X, x0, tau)
    theta = np.linalg.pinv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y)
    return x0_b @ theta
x_query = np.linspace(X.min(), X.max(), 300)
y_pred = np.array([lwlr(x, X, y, tau=0.5) for x in x_query])
plt.scatter(X, y, alpha=0.6)
plt.plot(x_query, y_pred, color='red')
plt.title("LWLR")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()