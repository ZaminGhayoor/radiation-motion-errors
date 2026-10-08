"""Small numerical methods used in the experiments. Arrays use float64."""

import numpy as np
from scipy.interpolate import CubicSpline


def check_nodes(t, y):
    """Return finite, increasing 1D nodes and matching values."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 2:
        raise ValueError("Need at least two matching one-dimensional arrays.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Nodes and values must be finite.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Nodes must be strictly increasing.")
    return t, y


def divided_differences(t, y):
    """Coefficients of the Newton polynomial, using a divided-difference table."""
    t, y = check_nodes(t, y)
    n = len(t)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (t[i + j] - t[i])
    return table[0].copy()


def newton_value(t, c, q):
    """Evaluate a Newton polynomial with nested multiplication."""
    q = np.asarray(q, dtype=float)
    p = np.zeros_like(q) + c[-1]
    for j in range(len(c) - 2, -1, -1):
        p = c[j] + (q - t[j]) * p
    return p


def reconstruct(t, y, q, method):
    """Interpolate only inside the observed interval; no silent extrapolation."""
    t, y = check_nodes(t, y)
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or not np.all(np.isfinite(q)):
        raise ValueError("Query times must be a finite one-dimensional array.")
    if np.any(q < t[0]) or np.any(q > t[-1]):
        raise ValueError("Query times must lie inside the observed interval.")
    if method == "linear":
        return np.interp(q, t, y)
    if method == "cubic":
        # Natural endpoints set the reconstructed acceleration to zero there.
        return CubicSpline(t, y, bc_type="natural", extrapolate=False)(q)
    if method == "local newton":
        if len(t) < 4:
            raise ValueError("Local cubic Newton interpolation needs four nodes.")
        out = np.zeros(len(q))
        interval = np.searchsorted(t, q, side="right") - 1
        # Four neighboring nodes per interval keep the polynomial degree at 3.
        for i in range(len(t) - 1):
            start = min(max(i - 1, 0), len(t) - 4)
            nodes = t[start:start + 4]
            c = divided_differences(nodes, y[start:start + 4])
            mask = interval == i
            if i == len(t) - 2:
                mask = mask | (q == t[-1])
            out[mask] = newton_value(nodes, c, q[mask])
        return out
    raise ValueError("Unknown interpolation method.")


def central_differences(t, y):
    """First and second derivatives at interior nodes of a uniform grid."""
    t, y = check_nodes(t, y)
    if len(t) < 3:
        raise ValueError("Need at least three nodes for central differences.")
    h = t[1] - t[0]
    if not np.allclose(np.diff(t), h, rtol=1e-9, atol=1e-12):
        raise ValueError("This formula requires uniform time spacing.")
    v = np.zeros(len(t) - 2)
    a = np.zeros(len(t) - 2)
    for i in range(1, len(t) - 1):
        v[i - 1] = (y[i + 1] - y[i - 1]) / (2 * h)
        a[i - 1] = (y[i + 1] - 2 * y[i] + y[i - 1]) / h ** 2
    return t[1:-1], v, a


def trapezoid(t, y):
    """Integrate over the last axis; y may hold many spatial positions."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or len(t) < 2 or y.ndim < 1 or y.shape[-1] != len(t):
        raise ValueError("The last data axis must match at least two times.")
    if np.any(np.diff(t) <= 0) or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Use finite data and increasing times.")
    return np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(t), axis=-1)


def rmse(estimate, reference):
    """Root mean square error, in the units of the inputs."""
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(reference)) ** 2)))


def gmres_steps(a, b, max_steps=30, tol=1e-12, reorthogonalize=True):
    """Unrestarted GMRES with x0=0 and modified Gram-Schmidt Arnoldi.

    Keep every iterate so residual and forward error can be compared later.
    This small dense implementation is for studying the algorithm, not speed.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.ndim != 1 or a.shape != (len(b), len(b)) or len(b) == 0:
        raise ValueError("Need a square matrix and a matching nonempty vector.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Matrix and vector must be finite.")
    if max_steps < 1 or tol <= 0:
        raise ValueError("Steps and tolerance must be positive.")
    size = min(int(max_steps), len(b))
    beta = np.linalg.norm(b)
    if beta == 0:
        return np.zeros_like(b), []
    q = np.zeros((len(b), size + 1))
    h = np.zeros((size + 1, size))
    q[:, 0] = b / beta
    history = []
    scale = max(np.linalg.norm(a, ord="fro"), 1.0)
    for k in range(size):
        w = a @ q[:, k]
        passes = 2 if reorthogonalize else 1
        for repeat in range(passes):
            for j in range(k + 1):
                dot = q[:, j] @ w
                h[j, k] = h[j, k] + dot
                w = w - dot * q[:, j]
        h[k + 1, k] = np.linalg.norm(w)
        breakdown = h[k + 1, k] <= 10 * np.finfo(float).eps * scale
        if not breakdown:
            q[:, k + 1] = w / h[k + 1, k]
        rhs = np.zeros(k + 2)
        rhs[0] = beta
        small_h = h[:k + 2, :k + 1]
        c = np.linalg.lstsq(small_h, rhs, rcond=None)[0]
        x = q[:, :k + 1] @ c
        rel_residual = np.linalg.norm(b - a @ x) / beta
        basis = q[:, :k + 1]
        orth_error = np.linalg.norm(basis.T @ basis - np.eye(k + 1))
        relation = a @ basis - q[:, :k + 2] @ small_h
        history.append({
            "iteration": k + 1,
            "relative_residual": float(rel_residual),
            "projected_residual": float(np.linalg.norm(rhs - small_h @ c) / beta),
            "orthogonality_error": float(orth_error),
            "arnoldi_relation_error": float(np.linalg.norm(relation) / scale),
            "x": x.copy()
        })
        if rel_residual <= tol or breakdown:
            break
    return x, history
