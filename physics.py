"""Synthetic moving-phantom benchmark. mm, s, cm, Gy unless stated otherwise."""

import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from methods import trapezoid


CONFIG = {
    "amplitude_mm": 4.0,
    "period_s": 4.0,
    "duration_s": 8.0,
    "mass_kg": 1.0,
    "field_half_width_mm": 7.0,
    "penumbra_sigma_mm": 1.5,
    "dose_rate_gy_s": 0.4,
    "depth_cm": 5.0,
    "attenuation_per_cm": 0.05,
    "sample_dt_s": 0.5,
    "noise_sd_mm": 0.15,
    "seed": 2026
}


def motion(t, cfg=CONFIG):
    """Exact position, velocity, acceleration of an undamped oscillator."""
    t = np.asarray(t, dtype=float)
    w = 2 * np.pi / cfg["period_s"]
    amp = cfg["amplitude_mm"]
    x = amp * np.cos(w * t)
    v = -amp * w * np.sin(w * t)
    a = -amp * w ** 2 * np.cos(w * t)
    return x, v, a


def oscillator(t, method="euler", cfg=CONFIG):
    """Integrate the separate states x'=v and v'=-omega^2*x."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Use at least two finite, increasing times.")
    if method not in ["euler", "heun"]:
        raise ValueError("Choose euler or heun.")
    w = 2 * np.pi / cfg["period_s"]
    states = np.zeros((len(t), 2))
    x, v, _ = motion(t[0], cfg)
    states[0] = [x, v]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        x, v = states[i]
        first = np.array([v, -w ** 2 * x])
        prediction = states[i] + dt * first
        if method == "euler":
            states[i + 1] = prediction
        else:
            second = np.array([prediction[1], -w ** 2 * prediction[0]])
            states[i + 1] = states[i] + 0.5 * dt * (first + second)
    return states


def beam_profile(x, cfg=CONFIG):
    """Prescribed softened field profile, not a photon-transport calculation."""
    x = np.asarray(x, dtype=float)
    width = cfg["field_half_width_mm"]
    sd = cfg["penumbra_sigma_mm"]
    return 0.5 * (erf((x + width) / (np.sqrt(2) * sd)) - erf((x - width) / (np.sqrt(2) * sd)))


def dose_rate(x, cfg=CONFIG):
    """Illustrative dose rate at a fixed depth. Depth dependence is assumed."""
    scale = cfg["dose_rate_gy_s"] * np.exp(-cfg["attenuation_per_cm"] * cfg["depth_cm"])
    return scale * beam_profile(x, cfg)


def accumulated_dose(t, displacement, points, cfg=CONFIG):
    """Follow material points of a rigid phantom through a stationary field."""
    t = np.asarray(t, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    points = np.atleast_1d(np.asarray(points, dtype=float))
    if displacement.shape != t.shape or not np.all(np.isfinite(points)):
        raise ValueError("Position samples must match the times; points must be finite.")
    rates = dose_rate(points[:, None] + displacement[None, :], cfg)
    return trapezoid(t, rates)


def reference_dose(points, cfg=CONFIG):
    """Independent adaptive quadrature along the exact trajectory."""
    points = np.atleast_1d(np.asarray(points, dtype=float))
    doses = np.zeros(len(points))
    estimates = np.zeros(len(points))
    for i in range(len(points)):
        def integrand(t):
            return float(dose_rate(points[i] + motion(t, cfg)[0], cfg))
        doses[i], estimates[i] = quad(integrand, 0, cfg["duration_s"], epsabs=1e-11, epsrel=1e-11, limit=200)
    return doses, estimates


def survival(dose, alpha=0.30, beta=0.03):
    """Illustrative acute-exposure LQ surviving fraction, not patient outcome."""
    dose = np.asarray(dose, dtype=float)
    if not np.all(np.isfinite(dose)) or np.any(dose < 0) or alpha < 0 or beta < 0:
        raise ValueError("Dose and LQ parameters must be finite/nonnegative.")
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("LQ parameters must be finite.")
    return np.exp(-alpha * dose - beta * dose ** 2)


def repair_system(dose, dt_min=60.0):
    """Separate toy post-exposure model, with states H, R, I per location.

    R repairs into H or becomes irreversibly damaged I. The arbitrary initial
    R fraction is 1-exp(-0.25*D). This is NOT a mechanistic derivation of LQ.
    Return the implicit-Euler system, its initial state, generator and rates.
    """
    dose = np.asarray(dose, dtype=float)
    if dose.ndim != 1 or len(dose) < 1 or np.any(dose < 0) or not np.all(np.isfinite(dose)):
        raise ValueError("Need a nonempty finite nonnegative dose vector.")
    if not np.isfinite(dt_min) or dt_min <= 0:
        raise ValueError("Time step must be positive and finite.")
    n = len(dose)
    rates = np.linspace(0.01, 0.06, n)  # Illustrative repair rates per minute
    death = 0.008                       # Illustrative loss rate per minute
    k = np.zeros((3 * n, 3 * n))
    initial = np.zeros(3 * n)
    for i in range(n):
        j = 3 * i
        damaged = 1 - np.exp(-0.25 * dose[i])
        initial[j:j + 3] = [1 - damaged, damaged, 0.0]
        k[j, j + 1] = rates[i]
        k[j + 1, j + 1] = -(rates[i] + death)
        k[j + 2, j + 1] = death
    a = np.eye(3 * n) - dt_min * k
    return a, initial, k, rates, death


def repair_exact(initial, rates, death, time_min):
    """Closed-form solution of the specific post-exposure three-state model."""
    out = np.asarray(initial, dtype=float).copy()
    for i in range(len(rates)):
        j = 3 * i
        h, r, loss = initial[j:j + 3]
        total = rates[i] + death
        left = r * np.exp(-total * time_min)
        out[j] = h + rates[i] / total * (r - left)
        out[j + 1] = left
        out[j + 2] = loss + death / total * (r - left)
    return out
