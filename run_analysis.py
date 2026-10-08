"""Run the synthetic experiments and rebuild the figures and CSV tables."""

from pathlib import Path
import csv
import json
import platform

import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullLocator

from methods import reconstruct, rmse, gmres_steps
from physics import CONFIG, motion, oscillator, dose_rate, accumulated_dose, reference_dose, survival, repair_system, repair_exact


ROOT = Path(__file__).resolve().parent
COLORS = {"linear": "#1b7a78", "local newton": "#b56939", "cubic": "#7356a4"}
METHODS = ["linear", "local newton", "cubic"]


def save_table(name, rows):
    with (ROOT / "results" / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish(fig, name):
    fig.savefig(ROOT / "figures" / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def setup():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "semibold", "axes.titlepad": 12,
        "axes.grid": True, "grid.alpha": 0.18,
        "lines.linewidth": 2, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False
    })


def diagram():
    fig, ax = plt.subplots(figsize=(10, 5.3), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 5.5))
    ax.axis("off")
    boxes = [
        (0.3, 3.8, "Exact phantom motion\nposition x, velocity v", "#e5f2f0"),
        (3.65, 3.8, "Synthetic sensor\nspacing + noise + bias", "#faf0e3"),
        (7.0, 3.8, "Reconstructed motion\nlinear / Newton / cubic", "#eee8f5"),
        (0.3, 1.8, "Reference dose\nadaptive quadrature", "#e5f2f0"),
        (7.0, 1.8, "Estimated dose\ntrapezoid integration", "#eee8f5"),
        (3.65, 0.2, "Error comparisons\nposition / dose / survival", "#e9edf2")
    ]
    for x, y, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.65, 1.0, boxstyle="round,pad=0.12", facecolor=color, edgecolor="#94a0a8"))
        ax.text(x + 1.325, y + 0.5, label, ha="center", va="center", fontsize=10.5)
    arrows = [((2.98, 4.3), (3.48, 4.3)), ((6.33, 4.3), (6.83, 4.3)),
              ((1.63, 3.65), (1.63, 2.96)), ((8.33, 3.65), (8.33, 2.96)),
              ((2.4, 1.65), (3.6, 0.85)), ((7.65, 1.65), (6.4, 0.85))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#455363", "lw": 1.5})
    ax.text(5, 5.2, "Where the errors enter", ha="center", fontsize=16, weight="semibold")
    finish(fig, "model")


def main_comparison():
    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])
    t = np.linspace(0, cfg["duration_s"], 1601)
    sample_count = int(round(cfg["duration_s"] / cfg["sample_dt_s"])) + 1
    ts = np.linspace(0, cfg["duration_s"], sample_count)
    x, v, a = motion(t)
    clean = motion(ts)[0]
    observed = clean + rng.normal(0, cfg["noise_sd_mm"], len(ts))
    estimates = {}
    rows = []
    points = np.linspace(-18, 18, 241)
    d_ref, quad_est = reference_dose(points)
    d_static = cfg["duration_s"] * dose_rate(points)
    dose_estimates = {}
    s_ref = survival(d_ref)
    for method in METHODS:
        estimate = reconstruct(ts, observed, t, method)
        clean_estimate = reconstruct(ts, clean, t, method)
        estimates[method] = estimate
        d = accumulated_dose(t, estimate, points)
        dose_estimates[method] = d
        interior = (t >= ts[1]) & (t <= ts[-2])
        rows.append({
            "method": method,
            "position_rmse_mm": rmse(estimate, x),
            "interior_position_rmse_mm": rmse(estimate[interior], x[interior]),
            "clean_position_rmse_mm": rmse(clean_estimate, x),
            "max_position_error_mm": float(np.max(np.abs(estimate - x))),
            "dose_rmse_gy": rmse(d, d_ref),
            "max_dose_error_gy": float(np.max(np.abs(d - d_ref))),
            "survival_rmse": rmse(survival(d), s_ref)
        })
    save_table("method_comparison.csv", rows)
    save_table("sensor_samples.csv", [{"time_s": float(ts[i]), "true_position_mm": float(clean[i]), "observed_position_mm": float(observed[i])} for i in range(len(ts))])
    dose_rows = []
    for i in range(len(points)):
        row = {"material_position_mm": float(points[i]), "stationary_dose_gy": float(d_static[i]), "reference_dose_gy": float(d_ref[i])}
        for method in METHODS:
            row[method.replace(" ", "_") + "_dose_gy"] = float(dose_estimates[method][i])
        dose_rows.append(row)
    save_table("dose_profiles.csv", dose_rows)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    axes[0].plot(t, x, color="#283443", label="Exact motion")
    axes[0].scatter(ts, observed, color="#283443", s=28, zorder=5, label="Noisy samples")
    for method in METHODS:
        axes[0].plot(t, estimates[method], color=COLORS[method], alpha=0.8, label=method.title())
        axes[1].plot(t, estimates[method] - x, color=COLORS[method], label=method.title())
    axes[0].set(title="The curves look close. Their errors tell a different story.", ylabel="Position (mm)")
    axes[0].legend(ncol=3, fontsize=9, loc="upper right")
    axes[1].axhline(0, color="#283443", lw=0.8)
    axes[1].set(xlabel="Time (s)", ylabel="Signed position error (mm)")
    finish(fig, "motion_reconstruction")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes[0, 0].plot(points, d_static, "--", color="#9b9b9b", label="Stationary assumption")
    axes[0, 0].plot(points, d_ref, color="#283443", label="Exact moving reference")
    for method in METHODS:
        axes[0, 0].plot(points, dose_estimates[method], color=COLORS[method], alpha=0.7, label=method.title())
        axes[0, 1].plot(points, dose_estimates[method] - d_ref, color=COLORS[method], label=method.title())
    axes[0, 0].set(title="Motion spreads the prescribed field", ylabel="Accumulated dose (Gy)", xlabel="Material position (mm)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Reconstruction error in dose", ylabel="Dose error (Gy)", xlabel="Material position (mm)")
    doses = np.linspace(0, 6, 201)
    axes[1, 0].plot(doses, survival(doses), color="#1b7a78", label="Example A: alpha=0.30, beta=0.03")
    axes[1, 0].plot(doses, survival(doses, 0.15, 0.05), color="#b56939", label="Example B: alpha=0.15, beta=0.05")
    axes[1, 0].set(title="Two illustrative LQ responses", xlabel="Acute dose (Gy)", ylabel="Surviving fraction", yscale="log")
    axes[1, 0].legend(fontsize=8)
    for method in METHODS:
        delta = dose_estimates[method] - d_ref
        axes[1, 1].plot(points, survival(dose_estimates[method]) - s_ref, color=COLORS[method], label=method.title())
    delta = dose_estimates["cubic"] - d_ref
    first_order = -(0.30 + 0.06 * d_ref) * s_ref * delta
    axes[1, 1].plot(points, first_order, ":", color="#283443", label="Cubic: first-order estimate")
    axes[1, 1].set(title="Dose error reaches the response model", xlabel="Material position (mm)", ylabel="Surviving-fraction error")
    axes[1, 1].legend(fontsize=8)
    finish(fig, "dose_and_survival")
    return {"rows": rows, "quadrature_max_estimate_gy": float(max(quad_est)), "t": t, "ts": ts, "observed": observed, "clean": clean, "points": points, "d_ref": d_ref}


def sampling_experiment(repeats=50):
    rng = np.random.default_rng(2027)
    t = np.linspace(0, 8, 801)
    true = motion(t)[0]
    points = np.linspace(-14, 14, 41)
    d_ref = reference_dose(points)[0]
    rows = []
    raw = []
    for dt in [0.125, 0.25, 0.5, 1.0]:
        ts = np.linspace(0, 8, int(round(8 / dt)) + 1)
        for noise in [0.0, 0.15]:
            values = {method: [] for method in METHODS}
            for trial in range(repeats if noise > 0 else 1):
                measured = motion(ts)[0] + rng.normal(0, noise, len(ts))
                # Each method sees exactly the same samples in a trial.
                for method in METHODS:
                    x = reconstruct(ts, measured, t, method)
                    d = accumulated_dose(t, x, points)
                    err = rmse(d, d_ref)
                    pos_err = rmse(x, true)
                    values[method].append(err)
                    raw.append({"sample_dt_s": dt, "noise_sd_mm": noise, "trial": trial, "method": method, "position_rmse_mm": pos_err, "dose_rmse_gy": err})
            for method in METHODS:
                val = np.asarray(values[method])
                rows.append({"sample_dt_s": dt, "noise_sd_mm": noise, "method": method, "repeats": len(val), "median_dose_rmse_gy": float(np.median(val)), "q25": float(np.quantile(val, 0.25)), "q75": float(np.quantile(val, 0.75))})
    save_table("sampling_trials.csv", raw)
    save_table("sampling_summary.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout="constrained")
    for ax, noise in zip(axes, [0.0, 0.15]):
        for method in METHODS:
            selected = [r for r in rows if r["noise_sd_mm"] == noise and r["method"] == method]
            dt = [r["sample_dt_s"] for r in selected]
            med = [r["median_dose_rmse_gy"] for r in selected]
            ax.plot(dt, med, "o-", color=COLORS[method], label=method.title())
            if noise > 0:
                ax.fill_between(dt, [r["q25"] for r in selected], [r["q75"] for r in selected], alpha=0.15, color=COLORS[method])
        title = "Exact samples" if noise == 0 else "0.15 mm noise; 50 trials"
        ax.set(title=title, xlabel="Sampling interval (s)", xscale="log", yscale="log")
        ax.set_xticks([0.125, 0.25, 0.5, 1.0], ["0.125", "0.25", "0.5", "1.0"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Dose RMSE across 41 positions (Gy)")
    fig.suptitle("Sampling error and sensor noise behave differently", fontsize=15)
    finish(fig, "sampling_and_noise")


def differentiation_experiment():
    rng = np.random.default_rng(2028)
    t0 = 0.37
    x0, exact_v, exact_a = motion(t0)
    rows = []
    for h in np.logspace(-9, -0.3, 60):
        xm = motion(t0 - h)[0]
        xp = motion(t0 + h)[0]
        forward = (xp - x0) / h
        central = (xp - xm) / (2 * h)
        rows.append({"h_s": float(h), "forward_velocity_error_mm_s": float(abs(forward - exact_v)), "central_velocity_error_mm_s": float(abs(central - exact_v))})
    save_table("difference_roundoff.csv", rows)
    noisy = []
    for h in np.logspace(-2, 0, 25):
        xm = motion(t0 - h)[0]
        xp = motion(t0 + h)[0]
        clean_a = (xp - 2 * x0 + xm) / h ** 2
        e = rng.normal(0, 0.15, (4000, 3))
        est = clean_a + (e[:, 0] - 2 * e[:, 1] + e[:, 2]) / h ** 2
        # Independent sensor errors give variance (1+4+1)*sigma^2/h^4.
        theory = np.sqrt((clean_a - exact_a) ** 2 + 6 * 0.15 ** 2 / h ** 4)
        noisy.append({"h_s": float(h), "acceleration_rmse_mm_s2": rmse(est, exact_a), "predicted_rmse_mm_s2": float(theory), "clean_acceleration_error_mm_s2": float(abs(clean_a - exact_a))})
    save_table("difference_noise.csv", noisy)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for key, label, color in [("forward_velocity_error_mm_s", "Forward difference", "#b56939"), ("central_velocity_error_mm_s", "Central difference", "#1b7a78")]:
        axes[0].loglog([r["h_s"] for r in rows], [r[key] for r in rows], color=color, label=label)
    axes[0].set(title="Truncation gives way to roundoff", xlabel="Difference step h (s)", ylabel="Absolute velocity error (mm/s)")
    axes[0].legend(fontsize=9)
    for key, label, color, style in [("acceleration_rmse_mm_s2", "4,000 noisy draws", "#7356a4", "-"), ("predicted_rmse_mm_s2", "Bias + noise formula", "#283443", ":"), ("clean_acceleration_error_mm_s2", "Noiseless truncation", "#1b7a78", "--")]:
        axes[1].loglog([r["h_s"] for r in noisy], [r[key] for r in noisy], style, color=color, label=label)
    axes[1].set(title="A second derivative amplifies noise", xlabel="Difference step h (s)", ylabel="Acceleration error (mm/s²)")
    axes[1].legend(fontsize=9)
    finish(fig, "differentiation")


def ode_experiment():
    rows = []
    for method in ["euler", "heun"]:
        last = None
        for dt in [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]:
            t = np.linspace(0, 8, int(round(8 / dt)) + 1)
            states = oscillator(t, method)
            ref_x, ref_v, _ = motion(t)
            err = rmse(states[:, 0], ref_x)
            order = "" if last is None else float(np.log2(last / err))
            rows.append({"method": method, "dt_s": dt, "position_rmse_mm": err, "velocity_rmse_mm_s": rmse(states[:, 1], ref_v), "observed_position_order": order})
            last = err
    save_table("ode_convergence.csv", rows)
    t = np.linspace(0, 8, 161)
    true_x, true_v, _ = motion(t)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    axes[0].plot(true_x, true_v, color="#283443", label="Exact orbit")
    for method, color in [("euler", "#b56939"), ("heun", "#1b7a78")]:
        states = oscillator(t, method)
        axes[0].plot(states[:, 0], states[:, 1], color=color, label=method.title())
        selected = [r for r in rows if r["method"] == method]
        axes[1].loglog([r["dt_s"] for r in selected], [r["position_rmse_mm"] for r in selected], "o-", color=color, label=method.title())
    axes[0].set(title="Separate states: position and velocity", xlabel="Position (mm)", ylabel="Velocity (mm/s)")
    axes[1].set(title="Smaller steps reduce integration error", xlabel="Time step (s)", ylabel="Position RMSE (mm)")
    for ax in axes:
        ax.legend(fontsize=9)
    finish(fig, "newtonian_states")


def quadrature_experiment():
    # A partial cycle avoids the special cancellation of full-period sampling.
    cfg = CONFIG.copy()
    cfg["duration_s"] = 7.3
    point = np.array([6.0])
    ref = float(reference_dose(point, cfg)[0][0])
    rows = []
    last = None
    for n in [16, 32, 64, 128, 256, 512]:
        t = np.linspace(0, cfg["duration_s"], n + 1)
        estimate = float(accumulated_dose(t, motion(t, cfg)[0], point, cfg)[0])
        err = abs(estimate - ref)
        order = "" if last is None else float(np.log2(last / err))
        rows.append({"intervals": n, "dt_s": cfg["duration_s"] / n, "dose_error_gy": err, "observed_order": order})
        last = err
    save_table("quadrature_convergence.csv", rows)


def error_budget(data):
    # Signed differences telescope exactly; these are not independent variances.
    point = np.array([6.0])
    t = data["t"]
    ts = data["ts"]
    d_static = float(8 * dose_rate(point)[0])
    d_ref = float(reference_dose(point)[0][0])
    d_grid = float(accumulated_dose(t, motion(t)[0], point)[0])
    d_clean = float(accumulated_dose(t, reconstruct(ts, data["clean"], t, "cubic"), point)[0])
    d_noisy = float(accumulated_dose(t, reconstruct(ts, data["observed"], t, "cubic"), point)[0])
    d_bias = float(accumulated_dose(t, reconstruct(ts, data["observed"] + 0.2, t, "cubic"), point)[0])
    values = [d_static, d_ref, d_grid, d_clean, d_noisy, d_bias]
    names = ["Physical motion", "Quadrature", "Interpolation", "Sensor noise", "0.2 mm bias"]
    rows = []
    for i in range(len(names)):
        rows.append({"source": names[i], "signed_dose_change_gy": values[i + 1] - values[i], "before_gy": values[i], "after_gy": values[i + 1]})
    save_table("signed_error_budget.csv", rows)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    changes = [r["signed_dose_change_gy"] for r in rows]
    ax.barh(names, changes, color=["#808c98", "#93b7b3", "#7356a4", "#b56939", "#1b7a78"])
    ax.axvline(0, color="#283443", lw=0.8)
    for i in range(len(names)):
        ax.annotate("%.2e Gy" % changes[i], (changes[i], i), xytext=(5 if changes[i] >= 0 else -5, 0), textcoords="offset points", ha="left" if changes[i] >= 0 else "right", va="center", fontsize=9)
    ax.margins(x=0.45)
    ax.invert_yaxis()
    ax.set(title="One point near the field edge: what changed the dose?", xlabel="Signed contribution to the final difference (Gy)")
    finish(fig, "error_budget")
    return float(sum(changes) - (d_bias - d_static))


def arnoldi_experiment(data):
    doses = np.interp(np.linspace(-12, 12, 40), data["points"], data["d_ref"])
    a, initial, k, rates, death = repair_system(doses)
    direct = np.linalg.solve(a, initial)
    condition = float(np.linalg.cond(a))
    rows = []
    final = None
    for reorth in [False, True]:
        estimate, history = gmres_steps(a, initial, max_steps=35, reorthogonalize=reorth)
        for row in history:
            forward_error = float(np.linalg.norm(row["x"] - direct) / np.linalg.norm(direct))
            record = {key: value for key, value in row.items() if key != "x"}
            record["passes"] = 2 if reorth else 1
            record["relative_forward_error"] = forward_error
            record["condition_times_residual"] = condition * row["relative_residual"]
            rows.append(record)
        if reorth:
            final = estimate
    save_table("arnoldi_gmres.csv", rows)
    exact = repair_exact(initial, rates, death, 60)
    time_rows = []
    for steps in [1, 2, 4, 8, 16, 32, 64]:
        dt = 60 / steps
        step_matrix = np.eye(len(initial)) - dt * k
        y = initial.copy()
        for i in range(steps):
            y = np.linalg.solve(step_matrix, y)
        time_rows.append({"steps": steps, "dt_min": dt, "relative_time_discretization_error": float(np.linalg.norm(y - exact) / np.linalg.norm(exact))})
    save_table("repair_time_error.csv", time_rows)
    states = final.reshape(-1, 3)
    save_table("repair_states.csv", [{"location": i, "H": float(states[i, 0]), "R": float(states[i, 1]), "I": float(states[i, 2])} for i in range(len(states))])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    selected = [r for r in rows if r["passes"] == 2]
    for key, label, color, style in [("relative_residual", "Residual", "#1b7a78", "-"), ("relative_forward_error", "Error vs direct solve", "#b56939", "--"), ("condition_times_residual", "Condition-number bound", "#808c98", ":")]:
        axes[0].semilogy([r["iteration"] for r in selected], [r[key] for r in selected], style, color=color, label=label)
    axes[0].set(title="Residual and actual solver error", xlabel="Arnoldi / GMRES step", ylabel="Relative 2-norm")
    axes[0].legend(fontsize=8)
    for passes, color in [(1, "#b56939"), (2, "#1b7a78")]:
        selected = [r for r in rows if r["passes"] == passes]
        axes[1].semilogy([r["iteration"] for r in selected], [max(r["orthogonality_error"], 1e-18) for r in selected], color=color, label=str(passes) + " Gram-Schmidt pass(es)")
    axes[1].set(title="Does the basis stay orthogonal?", xlabel="Arnoldi / GMRES step", ylabel="Frobenius norm of QᵀQ − I")
    axes[1].legend(fontsize=8)
    axes[2].loglog([r["dt_min"] for r in time_rows], [r["relative_time_discretization_error"] for r in time_rows], "o-", color="#7356a4")
    axes[2].set(title="A different error: the time step", xlabel="Implicit-Euler step (min)", ylabel="Relative error vs exact ODE")
    finish(fig, "arnoldi_errors")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    times = np.linspace(0, 120, 121)
    curves = np.zeros((len(times), 3))
    location = 20
    for i in range(len(times)):
        curves[i] = repair_exact(initial, rates, death, times[i]).reshape(-1, 3)[location]
    for i, label, color in [(0, "H: no damage in this model", "#1b7a78"), (1, "R: repairable damage", "#b56939"), (2, "I: irreversible damage", "#7356a4")]:
        ax.plot(times, curves[:, i], color=color, label=label)
    ax.set(title="A separate toy model of post-exposure cell states", xlabel="Time after exposure (min)", ylabel="Fraction", ylim=(0, 1))
    ax.legend(fontsize=9)
    finish(fig, "repair_states")
    return {"matrix_size": len(initial), "condition_number_2": condition, "final_gmres_relative_residual": rows[-1]["relative_residual"], "final_gmres_relative_error": rows[-1]["relative_forward_error"], "one_step_time_relative_error": time_rows[0]["relative_time_discretization_error"], "mass_error": float(np.max(np.abs(states.sum(axis=1) - 1)))}


def main():
    setup()
    diagram()
    data = main_comparison()
    sampling_experiment()
    differentiation_experiment()
    ode_experiment()
    quadrature_experiment()
    budget_error = error_budget(data)
    arnoldi = arnoldi_experiment(data)
    report = {"config": CONFIG, "methods": data["rows"], "quadrature_max_estimate_gy": data["quadrature_max_estimate_gy"], "signed_budget_closure_gy": budget_error, "arnoldi": arnoldi,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    (ROOT / "results" / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in data["rows"]:
        print("%-14s position RMSE %.4f mm; dose RMSE %.6f Gy" % (row["method"], row["position_rmse_mm"], row["dose_rmse_gy"]))
    print("GMRES relative residual: %.3e" % arnoldi["final_gmres_relative_residual"])
    print("One-step ODE relative error: %.3e" % arnoldi["one_step_time_relative_error"])
    print("Figures and tables saved next to this script.")
    return report


if __name__ == "__main__":
    main()
