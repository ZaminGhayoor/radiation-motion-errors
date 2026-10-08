# Radiation motion errors

How much can a small position error change a dose estimate?

This is a numerical-methods project built around that question. A simulated phantom moves back and forth through a fixed radiation field. Its position is only recorded at a few times, so the motion between measurements has to be reconstructed. I compare three interpolation methods, then follow the errors into accumulated dose and a simple biological response model.

The setup is deliberately small enough to work through by hand: one spatial dimension, a known motion equation, and synthetic measurements. There are no patient records, measured beam data, or trained machine-learning models here. This is a study model, not a treatment-planning tool.

![How the measurements and reference calculations connect](figures/model.png)

## Start here

Open [walkthrough.ipynb](walkthrough.ipynb). It has saved outputs, short explanations, and examples that call the functions directly.

To rerun everything from a terminal, start inside this folder:

```bash
python -m pip install -r requirements.txt
python run_analysis.py
python -m unittest discover -s tests -v
```

For an editable notebook, install `requirements-notebook.txt` and run `jupyter lab`.

## The experiment

The motion comes from an undamped mass-spring model: `x' = v` and `v' = -(k/m)x`. The default displacement has a 4 mm amplitude and a 4 s period. Newton's law describes the phantom's motion; radiation delivery is represented by a prescribed dose-rate profile.

Over 8 seconds, the main run records 17 positions, each with independent Gaussian noise of standard deviation 0.15 mm. The reconstruction methods are:

- Piecewise linear interpolation.
- Local cubic Newton interpolation using four nearby samples and divided differences.
- A natural cubic spline, including a check of the errors away from its endpoints.

An exact trajectory and independent adaptive quadrature provide references. The script also compares Euler and Heun integration, studies differentiation with and without sensor noise, and repeats the noisy interpolation experiment 50 times at each of four sampling intervals. Seeds and settings are recorded in [results/summary.json](results/summary.json).

## What came out of the default run

| Method | Position RMSE, mm | Dose RMSE, Gy |
| --- | ---: | ---: |
| Linear | 0.2020 | 0.025858 |
| Local Newton | 0.0958 | 0.006266 |
| Natural cubic | 0.1042 | 0.006799 |

These are errors against the synthetic reference, not clinical accuracy estimates. The position errors use 1,601 time points; the dose errors use 241 material positions from -18 to 18 mm.

![Motion reconstruction and signed errors](figures/motion_reconstruction.png)

Local Newton does best on this particular trace. With 50 noisy trials at 0.5 s spacing, the median dose RMSE is about 0.00855 Gy for local Newton and 0.00885 Gy for the spline, with overlapping interquartile ranges. That is a much more modest result than claiming one method always wins.

![Sampling intervals with and without noise](figures/sampling_and_noise.png)

One detail worth checking is the spline boundary condition. A natural spline sets its endpoint second derivatives to zero, while the oscillator has nonzero acceleration at those endpoints. The tables report both full-interval and interior position errors.

## Differentiation and separate states

Position, velocity, and acceleration have different error behavior. A position curve can look reasonable while its second derivative is very noisy. For independent position errors with standard deviation `sigma`, the central second-difference noise has standard deviation `sqrt(6)*sigma/h²`.

![Differentiation error and noise amplification](figures/differentiation.png)

The dose calculation follows fixed material points as the phantom moves. A separate signed error budget accounts for the stationary assumption, time quadrature, interpolation, measurement noise, and a 0.2 mm sensor offset. Those differences telescope; they are not combined as if they were independent random errors.

## The biology part

The first response model is the linear-quadratic surviving fraction, `S(D) = exp(-alpha*D - beta*D²)`. Two illustrative parameter pairs show how the same dose can produce different model responses. The code compares the actual change in `S` with its first-order derivative estimate.

![Dose profiles and illustrative cell survival](figures/dose_and_survival.png)

A second, independent toy model separates cells into no modeled damage, repairable damage, and irreversible damage. It is a way to practice coupled states and linear systems. Its rates are assumed, and it is not fitted to the LQ curves or a cell line.

## Arnoldi and GMRES

The three-state model produces a nonsymmetric implicit-Euler linear system. `methods.py` implements Arnoldi with modified Gram-Schmidt and uses the small least-squares problem to form GMRES iterates. The experiment checks residuals, error against a direct solve, the Arnoldi relation, and loss of orthogonality with one versus two Gram-Schmidt passes.

![Arnoldi, solver error, and time discretization error](figures/arnoldi_errors.png)

In the default run, the final relative residual is about `8.24e-13`. The error caused by taking one 60-minute time step is still about `8.84e-2` relative to the exact state evolution. Solving the linear system more accurately cannot fix that time-step error.

This matrix is small and block diagonal, so a direct solve is the practical baseline. GMRES is here to study the error analysis. Sparse matrices, spatial coupling, restarting, and preconditioning are reasonable next experiments.

## Files and next steps

| File | Purpose |
| --- | --- |
| `physics.py` | Motion, prescribed field, dose, and biological models |
| `methods.py` | Interpolation, differences, integration, and GMRES |
| `run_analysis.py` | Reproduce all nine figures and the result tables |
| `walkthrough.ipynb` | Worked examples with saved outputs |
| `model_notes.md` | Equations, units, assumptions, and error bounds |
| `tests/test_models.py` | 12 checks against analytic results and independent solvers |

The next useful changes would be an irregular motion trace, a smoothing fit for noisy data, and explicit tests of missing samples. A later version could use measured phantom data. The current periodic trajectory is a controlled benchmark, so it cannot establish how these methods perform on breathing data or a real delivery system.

The methods connect to Hunter's [numerical-methods core lectures](https://github.com/mathstat-hunter/numerical-methods/tree/main/lectures/1%20core): interpolation, least squares/GMRES, and numerical differentiation/integration. The exact course snapshot and the physics/biology references are in [model_notes.md](model_notes.md).
