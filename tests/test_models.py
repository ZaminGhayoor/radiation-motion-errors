"""Analytic checks and independent references for the numerical experiments."""

import unittest
import numpy as np
from scipy.sparse.linalg import gmres
from methods import divided_differences, newton_value, reconstruct, central_differences, trapezoid, gmres_steps, rmse
from physics import CONFIG, motion, oscillator, dose_rate, accumulated_dose, reference_dose, survival, repair_system, repair_exact


class MethodTests(unittest.TestCase):

    def test_newton_reproduces_cubic(self):
        t = np.array([-1.0, 0.0, 0.6, 1.5, 2.0])
        y = t ** 3 - 2 * t + 1
        q = np.linspace(-1, 2, 91)
        c = divided_differences(t, y)
        np.testing.assert_allclose(newton_value(t, c, q), q ** 3 - 2 * q + 1, atol=1e-13)
        np.testing.assert_allclose(reconstruct(t, y, q, "local newton"), q ** 3 - 2 * q + 1, atol=1e-13)

    def test_interpolation_hits_nodes_and_straight_line(self):
        t = np.linspace(0, 4, 9)
        q = np.linspace(0, 4, 61)
        for method in ["linear", "local newton", "cubic"]:
            np.testing.assert_allclose(reconstruct(t, np.sin(t), t, method), np.sin(t), atol=1e-13)
            np.testing.assert_allclose(reconstruct(t, 2 * t + 1, q, method), 2 * q + 1, atol=1e-13)

    def test_invalid_nodes_and_extrapolation(self):
        for t, y in [([], []), ([1, 1], [2, 3]), ([0, np.nan], [2, 3])]:
            with self.assertRaises(ValueError):
                divided_differences(t, y)
        with self.assertRaises(ValueError):
            reconstruct([0, 1], [0, 2], [-0.1], "linear")
        with self.assertRaises(ValueError):
            central_differences([0, 1, 3], [1, 2, 3])

    def test_derivatives_of_quadratic(self):
        t = np.linspace(0, 4, 41)
        q, v, a = central_differences(t, t ** 2 + 3 * t)
        np.testing.assert_allclose(v, 2 * q + 3, atol=1e-12)
        np.testing.assert_allclose(a, np.full(len(q), 2), atol=1e-11)

    def test_trapezoid_exact_for_affine_function(self):
        t = np.array([0, 0.1, 0.6, 1.0])
        self.assertAlmostEqual(float(trapezoid(t, 2 * t + 3)), 4.0, places=14)

    def test_gmres_against_direct_and_scipy(self):
        a = np.diag(np.linspace(1, 3, 12)) + np.diag(np.full(11, 0.4), 1)
        b = np.linspace(0.2, 2, 12)
        x, rows = gmres_steps(a, b, max_steps=12)
        ref = np.linalg.solve(a, b)
        other, status = gmres(a, b, rtol=1e-12, atol=0, restart=12)
        self.assertEqual(status, 0)
        np.testing.assert_allclose(x, ref, atol=1e-11)
        np.testing.assert_allclose(x, other, atol=1e-11)
        for row in rows:
            err = np.linalg.norm(row["x"] - ref) / np.linalg.norm(ref)
            self.assertLessEqual(err, np.linalg.cond(a) * row["relative_residual"] + 1e-14)
            self.assertAlmostEqual(row["relative_residual"], row["projected_residual"], places=12)
            self.assertLess(row["arnoldi_relation_error"], 1e-13)

    def test_gmres_zero_rhs_and_happy_breakdown(self):
        a = 2 * np.eye(4)
        x, rows = gmres_steps(a, np.zeros(4))
        np.testing.assert_array_equal(x, np.zeros(4))
        self.assertEqual(rows, [])
        x, rows = gmres_steps(a, np.ones(4))
        np.testing.assert_allclose(x, np.full(4, 0.5))
        self.assertEqual(len(rows), 1)


class PhysicsTests(unittest.TestCase):

    def test_newtons_law_and_energy_units(self):
        t = np.linspace(0, 8, 101)
        x, v, a = motion(t)
        mass = CONFIG["mass_kg"]
        spring = mass * (2 * np.pi / CONFIG["period_s"]) ** 2
        # Convert mm to m before calculating force (N) and energy (J).
        np.testing.assert_allclose(mass * a / 1000, -spring * x / 1000, atol=1e-15)
        energy = 0.5 * mass * (v / 1000) ** 2 + 0.5 * spring * (x / 1000) ** 2
        np.testing.assert_allclose(energy, np.full(len(t), energy[0]), atol=1e-18)

    def test_euler_and_heun_convergence(self):
        for method, expected in [("euler", 1), ("heun", 2)]:
            errors = []
            for n in [800, 1600]:
                t = np.linspace(0, 8, n + 1)
                errors.append(rmse(oscillator(t, method)[:, 0], motion(t)[0]))
            order = np.log2(errors[0] / errors[1])
            self.assertAlmostEqual(order, expected, delta=0.08)

    def test_stationary_dose_and_moving_quadrature(self):
        t = np.linspace(0, 8, 4001)
        points = np.array([-6.0, 0.0, 6.0])
        np.testing.assert_allclose(accumulated_dose(t, np.zeros(len(t)), points), 8 * dose_rate(points), atol=1e-13)
        ref, _ = reference_dose(points)
        dense = accumulated_dose(t, motion(t)[0], points)
        np.testing.assert_allclose(dense, ref, atol=1e-9)
        self.assertAlmostEqual(ref[0], ref[-1], places=12)
        self.assertTrue(np.all(ref >= 0))

    def test_survival_and_its_derivative(self):
        self.assertEqual(float(survival(0)), 1.0)
        d = np.linspace(0, 6, 101)
        self.assertTrue(np.all(np.diff(survival(d)) < 0))
        h = 1e-5
        numerical = (survival(2 + h) - survival(2 - h)) / (2 * h)
        exact = -(0.30 + 2 * 0.03 * 2) * survival(2)
        self.assertAlmostEqual(float(numerical), float(exact), places=10)

    def test_repair_conserves_population_and_separates_time_error(self):
        a, initial, k, rates, death = repair_system(np.array([0, 1, 2, 3]))
        np.testing.assert_allclose(k.sum(axis=0), 0, atol=1e-15)
        exact = repair_exact(initial, rates, death, 60)
        one_step = np.linalg.solve(a, initial)
        small_a = np.eye(len(initial)) - k
        many_steps = initial.copy()
        for i in range(60):
            many_steps = np.linalg.solve(small_a, many_steps)
        for y in [exact, one_step, many_steps]:
            self.assertTrue(np.min(y) >= -1e-14)
            np.testing.assert_allclose(y.reshape(-1, 3).sum(axis=1), 1, atol=1e-13)
        self.assertLess(np.linalg.norm(many_steps - exact), np.linalg.norm(one_step - exact) / 10)


if __name__ == "__main__":
    unittest.main()
