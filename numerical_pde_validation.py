"""PINN vs. an independent numerical BVP solution, with no experimental data.

This is a self-consistency check, not a comparison to the photographed
strips: a classical RK4 shooting solver finds the pinned-end elastica
(dphi/ds=0 at both ends) for a chosen end distance, five interior points are
sampled from that numerical curve, and a calibrated PaperPINN is trained on
those five synthetic points plus the physics residual. The PINN curve is
then checked against the numerical curve it was never directly given.

No SciPy dependency: the project intentionally keeps numpy/torch/matplotlib
as the only requirements, so the boundary value problem is solved with a
hand-rolled RK4 shooting method (three unknowns: dimensionless Q, F and the
left-end angle; three end conditions: x(L)=d, y(L)=0, dphi/ds(L)=0) instead
of scipy.integrate.solve_bvp.

Example
-------
python3 numerical_pde_validation.py --distance-mm 190
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bending_with_distributed_loading import (
    ROOT,
    TORCH_DTYPE,
    PaperPINN,
    PaperProperties,
    circular_arc_angle,
    prediction,
    train,
)


def _pointwise_derivative(xi: np.ndarray, states: np.ndarray, q: float, f: float, gamma: float) -> np.ndarray:
    """Vectorized [dxbar, dybar, dtheta, domega]/dxi at every node, states=(N,4)."""

    theta, omega = states[:, 2], states[:, 3]
    return np.stack(
        (
            np.cos(theta),
            np.sin(theta),
            omega,
            q * np.cos(theta) - f * np.sin(theta) - gamma * (1.0 - xi) * np.cos(theta),
        ),
        axis=1,
    )


def _collocation_residual(
    z: np.ndarray, xi_nodes: np.ndarray, h: float, distance_ratio: float, gamma: float
) -> np.ndarray:
    """Trapezoidal collocation residual for the whole discretized pinned BVP.

    Unknowns z pack N+1 states [xbar, ybar, theta, omega] plus (q, f). Global
    Newton on this system avoids the shooting method's exponential
    sensitivity to the initial guess, which is what actually diverged above.
    """

    n = len(xi_nodes)
    states = z[: 4 * n].reshape(n, 4)
    q, f = z[-2], z[-1]
    rate = _pointwise_derivative(xi_nodes, states, q, f, gamma)
    interval_residual = states[1:] - states[:-1] - 0.5 * h * (rate[:-1] + rate[1:])

    residual = np.empty(4 * (n - 1) + 6)
    residual[: 4 * (n - 1)] = interval_residual.reshape(-1)
    residual[-6] = states[0, 0]
    residual[-5] = states[0, 1]
    residual[-4] = states[0, 3]
    residual[-3] = states[-1, 0] - distance_ratio
    residual[-2] = states[-1, 1]
    residual[-1] = states[-1, 3]
    return residual


def solve_pinned_bvp(
    distance_ratio: float,
    gamma: float,
    nodes: int = 200,
    tol: float = 1.0e-11,
    max_newton: int = 60,
) -> tuple[np.ndarray, float, float]:
    """Solve the pinned-end elastica BVP by Newton's method on the full mesh.

    Returns the (nodes, 4) array of [xbar, ybar, theta, omega] on a uniform
    xi grid, plus the dimensionless Q*L^2/EI and F*L^2/EI used throughout
    ``bending_with_distributed_loading.py``.
    """

    xi_nodes = np.linspace(0.0, 1.0, nodes)
    h = xi_nodes[1] - xi_nodes[0]

    theta0_guess = circular_arc_angle(distance_ratio)
    theta_guess = theta0_guess * (1.0 - 2.0 * xi_nodes)
    omega_guess = np.full(nodes, -2.0 * theta0_guess)
    xbar_guess = distance_ratio * xi_nodes
    ybar_guess = 0.15 * distance_ratio * np.sin(np.pi * xi_nodes)
    states = np.stack((xbar_guess, ybar_guess, theta_guess, omega_guess), axis=1)
    z = np.concatenate((states.reshape(-1), [gamma / 2.0, 0.0]))

    def residual(vector: np.ndarray) -> np.ndarray:
        return _collocation_residual(vector, xi_nodes, h, distance_ratio, gamma)

    size = len(z)
    for _ in range(max_newton):
        r = residual(z)
        norm_r = np.linalg.norm(r)
        if norm_r < tol:
            break
        jacobian = np.empty((size, size))
        for j in range(size):
            step = 1.0e-7 * max(1.0, abs(z[j]))
            bumped = z.copy()
            bumped[j] += step
            jacobian[:, j] = (residual(bumped) - r) / step
        delta = np.linalg.solve(jacobian, r)
        scale = 1.0
        while scale > 1.0e-4 and np.linalg.norm(residual(z - scale * delta)) >= norm_r:
            scale *= 0.5
        z = z - scale * delta
    else:
        raise RuntimeError(f"Collocation Newton solve did not converge; final residual norm {norm_r:.3e}")

    return z[: 4 * nodes].reshape(nodes, 4), float(z[-2]), float(z[-1])


def plot_comparison(
    output_path: Path,
    distance_mm: float,
    xi_numerical: np.ndarray,
    numerical_xy_m: np.ndarray,
    xi_pinn: np.ndarray,
    pinn_xy_m: np.ndarray,
    training_xy_m: np.ndarray,
    rmse_m: float,
) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    axis.plot(
        numerical_xy_m[:, 0], numerical_xy_m[:, 1], color="tab:red", lw=2, label="Numerical PDE solution", zorder=2
    )
    axis.plot(
        pinn_xy_m[:, 0], pinn_xy_m[:, 1], color="tab:blue", lw=2, ls="--", label="PINN", zorder=3
    )
    axis.scatter(
        training_xy_m[:, 0],
        training_xy_m[:, 1],
        color="black",
        s=32,
        zorder=4,
        label=f"{len(training_xy_m)} training points",
    )
    axis.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        title=f"d = {distance_mm:g} mm, pinned ends\nRMSE(PINN vs numerical) = {rmse_m * 1000:.3f} mm",
    )
    axis.axis("equal")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--distance-mm", type=float, default=190.0)
    parser.add_argument("--data-points", type=int, default=5)
    parser.add_argument("--adam-epochs", type=int, default=5000)
    parser.add_argument("--lbfgs-iterations", type=int, default=500)
    parser.add_argument("--collocation-points", type=int, default=160)
    parser.add_argument("--data-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "numerical_validation")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_default_dtype(TORCH_DTYPE)

    properties = PaperProperties()
    distance = arguments.distance_mm / 1000.0
    distance_ratio = distance / properties.length

    fine_state, q, f = solve_pinned_bvp(distance_ratio, properties.gravity_number)
    fine_xi = np.linspace(0.0, 1.0, len(fine_state))
    numerical_xy_m = fine_state[:, :2] * properties.length
    F_N = f * properties.bending_stiffness / properties.length**2
    Q_N = q * properties.bending_stiffness / properties.length**2
    print(
        f"Numerical BVP (collocation): F={F_N:.5f} N, Q={Q_N:.5f} N, "
        f"phi(0)={fine_state[0, 2]:.5f} rad, phi(L)={fine_state[-1, 2]:.5f} rad"
    )

    targets = np.linspace(0.0, 1.0, arguments.data_points + 2)[1:-1]
    training_xi = targets.reshape(-1, 1)
    training_xy_m = np.column_stack(
        (np.interp(targets, fine_xi, numerical_xy_m[:, 0]), np.interp(targets, fine_xi, numerical_xy_m[:, 1]))
    )

    torch.manual_seed(arguments.seed + int(round(arguments.distance_mm)))
    model = PaperPINN(distance_ratio, support="calibrated")
    losses = train(
        model=model,
        properties=properties,
        xi_data=training_xi,
        xy_data_m=training_xy_m,
        adam_epochs=arguments.adam_epochs,
        lbfgs_iterations=arguments.lbfgs_iterations,
        collocation_points=arguments.collocation_points,
        data_weight=arguments.data_weight,
    )
    xi_pinn, predicted = prediction(model, properties, points=len(fine_xi))
    pinn_xy_m = predicted[:, :2]

    difference = pinn_xy_m - numerical_xy_m
    rmse_m = float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))
    print(
        f"d={arguments.distance_mm:g} mm | training_points={arguments.data_points} | "
        f"physics_loss={losses['physics_loss']:.3e} | RMSE(PINN vs numerical)={rmse_m * 1000:.3f} mm"
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    label = f"{arguments.distance_mm:g}"

    csv_path = arguments.output_dir / f"comparison_{label}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("xi", "x_numerical_m", "y_numerical_m", "x_pinn_m", "y_pinn_m"))
        writer.writerows(
            zip(fine_xi, numerical_xy_m[:, 0], numerical_xy_m[:, 1], pinn_xy_m[:, 0], pinn_xy_m[:, 1])
        )

    plot_comparison(
        arguments.output_dir / f"comparison_{label}.png",
        arguments.distance_mm,
        fine_xi,
        numerical_xy_m,
        xi_pinn,
        pinn_xy_m,
        training_xy_m,
        rmse_m,
    )

    with (arguments.output_dir / f"summary_{label}.json").open("w") as stream:
        json.dump(
            {
                "distance_mm": arguments.distance_mm,
                "training_points": arguments.data_points,
                "numerical_bvp": {"F_N": F_N, "Q_N": Q_N, "phi_left_rad": float(fine_state[0, 2])},
                "pinn": {
                    "F_N": float(model.force_horizontal.detach()) * properties.bending_stiffness / properties.length**2,
                    "Q_N": float(model.force_vertical.detach()) * properties.bending_stiffness / properties.length**2,
                    **losses,
                },
                "rmse_pinn_vs_numerical_m": rmse_m,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
