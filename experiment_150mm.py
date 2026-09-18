"""Two ablations for the stubborn d=150 mm looped-shape fit.

1. Curvature-weighted calibration points: instead of sampling uniformly in
   arc length, concentrate points where the measured curve turns sharply
   (the tight loop), leaving fewer points on the long straight run-out.
2. A ground-constraint-free model variant: disables the unilateral y>=0
   contact constraint (and its complementarity term) added for the ground
   support, reverting to the plain elastica ODE with unconstrained y.

Both are compared against the current baseline (uniform points, ground
constraint on) for d=150 mm only.

Usage: python3 experiment_150mm.py
"""

from __future__ import annotations

import numpy as np
import torch

from bending_with_distributed_loading import (
    EXPERIMENTS,
    ROOT,
    TORCH_DTYPE,
    PaperPINN,
    PaperProperties,
    align_to_boundary,
    evaluate,
    normalized_arclength,
    prediction,
    read_xy,
    train,
    write_outputs,
)

DISTANCE_MM = 150.0
ADAM_EPOCHS = 20000
LBFGS_ITERATIONS = 2000
COLLOCATION_POINTS = 160
DATA_WEIGHT = 10.0
SEED = 42
OUTPUT_DIR = ROOT / "results" / "experiment_150mm"


class FreeYPaperPINN(PaperPINN):
    """PaperPINN without the unilateral ground constraint.

    y is left unconstrained (may go negative) and the contact-pressure DOF
    is pinned at zero, which also zeroes physics_loss's contact-resultant
    and complementarity terms -- i.e. the plain pre-ground-constraint ODE.
    """

    def forward(self, xi: torch.Tensor):
        output = self.network(2.0 * xi - 1.0)
        bubble = xi * (1.0 - xi)
        x = self.distance_ratio * xi + bubble * output[:, 0:1]
        y = bubble * output[:, 1:2]
        if self.support == "pinned":
            h_left = 2.0 * xi**3 - 3.0 * xi**2 + 1.0
            h_right = 1.0 - h_left
            theta_bubble = xi**2 * (1.0 - xi) ** 2
            theta = (
                self.theta_left * h_left
                + self.theta_right * h_right
                + theta_bubble * output[:, 2:3]
            )
        else:
            theta = (
                self.theta_left * (1.0 - xi)
                + self.theta_right * xi
                + bubble * output[:, 2:3]
            )
        contact_pressure = torch.zeros_like(output[:, 3:4])
        return x, y, theta, contact_pressure


def select_curvature_weighted_observations(
    points: np.ndarray, arclength: np.ndarray, number: int
) -> tuple[np.ndarray, np.ndarray]:
    """Interior points concentrated where the measured curve turns sharply.

    Builds a cumulative weight that is mostly the local turning-angle
    magnitude (curvature proxy) plus a small floor so straight stretches
    still get baseline coverage, then samples evenly in that weighted
    measure instead of raw arc length.
    """

    if number == 0:
        return np.empty((0, 1)), np.empty((0, 2))
    tangents = np.diff(points, axis=0)
    angles = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    turning = np.abs(np.diff(angles))
    curvature = np.concatenate(([0.0], turning, [0.0]))
    weight = curvature + 0.15 * curvature.mean() + 1.0e-9
    panel = 0.5 * (weight[:-1] + weight[1:])
    cumulative = np.concatenate(([0.0], np.cumsum(panel)))
    cumulative /= cumulative[-1]
    targets = np.linspace(0.0, 1.0, number + 2)[1:-1]
    chosen = np.array(sorted({int(np.argmin(np.abs(cumulative - t))) for t in targets}))
    return arclength[chosen, None], points[chosen]


def run_variant(
    label: str,
    model_cls: type,
    selector,
    n_points: int,
    properties: PaperProperties,
    measured: np.ndarray,
    arc: np.ndarray,
    distance: float,
) -> None:
    observations_xi, observations_xy = selector(measured, arc, n_points)
    torch.manual_seed(SEED + int(round(DISTANCE_MM)))
    model = model_cls(distance / properties.length, support="calibrated")
    losses = train(
        model=model,
        properties=properties,
        xi_data=observations_xi,
        xy_data_m=observations_xy,
        adam_epochs=ADAM_EPOCHS,
        lbfgs_iterations=LBFGS_ITERATIONS,
        collocation_points=COLLOCATION_POINTS,
        data_weight=DATA_WEIGHT,
    )
    xi, predicted = prediction(model, properties)
    metrics = evaluate(xi, predicted, measured)
    write_outputs(
        OUTPUT_DIR,
        label,
        xi,
        predicted,
        measured,
        observations_xi,
        observations_xy,
        support=f"{label}",
        rmse_m=metrics["rmse_m"],
    )
    print(
        f"{label:38s} | points={len(observations_xi)} | physics={losses['physics_loss']:.3e} "
        f"| RMSE={metrics['rmse_m'] * 1000:6.2f} mm | R2_y={metrics['r2_y']:.4f}"
    )


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(TORCH_DTYPE)
    properties = PaperProperties()
    distance = DISTANCE_MM / 1000.0
    source = ROOT / EXPERIMENTS[int(DISTANCE_MM)]
    measured = align_to_boundary(read_xy(source), distance)
    arc = normalized_arclength(measured)

    def uniform(points, arclength, number):
        targets = np.linspace(0.0, 1.0, number + 2)[1:-1]
        chosen = np.array([np.argmin(np.abs(arclength - t)) for t in targets])
        return arclength[chosen, None], points[chosen]

    print(f"d={DISTANCE_MM:g} mm variants (Adam={ADAM_EPOCHS}, L-BFGS={LBFGS_ITERATIONS}):")
    run_variant("uniform_2pt_ground_on (baseline)", PaperPINN, uniform, 2, properties, measured, arc, distance)
    run_variant("curvature_2pt_ground_on", PaperPINN, select_curvature_weighted_observations, 2, properties, measured, arc, distance)
    run_variant("curvature_4pt_ground_on", PaperPINN, select_curvature_weighted_observations, 4, properties, measured, arc, distance)
    run_variant("uniform_2pt_ground_off", FreeYPaperPINN, uniform, 2, properties, measured, arc, distance)
    run_variant("curvature_2pt_ground_off", FreeYPaperPINN, select_curvature_weighted_observations, 2, properties, measured, arc, distance)


if __name__ == "__main__":
    main()
