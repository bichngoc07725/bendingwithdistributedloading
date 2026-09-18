"""Predict d=115, 170, 210 mm using only the neighbor fits at d=75, 150, 190 mm.

Each neighbor's calibrated end angles (phi_left, phi_right) and reaction
forces (F, Q) are read from ``results/summary.json`` -- the calibrated fit
that used that neighbor's own 2 interior data points. These four quantities
are interpolated in d (piecewise-linear, extrapolated for d=210 which lies
outside the [75, 190] mm training range) and then frozen as constants. No
experimental data from the target distances is used; only the network's
position/angle correction terms are trained, against the physics residual
alone, so the resulting shape solves the governing ODE subject to the
interpolated (not directly fitted) end conditions.

Usage: python3 predict_from_neighbors.py
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    physics_loss,
    prediction,
    read_xy,
)

TRAIN_DISTANCES_MM = (75.0, 150.0, 190.0)
TARGET_DISTANCES_MM = (115.0, 170.0, 210.0)
ADAM_EPOCHS = 20000
LBFGS_ITERATIONS = 2000
COLLOCATION_POINTS = 160
SEED = 42
OUTPUT_DIR = ROOT / "results" / "interpolated"


def interp_extrap(query: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Piecewise-linear interpolation; linear extrapolation past either end."""

    if query <= xs[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(ys[0] + slope * (query - xs[0]))
    if query >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(ys[-1] + slope * (query - xs[-1]))
    return float(np.interp(query, xs, ys))


def train_network_only(
    model: PaperPINN,
    properties: PaperProperties,
    adam_epochs: int,
    lbfgs_iterations: int,
    collocation_points: int,
) -> float:
    """Fit only model.network against the physics residual; BCs/forces are frozen."""

    xi = torch.linspace(0.0, 1.0, collocation_points, dtype=TORCH_DTYPE).reshape(-1, 1)
    xi.requires_grad_(True)

    def objective() -> torch.Tensor:
        return physics_loss(model, xi, properties.gravity_number)

    adam = torch.optim.Adam(model.network.parameters(), lr=1.0e-3)
    for _ in range(adam_epochs):
        adam.zero_grad()
        loss = objective()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.network.parameters(), max_norm=10.0)
        adam.step()

    lbfgs = torch.optim.LBFGS(
        model.network.parameters(),
        lr=0.5,
        max_iter=lbfgs_iterations,
        max_eval=lbfgs_iterations + 100,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    lbfgs.step(closure)
    return float(objective().detach())


def save_outputs(
    output_dir,
    label: str,
    xi: np.ndarray,
    predicted: np.ndarray,
    measured: np.ndarray,
    rmse_m: float,
    r2_y: float,
) -> None:
    """Exp (real photo data) vs PINN, for a distance the PINN never trained on."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"shape_{label}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("xi", "x_m", "y_m", "phi_rad"))
        writer.writerows(zip(xi, predicted[:, 0], predicted[:, 1], predicted[:, 2]))

    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    axis.scatter(measured[:, 0], measured[:, 1], color="tab:red", s=18, alpha=0.8, label="Exp", zorder=2)
    axis.plot(predicted[:, 0], predicted[:, 1], color="tab:blue", lw=2, ls="--", label="PINN", zorder=3)
    axis.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        title=(
            f"d = {label} mm (interpolated from neighbors)\n"
            f"RMSE = {rmse_m * 1000:.3f} mm, R² = {r2_y:.4f}"
        ),
    )
    axis.grid(alpha=0.3)
    # Headroom above the curves so the top-right legend doesn't sit on the data.
    x_values = np.concatenate((measured[:, 0], predicted[:, 0]))
    y_values = np.concatenate((measured[:, 1], predicted[:, 1]))
    x_span = float(x_values.max() - x_values.min())
    y_span = float(y_values.max() - y_values.min())
    axis.set_xlim(x_values.min() - 0.05 * x_span, x_values.max() + 0.05 * x_span)
    axis.set_ylim(y_values.min() - 0.05 * y_span, y_values.max() + 0.35 * y_span)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_dir / f"shape_{label}.png", dpi=170)
    plt.close(figure)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(TORCH_DTYPE)
    properties = PaperProperties()

    summary_path = ROOT / "results" / "summary.json"
    summary = json.loads(summary_path.read_text())
    by_distance = {r["distance_mm"]: r for r in summary["results"]}
    for d in TRAIN_DISTANCES_MM:
        if by_distance.get(d, {}).get("training_points", 0) < 2:
            raise ValueError(
                f"{summary_path} has no calibrated (>=2 point) fit for d={d} mm; "
                "run `python3 bending_with_distributed_loading.py` first."
            )

    # Convert the physical F_N, Q_N back to the dimensionless F*L^2/EI, Q*L^2/EI
    # used inside PaperPINN, matching the inverse of the conversion in run_case().
    scale = properties.length**2 / properties.bending_stiffness
    xs = np.array(TRAIN_DISTANCES_MM)
    phi_left = np.array([by_distance[d]["phi_left_rad"] for d in TRAIN_DISTANCES_MM])
    phi_right = np.array([by_distance[d]["phi_right_rad"] for d in TRAIN_DISTANCES_MM])
    f_h = np.array([by_distance[d]["F_N"] * scale for d in TRAIN_DISTANCES_MM])
    f_v = np.array([by_distance[d]["Q_N"] * scale for d in TRAIN_DISTANCES_MM])

    print(f"Interpolation basis from d = {TRAIN_DISTANCES_MM} mm (own calibrated fits):")
    for i, d in enumerate(TRAIN_DISTANCES_MM):
        print(
            f"  d={d:g} mm | phi_left={phi_left[i]:+.4f} | phi_right={phi_right[i]:+.4f} "
            f"| f_h*={f_h[i]:+.4f} | f_v*={f_v[i]:+.4f}"
        )

    results = []
    for distance_mm in TARGET_DISTANCES_MM:
        interpolated = {
            "phi_left": interp_extrap(distance_mm, xs, phi_left),
            "phi_right": interp_extrap(distance_mm, xs, phi_right),
            "f_h": interp_extrap(distance_mm, xs, f_h),
            "f_v": interp_extrap(distance_mm, xs, f_v),
        }
        distance = distance_mm / 1000.0
        torch.manual_seed(SEED + int(round(distance_mm)))
        model = PaperPINN(distance / properties.length, support="calibrated")
        with torch.no_grad():
            model.theta_left.copy_(torch.tensor(interpolated["phi_left"], dtype=TORCH_DTYPE))
            model.theta_right.copy_(torch.tensor(interpolated["phi_right"], dtype=TORCH_DTYPE))
            model.force_horizontal.copy_(torch.tensor(interpolated["f_h"], dtype=TORCH_DTYPE))
            model.force_vertical.copy_(torch.tensor(interpolated["f_v"], dtype=TORCH_DTYPE))
        model.theta_left.requires_grad_(False)
        model.theta_right.requires_grad_(False)
        model.force_horizontal.requires_grad_(False)
        model.force_vertical.requires_grad_(False)

        physics_residual = train_network_only(
            model, properties, ADAM_EPOCHS, LBFGS_ITERATIONS, COLLOCATION_POINTS
        )
        xi, predicted = prediction(model, properties)

        source = ROOT / EXPERIMENTS[int(distance_mm)]
        measured = align_to_boundary(read_xy(source), distance)
        metrics = evaluate(xi, predicted, measured)

        save_outputs(
            OUTPUT_DIR,
            f"{distance_mm:g}",
            xi,
            predicted,
            measured,
            rmse_m=metrics["rmse_m"],
            r2_y=metrics["r2_y"],
        )

        print(
            f"d={distance_mm:g} mm | physics={physics_residual:.3e} | "
            f"RMSE={metrics['rmse_m'] * 1000:.2f} mm | R2_y={metrics['r2_y']:.4f}"
        )
        results.append(
            {
                "distance_mm": distance_mm,
                "interpolated": interpolated,
                "physics_loss": physics_residual,
                **metrics,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "summary.json").open("w") as stream:
        json.dump({"train_distances_mm": TRAIN_DISTANCES_MM, "results": results}, stream, indent=2)


if __name__ == "__main__":
    main()
