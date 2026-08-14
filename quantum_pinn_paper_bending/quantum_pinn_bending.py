"""Hybrid Quantum Physics-Informed Neural Network for grounded paper bending.

The variational quantum circuit is a trainable feature map inside a PINN.  It
is evaluated on PennyLane's noiseless ``default.qubit`` simulator by default;
that is useful for research and comparison, not evidence of quantum advantage.

Examples
--------
python quantum_pinn_bending.py --case 190
python quantum_pinn_bending.py --case 75 --data-points 4 --epochs 3000
python quantum_pinn_bending.py --distance-mm 140
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Must be set before importing matplotlib (and before importing the classical
# helper module) because managed Python installations can have a read-only home.
os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "quantum_pinn_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
CLASSICAL_ROOT = ROOT.parent
sys.path.insert(0, str(CLASSICAL_ROOT))

# Both workspaces use precisely the same paper constants and image-coordinate
# preprocessing, so a classical-vs-quantum comparison is meaningful.
from bending_with_distributed_loading import (  # noqa: E402
    EXPERIMENTS,
    PaperProperties,
    align_to_boundary,
    normalized_arclength,
    numerical_pde_solution,
    read_xy,
    select_sparse_observations,
)


DTYPE = torch.float64


def circular_arc_angle(chord_ratio: float) -> float:
    """Find a in sin(a)/a=d/L, used only to initialize a simple arch."""

    if not 0.0 < chord_ratio <= 1.0:
        raise ValueError("The distance must satisfy 0 < d <= L.")
    if chord_ratio > 0.999999:
        return 0.001
    low, high = 1.0e-10, np.pi - 1.0e-10
    for _ in range(100):
        middle = (low + high) / 2.0
        if np.sin(middle) / middle > chord_ratio:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def make_quantum_layer(qubits: int, circuit_layers: int) -> nn.Module:
    """Create a differentiable VQC feature map for a PyTorch model.

    AngleEmbedding encodes the normalized arc-length coordinate into qubit
    rotations. StronglyEntanglingLayers supplies trainable rotations and CNOT
    entanglers. The measured Pauli-Z expectations are sent to a small
    classical readout head.
    """

    device = qml.device("default.qubit", wires=qubits, shots=None)
    weight_shape = qml.StronglyEntanglingLayers.shape(
        n_layers=circuit_layers, n_wires=qubits
    )

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(inputs, quantum_weights):
        qml.AngleEmbedding(inputs, wires=range(qubits), rotation="Y")
        qml.StronglyEntanglingLayers(quantum_weights, wires=range(qubits))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(qubits)]

    return qml.qnn.TorchLayer(circuit, {"quantum_weights": weight_shape})


class HybridQuantumPINN(nn.Module):
    """QPINN with exact endpoint and nonpenetrating-ground constraints."""

    def __init__(
        self,
        distance_ratio: float,
        support: str,
        qubits: int,
        circuit_layers: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, qubits), nn.Tanh())
        self.quantum_layer = make_quantum_layer(qubits, circuit_layers)
        # Matches the classical PaperPINN's readout capacity (width 48) so the
        # quantum feature map isn't bottlenecked by an undersized classical
        # head: 4 qubits into 1 hidden layer of 16 units converges to a much
        # worse fit than the classical model even with equal training budget.
        self.readout = nn.Sequential(
            nn.Linear(qubits, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 4),
        )
        # raw_y starts positive because y=xi(1-xi)*raw_y^2.
        with torch.no_grad():
            self.readout[-1].bias[1] = 1.1

        arc_angle = circular_arc_angle(distance_ratio)
        self.theta_left = nn.Parameter(torch.tensor(arc_angle, dtype=DTYPE))
        self.theta_right = nn.Parameter(torch.tensor(-arc_angle, dtype=DTYPE))
        self.force_horizontal = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.force_vertical = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.distance_ratio = distance_ratio
        self.support = support

    def forward(
        self, xi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        angles = np.pi * self.encoder(xi)
        quantum_features = self.quantum_layer(angles)
        if quantum_features.ndim == 1:
            quantum_features = quantum_features.unsqueeze(0)
        values = self.readout(quantum_features)
        bubble = xi * (1.0 - xi)

        x = self.distance_ratio * xi + bubble * values[:, 0:1]
        # Exact unilateral contact geometry: y>=0 and y(0)=y(1)=0.
        y = bubble * values[:, 1:2].square()

        if self.support == "pinned":
            h_left = 2.0 * xi**3 - 3.0 * xi**2 + 1.0
            h_right = 1.0 - h_left
            theta = (
                self.theta_left * h_left
                + self.theta_right * h_right
                + xi**2 * (1.0 - xi) ** 2 * values[:, 2:3]
            )
        else:
            theta = (
                self.theta_left * (1.0 - xi)
                + self.theta_right * xi
                + bubble * values[:, 2:3]
            )

        # Non-negative upward contact-pressure density, with p*y=0 in loss.
        pressure = values[:, 3:4].square()
        return x, y, theta, pressure


def derivative(value: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        value, coordinate, grad_outputs=torch.ones_like(value), create_graph=True
    )[0]


def right_resultant(density: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    intervals = xi[1:] - xi[:-1]
    panels = 0.5 * (density[:-1] + density[1:]) * intervals
    integral = torch.flip(torch.cumsum(torch.flip(panels, dims=(0,)), dim=0), dims=(0,))
    return torch.cat((integral, torch.zeros_like(density[-1:])), dim=0)


def physics_loss(
    model: HybridQuantumPINN, xi: torch.Tensor, gravity_number: float
) -> torch.Tensor:
    """Euler-elastica residual plus frictionless Signorini ground contact."""

    x, y, theta, pressure = model(xi)
    dx = derivative(x, xi)
    dy = derivative(y, xi)
    dtheta = derivative(theta, xi)
    d2theta = derivative(dtheta, xi)
    contact_force = right_resultant(pressure, xi)

    moment_residual = (
        d2theta
        - model.force_vertical * torch.cos(theta)
        + model.force_horizontal * torch.sin(theta)
        + gravity_number * (1.0 - xi) * torch.cos(theta)
        - contact_force * torch.cos(theta)
    )
    geometry_residual = (dx - torch.cos(theta)).square().mean() + (
        dy - torch.sin(theta)
    ).square().mean()
    complementarity = (pressure * y).square().mean()
    return (
        moment_residual.square().mean()
        + geometry_residual
        + 10.0 * complementarity
        + 1.0e-6 * pressure.square().mean()
    )


def sparse_data_loss(
    model: HybridQuantumPINN, xi_data: torch.Tensor, xy_data: torch.Tensor
) -> torch.Tensor:
    if xi_data.numel() == 0:
        return torch.zeros((), dtype=DTYPE)
    x, y, _, _ = model(xi_data)
    return (torch.cat((x, y), dim=1) - xy_data).square().mean()


def train(
    model: HybridQuantumPINN,
    properties: PaperProperties,
    xi_data: np.ndarray,
    xy_data_m: np.ndarray,
    epochs: int,
    collocation_points: int,
    data_weight: float,
    lbfgs_iterations: int = 0,
) -> dict[str, float]:
    xi = torch.linspace(0.0, 1.0, collocation_points, dtype=DTYPE).reshape(-1, 1)
    xi.requires_grad_(True)
    xi_data_t = torch.as_tensor(xi_data, dtype=DTYPE)
    xy_data_t = torch.as_tensor(xy_data_m / properties.length, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)

    def objective() -> torch.Tensor:
        pde = physics_loss(model, xi, properties.gravity_number)
        data = sparse_data_loss(model, xi_data_t, xy_data_t)
        return pde + data_weight * data

    for _ in range(epochs):
        optimizer.zero_grad()
        total = objective()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    # Adam brings the hybrid model into the solution basin; L-BFGS then
    # polishes it, mirroring bending_with_distributed_loading.py. Quantum
    # circuit evaluations are costly, so this stays opt-in and short by
    # default (a few hundred iterations, not thousands).
    if lbfgs_iterations > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=lbfgs_iterations,
            max_eval=lbfgs_iterations + 50,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            total = objective()
            total.backward()
            return total

        lbfgs.step(closure)

    pde = physics_loss(model, xi, properties.gravity_number)
    data = sparse_data_loss(model, xi_data_t, xy_data_t)
    return {
        "loss": float((pde + data_weight * data).detach()),
        "physics_loss": float(pde.detach()),
        "data_loss": float(data.detach()),
    }


def predict(
    model: HybridQuantumPINN, properties: PaperProperties, points: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    xi = torch.linspace(0.0, 1.0, points, dtype=DTYPE).reshape(-1, 1)
    with torch.no_grad():
        x, y, theta, pressure = model(xi)
    curve = torch.column_stack((x[:, 0], y[:, 0], theta[:, 0], pressure[:, 0]))
    return xi[:, 0].numpy(), curve.numpy() * np.array(
        [properties.length, properties.length, 1.0, 1.0]
    )


def evaluate(xi: np.ndarray, prediction: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    measured_xi = normalized_arclength(measured)
    reference = np.column_stack(
        (
            np.interp(xi, measured_xi, measured[:, 0]),
            np.interp(xi, measured_xi, measured[:, 1]),
        )
    )
    difference = prediction[:, :2] - reference
    rmse = float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))
    y_true = reference[:, 1]
    y_residual = prediction[:, 1] - y_true
    y_denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2_y = float(1.0 - np.sum(y_residual**2) / y_denominator) if y_denominator > 0.0 else float("nan")
    return {"rmse_m": rmse, "r2_y": r2_y}


def write_pde_consistency_plot(
    output_dir: Path,
    label: str,
    model: HybridQuantumPINN,
    properties: PaperProperties,
    xi: np.ndarray,
    prediction: np.ndarray,
    sparse_xy: np.ndarray,
) -> dict[str, float]:
    """Self-consistency check: does the QPINN's shape solve the elastica ODE?

    Mirrors bending_with_distributed_loading.py's own diagnostic: solve_bvp
    re-derives the shape from the QPINN's learned end angles and forces,
    seeded on the QPINN's own theta(xi) to land on the same elastica branch.
    This checks physics self-consistency, not agreement with the photograph
    (see quantum_shape_{label}.png / evaluate() for that).
    """

    theta_left = float(model.theta_left.detach())
    theta_right = float(model.theta_right.detach())
    force_horizontal = float(model.force_horizontal.detach())
    force_vertical = float(model.force_vertical.detach())
    reference = numerical_pde_solution(
        theta_left,
        theta_right,
        force_horizontal,
        force_vertical,
        properties.gravity_number,
        properties.length,
        theta_guess=prediction[:, 2],
        xi_guess=xi,
    )
    metrics = evaluate(xi, prediction, reference[:, :2])

    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    axis.plot(
        reference[:, 0], reference[:, 1], color="tab:red", lw=2, label="Numerical PDE solution", zorder=2
    )
    axis.plot(prediction[:, 0], prediction[:, 1], color="tab:blue", lw=2, ls="--", label="PINN", zorder=3)
    if len(sparse_xy):
        axis.scatter(
            sparse_xy[:, 0], sparse_xy[:, 1], color="black", s=28, zorder=4,
            label=f"{len(sparse_xy)} training points",
        )
    title = f"d = {label} mm, RMSE = {metrics['rmse_m'] * 1000:.3f} mm, R² = {metrics['r2_y']:.4f}"
    axis.set(xlabel="X (m)", ylabel="Y (m)", title=title)
    axis.grid(alpha=0.3)
    x_values = np.concatenate((reference[:, 0], prediction[:, 0]))
    y_values = np.concatenate((reference[:, 1], prediction[:, 1]))
    x_span = float(x_values.max() - x_values.min())
    y_span = float(y_values.max() - y_values.min())
    axis.set_xlim(x_values.min() - 0.05 * x_span, x_values.max() + 0.05 * x_span)
    axis.set_ylim(y_values.min() - 0.05 * y_span, y_values.max() + 0.35 * y_span)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_dir / f"quantum_pde_check_{label}.png", dpi=170)
    plt.close(figure)
    return {"pde_rmse_m": metrics["rmse_m"], "pde_r2_y": metrics["r2_y"]}


def write_outputs(
    output_dir: Path,
    label: str,
    xi: np.ndarray,
    prediction: np.ndarray,
    measured: Optional[np.ndarray],
    sparse_xy: np.ndarray,
    result: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"quantum_shape_{label}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("xi", "x_m", "y_m", "phi_rad", "contact_pressure_bar"))
        writer.writerows(zip(xi, *prediction.T))

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    if measured is not None:
        axis.plot(measured[:, 0], measured[:, 1], color="tab:red", lw=2, label="experiment", zorder=2)
    axis.plot(
        prediction[:, 0], prediction[:, 1], color="tab:blue", lw=2, ls="--", label="hybrid QPINN", zorder=3
    )
    if len(sparse_xy):
        axis.scatter(
            sparse_xy[:, 0], sparse_xy[:, 1], color="black", s=28, zorder=4, label="sparse data"
        )
    axis.axhline(0.0, color="black", alpha=0.45, lw=0.8)
    if "rmse_m" in result and "r2_y" in result:
        metric = f", RMSE = {result['rmse_m'] * 1000:.3f} mm, R² = {result['r2_y']:.4f}"
    else:
        metric = ", data-free"
    axis.set(title=f"d = {label} mm{metric}", xlabel="x (m)", ylabel="y (m)")
    axis.axis("equal")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"quantum_shape_{label}.png", dpi=170)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(map(str, EXPERIMENTS)), default="190")
    parser.add_argument("--distance-mm", type=float, help="New distance; overrides --case and uses no image data.")
    parser.add_argument("--data-points", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lbfgs-iterations", type=int, default=0, help="Optional L-BFGS polish after Adam.")
    parser.add_argument("--collocation-points", type=int, default=64)
    parser.add_argument("--data-weight", type=float, default=10.0)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--circuit-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.qubits < 2:
        raise ValueError("Use at least 2 qubits so the circuit contains an entangling gate.")
    torch.set_default_dtype(DTYPE)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    properties = PaperProperties()

    if arguments.distance_mm is None:
        distance_mm = float(arguments.case)
        source = CLASSICAL_ROOT / EXPERIMENTS[int(arguments.case)]
    else:
        distance_mm = arguments.distance_mm
        source = None
    distance = distance_mm / 1000.0
    if not 0.0 < distance <= properties.length:
        raise ValueError("distance-mm must be in (0, 297].")

    measured = None
    observations_xi = np.empty((0, 1))
    observations_xy = np.empty((0, 2))
    if source is not None:
        measured = align_to_boundary(read_xy(source), distance)
        number = arguments.data_points
        if number is None:
            number = 4 if distance / properties.length < 0.45 else 2
        observations_xi, observations_xy = select_sparse_observations(
            measured, normalized_arclength(measured), number
        )
        support = "calibrated"
    else:
        if arguments.data_points not in (None, 0):
            raise ValueError("A new distance has no experimental curve for sparse data.")
        support = "pinned"

    model = HybridQuantumPINN(
        distance_ratio=distance / properties.length,
        support=support,
        qubits=arguments.qubits,
        circuit_layers=arguments.circuit_layers,
    )
    losses = train(
        model, properties, observations_xi, observations_xy,
        arguments.epochs, arguments.collocation_points,
        arguments.data_weight if len(observations_xi) else 0.0,
        arguments.lbfgs_iterations,
    )
    xi, prediction = predict(model, properties)
    result: dict[str, float] = {
        **losses,
        "F_N": float(model.force_horizontal.detach()) * properties.bending_stiffness / properties.length**2,
        "Q_N": float(model.force_vertical.detach()) * properties.bending_stiffness / properties.length**2,
    }
    if measured is not None:
        result.update(evaluate(xi, prediction, measured))
    else:
        result["min_y_m"] = float(prediction[:, 1].min())
    label = f"{distance_mm:g}"
    write_outputs(arguments.output_dir, label, xi, prediction, measured, observations_xy, result)
    result.update(
        write_pde_consistency_plot(
            arguments.output_dir, label, model, properties, xi, prediction, observations_xy
        )
    )
    torch.save(model.state_dict(), arguments.output_dir / f"quantum_pinn_{label}.pt")
    with (arguments.output_dir / f"summary_{label}.json").open("w") as stream:
        json.dump({"configuration": vars(arguments), "support": support, "result": result}, stream, indent=2, default=str)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
