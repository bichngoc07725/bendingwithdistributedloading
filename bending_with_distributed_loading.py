"""PINN for the large-deflection shape of an A4 paper strip.

The model follows Eqs. (1)-(5) of ``Paper shape``.  The independent
coordinate is the material arc length ``s``; it is not the image x-coordinate.

For a new distance supplied with ``--distance-mm``, the default is deliberately
data-free: ideal pinned-end boundary conditions determine an equilibrium
shape.  When one of the supplied experimental cases is selected, the default
instead uses an adaptive sparse calibration set: two uniformly spaced,
*interior* measurements, or four for the two tightly folded cases where two
points cannot select the correct elastica branch.  The endpoint measurements
are never used as training data because they are hard boundary conditions.

Examples
--------
# Predict every supplied experiment using d only
python3 bending_with_distributed_loading.py --case all --data-points 0

# Calibrate a tightly folded 115 mm experiment with 4 sparse internal points
python3 bending_with_distributed_loading.py --case 115 --support calibrated --data-points 4

# Predict a new configuration from a known distance, without any image data
python3 bending_with_distributed_loading.py --distance-mm 140

Dependencies: Python 3, numpy, torch and matplotlib.  No TensorFlow, pandas,
Colab, or Google Drive is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Some managed Python installs have a read-only home folder.  Keep Matplotlib's
# cache in the system temporary directory instead of printing cache warnings.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper_pinn_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "paper_pinn_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_bvp
from torch import nn


ROOT = Path(__file__).resolve().parent
TORCH_DTYPE = torch.float64

# Publication-style geometry shared by every shape plot. Fixed physical axis
# limits make different end-distance cases directly comparable and prevent
# Matplotlib's equal-aspect adjustment from producing differently sized axes.
PLOT_FIGURE_SIZE_INCHES = (7.2, 5.2)
PLOT_DPI = 170
PLOT_X_LIMITS_M = (-0.05, 0.225)
PLOT_Y_LIMITS_M = (-0.01, 0.17)
PLOT_X_TICKS_M = np.arange(-0.05, 0.201, 0.05)
PLOT_Y_TICKS_M = np.arange(0.00, 0.161, 0.02)


@dataclass(frozen=True)
class PaperProperties:
    """SI properties from Table I of the paper.

    The table gives 70 gsm, which is *areal* mass density.  Equation (8)
    requires volume density, hence rho = 0.070 / delta [kg/m^3].  Treating
    0.070 as rho (as the old script did) is dimensionally inconsistent and
    makes EI about 11,000 times too small.
    """

    length: float = 0.297
    width: float = 0.210
    thickness: float = 0.00009
    areal_density: float = 0.070
    mass: float = 0.004
    gravity: float = 9.81
    youngs_modulus_override: Optional[float] = None

    @property
    def volumetric_density(self) -> float:
        return self.areal_density / self.thickness

    @property
    def second_moment(self) -> float:
        return self.width * self.thickness**3 / 12.0

    @property
    def youngs_modulus(self) -> float:
        if self.youngs_modulus_override is not None:
            return self.youngs_modulus_override
        # Eq. (8), with rho in kg/m^3.
        return (
            1.53113
            * self.volumetric_density
            * self.gravity
            * self.length**3
            / self.thickness**2
        )

    @property
    def bending_stiffness(self) -> float:
        return self.youngs_modulus * self.second_moment

    @property
    def distributed_weight(self) -> float:
        """lambda*g [N/m]."""

        return self.mass * self.gravity / self.length

    @property
    def gravity_number(self) -> float:
        """Dimensionless lambda*g*L^3/(EI)."""

        return self.distributed_weight * self.length**3 / self.bending_stiffness


# End distances declared by the data-file names.  We do not infer d from a
# noisy photograph because d is the input/control variable of the problem.
EXPERIMENTS = {
    75: "data/d=75mm.csv",
    115: "data/d=115mm.csv",
    150: "data/d=150mm.csv",
    170: "data/d=170mm.csv",
    190: "data/d=190mm.csv",
    210: "data/d=210mm.csv",
}


def _column_index(reference: str) -> int:
    """Convert the Excel column part of a reference (e.g. B17) to an index."""

    letters = re.match(r"[A-Z]+", reference).group(0)
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - ord("A") + 1
    return result - 1


def read_xy_xlsx(path: Path) -> np.ndarray:
    """Read X/Y columns from simple .xlsx files without pandas/openpyxl.

    All supplied spreadsheets are a single sheet with either ``Index, X, Y``
    or ``X, Y``.  Mapping values by their cell references avoids confusing the
    numeric shared-string identifiers in the header with measured coordinates.
    """

    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[int, float]] = []
    for row_number, row in enumerate(sheet.findall(f".//{namespace}row")):
        if row_number == 0:
            continue
        numeric: dict[int, float] = {}
        for cell in row.findall(f"{namespace}c"):
            value = cell.find(f"{namespace}v")
            if value is None:
                continue
            try:
                numeric[_column_index(cell.attrib["r"])] = float(value.text)
            except (KeyError, TypeError, ValueError):
                continue
        if numeric:
            rows.append(numeric)

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    # Two-column files are X,Y; three-column files are Index,X,Y.
    columns = sorted(set().union(*rows))
    x_col, y_col = columns[-2:]
    points = [(row[x_col], row[y_col]) for row in rows if x_col in row and y_col in row]
    return np.asarray(points, dtype=float)


def read_xy(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            points = [(float(row["X"]), float(row["Y"])) for row in reader]
        return np.asarray(points, dtype=float)
    if path.suffix.lower() == ".xlsx":
        return read_xy_xlsx(path)
    raise ValueError(f"Unsupported data file: {path}")


def align_to_boundary(points: np.ndarray, distance: float) -> np.ndarray:
    """Remove image translation/rotation and map its measured chord to d.

    This does not fit the curve.  It only puts photograph coordinates into the
    same coordinate system as x(0)=y(0)=0, x(L)=d, y(L)=0.
    """

    shifted = np.asarray(points, dtype=float) - points[0]
    endpoint = shifted[-1]
    chord_length = float(np.linalg.norm(endpoint))
    if chord_length <= 0.0:
        raise ValueError("The first and final measurement points coincide.")
    ex = endpoint / chord_length
    ey = np.array([-ex[1], ex[0]])
    rotated = np.column_stack((shifted @ ex, shifted @ ey))
    return rotated * (distance / chord_length)


def normalized_arclength(points: np.ndarray) -> np.ndarray:
    increments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(increments.sum())
    if total <= 0.0:
        raise ValueError("Measurements do not define a curve.")
    return np.concatenate(([0.0], np.cumsum(increments) / total))


def select_sparse_observations(
    points: np.ndarray, arclength: np.ndarray, number: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select uniformly spaced interior observations, never the two BCs."""

    if number == 0:
        return np.empty((0, 1)), np.empty((0, 2))
    if number < 0:
        raise ValueError("data_points must be non-negative")
    if number > len(points) - 2:
        raise ValueError(
            f"Requested {number} internal points, but only {len(points) - 2} are available."
        )
    targets = np.linspace(0.0, 1.0, number + 2)[1:-1]
    chosen = np.array([np.argmin(np.abs(arclength - target)) for target in targets])
    return arclength[chosen, None], points[chosen]


def circular_arc_angle(chord_ratio: float) -> float:
    """Useful low-energy initialization: sin(a)/a = d/L for a single arch."""

    if not 0.0 < chord_ratio <= 1.0:
        raise ValueError("The pinned-end distance must satisfy 0 < d <= L.")
    if chord_ratio > 0.999999:
        return 0.001
    lo, hi = 1.0e-10, math.pi - 1.0e-10
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if math.sin(mid) / mid > chord_ratio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


class PaperPINN(nn.Module):
    """One network for (x/L, y/L, phi), plus global physical F and Q.

    The old implementation assigned F and Q to every dense layer then averaged
    them.  Reaction forces are constants of a static equilibrium, so they are
    scalar trainable parameters here, not neural-network weights.
    """

    def __init__(
        self, distance_ratio: float, support: str = "pinned", width: int = 48, depth: int = 5
    ):
        super().__init__()
        layers: list[nn.Module] = []
        inputs = 1
        for _ in range(depth):
            layers.extend((nn.Linear(inputs, width), nn.Tanh()))
            inputs = width
        # x, raw_y, phi, and non-negative ground-contact pressure density.
        layers.append(nn.Linear(inputs, 4))
        self.network = nn.Sequential(*layers)
        # y = xi(1-xi) * raw_y^2 below.  A positive initial raw_y avoids the
        # near-zero-gradient state of a squared output and starts from a
        # reasonable arched configuration.
        with torch.no_grad():
            self.network[-1].bias[1] = 1.1
        initial_angle = circular_arc_angle(distance_ratio)
        self.theta_left = nn.Parameter(torch.tensor(initial_angle, dtype=TORCH_DTYPE))
        self.theta_right = nn.Parameter(torch.tensor(-initial_angle, dtype=TORCH_DTYPE))
        # Dimensionless F*L^2/EI and Q*L^2/EI.  gamma/2 is a stable Q guess.
        self.force_horizontal = nn.Parameter(torch.tensor(0.0, dtype=TORCH_DTYPE))
        self.force_vertical = nn.Parameter(torch.tensor(0.0, dtype=TORCH_DTYPE))
        self.distance_ratio = distance_ratio
        self.support = support

    def forward(
        self, xi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.network(2.0 * xi - 1.0)
        bubble = xi * (1.0 - xi)

        # Exact position boundary conditions.  Squaring raw_y is a hard
        # unilateral ground constraint: y(xi) >= 0 while y(0)=y(1)=0.  It can
        # also represent an interior touch-down (raw_y=0) without allowing
        # numerical penetration below the ground plane.
        x = self.distance_ratio * xi + bubble * output[:, 0:1]
        y = bubble * output[:, 1:2].square()

        if self.support == "pinned":
            # Cubic Hermite end-angle term and a bubble with zero derivative
            # at both ends.  Thus dphi/dxi=0 at xi=0,1 exactly: zero end
            # moments of ideal pinned supports, M=EI*dphi/ds.
            h_left = 2.0 * xi**3 - 3.0 * xi**2 + 1.0
            h_right = 1.0 - h_left
            theta_bubble = xi**2 * (1.0 - xi) ** 2
            theta = (
                self.theta_left * h_left
                + self.theta_right * h_right
                + theta_bubble * output[:, 2:3]
            )
        else:
            # This is the boundary model actually fitted in the paper: the
            # endpoint angles/curvatures are not assumed to be pin moments.
            # It requires sparse shape measurements to resolve the extra
            # freedom caused by a real clamp/frictional contact.
            theta = (
                self.theta_left * (1.0 - xi)
                + self.theta_right * xi
                + bubble * output[:, 2:3]
            )
        # Dimensionless upward contact-pressure density.  The complementarity
        # term in physics_loss enforces that it is active only at y=0.
        contact_pressure = output[:, 3:4].square()
        return x, y, theta, contact_pressure


def derivative(value: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        value,
        coordinate,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
    )[0]


def right_resultant(density: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Integral from xi to 1 using differentiable trapezoidal panels.

    ``density`` is the dimensionless upward normal pressure p_bar(xi).
    Its integral is the contact force carried by the portion of paper to the
    right of each collocation point.
    """

    intervals = xi[1:] - xi[:-1]
    panels = 0.5 * (density[:-1] + density[1:]) * intervals
    accumulated = torch.flip(torch.cumsum(torch.flip(panels, dims=(0,)), dim=0), dims=(0,))
    return torch.cat((accumulated, torch.zeros_like(density[-1:])), dim=0)


def physics_loss(model: PaperPINN, xi: torch.Tensor, gravity_number: float) -> torch.Tensor:
    """Dimensionless Eq. (5), coupled to x'=cos(phi), y'=sin(phi)."""

    x, y, theta, contact_pressure = model(xi)
    dx = derivative(x, xi)
    dy = derivative(y, xi)
    dtheta = derivative(theta, xi)
    d2theta = derivative(dtheta, xi)

    # Eq. (5) after s=L*xi and division by EI/L^2:
    # theta_xixi = q*cos(theta) - f*sin(theta) - gamma(1-xi)*cos(theta).
    resultant = right_resultant(contact_pressure, xi)
    residual_theta = (
        d2theta
        - model.force_vertical * torch.cos(theta)
        + model.force_horizontal * torch.sin(theta)
        + gravity_number * (1.0 - xi) * torch.cos(theta)
        - resultant * torch.cos(theta)
    )
    residual_x = dx - torch.cos(theta)
    residual_y = dy - torch.sin(theta)
    pde = (
        residual_theta.square().mean()
        + residual_x.square().mean()
        + residual_y.square().mean()
    )
    # Signorini complementarity: p >= 0, y >= 0 and p*y = 0.  Positivity of
    # p and y is built into the architecture; this term prevents a spurious
    # upward pressure while the sheet is above the ground.
    complementarity = (contact_pressure * y).square().mean()
    pressure_regularization = 1.0e-6 * contact_pressure.square().mean()
    return pde + 10.0 * complementarity + pressure_regularization


def geometric_closure_loss(model: PaperPINN, xi: torch.Tensor) -> torch.Tensor:
    """Enforce the global inextensibility/endpoint integrals from ``theta``.

    The coordinate outputs already satisfy both endpoint positions exactly,
    while the local residuals encourage ``x'=cos(theta)`` and
    ``y'=sin(theta)``.  A finite residual can nevertheless accumulate into a
    noticeable endpoint drift when an independent ODE solver integrates the
    learned tangent field, especially for the looped d=150 mm branch.  These
    two integral constraints explicitly remove that global inconsistency:

        integral cos(theta) dxi = d/L,  integral sin(theta) dxi = 0.

    This is a physics constraint, not additional curve supervision.
    """

    _, _, theta, _ = model(xi)
    coordinate = xi[:, 0]
    tangent_x = torch.trapezoid(torch.cos(theta[:, 0]), coordinate)
    tangent_y = torch.trapezoid(torch.sin(theta[:, 0]), coordinate)
    return (tangent_x - model.distance_ratio).square() + tangent_y.square()


def data_loss(
    model: PaperPINN,
    xi_data: Optional[torch.Tensor],
    xy_data: Optional[torch.Tensor],
) -> torch.Tensor:
    if xi_data is None or xi_data.numel() == 0:
        return torch.zeros((), dtype=TORCH_DTYPE)
    x, y, _, _ = model(xi_data)
    predicted = torch.cat((x, y), dim=1)
    return (predicted - xy_data).square().mean()


def train(
    model: PaperPINN,
    properties: PaperProperties,
    xi_data: np.ndarray,
    xy_data_m: np.ndarray,
    adam_epochs: int,
    lbfgs_iterations: int,
    collocation_points: int,
    data_weight: float,
    physics_weight: float = 1.0,
    closure_weight: float = 0.0,
) -> dict[str, float]:
    xi = torch.linspace(0.0, 1.0, collocation_points, dtype=TORCH_DTYPE).reshape(-1, 1)
    xi.requires_grad_(True)
    xi_data_t = torch.as_tensor(xi_data, dtype=TORCH_DTYPE)
    xy_data_t = torch.as_tensor(xy_data_m / properties.length, dtype=TORCH_DTYPE)

    def objective() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pde = physics_loss(model, xi, properties.gravity_number)
        closure = geometric_closure_loss(model, xi)
        observations = data_loss(model, xi_data_t, xy_data_t)
        total = physics_weight * pde + closure_weight * closure + data_weight * observations
        return total, pde, closure, observations

    adam = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    for _ in range(adam_epochs):
        adam.zero_grad()
        total, _, _, _ = objective()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        adam.step()

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=lbfgs_iterations,
        max_eval=lbfgs_iterations + 100,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def lbfgs_closure() -> torch.Tensor:
        lbfgs.zero_grad()
        total, _, _, _ = objective()
        total.backward()
        return total

    lbfgs.step(lbfgs_closure)
    total, pde, closure_constraint, observations = objective()
    return {
        "loss": float(total.detach()),
        "physics_loss": float(pde.detach()),
        "closure_loss": float(closure_constraint.detach()),
        "data_loss": float(observations.detach()),
    }


def prediction(model: PaperPINN, properties: PaperProperties, points: int = 600) -> tuple[np.ndarray, np.ndarray]:
    xi = torch.linspace(0.0, 1.0, points, dtype=TORCH_DTYPE).reshape(-1, 1)
    with torch.no_grad():
        x, y, theta, _ = model(xi)
    xy = torch.cat((x, y), dim=1).cpu().numpy() * properties.length
    return xi[:, 0].cpu().numpy(), np.column_stack((xy, theta.cpu().numpy()))


def numerical_pde_solution(
    theta_left: float,
    theta_right: float,
    force_horizontal: float,
    force_vertical: float,
    gravity_number: float,
    length: float,
    theta_guess: Optional[np.ndarray] = None,
    xi_guess: Optional[np.ndarray] = None,
    points: int = 600,
) -> np.ndarray:
    """Classical solve_bvp solution of Eq. (5), independent of the network.

    Uses the same dimensionless forces and end angles the PINN converged to
    as fixed boundary data: theta(0)=theta_left, theta(1)=theta_right. This
    checks whether the PINN's learned parameters describe an actual
    solution of the governing ODE, not whether they match a photograph.

    The ODE is nonlinear enough that fixing only the two end angles does not
    guarantee a unique curve -- distinct branches (e.g. a folded/looped
    shape vs. a simple arch) can share the same endpoint tangents.  Seeding
    solve_bvp with the PINN's own theta(xi) (``theta_guess``/``xi_guess``)
    steers the shooting method onto the same branch instead of an arbitrary
    one reachable from a naive straight-line guess.
    """

    def odes(xi: np.ndarray, state: np.ndarray) -> np.ndarray:
        theta, omega = state
        domega = (
            force_vertical * np.cos(theta)
            - force_horizontal * np.sin(theta)
            - gravity_number * (1.0 - xi) * np.cos(theta)
        )
        return np.vstack((omega, domega))

    def boundary(state_left: np.ndarray, state_right: np.ndarray) -> np.ndarray:
        return np.array([state_left[0] - theta_left, state_right[0] - theta_right])

    if theta_guess is not None and xi_guess is not None:
        mesh = np.asarray(xi_guess, dtype=float)
        theta_mesh = np.asarray(theta_guess, dtype=float)
        omega_mesh = np.gradient(theta_mesh, mesh)
    else:
        mesh = np.linspace(0.0, 1.0, 50)
        theta_mesh = theta_left + (theta_right - theta_left) * mesh
        omega_mesh = np.full_like(mesh, theta_right - theta_left)
    guess = np.vstack((theta_mesh, omega_mesh))
    solution = solve_bvp(odes, boundary, mesh, guess, tol=1.0e-8, max_nodes=20000)
    if not solution.success:
        raise RuntimeError(f"solve_bvp failed to converge: {solution.message}")

    xi_dense = np.linspace(0.0, 1.0, points)
    theta_dense = solution.sol(xi_dense)[0]
    x = np.concatenate(([0.0], np.cumsum(0.5 * (np.cos(theta_dense[:-1]) + np.cos(theta_dense[1:])) * np.diff(xi_dense))))
    y = np.concatenate(([0.0], np.cumsum(0.5 * (np.sin(theta_dense[:-1]) + np.sin(theta_dense[1:])) * np.diff(xi_dense))))
    return np.column_stack((x, y, theta_dense)) * [length, length, 1.0]


def evaluate(predicted_xi: np.ndarray, predicted: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    # Arc length must come from position alone; a 3rd (theta) column would
    # otherwise be treated as a spatial coordinate and corrupt the distances.
    measured_xi = normalized_arclength(measured[:, :2])
    measured_on_grid = np.column_stack(
        (
            np.interp(predicted_xi, measured_xi, measured[:, 0]),
            np.interp(predicted_xi, measured_xi, measured[:, 1]),
        )
    )
    difference = predicted[:, :2] - measured_on_grid
    rmse = float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))
    y_true = measured_on_grid[:, 1]
    y_residual = predicted[:, 1] - y_true
    y_denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2_y = float(1.0 - np.sum(y_residual**2) / y_denominator) if y_denominator > 0.0 else float("nan")
    return {"rmse_m": rmse, "r2_y": r2_y}


def plot_shape(
    output_path: Path,
    label: str,
    predicted: np.ndarray,
    reference: np.ndarray,
    sparse_xy: np.ndarray,
    rmse_m: Optional[float] = None,
    r2_y: Optional[float] = None,
) -> None:
    """Render one comparison with the same canvas, scale and visual style."""

    figure, axis = plt.subplots(figsize=PLOT_FIGURE_SIZE_INCHES)
    axis.plot(
        reference[:, 0],
        reference[:, 1],
        color="tab:red",
        lw=2.2,
        label="Numerical PDE solution",
        zorder=2,
    )
    axis.plot(
        predicted[:, 0],
        predicted[:, 1],
        color="tab:blue",
        lw=2.2,
        ls="--",
        label="PINN",
        zorder=3,
    )
    if len(sparse_xy):
        axis.scatter(
            sparse_xy[:, 0],
            sparse_xy[:, 1],
            color="black",
            s=28,
            zorder=4,
            label=f"{len(sparse_xy)} training points",
        )
    title = f"d = {label} mm"
    if rmse_m is not None:
        title += f", RMSE = {rmse_m * 1000:.3f} mm"
    if r2_y is not None:
        title += f", R² = {r2_y:.4f}"
    axis.set_title(title, fontsize=15, pad=10)
    axis.set_xlabel("X (m)", fontsize=12)
    axis.set_ylabel("Y (m)", fontsize=12)
    axis.set_xlim(*PLOT_X_LIMITS_M)
    axis.set_ylim(*PLOT_Y_LIMITS_M)
    axis.set_xticks(PLOT_X_TICKS_M)
    axis.set_yticks(PLOT_Y_TICKS_M)
    axis.tick_params(labelsize=10)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3, linewidth=0.8)
    axis.legend(loc="upper right", fontsize=10, framealpha=0.9)
    # Fixed margins plus fixed limits/aspect give every output the same axes
    # rectangle. Do not use bbox_inches="tight", which changes pixel size.
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.88)
    figure.savefig(output_path, dpi=PLOT_DPI, facecolor="white")
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    label: str,
    xi: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    sparse_xi: np.ndarray,
    sparse_xy: np.ndarray,
    support: str,
    rmse_m: Optional[float] = None,
    r2_y: Optional[float] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"shape_{label}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("xi", "x_m", "y_m", "phi_rad"))
        writer.writerows(zip(xi, predicted[:, 0], predicted[:, 1], predicted[:, 2]))

    plot_shape(
        output_dir / f"shape_{label}.png",
        label,
        predicted,
        reference,
        sparse_xy,
        rmse_m=rmse_m,
        r2_y=r2_y,
    )


def replot_saved_outputs(output_dir: Path) -> None:
    """Rebuild saved PNG files from CSV/summary data without retraining."""

    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text())
    length = float(summary["paper_properties"]["length"])
    bending_stiffness = float(summary["bending_stiffness_Nm2"])
    gravity_number = float(summary["gravity_number"])
    force_scale = length**2 / bending_stiffness

    for result in summary["results"]:
        distance_mm = float(result["distance_mm"])
        label = f"{distance_mm:g}"
        csv_path = output_dir / f"shape_{label}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}")
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        xi = np.asarray([float(row["xi"]) for row in rows])
        predicted = np.asarray(
            [
                [float(row["x_m"]), float(row["y_m"]), float(row["phi_rad"])]
                for row in rows
            ]
        )
        reference = numerical_pde_solution(
            float(result["phi_left_rad"]),
            float(result["phi_right_rad"]),
            float(result["F_N"]) * force_scale,
            float(result["Q_N"]) * force_scale,
            gravity_number,
            length,
            theta_guess=predicted[:, 2],
            xi_guess=xi,
            points=len(xi),
        )

        sparse_xy = np.empty((0, 2))
        number_of_points = int(result.get("training_points", 0))
        experiment_key = int(round(distance_mm))
        if number_of_points and experiment_key in EXPERIMENTS:
            measured = align_to_boundary(
                read_xy(ROOT / EXPERIMENTS[experiment_key]), distance_mm / 1000.0
            )
            measured_xi = normalized_arclength(measured)
            observations_xi, _ = select_sparse_observations(
                measured, measured_xi, number_of_points
            )
            reference_xi = np.linspace(0.0, 1.0, len(reference))
            sparse_xy = np.column_stack(
                (
                    np.interp(observations_xi[:, 0], reference_xi, reference[:, 0]),
                    np.interp(observations_xi[:, 0], reference_xi, reference[:, 1]),
                )
            )

        plot_shape(
            output_dir / f"shape_{label}.png",
            label,
            predicted,
            reference,
            sparse_xy,
            rmse_m=result.get("rmse_m"),
            r2_y=result.get("r2_y"),
        )
        print(f"Replotted {output_dir / f'shape_{label}.png'}")


def plot_six_case_comparison(output_dir: Path) -> None:
    """Create the 2x3 measured-vs-PINN figure used by the paper caption."""

    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text())
    by_distance = {
        int(round(float(result["distance_mm"]))): result
        for result in summary["results"]
    }
    distances = tuple(EXPERIMENTS)
    missing = [distance for distance in distances if distance not in by_distance]
    if missing:
        raise ValueError(f"{summary_path} is missing cases: {missing}")

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 7.4),
        sharex=True,
        sharey=True,
    )
    panel_labels = "abcdef"
    legend_handles = None

    for panel_index, (axis, distance_mm) in enumerate(zip(axes.flat, distances)):
        label = f"{distance_mm:g}"
        prediction_path = output_dir / f"shape_{label}.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing {prediction_path}")
        with prediction_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        predicted = np.asarray(
            [[float(row["x_m"]), float(row["y_m"])] for row in rows]
        )

        measured = align_to_boundary(
            read_xy(ROOT / EXPERIMENTS[distance_mm]), distance_mm / 1000.0
        )
        measured_xi = normalized_arclength(measured)
        number_of_points = int(by_distance[distance_mm]["training_points"])
        _, calibration_xy = select_sparse_observations(
            measured, measured_xi, number_of_points
        )

        measured_line = axis.plot(
            measured[:, 0],
            measured[:, 1],
            color="tab:red",
            lw=1.9,
            label="Measured",
            zorder=2,
        )[0]
        predicted_line = axis.plot(
            predicted[:, 0],
            predicted[:, 1],
            color="tab:blue",
            lw=2.1,
            ls="--",
            label="PINN",
            zorder=3,
        )[0]
        calibration_markers = axis.scatter(
            calibration_xy[:, 0],
            calibration_xy[:, 1],
            color="black",
            edgecolor="white",
            linewidth=0.4,
            s=30,
            label="Calibration points",
            zorder=4,
        )
        if legend_handles is None:
            legend_handles = (
                measured_line,
                predicted_line,
                calibration_markers,
            )

        axis.set_title(f"d = {label} mm", fontsize=13, pad=7)
        axis.text(
            0.025,
            0.955,
            f"({panel_labels[panel_index]})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlim(*PLOT_X_LIMITS_M)
        axis.set_ylim(*PLOT_Y_LIMITS_M)
        axis.set_xticks(PLOT_X_TICKS_M)
        axis.set_yticks(PLOT_Y_TICKS_M)
        axis.set_aspect("equal", adjustable="box")
        axis.tick_params(labelsize=9)
        axis.grid(alpha=0.28, linewidth=0.7)

    figure.legend(
        legend_handles,
        ("Measured", "PINN", "Calibration points"),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=3.0,
        columnspacing=2.0,
    )
    figure.supxlabel("X (m)", fontsize=13, y=0.025)
    figure.supylabel("Y (m)", fontsize=13, x=0.015)
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.105,
        top=0.88,
        wspace=0.14,
        hspace=0.27,
    )

    png_path = output_dir / "measured_vs_pinn_six_cases.png"
    pdf_path = output_dir / "measured_vs_pinn_six_cases.pdf"
    figure.savefig(png_path, dpi=PLOT_DPI, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def run_case(
    distance_mm: float,
    properties: PaperProperties,
    source: Optional[Path],
    arguments: argparse.Namespace,
) -> dict[str, float | int | str]:
    distance = distance_mm / 1000.0
    if distance > properties.length:
        raise ValueError(f"d={distance_mm:g} mm is greater than L={properties.length * 1000:g} mm.")

    data_points = arguments.data_points
    if data_points is None:
        # For d/L < 0.45 the sheet has a tight loop.  Two points are enough
        # algebraically but do not reliably distinguish its elastica branch;
        # four remains a deliberately sparse calibration set.
        data_points = 4 if source is not None and distance / properties.length < 0.45 else 2 if source is not None else 0
    support = arguments.support
    if support == "auto":
        support = "calibrated" if source is not None else "pinned"

    measured = None
    observations_xi = np.empty((0, 1))
    observations_xy = np.empty((0, 2))
    if source is not None:
        measured = align_to_boundary(read_xy(source), distance)
        arc = normalized_arclength(measured)
        observations_xi, observations_xy = select_sparse_observations(
            measured, arc, data_points
        )
    elif data_points:
        raise ValueError("Sparse data were requested, but no experimental file was supplied.")

    if support == "calibrated" and len(observations_xi) < 2:
        raise ValueError("--support calibrated needs at least 2 interior data points.")
    # A case-specific seed gives the same answer when a case is run alone or
    # as part of --case all.
    torch.manual_seed(arguments.seed + int(round(distance_mm)))
    model = PaperPINN(distance / properties.length, support=support)
    losses = train(
        model=model,
        properties=properties,
        xi_data=observations_xi,
        xy_data_m=observations_xy,
        adam_epochs=arguments.adam_epochs,
        lbfgs_iterations=arguments.lbfgs_iterations,
        collocation_points=arguments.collocation_points,
        data_weight=arguments.data_weight if len(observations_xi) else 0.0,
        physics_weight=arguments.physics_weight,
        closure_weight=arguments.closure_weight,
    )
    xi, predicted = prediction(model, properties)
    label = f"{distance_mm:g}"

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
        theta_guess=predicted[:, 2],
        xi_guess=xi,
    )

    result: dict[str, float | int | str] = {
        "distance_mm": distance_mm,
        "training_points": int(len(observations_xi)),
        "F_N": force_horizontal * properties.bending_stiffness / properties.length**2,
        "Q_N": force_vertical * properties.bending_stiffness / properties.length**2,
        "phi_left_rad": theta_left,
        "phi_right_rad": theta_right,
        "support": support,
        "data_weight": arguments.data_weight if len(observations_xi) else 0.0,
        "physics_weight": arguments.physics_weight,
        "closure_weight": arguments.closure_weight,
        "min_y_m": float(predicted[:, 1].min()),
        **losses,
    }
    result.update(evaluate(xi, predicted, reference))

    # Markers should sit on the numerical PDE solution (the ground-truth
    # curve being checked against), not at the raw photo coordinates the
    # PINN was fit to -- those hug the PINN curve almost exactly instead.
    reference_xi = np.linspace(0.0, 1.0, len(reference))
    sparse_xy_on_reference = np.column_stack(
        (
            np.interp(observations_xi[:, 0], reference_xi, reference[:, 0]),
            np.interp(observations_xi[:, 0], reference_xi, reference[:, 1]),
        )
    ) if len(observations_xi) else np.empty((0, 2))

    write_outputs(
        arguments.output_dir,
        label,
        xi,
        predicted,
        reference,
        observations_xi,
        sparse_xy_on_reference,
        support,
        rmse_m=result.get("rmse_m"),
        r2_y=result.get("r2_y"),
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("all", *map(str, EXPERIMENTS)), default="all")
    parser.add_argument("--distance-mm", type=float, help="Predict one new distance; no experimental data are used.")
    parser.add_argument(
        "--data-points",
        type=int,
        default=None,
        help="Interior measurements. Default: adaptive 2/4 for supplied experiments, 0 for --distance-mm.",
    )
    parser.add_argument("--data-weight", type=float, default=10.0, help="Weight of normalized sparse-data MSE.")
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=1.0,
        help="Weight of the local ODE/kinematic/contact residual (default: 1).",
    )
    parser.add_argument(
        "--closure-weight",
        type=float,
        default=0.0,
        help="Weight of global tangent-integral endpoint closure (default: 0).",
    )
    parser.add_argument(
        "--support",
        choices=("auto", "pinned", "calibrated"),
        default="auto",
        help="Auto: calibrated with data, pinned without data. Calibrated needs >=2 points.",
    )
    parser.add_argument("--adam-epochs", type=int, default=20000)
    parser.add_argument("--lbfgs-iterations", type=int, default=2000)
    parser.add_argument("--collocation-points", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate standardized PNGs from output-dir CSV/summary files without retraining.",
    )
    parser.add_argument(
        "--plot-six-cases",
        action="store_true",
        help="Create a 2x3 measured-vs-PINN comparison from saved results without retraining.",
    )
    parser.add_argument("--length-mm", type=float, default=297.0)
    parser.add_argument("--youngs-modulus-pa", type=float, help="Optional measured E; overrides Eq. (8).")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.replot_only and arguments.plot_six_cases:
        raise ValueError("Use either --replot-only or --plot-six-cases, not both.")
    if arguments.plot_six_cases:
        plot_six_case_comparison(arguments.output_dir)
        return
    if arguments.replot_only:
        replot_saved_outputs(arguments.output_dir)
        return
    if arguments.distance_mm is not None and arguments.case != "all":
        raise ValueError("Use either --distance-mm or --case, not both.")
    if arguments.data_weight < 0.0:
        raise ValueError("data_weight must be non-negative")
    if arguments.physics_weight < 0.0:
        raise ValueError("physics_weight must be non-negative")
    if arguments.closure_weight < 0.0:
        raise ValueError("closure_weight must be non-negative")

    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_default_dtype(TORCH_DTYPE)
    properties = PaperProperties(
        length=arguments.length_mm / 1000.0,
        youngs_modulus_override=arguments.youngs_modulus_pa,
    )

    if arguments.distance_mm is not None:
        jobs = [(arguments.distance_mm, None)]
    elif arguments.case == "all":
        jobs = [(float(distance), ROOT / filename) for distance, filename in EXPERIMENTS.items()]
    else:
        distance = int(arguments.case)
        jobs = [(float(distance), ROOT / EXPERIMENTS[distance])]

    print(
        f"E={properties.youngs_modulus:.6g} Pa, I={properties.second_moment:.6g} m^4, "
        f"EI={properties.bending_stiffness:.6g} N m^2, gamma={properties.gravity_number:.6g}"
    )
    results = []
    for distance_mm, source in jobs:
        number_of_points = arguments.data_points
        if number_of_points is None:
            number_of_points = (
                4 if source is not None and distance_mm / 1000.0 / properties.length < 0.45 else 2 if source is not None else 0
            )
        print(f"Training d={distance_mm:g} mm with {number_of_points} internal data points...")
        result = run_case(distance_mm, properties, source, arguments)
        results.append(result)
        summary = (
            f"d={distance_mm:g} mm | points={result['training_points']} | "
            f"physics={result['physics_loss']:.3e} | closure={result['closure_loss']:.3e} | "
            f"F={result['F_N']:.5f} N | Q={result['Q_N']:.5f} N"
        )
        if "rmse_m" in result:
            summary += f" | RMSE={result['rmse_m'] * 1000:.2f} mm | R2_y={result['r2_y']:.4f}"
        print(summary)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.output_dir / "summary.json").open("w") as stream:
        json.dump(
            {
                "paper_properties": asdict(properties),
                "youngs_modulus_pa": properties.youngs_modulus,
                "second_moment_m4": properties.second_moment,
                "bending_stiffness_Nm2": properties.bending_stiffness,
                "gravity_number": properties.gravity_number,
                "results": results,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
