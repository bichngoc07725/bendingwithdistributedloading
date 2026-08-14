# PINN for paper bending under distributed self-weight

This project rewrites `bending_with_distributed_loading.py` as an executable PyTorch PINN. It implements the large-deflection model in *Paper shape*, not a small-deflection beam approximation.

## Open and run with VS Code

1. Open `paper-bending-pinn.code-workspace` in VS Code and install the suggested **Python** extension if VS Code requests it.
2. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux), choose **Tasks: Run Task**, then run **Run: calibrate all data (adaptive 2/4 points)**. The task creates `.venv`, installs `requirements.txt`, and runs the validated configuration.
3. To use the debugger, select either `PINN: calibrate all data (adaptive 2/4 points)` or `PINN: predict d=140 mm (no data)` in **Run and Debug**, then press `F5`.

Use **Run: predict d=140 mm (no data)** when only the end distance is known. Change `140` in `.vscode/tasks.json` or `.vscode/launch.json` to predict another distance in mm.

## Physical model

With `s` as material arc length, `phi(s)` as the tangent angle and `lambda` the linear mass density, the model is

```text
EI d²phi/ds² = Q cos(phi) - F sin(phi) - lambda g (L - s) cos(phi)
dx/ds = cos(phi)
dy/ds = sin(phi)
```

The code uses SI units and the A4 dimensions from Table I: `L = 297 mm`, `w = 210 mm`, `delta = 0.09 mm`, mass `0.004 kg`. The table's `70 gsm` is an areal density, so Eq. (8) uses `rho = 0.070 / delta = 777.78 kg/m³`; this gives `E = 3.7785e10 Pa` and `EI = 4.8204e-4 N m²`.

The PINN input is the normalized arc length `xi = s/L`. A 5 x 48 `tanh` MLP outputs `x`, `y`, and `phi`. `F` and `Q` are one scalar parameter each, not weights embedded in each neural-network layer. The coordinate boundary conditions are hard constraints:

```text
x(0)=0, y(0)=0, x(L)=d, y(L)=0.
```

The experimental coordinate system also has a rigid ground at `y=0`. The architecture enforces the unilateral constraint `y(s) >= 0` exactly, so a predicted curve cannot penetrate below the ground line.

The loss contains the three dimensionless residuals for the ODE above. Adam brings the network into the solution basin and L-BFGS finishes the BVP solve.

## Two support models

`--support pinned` is the predictive, data-free model. It adds the ideal-pin conditions `dphi/ds(0)=dphi/ds(L)=0`, so knowing `d` is enough to predict a single low-energy equilibrium. The program selects this mode automatically when using `--distance-mm`:

```bash
python3 bending_with_distributed_loading.py --distance-mm 140
```

The supplied photographs are not perfectly symmetric, particularly at 75 and 115 mm. This is expected when a real clamp has friction, finite contact area, or an end moment. It is also consistent with the paper, which fits unknown edge angle/curvature parameters rather than setting pin moments to zero.

`--support calibrated` therefore leaves the end conditions to be identified from the curve. It needs at least two *interior* points; the two endpoint coordinates are never used as training data because they are already exact boundary conditions. This is also the **default** whenever `--case` uses one of the supplied data files. The adaptive default uses 4 points for `d=75/115 mm` (tight loops) and 2 points for the other cases. To force exactly two points in every case, pass `--data-points 2`.

```bash
python3 bending_with_distributed_loading.py --case all --output-dir results
```

For a single data set use, for example:

```bash
python3 bending_with_distributed_loading.py \
  --case 115 --support calibrated --data-points 2 --output-dir results
```

Each run writes `shape_<d>.csv` (`xi`, `x_m`, `y_m`, `phi_rad`), a comparison plot `shape_<d>.png`, and `summary.json` to the output directory.

## Validation using adaptive sparse internal points

The following runs used the supplied data files, the calibrated-end mode, the nonpenetrating-ground constraint `y >= 0`, 160 collocation points, 5,000 Adam steps, and 500 L-BFGS iterations. The tight loops at 75 and 115 mm need four interior points to identify their elastica branch; all other cases use two. RMSE is the Euclidean curve error after sampling both curves at the same normalized arc-length positions.

| End distance | Internal training points | RMSE | R² (y) |
|---:|---:|---:|---:|
| 75 mm | 4 | 3.27 mm | 0.9983 |
| 115 mm | 4 | 5.16 mm | 0.9844 |
| 150 mm | 2 | 6.64 mm | 0.9290 |
| 170 mm | 2 | 1.19 mm | 0.9993 |
| 190 mm | 2 | 1.95 mm | 0.9993 |
| 210 mm | 2 | 1.80 mm | 0.9978 |

For comparison, the ideal-pinned model at 170 mm used **zero** curve points and achieved RMSE `8.54 mm`, R²(y) `0.9960`; it captures the global arch but cannot reproduce the photographed asymmetry. Its unconstrained version can also penetrate the ground plane, so it is not suitable for contact configurations. Two calibration points are useful for ordinary cases and four are required for the tight loops; neither should be confused with the data-free physics prediction.

## Corrections relative to the old script

- Removed Colab-only syntax, Google Drive paths, TensorFlow/pandas dependency, undefined `loss_mse`, and attempts to load a nonexistent model.
- Replaced per-layer `F`/`Q` values and their arbitrary average with global physical reaction-force parameters.
- Preserved the PDE while enforcing all four position boundary conditions exactly; the old code rescaled the predicted curve after solving, which breaks the inextensibility/PDE relation.
- Avoids fitting finite-difference angles, which amplify pixel noise. The network learns the smooth tangent angle and coordinate curve together from the governing equations.
- Reads both provided CSV and XLSX files without notebook-only dependencies, aligns each photo to its known chord, and samples sparse observations by arc length.

## Limits

The equation assumes a uniform, planar, inextensible strip and point-like supports. The PINN approximates frictionless distributed ground contact through a non-negative pressure and the Signorini condition `p*y=0`; it does not model tangential friction, plastic deformation, or nonuniform paper stiffness. Contact is resolved only at the 160 collocation points, so a large flat contact zone may require more collocation points. If you independently measure Young's modulus, pass `--youngs-modulus-pa <value>`; that is preferable to estimating `E` from sparse image points when material identification is the goal.
