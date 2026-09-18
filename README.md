# PINN for paper bending under distributed self-weight

This project rewrites `bending_with_distributed_loading.py` as an executable PyTorch PINN. It implements the large-deflection model in *Paper shape*, not a small-deflection beam approximation.

The six measured input datasets are stored as CSV files in `data/`. Generated CSV files remain under `results/` with their corresponding figures and summaries.

## Open and run with VS Code

1. Open `paper-bending-pinn.code-workspace` in VS Code and install the suggested **Python** extension if VS Code requests it.
2. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux), choose **Tasks: Run Task**, then run **Run: calibrate all data (adaptive 2/4 points)**. The task creates `.venv`, installs `requirements.txt`, and runs the validated configuration.
3. To use the debugger, select a `PINN: ...` configuration in **Run and Debug**, then press `F5`.

Use **Run: predict d=140 mm (no data)** when only the end distance is known. Change `140` in `.vscode/tasks.json` or `.vscode/launch.json` to predict another distance in mm.

Use **Run: refine d=150 mm (physics closure)** to regenerate the improved
`results/refined_150/shape_150.png` described below.

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

The loss contains the three dimensionless residuals for the ODE above. An
optional global closure term also enforces the integrated inextensibility
conditions

```text
integral cos(phi) dxi = d/L,  integral sin(phi) dxi = 0.
```

This prevents a small local kinematic residual from accumulating into a visible
endpoint drift when the learned tangent field is integrated by the
contact-free BVP projection. That projection reuses the PINN's fitted
parameters and omits contact pressure, so it is a consistency diagnostic rather
than independent ground truth. Adam brings the network into the solution basin
and L-BFGS finishes the BVP solve.

## Two support models

`--support pinned` is the nominal data-free mode. It adds the ideal-pin conditions `dphi/ds(0)=dphi/ds(L)=0`; the current implementation can run from `d` alone, but an accurate branch is not guaranteed. The program selects this mode automatically when using `--distance-mm`:

```bash
python3 bending_with_distributed_loading.py --distance-mm 140
```

The supplied photographs are not perfectly symmetric, particularly at 75 and 115 mm. This is expected when a real clamp has friction, finite contact area, or an end moment. It is also consistent with the paper, which fits unknown edge angle/curvature parameters rather than setting pin moments to zero.

`--support calibrated` therefore leaves the end conditions to be identified from the curve. It needs at least two *interior* points; the two endpoint coordinates are never used in the data loss because they are already exact boundary conditions. This is also the **default** whenever `--case` uses one of the supplied data files. The current case defaults use 4 points for `d=75/115 mm` and 2 points for the other cases. To force exactly two points in every case, pass `--data-points 2`.

```bash
python3 bending_with_distributed_loading.py --case all --output-dir results
```

For a single data set use, for example:

```bash
python3 bending_with_distributed_loading.py \
  --case 115 --support calibrated --data-points 2 --output-dir results
```

Each run writes `shape_<d>.csv` (`xi`, `x_m`, `y_m`, `phi_rad`), a comparison plot `shape_<d>.png`, and `summary.json` to the output directory.

All shape plots use the same 7.2 x 5.2 inch canvas, 170 DPI, physical axis
limits, ticks, one-to-one data aspect ratio, fonts, legend, and margins. This
makes the different end-distance cases visually comparable. To restyle plots
already stored in `results` without retraining the PINN, run the VS Code task
**Replot: standardize figures in results** or:

```bash
python3 bending_with_distributed_loading.py \
  --replot-only --output-dir results
```

To create the publication figure with all six measured curves (red), PINN
predictions (blue dashed), and the two or four calibration points, run the VS
Code task **Plot: six measured vs PINN cases** or:

```bash
python3 bending_with_distributed_loading.py \
  --plot-six-cases --output-dir results
```

This writes both `results/measured_vs_pinn_six_cases.png` and a vector-quality
`results/measured_vs_pinn_six_cases.pdf`. The panels are ordered left-to-right,
top-to-bottom as 75, 115, 150, 170, 190, and 210 mm.

## Refined physical consistency for d=150 mm

The curated `results/shape_150.csv`, `results/shape_150.png`, and the 150 mm
entry in `results/summary.json` use this refined run. The command below saves
a separate rerun under `results/refined_150/`.

The d=150 mm curve is a sensitive looped branch. With only two interior
measurements, the default data-weighted objective can fit those points while
leaving enough local ODE error for the contact-free BVP projection
to miss the right endpoint. Use stronger local-physics and global-closure terms
for this case:

```bash
python3 bending_with_distributed_loading.py \
  --case 150 \
  --data-points 2 \
  --data-weight 10 \
  --physics-weight 100 \
  --closure-weight 100 \
  --output-dir results/refined_150
```

With seed 42, 160 collocation points, 20,000 Adam steps, and 2,000 L-BFGS
iterations, the same code and training budget gave:

| d=150 configuration | Physics loss | Closure loss | RMSE to BVP projection | R² (y) |
|---|---:|---:|---:|---:|
| Default weights | 5.556e-3 | 0 (disabled) | 4.962 mm | 0.9838 |
| Physics + closure weights = 100 | 6.227e-5 | 2.142e-7 | 0.112 mm | 0.999988 |

These weights deliberately prioritize agreement with the governing ODE BVP. The
normalized two-point data loss increases from `4.104e-4` to `2.929e-3`. To
restore the original data-prioritized weighting, pass `--physics-weight 1
--closure-weight 0`; the unified script now selects the refined weights for
the supplied 150 mm case by default.

## Corrections relative to the old script

- Removed Colab-only syntax, Google Drive paths, TensorFlow/pandas dependency, undefined `loss_mse`, and attempts to load a nonexistent model.
- Replaced per-layer `F`/`Q` values and their arbitrary average with global physical reaction-force parameters.
- Preserved the elastica ODE while enforcing all four position boundary conditions exactly; the old code rescaled the predicted curve after solving, which breaks the inextensibility/ODE relation.
- Avoids fitting finite-difference angles, which amplify pixel noise. The network learns the smooth tangent angle and coordinate curve together from the governing equations.
- Reads the supplied CSV files without notebook-only dependencies, aligns each photo to its known chord, and samples sparse observations by arc length.

## Limits

The equation assumes a uniform, planar, inextensible strip and point-like supports. The PINN approximates frictionless distributed ground contact through a non-negative pressure and the Signorini condition `p*y=0`; it does not model tangential friction, plastic deformation, or nonuniform paper stiffness. Contact is resolved only at the 160 collocation points, so a large flat contact zone may require more collocation points. If you independently measure Young's modulus, pass `--youngs-modulus-pa <value>`; that is preferable to estimating `E` from sparse image points when material identification is the goal.

## File hợp nhất

`bending_with_distributed_loading.py` là file chạy duy nhất, chứa toàn bộ bộ
giải và năm biến thể thí nghiệm qua `--experiment-150mm`. File vẫn cần
`data/` và các thư viện trong `requirements.txt`. Mặc định ghi vào
`results_unified/`.

```bash
# Chạy cả sáu khoảng cách với cấu hình đúng theo results/ và kiểm tra kết quả
python3 bending_with_distributed_loading.py --case all --verify-results

# Chạy riêng một trường hợp; dùng cùng cấu hình như khi chạy --case all
python3 bending_with_distributed_loading.py --case 150 --verify-results --output-dir results_unified/only_150

# So sánh 5 biến thể ở 150 mm; mỗi biến thể có thư mục riêng
python3 bending_with_distributed_loading.py --experiment-150mm --output-dir results_unified/experiment_150mm

# Chọn riêng một cấu hình, đối chiếu với dữ liệu đo
python3 bending_with_distributed_loading.py --case 150 --sampling curvature --data-points 4 --ground off --reference measured --output-dir results_unified/curvature_free

# Cấu hình refined 150 mm, đối chiếu với phép chiếu BVP
python3 bending_with_distributed_loading.py --case 150 --data-points 2 --physics-weight 100 --closure-weight 100 --output-dir results_unified/refined_150
```

`--reference bvp` (mặc định) tính sai số với phép chiếu BVP không tiếp xúc;
`--reference measured` tính sai số với đường đo. Chế độ `--experiment-150mm`
luôn dùng đường đo, support calibrated và năm tổ hợp uniform/curvature,
2/4 điểm, ground on/off của thí nghiệm gốc. Các trọng số loss, seed và ngân
sách huấn luyện dùng chung qua CLI. Bộ chọn curvature bảo đảm đủ số điểm
nội bộ phân biệt, tránh trùng điểm hoặc chọn đầu mút.

Dùng `--replot-only --output-dir <thư_mục_một_biến_thể>` để vẽ lại từng biến
thể. Thư mục tổng của thí nghiệm chứa summary của cả năm biến thể.

Mặc định của file hợp nhất tái hiện cấu hình đã lưu trong `results/`:

| Khoảng cách (mm) | Số điểm đo | Data weight | Physics weight | Closure weight |
|---|---:|---:|---:|---:|
| 75, 115 | 4 | 10 | 1 | 0 |
| 150 | 2 | 10 | 100 | 100 |
| 170, 190, 210 | 2 | 10 | 1 | 0 |

Cả sáu dùng lấy mẫu uniform, ground on, support calibrated, seed `42 + d`,
20.000 bước Adam, 2.000 vòng L-BFGS và 160 điểm collocation. Các tham số CLI
được truyền tường minh sẽ ghi đè mặc định. Thí nghiệm `--experiment-150mm`
giữ trọng số 1/0 ban đầu; các biến thể của thí nghiệm này không phải bộ kết
quả chuẩn trong `results/`.

`--verify-results` chỉ đọc kết quả chuẩn **sau khi huấn luyện**, so sánh toàn
bộ CSV (tọa độ, góc và xi) cùng các chỉ số trong summary, ghi
`verification.json` và báo lỗi nếu sai khác vượt ngưỡng làm tròn số
(`rtol=1e-7`, `atol=1e-9` cho CSV; `atol=1e-10` cho chỉ số).
File hợp nhất không lấy đường cong đã lưu làm đầu ra dự đoán. Khi đổi phiên
bản thư viện hoặc môi trường số, cần chạy lại phép kiểm tra này trước khi
khẳng định kết quả trùng bộ chuẩn.

Đã chạy lại đầy đủ cả sáu trường hợp ngày 18/09/2026: toàn bộ giá trị CSV
và chỉ số đã lưu khớp chính xác (chênh lệch bằng 0); sáu PNG riêng và PNG
tổng hợp cũng trùng từng byte. Chạy riêng 150 mm cho cùng kết quả với
`--case all`. Báo cáo tại `results_unified/verification.json` ghi môi trường
đã kiểm chứng: Python 3.9.6, NumPy 2.0.2, PyTorch 2.8.0, SciPy 1.13.1,
Matplotlib 3.9.4, macOS arm64, CPU 4 threads.
