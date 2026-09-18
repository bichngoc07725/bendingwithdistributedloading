# Quantum-PINN for grounded paper bending

## Mục tiêu

Workspace này là phiên bản **hybrid Quantum-PINN (QPINN)** của dự án uốn giấy. Nó giữ nguyên mô hình vật lý và dữ liệu từ workspace cổ điển, nhưng thay mạng MLP thuần bằng một mạch lượng tử biến phân (VQC) nhỏ kết hợp với một lớp đọc ra cổ điển.

Sơ đồ kiến trúc có thể chỉnh sửa trong VS Code: [`docs/quantum_pinn_architecture.svg`](docs/quantum_pinn_architecture.svg).

Workspace ghim PennyLane 0.38.x để tương thích với Python 3.9 và package mirror của môi trường hiện tại; các API dùng ở đây (`TorchLayer`, `AngleEmbedding`, `StronglyEntanglingLayers`) đều có trong phiên bản này.

Đây là mô phỏng quantum trên `default.qubit` của PennyLane. Nó không tự động chứng minh lợi thế lượng tử hay nhanh hơn PINN PyTorch; với bài toán ODE 1D nhỏ này, mô phỏng thường chậm hơn. Giá trị của workspace là để nghiên cứu biểu diễn VQC, gradient quantum và so sánh công bằng với PINN cổ điển.

## Kiến trúc hybrid

```text
xi=s/L
  -> Linear + tanh encoder
  -> 4 góc lượng tử
  -> AngleEmbedding (RY trên 4 qubit)
  -> 2 StronglyEntanglingLayers (Rot + CNOT)
  -> <Z0>, <Z1>, <Z2>, <Z3>
  -> classical readout 16 nút
  -> raw x, raw y, raw phi, raw p
```

`raw_y` được bình phương trong công thức `y=xi(1-xi) raw_y²`. Vì vậy mọi nghiệm đều thỏa `y>=0`, đồng thời `y(0)=y(1)=0`. `p=raw_p²` là mật độ phản lực pháp tuyến không âm của nền.

## Physics loss

Phần tử lượng tử chỉ thay bộ xấp xỉ hàm. Physics loss vẫn là Euler elastica của bài báo:

```text
EI phi'' = Q cos(phi) - F sin(phi) - lambda g (L-s) cos(phi)
x' = cos(phi)
y' = sin(phi)
```

Khi có nền, QPINN thêm lực tiếp xúc phân bố hướng lên. Với `p>=0` là phản lực nền, nó học điều kiện Signorini:

```text
y >= 0, p >= 0, p*y = 0.
```

Vì vậy phản lực chỉ được phép xuất hiện tại nơi giấy chạm nền. Tích phân của `p` từ vị trí hiện tại đến đầu phải được đưa vào residual mô-men.

## Dữ liệu tối thiểu

Hai điểm nội bộ là mức tối thiểu cho các ca cung đơn (`d=150,170,190,210 mm`). Với vòng cong chặt `d=75,115 mm`, dùng bốn điểm nội bộ để chọn đúng nhánh elastica. Các điểm đầu mút **không** được dùng làm training data, vì chúng đã là điều kiện biên cứng.

## Chạy trong VS Code

Mở `quantum-paper-bending-pinn.code-workspace`, tin cậy workspace, rồi chọn:

```text
Tasks: Run Task -> Run QPINN: d=190 mm
```

Task tạo `.venv`, cài PennyLane/PyTorch và chạy mô phỏng. Kết quả nằm trong
`../results/quantum/` của repository chính:

- `quantum_shape_<d>.png`: so sánh hình dạng;
- `quantum_shape_<d>.csv`: xi, x, y, phi, phản lực nền;
- `quantum_pinn_<d>.pt`: trọng số PyTorch;
- `summary_<d>.json`: metric và tham số lực.

## So sánh đúng cách

So sánh QPINN và PINN cổ điển bằng cùng: hằng số vật liệu, số collocation, số điểm dữ liệu, điều kiện nền, seed và RMSE. Không nên so RMSE của QPINN với kết quả cũ cho phép `y<0`, vì nghiệm đó không khả thi vật lý.

Để chạy trên phần cứng thật cần device/plugin phù hợp, shots hữu hạn, xử lý nhiễu và tối ưu không dùng `backprop` simulator. Hãy coi phiên bản này là baseline noiseless trước khi thực hiện bước đó.
