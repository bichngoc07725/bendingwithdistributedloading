"""Build the Vietnamese technical note for the QPINN architecture diagram.

The source citations in the PDF point to the current workspace implementation:
``quantum_pinn_bending.py`` and ``bending_with_distributed_loading.py``.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "giai_thich_qpinn_uon_giay.pdf"
FIGURE = ROOT / "docs" / "quantum_pinn_architecture_reference.png"
CODE = "quantum_pinn_paper_bending/quantum_pinn_bending.py"
CLASSICAL_CODE = "bending_with_distributed_loading.py"

FONT = "Verdana"
FONT_BOLD = "Verdana-Bold"
FONT_MONO = "Andale-Mono"


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(FONT, "/System/Library/Fonts/Supplemental/Verdana.ttf"))
    pdfmetrics.registerFont(
        TTFont(FONT_BOLD, "/System/Library/Fonts/Supplemental/Verdana Bold.ttf")
    )
    pdfmetrics.registerFont(
        TTFont(FONT_MONO, "/System/Library/Fonts/Supplemental/Andale Mono.ttf")
    )


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "VNTitle",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#173044"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "VNSubtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4d6272"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "VNH1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=23,
            textColor=colors.HexColor("#173044"),
            spaceBefore=4,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "VNH2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1d5e7f"),
            spaceBefore=4,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "VNBody",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.1,
            leading=13.1,
            textColor=colors.HexColor("#1f303e"),
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "VNSmall",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.7,
            leading=10.3,
            textColor=colors.HexColor("#284253"),
        ),
        "caption": ParagraphStyle(
            "VNCaption",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8,
            leading=10.4,
            textColor=colors.HexColor("#4a6070"),
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "VNTable",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.3,
            leading=9.2,
            textColor=colors.HexColor("#1f303e"),
        ),
        "tablehead": ParagraphStyle(
            "VNTableHead",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.5,
            leading=9.5,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(rows: list[list[str]], widths: list[float], styles: dict[str, ParagraphStyle]) -> Table:
    prepared = []
    for row_index, row in enumerate(rows):
        cell_style = styles["tablehead"] if row_index == 0 else styles["table"]
        prepared.append([P(cell, cell_style) for cell in row])
    result = Table(prepared, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d5e7f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b4c6d2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f8fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def code_box(title: str, source: str, body: str, styles: dict[str, ParagraphStyle]) -> KeepTogether:
    code_style = ParagraphStyle(
        "Code",
        fontName=FONT_MONO,
        fontSize=7.0,
        leading=8.7,
        textColor=colors.HexColor("#173044"),
        leftIndent=6,
        rightIndent=6,
        spaceBefore=2,
        spaceAfter=2,
    )
    label = P(f"<b>{title}</b> - <font color='#4d6272'>{source}</font>", styles["h2"])
    listing = Preformatted(body.strip(), code_style, maxLineLength=108)
    box = Table([[listing]], colWidths=[746])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f7fa")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9db5c5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return KeepTogether([label, box, Spacer(1, 7)])


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#b8c8d4"))
    canvas.setLineWidth(0.4)
    canvas.line(36, 28, 806, 28)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(colors.HexColor("#536b7a"))
    canvas.drawString(36, 16, "QPINN uon giay - tai lieu ky thuat, doi chieu voi ma nguon trong workspace")
    canvas.drawRightString(806, 16, f"Trang {document.page}")
    canvas.restoreState()


def build() -> Path:
    register_fonts()
    styles = make_styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUT),
        pagesize=landscape(A4),
        leftMargin=36,
        rightMargin=36,
        topMargin=31,
        bottomMargin=35,
        title="Giai thich QPINN uon giay",
        author="Paper Bending QPINN workspace",
    )

    story = []
    story.append(P("QPINN UỐN GIẤY: GIẢI THÍCH SƠ ĐỒ, KÝ HIỆU VÀ ĐỐI CHIẾU MÃ NGUỒN", styles["title"]))
    story.append(P("Tai lieu nay mo ta dung implementation hybrid Quantum Physics-Informed Neural Network trong workspace. Cac trich dan [C1]... [C6] la duong dan va dong ma nguon de kiem tra lai.", styles["subtitle"]))
    # This size leaves space for the long Vietnamese title and caption on page 1.
    image = Image(str(FIGURE), width=630, height=354.375)
    image.hAlign = "CENTER"
    story.append(image)
    story.append(Spacer(1, 4))
    story.append(P("Hinh 1. So do QPINN theo bo cuc tham chieu. Phan quantum chi thay bo xap xi ham; phuong trinh dan hoi, du lieu va loss van duoc tinh bang PyTorch co the lay dao ham tu dong.", styles["caption"]))
    story.append(PageBreak())

    story.append(P("1. Cach doc so do tu trai sang phai", styles["h1"]))
    architecture_rows = [
        ["Vung tren hinh", "Y nghia", "Doi chieu code"],
        ["Preprocessor (theta_enc)", "Nhan xi = s/L va tao bon gia tri a1...a4. Trong code, encoder la Linear(1,4) + tanh; sau do nhan pi de thanh goc quay.", f"[C2] {CODE}:109, 127"],
        ["VQC quantum layer (theta_q)", "Bon goc duoc ma hoa bang RY. Hai StronglyEntanglingLayers hoc trong so luong tu; do Pauli-Z tren bon qubit tao z1...z4.", f"[C1] {CODE}:75-95"],
        ["Classical readout (theta_read)", "z1...z4 di qua MLP 4 -> 16 -> 4, tao bốn raw outputs: r_x, r_y, r_phi, r_p.", f"[C2] {CODE}:111, 131"],
        ["Hard transform", "Chuyen raw outputs thanh x, y, phi, p. Binh phuong r_y va r_p ep y >= 0, p >= 0; factor xi(1-xi) ep y=0 tai hai dau.", f"[C2] {CODE}:132-155"],
        ["Physics + min L(theta)", "Cong residual PDE, residual hinh hoc, du lieu thua, complementarity tiep xuc va regularization. Adam cap nhat toan bo tham so trainable.", f"[C3] {CODE}:171-199; [C4]:221-268"],
    ]
    story.append(table(architecture_rows, [151, 430, 165], styles))
    story.append(Spacer(1, 12))
    story.append(P("Luu y quan trong ve hinh", styles["h2"]))
    story.append(P("O BC tren so do ghi MSE_BC = 0 de nhan manh rang dieu kien bien khong phai la mot penalty hoc duoc. No da duoc ap dat bang cong thuc x va y trong forward(). Do do physics_loss() hien tai khong co mot term MSE_BC rieng. Ky hieu phi tren so do cung chinh la bien theta trong code.", styles["body"]))
    story.append(P("Trong code hien tai, distance_ratio la thuoc tinh cua model chứ khong duoc dua vao input neuron cung xi. Cac hang vat lieu L, EI va lambda*g chi di vao physics loss thong qua PaperProperties va gravity_number.", styles["body"]))
    story.append(PageBreak())

    story.append(P("2. Giai thich tat ca ky hieu va tham so tren hinh", styles["h1"]))
    symbols_a = [
        ["Ky hieu", "Y nghia trong bai toan", "Don vi / ghi chu"],
        ["s", "Toa do do dai cung tren tam trung hoa cua to giay.", "m; chay tu 0 den L."],
        ["L", "Chieu dai giay. PaperProperties dat L = 0.297.", "m; [C5] bending_with_distributed_loading.py:72."],
        ["xi = s/L", "Toa do chuan hoa dua vao mang.", "Khong thu nguyen, 0 <= xi <= 1."],
        ["d", "Khoang cach ngang giua hai support. Tren code, distance_ratio = d/L.", "m khi tinh; CLI nhan mm. [C6]:353-385."],
        ["a1...a4", "Bon goc dau vao cua VQC sau encoder va phep nhan pi.", "radian trong RY(a_i)."],
        ["theta_enc", "Weights va biases cua Linear(1,4) + tanh.", "Tham so hoc co dien."],
        ["theta_q", "Quantum weights cua StronglyEntanglingLayers.", "2 layers, 4 wires mac dinh."],
        ["z1...z4", "Cac ky vong <Z_i> sau do qubit i.", "Moi gia tri nam trong [-1, 1] o simulator noiseless."],
        ["theta_read", "Weights/biases cua MLP 4 -> 16 -> 4.", "Tham so hoc co dien."],
        ["r_x, r_y, r_phi, r_p", "Bốn raw outputs cua readout.", "Khong phai toa do/vat ly cuoi cung."],
    ]
    story.append(table(symbols_a, [105, 432, 209], styles))
    story.append(Spacer(1, 10))
    symbols_b = [
        ["Ky hieu", "Y nghia trong bai toan", "Don vi / ghi chu"],
        ["x, y", "Toa do du doan cua duong tam giay. Trong forward la da chuan hoa theo L; predict() nhan L de tra ve met.", "m o ket qua; [C2]:134-136, [C4]:271-280."],
        ["phi (theta trong code)", "Goc tiep tuyen cua to giay theo chieu s.", "radian; theta_left/theta_right la tham so hoc."],
        ["p", "Mat do phan luc phap tuyen nen theo dang chuan hoa; p = r_p^2.", "Khong nen doc nhu ap suat SI. Ten CSV contact_pressure_bar la ten legacy, code khong quy doi sang bar."],
        ["EI", "Do cung uon: E * I, voi I = w*t^3/12.", "N m^2; [C5]:84-103."],
        ["lambda*g", "Trong luong phan bo tren mot don vi dai.", "N/m; gravity_number = lambda*g*L^3/(EI). [C5]:105-115."],
        ["F, Q", "Luc phan luc ngang va dung duoc hoc trong model (force_horizontal, force_vertical).", "Noi bo la dang chuan hoa; ket qua F_N, Q_N quy doi ve N. [C2]:117-120; [C6]:393-397."],
        ["MSE_D", "Mean-square error giua [x,y] du doan va cac diem do noi bo thua.", "Chi dung 2 hoac 4 diem theo ca d; [C3]:202-208."],
        ["MSE_PDE", "Residual moment Euler-elastica + residual hinh hoc x'=cos(phi), y'=sin(phi).", "Tinh tren cac collocation points."],
        ["MSE_C", "Complementarity cua tiep xuc: (p*y)^2.", "Ep nen chi tac dung tai cho y=0."],
        ["w_D, 10, epsilon", "Trong so data loss, trong so complementarity va regularization cua p.", "Mac dinh w_D=10; factors 10 va 1e-6 trong [C3]:193-199."],
    ]
    story.append(table(symbols_b, [105, 432, 209], styles))
    story.append(PageBreak())

    story.append(P("3. Phuong trinh, loss va y nghia vat ly", styles["h1"]))
    story.append(P("Trong bien chuan hoa xi, code khong viet truc tiep EI*phi'' theo don vi SI. No chia ty le de residual moment dung gravity_number. Dung dau cua cac luc trong residual hien tai la:", styles["body"]))
    story.append(P("phi'' - Q cos(phi) + F sin(phi) + G(1-xi) cos(phi) - contact_force cos(phi) = 0, voi G = lambda*g*L^3/(EI).", styles["h2"]))
    physics_rows = [
        ["Thanh phan loss", "Cong thuc / cach tinh trong code", "Vai tro"],
        ["Moment residual", "d2theta - Q cos(theta) + F sin(theta) + G(1-xi)cos(theta) - contact_force cos(theta)", "Can bang moment cua elastica khi co tu trong va nen."],
        ["Geometry residual", "mean[(dx - cos theta)^2] + mean[(dy - sin theta)^2]", "Buoc x,y va goc tiep tuyen phai nhat quan."],
        ["Contact force", "right_resultant(p, xi): tich phan p tu vi tri hien tai den dau phai", "Phan luc nen phan bo duoc dua vao moment residual."],
        ["Complementarity", "mean[(p*y)^2]", "Ket hop voi y >= 0 va p >= 0 da ap dat cung."],
        ["Sparse data", "mean ||[x,y]_pred - [x,y]_data||^2", "Chon dung nhanh nghiem voi it diem quan sat."],
    ]
    story.append(table(physics_rows, [133, 387, 226], styles))
    story.append(Spacer(1, 10))
    story.append(P("Vi sao y khong the am?", styles["h2"]))
    story.append(P("Voi 0 <= xi <= 1, bubble = xi(1-xi) khong am va r_y^2 khong am. Vi vay y = bubble*r_y^2 khong am moi luc. Day la rang buoc hard constraint, manh hon viec chi them penalty cho y am. Tuong tu, p = r_p^2 khong am; term (p*y)^2 buoc p bien mat khi giay tach nen.", styles["body"]))
    story.append(P("Can than khi dien giai: rang buoc nay mo hinh hoa nen phang, khong ma sat va khong cho to giay xuyen qua nen. No khong tu dong mo ta ma sat, tu contact hai chieu, plasticity hay self-contact cua to giay.", styles["body"]))
    story.append(PageBreak())

    story.append(P("4. Dan chung tu ma nguon", styles["h1"]))
    story.append(P("Cac excerpt duoi day duoc rut gon nhung giu nguyen logic va ten bien cua implementation. Duong dan/dong cho phep doi chieu truc tiep trong VS Code.", styles["body"]))
    story.append(code_box(
        "[C1] Tao VQC va phep do",
        f"{CODE}:84-95",
        "device = qml.device(\"default.qubit\", wires=qubits, shots=None)\n@qml.qnode(device, interface=\"torch\", diff_method=\"backprop\")\ndef circuit(inputs, quantum_weights):\n    qml.AngleEmbedding(inputs, wires=range(qubits), rotation=\"Y\")\n    qml.StronglyEntanglingLayers(quantum_weights, wires=range(qubits))\n    return [qml.expval(qml.PauliZ(wire)) for wire in range(qubits)]\nreturn qml.qnn.TorchLayer(circuit, {\"quantum_weights\": weight_shape})",
        styles,
    ))
    story.append(code_box(
        "[C2] Encoder, readout va hard constraints",
        f"{CODE}:109-111, 127-136, 153-155",
        "self.encoder = nn.Sequential(nn.Linear(1, qubits), nn.Tanh())\nself.quantum_layer = make_quantum_layer(qubits, circuit_layers)\nself.readout = nn.Sequential(nn.Linear(qubits, 16), nn.Tanh(), nn.Linear(16, 4))\n\nangles = np.pi * self.encoder(xi)\nquantum_features = self.quantum_layer(angles)\nvalues = self.readout(quantum_features)\nbubble = xi * (1.0 - xi)\nx = self.distance_ratio * xi + bubble * values[:, 0:1]\ny = bubble * values[:, 1:2].square()\npressure = values[:, 3:4].square()",
        styles,
    ))
    story.append(PageBreak())
    story.append(P("4.2. Dan chung physics loss va optimisation", styles["h1"]))
    story.append(P("C3 cho thay term vat ly va contact trong so do la cac phep tinh co that trong code. C4 cho thay gradient duoc lan truyen qua ca classical layers va TorchLayer cua PennyLane trong vong Adam.", styles["body"]))
    story.append(code_box(
        "[C3] Physics loss, contact va data loss",
        f"{CODE}:176-199, 202-208",
        "moment_residual = (d2theta - model.force_vertical * torch.cos(theta)\n    + model.force_horizontal * torch.sin(theta)\n    + gravity_number * (1.0 - xi) * torch.cos(theta)\n    - contact_force * torch.cos(theta))\ngeometry_residual = (dx - torch.cos(theta)).square().mean() + \\\n    (dy - torch.sin(theta)).square().mean()\ncomplementarity = (pressure * y).square().mean()\nreturn moment_residual.square().mean() + geometry_residual \\\n    + 10.0 * complementarity + 1.0e-6 * pressure.square().mean()",
        styles,
    ))
    story.append(code_box(
        "[C4] Optimisation",
        f"{CODE}:221-237, 262-268",
        "xi = torch.linspace(0.0, 1.0, collocation_points, dtype=DTYPE).reshape(-1, 1)\nxi.requires_grad_(True)\noptimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)\nfor _ in range(epochs):\n    optimizer.zero_grad()\n    total = physics_loss(model, xi, properties.gravity_number) + data_weight * data\n    total.backward()\n    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)\n    optimizer.step()",
        styles,
    ))
    story.append(P("Doi chieu nhanh: moment_residual va geometry_residual gop thanh physics loss; sparse_data_loss duoc cong rieng trong objective voi data_weight. Vi y va p da duoc hard-code khong am, complementarity khong can dung de sua dau cua y hay p ma chi hoc vi tri contact phu hop.", styles["body"]))
    story.append(PageBreak())

    story.append(P("5. Cau hinh hien tai, ket qua mau va cach dung", styles["h1"]))
    config_rows = [
        ["Tham so CLI", "Mac dinh trong code", "Dien giai"],
        ["--case", "190", "Ca thu nghiem mac dinh. Neu co data, model tu chon che do calibrated."],
        ["--qubits", "4", "So qubit cua VQC, tuong ung a1...a4 va z1...z4."],
        ["--circuit-layers", "2", "So StronglyEntanglingLayers."],
        ["--epochs", "1800", "So buoc Adam."],
        ["--collocation-points", "64", "So diem xi dung de tinh PDE loss."],
        ["--data-weight", "10", "w_D nhan vao sparse_data_loss."],
        ["--lbfgs-iterations", "0", "Optional polish sau Adam; mac dinh tat."],
        ["--seed", "42", "Seed cua PyTorch va NumPy."],
    ]
    story.append(table(config_rows, [170, 128, 448], styles))
    story.append(Spacer(1, 11))
    story.append(P("Che do sparse data", styles["h2"]))
    story.append(P("Khi chay mot case co file do, code mac dinh dung 4 diem noi bo neu d/L < 0.45 (cac vong cong chat nhu 75 va 115 mm), nguoc lai dung 2 diem. Hai diem dau mut khong duoc tinh la training data vi boundary condition da hard-code. Xem [C6] quantum_pinn_bending.py:366-378.", styles["body"]))
    story.append(P("Ket qua mau da luu cho d=190 mm", styles["h2"]))
    result_rows = [
        ["Metric", "Gia tri", "Y nghia"],
        ["So diem noi bo", "2", "Hai diem den tren figure ket qua."],
        ["RMSE", "3.77 mm", "So sanh duong QPINN va curve thuc nghiem tren 300 diem noi suy."],
        ["min y", "0.0 m", "Kiem tra rang buoc nen y >= 0."],
        ["Physics loss", "0.00195", "Gia tri da chuan hoa cua residual sau training mau."],
    ]
    story.append(table(result_rows, [160, 140, 446], styles))
    story.append(Spacer(1, 11))
    story.append(P("Lenh chay de tai lap", styles["h2"]))
    command = Preformatted(
        "cd quantum_pinn_paper_bending\n.venv/bin/python quantum_pinn_bending.py --case 190\n# Ca cong chat hon: --case 75 --data-points 4 --epochs 3000",
        ParagraphStyle("Command", fontName=FONT_MONO, fontSize=8, leading=10, textColor=colors.HexColor("#173044")),
    )
    command_box = Table([[command]], colWidths=[746])
    command_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef6fb")), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9db5c5")), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(command_box)
    story.append(Spacer(1, 11))
    story.append(P("Pham vi", styles["h2"]))
    story.append(P("Day la hybrid QPINN chay tren simulator default.qubit khong shot noise, khong phai ket qua tren quantum hardware va khong khang dinh quantum advantage. Gia tri cua no la mot baseline co the dao ham, co rang buoc vat ly va co the so sanh cong bang voi PINN co dien.", styles["body"]))

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    print(build())
