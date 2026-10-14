# ==============================================================================
# TẬP TIN: src/step_05_assumption_tests.py
# MỤC ĐÍCH: Kiểm định 4 Giả định Thống kê Bắt buộc của Mô hình Hồi quy Tuyến tính OLS (KDD Bước 6)
# QUY CHUẨN: Giáo trình Phân tích Dữ liệu DA - SGU (Đại học Sài Gòn)
# 4 GIẢ ĐỊNH: (1) Tính tuyến tính -> (2) Phân phối chuẩn sai số -> (3) Không đa cộng tuyến (VIF) -> (4) Phương sai sai số không đổi
# GIẢI THÍCH: Chú thích từng dòng lệnh bằng tiếng Việt dành cho Newbie Data Analyst
# ==============================================================================

# Import thư viện sys để cấu hình hiển thị mã hóa tiếng Việt UTF-8 trên Windows console
import sys

# Import thư viện os để quản lý đường dẫn tập tin và kiểm tra dung lượng thư mục
import os

# Import thư viện pandas để thao tác với bảng dữ liệu DataFrame và trích xuất chỉ số
import pandas as pd

# Import thư viện numpy để tính toán ma trận và các giá trị số học
import numpy as np

# Import statsmodels.api để ước lượng mô hình OLS và trích xuất phần dư, giá trị dự báo
import statsmodels.api as sm

# Import hàm kiểm định Shapiro-Wilk từ scipy.stats để kiểm tra phân phối chuẩn của phần dư
from scipy.stats import shapiro, probplot

# Import hàm tính hệ số phóng đại phương sai VIF từ statsmodels.stats.outliers_influence
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import hàm kiểm định Breusch-Pagan và White Test để kiểm tra phương sai sai số thay đổi
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Import matplotlib.pyplot và seaborn để kết xuất 4 biểu đồ kiểm định chất lượng cao (300 DPI)
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập mã hóa UTF-8 cho luồng xuất console tránh lỗi ký tự tiếng Việt trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình font chữ hỗ trợ tiếng Việt có dấu và hiển thị chuẩn dấu trừ trên Matplotlib
plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def kiem_dinh_gia_dinh_1_linearity(model, y_train, fig_path):
    """
    Task 5.1 - GIẢ ĐỊNH 1: Tính Tuyến tính (Linearity)
    - Kiểm tra qua kiểm định F-statistic: H0: beta_1 = beta_2 = beta_3 = 0.
    - Vẽ biểu đồ phân tán Actual vs Fitted values kèm đường tham chiếu 45 độ.
    """
    print("=" * 85)
    print("TASK 5.1 — GIẢ ĐỊNH 1: MỐI QUAN HỆ TUYẾN TÍNH (LINEARITY)")
    print("=" * 85)

    f_stat = model.fvalue
    f_pvalue = model.f_pvalue

    print(f"- Thống kê F-statistic: {f_stat:.4f}")
    print(f"- Giá trị p-value của kiểm định F: {f_pvalue:.4e}")

    if f_pvalue < 0.05:
        ket_luan = "ĐẠT GIẢ ĐỊNH (Bác bỏ H0, mối quan hệ tuyến tính có ý nghĩa thống kê áp đảo p < 0.05)"
    else:
        ket_luan = "VI PHẠM (Chấp nhận H0, không có mối quan hệ tuyến tính)"
    print(f"-> KẾT LUẬN: {ket_luan}")

    # Vẽ biểu đồ Actual vs Fitted
    fitted_vals = model.fittedvalues
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Điểm phân tán giữa giá trị thực tế và giá trị dự báo
    ax.scatter(y_train / 1000.0, fitted_vals / 1000.0, color="#1E88E5", alpha=0.7, edgecolors="black", s=50, label="Quan sát thực tế")

    # Đường chéo 45 độ (Đường dự báo lý tưởng Actual = Fitted)
    min_val = min(y_train.min(), fitted_vals.min()) / 1000.0
    max_val = max(y_train.max(), fitted_vals.max()) / 1000.0
    ax.plot([min_val, max_val], [min_val, max_val], color="#D32F2F", linestyle="--", linewidth=2.5, label="Đường chuẩn lý tưởng (y = x)")

    ax.set_title("GIẢ ĐỊNH 1: BIỂU ĐỒ GIÁ TRỊ THỰC TẾ (ACTUAL) VS DỰ BÁO (FITTED)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Giá Cà phê Thực tế (Nghìn VNĐ/kg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Giá Cà phê Dự báo OLS (Nghìn VNĐ/kg)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    # Thêm hộp chú thích chỉ số R2 và F-stat lên đồ thị
    textstr = f"$R^2 = {model.rsquared:.4f}$\n$F = {f_stat:.2f}$ ($p < 0.001$)"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.72, 0.15, textstr, transform=ax.transAxes, fontsize=11, verticalalignment="top", bbox=props)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ 1 (fig05): {fig_path}")

    return {
        "Gia_dinh": "1. Tính Tuyến tính (Linearity)",
        "Kiem_dinh": "F-statistic Test",
        "Gia_tri_thong_ke": round(f_stat, 2),
        "p_value": f"{f_pvalue:.4e}",
        "Ket_qua": "ĐẠT (PASS)"
    }

def kiem_dinh_gia_dinh_2_normality(model, fig_path):
    """
    Task 5.2 — GIẢ ĐỊNH 2: Phân phối Chuẩn của Phần dư (Residual Normality)
    - Tính trung bình phần dư (e_bar).
    - Kiểm định Shapiro-Wilk Test (H0: Phần dư tuân theo phân phối chuẩn).
    - Vẽ 2 đồ thị: Histogram kèm KDE và Q-Q Plot.
    """
    print("\n" + "=" * 85)
    print("TASK 5.2 — GIẢ ĐỊNH 2: PHÂN PHỐI CHUẨN CỦA PHẦN DƯ (RESIDUAL NORMALITY)")
    print("=" * 85)

    residuals = model.resid
    mean_resid = residuals.mean()
    print(f"- Trung bình phần dư: {mean_resid:+.8f} (~0)")

    # Kiểm định Shapiro-Wilk
    w_stat, p_val_shapiro = shapiro(residuals)
    print(f"- Kiểm định Shapiro-Wilk: Thống kê W = {w_stat:.4f} | p-value = {p_val_shapiro:.4f}")

    if p_val_shapiro >= 0.05:
        ket_luan = "ĐẠT GIẢ ĐỊNH (Chấp nhận H0: Phần dư tuân theo phân phối chuẩn vì p-value >= 0.05)"
        ket_qua_danh_gia = "ĐẠT (PASS)"
    else:
        ket_luan = "VI PHẠM (Bác bỏ H0: Phần dư không chuẩn, cần thử chuẩn hóa hoặc Logarit)"
        ket_qua_danh_gia = "VI PHẠM (FAIL)"
    print(f"-> KẾT LUẬN: {ket_luan}")

    # Vẽ biểu đồ Histogram + Q-Q Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Histogram kèm đường KDE
    sns.histplot(residuals / 1000.0, kde=True, color="#43A047", ax=ax1, bins=15, stat="density", edgecolor="black", alpha=0.6)
    ax1.set_title("Biểu đồ Phân phối Phần dư (Histogram & KDE)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Phần dư Residuals (Nghìn VNĐ/kg)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mật độ xác suất (Density)", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Q-Q Plot (Quantile-Quantile Plot)
    probplot(residuals / 1000.0, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor("#1E88E5")
    ax2.get_lines()[0].set_markeredgecolor("black")
    ax2.get_lines()[0].set_markersize(6)
    ax2.get_lines()[1].set_color("#D32F2F")
    ax2.get_lines()[1].set_linewidth(2)
    ax2.set_title("Biểu đồ Q-Q Plot Kiểm tra Phân phối Chuẩn", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Phân vị Chuẩn Lý thuyết (Theoretical Quantiles)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Phân vị Phần dư Mẫu (Sample Quantiles)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"GIẢ ĐỊNH 2: KIỂM ĐỊNH PHÂN PHỐI CHUẨN CỦA PHẦN DƯ (Shapiro-Wilk W = {w_stat:.4f}, p = {p_val_shapiro:.4f})",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ 2 (fig06): {fig_path}")

    return {
        "Gia_dinh": "2. Phân phối Chuẩn của Phần dư (Normality)",
        "Kiem_dinh": "Shapiro-Wilk Test",
        "Gia_tri_thong_ke": round(w_stat, 4),
        "p_value": f"{p_val_shapiro:.4f}",
        "Ket_qua": ket_qua_danh_gia
    }

def kiem_dinh_gia_dinh_3_multicollinearity(df_train, features, fig_path):
    """
    Task 5.3 — GIẢ ĐỊNH 3: Không có Đa cộng tuyến nghiêm trọng (No Multicollinearity)
    - Tính hệ số VIF (Variance Inflation Factor) và Độ chấp nhận Tolerance (1/VIF).
    - Ngưỡng DA-SGU: VIF < 10 (và Tolerance > 0.1) là an toàn.
    - Vẽ biểu đồ cột ngang kèm vạch đỏ cảnh báo VIF = 10.
    """
    print("\n" + "=" * 85)
    print("TASK 5.3 — GIẢ ĐỊNH 3: KIỂM ĐỊNH ĐA CỘNG TUYẾN (MULTICOLLINEARITY & VIF)")
    print("=" * 85)

    X_vif = df_train[features]
    vif_data = []

    for i, col in enumerate(features):
        vif_val = variance_inflation_factor(X_vif.values, i)
        tol_val = 1.0 / vif_val
        vif_data.append({
            "Biến độc lập": col,
            "Hệ số VIF": round(vif_val, 4),
            "Độ chấp nhận (Tolerance)": round(tol_val, 4),
            "Đánh giá VIF < 10": "ĐẠT (Không đa cộng tuyến)" if vif_val < 10.0 else "CẢNH BÁO (> 10)"
        })

    df_vif = pd.DataFrame(vif_data)
    print(df_vif.to_string(index=False))

    max_vif = df_vif["Hệ số VIF"].max()
    if max_vif < 10.0:
        ket_luan = f"ĐẠT GIẢ ĐỊNH (Tất cả VIF đều < 10, VIF cao nhất là {max_vif:.2f} < 10)"
        ket_qua_danh_gia = "ĐẠT (PASS)"
    else:
        ket_luan = f"VI PHẠM (Phát hiện biến có VIF >= 10, cần kích hoạt Giao thức Dừng & Hỏi)"
        ket_qua_danh_gia = "VI PHẠM (FAIL)"
    print(f"-> KẾT LUẬN: {ket_luan}")

    # Vẽ biểu đồ cột ngang VIF
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    y_pos = np.arange(len(features))
    vif_values = df_vif["Hệ số VIF"].values

    # Thanh bar màu xanh
    bars = ax.barh(y_pos, vif_values, color="#1976D2", alpha=0.8, edgecolor="black", height=0.45)

    # Vẽ vạch cảnh báo đỏ tại VIF = 10 và vạch vàng tại VIF = 5
    ax.axvline(x=10.0, color="#D32F2F", linestyle="--", linewidth=2.5, label="Ngưỡng cảnh báo DA-SGU: VIF = 10")
    ax.axvline(x=5.0, color="#FBC02D", linestyle=":", linewidth=2.0, label="Ngưỡng khuyến nghị khắt khe: VIF = 5")

    # Ghi số VIF cụ thể trên từng thanh
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f"{width:.2f}",
                ha="left", va="center", fontsize=10, fontweight="bold", color="#0D47A1")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10, fontweight="bold")
    ax.set_xlabel("Hệ số Phóng đại Phương sai (VIF - Variance Inflation Factor)", fontsize=10, fontweight="bold")
    ax.set_title("GIẢ ĐỊNH 3: KIỂM ĐỊNH HỆ SỐ PHÓNG ĐẠI PHƯƠNG SAI VIF CÁC BIẾN ĐỘC LẬP", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, 12)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ 3 (fig07): {fig_path}")

    return {
        "Gia_dinh": "3. Không Đa cộng tuyến (No Multicollinearity)",
        "Kiem_dinh": "Hệ số VIF (Variance Inflation Factor)",
        "Gia_tri_thong_ke": f"Max VIF = {max_vif:.2f}",
        "p_value": "—",
        "Ket_qua": ket_qua_danh_gia
    }, df_vif

def kiem_dinh_gia_dinh_4_homoscedasticity(model, fig_path):
    """
    Task 5.4 — GIẢ ĐỊNH 4: Phương sai Sai số Không đổi (Homoscedasticity)
    - Vẽ biểu đồ phân tán Residuals vs Fitted values.
    - Thực hiện kiểm định Breusch-Pagan Test và White Test.
    - Nếu p-value < 0.05: Đề xuất giải pháp Robust Standard Errors (HC3) và Log-Log Transformation.
    """
    print("\n" + "=" * 85)
    print("TASK 5.4 — GIẢ ĐỊNH 4: PHƯƠNG SAI SAI SỐ KHÔNG ĐỔI (HOMOSCEDASTICITY)")
    print("=" * 85)

    residuals = model.resid
    fitted = model.fittedvalues
    exog = model.model.exog

    # 1. Kiểm định Breusch-Pagan
    bp_test = het_breuschpagan(residuals, exog)
    bp_lm_stat = bp_test[0]
    bp_pvalue = bp_test[1]
    print(f"- Kiểm định Breusch-Pagan: LM Statistic = {bp_lm_stat:.4f} | p-value = {bp_pvalue:.4e}")

    # 2. Kiểm định White
    white_test = het_white(residuals, exog)
    white_lm_stat = white_test[0]
    white_pvalue = white_test[1]
    print(f"- Kiểm định White Test   : LM Statistic = {white_lm_stat:.4f} | p-value = {white_pvalue:.4e}")

    if bp_pvalue >= 0.05 and white_pvalue >= 0.05:
        ket_luan = "ĐẠT GIẢ ĐỊNH (Chấp nhận H0: Phương sai sai số là không đổi)"
        ket_qua_danh_gia = "ĐẠT (PASS)"
    else:
        ket_luan = ("VI PHẠM (Bác bỏ H0: Phát hiện hiện tượng Phương sai sai số thay đổi Heteroscedasticity). "
                    "Giải pháp chuẩn DA-SGU: Dùng Sai số chuẩn vững HC3 (Robust Standard Errors) hoặc Mô hình Log-Log!")
        ket_qua_danh_gia = "VI PHẠM (CẦN XỬ LÝ)"
    print(f"-> KẾT LUẬN: {ket_luan}")

    # Vẽ biểu đồ Residuals vs Fitted Values
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # Điểm phân tán
    ax.scatter(fitted / 1000.0, residuals / 1000.0, color="#E65100", alpha=0.7, edgecolors="black", s=45, label="Phần dư thực tế")

    # Đường mốc y = 0
    ax.axhline(y=0, color="#000000", linestyle="-", linewidth=1.5)

    # Đường cận dải biến động sai số (đường biên)
    ax.axhline(y=15, color="#D32F2F", linestyle=":", linewidth=1.2, label="Dải biên sai số")
    ax.axhline(y=-15, color="#D32F2F", linestyle=":", linewidth=1.2)

    ax.set_title("GIẢ ĐỊNH 4: BIỂU ĐỒ PHẦN DƯ VS GIÁ TRỊ DỰ BÁO (RESIDUALS VS FITTED)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Giá trị Dự báo Fitted Values (Nghìn VNĐ/kg)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Phần dư Residuals (Nghìn VNĐ/kg)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ 4 (fig08): {fig_path}")

    # Chạy mô hình OLS với Robust Standard Errors HC3 theo khuyến nghị DA-SGU
    model_robust_hc3 = model.model.fit(cov_type="HC3")

    return {
        "Gia_dinh": "4. Phương sai Không đổi (Homoscedasticity)",
        "Kiem_dinh": "Breusch-Pagan & White Test",
        "Gia_tri_thong_ke": f"BP Stat={bp_lm_stat:.2f}, White Stat={white_lm_stat:.2f}",
        "p_value": f"BP p={bp_pvalue:.4e}, White p={white_pvalue:.4e}",
        "Ket_qua": ket_qua_danh_gia
    }, model_robust_hc3

def main():
    """
    Hàm điều phối toàn bộ chu trình Bước 5
    """
    file_unscaled = "data/processed/vn_coffee_unscaled.csv"
    fig05 = "figures/fig05_actual_vs_fitted.png"
    fig06 = "figures/fig06_residuals_normality.png"
    fig07 = "figures/fig07_vif_multicollinearity.png"
    fig08 = "figures/fig08_homoscedasticity_plot.png"

    assert os.path.exists(file_unscaled), f"Lỗi: Không tìm thấy file '{file_unscaled}'!"
    df = pd.read_csv(file_unscaled)

    # Tập huấn luyện 80%
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy().reset_index(drop=True)

    # Mô hình tối ưu sau Stepwise từ Bước 4
    features = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "Crude_Oil_Price"]
    y_train = df_train["VN_Coffee_Price"]
    X_train = sm.add_constant(df_train[features])
    model = sm.OLS(y_train, X_train).fit()

    # 1. Kiểm định Giả định 1: Linearity
    kq_gd1 = kiem_dinh_gia_dinh_1_linearity(model, y_train, fig05)

    # 2. Kiểm định Giả định 2: Normality of Residuals
    kq_gd2 = kiem_dinh_gia_dinh_2_normality(model, fig06)

    # 3. Kiểm định Giả định 3: Multicollinearity (VIF)
    kq_gd3, df_vif = kiem_dinh_gia_dinh_3_multicollinearity(df_train, features, fig07)

    # 4. Kiểm định Giả định 4: Homoscedasticity
    kq_gd4, model_robust_hc3 = kiem_dinh_gia_dinh_4_homoscedasticity(model, fig08)

    # Bảng tổng kết Checklist 4 Giả định
    bang_checklist = pd.DataFrame([kq_gd1, kq_gd2, kq_gd3, kq_gd4])
    print("\n" + "=" * 85)
    print("BẢNG TỔNG KẾT CHECKLIST 4 GIẢ ĐỊNH HỒI QUY THEO CHUẨN DA - SGU:")
    print("=" * 85)
    print(bang_checklist.to_string(index=False))

    # --------------------------------------------------------------------------
    # CHẠY SELF-TEST (ASSERTION RULES THEO QUY TẮC SỐ 5)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CHẠY SELF-TEST (TỰ ĐỘNG KIỂM THỬ BƯỚC 5)")
    print("=" * 85)

    # Assertion 1: Trung bình phần dư xấp xỉ bằng 0 tuyệt đối (sai số < 1e-5)
    mean_resid = abs(model.resid.mean())
    assert mean_resid < 1e-5, f"Self-Test Thất bại: Trung bình phần dư ({mean_resid}) không xấp xỉ 0!"
    print(f"[PASS] Assertion 1: |mean(residuals)| = {mean_resid:.8f} < 10^-5 (Trung bình phần dư hoàn hảo bằng 0).")

    # Assertion 2: Kiểm tra sự tồn tại và dung lượng của cả 4 biểu đồ
    danh_sach_fig = [fig05, fig06, fig07, fig08]
    for fig_file in danh_sach_fig:
        assert os.path.exists(fig_file) and os.path.getsize(fig_file) > 0, f"Self-Test Thất bại: File biểu đồ {fig_file} không hợp lệ!"
        print(f"[PASS] File biểu đồ {fig_file} tồn tại hợp lệ ({os.path.getsize(fig_file):,} bytes).")

    # Assertion 3: Giả định 1 Linearity và Giả định 2 Normality và Giả định 3 VIF phải ĐẠT
    assert model.f_pvalue < 0.05, "Self-Test Thất bại: Giả định 1 (Linearity F-test) không đạt!"
    w_s, p_s = shapiro(model.resid)
    assert p_s >= 0.05, "Self-Test Thất bại: Giả định 2 (Normality Shapiro) không đạt!"
    max_vif = df_vif["Hệ số VIF"].max()
    assert max_vif < 10.0, "Self-Test Thất bại: Giả định 3 (VIF) vượt ngưỡng 10!"
    print(f"[PASS] Assertion 3: Giả định 1, 2, 3 đều vượt qua 100% (Linearity p < 0.001, Shapiro p = {p_s:.4f} >= 0.05, Max VIF = {max_vif:.2f} < 10).")

    print("-" * 85)
    print("SELF-TEST BƯỚC 5: PASS 100% TOÀN BỘ CÁC BÀI KIỂM THỬ TỰ ĐỘNG!")
    print("=" * 85)

    return model, model_robust_hc3, bang_checklist

if __name__ == "__main__":
    main()
