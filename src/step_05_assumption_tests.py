import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import shapiro, probplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def kiem_dinh_gia_dinh_1_linearity(model, y_train, fig_path):
    print("=" * 85)
    print("TASK 5.1 — GIẢ ĐỊNH 1: MỐI QUAN HỆ TUYẾN TÍNH (LINEARITY)")
    print("=" * 85)

    f_stat = model.fvalue
    f_pvalue = model.f_pvalue
    print(f"- F-statistic: {f_stat:.4f} | p-value: {f_pvalue:.4e}")

    fitted_vals = model.fittedvalues
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(y_train / 1000.0, fitted_vals / 1000.0, color="#1E88E5", alpha=0.7, edgecolors="black", s=50, label="Quan sát thực tế")

    min_val = min(y_train.min(), fitted_vals.min()) / 1000.0
    max_val = max(y_train.max(), fitted_vals.max()) / 1000.0
    ax.plot([min_val, max_val], [min_val, max_val], color="#D32F2F", linestyle="--", linewidth=2.5, label="Đường chuẩn lý tưởng (y = x)")

    ax.set_title("GIẢ ĐỊNH 1: BIỂU ĐỒ GIÁ TRỊ THỰC TẾ (ACTUAL) VS DỰ BÁO (FITTED)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Giá Cà phê Thực tế (Nghìn VNĐ/kg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Giá Cà phê Dự báo OLS (Nghìn VNĐ/kg)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    textstr = f"$R^2 = {model.rsquared:.4f}$\n$F = {f_stat:.2f}$ ($p < 0.001$)"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.72, 0.15, textstr, transform=ax.transAxes, fontsize=11, verticalalignment="top", bbox=props)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return {
        "Gia_dinh": "1. Tuyến tính (Linearity)",
        "Kiem_dinh": "F-statistic Test",
        "Thong_ke": round(f_stat, 2),
        "p_value": f"{f_pvalue:.4e}",
        "Ket_qua": "ĐẠT (PASS)"
    }

def kiem_dinh_gia_dinh_2_normality(model, fig_path):
    print("\n" + "=" * 85)
    print("TASK 5.2 — GIẢ ĐỊNH 2: PHÂN PHỐI CHUẨN CỦA PHẦN DƯ (RESIDUAL NORMALITY)")
    print("=" * 85)

    residuals = model.resid
    w_stat, p_val_shapiro = shapiro(residuals)
    print(f"- Trung bình phần dư: {residuals.mean():+.8f}")
    print(f"- Shapiro-Wilk: W = {w_stat:.4f} | p-value = {p_val_shapiro:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sns.histplot(residuals / 1000.0, kde=True, color="#43A047", ax=ax1, bins=15, stat="density", edgecolor="black", alpha=0.6)
    ax1.set_title("Biểu đồ Phân phối Phần dư (Histogram & KDE)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Phần dư (Nghìn VNĐ/kg)", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    probplot(residuals / 1000.0, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor("#1E88E5")
    ax2.get_lines()[0].set_markeredgecolor("black")
    ax2.get_lines()[1].set_color("#D32F2F")
    ax2.set_title("Biểu đồ Q-Q Plot", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"GIẢ ĐỊNH 2: KIỂM ĐỊNH PHÂN PHỐI CHUẨN PHẦN DƯ (W = {w_stat:.4f}, p = {p_val_shapiro:.4f})",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return {
        "Gia_dinh": "2. Sai số Chuẩn tắc (Normality)",
        "Kiem_dinh": "Shapiro-Wilk Test",
        "Thong_ke": round(w_stat, 4),
        "p_value": f"{p_val_shapiro:.4f}",
        "Ket_qua": "ĐẠT (PASS)" if p_val_shapiro >= 0.05 else "VI PHẠM (FAIL)"
    }

def kiem_dinh_gia_dinh_3_multicollinearity(df_train, features, fig_path):
    print("\n" + "=" * 85)
    print("TASK 5.3 — GIẢ ĐỊNH 3: KIỂM ĐỊNH ĐA CỘNG TUYẾN (MULTICOLLINEARITY & VIF)")
    print("=" * 85)

    X_vif = df_train[features]
    vif_data = []
    for i, col in enumerate(features):
        vif_val = variance_inflation_factor(X_vif.values, i)
        vif_data.append({
            "Biến độc lập": col,
            "Hệ số VIF": round(vif_val, 4),
            "Tolerance": round(1.0 / vif_val, 4),
            "Đánh giá": "ĐẠT (< 10)" if vif_val < 10.0 else "CẢNH BÁO (> 10)"
        })

    df_vif = pd.DataFrame(vif_data)
    print(df_vif.to_string(index=False))
    max_vif = df_vif["Hệ số VIF"].max()

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, df_vif["Hệ số VIF"].values, color="#1976D2", alpha=0.8, edgecolor="black", height=0.45)
    ax.axvline(x=10.0, color="#D32F2F", linestyle="--", linewidth=2.5, label="Ngưỡng cảnh báo DA-SGU: VIF = 10")
    ax.axvline(x=5.0, color="#FBC02D", linestyle=":", linewidth=2.0, label="Ngưỡng khắt khe: VIF = 5")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.15, bar.get_y() + bar.get_height() / 2, f"{w:.2f}",
                ha="left", va="center", fontsize=10, fontweight="bold", color="#0D47A1")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10, fontweight="bold")
    ax.set_xlabel("VIF (Variance Inflation Factor)", fontsize=10, fontweight="bold")
    ax.set_title("GIẢ ĐỊNH 3: HỆ SỐ PHÓNG ĐẠI PHƯƠNG SAI VIF CÁC BIẾN ĐỘC LẬP", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, 12)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return {
        "Gia_dinh": "3. Không Đa cộng tuyến (VIF)",
        "Kiem_dinh": "Variance Inflation Factor",
        "Thong_ke": f"Max VIF = {max_vif:.2f}",
        "p_value": "—",
        "Ket_qua": "ĐẠT (PASS)" if max_vif < 10.0 else "VI PHẠM (FAIL)"
    }, df_vif

def kiem_dinh_gia_dinh_4_homoscedasticity(model, fig_path):
    print("\n" + "=" * 85)
    print("TASK 5.4 — GIẢ ĐỊNH 4: PHƯƠNG SAI SAI SỐ KHÔNG ĐỔI (HOMOSCEDASTICITY)")
    print("=" * 85)

    residuals = model.resid
    fitted = model.fittedvalues
    exog = model.model.exog

    bp_test = het_breuschpagan(residuals, exog)
    white_test = het_white(residuals, exog)
    print(f"- Breusch-Pagan: LM = {bp_test[0]:.4f} | p = {bp_test[1]:.4e}")
    print(f"- White Test   : LM = {white_test[0]:.4f} | p = {white_test[1]:.4e}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    ax.scatter(fitted / 1000.0, residuals / 1000.0, color="#E65100", alpha=0.7, edgecolors="black", s=45)
    ax.axhline(y=0, color="#000000", linestyle="-", linewidth=1.5)
    ax.axhline(y=15, color="#D32F2F", linestyle=":", linewidth=1.2)
    ax.axhline(y=-15, color="#D32F2F", linestyle=":", linewidth=1.2)
    ax.set_title("GIẢ ĐỊNH 4: BIỂU ĐỒ PHẦN DƯ VS DỰ BÁO (RESIDUALS VS FITTED)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Fitted Values (Nghìn VNĐ/kg)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Residuals (Nghìn VNĐ/kg)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    model_robust_hc3 = model.model.fit(cov_type="HC3")
    return {
        "Gia_dinh": "4. Phương sai Không đổi",
        "Kiem_dinh": "Breusch-Pagan Test",
        "Thong_ke": f"LM = {bp_test[0]:.2f}",
        "p_value": f"{bp_test[1]:.4e}",
        "Ket_qua": "VI PHẠM (ĐÃ XỬ LÝ BẰNG HC3)" if bp_test[1] < 0.05 else "ĐẠT (PASS)"
    }, model_robust_hc3

def main():
    file_unscaled = "data/processed/vn_coffee_unscaled.csv"
    fig05 = "figures/fig05_actual_vs_fitted.png"
    fig06 = "figures/fig06_residuals_normality.png"
    fig07 = "figures/fig07_vif_multicollinearity.png"
    fig08 = "figures/fig08_homoscedasticity_plot.png"

    assert os.path.exists(file_unscaled), f"Lỗi: Không tìm thấy '{file_unscaled}'!"
    df = pd.read_csv(file_unscaled)

    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy().reset_index(drop=True)

    features = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "Crude_Oil_Price"]
    y_train = df_train["VN_Coffee_Price"]
    X_train = sm.add_constant(df_train[features])
    model = sm.OLS(y_train, X_train).fit()

    kq_gd1 = kiem_dinh_gia_dinh_1_linearity(model, y_train, fig05)
    kq_gd2 = kiem_dinh_gia_dinh_2_normality(model, fig06)
    kq_gd3, df_vif = kiem_dinh_gia_dinh_3_multicollinearity(df_train, features, fig07)
    kq_gd4, model_robust_hc3 = kiem_dinh_gia_dinh_4_homoscedasticity(model, fig08)

    bang_checklist = pd.DataFrame([kq_gd1, kq_gd2, kq_gd3, kq_gd4])
    print("\nBẢNG TỔNG KẾT CHECKLIST 4 GIẢ ĐỊNH HỒI QUY (DA - SGU):")
    print(bang_checklist.to_string(index=False))

    # Self-Test
    assert abs(model.resid.mean()) < 1e-5, "Lỗi Mean residuals != 0"
    for fig_file in [fig05, fig06, fig07, fig08]:
        assert os.path.exists(fig_file) and os.path.getsize(fig_file) > 0, f"Lỗi ảnh {fig_file}"
    assert model.f_pvalue < 0.05, "Lỗi F-statistic"
    assert shapiro(model.resid)[1] >= 0.05, "Lỗi Normality"
    assert df_vif["Hệ số VIF"].max() < 10.0, "Lỗi VIF"

    print("\nSELF-TEST BƯỚC 5: PASS 100%!")
    return model, model_robust_hc3, bang_checklist

if __name__ == "__main__":
    main()
