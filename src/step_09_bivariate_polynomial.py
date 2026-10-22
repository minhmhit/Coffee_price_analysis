"""
====================================================================================================
DỰ ÁN PHÂN TÍCH VÀ DỰ BÁO GIÁ CÀ PHÊ TÂY NGUYÊN (VIETNAM COFFEE PRICE ANALYSIS)
HỌC PHẦN: KHAI PHÁ DỮ LIỆU & PHÂN TÍCH KINH DOANH (DATA ANALYSIS - SGU)
FILE: src/step_09_bivariate_polynomial.py
MỤC TIÊU: BƯỚC 9 — MÔ HÌNH HÓA ĐA THỨC VÀ TIỂU PHÂN HỆ (POLYNOMIAL & BIVARIATE REGRESSION)
          - Phân tích tương tác cục bộ từng cặp biến:
            + Phân hệ 1: Sàn London (X1) -> Giá Tây Nguyên (Y)
            + Phân hệ 2: Giá Dầu thô (X6) -> Giá Tây Nguyên (Y)
          - Đánh giá dạng hàm phi tuyến (Linear, Bậc 2 Quadratic, Bậc 3 Cubic) bằng PolynomialFeatures
====================================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cấu hình UTF-8 tiếng Việt cho Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình đồ họa matplotlib chuẩn khoa học
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-v0_8-whitegrid')

# Đường dẫn file
DATA_PATH = os.path.join("data", "processed", "vn_coffee_unscaled.csv")
FIGURES_DIR = "figures"
REPORTS_DIR = "Reports"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_du_lieu(filepath=DATA_PATH):
    """Đọc dữ liệu sạch thực tế 153 tháng từ vn_coffee_unscaled.csv"""
    if not os.path.exists(filepath):
        alternate_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alternate_path):
            filepath = alternate_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại '{filepath}'!")
    df = pd.read_csv(filepath)
    print(f"- Tải thành công dữ liệu: {len(df)} dòng quan sát.")
    return df


def danh_gia_phan_he_da_thuc(df, ten_bien_x, ten_bien_y, nhan_phan_he, don_vi_x):
    """
    Huấn luyện và so sánh mô hình hồi quy đa thức bậc 1, 2, 3 cho một tiểu phân hệ.
    Trả về bảng tổng hợp các chỉ số R2, Adj-R2, RMSE, MAE, AIC, BIC và các mô hình đã fit.
    """
    print("\n" + "=" * 80)
    print(f"TIỂU PHÂN HỆ: {nhan_phan_he.upper()} ({ten_bien_x} -> {ten_bien_y})")
    print("=" * 80)

    y = df[ten_bien_y].values
    x_raw = df[ten_bien_x].values.reshape(-1, 1)

    ket_qua = []
    cac_mo_hinh = {}
    cac_bien_doi = {}

    for deg in [1, 2, 3]:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_poly = poly.fit_transform(x_raw)
        X_sm = sm.add_constant(X_poly)
        
        # Ước lượng OLS
        model = sm.OLS(y, X_sm).fit()
        y_pred = model.predict(X_sm)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100

        cac_mo_hinh[deg] = model
        cac_bien_doi[deg] = poly

        thong_tin = {
            "Bậc đa thức": f"Degree {deg}",
            "Tên hàm": "Tuyến tính (Linear)" if deg == 1 else ("Bậc 2 (Quadratic)" if deg == 2 else "Bậc 3 (Cubic)"),
            "R2": round(model.rsquared, 4),
            "Adj R2": round(model.rsquared_adj, 4),
            "RMSE (VNĐ/kg)": round(rmse, 2),
            "MAE (VNĐ/kg)": round(mae, 2),
            "MAPE (%)": round(mape, 2),
            "AIC": round(model.aic, 2),
            "BIC": round(model.bic, 2),
            "F-stat": round(model.fvalue, 2),
            "p-value F": model.f_pvalue
        }
        ket_qua.append(thong_tin)

        print(f"\n[BẬC {deg} - {thong_tin['Tên hàm']}]:")
        print(f"  + R2 = {model.rsquared:.4f} | Adj R2 = {model.rsquared_adj:.4f} | RMSE = {rmse:,.2f} VNĐ/kg")
        print(f"  + AIC = {model.aic:.2f} | BIC = {model.bic:.2f}")
        for i, coef in enumerate(model.params):
            p_val = model.pvalues[i]
            label = "const (beta_0)" if i == 0 else f"X^{i} (beta_{i})"
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"    - {label:<16}: coef = {coef:>14.4f} | p-value = {p_val:.4e} ({sig})")

    df_so_sanh = pd.DataFrame(ket_qua)
    print("\n--- BẢNG TỔNG HỢP SO SÁNH 3 BẬC ĐA THỨC ---")
    print(df_so_sanh[["Bậc đa thức", "Tên hàm", "R2", "Adj R2", "RMSE (VNĐ/kg)", "AIC", "BIC"]].to_string(index=False))

    return df_so_sanh, cac_mo_hinh, cac_bien_doi


def ve_bieu_do_poly_fit(df, ten_bien_x, ten_bien_y, cac_mo_hinh, cac_bien_doi, df_so_sanh, 
                        tieu_de, nhan_x, nhan_y, ten_file_anh):
    """
    Vẽ biểu đồ Scatter Plot kèm 3 đường cong hồi quy (Linear, Poly 2, Poly 3) chuẩn 300 DPI.
    """
    x_raw = df[ten_bien_x].values
    y_raw = df[ten_bien_y].values

    # Tạo dải giá trị X mịn để vẽ đường cong liên tục (Smooth curve)
    x_min, x_max = x_raw.min(), x_raw.max()
    x_padding = (x_max - x_min) * 0.05
    x_grid = np.linspace(x_min - x_padding, x_max + x_padding, 400).reshape(-1, 1)

    plt.figure(figsize=(10, 6.5))

    # 1. Vẽ các điểm dữ liệu thực tế (Scatter)
    plt.scatter(x_raw, y_raw, color="#4a5568", alpha=0.55, edgecolors="#1a202c", 
                s=45, label="Quan sát thực tế (153 tháng: 2014-2026)", zorder=3)

    # Cấu hình màu và nét vẽ cho 3 bậc
    mau_sac = {1: "#2b6cb0", 2: "#2f855a", 3: "#c53030"}
    net_ve = {1: "--", 2: "-", 3: "-."}
    do_day = {1: 2.2, 2: 2.8, 3: 2.2}

    for deg in [1, 2, 3]:
        poly = cac_bien_doi[deg]
        model = cac_mo_hinh[deg]
        
        X_grid_poly = poly.transform(x_grid)
        X_grid_sm = sm.add_constant(X_grid_poly)
        y_grid_pred = model.predict(X_grid_sm)

        r2_val = df_so_sanh.loc[df_so_sanh["Bậc đa thức"] == f"Degree {deg}", "R2"].values[0]
        rmse_val = df_so_sanh.loc[df_so_sanh["Bậc đa thức"] == f"Degree {deg}", "RMSE (VNĐ/kg)"].values[0]
        
        ten_bac = "Bậc 1 (Tuyến tính)" if deg == 1 else ("Bậc 2 (Quadratic)" if deg == 2 else "Bậc 3 (Cubic)")
        nhan_chu_thich = f"{ten_bac}: $R^2 = {r2_val:.4f}$ | RMSE = {rmse_val:,.0f}"

        plt.plot(x_grid.flatten(), y_grid_pred, color=mau_sac[deg], linestyle=net_ve[deg],
                 linewidth=do_day[deg], label=nhan_chu_thich, zorder=4)

    plt.title(tieu_de, fontsize=13, fontweight="bold", pad=14, color="#1a202c")
    plt.xlabel(nhan_x, fontsize=11, fontweight="semibold", labelpad=8)
    plt.ylabel(nhan_y, fontsize=11, fontweight="semibold", labelpad=8)

    # Định dạng trục tọa độ đẹp mắt
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"{int(val):,}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"{int(val):,}"))

    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e0", fontsize=9.5, loc="upper left")
    plt.tight_layout()

    duong_dan_anh = os.path.join(FIGURES_DIR, ten_file_anh)
    plt.savefig(duong_dan_anh, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"- Đã lưu biểu đồ: '{duong_dan_anh}' (Độ phân giải 300 DPI).")
    return duong_dan_anh


def main():
    print("=" * 85)
    print("BƯỚC 9: MÔ HÌNH HÓA ĐA THỨC VÀ TIỂU PHÂN HỆ (POLYNOMIAL & BIVARIATE REGRESSION)")
    print("=" * 85)

    df = load_du_lieu()

    # TASK 9.1 & 9.2 & 9.3:
    # -----------------------------------------------------------------------------------------
    # PHÂN HỆ 1: Sàn London -> Giá Tây Nguyên
    # -----------------------------------------------------------------------------------------
    df_london, models_london, polys_london = danh_gia_phan_he_da_thuc(
        df=df,
        ten_bien_x="London_Robusta_Price",
        ten_bien_y="VN_Coffee_Price",
        nhan_phan_he="Phân hệ 1: Sàn London (ICE Futures Europe) tác động Giá Cà phê Tây Nguyên",
        don_vi_x="USD/Tấn"
    )

    # Vẽ biểu đồ Phân hệ 1 (fig15)
    ve_bieu_do_poly_fit(
        df=df,
        ten_bien_x="London_Robusta_Price",
        ten_bien_y="VN_Coffee_Price",
        cac_mo_hinh=models_london,
        cac_bien_doi=polys_london,
        df_so_sanh=df_london,
        tieu_de="PHÂN HỆ 1: QUAN HỆ HỒI QUY ĐA THỨC GIỮA GIÁ SÀN LONDON VÀ GIÁ CÀ PHÊ TÂY NGUYÊN",
        nhan_x="Giá Cà phê Robusta kỳ hạn London (USD/Tấn)",
        nhan_y="Giá Cà phê Tây Nguyên (VNĐ/kg)",
        ten_file_anh="fig15_london_poly_fit.png"
    )

    # -----------------------------------------------------------------------------------------
    # PHÂN HỆ 2: Giá Dầu thô -> Giá Tây Nguyên
    # -----------------------------------------------------------------------------------------
    df_oil, models_oil, polys_oil = danh_gia_phan_he_da_thuc(
        df=df,
        ten_bien_x="Crude_Oil_Price",
        ten_bien_y="VN_Coffee_Price",
        nhan_phan_he="Phân hệ 2: Giá Dầu thô Brent tác động Giá Cà phê Tây Nguyên",
        don_vi_x="USD/Thùng"
    )

    # Vẽ biểu đồ Phân hệ 2 (fig16)
    ve_bieu_do_poly_fit(
        df=df,
        ten_bien_x="Crude_Oil_Price",
        ten_bien_y="VN_Coffee_Price",
        cac_mo_hinh=models_oil,
        cac_bien_doi=polys_oil,
        df_so_sanh=df_oil,
        tieu_de="PHÂN HỆ 2: QUAN HỆ HỒI QUY ĐA THỨC GIỮA GIÁ DẦU THÔ BRENT VÀ GIÁ CÀ PHÊ TÂY NGUYÊN",
        nhan_x="Giá Dầu thô Brent thế giới (USD/Thùng)",
        nhan_y="Giá Cà phê Tây Nguyên (VNĐ/kg)",
        ten_file_anh="fig16_oil_poly_fit.png"
    )

    # TỰ KIỂM TRA (SELF-TEST) THEO YÊU CẦU ĐỀ BÀI:
    print("\n" + "=" * 80)
    print("TỰ KIỂM TRA (SELF-TEST) THEO CHUẨN ĐỒ ÁN:")
    print("=" * 80)
    
    # Kiểm tra Bậc nào có R2 cao nhất
    max_r2_london_deg = df_london.loc[df_london["R2"].idxmax(), "Bậc đa thức"]
    max_r2_oil_deg = df_oil.loc[df_oil["R2"].idxmax(), "Bậc đa thức"]
    print(f"- Phân hệ 1 (London): Bậc có R2 cao nhất là {max_r2_london_deg} (R2 = {df_london['R2'].max():.4f})")
    print(f"- Phân hệ 2 (Dầu thô): Bậc có R2 cao nhất là {max_r2_oil_deg} (R2 = {df_oil['R2'].max():.4f})")

    # Kiểm tra tính hợp lệ thống kê
    assert df_london["R2"].max() > 0.80, "Lỗi: R2 của Phân hệ London không đạt chuẩn > 0.80"
    assert df_oil["R2"].max() > 0.15, "Lỗi: R2 của Phân hệ Dầu thô không đạt chuẩn > 0.15"
    print("- Toàn bộ các kiểm định self-test: PASS 100%!")

    return df_london, df_oil


if __name__ == "__main__":
    main()
