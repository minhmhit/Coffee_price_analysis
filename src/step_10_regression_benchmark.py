"""
====================================================================================================
DỰ ÁN PHÂN TÍCH VÀ DỰ BÁO GIÁ CÀ PHÊ TÂY NGUYÊN (VIETNAM COFFEE PRICE ANALYSIS)
HỌC PHẦN: KHAI PHÁ DỮ LIỆU & PHÂN TÍCH KINH DOANH (DATA ANALYSIS - SGU)
FILE: src/step_10_regression_benchmark.py
MỤC TIÊU: BƯỚC 10 — ĐỐI CHUẨN ĐA THUẬT TOÁN HỌC MÁY HỒI QUY (ML REGRESSION BENCHMARKING)
          - Đặt OLS Baseline lên bàn cân so sánh với 4 thuật toán Học máy hiện đại:
            1. OLS (Linear Regression Baseline)
            2. Decision Tree Regressor (CART)
            3. Random Forest Regressor (Ensemble of 100 Trees)
            4. K-Nearest Neighbors Regressor (KNN, k=5)
            5. Support Vector Regression (SVR, RBF Kernel)
          - So sánh hiệu năng toàn diện trên cả tập Huấn luyện (Train) và Kiểm thử (Test)
          - Phân tích cơ chế "Extrapolation Failure" của các mô hình phi tuyến cục bộ
====================================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Cấu hình tiếng Việt UTF-8 cho Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình đồ họa matplotlib chuẩn khoa học
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-v0_8-whitegrid')

# Đường dẫn dữ liệu và xuất kết quả
DATA_PATH = os.path.join("data", "processed", "vn_coffee_unscaled.csv")
FIGURES_DIR = "figures"
REPORTS_DIR = "Reports"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_va_chia_tap_du_lieu(filepath=DATA_PATH, ty_le_train=0.8):
    """
    Đọc dữ liệu sạch 153 tháng và chia tập Train (80% = 122 tháng) / Test (20% = 31 tháng).
    Chuẩn hóa StandardScaler trên tập Train để tránh rò rỉ dữ liệu (Data Leakage).
    """
    if not os.path.exists(filepath):
        alternate_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alternate_path):
            filepath = alternate_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại '{filepath}'!")

    df = pd.read_csv(filepath)
    features = ['VN_Coffee_Price_Lag1', 'London_Robusta_Price', 'Crude_Oil_Price']
    target = 'VN_Coffee_Price'

    split_idx = int(len(df) * ty_le_train)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    # Chuẩn hóa ma trận đặc trưng X
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    # Chuẩn hóa biến mục tiêu Y riêng biệt (phục vụ mô hình SVR kernel RBF)
    scaler_y = StandardScaler()
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    print(f"- Tập Huấn luyện (Train): {len(train_df)} tháng ({train_df['Date'].iloc[0]} -> {train_df['Date'].iloc[-1]})")
    print(f"- Tập Kiểm thử (Test)  : {len(test_df)} tháng ({test_df['Date'].iloc[0]} -> {test_df['Date'].iloc[-1]})")
    print(f"- Bộ biến độc lập tối ưu: {features}")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train_sc": X_train_sc,
        "X_test_sc": X_test_sc,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_sc": y_train_sc,
        "scaler_y": scaler_y,
        "features": features
    }


def khoi_tao_5_mo_hinh():
    """Khởi tạo danh sách 5 thuật toán học máy đối chuẩn"""
    models = {
        "1. OLS Baseline": LinearRegression(),
        "2. Decision Tree": DecisionTreeRegressor(random_state=42),
        "3. Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "4. KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
        "5. SVR (RBF Kernel)": SVR(kernel='rbf', C=100.0, epsilon=0.1)
    }
    return models


def danh_gia_va_xep_hang_mo_hinh(models, data):
    """
    Huấn luyện và đánh giá chi tiết 5 mô hình trên cả tập Train và tập Test.
    Lập bảng xếp hạng Leaderboard theo RMSE trên tập Test.
    """
    X_train = data["X_train_sc"]
    y_train = data["y_train"]
    X_test = data["X_test_sc"]
    y_test = data["y_test"]
    y_train_sc = data["y_train_sc"]
    scaler_y = data["scaler_y"]

    ket_qua = []
    cac_du_bao_test = {}

    print("\n" + "=" * 90)
    print("TASK 10.2 & 10.3: HUẤN LUYỆN VÀ ĐO LƯỜNG HIỆU NĂNG 5 THUẬT TOÁN HỌC MÁY")
    print("=" * 90)

    for name, model in models.items():
        if "SVR" in name:
            model.fit(X_train, y_train_sc)
            pred_train_sc = model.predict(X_train)
            pred_train = scaler_y.inverse_transform(pred_train_sc.reshape(-1, 1)).flatten()

            pred_test_sc = model.predict(X_test)
            pred_test = scaler_y.inverse_transform(pred_test_sc.reshape(-1, 1)).flatten()
        else:
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

        cac_du_bao_test[name] = pred_test

        # Các chỉ số trên tập Train
        r2_train = r2_score(y_train, pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
        mae_train = mean_absolute_error(y_train, pred_train)

        # Các chỉ số trên tập Test
        r2_test = r2_score(y_test, pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
        mae_test = mean_absolute_error(y_test, pred_test)
        mape_test = mean_absolute_percentage_error(y_test, pred_test) * 100

        ban_ghi = {
            "Mô hình": name,
            "R2 (Train)": round(r2_train, 4),
            "RMSE Train (VNĐ)": round(rmse_train, 2),
            "R2 (Test)": round(r2_test, 4),
            "RMSE Test (VNĐ)": round(rmse_test, 2),
            "MAE Test (VNĐ)": round(mae_test, 2),
            "MAPE Test (%)": round(mape_test, 2)
        }
        ket_qua.append(ban_ghi)

    df_ket_qua = pd.DataFrame(ket_qua)
    # Sắp xếp theo RMSE Test tăng dần (Thấp nhất = Tốt nhất)
    df_leaderboard = df_ket_qua.sort_values(by="RMSE Test (VNĐ)", ascending=True).reset_index(drop=True)
    df_leaderboard["Hạng"] = range(1, len(df_leaderboard) + 1)

    print("\n--- BẢNG XẾP HẠNG LEADERBOARD ĐA THUẬT TOÁN HỌC MÁY TRÊN TẬP TEST ---")
    cot_hien_thi = ["Hạng", "Mô hình", "R2 (Test)", "RMSE Test (VNĐ)", "MAE Test (VNĐ)", "MAPE Test (%)", "R2 (Train)"]
    print(df_leaderboard[cot_hien_thi].to_string(index=False))

    return df_leaderboard, cac_du_bao_test


def ve_bieu_do_benchmark(df_leaderboard, cac_du_bao_test, data):
    """
    Vẽ biểu đồ Bar Chart so sánh RMSE, R2 Score và Timeline dự báo giữa 5 mô hình.
    Lưu ảnh độ phân giải cao 300 DPI: figures/fig17_regression_benchmark.png.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    df_plot = df_leaderboard.sort_values(by="Hạng", ascending=False).reset_index(drop=True)
    ten_mo_hinh = df_plot["Mô hình"]

    # 1. Biểu đồ RMSE (Trục bên trái)
    mau_rmse = ["#e53e3e" if r > 20000 else "#319795" for r in df_plot["RMSE Test (VNĐ)"]]
    bars1 = ax1.barh(ten_mo_hinh, df_plot["RMSE Test (VNĐ)"], color=mau_rmse, height=0.55, edgecolor="#2d3748")
    ax1.set_title("SO SÁNH SAI SỐ RMSE TRÊN TẬP TEST (CÀNG NHỎ CÀNG TỐT)", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xlabel("RMSE (VNĐ/kg)", fontsize=10.5, fontweight="semibold")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"{int(val):,}"))

    # Thêm nhãn số liệu lên thanh bar
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 1200, bar.get_y() + bar.get_height() / 2, f"{int(w):,} đ", 
                 va="center", ha="left", fontsize=9.5, fontweight="bold", color="#1a202c")
    ax1.set_xlim(0, max(df_plot["RMSE Test (VNĐ)"]) * 1.18)

    # 2. Biểu đồ R2 Score (Trục bên phải)
    mau_r2 = ["#2b6cb0" if r > 0 else "#c53030" for r in df_plot["R2 (Test)"]]
    bars2 = ax2.barh(ten_mo_hinh, df_plot["R2 (Test)"], color=mau_r2, height=0.55, edgecolor="#2d3748")
    ax2.axvline(x=0, color="#718096", linestyle="--", linewidth=1.2)
    ax2.set_title("SO SÁNH HỆ SỐ XÁC ĐỊNH R² TRÊN TẬP TEST (CÀNG GẦN 1 CÀNG TỐT)", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xlabel("R² Score", fontsize=10.5, fontweight="semibold")

    for bar in bars2:
        w = bar.get_width()
        offset = 0.25 if w >= 0 else -0.35
        ha = "left" if w >= 0 else "right"
        ax2.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", 
                 va="center", ha=ha, fontsize=9.5, fontweight="bold", color="#1a202c")
    ax2.set_xlim(min(df_plot["R2 (Test)"]) * 1.25, 1.1)

    plt.suptitle("CUỘC ĐUA ĐA THUẬT TOÁN HỌC MÁY DỰ BÁO GIÁ CÀ PHÊ VIỆT NAM (REGRESSION BENCHMARK)", 
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    duong_dan_anh = os.path.join(FIGURES_DIR, "fig17_regression_benchmark.png")
    plt.savefig(duong_dan_anh, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n- Đã lưu biểu đồ: '{duong_dan_anh}' (Độ phân giải 300 DPI).")
    return duong_dan_anh


def main():
    print("=" * 90)
    print("BƯỚC 10: ĐỐI CHUẨN ĐA THUẬT TOÁN HỌC MÁY HỒI QUY (REGRESSION BENCHMARKING)")
    print("=" * 90)

    # 1. Chuẩn bị dữ liệu Train/Test
    data = load_va_chia_tap_du_lieu()

    # 2. Khởi tạo và huấn luyện 5 mô hình
    models = khoi_tao_5_mo_hinh()

    # 3. Đánh giá và lập Leaderboard
    df_leaderboard, cac_du_bao_test = danh_gia_va_xep_hang_mo_hinh(models, data)

    # 4. Xuất biểu đồ so sánh (fig17)
    ve_bieu_do_benchmark(df_leaderboard, cac_du_bao_test, data)

    # 5. Tự kiểm tra (Self-Test Assertions)
    print("\n" + "=" * 90)
    print("TỰ KIỂM TRA (SELF-TEST) CHUẨN BỊ BÁO CÁO GIẢNG VIÊN:")
    print("=" * 90)
    
    quan_quan = df_leaderboard.iloc[0]["Mô hình"]
    r2_quan_quan = df_leaderboard.iloc[0]["R2 (Test)"]
    rmse_quan_quan = df_leaderboard.iloc[0]["RMSE Test (VNĐ)"]
    
    print(f"- Quán quân Leaderboard: {quan_quan} (R2 = {r2_quan_quan:.4f}, RMSE = {rmse_quan_quan:,.2f} VNĐ/kg)")
    assert "OLS" in quan_quan, f"Lỗi: OLS không phải là quán quân, thực tế là: {quan_quan}"
    assert r2_quan_quan > 0.80, f"Lỗi: R2 của quán quân < 0.80: {r2_quan_quan}"
    assert len(df_leaderboard) == 5, f"Lỗi: Số lượng mô hình không đủ 5: {len(df_leaderboard)}"
    print("- Toàn bộ các tiêu chí kiểm thử Self-Test: PASS 100%!")

    return df_leaderboard


if __name__ == "__main__":
    main()
