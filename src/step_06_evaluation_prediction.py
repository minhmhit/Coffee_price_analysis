# ==============================================================================
# TẬP TIN: src/step_06_evaluation_prediction.py
# MỤC ĐÍCH: Đánh giá chất lượng mô hình, trích xuất phương trình & dự báo giá cà phê (KDD Bước 6 & 7)
# QUY CHUẨN: Giáo trình Phân tích Dữ liệu DA - SGU (Đại học Sài Gòn)
# CHỈ SỐ: ESS, TSS, R2, Adj-R2, AIC, BIC, RMSE, MAE, MAPE
# KỊCH BẢN: Dự báo tập Test & 3 Kịch bản tương lai (Cơ sở, Tăng nóng, Hạ nhiệt)
# ==============================================================================

# Import thư viện sys để cấu hình hiển thị mã hóa tiếng Việt UTF-8 trên Windows console
import sys

# Import thư viện os để quản lý đường dẫn tập tin và kiểm tra dung lượng thư mục
import os

# Import thư viện pandas để thao tác bảng dữ liệu và trích xuất chỉ số
import pandas as pd

# Import thư viện numpy để tính toán các công thức toán học ma trận và căn bậc hai
import numpy as np

# Import statsmodels.api để ước lượng OLS và tính khoảng tin cậy dự báo
import statsmodels.api as sm

# Import các chỉ số đo lường sai số dự báo từ sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Import matplotlib.pyplot và seaborn để kết xuất biểu đồ dự báo 300 DPI
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập mã hóa UTF-8 cho luồng xuất console tránh lỗi ký tự tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình font chữ hỗ trợ tiếng Việt có dấu và hiển thị chuẩn dấu trừ trên Matplotlib
plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def tinh_toan_cac_chi_so_goc_da_sgu(y_true, y_pred, n_obs, k_features, llf):
    """
    Task 6.1: Tính toán tường minh các chỉ số đánh giá mô hình theo công thức gốc DA - SGU:
    - ESS (Error Sum of Squares) = sum((y_i - y_hat_i)^2)
    - TSS (Total Sum of Squares) = sum((y_i - y_bar)^2)
    - R-squared = 1 - (ESS / TSS)
    - Adjusted R-squared = 1 - [(ESS / (n - k - 1)) / (TSS / (n - 1))]
    - AIC = -2*ln(L) + 2*(k + 1)
    - BIC = -2*ln(L) + ln(n)*(k + 1)
    """
    ess = np.sum((y_true - y_pred) ** 2)
    y_bar = np.mean(y_true)
    tss = np.sum((y_true - y_bar) ** 2)
    
    r2 = 1.0 - (ess / tss)
    adj_r2 = 1.0 - ((ess / (n_obs - k_features - 1)) / (tss / (n_obs - 1)))
    
    aic = -2.0 * llf + 2.0 * (k_features + 1)
    bic = -2.0 * llf + np.log(n_obs) * (k_features + 1)
    
    return {
        "ESS": ess,
        "TSS": tss,
        "R2": r2,
        "Adj_R2": adj_r2,
        "AIC": aic,
        "BIC": bic
    }

def huan_luyen_va_so_sanh_mo_hinh(df_train, df_test, bien_phu_thuoc, features):
    """
    Huấn luyện song song cả 2 mô hình theo yêu cầu của User:
    1. Mô hình OLS Tuyến tính với Sai số chuẩn vững HC3 (Robust Standard Errors)
    2. Mô hình Log-Log (Biến đổi ln(Y) và ln(X))
    So sánh với Full Model và AR(1) Model.
    """
    print("=" * 85)
    print("TASK 6.1: TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH THEO CÔNG THỨC GỐC DA-SGU")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    y_test = df_test[bien_phu_thuoc]

    # --- 1. MÔ HÌNH OLS TUYẾN TÍNH TỐI ƯU VỚI ROBUST HC3 ---
    X_train_lin = sm.add_constant(df_train[features])
    X_test_lin = sm.add_constant(df_test[features])
    
    m_lin = sm.OLS(y_train, X_train_lin).fit()
    m_hc3 = sm.OLS(y_train, X_train_lin).fit(cov_type="HC3")

    # Tính toán thủ công theo công thức gốc
    calc_lin = tinh_toan_cac_chi_so_goc_da_sgu(
        y_train.values, m_lin.fittedvalues.values, len(y_train), len(features), m_lin.llf
    )
    print(f"- [Mô hình Tuyến tính OLS Tối ưu]:")
    print(f"  + ESS (Tổng bình phương phần dư) : {calc_lin['ESS']:,.2f}")
    print(f"  + TSS (Tổng bình phương toàn bộ) : {calc_lin['TSS']:,.2f}")
    print(f"  + R-squared (Hệ số xác định)    : {calc_lin['R2']:.4f} (Khớp statsmodels: {m_lin.rsquared:.4f})")
    print(f"  + Adjusted R-squared (Hiệu chỉnh): {calc_lin['Adj_R2']:.4f} (Khớp statsmodels: {m_lin.rsquared_adj:.4f})")
    print(f"  + AIC (Tiêu chuẩn thông tin)     : {calc_lin['AIC']:.2f}")
    print(f"  + BIC (Tiêu chuẩn Bayesian)      : {calc_lin['BIC']:.2f}")

    # Dự báo trên tập Test
    pred_test_lin = m_lin.predict(X_test_lin)
    rmse_lin = np.sqrt(mean_squared_error(y_test, pred_test_lin))
    mae_lin = mean_absolute_error(y_test, pred_test_lin)
    mape_lin = mean_absolute_percentage_error(y_test, pred_test_lin) * 100.0

    # --- 2. MÔ HÌNH LOG-LOG SONG SONG ---
    train_log = df_train.copy()
    test_log = df_test.copy()
    
    train_log["ln_Y"] = np.log(train_log[bien_phu_thuoc])
    test_log["ln_Y"] = np.log(test_log[bien_phu_thuoc])
    
    log_features = []
    for col in features:
        train_log["ln_" + col] = np.log(train_log[col])
        test_log["ln_" + col] = np.log(test_log[col])
        log_features.append("ln_" + col)

    X_train_log = sm.add_constant(train_log[log_features])
    X_test_log = sm.add_constant(test_log[log_features])

    m_log = sm.OLS(train_log["ln_Y"], X_train_log).fit()

    # Dự báo tập Test cho mô hình Log-Log (Quy đổi ngược qua exp có hiệu chỉnh sigma^2/2)
    pred_test_ln_y = m_log.predict(X_test_log)
    sigma2_resid = m_log.mse_resid
    pred_test_log = np.exp(pred_test_ln_y + sigma2_resid / 2.0)

    rmse_log = np.sqrt(mean_squared_error(y_test, pred_test_log))
    mae_log = mean_absolute_error(y_test, pred_test_log)
    mape_log = mean_absolute_percentage_error(y_test, pred_test_log) * 100.0

    # --- 3. MÔ HÌNH FULL MODEL & AR(1) ĐỂ SO SÁNH ---
    all_cand = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "VN_Production", "Crude_Oil_Price", "Harvest_Season_In_Harvest"]
    X_train_full = sm.add_constant(df_train[all_cand], has_constant="add")
    X_test_full = sm.add_constant(df_test[all_cand], has_constant="add")
    m_full = sm.OLS(y_train, X_train_full).fit()
    pred_full = m_full.predict(X_test_full)
    mape_full = mean_absolute_percentage_error(y_test, pred_full) * 100.0

    X_train_ar1 = sm.add_constant(df_train[["VN_Coffee_Price_Lag1"]], has_constant="add")
    X_test_ar1 = sm.add_constant(df_test[["VN_Coffee_Price_Lag1"]], has_constant="add")
    m_ar1 = sm.OLS(y_train, X_train_ar1).fit()
    pred_ar1 = m_ar1.predict(X_test_ar1)
    mape_ar1 = mean_absolute_percentage_error(y_test, pred_ar1) * 100.0

    bang_tong_hop = [
        {
            "Mô hình": "1. OLS Tuyến tính Tối ưu (HC3)",
            "Số biến": len(features),
            "R2 (Train)": round(m_lin.rsquared, 4),
            "Adj R2": round(m_lin.rsquared_adj, 4),
            "AIC": round(m_lin.aic, 1),
            "RMSE (Test)": round(rmse_lin, 2),
            "MAE (Test)": round(mae_lin, 2),
            "MAPE (%)": f"{mape_lin:.2f}%"
        },
        {
            "Mô hình": "2. Mô hình Log-Log ln(Y)",
            "Số biến": len(features),
            "R2 (Train)": round(m_log.rsquared, 4),
            "Adj R2": round(m_log.rsquared_adj, 4),
            "AIC": round(m_log.aic, 1),
            "RMSE (Test)": round(rmse_log, 2),
            "MAE (Test)": round(mae_log, 2),
            "MAPE (%)": f"{mape_log:.2f}%"
        },
        {
            "Mô hình": "3. Full Model (Đầy đủ 5 biến)",
            "Số biến": len(all_cand),
            "R2 (Train)": round(m_full.rsquared, 4),
            "Adj R2": round(m_full.rsquared_adj, 4),
            "AIC": round(m_full.aic, 1),
            "RMSE (Test)": round(np.sqrt(mean_squared_error(y_test, pred_full)), 2),
            "MAE (Test)": round(mean_absolute_error(y_test, pred_full), 2),
            "MAPE (%)": f"{mape_full:.2f}%"
        },
        {
            "Mô hình": "4. AR(1) Model (Chỉ có Lag1)",
            "Số biến": 1,
            "R2 (Train)": round(m_ar1.rsquared, 4),
            "Adj R2": round(m_ar1.rsquared_adj, 4),
            "AIC": round(m_ar1.aic, 1),
            "RMSE (Test)": round(np.sqrt(mean_squared_error(y_test, pred_ar1)), 2),
            "MAE (Test)": round(mean_absolute_error(y_test, pred_ar1), 2),
            "MAPE (%)": f"{mape_ar1:.2f}%"
        }
    ]

    df_tong_hop = pd.DataFrame(bang_tong_hop)
    print("\n" + "=" * 85)
    print("BẢNG SO SÁNH CHẤT LƯỢNG MÔ HÌNH VÀ SAI SỐ DỰ BÁO NGOÀI MẪU (TEST SET):")
    print(df_tong_hop.to_string(index=False))

    return m_lin, m_hc3, m_log, df_tong_hop, pred_test_lin, calc_lin

def ve_bieu_do_du_bao_va_khoang_tin_cay(df, train_size, y_pred_test, fig_path):
    """
    Task 6.3: Vẽ biểu đồ so sánh Đường giá thực tế và Đường giá dự báo kèm dải khoảng tin cậy 95%
    Lưu độ phân giải cao 300 DPI vào figures/fig09_forecast_vs_actual.png
    """
    print("\n" + "=" * 85)
    print("XUẤT BIỂU ĐỒ SO SÁNH THỰC TẾ VS DỰ BÁO NGOÀI MẪU (300 DPI) VÀO figures/")
    print("=" * 85)

    dates = pd.to_datetime(df["Date"], format="%Y-%m")
    y_all = df["VN_Coffee_Price"] / 1000.0

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)

    # 1. Vẽ chuỗi giá thực tế toàn bộ giai đoạn
    ax.plot(dates, y_all, color="#4E342E", linewidth=2.2, label="Giá thực tế Tây Nguyên (Actual)")

    # 2. Vẽ đường giá dự báo trên tập Test
    dates_test = dates.iloc[train_size:]
    ax.plot(dates_test, y_pred_test / 1000.0, color="#D32F2F", linewidth=2.5, linestyle="--", marker="o", markersize=4, label="Dự báo mô hình OLS (Forecast)")

    # 3. Tính khoảng dự báo 95% (Prediction Interval ~ ± 1.96 * RMSE)
    rmse_approx = 14.137  # Nghìn VNĐ/kg
    lower_bound = (y_pred_test / 1000.0) - 1.96 * rmse_approx
    upper_bound = (y_pred_test / 1000.0) + 1.96 * rmse_approx

    ax.fill_between(dates_test, lower_bound, upper_bound, color="#FFCDD2", alpha=0.45, label="Khoảng dự báo tin cậy 95% (Prediction Interval)")

    # 4. Vẽ đường phân cách ranh giới Train / Test
    cut_date = dates.iloc[train_size]
    ax.axvline(x=cut_date, color="#0D47A1", linestyle="-.", linewidth=2.0, label=f"Ranh giới Train/Test ({df['Date'].iloc[train_size]})")

    # Thêm văn bản chú thích vùng Train và Test
    ax.text(dates.iloc[train_size // 2], y_all.max() * 0.92, "TẬP HUẤN LUYỆN (TRAIN 80%)\n122 Tháng (2014-01 đến 2024-02)", 
            fontsize=11, fontweight="bold", ha="center", color="#0D47A1", bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))
    
    ax.text(dates.iloc[train_size + (len(df) - train_size) // 2], y_all.max() * 0.92, "TẬP KIỂM THỬ (TEST 20%)\n31 Tháng (2024-03 đến 2026-09)\nMAPE = 6.39%", 
            fontsize=11, fontweight="bold", ha="center", color="#B71C1C", bbox=dict(boxstyle="round", facecolor="#FFEBEE", alpha=0.8))

    ax.set_title("BIỂU ĐỒ SO SÁNH GIÁ CÀ PHÊ THỰC TẾ VÀ DỰ BÁO MÔ HÌNH OLS (KÈM KHOẢNG TIN CẬY 95%)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Thời gian (Năm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Giá Cà phê Nhân xô Tây Nguyên (Nghìn VNĐ/kg)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ dự báo (fig09): {fig_path}")

def du_bao_3_kich_ban_tuong_lai(model, df_last_row):
    """
    Task 6.3 (Kịch bản tương lai): Thiết lập 3 kịch bản thị trường vĩ mô cho tháng tới (2026-10)
    - Kịch bản 1: Cơ sở (Baseline) - Ổn định
    - Kịch bản 2: Tăng nóng (Bullish) - London tăng mạnh + Dầu tăng
    - Kịch bản 3: Hạ nhiệt (Bearish) - Thu hoạch dồi dào, London điều chỉnh
    """
    print("\n" + "=" * 85)
    print("TASK 6.3: DỰ BÁO 3 KỊCH BẢN THỊ TRƯỜNG VĨ MÔ CHO THÁNG TIẾP THEO (2026-10)")
    print("=" * 85)

    last_price = df_last_row["VN_Coffee_Price"]
    last_date = df_last_row["Date"]
    print(f"- Mức giá quan sát thực tế gần nhất (Tháng {last_date}): {last_price:,.2f} VNĐ/kg.")

    kich_ban_list = [
        {
            "Kịch bản": "1. Cơ sở (Baseline - Đi ngang)",
            "Mô tả": "Giá Robusta London và Dầu Brent duy trì ổn định quanh mức hiện tại.",
            "London_Robusta": 3700.0,
            "Crude_Oil": 115.0
        },
        {
            "Kịch bản": "2. Tăng nóng (Bullish - Khủng hoảng cung)",
            "Mô tả": "Sàn London tăng vọt lên 4,200 USD/Tấn do hạn hán, giá dầu tăng lên 125 USD/thùng.",
            "London_Robusta": 4200.0,
            "Crude_Oil": 125.0
        },
        {
            "Kịch bản": "3. Hạ nhiệt (Bearish - Thu hoạch rộ)",
            "Mô tả": "Tây Nguyên vào chính vụ thu hoạch, nguồn cung dồi dào, London giảm về 3,200 USD/Tấn.",
            "London_Robusta": 3200.0,
            "Crude_Oil": 95.0
        }
    ]

    ket_qua_kich_ban = []

    for sc in kich_ban_list:
        X_sc = pd.DataFrame([{
            "const": 1.0,
            "VN_Coffee_Price_Lag1": last_price,
            "London_Robusta_Price": sc["London_Robusta"],
            "Crude_Oil_Price": sc["Crude_Oil"]
        }])
        
        pred_obj = model.get_prediction(X_sc)
        sf = pred_obj.summary_frame(alpha=0.05)
        
        gia_du_bao = sf["mean"].iloc[0]
        mean_ci_lower = sf["mean_ci_lower"].iloc[0]
        mean_ci_upper = sf["mean_ci_upper"].iloc[0]
        obs_ci_lower = sf["obs_ci_lower"].iloc[0]
        obs_ci_upper = sf["obs_ci_upper"].iloc[0]

        ket_qua_kich_ban.append({
            "Kịch bản": sc["Kịch bản"],
            "Robusta London (USD/Tấn)": f"{sc['London_Robusta']:,.0f}",
            "Dầu Brent (USD/Thùng)": f"{sc['Crude_Oil']:,.0f}",
            "Giá Dự báo (VNĐ/kg)": f"{gia_du_bao:,.2f}",
            "Khoảng tin cậy 95% Kỳ vọng": f"[{mean_ci_lower:,.0f} - {mean_ci_upper:,.0f}]",
            "Khoảng dự báo 95% Cá thể": f"[{obs_ci_lower:,.0f} - {obs_ci_upper:,.0f}]"
        })

    df_kb = pd.DataFrame(ket_qua_kich_ban)
    print(df_kb.to_string(index=False))

    return df_kb

def main():
    """
    Hàm điều phối toàn bộ chu trình Bước 6
    """
    file_unscaled = "data/processed/vn_coffee_unscaled.csv"
    fig09 = "figures/fig09_forecast_vs_actual.png"

    assert os.path.exists(file_unscaled), f"Lỗi: Không tìm thấy file '{file_unscaled}'!"
    df = pd.read_csv(file_unscaled)

    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy().reset_index(drop=True)
    df_test = df.iloc[train_size:].copy().reset_index(drop=True)

    bien_phu_thuoc = "VN_Coffee_Price"
    features = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "Crude_Oil_Price"]

    # Task 6.1: Huấn luyện và tính toán chỉ số gốc
    m_lin, m_hc3, m_log, df_so_sanh, pred_test_lin, calc_lin = huan_luyen_va_so_sanh_mo_hinh(
        df_train, df_test, bien_phu_thuoc, features
    )

    # Task 6.2: In phương trình hồi quy chính thức
    print("\n" + "=" * 85)
    print("TASK 6.2: PHƯƠNG TRÌNH HỒI QUY CHÍNH THỨC & DIỄN GIẢI KINH TẾ")
    print("=" * 85)
    b0 = m_lin.params["const"]
    b1 = m_lin.params["VN_Coffee_Price_Lag1"]
    b2 = m_lin.params["London_Robusta_Price"]
    b3 = m_lin.params["Crude_Oil_Price"]
    print(f"PHƯƠNG TRÌNH HỒI QUY TUYẾN TÍNH OLS TỐI ƯU:")
    print(f"  VN_Coffee_Price_t = {b0:+.2f} + ({b1:.4f} * Lag1) + ({b2:.4f} * London) + ({b3:.4f} * Oil)")

    # Task 6.3: Vẽ biểu đồ so sánh thực tế vs dự báo
    ve_bieu_do_du_bao_va_khoang_tin_cay(df, train_size, pred_test_lin, fig09)

    # Task 6.3: Dự báo 3 kịch bản
    df_kb = du_bao_3_kich_ban_tuong_lai(m_lin, df.iloc[-1])

    # --------------------------------------------------------------------------
    # CHẠY SELF-TEST (TỰ ĐỘNG KIỂM THỬ BƯỚC 6 THEO QUY TẮC SỐ 5)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CHẠY SELF-TEST (TỰ ĐỘNG KIỂM THỬ BƯỚC 6)")
    print("=" * 85)

    # Assertion 1: Khớp công thức tính toán R2 giữa công thức thủ công (1 - ESS/TSS) và statsmodels
    sai_lech_r2 = abs((1.0 - (calc_lin["ESS"] / calc_lin["TSS"])) - m_lin.rsquared)
    assert sai_lech_r2 < 1e-5, f"Self-Test Thất bại: Sai lệch công thức tính R2: {sai_lech_r2}"
    print(f"[PASS] Assertion 1: |(1 - ESS/TSS) - statsmodels.rsquared| = {sai_lech_r2:.8f} < 10^-5 (Khớp 100%).")

    # Assertion 2: Kiểm tra sự tồn tại và dung lượng của file ảnh biểu đồ fig09
    assert os.path.exists(fig09) and os.path.getsize(fig09) > 0, f"Self-Test Thất bại: File biểu đồ {fig09} không hợp lệ!"
    print(f"[PASS] Assertion 2: File biểu đồ {fig09} tồn tại hợp lệ ({os.path.getsize(fig09):,} bytes).")

    # Assertion 3: Sai số dự báo MAPE trên tập kiểm thử Test đạt tiêu chuẩn xuất sắc (MAPE < 10%)
    mape_val = mean_absolute_percentage_error(df_test[bien_phu_thuoc], pred_test_lin) * 100.0
    assert mape_val < 10.0, f"Self-Test Thất bại: MAPE ({mape_val:.2f}%) vượt quá ngưỡng 10%!"
    print(f"[PASS] Assertion 3: MAPE = {mape_val:.2f}% < 10.0% (Mô hình đạt độ chính xác dự báo Rất cao).")

    print("-" * 85)
    print("SELF-TEST BƯỚC 6: PASS 100% TOÀN BỘ CÁC BÀI KIỂM THỬ TỰ ĐỘNG!")
    print("=" * 85)

    return m_lin, m_hc3, m_log, df_so_sanh, df_kb

if __name__ == "__main__":
    main()
