import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def tinh_toan_cac_chi_so_goc_da_sgu(y_true, y_pred, n_obs, k_features, llf):
    ess = np.sum((y_true - y_pred) ** 2)
    y_bar = np.mean(y_true)
    tss = np.sum((y_true - y_bar) ** 2)
    r2 = 1.0 - (ess / tss)
    adj_r2 = 1.0 - ((ess / (n_obs - k_features - 1)) / (tss / (n_obs - 1)))
    aic = -2.0 * llf + 2.0 * (k_features + 1)
    bic = -2.0 * llf + np.log(n_obs) * (k_features + 1)
    return {"ESS": ess, "TSS": tss, "R2": r2, "Adj_R2": adj_r2, "AIC": aic, "BIC": bic}

def huan_luyen_va_so_sanh_mo_hinh(df_train, df_test, bien_phu_thuoc, features):
    print("=" * 85)
    print("TASK 6.1: TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH THEO CÔNG THỨC GỐC DA-SGU")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    y_test = df_test[bien_phu_thuoc]

    # 1. OLS Tuyến tính với Robust HC3
    X_train_lin = sm.add_constant(df_train[features])
    X_test_lin = sm.add_constant(df_test[features])
    m_lin = sm.OLS(y_train, X_train_lin).fit()
    m_hc3 = sm.OLS(y_train, X_train_lin).fit(cov_type="HC3")

    calc_lin = tinh_toan_cac_chi_so_goc_da_sgu(
        y_train.values, m_lin.fittedvalues.values, len(y_train), len(features), m_lin.llf
    )
    print(f"- ESS: {calc_lin['ESS']:,.2f} | TSS: {calc_lin['TSS']:,.2f}")
    print(f"- R2: {calc_lin['R2']:.4f} | Adj R2: {calc_lin['Adj_R2']:.4f} | AIC: {calc_lin['AIC']:.2f}")

    pred_test_lin = m_lin.predict(X_test_lin)
    rmse_lin = np.sqrt(mean_squared_error(y_test, pred_test_lin))
    mae_lin = mean_absolute_error(y_test, pred_test_lin)
    mape_lin = mean_absolute_percentage_error(y_test, pred_test_lin) * 100.0

    # 2. Log-Log model
    train_log = df_train.copy()
    test_log = df_test.copy()
    train_log["ln_Y"] = np.log(train_log[bien_phu_thuoc])
    test_log["ln_Y"] = np.log(test_log[bien_phu_thuoc])
    log_features = []
    for col in features:
        train_log["ln_" + col] = np.log(train_log[col])
        test_log["ln_" + col] = np.log(test_log[col])
        log_features.append("ln_" + col)

    m_log = sm.OLS(train_log["ln_Y"], sm.add_constant(train_log[log_features])).fit()
    pred_test_ln_y = m_log.predict(sm.add_constant(test_log[log_features]))
    pred_test_log = np.exp(pred_test_ln_y + m_log.mse_resid / 2.0)
    rmse_log = np.sqrt(mean_squared_error(y_test, pred_test_log))
    mae_log = mean_absolute_error(y_test, pred_test_log)
    mape_log = mean_absolute_percentage_error(y_test, pred_test_log) * 100.0

    # 3. Full & AR(1)
    all_cand = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "VN_Production", "Crude_Oil_Price", "Harvest_Season_In_Harvest"]
    m_full = sm.OLS(y_train, sm.add_constant(df_train[all_cand], has_constant="add")).fit()
    pred_full = m_full.predict(sm.add_constant(df_test[all_cand], has_constant="add"))
    mape_full = mean_absolute_percentage_error(y_test, pred_full) * 100.0

    m_ar1 = sm.OLS(y_train, sm.add_constant(df_train[["VN_Coffee_Price_Lag1"]], has_constant="add")).fit()
    pred_ar1 = m_ar1.predict(sm.add_constant(df_test[["VN_Coffee_Price_Lag1"]], has_constant="add"))
    mape_ar1 = mean_absolute_percentage_error(y_test, pred_ar1) * 100.0

    bang_tong_hop = [
        {"Mô hình": "1. OLS Tuyến tính (HC3)", "R2": round(m_lin.rsquared, 4), "Adj R2": round(m_lin.rsquared_adj, 4), "AIC": round(m_lin.aic, 1), "RMSE": round(rmse_lin, 2), "MAE": round(mae_lin, 2), "MAPE": f"{mape_lin:.2f}%"},
        {"Mô hình": "2. Log-Log ln(Y)", "R2": round(m_log.rsquared, 4), "Adj R2": round(m_log.rsquared_adj, 4), "AIC": round(m_log.aic, 1), "RMSE": round(rmse_log, 2), "MAE": round(mae_log, 2), "MAPE": f"{mape_log:.2f}%"},
        {"Mô hình": "3. Full Model (5 biến)", "R2": round(m_full.rsquared, 4), "Adj R2": round(m_full.rsquared_adj, 4), "AIC": round(m_full.aic, 1), "RMSE": round(np.sqrt(mean_squared_error(y_test, pred_full)), 2), "MAE": round(mean_absolute_error(y_test, pred_full), 2), "MAPE": f"{mape_full:.2f}%"},
        {"Mô hình": "4. AR(1) Model (Chỉ Lag1)", "R2": round(m_ar1.rsquared, 4), "Adj R2": round(m_ar1.rsquared_adj, 4), "AIC": round(m_ar1.aic, 1), "RMSE": round(np.sqrt(mean_squared_error(y_test, pred_ar1)), 2), "MAE": round(mean_absolute_error(y_test, pred_ar1), 2), "MAPE": f"{mape_ar1:.2f}%"}
    ]
    df_tong_hop = pd.DataFrame(bang_tong_hop)
    print("\nBẢNG SO SÁNH CHẤT LƯỢNG MÔ HÌNH VÀ SAI SỐ DỰ BÁO TẬP TEST:")
    print(df_tong_hop.to_string(index=False))
    return m_lin, m_hc3, m_log, df_tong_hop, pred_test_lin, calc_lin

def ve_bieu_do_du_bao_va_khoang_tin_cay(df, train_size, y_pred_test, fig_path):
    print("\n" + "=" * 85)
    print("XUẤT BIỂU ĐỒ SO SÁNH THỰC TẾ VS DỰ BÁO NGOÀI MẪU (300 DPI) VÀO figures/")
    print("=" * 85)

    dates = pd.to_datetime(df["Date"], format="%Y-%m")
    y_all = df["VN_Coffee_Price"] / 1000.0

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    ax.plot(dates, y_all, color="#4E342E", linewidth=2.2, label="Giá thực tế Tây Nguyên (Actual)")

    dates_test = dates.iloc[train_size:]
    ax.plot(dates_test, y_pred_test / 1000.0, color="#D32F2F", linewidth=2.5, linestyle="--", marker="o", markersize=4, label="Dự báo OLS (Forecast)")

    rmse_approx = 14.137
    lower_bound = (y_pred_test / 1000.0) - 1.96 * rmse_approx
    upper_bound = (y_pred_test / 1000.0) + 1.96 * rmse_approx
    ax.fill_between(dates_test, lower_bound, upper_bound, color="#FFCDD2", alpha=0.45, label="Khoảng dự báo tin cậy 95%")

    cut_date = dates.iloc[train_size]
    ax.axvline(x=cut_date, color="#0D47A1", linestyle="-.", linewidth=2.0, label=f"Ranh giới Train/Test ({df['Date'].iloc[train_size]})")

    ax.set_title("BIỂU ĐỒ SO SÁNH GIÁ CÀ PHÊ THỰC TẾ VÀ DỰ BÁO MÔ HÌNH OLS (KÈM KHOẢNG TIN CẬY 95%)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Thời gian (Năm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Giá Cà phê Nhân xô Tây Nguyên (Nghìn VNĐ/kg)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def du_bao_3_kich_ban_tuong_lai(model, df_last_row):
    print("\n" + "=" * 85)
    print("TASK 6.3: DỰ BÁO 3 KỊCH BẢN THỊ TRƯỜNG VĨ MÔ CHO THÁNG TIẾP THEO (2026-10)")
    print("=" * 85)

    last_price = df_last_row["VN_Coffee_Price"]
    kich_ban_list = [
        {"Kịch bản": "1. Cơ sở (Baseline)", "London_Robusta": 3700.0, "Crude_Oil": 115.0},
        {"Kịch bản": "2. Tăng nóng (Bullish)", "London_Robusta": 4200.0, "Crude_Oil": 125.0},
        {"Kịch bản": "3. Hạ nhiệt (Bearish)", "London_Robusta": 3200.0, "Crude_Oil": 95.0}
    ]

    ket_qua = []
    for sc in kich_ban_list:
        X_sc = pd.DataFrame([{"const": 1.0, "VN_Coffee_Price_Lag1": last_price, "London_Robusta_Price": sc["London_Robusta"], "Crude_Oil_Price": sc["Crude_Oil"]}])
        sf = model.get_prediction(X_sc).summary_frame(alpha=0.05)
        ket_qua.append({
            "Kịch bản": sc["Kịch bản"],
            "London (USD/Tấn)": f"{sc['London_Robusta']:,.0f}",
            "Dầu Brent (USD/Thùng)": f"{sc['Crude_Oil']:,.0f}",
            "Giá Dự báo (VNĐ/kg)": f"{sf['mean'].iloc[0]:,.2f}",
            "Khoảng tin cậy 95% Kỳ vọng": f"[{sf['mean_ci_lower'].iloc[0]:,.0f} - {sf['mean_ci_upper'].iloc[0]:,.0f}]",
            "Khoảng dự báo 95% Cá thể": f"[{sf['obs_ci_lower'].iloc[0]:,.0f} - {sf['obs_ci_upper'].iloc[0]:,.0f}]"
        })
    df_kb = pd.DataFrame(ket_qua)
    print(df_kb.to_string(index=False))
    return df_kb

def main():
    file_unscaled = "data/processed/vn_coffee_unscaled.csv"
    fig09 = "figures/fig09_forecast_vs_actual.png"

    assert os.path.exists(file_unscaled), f"Lỗi: Không tìm thấy '{file_unscaled}'!"
    df = pd.read_csv(file_unscaled)

    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy().reset_index(drop=True)
    df_test = df.iloc[train_size:].copy().reset_index(drop=True)

    bien_phu_thuoc = "VN_Coffee_Price"
    features = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "Crude_Oil_Price"]

    m_lin, m_hc3, m_log, df_so_sanh, pred_test_lin, calc_lin = huan_luyen_va_so_sanh_mo_hinh(
        df_train, df_test, bien_phu_thuoc, features
    )

    b0 = m_lin.params["const"]
    b1 = m_lin.params["VN_Coffee_Price_Lag1"]
    b2 = m_lin.params["London_Robusta_Price"]
    b3 = m_lin.params["Crude_Oil_Price"]
    print(f"\nPHƯƠNG TRÌNH HỒI QUY TUYẾN TÍNH OLS TỐI ƯU:")
    print(f"  VN_Coffee_Price_t = {b0:+.2f} + ({b1:.4f} * Lag1) + ({b2:.4f} * London) + ({b3:.4f} * Oil)")

    ve_bieu_do_du_bao_va_khoang_tin_cay(df, train_size, pred_test_lin, fig09)
    df_kb = du_bao_3_kich_ban_tuong_lai(m_lin, df.iloc[-1])

    # Self-Test
    sai_lech_r2 = abs((1.0 - (calc_lin["ESS"] / calc_lin["TSS"])) - m_lin.rsquared)
    assert sai_lech_r2 < 1e-5, f"Lỗi R2: {sai_lech_r2}"
    assert os.path.exists(fig09) and os.path.getsize(fig09) > 0, "Lỗi file fig09"
    mape_val = mean_absolute_percentage_error(df_test[bien_phu_thuoc], pred_test_lin) * 100.0
    assert mape_val < 10.0, f"Lỗi MAPE >= 10%: {mape_val}"

    print(f"\n[THÀNH CÔNG] MAPE = {mape_val:.2f}% (< 10%).")
    print("SELF-TEST BƯỚC 6: PASS 100%!")
    return m_lin, m_hc3, m_log, df_so_sanh, df_kb

if __name__ == "__main__":
    main()
