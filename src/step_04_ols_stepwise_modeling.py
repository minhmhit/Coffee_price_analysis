import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def chia_tap_du_lieu_chuoi_thoi_gian(df, ty_le_train=0.8):
    print("=" * 85)
    print("TASK 4.1: CHIA TẬP DỮ LIỆU HUẤN LUYỆN (TRAIN 80%) & KIỂM THỬ (TEST 20%)")
    print("=" * 85)

    tong_so_mau = len(df)
    diem_cat = int(tong_so_mau * ty_le_train)
    df_train = df.iloc[:diem_cat].copy().reset_index(drop=True)
    df_test = df.iloc[diem_cat:].copy().reset_index(drop=True)

    print(f"- Train Set: {len(df_train)} tháng ({df_train['Date'].iloc[0]} -> {df_train['Date'].iloc[-1]})")
    print(f"- Test Set : {len(df_test)} tháng ({df_test['Date'].iloc[0]} -> {df_test['Date'].iloc[-1]})")
    return df_train, df_test

def uoc_luong_mo_hinh_day_du(df_train, cac_bien_doc_lap, bien_phu_thuoc):
    print("\n" + "=" * 85)
    print("TASK 4.2: XÂY DỰNG MÔ HÌNH HỒI QUY ĐẦY ĐỦ (FULL MULTIPLE REGRESSION MODEL)")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    X_train = sm.add_constant(df_train[cac_bien_doc_lap])
    model_full = sm.OLS(y_train, X_train).fit()
    print(model_full.summary())
    return model_full

def thuat_toan_backward_elimination(df_train, cac_bien_ung_vien, bien_phu_thuoc, nguong_p=0.05):
    print("\n" + "=" * 85)
    print("TASK 4.3: THỰC THI THUẬT TOÁN LỰA CHỌN BIẾN LÙI (BACKWARD ELIMINATION)")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    bien_hien_tai = list(cac_bien_ung_vien)
    nhat_ky_vong_lap = []
    vong_lap = 1

    while True:
        X_curr = sm.add_constant(df_train[bien_hien_tai])
        model_curr = sm.OLS(y_train, X_curr).fit()

        p_values = model_curr.pvalues.drop("const")
        max_p = p_values.max()
        bien_kem_nhat = p_values.idxmax()

        thong_tin_vong = {
            "Vòng": vong_lap,
            "Số biến": len(bien_hien_tai),
            "Adj R-squared": round(model_curr.rsquared_adj, 4),
            "AIC": round(model_curr.aic, 2),
            "Biến có p lớn nhất": bien_kem_nhat,
            "Max p-value": round(max_p, 4)
        }
        nhat_ky_vong_lap.append(thong_tin_vong)

        print(f"[VÒNG {vong_lap}] Số biến: {len(bien_hien_tai)} | Adj-R2 = {model_curr.rsquared_adj:.4f} | AIC = {model_curr.aic:.2f}")
        print(f"  + Max p-value: '{bien_kem_nhat}' (p = {max_p:.4f})")

        if max_p > nguong_p:
            print(f"  -> LOẠI BỎ: '{bien_kem_nhat}' vì p > {nguong_p}")
            bien_hien_tai.remove(bien_kem_nhat)
            vong_lap += 1
        else:
            print(f"  -> DỪNG THUẬT TOÁN: Tất cả biến có p <= {nguong_p}!")
            mo_hinh_toi_uu = model_curr
            cac_bien_toi_uu = list(bien_hien_tai)
            break

    df_nhat_ky = pd.DataFrame(nhat_ky_vong_lap)
    print("\nBẢNG NHẬT KÝ LỰA CHỌN BIẾN:")
    print(df_nhat_ky.to_string(index=False))
    print("\nMÔ HÌNH TỐI ƯU CUỐI CÙNG:")
    print(mo_hinh_toi_uu.summary())
    return mo_hinh_toi_uu, cac_bien_toi_uu, df_nhat_ky

def so_sanh_cac_mo_hinh(df_train, bien_phu_thuoc):
    print("\n" + "=" * 85)
    print("TASK 4.4: SO SÁNH HIỆU QUẢ CÁC MÔ HÌNH HỒI QUY (MODEL BENCHMARKING)")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]

    # 1. Macro Only
    bien_macro = ["London_Robusta_Price", "VN_Production", "Crude_Oil_Price", "Harvest_Season_In_Harvest"]
    X_m1 = sm.add_constant(df_train[bien_macro])
    m1 = sm.OLS(y_train, X_m1).fit()

    # 2. AR(1) Only
    bien_ar = ["VN_Coffee_Price_Lag1"]
    X_m2 = sm.add_constant(df_train[bien_ar])
    m2 = sm.OLS(y_train, X_m2).fit()

    # 3. Stepwise Optimal
    bien_stepwise = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "Crude_Oil_Price"]
    X_m3 = sm.add_constant(df_train[bien_stepwise])
    m3 = sm.OLS(y_train, X_m3).fit()

    danh_sach_so_sanh = [
        {"Mô hình": "1. Macro Only", "Số biến": len(bien_macro), "R2": round(m1.rsquared, 4), "Adj-R2": round(m1.rsquared_adj, 4), "AIC": round(m1.aic, 2), "Durbin-Watson": round(sm.stats.stattools.durbin_watson(m1.resid), 3)},
        {"Mô hình": "2. AR(1) Only", "Số biến": len(bien_ar), "R2": round(m2.rsquared, 4), "Adj-R2": round(m2.rsquared_adj, 4), "AIC": round(m2.aic, 2), "Durbin-Watson": round(sm.stats.stattools.durbin_watson(m2.resid), 3)},
        {"Mô hình": "3. Stepwise Optimal", "Số biến": len(bien_stepwise), "R2": round(m3.rsquared, 4), "Adj-R2": round(m3.rsquared_adj, 4), "AIC": round(m3.aic, 2), "Durbin-Watson": round(sm.stats.stattools.durbin_watson(m3.resid), 3)}
    ]
    df_so_sanh = pd.DataFrame(danh_sach_so_sanh)
    print(df_so_sanh.to_string(index=False))
    return m1, m2, m3, df_so_sanh

def trich_xuat_bang_he_so(mo_hinh):
    return pd.DataFrame({
        "Beta": mo_hinh.params,
        "Std.Err": mo_hinh.bse,
        "t-stat": mo_hinh.tvalues,
        "p-value": mo_hinh.pvalues,
        "CI 2.5%": mo_hinh.conf_int()[0],
        "CI 97.5%": mo_hinh.conf_int()[1]
    })

def main():
    file_unscaled = "data/processed/vn_coffee_unscaled.csv"
    file_scaled = "data/processed/vn_coffee_scaled.csv"

    assert os.path.exists(file_unscaled), f"Lỗi: Không tìm thấy '{file_unscaled}'!"
    df = pd.read_csv(file_unscaled)

    bien_phu_thuoc = "VN_Coffee_Price"
    cac_bien_ung_vien = [
        "VN_Coffee_Price_Lag1",
        "London_Robusta_Price",
        "VN_Production",
        "Crude_Oil_Price",
        "Harvest_Season_In_Harvest"
    ]

    df_train, df_test = chia_tap_du_lieu_chuoi_thoi_gian(df, ty_le_train=0.8)
    model_full = uoc_luong_mo_hinh_day_du(df_train, cac_bien_ung_vien, bien_phu_thuoc)
    mo_hinh_toi_uu, cac_bien_toi_uu, df_nhat_ky = thuat_toan_backward_elimination(df_train, cac_bien_ung_vien, bien_phu_thuoc)
    m_macro, m_ar, m_opt, df_so_sanh = so_sanh_cac_mo_hinh(df_train, bien_phu_thuoc)

    df_sc = pd.read_csv(file_scaled)
    df_train_sc = df_sc.iloc[:len(df_train)].copy().reset_index(drop=True)
    X_train_sc = sm.add_constant(df_train_sc[cac_bien_toi_uu])
    y_train_sc = df_train_sc[bien_phu_thuoc]
    model_scaled_opt = sm.OLS(y_train_sc, X_train_sc).fit()

    print("\nHỆ SỐ HỒI QUY CHUẨN HÓA (STANDARDIZED BETA):")
    for col in cac_bien_toi_uu:
        print(f"- {col:<25}: Beta* = {model_scaled_opt.params[col]:+10.4f} | p = {model_scaled_opt.pvalues[col]:.4e}")

    # Self-Test
    p_values_final = mo_hinh_toi_uu.pvalues.drop("const")
    assert (p_values_final < 0.05).all(), f"Lỗi: Có biến p >= 0.05: {p_values_final}"
    assert mo_hinh_toi_uu.f_pvalue < 0.05, "Lỗi: F-statistic không có ý nghĩa"
    assert mo_hinh_toi_uu.rsquared >= 0.80, "Lỗi: R2 < 0.80"
    assert len(df_train) == 122 and len(df_test) == 31, "Lỗi kích thước Train/Test"

    print("\nSELF-TEST BƯỚC 4: PASS 100%!")
    return mo_hinh_toi_uu, df_nhat_ky, df_so_sanh, model_scaled_opt

if __name__ == "__main__":
    main()
