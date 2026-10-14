# ==============================================================================
# TẬP TIN: src/step_04_ols_stepwise_modeling.py
# MỤC ĐÍCH: Huấn luyện mô hình Hồi quy Tuyến tính OLS và Lựa chọn biến Stepwise (KDD Bước 5)
# QUY CHUẨN: Giáo trình Phân tích Dữ liệu DA - SGU (Đại học Sài Gòn)
# KỸ THUẬT: Train/Test Split (Time Series), Full Model, Backward Elimination, Stepwise, AR(1)
# GIẢI THÍCH: Chú thích từng dòng lệnh bằng tiếng Việt dành cho Newbie Data Analyst
# ==============================================================================

# Import thư viện sys để cấu hình hiển thị mã hóa tiếng Việt UTF-8 trên Windows console
import sys

# Import thư viện os để quản lý đường dẫn tập tin và thư mục
import os

# Import thư viện pandas để thao tác với bảng dữ liệu DataFrame và trích xuất chỉ số
import pandas as pd

# Import thư viện numpy để tính toán ma trận và các giá trị số học
import numpy as np

# Import thư viện statsmodels.api để ước lượng mô hình hồi quy OLS kinh tế lượng chuyên sâu
import statsmodels.api as sm

# Thiết lập bảng mã UTF-8 cho luồng xuất console tránh lỗi ký tự tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def chia_tap_du_lieu_chuoi_thoi_gian(df, ty_le_train=0.8):
    """
    Task 4.1: Chia tập dữ liệu thành Tập huấn luyện (Train Set - 80%) và Tập kiểm thử (Test Set - 20%).
    QUY TẮC BẮT BUỘC CHO CHUỖI THỜI GIAN:
    Tuyệt đối KHÔNG dùng train_test_split(shuffle=True) vì sẽ làm xáo trộn thứ tự thời gian,
    gây ra hiện tượng Rò rỉ dữ liệu tương lai (Data Leakage / Look-ahead Bias).
    Bắt buộc phải cắt theo thứ tự thời gian tuyến tính (Chronological Split).
    """
    print("=" * 85)
    print("TASK 4.1: CHIA TẬP DỮ LIỆU HUẤN LUYỆN (TRAIN 80%) & KIỂM THỬ (TEST 20%)")
    print("=" * 85)

    tong_so_mau = len(df)
    diem_cat = int(tong_so_mau * ty_le_train)

    # 80% quan sát đầu tiên (từ quá khứ đến mốc cắt) dùng để Train mô hình
    df_train = df.iloc[:diem_cat].copy().reset_index(drop=True)
    # 20% quan sát gần nhất (tương lai) dùng để Test và đánh giá dự báo
    df_test = df.iloc[diem_cat:].copy().reset_index(drop=True)

    print(f"- Tổng số quan sát: {tong_so_mau} tháng.")
    print(f"- Tập huấn luyện (Train Set): {len(df_train)} tháng ({len(df_train)/tong_so_mau*100:.1f}%) | "
          f"Từ {df_train['Date'].iloc[0]} đến {df_train['Date'].iloc[-1]}")
    print(f"- Tập kiểm thử (Test Set)  : {len(df_test)} tháng ({len(df_test)/tong_so_mau*100:.1f}%) | "
          f"Từ {df_test['Date'].iloc[0]} đến {df_test['Date'].iloc[-1]}")

    return df_train, df_test

def uoc_luong_mo_hinh_day_du(df_train, cac_bien_doc_lap, bien_phu_thuoc):
    """
    Task 4.2: Xây dựng Mô hình Hồi quy Tuyến tính Đa biến Đầy đủ (Full Model)
    Y = beta_0 + beta_1*X1 + beta_3*X3 + beta_6*X6 + beta_7*X7 + beta_Lag*Y_{t-1} + epsilon
    Sử dụng statsmodels.api.OLS và bắt buộc thêm hệ số chặn beta_0 bằng sm.add_constant().
    """
    print("\n" + "=" * 85)
    print("TASK 4.2: XÂY DỰNG MÔ HÌNH HỒI QUY ĐẦY ĐỦ (FULL MULTIPLE REGRESSION MODEL)")
    print("=" * 85)

    # Tách biến phụ thuộc Y
    y_train = df_train[bien_phu_thuoc]
    
    # Tách ma trận biến độc lập X và bổ sung cột hằng số 1 đại diện cho hệ số chặn beta_0
    X_train = sm.add_constant(df_train[cac_bien_doc_lap])

    # Khởi tạo mô hình OLS và khớp dữ liệu (fit)
    model_full = sm.OLS(y_train, X_train).fit()

    print(model_full.summary())
    return model_full

def thuat_toan_backward_elimination(df_train, cac_bien_ung_vien, bien_phu_thuoc, nguong_p=0.05):
    """
    Task 4.3: Thực thi thuật toán Lựa chọn Biến Backward Elimination chuẩn giáo trình DA - SGU:
    - Bắt đầu với tất cả các biến độc lập ứng viên.
    - Tại mỗi vòng lặp:
      1. Khớp mô hình OLS.
      2. Tìm biến có p-value lớn nhất (hệ số t-test kém ý nghĩa nhất).
      3. Nếu max(p-value) > 0.05: Loại biến đó, ghi nhật ký và lặp lại.
      4. Nếu max(p-value) <= 0.05: Dừng lại và kết luận mô hình tối ưu!
    """
    print("\n" + "=" * 85)
    print("TASK 4.3: THỰC THI THUẬT TOÁN LỰA CHỌN BIẾN LÙI (BACKWARD ELIMINATION)")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    bien_hien_tai = list(cac_bien_ung_vien)
    nhat_ky_vong_lap = []
    vong_lap = 1

    while True:
        # Thêm hệ số chặn const
        X_curr = sm.add_constant(df_train[bien_hien_tai])
        model_curr = sm.OLS(y_train, X_curr).fit()

        # Lấy p-values của các biến độc lập (loại trừ hằng số const)
        p_values = model_curr.pvalues.drop("const")
        max_p = p_values.max()
        bien_kem_nhat = p_values.idxmax()

        # Ghi nhận các chỉ số của vòng lặp hiện tại
        thong_tin_vong = {
            "Vòng": vong_lap,
            "Số biến": len(bien_hien_tai),
            "Danh sách biến": list(bien_hien_tai),
            "R-squared": round(model_curr.rsquared, 4),
            "Adj R-squared": round(model_curr.rsquared_adj, 4),
            "F-statistic": round(model_curr.fvalue, 2),
            "Prob (F)": f"{model_curr.f_pvalue:.4e}",
            "AIC": round(model_curr.aic, 2),
            "BIC": round(model_curr.bic, 2),
            "Biến có p lớn nhất": bien_kem_nhat,
            "Max p-value": round(max_p, 4)
        }
        nhat_ky_vong_lap.append(thong_tin_vong)

        print(f"\n[VÒNG LẶP {vong_lap}] Số biến: {len(bien_hien_tai)} | "
              f"R2 = {model_curr.rsquared:.4f} | Adj-R2 = {model_curr.rsquared_adj:.4f} | AIC = {model_curr.aic:.2f}")
        print(f"  + Biến có p-value lớn nhất: '{bien_kem_nhat}' (p = {max_p:.4f})")

        # Kiểm tra điều kiện loại bỏ biến
        if max_p > nguong_p:
            print(f"  -> LOẠI BỎ: '{bien_kem_nhat}' vì p-value ({max_p:.4f}) > ngưỡng {nguong_p}")
            bien_hien_tai.remove(bien_kem_nhat)
            vong_lap += 1
        else:
            print(f"  -> DỪNG THUẬT TOÁN: Tất cả các biến còn lại đều có p-value <= {nguong_p}!")
            mo_hinh_toi_uu = model_curr
            cac_bien_toi_uu = list(bien_hien_tai)
            break

    df_nhat_ky = pd.DataFrame(nhat_ky_vong_lap)
    print("\n" + "=" * 85)
    print("BẢNG NHẬT KÝ QUÁ TRÌNH LỰA CHỌN BIẾN (ITERATION LOG):")
    print(df_nhat_ky[["Vòng", "Số biến", "Adj R-squared", "AIC", "Biến có p lớn nhất", "Max p-value"]].to_string(index=False))
    
    print("\n" + "=" * 85)
    print("KẾT QUẢ MÔ HÌNH TỐI ƯU SAU STEPWISE (STEPWISE OPTIMAL MODEL):")
    print(mo_hinh_toi_uu.summary())

    return mo_hinh_toi_uu, cac_bien_toi_uu, df_nhat_ky

def so_sanh_cac_mo_hinh(df_train, bien_phu_thuoc):
    """
    Task 4.4: Huấn luyện và so sánh hiệu quả giữa 3 mô hình theo đúng giáo trình DA-SGU:
    1. Mô hình Vĩ mô thuần túy (Macro Only): Chỉ dùng các biến thị trường bên ngoài.
    2. Mô hình Tự hồi quy thuần túy (AR(1) Model): Y_t = beta_0 + beta_1 * Y_{t-1}.
    3. Mô hình Tối ưu kết hợp (Stepwise Optimal / ARDL): Kết hợp AR(1) và các biến vĩ mô có ý nghĩa.
    """
    print("\n" + "=" * 85)
    print("TASK 4.4: BẢNG SO SÁNH HIỆU QUẢ CÁC MÔ HÌNH HỒI QUY (MODEL BENCHMARKING)")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]

    # Mô hình 1: Macro Only
    bien_macro = ["London_Robusta_Price", "VN_Production", "Crude_Oil_Price", "Harvest_Season_In_Harvest"]
    X_m1 = sm.add_constant(df_train[bien_macro])
    m1 = sm.OLS(y_train, X_m1).fit()

    # Mô hình 2: AR(1) thuần túy
    bien_ar = ["VN_Coffee_Price_Lag1"]
    X_m2 = sm.add_constant(df_train[bien_ar])
    m2 = sm.OLS(y_train, X_m2).fit()

    # Mô hình 3: Stepwise Optimal (AR + Vĩ mô chọn lọc)
    bien_stepwise = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "Crude_Oil_Price"]
    X_m3 = sm.add_constant(df_train[bien_stepwise])
    m3 = sm.OLS(y_train, X_m3).fit()

    danh_sach_so_sanh = [
        {
            "Mô hình": "1. Macro Only (Chỉ biến Vĩ mô)",
            "Số biến": len(bien_macro),
            "R-squared": round(m1.rsquared, 4),
            "Adj R-squared": round(m1.rsquared_adj, 4),
            "F-statistic": round(m1.fvalue, 2),
            "AIC": round(m1.aic, 2),
            "BIC": round(m1.bic, 2),
            "Durbin-Watson": round(sm.stats.stattools.durbin_watson(m1.resid), 3)
        },
        {
            "Mô hình": "2. AR(1) Model (Tự hồi quy bậc 1)",
            "Số biến": len(bien_ar),
            "R-squared": round(m2.rsquared, 4),
            "Adj R-squared": round(m2.rsquared_adj, 4),
            "F-statistic": round(m2.fvalue, 2),
            "AIC": round(m2.aic, 2),
            "BIC": round(m2.bic, 2),
            "Durbin-Watson": round(sm.stats.stattools.durbin_watson(m2.resid), 3)
        },
        {
            "Mô hình": "3. Stepwise Optimal (Kết hợp tối ưu)",
            "Số biến": len(bien_stepwise),
            "R-squared": round(m3.rsquared, 4),
            "Adj R-squared": round(m3.rsquared_adj, 4),
            "F-statistic": round(m3.fvalue, 2),
            "AIC": round(m3.aic, 2),
            "BIC": round(m3.bic, 2),
            "Durbin-Watson": round(sm.stats.stattools.durbin_watson(m3.resid), 3)
        }
    ]

    df_so_sanh = pd.DataFrame(danh_sach_so_sanh)
    print(df_so_sanh.to_string(index=False))

    return m1, m2, m3, df_so_sanh

def trich_xuat_bang_he_so(mo_hinh, ten_mo_hinh="Mô hình"):
    """
    Hàm tiện ích trích xuất bảng hệ số hồi quy chuẩn hóa gồm:
    Hệ số Beta, Sai số chuẩn (Std. Error), giá trị t-stat, p-value và khoảng tin cậy 95%.
    """
    bang_he_so = pd.DataFrame({
        "Hệ số (Beta)": mo_hinh.params,
        "Sai số chuẩn (Std.Err)": mo_hinh.bse,
        "Thống kê t (t-stat)": mo_hinh.tvalues,
        "Giá trị p (p-value)": mo_hinh.pvalues,
        "Khoảng tin cậy dưới 95%": mo_hinh.conf_int()[0],
        "Khoảng tin cậy trên 95%": mo_hinh.conf_int()[1]
    })
    return bang_he_so

def main():
    """
    Hàm điều phối toàn bộ chu trình Bước 4
    """
    file_unscaled = "data/processed/vn_coffee_unscaled.csv"
    file_scaled = "data/processed/vn_coffee_scaled.csv"

    assert os.path.exists(file_unscaled), f"Lỗi: Không tìm thấy file '{file_unscaled}'!"
    df = pd.read_csv(file_unscaled)

    bien_phu_thuoc = "VN_Coffee_Price"
    cac_bien_ung_vien = [
        "VN_Coffee_Price_Lag1",
        "London_Robusta_Price",
        "VN_Production",
        "Crude_Oil_Price",
        "Harvest_Season_In_Harvest"
    ]

    # Task 4.1: Chia tập dữ liệu 80% Train và 20% Test theo chuỗi thời gian
    df_train, df_test = chia_tap_du_lieu_chuoi_thoi_gian(df, ty_le_train=0.8)

    # Task 4.2: Ước lượng Mô hình Đầy đủ
    model_full = uoc_luong_mo_hinh_day_du(df_train, cac_bien_ung_vien, bien_phu_thuoc)

    # Task 4.3: Thực thi thuật toán Backward Elimination
    mo_hinh_toi_uu, cac_bien_toi_uu, df_nhat_ky = thuat_toan_backward_elimination(
        df_train, cac_bien_ung_vien, bien_phu_thuoc, nguong_p=0.05
    )

    # Task 4.4: So sánh 3 mô hình (Macro Only, AR(1), Stepwise Optimal)
    m_macro, m_ar, m_opt, df_so_sanh = so_sanh_cac_mo_hinh(df_train, bien_phu_thuoc)

    # Trích xuất bảng hệ số của mô hình tối ưu
    bang_he_so_toi_uu = trich_xuat_bang_he_so(mo_hinh_toi_uu, "Stepwise Optimal")

    # Mô hình chuẩn hóa (Scaled Model) để lấy Standardized Beta
    df_sc = pd.read_csv(file_scaled)
    df_train_sc = df_sc.iloc[:len(df_train)].copy().reset_index(drop=True)
    X_train_sc = sm.add_constant(df_train_sc[cac_bien_toi_uu])
    y_train_sc = df_train_sc[bien_phu_thuoc]
    model_scaled_opt = sm.OLS(y_train_sc, X_train_sc).fit()

    print("\n" + "=" * 85)
    print("HỆ SỐ HỒI QUY CHUẨN HÓA (STANDARDIZED BETA) CỦA MÔ HÌNH TỐI ƯU:")
    print("=" * 85)
    for col in cac_bien_toi_uu:
        b_sc = model_scaled_opt.params[col]
        p_sc = model_scaled_opt.pvalues[col]
        print(f"- {col:<25}: Beta* = {b_sc:+10.4f} | p-value = {p_sc:.4e}")

    # --------------------------------------------------------------------------
    # CHẠY SELF-TEST (TỰ ĐỘNG KIỂM THỬ BƯỚC 4 THEO QUY TẮC SỐ 5)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CHẠY SELF-TEST (TỰ ĐỘNG KIỂM THỬ BƯỚC 4)")
    print("=" * 85)

    # Assertion 1: Tất cả các biến độc lập trong mô hình Stepwise đều có p-value < 0.05
    p_values_final = mo_hinh_toi_uu.pvalues.drop("const")
    assert (p_values_final < 0.05).all(), f"Self-Test Thất bại: Vẫn còn biến có p-value >= 0.05: {p_values_final}"
    print(f"[PASS] Assertion 1: Toàn bộ biến giữ lại đều đạt ý nghĩa thống kê p < 0.05 (Max p = {p_values_final.max():.4f}).")

    # Assertion 2: Kiểm định F-statistic của mô hình tối ưu có ý nghĩa thống kê cao (p < 0.05)
    assert mo_hinh_toi_uu.f_pvalue < 0.05, f"Self-Test Thất bại: Prob(F-statistic) ({mo_hinh_toi_uu.f_pvalue}) >= 0.05!"
    print(f"[PASS] Assertion 2: Prob (F-statistic) = {mo_hinh_toi_uu.f_pvalue:.4e} < 0.05 (Mô hình giải thích có ý nghĩa 100%).")

    # Assertion 3: Hệ số R-squared của mô hình tối ưu đạt mức xuất sắc (R2 >= 0.8)
    assert mo_hinh_toi_uu.rsquared >= 0.80, f"Self-Test Thất bại: R-squared ({mo_hinh_toi_uu.rsquared}) < 0.80!"
    print(f"[PASS] Assertion 3: R-squared = {mo_hinh_toi_uu.rsquared:.4f} (>= 80% độ biến thiên được giải thích).")

    # Assertion 4: Tỷ lệ phân chia Train / Test đúng 80% / 20%
    assert len(df_train) == 122 and len(df_test) == 31, "Self-Test Thất bại: Số lượng mẫu Train/Test không đúng!"
    print(f"[PASS] Assertion 4: Phân chia tập dữ liệu chuẩn 80% ({len(df_train)}) và 20% ({len(df_test)}).")

    print("-" * 85)
    print("SELF-TEST BƯỚC 4: PASS 100% TOÀN BỘ CÁC BÀI KIỂM THỬ TỰ ĐỘNG!")
    print("=" * 85)

    return mo_hinh_toi_uu, df_nhat_ky, df_so_sanh, model_scaled_opt

if __name__ == "__main__":
    main()
