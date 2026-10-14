"""
====================================================================================================
DỰ ÁN PHÂN TÍCH VÀ DỰ BÁO GIÁ CÀ PHÊ TÂY NGUYÊN (VIETNAM COFFEE PRICE ANALYSIS)
HỌC PHẦN: KHAI PHÁ DỮ LIỆU & PHÂN TÍCH KINH DOANH (DATA ANALYSIS - SGU)
FILE: export_coffee_ols_model.py
MỤC ĐÍCH: MÔ HÌNH HỒI QUY TUYẾN TÍNH ĐA BIẾN TỐI ƯU (OLS REGRESSION MODEL) ĐÃ TINH CHỌN
NGƯỜI THỰC HIỆN: SINH VIÊN BỘ MÔN CNTT - ĐẠI HỌC SÀI GÒN (SGU)
====================================================================================================

TỔNG QUAN VỀ PHƯƠNG TRÌNH HỒI QUY:
Phương trình kinh tế lượng ước lượng giá cà phê Tây Nguyên (VN_Coffee_Price):
    VN_Coffee_Price_t = beta_0 
                      + beta_1 * VN_Coffee_Price_Lag1_t 
                      + beta_2 * London_Robusta_Price_t 
                      + beta_3 * Crude_Oil_Price_t 
                      + epsilon_t

Trong đó:
- R-squared (R2)            : 0.9253 (Mô hình giải thích được 92.53% biến thiên của giá cà phê)
- Adjusted R-squared (Adj-R2): 0.9234
- F-statistic               : 487.43 (p-value = 2.75e-66 < 0.001 -> Toàn bộ mô hình có ý nghĩa)
- Sai số MAPE trên tập Test : 6.39% (< 10% -> Độ chính xác dự báo ở mức Xuất sắc)
====================================================================================================
"""

import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Đảm bảo hiển thị tiếng Việt UTF-8 chuẩn trên console Windows/Linux
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ==================================================================================================
# 1. TỪ ĐIỂN ĐỊNH NGHĨA VÀ GIẢI THÍCH CHI TIẾT CÁC BIẾN SỐ (VARIABLE DICTIONARY)
# ==================================================================================================
"""
GIẢI THÍCH CHI TIẾT CÁC BIẾN TRONG NGHIÊN CỨU:

1. BIẾN PHỤ THUỘC (DEPENDENT VARIABLE - Y):
   - Tên biến: VN_Coffee_Price (Y_t)
   - Đơn vị tính: VNĐ / kg cà phê nhân xô.
   - Định nghĩa: Giá trung bình hàng tháng của cà phê vối Robusta tại 4 tỉnh trọng điểm Tây Nguyên 
                 (Đắk Lắk, Lâm Đồng, Gia Lai, Đắk Nông) - thủ phủ cà phê chiếm >90% sản lượng cả nước.
   - Vai trò: Mục tiêu trọng tâm cần dự báo và lượng hóa các yếu tố tác động.

2. CÁC BIẾN ĐỘC LẬP TRONG MÔ HÌNH TỐI ƯU (SELECTED INDEPENDENT VARIABLES - X):
   
   a) VN_Coffee_Price_Lag1 (Y_{t-1}) - Biến trễ giá nội địa tháng trước:
      - Đơn vị: VNĐ / kg.
      - Định nghĩa: Giá cà phê nội địa Tây Nguyên của tháng liền kề trước đó (thời điểm t-1).
      - Cơ sở kinh tế: Phản ánh tính quán tính chuỗi thời gian (Autoregressive - AR(1)), tâm lý găm hàng 
        của nông dân và các hợp đồng thương mại kỳ hạn đã ký từ tháng trước.
      - Hệ số hồi quy ước lượng: beta_1 = +0.8631 (t-stat = 21.611, p-value < 0.0001).
      - Ý nghĩa kinh tế: Tác động thuận chiều cực kỳ mạnh. Trong điều kiện các yếu tố khác không đổi, 
        nếu giá tháng trước tăng thêm 1.000 VNĐ/kg thì giá tháng này bình quân giữ lại đà tăng 863.1 VNĐ/kg.
        Điều này khẳng định thị trường có xu hướng duy trì đà giá (price momentum) rất cao.

   b) London_Robusta_Price (X_1) - Giá cà phê Robusta kỳ hạn sàn London (ICE Futures Europe):
      - Đơn vị: USD / Tấn (Metric Ton).
      - Định nghĩa: Giá đóng cửa trung bình tháng của hợp đồng kỳ hạn Robusta giao dịch trên sàn ICE London.
      - Cơ sở kinh tế: Việt Nam là quốc gia xuất khẩu Robusta số 1 thế giới (>40% thị phần toàn cầu). 
        Các doanh nghiệp thu mua lớn (Nestle, Trung Nguyên, Simexco...) định giá thu mua nội địa theo 
        công thức: Giá London quy đổi sang VNĐ trừ/cộng biên độ chiết khấu (FOB differential).
      - Hệ số hồi quy ước lượng: beta_2 = +4.0976 (t-stat = 2.208, p-value = 0.0292 < 0.05).
      - Ý nghĩa kinh tế: Tác động thuận chiều có ý nghĩa thống kê ở mức ý nghĩa 5%. Khi giá kỳ hạn trên 
        sàn London tăng thêm 100 USD/Tấn, giá thu mua nội địa tại Tây Nguyên sẽ tăng tương ứng khoảng 
        409.76 VNĐ/kg (tương đương 409.760 VNĐ/tấn).

   c) Crude_Oil_Price (X_2) - Giá Dầu thô thế giới (Brent Crude Oil):
      - Đơn vị: USD / Thùng (Barrel).
      - Định nghĩa: Giá dầu thô Brent trung bình hàng tháng trên thị trường tài chính quốc tế.
      - Cơ sở kinh tế: Dầu thô là nguyên liệu đầu vào thiết yếu của nền kinh tế toàn cầu, tác động đến:
        + Chi phí vận tải biển quốc tế (Container logistics, cước tàu rời từ cảng Sài Gòn đi EU/Mỹ).
        + Chi phí sản xuất: phân bón hóa học (đạm Urê, NPK phụ thuộc dầu khí) và dầu diesel tưới tiêu Tây Nguyên.
      - Hệ số hồi quy ước lượng: beta_3 = +61.3689 (t-stat = 2.001, p-value = 0.0477 < 0.05).
      - Ý nghĩa kinh tế: Tác động thuận chiều có ý nghĩa thống kê ở mức 5%. Khi giá dầu thô thế giới tăng 
        thêm 10 USD/Thùng, chi phí logistics và lạm phát đẩy giá cà phê nội địa tăng tương ứng khoảng 
        613.69 VNĐ/kg.

   d) Hằng số tự do (Intercept / Constant - beta_0):
      - Giá trị ước lượng: beta_0 = -1,671.3221 (t-stat = -0.618, p-value = 0.5377).
      - Ý nghĩa thống kê: Giá trị chặn kỹ thuật của hàm hồi quy, p-value > 0.05 mang tính chuẩn hóa đại số, 
        giúp mặt phẳng hồi quy đi qua trọng tâm dữ liệu mà không làm sai lệch các hệ số góc.

3. CÁC BIẾN BỊ LOẠI BỎ QUA THUẬT TOÁN BACKWARD ELIMINATION (EXCLUDED VARIABLES):
   - VN_Production (Sản lượng cà phê): p-value = 0.5422 > 0.05. Bị loại vì sản lượng công bố theo chu kỳ 
     năm/vụ mùa, không phản ánh biến động giá linh hoạt theo từng tháng.
   - Harvest_Season_In_Harvest (Biến giả mùa vụ thu hoạch): p-value = 0.7202 > 0.05. Bị loại vì yếu tố 
     mùa vụ đã được hấp thụ đầy đủ thông qua biến trễ giá Lag1 và khối lượng giao dịch trên sàn London.
"""


def load_va_chuan_bi_du_lieu(filepath="data/processed/vn_coffee_unscaled.csv"):
    """
    Đọc dữ liệu sạch thực tế và phân chia tập Train (80%) / Test (20%) theo đúng thứ tự chuỗi thời gian.
    """
    if not os.path.exists(filepath):
        # Hỗ trợ tìm kiếm file nếu chạy từ các thư mục con
        alternate_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alternate_path):
            filepath = alternate_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại '{filepath}'!")

    df = pd.read_csv(filepath)
    assert len(df) == 153, f"Dữ liệu cần 153 quan sát, thực tế: {len(df)}"

    # Chia Train (80% = 122 tháng: 2014-01 -> 2024-02) và Test (20% = 31 tháng: 2024-03 -> 2026-09)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    return df, df_train, df_test


def huan_luyen_mo_hinh_ols(df_train):
    """
    Huấn luyện mô hình OLS tối ưu với các biến độc lập đã được chọn lọc qua Stepwise / Backward Elimination.
    """
    # Khai báo danh sách biến giải thích tối ưu
    cac_bien_doc_lap = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "Crude_Oil_Price"]
    bien_phu_thuoc = "VN_Coffee_Price"

    # Chuẩn bị ma trận X và vector y
    X_train = sm.add_constant(df_train[cac_bien_doc_lap])
    y_train = df_train[bien_phu_thuoc]

    # Ước lượng mô hình OLS chuẩn
    model_ols = sm.OLS(y_train, X_train).fit()

    # Ước lượng thêm mô hình OLS với Sai số chuẩn vững HC3 (để khắc phục triệt để phương sai sai số thay đổi)
    model_robust = sm.OLS(y_train, X_train).fit(cov_type="HC3")

    return model_ols, model_robust, cac_bien_doc_lap, bien_phu_thuoc


def danh_gia_tren_tap_test(model, df_test, cac_bien_doc_lap, bien_phu_thuoc):
    """
    Thực hiện dự báo trên tập kiểm thử độc lập (Test Set - 31 tháng) và tính toán sai số thực nghiệm.
    """
    X_test = sm.add_constant(df_test[cac_bien_doc_lap])
    y_true = df_test[bien_phu_thuoc].values
    y_pred = model.predict(X_test).values

    # Tính toán các chỉ số sai số theo chuẩn giáo trình DA-SGU
    sai_so = y_true - y_pred
    mape = np.mean(np.abs(sai_so / y_true)) * 100
    rmse = np.sqrt(np.mean(sai_so ** 2))
    mae = np.mean(np.abs(sai_so))

    df_ket_qua_test = pd.DataFrame({
        "Tháng": df_test["Date"],
        "Thực tế (VNĐ/kg)": y_true,
        "Dự báo (VNĐ/kg)": np.round(y_pred, 2),
        "Sai lệch (VNĐ/kg)": np.round(sai_so, 2),
        "Sai số (%)": np.round(np.abs(sai_so / y_true) * 100, 2)
    })

    metrics = {"MAPE": mape, "RMSE": rmse, "MAE": mae}
    return df_ket_qua_test, metrics


def in_bao_cao_mo_hinh(model_ols, model_robust, metrics_test, df_test_preview):
    """
    In kết quả tổng hợp của mô hình dưới dạng báo cáo chuyên nghiệp để sinh viên trình giảng viên.
    """
    print("\n" + "=" * 90)
    print("BÁO CÁO MÔ HÌNH HỒI QUY TUYẾN TÍNH OLS TỐI ƯU - DỰ BÁO GIÁ CÀ PHÊ VIỆT NAM")
    print("=" * 90)
    print(f"1. THÔNG SỐ TỔNG THỂ CỦA MÔ HÌNH (MODEL GOODNESS-OF-FIT):")
    print(f"   - Số quan sát huấn luyện (N_train) : {int(model_ols.nobs)} tháng (2014-01 đến 2024-02)")
    print(f"   - Hệ số xác định R-squared (R2)   : {model_ols.rsquared:.4f} (Giải thích {model_ols.rsquared*100:.2f}% phương sai)")
    print(f"   - R2 hiệu chỉnh (Adjusted R2)     : {model_ols.rsquared_adj:.4f}")
    print(f"   - Thống kê F (F-statistic)        : {model_ols.fvalue:.2f} (p-value = {model_ols.f_pvalue:.2e}) -> Rất có ý nghĩa")
    print(f"   - Tiêu chuẩn Akaike (AIC)          : {model_ols.aic:.2f}")
    print(f"   - Tiêu chuẩn Bayesian (BIC)        : {model_ols.bic:.2f}")

    print("\n2. PHƯƠNG TRÌNH HỒI QUY DẠNG TOÁN HỌC (MATHEMATICAL EQUATION):")
    b0 = model_ols.params["const"]
    b1 = model_ols.params["VN_Coffee_Price_Lag1"]
    b2 = model_ols.params["London_Robusta_Price"]
    b3 = model_ols.params["Crude_Oil_Price"]
    print(f"   VN_Coffee_Price_t = {b0:,.2f}")
    print(f"                     + ({b1:.4f}) * VN_Coffee_Price_Lag1_t")
    print(f"                     + ({b2:.4f}) * London_Robusta_Price_t")
    print(f"                     + ({b3:.4f}) * Crude_Oil_Price_t")

    print("\n3. BẢNG ƯỚC LƯỢNG HỆ SỐ HỒI QUY VÀ Ý NGHĨA THỐNG KÊ:")
    print("-" * 90)
    header = f"{'Tên biến':<25} | {'Hệ số Beta':>12} | {'Sai số chuẩn':>12} | {'t-statistic':>11} | {'p-value':>10} | {'Kết luận':<10}"
    print(header)
    print("-" * 90)
    for col in model_ols.params.index:
        coef = model_ols.params[col]
        se = model_ols.bse[col]
        t = model_ols.tvalues[col]
        p = model_ols.pvalues[col]
        sig = "Có ý nghĩa" if p < 0.05 else "Không ý nghĩa"
        print(f"{col:<25} | {coef:>12.4f} | {se:>12.4f} | {t:>11.3f} | {p:>10.4f} | {sig:<10}")
    print("-" * 90)

    print("\n4. SO SÁNH VỚI MÔ HÌNH HIỆU CHỈNH SAI SỐ CHUẨN VỮNG (OLS ROBUST HC3):")
    print("   (Giúp đảm bảo p-value hoàn toàn chuẩn mực ngay cả khi thị trường biến động mạnh)")
    print("-" * 90)
    print(f"{'Tên biến':<25} | {'Beta OLS':>12} | {'SE OLS Thường':>14} | {'SE Robust HC3':>14} | {'p-val Robust':>12}")
    print("-" * 90)
    for col in model_ols.params.index:
        print(f"{col:<25} | {model_ols.params[col]:>12.4f} | {model_ols.bse[col]:>14.4f} | {model_robust.bse[col]:>14.4f} | {model_robust.pvalues[col]:>12.4f}")
    print("-" * 90)

    print("\n5. KẾT QUẢ KIỂM THỬ THỰC TẾ TRÊN TẬP DỮ LIỆU ĐỘC LẬP (TEST SET - 31 THÁNG):")
    print(f"   - Sai số phần trăm tuyệt đối trung bình (MAPE) : {metrics_test['MAPE']:.2f}% (Đạt tiêu chuẩn < 10% - Rất xuất sắc)")
    print(f"   - Căn bậc hai sai số bình phương trung bình (RMSE): {metrics_test['RMSE']:,.2f} VNĐ/kg")
    print(f"   - Sai số tuyệt đối trung bình (MAE)             : {metrics_test['MAE']:,.2f} VNĐ/kg")
    print("\n   Dự báo 5 tháng gần nhất trên tập Test:")
    print(df_test_preview.tail(5).to_string(index=False))
    print("=" * 90)


def main():
    print("Đang tải dữ liệu và ước lượng mô hình OLS...")
    df, df_train, df_test = load_va_chuan_bi_du_lieu()
    
    # Huấn luyện
    model_ols, model_robust, cac_bien_doc_lap, bien_phu_thuoc = huan_luyen_mo_hinh_ols(df_train)
    
    # Đánh giá trên tập test
    df_test_res, metrics_test = danh_gia_tren_tap_test(model_ols, df_test, cac_bien_doc_lap, bien_phu_thuoc)
    
    # In báo cáo đầy đủ
    in_bao_cao_mo_hinh(model_ols, model_robust, metrics_test, df_test_res)

    # Tự kiểm tra (Self-Test Assertions) theo chuẩn DA-SGU
    assert (model_ols.pvalues.drop("const") < 0.05).all(), "Lỗi: Biến độc lập không đạt ý nghĩa thống kê p < 0.05!"
    assert model_ols.rsquared > 0.85, f"Lỗi: R2 không đạt chuẩn (> 0.85): {model_ols.rsquared:.4f}"
    assert metrics_test["MAPE"] < 10.0, f"Lỗi: MAPE vượt ngưỡng 10%: {metrics_test['MAPE']:.2f}%"
    print("\n[SELF-TEST]: TOÀN BỘ CÁC CHỈ SỐ ĐẠT CHUẨN XUẤT SẮC - MÔ HÌNH SẴN SÀNG NỘP GIẢNG VIÊN!")


if __name__ == "__main__":
    main()
