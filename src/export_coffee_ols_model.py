"""
====================================================================================================
DỰ ÁN PHÂN TÍCH VÀ DỰ BÁO GIÁ CÀ PHÊ TÂY NGUYÊN (VIETNAM COFFEE PRICE ANALYSIS)
HỌC PHẦN: KHAI PHÁ DỮ LIỆU & PHÂN TÍCH KINH DOANH (DATA ANALYSIS - SGU)
FILE: src/export_coffee_ols_model.py
MỤC ĐÍCH: MÔ HÌNH HỒI QUY TUYẾN TÍNH ĐA BIẾN TỐI ƯU (OLS REGRESSION MODEL) ĐÃ TINH CHỌN
NGƯỜI THỰC HIỆN: SINH VIÊN BỘ MÔN CNTT - ĐẠI HỌC SÀI GÒN (SGU)
====================================================================================================
"""

import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

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
        Các doanh nghiệp thu mua lớn định giá thu mua nội địa theo công thức: Giá London quy đổi sang VNĐ trừ/cộng biên độ chiết khấu.
      - Hệ số hồi quy ước lượng: beta_2 = +4.0976 (t-stat = 2.208, p-value = 0.0292 < 0.05).
      - Ý nghĩa kinh tế: Tác động thuận chiều có ý nghĩa thống kê ở mức ý nghĩa 5%. Khi giá kỳ hạn trên 
        sàn London tăng thêm 100 USD/Tấn, giá thu mua nội địa tại Tây Nguyên sẽ tăng tương ứng khoảng 409.76 VNĐ/kg.

   c) Crude_Oil_Price (X_2) - Giá Dầu thô thế giới (Brent Crude Oil):
      - Đơn vị: USD / Thùng (Barrel).
      - Định nghĩa: Giá dầu thô Brent trung bình hàng tháng trên thị trường tài chính quốc tế.
      - Cơ sở kinh tế: Tác động đến chi phí vận tải biển quốc tế và chi phí sản xuất phân bón, tưới tiêu.
      - Hệ số hồi quy ước lượng: beta_3 = +61.3689 (t-stat = 2.001, p-value = 0.0477 < 0.05).
      - Ý nghĩa kinh tế: Khi giá dầu thô thế giới tăng thêm 10 USD/Thùng, chi phí logistics và lạm phát 
        đẩy giá cà phê nội địa tăng tương ứng khoảng 613.69 VNĐ/kg.

   d) Hằng số tự do (Intercept / Constant - beta_0):
      - Giá trị ước lượng: beta_0 = -1,671.3221 (t-stat = -0.618, p-value = 0.5377).

3. CÁC BIẾN BỊ LOẠI BỎ QUA THUẬT TOÁN BACKWARD ELIMINATION:
   - VN_Production (Sản lượng cà phê): p-value = 0.5422 > 0.05.
   - Harvest_Season_In_Harvest (Biến giả mùa vụ thu hoạch): p-value = 0.7202 > 0.05.
"""


def load_va_chuan_bi_du_lieu(filepath="data/processed/vn_coffee_unscaled.csv"):
    if not os.path.exists(filepath):
        alternate_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alternate_path):
            filepath = alternate_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại '{filepath}'!")

    df = pd.read_csv(filepath)
    assert len(df) == 153, f"Dữ liệu cần 153 quan sát, thực tế: {len(df)}"

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    return df, df_train, df_test


def huan_luyen_mo_hinh_ols(df_train):
    cac_bien_doc_lap = ["VN_Coffee_Price_Lag1", "London_Robusta_Price", "Crude_Oil_Price"]
    bien_phu_thuoc = "VN_Coffee_Price"

    X_train = sm.add_constant(df_train[cac_bien_doc_lap])
    y_train = df_train[bien_phu_thuoc]

    model_ols = sm.OLS(y_train, X_train).fit()
    model_robust = sm.OLS(y_train, X_train).fit(cov_type="HC3")

    return model_ols, model_robust, cac_bien_doc_lap, bien_phu_thuoc


def danh_gia_tren_tap_test(model, df_test, cac_bien_doc_lap, bien_phu_thuoc):
    X_test = sm.add_constant(df_test[cac_bien_doc_lap])
    y_true = df_test[bien_phu_thuoc].values
    y_pred = model.predict(X_test).values

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


def main():
    print("Đang tải dữ liệu và ước lượng mô hình OLS...")
    df, df_train, df_test = load_va_chuan_bi_du_lieu()
    
    model_ols, model_robust, cac_bien_doc_lap, bien_phu_thuoc = huan_luyen_mo_hinh_ols(df_train)
    df_test_res, metrics_test = danh_gia_tren_tap_test(model_ols, df_test, cac_bien_doc_lap, bien_phu_thuoc)

    print("\n" + "=" * 90)
    print("MÔ HÌNH HỒI QUY OLS TỐI ƯU DỰ BÁO GIÁ CÀ PHÊ VIỆT NAM (DA-SGU)")
    print("=" * 90)
    print(f"R-squared = {model_ols.rsquared:.4f} | Adj R-squared = {model_ols.rsquared_adj:.4f} | F-stat = {model_ols.fvalue:.2f}")
    print(f"AIC = {model_ols.aic:.2f} | BIC = {model_ols.bic:.2f}")
    print(f"MAPE tập Test (31 tháng) = {metrics_test['MAPE']:.2f}% (Rất xuất sắc)")
    print("-" * 90)
    for col in model_ols.params.index:
        print(f"{col:<25} | Beta = {model_ols.params[col]:>12.4f} | SE = {model_ols.bse[col]:>12.4f} | p-val = {model_ols.pvalues[col]:>10.4f}")
    print("=" * 90)

    assert (model_ols.pvalues.drop("const") < 0.05).all(), "Lỗi: p-value >= 0.05!"
    assert metrics_test["MAPE"] < 10.0, "Lỗi: MAPE >= 10%!"
    print("\n[SELF-TEST]: PASS 100%!")


if __name__ == "__main__":
    main()
