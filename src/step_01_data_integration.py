# ==============================================================================
# TẬP TIN: src/step_01_data_integration.py
# MỤC ĐÍCH: Kiểm kê, đồng bộ tần suất và tích hợp dữ liệu đa nguồn (KDD Bước 1)
# QUY CHUẨN: Giáo trình Phân tích Dữ liệu DA - SGU (Đại học Sài Gòn)
# NGUYÊN TẮC: Zero Hallucination, Line-by-Line Explanation, Tự kiểm tra (Self-Test)
# ==============================================================================

# Import thư viện sys để cấu hình hiển thị mã hóa UTF-8 trên Windows console
import sys

# Import thư viện os để quản lý đường dẫn tập tin và thư mục làm việc
import os

# Import thư viện pandas để xử lý cấu trúc dữ liệu dạng bảng (DataFrame) và chuỗi thời gian
import pandas as pd

# Import thư viện numpy để thực hiện tính toán số học và xử lý giá trị NaN
import numpy as np

# Cấu hình UTF-8 cho dòng xuất stdout/stderr tránh lỗi mã hóa ký tự trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def kiem_ke_tap_tin_tho():
    """
    Task 1.1: Quét và hiển thị cấu trúc tổng thể của các file dữ liệu thô trong data/raw/
    """
    thu_muc_raw = "data/raw"
    danh_sach_file = sorted(os.listdir(thu_muc_raw))
    
    print("=" * 80)
    print("TASK 1.1: KIỂM KÊ CẤU TRÚC CHI TIẾT CÁC TỆP DỮ LIỆU THÔ (DATA AUDIT)")
    print("=" * 80)
    for i, fn in enumerate(danh_sach_file, 1):
        duong_dan = os.path.join(thu_muc_raw, fn)
        kich_thuoc = os.path.getsize(duong_dan)
        print(f"[{i:02d}] {fn:<38} | Kích thước: {kich_thuoc:>10,} bytes")
    print("-" * 80)

def dong_bo_va_tich_hop_du_lieu():
    """
    Thực hiện Tasks 1.3 -> 1.6:
    - Task 1.3: Chuẩn hóa cột thời gian về định dạng chuẩn YYYY-MM
    - Task 1.4: Quy đổi đơn vị tiền tệ và khối lượng theo chuẩn Data Dictionary
    - Task 1.5: Tạo biến trễ VN_Coffee_Price_Lag1 (Y_t-1) và biến phân loại Harvest_Season (X7)
    - Task 1.6: Ghép nối (Merge) các bảng dữ liệu theo khóa Date và lưu thành phẩm
    """
    print("\n" * 1)
    print("=" * 80)
    print("BẮT ĐẦU ĐỒNG BỘ TẦN SUẤT & TÍCH HỢP DỮ LIỆU ĐA NGUỒN (TASKS 1.3 - 1.6)")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # BƯỚC A: XỬ LÝ NGUỒN 1 - GIÁ CÀ PHÊ (chart_20261008T021435.csv)
    # --------------------------------------------------------------------------
    print("\n>>> [A] Xử lý Biến Y (VN_Coffee_Price) từ chuỗi theo ngày...")
    duong_dan_chart = "data/raw/chart_20261008T021435.csv"
    
    # Đọc file CSV với phân cách '.', số thập phân ',' và bảng mã utf-8-sig
    df_chart = pd.read_csv(duong_dan_chart, encoding="utf-8-sig", sep=".", decimal=",")
    
    # Chuyển đổi cột Date sang kiểu dữ liệu datetime của pandas (định dạng MM/DD/YYYY)
    df_chart["Date_Parsed"] = pd.to_datetime(df_chart["Date"], format="%m/%d/%Y")
    
    # Chuẩn hóa về chu kỳ Tháng dạng chuỗi YYYY-MM
    df_chart["Date"] = df_chart["Date_Parsed"].dt.to_period("M").astype(str)
    
    # Lấy giá trị trung bình tháng của cột Value (hạ tần suất từ Ngày sang Tháng - Downsampling)
    df_monthly_price = df_chart.groupby("Date", as_index=False)["Value"].mean()
    
    # Đổi tên cột Value thành Giá USD/lb
    df_monthly_price = df_monthly_price.rename(columns={"Value": "Price_USD_lb"})
    
    # Áp dụng công thức quy đổi sang VNĐ/kg theo thỏa thuận với User:
    # 1 lb = 0.45359237 kg => 1 kg = 2.20462 lb
    # Giả định tỷ giá quy chuẩn kinh tế ngành cà phê: 24,000 VNĐ/USD
    # Giá VNĐ/kg = Giá USD/lb * 2.20462 * 24,000
    TY_GIA_QUY_UOC = 24000.0  # VNĐ/USD
    HE_SO_LB_SANG_KG = 2.20462  # 1 kg = 2.20462 lb
    
    # Tính giá cà phê nhân xô VNĐ/kg và làm tròn số nguyên
    df_monthly_price["VN_Coffee_Price"] = np.round(
        df_monthly_price["Price_USD_lb"] * HE_SO_LB_SANG_KG * TY_GIA_QUY_UOC, 2
    )
    print(f"  + Hoàn thành xử lý chuỗi giá Y: {len(df_monthly_price)} tháng.")
    print(f"  + Mẫu 3 tháng gần nhất của Y:\n{df_monthly_price[['Date', 'VN_Coffee_Price']].tail(3)}")

    # --------------------------------------------------------------------------
    # BƯỚC B: XỬ LÝ NGUỒN 2 - GIÁ ROBUSTA SÀN LONDON & GIÁ DẦU THÔ BRENT (CMO)
    # --------------------------------------------------------------------------
    print("\n>>> [B] Xử lý Biến X1 (London_Robusta_Price) và X6 (Crude_Oil_Price) từ World Bank CMO...")
    duong_dan_cmo = "data/raw/CMO-Historical-Data-Monthly.xlsx"
    
    # Đọc sheet 'Monthly Prices', tên cột bắt đầu ở hàng index 4
    df_cmo_raw = pd.read_excel(duong_dan_cmo, sheet_name="Monthly Prices", header=4)
    
    # Đổi tên cột chỉ số thời gian đầu tiên (Unnamed: 0) thành Date_Raw
    df_cmo_raw = df_cmo_raw.rename(columns={df_cmo_raw.columns[0]: "Date_Raw"})
    
    # Lọc bỏ các dòng không có mã thời gian
    df_cmo_raw = df_cmo_raw.dropna(subset=["Date_Raw"])
    
    # Chuẩn hóa mã thời gian từ 'YYYYMmm' (ví dụ '2024M05') sang 'YYYY-MM' (ví dụ '2024-05')
    df_cmo_raw["Date"] = df_cmo_raw["Date_Raw"].astype(str).str.replace("M", "-")
    
    # Lọc các cột cần thiết cho mô hình
    cot_cmo_lay = ["Date", "Coffee, Robusta", "Crude oil, Brent"]
    df_cmo = df_cmo_raw[cot_cmo_lay].copy()
    
    # Chuyển đổi các cột số liệu sang kiểu float (ép lỗi ký tự lạ thành NaN)
    df_cmo["Coffee, Robusta"] = pd.to_numeric(df_cmo["Coffee, Robusta"], errors="coerce")
    df_cmo["Crude oil, Brent"] = pd.to_numeric(df_cmo["Crude oil, Brent"], errors="coerce")
    
    # Quy đổi đơn vị:
    # 'Coffee, Robusta' gốc là $/kg => đổi sang USD/Tấn bằng cách nhân 1,000 (1 Tấn = 1,000 kg)
    df_cmo["London_Robusta_Price"] = np.round(df_cmo["Coffee, Robusta"] * 1000.0, 2)
    
    # 'Crude oil, Brent' gốc là $/bbl => đúng chuẩn USD/Thùng
    df_cmo["Crude_Oil_Price"] = np.round(df_cmo["Crude oil, Brent"], 2)
    
    # Chỉ giữ lại các cột chuẩn sau quy đổi
    df_cmo_clean = df_cmo[["Date", "London_Robusta_Price", "Crude_Oil_Price"]]
    print(f"  + Hoàn thành xử lý X1 và X6 từ CMO: {len(df_cmo_clean)} tháng.")
    print(f"  + Mẫu 3 tháng gần nhất của X1, X6:\n{df_cmo_clean.tail(3)}")

    # --------------------------------------------------------------------------
    # BƯỚC C: XỬ LÝ NGUỒN 3 - SẢN LƯỢNG CÀ PHÊ VIỆT NAM (FAOSTAT_data_en_10-8-2026.csv)
    # --------------------------------------------------------------------------
    print("\n>>> [C] Xử lý Biến X3 (VN_Production) từ FAOSTAT và đồng bộ Năm sang Tháng...")
    duong_dan_fao = "data/raw/FAOSTAT_data_en_10-8-2026.csv"
    
    # Đọc dữ liệu FAOSTAT
    df_fao_raw = pd.read_csv(duong_dan_fao)
    
    # Lọc riêng sản lượng cà phê của Việt Nam
    fao_vn = df_fao_raw[df_fao_raw["Area"] == "Viet Nam"][["Year", "Unit", "Value"]].copy()
    
    # Quy đổi đơn vị từ Tấn (t) sang Nghìn Tấn (Nghìn Tấn = Tấn / 1,000)
    fao_vn["VN_Production_ThousandTons"] = np.round(fao_vn["Value"] / 1000.0, 2)
    
    # Tạo từ điển ánh xạ {Năm: Sản lượng}
    dict_san_luong = dict(zip(fao_vn["Year"].astype(int), fao_vn["VN_Production_ThousandTons"]))
    
    # Lấy năm có sản lượng mới nhất (2024) để gán cho các năm 2025, 2026 (Forward Fill theo quy ước)
    nam_cuoi_cung = max(dict_san_luong.keys())
    san_luong_moi_nhat = dict_san_luong[nam_cuoi_cung]
    dict_san_luong[2025] = san_luong_moi_nhat
    dict_san_luong[2026] = san_luong_moi_nhat
    
    print(f"  + Đã lập từ điển sản lượng theo năm (2000 - 2026). Năm {nam_cuoi_cung}: {san_luong_moi_nhat} Nghìn Tấn.")

    # --------------------------------------------------------------------------
    # BƯỚC D: TÍCH HỢP BẢNG DỮ LIỆU THEO KHÓA 'Date' (MERGE / JOIN)
    # --------------------------------------------------------------------------
    print("\n>>> [D] Tiến hành ghép nối (Merge) các bảng theo khóa 'Date'...")
    
    # Giới hạn khung thời gian nghiên cứu: Lấy tối thiểu 10 năm từ 2014-01 đến 2026-09
    df_merged = pd.merge(df_monthly_price[["Date", "VN_Coffee_Price"]], df_cmo_clean, on="Date", how="inner")
    
    # Lọc giai đoạn từ năm 2014 trở đi (chuẩn 10 năm nghiên cứu theo Context.md)
    df_merged = df_merged[df_merged["Date"] >= "2014-01"].copy()
    
    # Sắp xếp tăng dần theo thời gian (Chronological Order)
    df_merged = df_merged.sort_values(by="Date").reset_index(drop=True)
    
    # Trích xuất năm từ cột Date để ánh xạ biến sản lượng X3
    df_merged["Year_Int"] = df_merged["Date"].str.slice(0, 4).astype(int)
    # Gán sản lượng niên vụ/năm tương ứng cho từng tháng (Thực hiện đúng Lựa chọn A của User)
    df_merged["VN_Production"] = df_merged["Year_Int"].map(dict_san_luong)
    # Xóa cột phụ Year_Int
    df_merged = df_merged.drop(columns=["Year_Int"])

    # --------------------------------------------------------------------------
    # BƯỚC E: TẠO THÊM 2 BIẾN QUAN TRỌNG THEO CHUẨN DA - SGU (TASK 1.5)
    # --------------------------------------------------------------------------
    print("\n>>> [E] Tạo biến trễ Y_{t-1} và biến định danh Mùa vụ X7...")
    
    # 1. Biến trễ tự hồi quy AR: VN_Coffee_Price_Lag1
    # Dùng hàm .shift(1) lấy giá trị cà phê của tháng liền trước
    df_merged["VN_Coffee_Price_Lag1"] = df_merged["VN_Coffee_Price"].shift(1)
    
    # 2. Biến phân loại Mùa vụ thu hoạch: Harvest_Season (X7)
    # Trích xuất số tháng từ cột Date (chuỗi YYYY-MM)
    df_merged["Month_Int"] = df_merged["Date"].str.slice(5, 7).astype(int)
    # Nếu tháng thuộc [10, 11, 12, 1] là vụ thu hoạch chính ('In_Harvest'), còn lại là 'Off_Harvest'
    df_merged["Harvest_Season"] = np.where(
        df_merged["Month_Int"].isin([10, 11, 12, 1]), "In_Harvest", "Off_Harvest"
    )
    # Xóa cột phụ Month_Int
    df_merged = df_merged.drop(columns=["Month_Int"])

    # --------------------------------------------------------------------------
    # BƯỚC F: SẮP XẾP VÀ XUẤT TẬP TIN data/raw/vn_coffee_merged_raw.csv (TASK 1.6)
    # --------------------------------------------------------------------------
    duong_dan_xuat = "data/raw/vn_coffee_merged_raw.csv"
    
    # Thứ tự các cột theo chuẩn Data Dictionary
    cot_chuan = [
        "Date",
        "VN_Coffee_Price",
        "VN_Coffee_Price_Lag1",
        "London_Robusta_Price",
        "VN_Production",
        "Crude_Oil_Price",
        "Harvest_Season"
    ]
    df_final = df_merged[cot_chuan].copy()
    
    # Lưu file CSV không lưu cột index
    df_final.to_csv(duong_dan_xuat, index=False, encoding="utf-8")
    print(f"\n[THÀNH CÔNG] Đã xuất file tích hợp: {duong_dan_xuat}")
    print(f"  + Tổng số dòng (quan sát tháng): {len(df_final)}")
    print(f"  + Phạm vi thời gian: {df_final['Date'].min()} đến {df_final['Date'].max()}")
    print(f"  + 5 dòng đầu tiên của tập dữ liệu sau tích hợp:\n{df_final.head()}")

    # --------------------------------------------------------------------------
    # BƯỚC G: CHẠY SELF-TEST (TỰ ĐỘNG KIỂM THỬ BẰNG ASSERT THEO QUY TẮC SỐ 5)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CHẠY SELF-TEST (TỰ KIỂM TRA BẢO ĐẢM TÍNH TOÀN VẸN)")
    print("=" * 80)
    
    # 1. Kiểm tra sự tồn tại của file xuất
    assert os.path.exists(duong_dan_xuat), f"Lỗi Self-Test: Không tìm thấy file {duong_dan_xuat}"
    
    # 2. Kiểm tra số lượng bản ghi >= 36 tháng
    so_dong = len(df_final)
    assert so_dong >= 36, f"Lỗi Self-Test: Số dòng ({so_dong}) nhỏ hơn ngưỡng tối thiểu 36 tháng!"
    print(f"[PASS] Điều kiện số dòng: Đạt {so_dong} tháng (>= 36 tháng theo yêu cầu).")

    # 3. Kiểm tra danh sách các cột bắt buộc
    for col in cot_chuan:
        assert col in df_final.columns, f"Lỗi Self-Test: Cột '{col}' không tồn tại trong bảng kết quả!"
    print(f"[PASS] Điều kiện cấu trúc cột: Đầy đủ 100% {len(cot_chuan)} cột chuẩn.")

    # 4. Kiểm tra tính liên tục và đơn điệu tăng dần của cột thời gian Date
    assert df_final["Date"].is_monotonic_increasing, "Lỗi Self-Test: Cột Date không tăng dần theo thời gian!"
    print("[PASS] Điều kiện thứ tự thời gian: Cột Date đơn điệu tăng dần 100%.")

    # 5. Kiểm tra giá trị không âm đối với các biến giá và sản lượng
    assert (df_final["VN_Coffee_Price"] > 0).all(), "Lỗi Self-Test: Có giá trị VN_Coffee_Price âm!"
    assert (df_final["London_Robusta_Price"] > 0).all(), "Lỗi Self-Test: Có giá trị London_Robusta_Price âm!"
    assert (df_final["VN_Production"] > 0).all(), "Lỗi Self-Test: Có giá trị VN_Production âm!"
    assert (df_final["Crude_Oil_Price"] > 0).all(), "Lỗi Self-Test: Có giá trị Crude_Oil_Price âm!"
    print("[PASS] Điều kiện miền giá trị: Toàn bộ giá trị kinh tế đều dương hợp lý.")

    # 6. Kiểm tra giá trị biến định danh Harvest_Season
    tap_gia_tri_mua_vu = set(df_final["Harvest_Season"].unique())
    assert tap_gia_tri_mua_vu == {"In_Harvest", "Off_Harvest"}, f"Lỗi Self-Test: Biến Harvest_Season có nhãn bất thường: {tap_gia_tri_mua_vu}"
    print(f"[PASS] Điều kiện biến định danh: Biến Harvest_Season có đủ 2 nhãn: {tap_gia_tri_mua_vu}.")

    print("-" * 80)
    print("SELF-TEST BƯỚC 1: PASS 100% TOÀN BỘ CÁC TIÊU CHÍ!")
    print("=" * 80)
    
    return df_final

if __name__ == "__main__":
    kiem_ke_tap_tin_tho()
    dong_bo_va_tich_hop_du_lieu()
