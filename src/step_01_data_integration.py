import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def kiem_ke_tap_tin_tho():
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
    print("=" * 80)
    print("BẮT ĐẦU ĐỒNG BỘ TẦN SUẤT & TÍCH HỢP DỮ LIỆU ĐA NGUỒN (TASKS 1.3 - 1.6)")
    print("=" * 80)

    # 1. Nguồn 1: Giá cà phê nội địa từ chuỗi ngày (chart_20261008T021435.csv)
    duong_dan_chart = "data/raw/chart_20261008T021435.csv"
    df_chart = pd.read_csv(duong_dan_chart, encoding="utf-8-sig", sep=".", decimal=",")
    df_chart["Date_Parsed"] = pd.to_datetime(df_chart["Date"], format="%m/%d/%Y")
    df_chart["Date"] = df_chart["Date_Parsed"].dt.to_period("M").astype(str)
    
    df_monthly_price = df_chart.groupby("Date", as_index=False)["Value"].mean()
    df_monthly_price = df_monthly_price.rename(columns={"Value": "Price_USD_lb"})
    
    TY_GIA_QUY_UOC = 24000.0
    HE_SO_LB_SANG_KG = 2.20462
    df_monthly_price["VN_Coffee_Price"] = np.round(
        df_monthly_price["Price_USD_lb"] * HE_SO_LB_SANG_KG * TY_GIA_QUY_UOC, 2
    )

    # 2. Nguồn 2: World Bank Pink Sheet (CMO-Historical-Data-Monthly.xlsx)
    duong_dan_cmo = "data/raw/CMO-Historical-Data-Monthly.xlsx"
    df_cmo_raw = pd.read_excel(duong_dan_cmo, sheet_name="Monthly Prices", header=4)
    df_cmo_raw = df_cmo_raw.rename(columns={df_cmo_raw.columns[0]: "Date_Raw"})
    df_cmo_raw = df_cmo_raw.dropna(subset=["Date_Raw"])
    df_cmo_raw["Date"] = df_cmo_raw["Date_Raw"].astype(str).str.replace("M", "-")
    
    df_cmo = df_cmo_raw[["Date", "Coffee, Robusta", "Crude oil, Brent"]].copy()
    df_cmo["Coffee, Robusta"] = pd.to_numeric(df_cmo["Coffee, Robusta"], errors="coerce")
    df_cmo["Crude oil, Brent"] = pd.to_numeric(df_cmo["Crude oil, Brent"], errors="coerce")
    
    df_cmo["London_Robusta_Price"] = np.round(df_cmo["Coffee, Robusta"] * 1000.0, 2)
    df_cmo["Crude_Oil_Price"] = np.round(df_cmo["Crude oil, Brent"], 2)
    df_cmo_clean = df_cmo[["Date", "London_Robusta_Price", "Crude_Oil_Price"]]

    # 3. Nguồn 3: Sản lượng cà phê từ FAOSTAT (FAOSTAT_data_en_10-8-2026.csv)
    duong_dan_fao = "data/raw/FAOSTAT_data_en_10-8-2026.csv"
    df_fao_raw = pd.read_csv(duong_dan_fao)
    fao_vn = df_fao_raw[df_fao_raw["Area"] == "Viet Nam"][["Year", "Unit", "Value"]].copy()
    fao_vn["VN_Production_ThousandTons"] = np.round(fao_vn["Value"] / 1000.0, 2)
    dict_san_luong = dict(zip(fao_vn["Year"].astype(int), fao_vn["VN_Production_ThousandTons"]))
    
    nam_cuoi_cung = max(dict_san_luong.keys())
    san_luong_moi_nhat = dict_san_luong[nam_cuoi_cung]
    dict_san_luong[2025] = san_luong_moi_nhat
    dict_san_luong[2026] = san_luong_moi_nhat

    # 4. Tích hợp các bảng theo Date
    df_merged = pd.merge(df_monthly_price[["Date", "VN_Coffee_Price"]], df_cmo_clean, on="Date", how="inner")
    df_merged = df_merged[df_merged["Date"] >= "2014-01"].copy()
    df_merged = df_merged.sort_values(by="Date").reset_index(drop=True)
    
    df_merged["Year_Int"] = df_merged["Date"].str.slice(0, 4).astype(int)
    df_merged["VN_Production"] = df_merged["Year_Int"].map(dict_san_luong)
    df_merged = df_merged.drop(columns=["Year_Int"])

    # 5. Tạo biến trễ Lag1 và biến Mùa vụ Harvest_Season
    df_merged["VN_Coffee_Price_Lag1"] = df_merged["VN_Coffee_Price"].shift(1)
    df_merged["Month_Int"] = df_merged["Date"].str.slice(5, 7).astype(int)
    df_merged["Harvest_Season"] = np.where(
        df_merged["Month_Int"].isin([10, 11, 12, 1]), "In_Harvest", "Off_Harvest"
    )
    df_merged = df_merged.drop(columns=["Month_Int"])

    # 6. Sắp xếp cấu trúc cột và xuất tệp
    duong_dan_xuat = "data/raw/vn_coffee_merged_raw.csv"
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
    df_final.to_csv(duong_dan_xuat, index=False, encoding="utf-8")

    # 7. Self-Test assertions
    assert os.path.exists(duong_dan_xuat), f"Lỗi Self-Test: Không tìm thấy file {duong_dan_xuat}"
    assert len(df_final) >= 36, f"Lỗi Self-Test: Số dòng nhỏ hơn 36 tháng!"
    for col in cot_chuan:
        assert col in df_final.columns, f"Lỗi Self-Test: Cột '{col}' không tồn tại!"
    assert df_final["Date"].is_monotonic_increasing, "Lỗi Self-Test: Cột Date không tăng dần!"
    assert (df_final["VN_Coffee_Price"] > 0).all(), "Lỗi Self-Test: Có giá trị VN_Coffee_Price âm!"
    assert (df_final["London_Robusta_Price"] > 0).all(), "Lỗi Self-Test: Có giá trị London_Robusta_Price âm!"
    assert (df_final["VN_Production"] > 0).all(), "Lỗi Self-Test: Có giá trị VN_Production âm!"
    assert (df_final["Crude_Oil_Price"] > 0).all(), "Lỗi Self-Test: Có giá trị Crude_Oil_Price âm!"
    assert set(df_final["Harvest_Season"].unique()) == {"In_Harvest", "Off_Harvest"}

    print(f"[THÀNH CÔNG] Đã tích hợp {len(df_final)} tháng vào: {duong_dan_xuat}")
    print("SELF-TEST BƯỚC 1: PASS 100%!")
    return df_final

if __name__ == "__main__":
    kiem_ke_tap_tin_tho()
    dong_bo_va_tich_hop_du_lieu()
