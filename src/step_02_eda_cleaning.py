import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def thong_ke_mo_ta_ban_dau(df, cac_cot_dinh_luong):
    print("=" * 85)
    print("TASK 2.1: BẢNG THỐNG KÊ MÔ TẢ TẬP DỮ LIỆU BAN ĐẦU (RAW EDA STATISTICAL SUMMARY)")
    print("=" * 85)
    
    bang_thong_ke = []
    for cot in cac_cot_dinh_luong:
        chuoi = df[cot].dropna()
        mode_series = chuoi.mode()
        bang_thong_ke.append({
            "Biến số": cot,
            "Số mẫu (N)": int(chuoi.count()),
            "Mean": round(chuoi.mean(), 2),
            "Std": round(chuoi.std(), 2),
            "Median": round(chuoi.median(), 2),
            "Mode": round(mode_series.iloc[0] if len(mode_series) > 0 else np.nan, 2),
            "Min": round(chuoi.min(), 2),
            "Q25 (25%)": round(chuoi.quantile(0.25), 2),
            "Q75 (75%)": round(chuoi.quantile(0.75), 2),
            "Max": round(chuoi.max(), 2)
        })
        
    df_thong_ke = pd.DataFrame(bang_thong_ke)
    print(df_thong_ke.to_string(index=False))
    return df_thong_ke

def xu_ly_trung_lap(df):
    print("\n" + "=" * 85)
    print("TASK 2.2: KIỂM TRA VÀ XỬ LÝ DỮ LIỆU TRÙNG LẶP (DUPLICATE REMOVAL)")
    print("=" * 85)
    
    so_dong_trung = df.duplicated(subset=["Date"]).sum()
    print(f"- Số bản ghi trùng lặp mốc thời gian 'Date': {so_dong_trung} dòng.")
    if so_dong_trung > 0:
        df_khong_trung = df.drop_duplicates(subset=["Date"], keep="first").copy()
    else:
        df_khong_trung = df.copy()
    return df_khong_trung

def xu_ly_gia_tri_khuyet(df, cac_cot_dinh_luong):
    print("\n" + "=" * 85)
    print("TASK 2.3: KIỂM TRA VÀ XỬ LÝ GIÁ TRỊ KHUYẾT (MISSING VALUES IMPUTATION)")
    print("=" * 85)
    
    tong_so_dong = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / tong_so_dong) * 100.0
    
    bang_missing = pd.DataFrame({
        "Cột dữ liệu": df.columns,
        "Số ô trống": missing_count.values,
        "Tỷ lệ thiếu (%)": np.round(missing_pct.values, 2)
    })
    print(bang_missing.to_string(index=False))
    
    cot_vuot_nguong = bang_missing[bang_missing["Tỷ lệ thiếu (%)"] > 5.0]
    if len(cot_vuot_nguong) > 0:
        raise ValueError(f"CẢNH BÁO: Cột {cot_vuot_nguong['Cột dữ liệu'].tolist()} thiếu > 5%!")
        
    df_imputed = df.copy()
    if df_imputed["VN_Coffee_Price_Lag1"].isnull().sum() > 0:
        df_imputed["VN_Coffee_Price_Lag1"] = df_imputed["VN_Coffee_Price_Lag1"].bfill()
        
    for cot in cac_cot_dinh_luong:
        if df_imputed[cot].isnull().sum() > 0:
            df_imputed[cot] = df_imputed[cot].fillna(df_imputed[cot].median())
            
    if "Harvest_Season" in df_imputed.columns and df_imputed["Harvest_Season"].isnull().sum() > 0:
        df_imputed["Harvest_Season"] = df_imputed["Harvest_Season"].fillna(df_imputed["Harvest_Season"].mode()[0])
        
    return df_imputed

def phat_hien_va_xu_ly_ngoai_lai_3sigma(df, cac_cot_dinh_luong):
    print("\n" + "=" * 85)
    print("TASK 2.4: KIỂM ĐỊNH NGOẠI LAI THEO QUY TẮC 3-SIGMA [Mean - 3*Std, Mean + 3*Std]")
    print("=" * 85)
    
    df_cleaned = df.copy()
    bang_kiem_dinh = []
    
    for cot in cac_cot_dinh_luong:
        mu = df[cot].mean()
        sigma = df[cot].std()
        nguong_duoi = mu - 3 * sigma
        nguong_tren = mu + 3 * sigma
        
        ngoai_lai = df[(df[cot] < nguong_duoi) | (df[cot] > nguong_tren)]
        q1 = df[cot].quantile(0.25)
        q3 = df[cot].quantile(0.75)
        iqr = q3 - q1
        ngoai_lai_iqr = df[(df[cot] < q1 - 1.5 * iqr) | (df[cot] > q3 + 1.5 * iqr)]
        
        bang_kiem_dinh.append({
            "Biến số": cot,
            "Mean": round(mu, 2),
            "Std": round(sigma, 2),
            "Ngưỡng dưới 3-Sigma": round(nguong_duoi, 2),
            "Ngưỡng trên 3-Sigma": round(nguong_tren, 2),
            "Số ngoại lai 3-Sigma": len(ngoai_lai),
            "Số ngoại lai IQR (Boxplot)": len(ngoai_lai_iqr)
        })
        
        if len(ngoai_lai) > 0:
            df_cleaned[cot] = np.clip(df_cleaned[cot], nguong_duoi, nguong_tren)
            
    df_kq_3sigma = pd.DataFrame(bang_kiem_dinh)
    print(df_kq_3sigma.to_string(index=False))
    return df_cleaned, df_kq_3sigma

def truc_quan_hoa_eda(df_raw, df_cleaned, duong_dan_fig01, duong_dan_fig02):
    print("\n" + "=" * 85)
    print("TASK 2.5: XUẤT BIỂU ĐỒ TRỰC QUAN HÓA EDA (300 DPI) VÀO figures/")
    print("=" * 85)
    
    # 1. Biểu đồ chuỗi thời gian 2 trục Y (fig01)
    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)
    dates = pd.to_datetime(df_cleaned["Date"], format="%Y-%m")
    
    color_y = "#8B4513"
    ax1.plot(dates, df_cleaned["VN_Coffee_Price"] / 1000.0, color=color_y, linewidth=2.5, label="Giá Cà phê Tây Nguyên (Y)")
    ax1.set_xlabel("Thời gian (Năm)", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Giá Cà phê Tây Nguyên (Nghìn VNĐ/kg)", color=color_y, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_y, labelsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color_x1 = "#1E88E5"
    ax2.plot(dates, df_cleaned["London_Robusta_Price"], color=color_x1, linewidth=2.0, linestyle="--", label="Giá Robusta London (X1)")
    ax2.set_ylabel("Giá Robusta Sàn London ICE (USD/Tấn)", color=color_x1, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_x1, labelsize=10)
    
    plt.title("BIỂU ĐỒ XU HƯỚNG GIÁ CÀ PHÊ TÂY NGUYÊN (Y) VÀ GIÁ ROBUSTA LONDON (X1) GIAI ĐOẠN 2014 - 2026", 
              fontsize=13, fontweight="bold", pad=15)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=True, framealpha=0.9, fontsize=10)
    
    fig.tight_layout()
    fig.savefig(duong_dan_fig01, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Biểu đồ Boxplot phân phối các biến (fig02)
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), dpi=300)
    danh_sach_ve = [
        ("VN_Coffee_Price", "Giá Tây Nguyên (VNĐ/kg)", "#8D6E63"),
        ("London_Robusta_Price", "Robusta London (USD/Tấn)", "#42A5F5"),
        ("Crude_Oil_Price", "Dầu thô Brent (USD/Thùng)", "#66BB6A"),
        ("VN_Production", "Sản lượng VN (Nghìn Tấn)", "#FFA726")
    ]
    
    for i, (col, ten_hien_thi, mau) in enumerate(danh_sach_ve):
        ax = axes[i]
        sns.boxplot(y=df_cleaned[col], ax=ax, color=mau, width=0.4, fliersize=5, 
                    flierprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "alpha": 0.7})
        sns.stripplot(y=df_cleaned[col], ax=ax, color="black", alpha=0.25, jitter=0.2, size=3)
        ax.set_title(ten_hien_thi, fontsize=11, fontweight="bold")
        ax.set_ylabel("")
        ax.grid(True, linestyle=":", alpha=0.6)
        
    fig.suptitle("PHÂN PHỐI VÀ HÀNG RÀO NGOẠI LAI (BOXPLOT & 3-SIGMA) CỦA CÁC BIẾN KINH TẾ LƯỢNG", 
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(duong_dan_fig02, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    file_raw = "data/raw/vn_coffee_merged_raw.csv"
    file_cleaned = "data/processed/vn_coffee_cleaned.csv"
    fig01 = "figures/fig01_timeseries_overview.png"
    fig02 = "figures/fig02_boxplots_before_after.png"
    
    assert os.path.exists(file_raw), f"Lỗi: Không tìm thấy '{file_raw}'!"
    df_raw = pd.read_csv(file_raw)
    
    cac_cot_dinh_luong = [
        "VN_Coffee_Price",
        "VN_Coffee_Price_Lag1",
        "London_Robusta_Price",
        "VN_Production",
        "Crude_Oil_Price"
    ]
    
    df_stat_raw = thong_ke_mo_ta_ban_dau(df_raw, cac_cot_dinh_luong)
    df_no_dup = xu_ly_trung_lap(df_raw)
    df_imputed = xu_ly_gia_tri_khuyet(df_no_dup, cac_cot_dinh_luong)
    df_cleaned, df_3sigma = phat_hien_va_xu_ly_ngoai_lai_3sigma(df_imputed, cac_cot_dinh_luong)
    truc_quan_hoa_eda(df_raw, df_cleaned, fig01, fig02)
    
    df_cleaned.to_csv(file_cleaned, index=False, encoding="utf-8")
    
    assert df_cleaned.isnull().sum().sum() == 0, "Self-Test Lỗi: Vẫn còn ô trống (NaN)!"
    assert df_cleaned["Date"].duplicated().sum() == 0, "Self-Test Lỗi: Trùng mốc thời gian!"
    assert os.path.exists(fig01) and os.path.getsize(fig01) > 0, f"Lỗi ảnh {fig01}"
    assert os.path.exists(fig02) and os.path.getsize(fig02) > 0, f"Lỗi ảnh {fig02}"
    assert os.path.exists(file_cleaned) and os.path.getsize(file_cleaned) > 0, "Lỗi file cleaned rỗng!"
    
    print(f"\n[THÀNH CÔNG] Đã lưu file sạch: {file_cleaned} ({len(df_cleaned)} dòng).")
    print("SELF-TEST BƯỚC 2: PASS 100%!")

if __name__ == "__main__":
    main()
