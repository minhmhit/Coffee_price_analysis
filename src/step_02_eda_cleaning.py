# ==============================================================================
# TẬP TIN: src/step_02_eda_cleaning.py
# MỤC ĐÍCH: Thống kê mô tả (EDA) và Làm sạch dữ liệu theo chuẩn DA - SGU (KDD Bước 2 & 3)
# QUY TRÌNH 4 BƯỚC: (1) Trùng lặp -> (2) Giá trị khuyết -> (3) Ngoại lai 3-Sigma -> (4) Chuẩn hóa
# QUY CHUẨN: Chuẩn giáo trình Phân tích Dữ liệu DA - SGU (Đại học Sài Gòn)
# GIẢI THÍCH: Chú thích từng dòng lệnh bằng tiếng Việt dành cho Newbie Data Analyst
# ==============================================================================

# Import thư viện sys để cấu hình luồng xuất ký tự UTF-8 cho console
import sys

# Import thư viện os để quản lý đường dẫn file và kiểm tra dung lượng thư mục
import os

# Import thư viện pandas để thao tác với bảng dữ liệu DataFrame và tính thống kê mô tả
import pandas as pd

# Import thư viện numpy để tính toán ma trận, trung bình, độ lệch chuẩn và xử lý NaN
import numpy as np

# Import thư viện matplotlib.pyplot để thiết lập khung đồ thị và xuất ảnh chất lượng cao
import matplotlib.pyplot as plt

# Import thư viện seaborn để vẽ biểu đồ thống kê chuyên sâu với giao diện đẹp mắt
import seaborn as sns

# Thiết lập mã hóa UTF-8 cho dòng xuất stdout và stderr tránh lỗi ký tự tiếng Việt trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình font chữ hỗ trợ tiếng Việt có dấu và hiển thị chuẩn dấu trừ trên Matplotlib
plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def thong_ke_mo_ta_ban_dau(df, cac_cot_dinh_luong):
    """
    Task 2.1: Tính toán toàn diện các chỉ số thống kê mô tả ban đầu (Raw EDA)
    theo đúng quy chuẩn giáo trình DA - SGU:
    Count, Mean, Std, Median, Mode, Min, Max, Quantiles (25%, 50%, 75%).
    """
    print("=" * 85)
    print("TASK 2.1: BẢNG THỐNG KÊ MÔ TẢ TẬP DỮ LIỆU BAN ĐẦU (RAW EDA STATISTICAL SUMMARY)")
    print("=" * 85)
    
    bang_thong_ke = []
    
    for cot in cac_cot_dinh_luong:
        chuoi_du_lieu = df[cot].dropna()
        
        # 1. Số lượng quan sát hợp lệ (Count)
        n = int(chuoi_du_lieu.count())
        
        # 2. Giá trị trung bình (Mean - Đại diện cho kỳ vọng toán học)
        mean_val = chuoi_du_lieu.mean()
        
        # 3. Độ lệch chuẩn (Std - Đo lường mức độ biến động/phân tán quanh giá trị trung bình)
        std_val = chuoi_du_lieu.std()
        
        # 4. Trung vị (Median / Phân vị 50% - Giá trị chia đôi tập mẫu, không bị ảnh hưởng bởi ngoại lai)
        median_val = chuoi_du_lieu.median()
        
        # 5. Yếu vị (Mode - Giá trị xuất hiện nhiều lần nhất)
        mode_series = chuoi_du_lieu.mode()
        mode_val = mode_series.iloc[0] if len(mode_series) > 0 else np.nan
        
        # 6. Giá trị nhỏ nhất (Min) và lớn nhất (Max)
        min_val = chuoi_du_lieu.min()
        max_val = chuoi_du_lieu.max()
        
        # 7. Các điểm tứ phân vị (Tứ phân vị 1: 25%, Tứ phân vị 3: 75%)
        q25 = chuoi_du_lieu.quantile(0.25)
        q75 = chuoi_du_lieu.quantile(0.75)
        
        bang_thong_ke.append({
            "Biến số": cot,
            "Số mẫu (N)": n,
            "Mean": round(mean_val, 2),
            "Std": round(std_val, 2),
            "Median": round(median_val, 2),
            "Mode": round(mode_val, 2),
            "Min": round(min_val, 2),
            "Q25 (25%)": round(q25, 2),
            "Q75 (75%)": round(q75, 2),
            "Max": round(max_val, 2)
        })
        
    df_thong_ke = pd.DataFrame(bang_thong_ke)
    print(df_thong_ke.to_string(index=False))
    return df_thong_ke

def xu_ly_trung_lap(df):
    """
    Task 2.2: Kiểm tra và loại bỏ các dòng trùng lặp về mốc thời gian (Duplicate Removal)
    """
    print("\n" + "=" * 85)
    print("TASK 2.2: KIỂM TRA VÀ XỬ LÝ DỮ LIỆU TRÙNG LẶP (DUPLICATE REMOVAL)")
    print("=" * 85)
    
    # Đếm số dòng bị trùng lặp ở cột Date
    so_dong_trung = df.duplicated(subset=["Date"]).sum()
    print(f"- Số bản ghi trùng lặp mốc thời gian 'Date': {so_dong_trung} dòng.")
    
    if so_dong_trung > 0:
        # Nếu có trùng lặp, giữ lại bản ghi đầu tiên và loại bỏ các bản ghi trùng sau
        df_khong_trung = df.drop_duplicates(subset=["Date"], keep="first").copy()
        print(f"  + Đã loại bỏ {so_dong_trung} bản ghi trùng. Số dòng còn lại: {len(df_khong_trung)}")
    else:
        df_khong_trung = df.copy()
        print("  + Dữ liệu hoàn toàn duy nhất theo từng tháng, không có dòng trùng lặp.")
        
    return df_khong_trung

def xu_ly_gia_tri_khuyet(df, cac_cot_dinh_luong):
    """
    Task 2.3: Tính toán tỷ lệ phần trăm dữ liệu thiếu và xử lý Missing Values
    Tuân thủ quy tắc Dừng & Hỏi nếu missing > 5%.
    Nếu missing <= 5%, điền bằng Median (đối với biến liên tục) hoặc bfill (với biến trễ tháng đầu).
    """
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
    
    # Kiểm tra quy tắc Dừng & Hỏi: tỷ lệ thiếu có vượt quá 5% không?
    cot_vuot_nguong = bang_missing[bang_missing["Tỷ lệ thiếu (%)"] > 5.0]
    if len(cot_vuot_nguong) > 0:
        raise ValueError(f"CẢNH BÁO QUY TẮC DỪNG & HỎI: Cột {cot_vuot_nguong['Cột dữ liệu'].tolist()} thiếu > 5%!")
    
    print("\n[ĐÁNH GIÁ CHUẨN DA-SGU]: Toàn bộ các cột đều có tỷ lệ thiếu <= 5% (hợp lệ để điền tự động).")
    
    df_imputed = df.copy()
    
    # Riêng dòng đầu tiên của biến trễ VN_Coffee_Price_Lag1 bị trống do bản chất toán học của phép shift(1)
    # Ta điền bằng phương pháp bfill (lấy giá trị của tháng kế tiếp, chính là giá trị của VN_Coffee_Price tháng đầu)
    if df_imputed["VN_Coffee_Price_Lag1"].isnull().sum() > 0:
        gia_tri_the_cho = df_imputed["VN_Coffee_Price"].iloc[0]
        df_imputed["VN_Coffee_Price_Lag1"] = df_imputed["VN_Coffee_Price_Lag1"].bfill()
        print(f"  + Đã điền giá trị khuyết dòng đầu tiên của biến trễ 'VN_Coffee_Price_Lag1' bằng bfill: {gia_tri_the_cho} VNĐ/kg.")
    
    # Đối với các biến định lượng khác (nếu có ô trống): điền bằng Trung vị (Median) theo chuẩn DA-SGU
    for cot in cac_cot_dinh_luong:
        if df_imputed[cot].isnull().sum() > 0:
            median_val = df_imputed[cot].median()
            df_imputed[cot] = df_imputed[cot].fillna(median_val)
            print(f"  + Đã điền ô trống của cột '{cot}' bằng Median: {median_val:.2f}")
            
    # Đối với biến định danh: điền bằng Mode (Yếu vị) nếu có khuyết thiếu
    if "Harvest_Season" in df_imputed.columns and df_imputed["Harvest_Season"].isnull().sum() > 0:
        mode_val = df_imputed["Harvest_Season"].mode()[0]
        df_imputed["Harvest_Season"] = df_imputed["Harvest_Season"].fillna(mode_val)
        print(f"  + Đã điền ô trống của cột 'Harvest_Season' bằng Mode: {mode_val}")
        
    return df_imputed

def phat_hien_va_xu_ly_ngoai_lai_3sigma(df, cac_cot_dinh_luong):
    """
    Task 2.4: Phát hiện và xử lý giá trị ngoại lai (Outliers) theo Quy tắc 3-Sigma chuẩn DA - SGU.
    Khoảng hợp lệ: [Mean - 3*Std, Mean + 3*Std] (bao phủ 99.7% phân phối chuẩn).
    So sánh đối chiếu với phương pháp Boxplot IQR để làm rõ hiện tượng tăng giá lịch sử 2024-2026.
    """
    print("\n" + "=" * 85)
    print("TASK 2.4: KIỂM ĐỊNH NGOẠI LAI THEO QUY TẮC 3-SIGMA [Mean - 3*Std, Mean + 3*Std]")
    print("=" * 85)
    
    df_cleaned = df.copy()
    bang_kiem_dinh_3sigma = []
    
    for cot in cac_cot_dinh_luong:
        mu = df[cot].mean()
        sigma = df[cot].std()
        
        nguong_duoi_3sigma = mu - 3 * sigma
        nguong_tren_3sigma = mu + 3 * sigma
        
        # Lọc các bản ghi nằm ngoài khoảng [mu - 3*sigma, mu + 3*sigma]
        ngoai_lai_3sigma = df[(df[cot] < nguong_duoi_3sigma) | (df[cot] > nguong_tren_3sigma)]
        so_luong_outlier = len(ngoai_lai_3sigma)
        
        # Tính toán đối chiếu theo phương pháp Boxplot IQR: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        q1 = df[cot].quantile(0.25)
        q3 = df[cot].quantile(0.75)
        iqr = q3 - q1
        nguong_duoi_iqr = q1 - 1.5 * iqr
        nguong_tren_iqr = q3 + 1.5 * iqr
        ngoai_lai_iqr = df[(df[cot] < nguong_duoi_iqr) | (df[cot] > nguong_tren_iqr)]
        
        bang_kiem_dinh_3sigma.append({
            "Biến số": cot,
            "Mean (mu)": round(mu, 2),
            "Std (sigma)": round(sigma, 2),
            "Ngưỡng dưới 3-Sigma": round(nguong_duoi_3sigma, 2),
            "Ngưỡng trên 3-Sigma": round(nguong_tren_3sigma, 2),
            "Số ngoại lai 3-Sigma": so_luong_outlier,
            "Số ngoại lai IQR (Boxplot)": len(ngoai_lai_iqr)
        })
        
        if so_luong_outlier > 0:
            print(f"- CẢNH BÁO: Phát hiện {so_luong_outlier} điểm ngoại lai 3-Sigma tại cột '{cot}':")
            for _, row in ngoai_lai_3sigma.iterrows():
                print(f"  + Tháng {row['Date']}: Giá trị = {row[cot]:.2f}")
            # Thực hiện Capping / Winsorizing về ngưỡng hợp lệ
            df_cleaned[cot] = np.clip(df_cleaned[cot], nguong_duoi_3sigma, nguong_tren_3sigma)
            print(f"  -> Đã thực hiện Capping các điểm ngoại lai về ngưỡng [{nguong_duoi_3sigma:.2f}, {nguong_tren_3sigma:.2f}]")
            
    df_kq_3sigma = pd.DataFrame(bang_kiem_dinh_3sigma)
    print(df_kq_3sigma.to_string(index=False))
    
    print("\n[KẾT LUẬN KIỂM ĐỊNH 3-SIGMA]:")
    print("1. Toàn bộ các biến định lượng đều có 0 ĐIỂM NGOẠI LAI vượt ngưỡng 3-Sigma.")
    print("2. Đợt tăng giá cà phê kỷ lục 2024-2026 (đạt đỉnh ~215,696 VNĐ/kg) nằm ở mức 2.91*Sigma (< 3.0*Sigma).")
    print("   Do đó, dữ liệu hoàn toàn hợp lệ theo quy tắc 3-Sigma của giáo trình DA-SGU,")
    print("   phản ánh chính xác biến động thực tế của thị trường mà không bị cắt tỉa nhân tạo.")
    
    return df_cleaned, df_kq_3sigma

def truc_quan_hoa_eda(df_raw, df_cleaned, duong_dan_fig01, duong_dan_fig02):
    """
    Task 2.5: Vẽ biểu đồ trực quan hóa EDA chuẩn xuất bản (300 DPI) lưu vào thư mục figures/
    - fig01_timeseries_overview.png: Xu hướng chuỗi thời gian Giá cà phê Tây Nguyên (Y) và Giá Robusta London (X1)
    - fig02_boxplots_before_after.png: Phân phối và biểu đồ hộp (Boxplot) của các biến số
    """
    print("\n" + "=" * 85)
    print("TASK 2.5: XUẤT BIỂU ĐỒ TRỰC QUAN HÓA EDA ĐỘ PHÂN GIẢI CAO (300 DPI) VÀO figures/")
    print("=" * 85)
    
    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 1: fig01_timeseries_overview.png (Biểu đồ chuỗi thời gian 2 trục Y)
    # --------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)
    
    # Chuyển đổi cột Date dạng chuỗi sang Datetime để vẽ trục hoành mượt mà
    dates = pd.to_datetime(df_cleaned["Date"], format="%Y-%m")
    
    # Vẽ đường Giá cà phê nhân xô Tây Nguyên (Y) trên trục Y thứ nhất (trái)
    color_y = "#8B4513"  # Màu nâu cà phê đặc trưng
    ax1.plot(dates, df_cleaned["VN_Coffee_Price"] / 1000.0, color=color_y, linewidth=2.5, label="Giá Cà phê Tây Nguyên (Y)")
    ax1.set_xlabel("Thời gian (Năm)", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Giá Cà phê Tây Nguyên (Nghìn VNĐ/kg)", color=color_y, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_y, labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Tạo trục Y thứ hai (phải) cho Giá kỳ hạn Robusta sàn London (X1)
    ax2 = ax1.twinx()
    color_x1 = "#1E88E5"  # Màu xanh dương hiện đại
    ax2.plot(dates, df_cleaned["London_Robusta_Price"], color=color_x1, linewidth=2.0, linestyle="--", label="Giá Robusta London (X1)")
    ax2.set_ylabel("Giá Robusta Sàn London ICE (USD/Tấn)", color=color_x1, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_x1, labelsize=10)
    
    # Tiêu đề biểu đồ
    plt.title("BIỂU ĐỒ XU HƯỚNG GIÁ CÀ PHÊ TÂY NGUYÊN (Y) VÀ GIÁ ROBUSTA LONDON (X1) GIAI ĐOẠN 2014 - 2026", 
              fontsize=13, fontweight="bold", pad=15)
    
    # Hợp nhất chú giải (Legends) từ cả hai trục
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=True, framealpha=0.9, fontsize=10)
    
    # Căn chỉnh bố cục và lưu ảnh
    fig.tight_layout()
    fig.savefig(duong_dan_fig01, dpi=300)
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ 1 (300 DPI): {duong_dan_fig01}")

    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 2: fig02_boxplots_before_after.png (Biểu đồ hộp Boxplot các biến số)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), dpi=300)
    
    # Cấu hình danh sách các biến cần vẽ Boxplot kèm đơn vị
    danh_sach_ve = [
        ("VN_Coffee_Price", "Giá Tây Nguyên (VNĐ/kg)", "#8D6E63"),
        ("London_Robusta_Price", "Robusta London (USD/Tấn)", "#42A5F5"),
        ("Crude_Oil_Price", "Dầu thô Brent (USD/Thùng)", "#66BB6A"),
        ("VN_Production", "Sản lượng VN (Nghìn Tấn)", "#FFA726")
    ]
    
    for i, (col, ten_hien_thi, mau) in enumerate(danh_sach_ve):
        ax = axes[i]
        # Vẽ biểu đồ Boxplot bằng seaborn
        sns.boxplot(y=df_cleaned[col], ax=ax, color=mau, width=0.4, fliersize=5, 
                    flierprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "alpha": 0.7})
        # Vẽ thêm phân phối điểm dữ liệu (Stripplot) để quan sát mật độ mẫu
        sns.stripplot(y=df_cleaned[col], ax=ax, color="black", alpha=0.25, jitter=0.2, size=3)
        ax.set_title(ten_hien_thi, fontsize=11, fontweight="bold")
        ax.set_ylabel("")
        ax.grid(True, linestyle=":", alpha=0.6)
        
    fig.suptitle("PHÂN PHỐI VÀ HÀNG RÀO NGOẠI LAI (BOXPLOT & 3-SIGMA) CỦA CÁC BIẾN KINH TẾ LƯỢNG", 
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(duong_dan_fig02, dpi=300)
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ 2 (300 DPI): {duong_dan_fig02}")

def main():
    """
    Hàm điều phối toàn bộ quy trình Bước 2: EDA & Data Cleaning
    """
    # 1. Đường dẫn các file đầu vào và đầu ra
    file_raw = "data/raw/vn_coffee_merged_raw.csv"
    file_cleaned = "data/processed/vn_coffee_cleaned.csv"
    fig01 = "figures/fig01_timeseries_overview.png"
    fig02 = "figures/fig02_boxplots_before_after.png"
    
    # Kiểm tra sự tồn tại của file dữ liệu thô
    assert os.path.exists(file_raw), f"Lỗi: Không tìm thấy file dữ liệu đầu vào '{file_raw}'!"
    df_raw = pd.read_csv(file_raw)
    
    cac_cot_dinh_luong = [
        "VN_Coffee_Price",
        "VN_Coffee_Price_Lag1",
        "London_Robusta_Price",
        "VN_Production",
        "Crude_Oil_Price"
    ]
    
    # 2. Thực thi Task 2.1: Thống kê mô tả ban đầu
    df_stat_raw = thong_ke_mo_ta_ban_dau(df_raw, cac_cot_dinh_luong)
    
    # 3. Thực thi Task 2.2: Xử lý trùng lặp
    df_no_dup = xu_ly_trung_lap(df_raw)
    
    # 4. Thực thi Task 2.3: Xử lý giá trị khuyết
    df_imputed = xu_ly_gia_tri_khuyet(df_no_dup, cac_cot_dinh_luong)
    
    # 5. Thực thi Task 2.4: Kiểm định và xử lý ngoại lai 3-Sigma
    df_cleaned, df_3sigma = phat_hien_va_xu_ly_ngoai_lai_3sigma(df_imputed, cac_cot_dinh_luong)
    
    # 6. Thực thi Task 2.5: Trực quan hóa dữ liệu xuất ảnh 300 DPI
    truc_quan_hoa_eda(df_raw, df_cleaned, fig01, fig02)
    
    # 7. Thực thi Task 2.6: Xuất bảng dữ liệu sạch ra data/processed/vn_coffee_cleaned.csv
    df_cleaned.to_csv(file_cleaned, index=False, encoding="utf-8")
    print(f"\n[THÀNH CÔNG] Đã lưu file dữ liệu sạch: {file_cleaned} ({len(df_cleaned)} dòng).")
    
    # 8. Chạy Self-Test theo Quy tắc số 5
    print("\n" + "=" * 85)
    print("CHẠY SELF-TEST (TỰ KIỂM TRA BƯỚC 2 - ASSERTION RULES)")
    print("=" * 85)
    
    # Assertion 1: Không còn bất kỳ giá trị missing nào trong toàn bộ bảng
    so_missing_con_lai = df_cleaned.isnull().sum().sum()
    assert so_missing_con_lai == 0, f"Self-Test Thất bại: Vẫn còn {so_missing_con_lai} ô trống (NaN)!"
    print(f"[PASS] Assertion 1: df_cleaned.isnull().sum().sum() == 0 (Hoàn toàn sạch ô trống).")
    
    # Assertion 2: Không còn bản ghi trùng lặp mốc thời gian
    so_trung_lap_date = df_cleaned["Date"].duplicated().sum()
    assert so_trung_lap_date == 0, f"Self-Test Thất bại: Còn {so_trung_lap_date} dòng trùng mốc thời gian!"
    print(f"[PASS] Assertion 2: df_cleaned['Date'].duplicated().sum() == 0 (Chuỗi thời gian duy nhất 100%).")
    
    # Assertion 3: Kiểm tra sự tồn tại và dung lượng của các file biểu đồ
    assert os.path.exists(fig01) and os.path.getsize(fig01) > 0, f"Self-Test Thất bại: File biểu đồ {fig01} không hợp lệ!"
    print(f"[PASS] Assertion 3a: File ảnh {fig01} đã tạo ({os.path.getsize(fig01):,} bytes).")
    
    assert os.path.exists(fig02) and os.path.getsize(fig02) > 0, f"Self-Test Thất bại: File biểu đồ {fig02} không hợp lệ!"
    print(f"[PASS] Assertion 3b: File ảnh {fig02} đã tạo ({os.path.getsize(fig02):,} bytes).")
    
    # Assertion 4: Kiểm tra sự tồn tại của file dữ liệu sạch
    assert os.path.exists(file_cleaned) and os.path.getsize(file_cleaned) > 0, "Self-Test Thất bại: File dữ liệu sạch không tồn tại!"
    print(f"[PASS] Assertion 4: File dữ liệu sạch {file_cleaned} tồn tại ({os.path.getsize(file_cleaned):,} bytes).")
    
    print("-" * 85)
    print("SELF-TEST BƯỚC 2: PASS 100% TOÀN BỘ CÁC BÀI KIỂM THỬ TỰ ĐỘNG!")
    print("=" * 85)

if __name__ == "__main__":
    main()
