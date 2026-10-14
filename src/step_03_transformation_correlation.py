# ==============================================================================
# TẬP TIN: src/step_03_transformation_correlation.py
# MỤC ĐÍCH: Biến đổi dữ liệu, mã hóa biến định danh, rời rạc hóa và phân tích tương quan Pearson (KDD Bước 4)
# QUY CHUẨN: Giáo trình Phân tích Dữ liệu DA - SGU (Đại học Sài Gòn)
# KỸ THUẬT: One-Hot Encoding, Label Encoding, pd.cut(), StandardScaler(), pearsonr()
# GIẢI THÍCH: Chú thích từng dòng lệnh bằng tiếng Việt dành cho Newbie Data Analyst
# ==============================================================================

# Import thư viện sys để cấu hình hiển thị tiếng Việt UTF-8 trên Windows console
import sys

# Import thư viện os để quản lý đường dẫn file và kiểm tra dung lượng thư mục
import os

# Import thư viện pandas để thao tác bảng dữ liệu và thực hiện biến đổi
import pandas as pd

# Import thư viện numpy để tính toán số học, trung bình và độ lệch chuẩn
import numpy as np

# Import thư viện scipy.stats để thực hiện kiểm định tương quan Pearson và tính p-value
from scipy.stats import pearsonr

# Import thư viện sklearn.preprocessing để sử dụng LabelEncoder và StandardScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import matplotlib.pyplot và seaborn để vẽ biểu đồ ma trận tương quan và phân tán
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập bảng mã UTF-8 cho dòng xuất stdout/stderr trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình font chữ hỗ trợ tiếng Việt có dấu và hiển thị chuẩn dấu trừ trên biểu đồ
plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def thuc_hien_roi_rac_hoa_va_label_encoding(df):
    """
    Task 3.1: Thực hành kỹ thuật Rời rạc hóa (Discretization) bằng pd.cut()
    và Mã hóa nhãn (Label Encoding) bằng LabelEncoder() theo chuẩn DA - SGU.
    Áp dụng phân chia biến định lượng (Crude_Oil_Price / Lượng mưa đại diện) thành 3 khoảng: 'Thap', 'Trung_Binh', 'Cao'.
    """
    print("=" * 85)
    print("TASK 3.1: RỜI RẠC HÓA DỮ LIỆU (pd.cut) & MÃ HÓA NHÃN (LabelEncoder)")
    print("=" * 85)

    df_trans = df.copy()

    # Sử dụng pd.cut để chia dải giá trị của Crude_Oil_Price thành 3 khoảng bằng nhau
    # 'Thap' (Thấp), 'Trung_Binh' (Trung bình), 'Cao' (Cao)
    nhan_danh_muc = ["Thap", "Trung_Binh", "Cao"]
    df_trans["Crude_Oil_Category"] = pd.cut(
        df_trans["Crude_Oil_Price"], 
        bins=3, 
        labels=nhan_danh_muc
    )
    
    print("- Kết quả phân chia rời rạc hóa biến Crude_Oil_Price bằng pd.cut():")
    print(df_trans["Crude_Oil_Category"].value_counts())

    # Khởi tạo đối tượng LabelEncoder từ thư viện scikit-learn
    le = LabelEncoder()
    # Chuyển đổi nhãn chuỗi ['Thap', 'Trung_Binh', 'Cao'] thành các số nguyên [0, 1, 2]
    df_trans["Crude_Oil_Encoded"] = le.fit_transform(df_trans["Crude_Oil_Category"].astype(str))
    
    print("\n- Ánh xạ mã hóa LabelEncoder (Số hóa nhãn phân loại):")
    for idx, label in enumerate(le.classes_):
        print(f"  + Nhãn '{label}' -> Giá trị số: {idx}")

    return df_trans, le

def thuc_hien_one_hot_encoding(df):
    """
    Task 3.2: Mã hóa biến định danh Mùa vụ thu hoạch (Harvest_Season) bằng pd.get_dummies()
    Sử dụng tham số drop_first=True để loại bỏ 1 cột cơ sở, tránh bẫy biến giả (Dummy Variable Trap).
    """
    print("\n" + "=" * 85)
    print("TASK 3.2: MÃ HÓA BIẾN ĐỊNH DANH BẰNG ONE-HOT ENCODING (pd.get_dummies)")
    print("=" * 85)

    df_encoded = df.copy()

    # Định nghĩa thứ tự phân loại để khi drop_first=True, nhãn 'Off_Harvest' làm cơ sở (0),
    # và nhãn 'In_Harvest' giữ lại làm biến đại diện (1)
    df_encoded["Harvest_Season"] = pd.Categorical(
        df_encoded["Harvest_Season"], 
        categories=["Off_Harvest", "In_Harvest"]
    )

    # Sử dụng pd.get_dummies với drop_first=True và dtype=int theo đúng giáo trình DA-SGU
    df_encoded = pd.get_dummies(
        df_encoded, 
        columns=["Harvest_Season"], 
        drop_first=True, 
        dtype=int
    )

    # Đổi tên cột cho trực quan và thống nhất: Harvest_Season_In_Harvest
    print("- Danh sách các cột sau khi One-Hot Encoding:")
    print(df_encoded.columns.tolist())
    print("\n- Phân phối tần số của biến giả Harvest_Season_In_Harvest:")
    print(df_encoded["Harvest_Season_In_Harvest"].value_counts().rename(
        index={1: "1 (Trong mùa thu hoạch)", 0: "0 (Ngoài mùa thu hoạch)"}
    ))

    return df_encoded

def phan_tich_tuong_quan_pearson(df, cac_bien_doc_lap, bien_phu_thuoc):
    """
    Task 3.3: Tính toán hệ số tương quan Pearson (r) và kiểm định giả thuyết thống kê (p-value)
    giữa từng biến độc lập X_i với biến phụ thuộc Y (VN_Coffee_Price).
    Giả thuyết H0: r = 0 (Không tương quan tuyến tính).
    Giả thuyết H1: r != 0 (Có tương quan tuyến tính).
    """
    print("\n" + "=" * 85)
    print("TASK 3.3: PHÂN TÍCH TƯƠNG QUAN TUYẾN TÍNH PEARSON & KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ")
    print("=" * 85)

    y = df[bien_phu_thuoc]
    bang_tuong_quan = []

    for col in cac_bien_doc_lap:
        # Tính hệ số tương quan r và p-value bằng hàm pearsonr của scipy
        r, p_value = pearsonr(df[col], y)
        
        # Đánh giá ý nghĩa thống kê theo mức ý nghĩa alpha = 0.05
        if p_value < 0.05:
            ket_luan = "Bác bỏ H0 (Có ý nghĩa TK p < 0.05)"
            y_nghia = "Tương quan có ý nghĩa"
        else:
            ket_luan = "Chấp nhận H0 (Không có ý nghĩa p >= 0.05)"
            y_nghia = "Không có tương quan tuyến tính"

        # Phân loại mức độ tương quan theo thang đo chuẩn DA-SGU
        abs_r = abs(r)
        if abs_r >= 0.8:
            muc_do = "Rất mạnh"
        elif abs_r >= 0.6:
            muc_do = "Mạnh"
        elif abs_r >= 0.4:
            muc_do = "Vừa phải"
        elif abs_r >= 0.2:
            muc_do = "Yếu"
        else:
            muc_do = "Rất yếu / Không đáng kể"

        chieu_huong = "Thuận (+)" if r > 0 else "Nghịch (-)"

        bang_tuong_quan.append({
            "Biến độc lập": col,
            "Hệ số r": round(r, 4),
            "p-value": f"{p_value:.4e}",
            "Chiều hướng": chieu_huong,
            "Mức độ tương quan": muc_do,
            "Kết luận kiểm định (alpha=0.05)": ket_luan
        })

    df_kq_tuong_quan = pd.DataFrame(bang_tuong_quan)
    print(df_kq_tuong_quan.to_string(index=False))

    return df_kq_tuong_quan

def ve_bieu_do_tuong_quan(df, cac_bien_nghien_cuu, duong_dan_heatmap, duong_dan_scatter):
    """
    Task 3.3 (Trực quan hóa):
    - figures/fig03_correlation_heatmap.png: Ma trận nhiệt tương quan Pearson giữa tất cả các cặp biến
    - figures/fig04_scatter_matrix_linearity.png: Lưới biểu đồ phân tán kiểm tra tính tuyến tính với Y
    """
    print("\n" + "=" * 85)
    print("XUẤT BIỂU ĐỒ TRỰC QUAN HÓA TƯƠNG QUAN (300 DPI) VÀO figures/")
    print("=" * 85)

    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 1: fig03_correlation_heatmap.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Tính ma trận tương quan giữa các biến nghiên cứu
    corr_matrix = df[cac_bien_nghien_cuu].corr(method="pearson")
    
    # Vẽ biểu đồ Heatmap với bảng màu coolwarm, hiển thị giá trị số tương quan
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        center=0,
        linewidths=1, 
        linecolor="white", 
        cbar_kws={"label": "Hệ số tương quan Pearson (r)"},
        ax=ax
    )
    ax.set_title("MA TRẬN HỆ SỐ TƯƠNG QUAN PEARSON (PEARSON CORRELATION HEATMAP)", 
                 fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(duong_dan_heatmap, dpi=300)
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu Heatmap: {duong_dan_heatmap}")

    # --------------------------------------------------------------------------
    # BIỂU ĐỒ 2: fig04_scatter_matrix_linearity.png
    # --------------------------------------------------------------------------
    # Vẽ lưới biểu đồ phân tán (Scatter plots) giữa từng biến độc lập và Y
    cac_bien_x = [c for c in cac_bien_nghien_cuu if c != "VN_Coffee_Price"]
    so_bien = len(cac_bien_x)
    
    fig, axes = plt.subplots(1, so_bien, figsize=(4.5 * so_bien, 4.5), dpi=300)
    
    for i, col in enumerate(cac_bien_x):
        ax = axes[i]
        # Vẽ điểm dữ liệu phân tán
        sns.scatterplot(
            x=df[col], 
            y=df["VN_Coffee_Price"] / 1000.0, 
            ax=ax, 
            color="#1E88E5", 
            alpha=0.65, 
            edgecolor="black", 
            s=40
        )
        # Vẽ đường xu hướng hồi quy tuyến tính (Trendline)
        sns.regplot(
            x=df[col], 
            y=df["VN_Coffee_Price"] / 1000.0, 
            ax=ax, 
            scatter=False, 
            color="#D32F2F", 
            line_kws={"linewidth": 2}
        )
        ax.set_title(f"{col}\nvs Giá Tây Nguyên", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Giá Cà phê Tây Nguyên (Nghìn VNĐ/kg)", fontsize=10, fontweight="bold")
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("KIỂM TRA TÍNH TUYẾN TÍNH (LINEARITY) GIỮA CÁC BIẾN ĐỘC LẬP VÀ BIẾN PHỤ THUỘC Y", 
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(duong_dan_scatter, dpi=300)
    plt.close(fig)
    print(f"[THÀNH CÔNG] Đã lưu Scatter Matrix: {duong_dan_scatter}")

def thuc_hien_feature_scaling(df, cac_bien_dinh_luong_x, bien_y, cot_date, file_unscaled, file_scaled):
    """
    Task 3.4: Tạo 2 phiên bản dữ liệu:
    1. vn_coffee_unscaled.csv: Giữ nguyên đơn vị đo lường ban đầu
    2. vn_coffee_scaled.csv: Chuẩn hóa StandardScaler() đưa các biến độc lập định lượng về N(0, 1)
    """
    print("\n" + "=" * 85)
    print("TASK 3.4: CHUẨN HÓA THANG ĐO (FEATURE SCALING VỚI StandardScaler)")
    print("=" * 85)

    # 1. Phiên bản Unscaled: Lưu đầy đủ các biến với đơn vị đo tự nhiên
    df_unscaled = df.copy()
    df_unscaled.to_csv(file_unscaled, index=False, encoding="utf-8")
    print(f"[LƯU THÀNH CÔNG] Dữ liệu gốc chưa chuẩn hóa: {file_unscaled}")

    # 2. Phiên bản Scaled: Áp dụng StandardScaler cho các biến độc lập định lượng
    df_scaled = df.copy()
    scaler = StandardScaler()
    
    # Thực hiện fit và transform trên các cột định lượng độc lập
    scaled_matrix = scaler.fit_transform(df_scaled[cac_bien_dinh_luong_x])
    df_scaled[cac_bien_dinh_luong_x] = scaled_matrix

    df_scaled.to_csv(file_scaled, index=False, encoding="utf-8")
    print(f"[LƯU THÀNH CÔNG] Dữ liệu chuẩn hóa StandardScaler: {file_scaled}")

    # In kiểm tra Mean và Std của các biến sau khi chuẩn hóa
    print("\n- Kiểm tra thông số thống kê sau khi áp dụng StandardScaler:")
    for col in cac_bien_dinh_luong_x:
        m = df_scaled[col].mean()
        s = df_scaled[col].std(ddof=0)
        print(f"  + Cột '{col:<25}': Mean = {m:+.8f} (~0) | Std = {s:.8f} (~1)")

    return df_unscaled, df_scaled, scaler

def main():
    """
    Hàm điều phối toàn bộ chu trình Bước 3
    """
    # Khai báo đường dẫn tập tin
    file_in = "data/processed/vn_coffee_cleaned.csv"
    file_unscaled = "data/processed/vn_coffee_unscaled.csv"
    file_scaled = "data/processed/vn_coffee_scaled.csv"
    fig03 = "figures/fig03_correlation_heatmap.png"
    fig04 = "figures/fig04_scatter_matrix_linearity.png"

    assert os.path.exists(file_in), f"Lỗi: Không tìm thấy file dữ liệu sạch '{file_in}'!"
    df_clean = pd.read_csv(file_in)

    # Task 3.1: Rời rạc hóa pd.cut & LabelEncoder
    df_trans, le = thuc_hien_roi_rac_hoa_va_label_encoding(df_clean)

    # Task 3.2: One-Hot Encoding biến Harvest_Season
    df_encoded = thuc_hien_one_hot_encoding(df_clean)

    # Danh sách các biến phục vụ phân tích tương quan và mô hình hóa
    bien_phu_thuoc = "VN_Coffee_Price"
    cac_bien_doc_lap = [
        "VN_Coffee_Price_Lag1",
        "London_Robusta_Price",
        "VN_Production",
        "Crude_Oil_Price",
        "Harvest_Season_In_Harvest"
    ]
    cac_bien_toan_bo = [bien_phu_thuoc] + cac_bien_doc_lap

    # Task 3.3: Phân tích tương quan Pearson
    df_kq_pearson = phan_tich_tuong_quan_pearson(df_encoded, cac_bien_doc_lap, bien_phu_thuoc)

    # Vẽ biểu đồ tương quan Heatmap và Scatter Linearity
    ve_bieu_do_tuong_quan(df_encoded, cac_bien_toan_bo, fig03, fig04)

    # Task 3.4: Feature Scaling (StandardScaler)
    cac_bien_dinh_luong_x = [
        "VN_Coffee_Price_Lag1",
        "London_Robusta_Price",
        "VN_Production",
        "Crude_Oil_Price"
    ]
    df_unscaled, df_scaled, scaler = thuc_hien_feature_scaling(
        df_encoded, cac_bien_dinh_luong_x, bien_phu_thuoc, "Date", file_unscaled, file_scaled
    )

    # --------------------------------------------------------------------------
    # CHẠY SELF-TEST (ASSERTION RULES THEO QUY TẮC SỐ 5)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("CHẠY SELF-TEST (TỰ ĐỘNG KIỂM THỬ BƯỚC 3)")
    print("=" * 85)

    # Assertion 1: Kiểm tra tính chuẩn hóa Mean xấp xỉ 0 (sai số < 1e-6) và Std xấp xỉ 1
    for col in cac_bien_dinh_luong_x:
        mean_val = df_scaled[col].mean()
        std_val = df_scaled[col].std(ddof=0)
        assert abs(mean_val) < 1e-6, f"Self-Test Thất bại: Mean của cột {col} ({mean_val}) không xấp xỉ 0!"
        assert abs(std_val - 1.0) < 1e-6, f"Self-Test Thất bại: Std của cột {col} ({std_val}) không xấp xỉ 1!"
    print("[PASS] Assertion 1: Toàn bộ các biến chuẩn hóa đều có Mean ≈ 0 và Std ≈ 1 (sai số < 10^-6).")

    # Assertion 2: Kiểm tra sự tồn tại và tính hợp lệ của 2 file dữ liệu xuất ra
    assert os.path.exists(file_unscaled) and os.path.getsize(file_unscaled) > 0, "Lỗi: File vn_coffee_unscaled.csv rỗng!"
    assert os.path.exists(file_scaled) and os.path.getsize(file_scaled) > 0, "Lỗi: File vn_coffee_scaled.csv rỗng!"
    print(f"[PASS] Assertion 2: Cả 2 file dữ liệu (unscaled: {os.path.getsize(file_unscaled):,} bytes, scaled: {os.path.getsize(file_scaled):,} bytes) đều hợp lệ.")

    # Assertion 3: Kiểm tra sự tồn tại của 2 biểu đồ
    assert os.path.exists(fig03) and os.path.getsize(fig03) > 0, f"Lỗi: File {fig03} không hợp lệ!"
    assert os.path.exists(fig04) and os.path.getsize(fig04) > 0, f"Lỗi: File {fig04} không hợp lệ!"
    print(f"[PASS] Assertion 3: Cả 2 biểu đồ phân giải cao (Heatmap: {os.path.getsize(fig03):,} bytes, Scatter: {os.path.getsize(fig04):,} bytes) đã tạo thành công.")

    print("-" * 85)
    print("SELF-TEST BƯỚC 3: PASS 100% TOÀN BỘ CÁC BÀI KIỂM THỬ TỰ ĐỘNG!")
    print("=" * 85)

if __name__ == "__main__":
    main()
