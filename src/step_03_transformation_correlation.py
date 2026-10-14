import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

def thuc_hien_roi_rac_hoa_va_label_encoding(df):
    print("=" * 85)
    print("TASK 3.1: RỜI RẠC HÓA DỮ LIỆU (pd.cut) & MÃ HÓA NHÃN (LabelEncoder)")
    print("=" * 85)

    df_trans = df.copy()
    nhan_danh_muc = ["Thap", "Trung_Binh", "Cao"]
    df_trans["Crude_Oil_Category"] = pd.cut(
        df_trans["Crude_Oil_Price"], 
        bins=3, 
        labels=nhan_danh_muc
    )
    
    le = LabelEncoder()
    df_trans["Crude_Oil_Encoded"] = le.fit_transform(df_trans["Crude_Oil_Category"].astype(str))
    return df_trans, le

def thuc_hien_one_hot_encoding(df):
    print("\n" + "=" * 85)
    print("TASK 3.2: MÃ HÓA BIẾN ĐỊNH DANH BẰNG ONE-HOT ENCODING (pd.get_dummies)")
    print("=" * 85)

    df_encoded = df.copy()
    df_encoded["Harvest_Season"] = pd.Categorical(
        df_encoded["Harvest_Season"], 
        categories=["Off_Harvest", "In_Harvest"]
    )
    df_encoded = pd.get_dummies(
        df_encoded, 
        columns=["Harvest_Season"], 
        drop_first=True, 
        dtype=int
    )
    return df_encoded

def phan_tich_tuong_quan_pearson(df, cac_bien_doc_lap, bien_phu_thuoc):
    print("\n" + "=" * 85)
    print("TASK 3.3: PHÂN TÍCH TƯƠNG QUAN TUYẾN TÍNH PEARSON & KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ")
    print("=" * 85)

    y = df[bien_phu_thuoc]
    bang_tuong_quan = []

    for col in cac_bien_doc_lap:
        r, p_value = pearsonr(df[col], y)
        ket_luan = "Bác bỏ H0 (Có ý nghĩa TK p < 0.05)" if p_value < 0.05 else "Chấp nhận H0 (p >= 0.05)"
        
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
            muc_do = "Rất yếu"

        chieu_huong = "Thuận (+)" if r > 0 else "Nghịch (-)"
        bang_tuong_quan.append({
            "Biến độc lập": col,
            "Hệ số r": round(r, 4),
            "p-value": f"{p_value:.4e}",
            "Chiều hướng": chieu_huong,
            "Mức độ": muc_do,
            "Kết luận": ket_luan
        })

    df_kq_tuong_quan = pd.DataFrame(bang_tuong_quan)
    print(df_kq_tuong_quan.to_string(index=False))
    return df_kq_tuong_quan

def ve_bieu_do_tuong_quan(df, cac_bien_nghien_cuu, duong_dan_heatmap, duong_dan_scatter):
    print("\n" + "=" * 85)
    print("XUẤT BIỂU ĐỒ TRỰC QUAN HÓA TƯƠNG QUAN (300 DPI) VÀO figures/")
    print("=" * 85)

    # 1. Heatmap ma trận tương quan Pearson (fig03)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    corr_matrix = df[cac_bien_nghien_cuu].corr(method="pearson")
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

    # 2. Scatter plot kiểm tra tính tuyến tính (fig04)
    cac_bien_x = [c for c in cac_bien_nghien_cuu if c != "VN_Coffee_Price"]
    so_bien = len(cac_bien_x)
    fig, axes = plt.subplots(1, so_bien, figsize=(4.5 * so_bien, 4.5), dpi=300)
    
    for i, col in enumerate(cac_bien_x):
        ax = axes[i]
        sns.scatterplot(
            x=df[col], 
            y=df["VN_Coffee_Price"] / 1000.0, 
            ax=ax, 
            color="#1E88E5", 
            alpha=0.65, 
            edgecolor="black", 
            s=40
        )
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

def thuc_hien_feature_scaling(df, cac_bien_dinh_luong_x, bien_y, cot_date, file_unscaled, file_scaled):
    print("\n" + "=" * 85)
    print("TASK 3.4: CHUẨN HÓA THANG ĐO (FEATURE SCALING VỚI StandardScaler)")
    print("=" * 85)

    df_unscaled = df.copy()
    df_unscaled.to_csv(file_unscaled, index=False, encoding="utf-8")

    df_scaled = df.copy()
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_scaled[cac_bien_dinh_luong_x])
    df_scaled[cac_bien_dinh_luong_x] = scaled_matrix
    df_scaled.to_csv(file_scaled, index=False, encoding="utf-8")

    return df_unscaled, df_scaled, scaler

def main():
    file_in = "data/processed/vn_coffee_cleaned.csv"
    file_unscaled = "data/processed/vn_coffee_unscaled.csv"
    file_scaled = "data/processed/vn_coffee_scaled.csv"
    fig03 = "figures/fig03_correlation_heatmap.png"
    fig04 = "figures/fig04_scatter_matrix_linearity.png"

    assert os.path.exists(file_in), f"Lỗi: Không tìm thấy '{file_in}'!"
    df_clean = pd.read_csv(file_in)

    df_trans, le = thuc_hien_roi_rac_hoa_va_label_encoding(df_clean)
    df_encoded = thuc_hien_one_hot_encoding(df_clean)

    bien_phu_thuoc = "VN_Coffee_Price"
    cac_bien_doc_lap = [
        "VN_Coffee_Price_Lag1",
        "London_Robusta_Price",
        "VN_Production",
        "Crude_Oil_Price",
        "Harvest_Season_In_Harvest"
    ]
    cac_bien_toan_bo = [bien_phu_thuoc] + cac_bien_doc_lap

    df_kq_pearson = phan_tich_tuong_quan_pearson(df_encoded, cac_bien_doc_lap, bien_phu_thuoc)
    ve_bieu_do_tuong_quan(df_encoded, cac_bien_toan_bo, fig03, fig04)

    cac_bien_dinh_luong_x = [
        "VN_Coffee_Price_Lag1",
        "London_Robusta_Price",
        "VN_Production",
        "Crude_Oil_Price"
    ]
    df_unscaled, df_scaled, scaler = thuc_hien_feature_scaling(
        df_encoded, cac_bien_dinh_luong_x, bien_phu_thuoc, "Date", file_unscaled, file_scaled
    )

    # Self-Test
    for col in cac_bien_dinh_luong_x:
        assert abs(df_scaled[col].mean()) < 1e-6, f"Lỗi Mean {col} != 0"
        assert abs(df_scaled[col].std(ddof=0) - 1.0) < 1e-6, f"Lỗi Std {col} != 1"

    assert os.path.exists(file_unscaled) and os.path.getsize(file_unscaled) > 0
    assert os.path.exists(file_scaled) and os.path.getsize(file_scaled) > 0
    assert os.path.exists(fig03) and os.path.getsize(fig03) > 0
    assert os.path.exists(fig04) and os.path.getsize(fig04) > 0

    print(f"\n[THÀNH CÔNG] Đã lưu 2 phiên bản dữ liệu và 2 biểu đồ tương quan!")
    print("SELF-TEST BƯỚC 3: PASS 100%!")

if __name__ == "__main__":
    main()
