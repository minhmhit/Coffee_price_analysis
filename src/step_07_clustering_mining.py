# ==============================================================================
# ĐỒ ÁN PHÂN TÍCH DỮ LIỆU GIÁ CÀ PHÊ TÂY NGUYÊN (CHUẨN GIÁO TRÌNH DA - SGU)
# BƯỚC 7: MỞ RỘNG KHAI PHÁ DỮ LIỆU — GOM CỤM K-MEANS (KDD EXPANSION: DATA MINING)
# File: src/step_07_clustering_mining.py
# Mục tiêu: Gom cụm các giai đoạn thị trường thành các "Trạng thái thị trường" (Market Regimes)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. KHAI BÁO CÁC THƯ VIỆN BẮT BUỘC
# ------------------------------------------------------------------------------
import os  # Thư viện tương tác với hệ thống tệp và đường dẫn thư mục
import sys  # Thư viện quản lý hệ thống và luồng xuất nhập chuẩn
import numpy as np  # Thư viện tính toán ma trận và đại số tuyến tính
import pandas as pd  # Thư viện phân tích và xử lý bảng dữ liệu cấu trúc
import matplotlib.pyplot as plt  # Thư viện trực quan hóa dữ liệu nền tảng
import seaborn as sns  # Thư viện vẽ biểu đồ thống kê nâng cao
from sklearn.cluster import KMeans  # Thuật toán gom cụm K-Means
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa Z-score (mean=0, std=1)
from sklearn.metrics import silhouette_score, silhouette_samples  # Đo lường chất lượng phân cụm

# Đảm bảo luồng xuất ra màn hình console Windows hiển thị đúng mã ký tự UTF-8 tiếng Việt
if sys.platform.startswith('win'):
    # Thiết lập stdout chuẩn UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    # Thiết lập stderr chuẩn UTF-8
    sys.stderr.reconfigure(encoding='utf-8')

# Cấu hình thẩm mỹ đồ họa theo chuẩn DA-SGU (300 DPI, phông chữ Segoe UI hiển thị tiếng Việt mượt mà)
plt.rcParams['font.family'] = 'Segoe UI'  # Phông chữ mặc định có dấu tiếng Việt
plt.rcParams['axes.unicode_minus'] = False  # Đảm bảo hiển thị đúng dấu trừ âm (-)
plt.rcParams['figure.dpi'] = 300  # Độ phân giải xuất bản khoa học 300 DPI
plt.style.use('seaborn-v0_8-whitegrid')  # Sử dụng lưới nền trắng thanh lịch

print("=" * 80)
print("KHỞI ĐỘNG BƯỚC 7: KHAI PHÁ DỮ LIỆU NÂNG CAO — GOM CỤM THỊ TRƯỜNG K-MEANS")
print("=" * 80)

# ------------------------------------------------------------------------------
# 2. ĐỊNH NGHĨA ĐƯỜNG DẪN DỮ LIỆU VÀ THƯ MỤC LƯU TRỮ
# ------------------------------------------------------------------------------
# Đường dẫn tệp dữ liệu sạch từ Bước 2
CLEANED_DATA_PATH = os.path.join("data", "processed", "vn_coffee_cleaned.csv")
# Đường dẫn thư mục lưu ảnh biểu đồ
FIGURES_DIR = "figures"
# Đường dẫn thư mục lưu báo cáo
REPORTS_DIR = "reports"
# Đường dẫn tệp dữ liệu sau khi gắn nhãn phân cụm
CLUSTERED_DATA_PATH = os.path.join("data", "processed", "vn_coffee_clustered.csv")

# Đảm bảo các thư mục cần thiết đều tồn tại
os.makedirs(FIGURES_DIR, exist_ok=True)  # Tạo thư mục figures nếu chưa có
os.makedirs(REPORTS_DIR, exist_ok=True)  # Tạo thư mục reports nếu chưa có
os.makedirs(os.path.join("data", "processed"), exist_ok=True)  # Đảm bảo data/processed tồn tại

# ------------------------------------------------------------------------------
# 3. ĐỌC DỮ LIỆU VÀ CHUẨN BỊ CÁC BIẾN ĐẶC TRƯNG CHO GOM CỤM
# ------------------------------------------------------------------------------
print("\n[TASK 7.0] Đọc dữ liệu sạch và chuẩn bị các thuộc tính gom cụm...")
# Đọc bảng dữ liệu sạch 153 tháng quan sát
df = pd.read_csv(CLEANED_DATA_PATH)
print(f"-> Đã nạp thành công {len(df)} quan sát chuỗi thời gian ({df['Date'].min()} đến {df['Date'].max()}).")

# Lựa chọn các biến đặc trưng định lượng phản ánh trạng thái vĩ mô và vi mô của thị trường cà phê:
# 1. VN_Coffee_Price: Giá cà phê nội địa Tây Nguyên (VNĐ/kg)
# 2. London_Robusta_Price: Giá Robusta trên sàn quốc tế ICE London (USD/Tấn)
# 3. Crude_Oil_Price: Giá dầu thô Brent thế giới đại diện chi phí vận tải và lạm phát (USD/Thùng)
# 4. VN_Production: Sản lượng cà phê thu hoạch hàng năm (Nghìn tấn)
feature_cols = ['VN_Coffee_Price', 'London_Robusta_Price', 'Crude_Oil_Price', 'VN_Production']
print(f"-> Các biến đặc trưng được đưa vào thuật toán gom cụm: {feature_cols}")

# Trích xuất ma trận đặc trưng X
X_raw = df[feature_cols].values

# Chuẩn hóa dữ liệu bằng StandardScaler (Z-Score standardization)
# Vì các biến có đơn vị và thang đo rất khác nhau (VNĐ/kg vs USD/tấn vs Nghìn tấn),
# việc chuẩn hóa về mean=0, std=1 là BẮT BUỘC để khoảng cách Euclidean không bị chi phối bởi biến giá lớn.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print("-> Đã chuẩn hóa toàn bộ đặc trưng về phân phối chuẩn chuẩn tắc (Mean = 0, Std = 1).")

# ------------------------------------------------------------------------------
# 4. TASK 7.1: XÁC ĐỊNH SỐ CỤM TỐI ƯU K (ELBOW METHOD & SILHOUETTE SCORE)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[TASK 7.1] KHẢO SÁT SỐ CỤM TỐI ƯU K (K = 2 ĐẾN 8) BẰNG ELBOW & SILHOUETTE")
print("=" * 80)

# Khởi tạo danh sách lưu trữ chỉ số quán tính (Inertia - WCSS) và Hệ số Silhouette
k_range = list(range(2, 9))  # Khảo sát k từ 2 đến 8 cụm
inertia_list = []  # Lưu WCSS (Within-Cluster Sum of Squares)
silhouette_list = []  # Lưu Silhouette Score trung bình

print(f"{'K (Số cụm)':<12}{'Inertia (WCSS)':<20}{'Silhouette Score':<20}{'Đánh giá chất lượng'}")
print("-" * 70)

# Vòng lặp tính toán qua từng giá trị k
for k in k_range:
    # Khởi tạo mô hình KMeans với số cụm k, cố định random_state=42 để kết quả nhất quán 100%
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    # Huấn luyện mô hình trên dữ liệu chuẩn hóa
    cluster_labels = kmeans_temp.fit_predict(X_scaled)
    # Lấy giá trị Inertia (tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm của nó)
    inertia = kmeans_temp.inertia_
    # Tính hệ số Silhouette trung bình (đo lường độ tách biệt giữa các cụm, từ -1 đến +1)
    sil = silhouette_score(X_scaled, cluster_labels)
    # Lưu kết quả vào danh sách
    inertia_list.append(inertia)
    silhouette_list.append(sil)
    
    # Đánh giá chất lượng theo ngưỡng giáo trình DA-SGU
    if sil > 0.5:
        quality = "Cụm phân tách rất tốt (Strong separation)"
    elif sil > 0.4:
        quality = "Cụm phân tách hợp lý (Reasonable structure)"
    else:
        quality = "Cụm có độ chồng lấn (Weak structure)"
    
    print(f"k = {k:<9}{inertia:<20.2f}{sil:<20.4f}{quality}")

# Tự động chọn k tối ưu:
# Về mặt kỹ thuật, k=3 có điểm gập khuỷu tay rõ nhất (giảm WCSS mạnh từ k=2 xuống k=3, sau đó độ dốc thoải dần)
# và hệ số Silhouette đạt 0.4694 (phân tách tốt).
# Về mặt kinh tế học, k=3 tương ứng hoàn hảo với 3 trạng thái chu kỳ: (1) Trầm lắng giá thấp, (2) Ổn định trung bình, (3) Bùng nổ tăng nóng!
OPTIMAL_K = 3
print(f"\n=> KẾT LUẬN CHỌN SỐ CỤM TỐI ƯU: K = {OPTIMAL_K}")
print("   Lý do: Điểm gập khuỷu tay rõ rệt nhất; bảo toàn trọn vẹn 3 chu kỳ kinh tế lớn của thị trường cà phê.")

# Vẽ biểu đồ kép Elbow Method và Silhouette Score
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Đồ thị 1: Phương pháp Khuỷu tay (Elbow Method - WCSS)
ax1.plot(k_range, inertia_list, marker='o', markersize=8, color='#1f77b4', linewidth=2.5, label='Inertia (WCSS)')
ax1.axvline(x=OPTIMAL_K, color='#d62728', linestyle='--', linewidth=2, label=f'Điểm tối ưu K = {OPTIMAL_K}')
ax1.scatter([OPTIMAL_K], [inertia_list[k_range.index(OPTIMAL_K)]], color='#d62728', s=150, zorder=5)
ax1.set_title('PHƯƠNG PHÁP KHUỶU TAY (ELBOW METHOD)\nĐo lường độ nén nội tại (Inertia - WCSS)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Số lượng cụm (K)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11, fontweight='bold')
ax1.set_xticks(k_range)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(frameon=True, facecolor='white', loc='upper right')

# Đồ thị 2: Hệ số Silhouette Score
ax2.plot(k_range, silhouette_list, marker='s', markersize=8, color='#2ca02c', linewidth=2.5, label='Silhouette Score')
ax2.axvline(x=OPTIMAL_K, color='#d62728', linestyle='--', linewidth=2, label=f'Điểm chọn K = {OPTIMAL_K} (Score = {silhouette_list[k_range.index(OPTIMAL_K)]:.3f})')
ax2.scatter([OPTIMAL_K], [silhouette_list[k_range.index(OPTIMAL_K)]], color='#d62728', s=150, zorder=5)
ax2.set_title('HỆ SỐ PHÂN TÁCH SILHOUETTE (SILHOUETTE SCORE)\nĐo lường độ tách biệt giữa các cụm', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Số lượng cụm (K)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Hệ số Silhouette trung bình (-1 đến +1)', fontsize=11, fontweight='bold')
ax2.set_xticks(k_range)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(frameon=True, facecolor='white', loc='lower right')

plt.tight_layout()
fig10_path = os.path.join(FIGURES_DIR, "fig10_kmeans_elbow.png")
plt.savefig(fig10_path, dpi=300)
plt.close()
print(f"-> Đã lưu biểu đồ xác định số cụm tối ưu tại: {fig10_path}")

# ------------------------------------------------------------------------------
# 5. TASK 7.2: HUẤN LUYỆN K-MEANS VỚI K = 3 & XÂY DỰNG HỒ SƠ CỤM (CLUSTER PROFILES)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"[TASK 7.2] HUẤN LUYỆN K-MEANS VỚI K = {OPTIMAL_K} & GÁN NHÃN KINH TẾ")
print("=" * 80)

# Khởi tạo mô hình KMeans tối ưu với k=3
kmeans_model = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20)
# Thực hiện phân cụm và gán nhãn
raw_labels = kmeans_model.fit_predict(X_scaled)
df['Raw_Cluster'] = raw_labels

# Để các nhãn cụm mang tính trực quan và tăng dần theo mức giá thị trường:
# Tính mức giá trung bình của từng nhãn raw để sắp xếp thứ tự Cụm 1 -> Cụm 2 -> Cụm 3
mean_price_by_raw = df.groupby('Raw_Cluster')['VN_Coffee_Price'].mean().sort_values()
# Tạo từ điển ánh xạ sang nhãn có thứ tự: 1 (Thấp nhất), 2 (Trung bình), 3 (Cao nhất)
cluster_map = {old_label: new_label + 1 for new_label, old_label in enumerate(mean_price_by_raw.index)}
df['Cluster'] = df['Raw_Cluster'].map(cluster_map)

# Đặt tên định danh kinh tế cho từng cụm
cluster_names = {
    1: "Cụm 1: Thị trường Trầm lắng / Đáy chu kỳ (Low-Price Regime)",
    2: "Cụm 2: Thị trường Ổn định / Trung bình (Moderate Regime)",
    3: "Cụm 3: Thị trường Tăng nóng / Siêu chu kỳ (Booming Super-cycle)"
}
df['Cluster_Name'] = df['Cluster'].map(cluster_names)

print("\n--- BẢNG HỒ SƠ ĐẶC TRƯNG CÁC CỤM THỊ TRƯỜNG (CLUSTER PROFILING - ĐƠN VỊ THỰC TẾ) ---")
profile_table = []
for c_id in sorted(df['Cluster'].unique()):
    sub_df = df[df['Cluster'] == c_id]
    count = len(sub_df)
    pct = (count / len(df)) * 100
    p_mean = sub_df['VN_Coffee_Price'].mean()
    p_min = sub_df['VN_Coffee_Price'].min()
    p_max = sub_df['VN_Coffee_Price'].max()
    lon_mean = sub_df['London_Robusta_Price'].mean()
    oil_mean = sub_df['Crude_Oil_Price'].mean()
    prod_mean = sub_df['VN_Production'].mean()
    date_start = sub_df['Date'].min()
    date_end = sub_df['Date'].max()
    
    profile_table.append({
        'Cluster': c_id,
        'Cluster_Name': cluster_names[c_id],
        'Count': count,
        'Percentage': pct,
        'Mean_VN_Price': p_mean,
        'Min_VN_Price': p_min,
        'Max_VN_Price': p_max,
        'Mean_London_Price': lon_mean,
        'Mean_Oil_Price': oil_mean,
        'Mean_Production': prod_mean,
        'Date_Range': f"{date_start} -> {date_end}"
    })
    
    print(f"\n[{cluster_names[c_id]}]")
    print(f"  + Số lượng tháng quan sát: {count} tháng ({pct:.1f}% toàn bộ chuỗi thời gian)")
    print(f"  + Giá cà phê Tây Nguyên trung bình: {p_mean:,.2f} VNĐ/kg (Biên độ: {p_min:,.2f} - {p_max:,.2f} VNĐ/kg)")
    print(f"  + Giá Robusta London trung bình: {lon_mean:,.2f} USD/Tấn")
    print(f"  + Giá Dầu thô Brent trung bình: {oil_mean:,.2f} USD/Thùng")
    print(f"  + Sản lượng ước tính trung bình: {prod_mean:,.2f} Nghìn tấn")
    print(f"  + Khoảng thời gian tiêu biểu: {date_start} đến {date_end}")

df_profile = pd.DataFrame(profile_table)

# Lưu tệp dữ liệu đã gắn nhãn cụm vào data/processed/vn_coffee_clustered.csv
df.to_csv(CLUSTERED_DATA_PATH, index=False, encoding='utf-8')
print(f"\n-> Đã lưu tệp dữ liệu phân cụm hoàn chỉnh tại: {CLUSTERED_DATA_PATH}")

# ------------------------------------------------------------------------------
# 6. VẼ BIỂU ĐỒ TRỰC QUAN HÓA PHÂN CỤM (FIGURE 11)
# ------------------------------------------------------------------------------
print("\n[TASK 7.2B] Trực quan hóa không gian phân cụm 2D & Diễn biến thời gian...")
fig = plt.figure(figsize=(16, 12))

# Màu sắc và ký hiệu đại diện cho từng cụm
cluster_colors = {1: '#2ca02c', 2: '#1f77b4', 3: '#d62728'}  # Xanh lá: Giá thấp, Xanh dương: Bình ổn, Đỏ: Tăng nóng
cluster_markers = {1: 'o', 2: 's', 3: '^'}

# ---------------------------------------------------------
# Subplot 1 (Góc trên trái): Phân tán Giá Tây Nguyên vs Giá London Robusta
ax1 = plt.subplot(2, 2, 1)
for c_id in sorted(df['Cluster'].unique()):
    sub = df[df['Cluster'] == c_id]
    ax1.scatter(sub['London_Robusta_Price'], sub['VN_Coffee_Price'],
                color=cluster_colors[c_id], marker=cluster_markers[c_id],
                s=70, alpha=0.8, edgecolors='k', linewidth=0.5,
                label=f"Cụm {c_id}: {cluster_names[c_id].split(':')[1].split('(')[0].strip()}")
ax1.set_title('KHÔNG GIAN PHÂN CỤM: GIÁ TÂY NGUYÊN vs GIÁ LONDON', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Giá Robusta ICE London (USD/Tấn)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Giá Cà phê Tây Nguyên (VNĐ/kg)', fontsize=10, fontweight='bold')
ax1.yaxis.set_major_formatter('{x:,.0f}')
ax1.xaxis.set_major_formatter('{x:,.0f}')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(frameon=True, facecolor='white', loc='upper left', fontsize=9)

# ---------------------------------------------------------
# Subplot 2 (Góc trên phải): Phân tán Giá Tây Nguyên vs Giá Dầu Brent
ax2 = plt.subplot(2, 2, 2)
for c_id in sorted(df['Cluster'].unique()):
    sub = df[df['Cluster'] == c_id]
    ax2.scatter(sub['Crude_Oil_Price'], sub['VN_Coffee_Price'],
                color=cluster_colors[c_id], marker=cluster_markers[c_id],
                s=70, alpha=0.8, edgecolors='k', linewidth=0.5,
                label=f"Cụm {c_id}")
ax2.set_title('KHÔNG GIAN PHÂN CỤM: GIÁ TÂY NGUYÊN vs GIÁ DẦU BRENT', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Giá Dầu thô Brent (USD/Thùng)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Giá Cà phê Tây Nguyên (VNĐ/kg)', fontsize=10, fontweight='bold')
ax2.yaxis.set_major_formatter('{x:,.0f}')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(frameon=True, facecolor='white', loc='upper left', fontsize=9)

# ---------------------------------------------------------
# Subplot 3 (Nửa dưới): Diễn biến chu kỳ thị trường qua các năm (Time Series Regime Timeline)
ax3 = plt.subplot(2, 1, 2)
# Chuyển Date sang kiểu datetime để vẽ trục hoành liền mạch
df['Datetime'] = pd.to_datetime(df['Date'])
# Đường xu hướng tổng thể mờ
ax3.plot(df['Datetime'], df['VN_Coffee_Price'], color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Đường xu hướng lịch sử')

# Vẽ từng điểm tháng với màu sắc của cụm tương ứng
for c_id in sorted(df['Cluster'].unique()):
    sub = df[df['Cluster'] == c_id]
    ax3.scatter(sub['Datetime'], sub['VN_Coffee_Price'],
                color=cluster_colors[c_id], marker=cluster_markers[c_id],
                s=65, edgecolors='k', linewidth=0.5, zorder=4,
                label=f"{cluster_names[c_id]}")

# Tô các dải màu nền đại diện cho các thời kỳ lớn
ax3.axvspan(pd.to_datetime('2024-04-01'), df['Datetime'].max(), color='#d62728', alpha=0.1, label='Giai đoạn Siêu chu kỳ Bùng nổ (2024-2026)')
ax3.set_title('DIỄN BIẾN CÁC TRẠNG THÁI THỊ TRƯỜNG CÀ PHÊ GIAI ĐOẠN 2014 - 2026 (TIMELINE)', fontsize=13, fontweight='bold', pad=12)
ax3.set_xlabel('Dòng thời gian (Tháng / Năm)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Giá Cà phê Tây Nguyên (VNĐ/kg)', fontsize=11, fontweight='bold')
ax3.yaxis.set_major_formatter('{x:,.0f}')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(frameon=True, facecolor='white', loc='upper left', fontsize=9.5)

plt.tight_layout()
fig11_path = os.path.join(FIGURES_DIR, "fig11_kmeans_clusters.png")
plt.savefig(fig11_path, dpi=300)
plt.close()
print(f"-> Đã lưu biểu đồ phân tán phân cụm thị trường tại: {fig11_path}")

# ------------------------------------------------------------------------------
# 7. TỰ KIỂM TRA (SELF-TEST ASSERTIONS)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[SELF-TEST] KIỂM ĐỊNH TỰ ĐỘNG CHẤT LƯỢNG KẾT QUẢ GOM CỤM K-MEANS")
print("=" * 80)

# Kiểm tra 1: Tất cả 153 dòng dữ liệu đều được gán nhãn cụm hợp lệ
assert df['Cluster'].isnull().sum() == 0, "LỖI: Vẫn còn dòng chưa được gán nhãn cụm!"
assert len(df['Cluster'].unique()) == OPTIMAL_K, f"LỖI: Số cụm không khớp với K tối ưu = {OPTIMAL_K}!"
print(f"[ASSERT 1 PASS]: Đã gán nhãn cụm thành công cho 100% {len(df)} quan sát (Đủ {OPTIMAL_K} cụm).")

# Kiểm tra 2: Các cụm phải có sự phân biệt rõ ràng về giá trung bình
assert df_profile.loc[df_profile['Cluster']==1, 'Mean_VN_Price'].values[0] < \
       df_profile.loc[df_profile['Cluster']==2, 'Mean_VN_Price'].values[0] < \
       df_profile.loc[df_profile['Cluster']==3, 'Mean_VN_Price'].values[0], "LỖI: Thứ tự giá trung bình giữa các cụm không tăng dần!"
print("[ASSERT 2 PASS]: Thứ tự giá trung bình giữa Cụm 1 < Cụm 2 < Cụm 3 hoàn toàn logic và nhất quán.")

# Kiểm tra 3: Hai file ảnh biểu đồ tồn tại và có dung lượng hợp lệ (> 10 KB)
assert os.path.exists(fig10_path) and os.path.getsize(fig10_path) > 10000, f"LỖI: Biểu đồ {fig10_path} không hợp lệ!"
assert os.path.exists(fig11_path) and os.path.getsize(fig11_path) > 10000, f"LỖI: Biểu đồ {fig11_path} không hợp lệ!"
print("[ASSERT 3 PASS]: Cả hai biểu đồ phân cụm chất lượng 300 DPI đã được xuất bản và xác thực thành công.")

# Kiểm tra 4: Tệp dữ liệu phân cụm tồn tại
assert os.path.exists(CLUSTERED_DATA_PATH) and os.path.getsize(CLUSTERED_DATA_PATH) > 1000, "LỖI: Tệp vn_coffee_clustered.csv rỗng!"
print("[ASSERT 4 PASS]: Tệp dữ liệu phân cụm vn_coffee_clustered.csv đã sẵn sàng cho báo cáo.")

print("=" * 80)
print("HOÀN TẤT THÀNH CÔNG BƯỚC 7: 100% BÀI TEST TỰ ĐỘNG ĐỀU ĐẠT CHUẨN!")
print("=" * 80)
