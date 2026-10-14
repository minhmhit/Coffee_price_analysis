import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-v0_8-whitegrid')

CLEANED_DATA_PATH = os.path.join("data", "processed", "vn_coffee_cleaned.csv")
FIGURES_DIR = "figures"
REPORTS_DIR = "reports"
CLUSTERED_DATA_PATH = os.path.join("data", "processed", "vn_coffee_clustered.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("BƯỚC 7: KHAI PHÁ DỮ LIỆU NÂNG CAO — GOM CỤM THỊ TRƯỜNG K-MEANS")
    print("=" * 80)

    df = pd.read_csv(CLEANED_DATA_PATH)
    feature_cols = ['VN_Coffee_Price', 'London_Robusta_Price', 'Crude_Oil_Price', 'VN_Production']
    X_raw = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. Khảo sát K (Elbow & Silhouette)
    k_range = list(range(2, 9))
    inertia_list = []
    silhouette_list = []

    print(f"{'K':<6}{'Inertia':<18}{'Silhouette Score'}")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        inertia_list.append(inertia)
        silhouette_list.append(sil)
        print(f"k={k:<4}{inertia:<18.2f}{sil:.4f}")

    OPTIMAL_K = 3

    # Vẽ biểu đồ Elbow & Silhouette (fig10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(k_range, inertia_list, marker='o', markersize=8, color='#1f77b4', linewidth=2.5, label='Inertia')
    ax1.axvline(x=OPTIMAL_K, color='#d62728', linestyle='--', linewidth=2, label=f'K = {OPTIMAL_K}')
    ax1.scatter([OPTIMAL_K], [inertia_list[k_range.index(OPTIMAL_K)]], color='#d62728', s=150, zorder=5)
    ax1.set_title('PHƯƠNG PHÁP KHUỶU TAY (ELBOW METHOD)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Số lượng cụm (K)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Inertia (WCSS)', fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(k_range, silhouette_list, marker='s', markersize=8, color='#2ca02c', linewidth=2.5, label='Silhouette')
    ax2.axvline(x=OPTIMAL_K, color='#d62728', linestyle='--', linewidth=2, label=f'K = {OPTIMAL_K}')
    ax2.scatter([OPTIMAL_K], [silhouette_list[k_range.index(OPTIMAL_K)]], color='#d62728', s=150, zorder=5)
    ax2.set_title('HỆ SỐ SILHOUETTE SCORE', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Số lượng cụm (K)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    fig10_path = os.path.join(FIGURES_DIR, "fig10_kmeans_elbow.png")
    plt.savefig(fig10_path, dpi=300)
    plt.close()

    # 2. Huấn luyện K-Means K=3 & Gán nhãn
    kmeans_model = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20)
    raw_labels = kmeans_model.fit_predict(X_scaled)
    df['Raw_Cluster'] = raw_labels

    mean_price_by_raw = df.groupby('Raw_Cluster')['VN_Coffee_Price'].mean().sort_values()
    cluster_map = {old_label: new_label + 1 for new_label, old_label in enumerate(mean_price_by_raw.index)}
    df['Cluster'] = df['Raw_Cluster'].map(cluster_map)

    cluster_names = {
        1: "Cụm 1: Thị trường Trầm lắng (Low-Price)",
        2: "Cụm 2: Thị trường Ổn định (Moderate)",
        3: "Cụm 3: Siêu chu kỳ Tăng nóng (Booming)"
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)

    profile_table = []
    print("\n--- HỒ SƠ ĐẶC TRƯNG CÁC CỤM THỊ TRƯỜNG ---")
    for c_id in sorted(df['Cluster'].unique()):
        sub_df = df[df['Cluster'] == c_id]
        p_mean = sub_df['VN_Coffee_Price'].mean()
        p_min = sub_df['VN_Coffee_Price'].min()
        p_max = sub_df['VN_Coffee_Price'].max()
        lon_mean = sub_df['London_Robusta_Price'].mean()
        oil_mean = sub_df['Crude_Oil_Price'].mean()
        prod_mean = sub_df['VN_Production'].mean()
        
        profile_table.append({
            'Cluster': c_id,
            'Cluster_Name': cluster_names[c_id],
            'Count': len(sub_df),
            'Mean_VN_Price': p_mean,
            'Min_VN_Price': p_min,
            'Max_VN_Price': p_max,
            'Mean_London_Price': lon_mean,
            'Mean_Oil_Price': oil_mean,
            'Mean_Production': prod_mean
        })
        print(f"[{cluster_names[c_id]}]: {len(sub_df)} tháng | Giá TB = {p_mean:,.0f} VNĐ/kg | London = {lon_mean:,.0f} USD/tấn")

    df_profile = pd.DataFrame(profile_table)
    df.to_csv(CLUSTERED_DATA_PATH, index=False, encoding='utf-8')

    # 3. Vẽ biểu đồ phân cụm (fig11)
    fig = plt.figure(figsize=(16, 12))
    cluster_colors = {1: '#2ca02c', 2: '#1f77b4', 3: '#d62728'}
    cluster_markers = {1: 'o', 2: 's', 3: '^'}

    # Subplot 1: Giá Tây Nguyên vs London
    ax1 = plt.subplot(2, 2, 1)
    for c_id in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == c_id]
        ax1.scatter(sub['London_Robusta_Price'], sub['VN_Coffee_Price'],
                    color=cluster_colors[c_id], marker=cluster_markers[c_id],
                    s=70, alpha=0.8, edgecolors='k', linewidth=0.5,
                    label=cluster_names[c_id])
    ax1.set_title('GIÁ TÂY NGUYÊN vs GIÁ LONDON', fontsize=11, fontweight='bold')
    ax1.set_xlabel('London Robusta (USD/Tấn)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Giá Tây Nguyên (VNĐ/kg)', fontsize=10, fontweight='bold')
    ax1.yaxis.set_major_formatter('{x:,.0f}')
    ax1.xaxis.set_major_formatter('{x:,.0f}')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9)

    # Subplot 2: Giá Tây Nguyên vs Dầu Brent
    ax2 = plt.subplot(2, 2, 2)
    for c_id in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == c_id]
        ax2.scatter(sub['Crude_Oil_Price'], sub['VN_Coffee_Price'],
                    color=cluster_colors[c_id], marker=cluster_markers[c_id],
                    s=70, alpha=0.8, edgecolors='k', linewidth=0.5,
                    label=cluster_names[c_id])
    ax2.set_title('GIÁ TÂY NGUYÊN vs GIÁ DẦU BRENT', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Dầu thô Brent (USD/Thùng)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Giá Tây Nguyên (VNĐ/kg)', fontsize=10, fontweight='bold')
    ax2.yaxis.set_major_formatter('{x:,.0f}')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=9)

    # Subplot 3: Timeline chu kỳ
    ax3 = plt.subplot(2, 1, 2)
    df['Datetime'] = pd.to_datetime(df['Date'])
    ax3.plot(df['Datetime'], df['VN_Coffee_Price'], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    for c_id in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == c_id]
        ax3.scatter(sub['Datetime'], sub['VN_Coffee_Price'],
                    color=cluster_colors[c_id], marker=cluster_markers[c_id],
                    s=65, edgecolors='k', linewidth=0.5, zorder=4,
                    label=cluster_names[c_id])
    ax3.axvspan(pd.to_datetime('2024-04-01'), df['Datetime'].max(), color='#d62728', alpha=0.1, label='Siêu chu kỳ Bùng nổ (2024-2026)')
    ax3.set_title('DIỄN BIẾN CHU KỲ THỊ TRƯỜNG CÀ PHÊ (2014 - 2026)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Thời gian (Năm)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Giá Tây Nguyên (VNĐ/kg)', fontsize=11, fontweight='bold')
    ax3.yaxis.set_major_formatter('{x:,.0f}')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper left', fontsize=9.5)

    plt.tight_layout()
    fig11_path = os.path.join(FIGURES_DIR, "fig11_kmeans_clusters.png")
    plt.savefig(fig11_path, dpi=300)
    plt.close()

    # Self-Test
    assert df['Cluster'].isnull().sum() == 0, "Lỗi: Vẫn còn dòng chưa gán cụm!"
    assert len(df['Cluster'].unique()) == OPTIMAL_K, "Lỗi: Số cụm != 3"
    assert df_profile.loc[df_profile['Cluster']==1, 'Mean_VN_Price'].values[0] < \
           df_profile.loc[df_profile['Cluster']==2, 'Mean_VN_Price'].values[0] < \
           df_profile.loc[df_profile['Cluster']==3, 'Mean_VN_Price'].values[0], "Lỗi thứ tự giá trung bình"
    assert os.path.exists(fig10_path) and os.path.getsize(fig10_path) > 10000
    assert os.path.exists(fig11_path) and os.path.getsize(fig11_path) > 10000
    assert os.path.exists(CLUSTERED_DATA_PATH) and os.path.getsize(CLUSTERED_DATA_PATH) > 1000

    print("\nSELF-TEST BƯỚC 7: PASS 100%!")

if __name__ == "__main__":
    main()
