"""
====================================================================================================
DỰ ÁN PHÂN TÍCH VÀ DỰ BÁO GIÁ CÀ PHÊ TÂY NGUYÊN (VIETNAM COFFEE PRICE ANALYSIS)
HỌC PHẦN: KHAI PHÁ DỮ LIỆU & PHÂN TÍCH KINH DOANH (DATA ANALYSIS - SGU)
FILE: src/step_11_decision_tree_classification.py
MỤC TIÊU: BƯỚC 11 — MÔ HÌNH PHÂN LOẠI CÓ GIÁM SÁT VÀ TRÍCH XUẤT TẬP LUẬT CÂY QUYẾT ĐỊNH
          (SUPERVISED CLASSIFICATION & DECISION TREE RULE EXTRACTION)
          - Sử dụng nhãn cụm K-Means (Trầm lắng, Ổn định, Siêu chu kỳ) làm biến mục tiêu Y_class
          - Huấn luyện Cây quyết định CART (DecisionTreeClassifier)
          - Đánh giá Confusion Matrix, Accuracy, Precision, Recall, F1-Score
          - Trực quan hóa Cây quyết định (fig18_decision_tree_rules.png - 300 DPI)
          - Trích xuất tập luật kinh doanh (Business Rules) dạng NẾU - THÌ (IF - THEN)
====================================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Cấu hình UTF-8 tiếng Việt cho Windows Console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình đồ họa matplotlib chuẩn khoa học
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-v0_8-whitegrid')

# Đường dẫn dữ liệu và thư mục xuất
DATA_PATH = os.path.join("data", "processed", "vn_coffee_clustered.csv")
FIGURES_DIR = "figures"
REPORTS_DIR = "Reports"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_du_lieu_phan_loai(filepath=DATA_PATH):
    """
    Đọc dữ liệu đã gom cụm từ vn_coffee_clustered.csv, trích xuất đặc trưng Tháng (Month)
    và chuẩn bị ma trận X cùng vector nhãn mục tiêu Y_class.
    """
    if not os.path.exists(filepath):
        alternate_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alternate_path):
            filepath = alternate_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại '{filepath}'!")

    df = pd.read_csv(filepath)
    df['Month'] = pd.to_datetime(df['Date']).dt.month

    feature_cols = ['London_Robusta_Price', 'Crude_Oil_Price', 'VN_Production', 'Month']
    target_col = 'Cluster'

    # Ánh xạ tên lớp trực quan
    class_names = [
        "Cụm 1: Trầm lắng (Low-Price)",
        "Cụm 2: Ổn định (Moderate)",
        "Cụm 3: Siêu chu kỳ (Booming)"
    ]

    print(f"- Tải thành công dữ liệu phân loại: {len(df)} dòng quan sát.")
    print("- Phân phối các trạng thái thị trường:")
    for c_id, c_name in zip([1, 2, 3], class_names):
        count = (df[target_col] == c_id).sum()
        pct = count / len(df) * 100
        print(f"  + {c_name:<30}: {count:>3} tháng ({pct:>5.1f}%)")

    return df, feature_cols, target_col, class_names


def huan_luyen_cay_quyet_dinh(df, feature_cols, target_col, class_names):
    """
    Huấn luyện Cây quyết định CART (DecisionTreeClassifier).
    Đánh giá trên tập Train/Test (80/20) và huấn luyện trên toàn bộ dữ liệu để sinh tập luật tổng quát.
    """
    X = df[feature_cols]
    y = df[target_col]

    # Chia Train (80%) và Test (20%) có phân tầng (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf_split = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
    clf_split.fit(X_train, y_train)
    y_pred_test = clf_split.predict(X_test)

    acc_test = accuracy_score(y_test, y_pred_test)

    # Huấn luyện trên toàn bộ dữ liệu để trích xuất cây luật hoàn chỉnh
    clf_full = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
    clf_full.fit(X, y)
    y_pred_full = clf_full.predict(X)

    acc_full = accuracy_score(y, y_pred_full)
    cm = confusion_matrix(y, y_pred_full)
    cr = classification_report(y, y_pred_full, target_names=class_names, digits=4)

    print("\n" + "=" * 85)
    print("TASK 11.2 & 11.3: KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI CÂY QUYẾT ĐỊNH CART")
    print("=" * 85)
    print(f"- Độ chính xác kiểm thử (Test Accuracy - 20% Holdout): {acc_test * 100:.2f}%")
    print(f"- Độ chính xác toàn bộ mẫu (Full Dataset Accuracy)     : {acc_full * 100:.2f}%")
    
    print("\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT) ---")
    print(cr)

    print("--- MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
    print(pd.DataFrame(cm, index=[f"Thực tế: Cụm {i}" for i in [1, 2, 3]],
                          columns=[f"Dự báo: Cụm {i}" for i in [1, 2, 3]]))

    return clf_full, cm, cr, acc_full, acc_test


def ve_truc_quan_cay_quyet_dinh(clf, feature_cols, class_names):
    """
    Vẽ cấu trúc Cây quyết định siêu nét (300 DPI) lưu vào figures/fig18_decision_tree_rules.png.
    """
    plt.figure(figsize=(16, 9))
    
    # Định dạng tên hiển thị của các biến
    feature_labels = [
        "Giá London (USD/Tấn)",
        "Giá Dầu Brent (USD/Thùng)",
        "Sản lượng VN (Nghìn Tấn)",
        "Tháng (Month)"
    ]

    short_class_names = ["1. Trầm lắng", "2. Ổn định", "3. Siêu chu kỳ"]

    plot_tree(
        clf,
        feature_names=feature_labels,
        class_names=short_class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        precision=1,
        impurity=True
    )

    plt.title("CẤU TRÚC CÂY QUYẾT ĐỊNH CART PHÂN LOẠI TRẠNG THÁI THỊ TRƯỜNG CÀ PHÊ VIỆT NAM",
              fontsize=14, fontweight="bold", pad=16, color="#1a202c")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "fig18_decision_tree_rules.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n- Đã lưu cấu trúc Cây quyết định: '{out_path}' (Độ phân giải 300 DPI).")
    return out_path


def trich_xuat_tap_luat_if_then(clf, feature_names, class_names):
    """
    Duyệt đệ quy cây quyết định để trích xuất tập luật kinh doanh dạng IF ... AND ... THEN ...
    """
    tree_ = clf.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, path, conditions):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Nhánh trái (<= threshold)
            left_cond = f"{name} <= {threshold:.2f}"
            recurse(tree_.children_left[node], path + [node], conditions + [left_cond])

            # Nhánh phải (> threshold)
            right_cond = f"{name} > {threshold:.2f}"
            recurse(tree_.children_right[node], path + [node], conditions + [right_cond])
        else:
            # Nút lá (Leaf Node)
            samples = tree_.n_node_samples[node]
            val = tree_.value[node][0]
            pred_class_idx = np.argmax(val)
            pred_class_name = class_names[pred_class_idx]
            confidence = val[pred_class_idx] * 100.0
            n_class_samples = int(round(val[pred_class_idx] * samples))
            rule_str = " AND ".join(conditions)
            rule_full = (
                f"NẾU (IF) {rule_str}\n"
                f"--> THÌ (THEN) TRẠNG THÁI = [{pred_class_name.upper()}]\n"
                f"    [Độ chính xác / Tin cậy: {confidence:.2f}% | Mẫu thuộc lớp: {n_class_samples}/{samples} tháng]"
            )
            rules.append((pred_class_idx + 1, rule_full))

    recurse(0, [], [])

    print("\n" + "=" * 85)
    print("TASK 11.5: TẬP LUẬT KINH DOANH (BUSINESS RULES) TRÍCH XUẤT TỪ CÂY QUYẾT ĐỊNH")
    print("=" * 85)
    for idx, (c_id, rule) in enumerate(rules, 1):
        print(f"\n[LUẬT SỐ {idx:02d}]:")
        print(rule)

    return rules


def main():
    print("=" * 90)
    print("BƯỚC 11: MÔ HÌNH PHÂN LOẠI CÓ GIÁM SÁT VÀ TRÍCH XUẤT TẬP LUẬT CÂY QUYẾT ĐỊNH")
    print("=" * 90)

    # 1. Nạp dữ liệu
    df, feature_cols, target_col, class_names = load_du_lieu_phan_loai()

    # 2. Huấn luyện và đánh giá Decision Tree CART
    clf, cm, cr, acc_full, acc_test = huan_luyen_cay_quyet_dinh(
        df, feature_cols, target_col, class_names
    )

    # 3. Vẽ biểu đồ cấu trúc cây (fig18)
    ve_truc_quan_cay_quyet_dinh(clf, feature_cols, class_names)

    # 4. Trích xuất tập luật IF - THEN
    rules = trich_xuat_tap_luat_if_then(clf, feature_cols, class_names)

    # 5. Tự kiểm tra (Self-Test Assertions)
    print("\n" + "=" * 90)
    print("TỰ KIỂM TRA (SELF-TEST) CHUẨN ĐỒ ÁN DA - SGU:")
    print("=" * 90)
    assert acc_full >= 0.90, f"Lỗi: Độ chính xác toàn cục < 90%: {acc_full*100:.2f}%"
    assert acc_test >= 0.85, f"Lỗi: Độ chính xác kiểm thử < 85%: {acc_test*100:.2f}%"
    assert len(rules) >= 4, f"Lỗi: Số lượng luật trích xuất < 4: {len(rules)}"
    print(f"- Độ chính xác toàn bộ mẫu : {acc_full*100:.2f}% (Đạt tiêu chuẩn xuất sắc > 90%)")
    print(f"- Độ chính xác kiểm thử Test: {acc_test*100:.2f}% (Đạt tiêu chuẩn vững > 85%)")
    print(f"- Số luật kinh doanh trích xuất: {len(rules)} luật.")
    print("- Toàn bộ các kiểm định Self-Test: PASS 100%!")

    return clf, rules, cm, cr


if __name__ == "__main__":
    main()
