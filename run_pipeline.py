# ==============================================================================
# ĐỒ ÁN PHÂN TÍCH DỮ LIỆU GIÁ CÀ PHÊ TÂY NGUYÊN (CHUẨN GIÁO TRÌNH DA - SGU)
# SCRIPT TỰ ĐỘNG CHẠY TOÀN BỘ QUY TRÌNH (RUN-ALL PIPELINE AUTOMATION)
# File: run_pipeline.py
# Mục tiêu: Thực thi tuần tự từ Bước 0 đến Bước 7 và xác thực kết quả toàn diện
# ==============================================================================

import os
import sys
import time
import subprocess

# Cấu hình UTF-8 cho Windows console
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Danh sách 8 bước nghiên cứu tuần tự theo đúng tiến trình KDD
STEPS = [
    ("Bước 0: Kiểm tra môi trường hệ thống & Python venv", "src/check_environment.py"),
    ("Bước 1: Tích hợp dữ liệu đa nguồn & Đồng bộ tần suất", "src/step_01_data_integration.py"),
    ("Bước 2: Khám phá EDA & Làm sạch dữ liệu theo chuẩn 3-Sigma", "src/step_02_eda_cleaning.py"),
    ("Bước 3: Biến đổi dữ liệu, Mã hóa & Phân tích tương quan Pearson", "src/step_03_transformation_correlation.py"),
    ("Bước 4: Lựa chọn biến Stepwise & Ước lượng mô hình OLS", "src/step_04_ols_stepwise_modeling.py"),
    ("Bước 5: Kiểm định nghiêm ngặt 4 giả định kinh tế lượng", "src/step_05_assumption_tests.py"),
    ("Bước 6: Đánh giá chất lượng mô hình & Dự báo 3 kịch bản", "src/step_06_evaluation_prediction.py"),
    ("Bước 7: Khai phá dữ liệu nâng cao — Gom cụm thị trường K-Means", "src/step_07_clustering_mining.py"),
]

def main():
    print("=" * 80)
    print("CHƯƠNG TRÌNH TỰ ĐỘNG CHẠY TOÀN BỘ ĐỒ ÁN PHÂN TÍCH DỮ LIỆU GIÁ CÀ PHÊ (DA - SGU)")
    print("=" * 80)
    start_total_time = time.time()
    
    python_executable = sys.executable
    print(f"-> Trình thông dịch Python đang sử dụng: {python_executable}\n")
    
    success_count = 0
    for idx, (step_name, script_path) in enumerate(STEPS):
        print("-" * 80)
        print(f"[{idx}/7] ĐANG THỰC THI: {step_name}")
        print(f"      Đường dẫn script: {script_path}")
        print("-" * 80)
        
        t0 = time.time()
        result = subprocess.run([python_executable, script_path], capture_output=False)
        elapsed = time.time() - t0
        
        if result.returncode == 0:
            print(f"\n=> [KẾT QUẢ]: THÀNH CÔNG (Thời gian: {elapsed:.2f} giây)\n")
            success_count += 1
        else:
            print(f"\n=> [LỖI]: Bước {step_name} gặp sự cố! Mã lỗi: {result.returncode}")
            print("Dừng quy trình tự động.")
            sys.exit(1)
            
    total_elapsed = time.time() - start_total_time
    print("=" * 80)
    print("TỔNG KẾT: TẤT CẢ CÁC BƯỚC ĐÃ ĐƯỢC THỰC THI THÀNH CÔNG 100%!")
    print(f"Số bước hoàn thành: {success_count}/{len(STEPS)}")
    print(f"Tổng thời gian chạy toàn bộ đồ án: {total_elapsed:.2f} giây")
    print("=" * 80)
    print("Tất cả tệp dữ liệu sạch, bảng phân cụm, 11 biểu đồ 300 DPI và các báo cáo")
    print("trong thư mục data/, figures/, reports/ đã được cập nhật đồng bộ hoàn toàn!")

if __name__ == '__main__':
    main()
