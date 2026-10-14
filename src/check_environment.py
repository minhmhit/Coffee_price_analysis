# ==============================================================================
# TẬP TIN: src/check_environment.py
# MỤC ĐÍCH: Kiểm tra tính sẵn sàng của môi trường ảo Python (.venv) và các thư viện Phân tích Dữ liệu
# QUY CHUẨN: Tuân thủ quy định DA-SGU - Chú thích chi tiết từng dòng cho Newbie DA
# ==============================================================================

# Import thư viện sys để kiểm tra phiên bản và cấu hình luồng xuất ký tự UTF-8
import sys

# Import thư viện os để thao tác với đường dẫn thư mục và hệ điều hành
import os

# Import thư viện platform để lấy thông tin hệ điều hành (Windows, Linux, macOS)
import platform

# Thiết lập bảng mã UTF-8 cho đầu ra console trên Windows tránh lỗi cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def kiem_tra_moi_truong():
    """
    Hàm thực hiện kiểm tra cấu hình môi trường Python và xác nhận cài đặt các thư viện cần thiết.
    Sử dụng assertion để tự động kiểm thử (Self-Test).
    """
    print("=" * 65)
    print("BẮT ĐẦU KIỂM TRA MÔI TRƯỜNG PHÂN TÍCH DỮ LIỆU (DA-SGU)")
    print("=" * 65)

    # 1. Kiểm tra phiên bản Python
    # sys.version chứa chuỗi mô tả chi tiết phiên bản Python đang chạy
    python_version = sys.version.replace('\n', ' ')
    print(f"[THÔNG TIN] Phiên bản Python: {python_version}")
    print(f"[THÔNG TIN] Đường dẫn thực thi Python: {sys.executable}")
    print(f"[THÔNG TIN] Hệ điều hành: {platform.system()} {platform.release()} ({platform.architecture()[0]})")

    # Đảm bảo Python đang chạy là phiên bản 3.9 trở lên
    assert sys.version_info >= (3, 9), "Lỗi: Phiên bản Python phải từ 3.9 trở lên!"

    # 2. Danh sách các thư viện cốt lõi bắt buộc cho môn Phân tích Dữ liệu
    cac_thu_vien_bat_buoc = [
        "numpy",        # Thư viện tính toán đại số tuyến tính và xử lý ma trận/mảng đa chiều
        "pandas",       # Thư viện xử lý và phân tích dữ liệu dạng bảng (DataFrame)
        "scipy",        # Thư viện tính toán khoa học và kiểm định thống kê (t-test, Shapiro, v.v.)
        "statsmodels",  # Thư viện xây dựng mô hình kinh tế lượng và hồi quy OLS chuyên sâu
        "sklearn",      # Thư viện học máy (Machine Learning) hỗ trợ tiền xử lý và phân cụm K-Means
        "matplotlib",   # Thư viện trực quan hóa dữ liệu và vẽ biểu đồ nền tảng
        "seaborn",      # Thư viện vẽ biểu đồ thống kê nâng cao với giao diện trực quan
        "openpyxl"      # Thư viện đọc và ghi file Excel (.xlsx) như tập dữ liệu Pink Sheet
    ]

    ket_qua_kiem_tra = {}

    # Lặp qua từng thư viện để import và kiểm tra phiên bản
    for ten_thu_vien in cac_thu_vien_bat_buoc:
        try:
            # Sử dụng __import__ để nạp động thư viện theo tên chuỗi
            module = __import__(ten_thu_vien)
            # Lấy số phiên bản của thư viện (nếu có)
            phien_ban = getattr(module, "__version__", "Không xác định")
            ket_qua_kiem_tra[ten_thu_vien] = {
                "trang_thai": "PASS",
                "phien_ban": phien_ban
            }
            print(f"[PASS] Thư viện '{ten_thu_vien:<12}' | Trạng thái: SẴN SÀNG | Phiên bản: {phien_ban}")
        except ImportError as e:
            ket_qua_kiem_tra[ten_thu_vien] = {
                "trang_thai": "FAIL",
                "loi": str(e)
            }
            print(f"[FAIL] Thư viện '{ten_thu_vien:<12}' | CHƯA được cài đặt! Lỗi: {e}")

    # 3. Chạy Self-Test với câu lệnh assert theo quy tắc số 5
    # Kiểm tra xem có thư viện nào bị lỗi (FAIL) hay không
    cac_thu_vien_loi = [k for k, v in ket_qua_kiem_tra.items() if v["trang_thai"] == "FAIL"]
    
    assert len(cac_thu_vien_loi) == 0, f"Self-Test Thất bại! Các thư viện chưa cài: {cac_thu_vien_loi}"

    print("-" * 65)
    print("SELF-TEST THÀNH CÔNG: Mọi thư viện bắt buộc đã được cài đặt và kiểm thử 100%!")
    print("=" * 65)
    
    return ket_qua_kiem_tra

if __name__ == "__main__":
    # Điểm khởi đầu của chương trình kiểm tra
    kiem_tra_moi_truong()
