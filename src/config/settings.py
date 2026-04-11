"""
Tập lệnh cấu hình trung tâm (Central Configuration) cho dự án MVS-XAI.
Nhà phát triển: MVS Team
"""
import os

# --- MAPPING TỪ ĐIỂN VESTA IEEE-CIS (BIẾN ĐỔI NGỮ NGHĨA KINH DOANH) ---
GLOBAL_DATA_DICTIONARY = {
    "TransactionAmt": "Số Tiền Giao Dịch",
    "ProductCD": "Kênh Sản Phẩm",
    "card1": "Mã Thẻ Chính",
    "card2": "Mã Chi Nhánh Thẻ",
    "card6": "Trạng Thái Thẻ",
    "dist1": "Khoảng Cách Địa Lý",
    "C1": "SL Email Lạ Trùng Lặp",
    "C2": "SL Thiết Bị Cùng IP",
    "C4": "SL Thiết Bị Đáng Ngờ",
    "C5": "Khóa OTP Thất Bại",
    "C6": "Tần Suất Bị Từ Chối",
    "C13": "Mật Độ Giao Dịch Cùng ID",
    "C14": "SL Số Điện Thoại Ảo",
    "V303": "Chỉ Báo Botnet",
    "Card_Velocity": "Vận Tốc Quẹt Thẻ"
}

# --- CẤU HÌNH LIÊN KẾT BÊN NGOÀI (EXTERNAL API) ---
# Tự động lấy từ Biến môi trường, nếu không có thì chạy MOCK MODE (giả lập)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "MOCK_MODE")

# --- SIÊU THAM SỐ (HYPERPARAMETERS) ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
