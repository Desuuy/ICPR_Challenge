import torch
import torch.nn as nn
from configs.config import get_default_config

def debug_loss_issue():
    config = get_default_config()
    print(f"--- Đang kiểm tra cấu hình ---")
    print(f"Số lượng class: {config.NUM_CLASSES}") # Phải là 37
    print(f"Thiết bị: {config.DEVICE}")

    # Giả lập đầu ra của Model (Logits)
    # Giả sử ảnh qua model tạo ra 32 frame (Time steps)
    # Batch size = 2, Num classes = 37
    logits = torch.randn(32, 2, config.NUM_CLASSES).to(config.DEVICE)
    
    # Giả lập Label (Mục tiêu)
    # Ví dụ biển số "ABC1234" có độ dài 7
    targets = torch.randint(1, 37, (14,), dtype=torch.long).to(config.DEVICE)
    target_lengths = torch.tensor([7, 7], dtype=torch.long).to(config.DEVICE)
    input_lengths = torch.tensor([32, 32], dtype=torch.long).to(config.DEVICE)

    # Khởi tạo CTC Loss chuẩn (Không Focal)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Tính loss thử
    loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths)
    
    print(f"\n--- Kết quả chạy thử ---")
    print(f"Loss thực tế: {loss.item()}")
    
    if loss.item() > 15:
        print("⚠️ CẢNH BÁO: Loss rất cao. Nguyên nhân thường do Model khởi tạo ngẫu nhiên hoặc ảnh quá mờ.")
    if torch.isinf(loss):
        print("❌ LỖI NẶNG: Loss là INF. Nguyên nhân: Độ dài biển số dài hơn số Frame model tạo ra!")

if __name__ == "__main__":
    debug_loss_issue()