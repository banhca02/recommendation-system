import argparse
import subprocess
import sys
import os

def run_command(command, step_name):
    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU: {step_name}")
    print(f"Lệnh: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(command, check=True)
        print(f"\nHOÀN TẤT: {step_name}")
    except subprocess.CalledProcessError as e:
        print(f"\nLỖI NGHIÊM TRỌNG: Bước '{step_name}' thất bại!")
        print("Dừng toàn bộ Pipeline.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="One-Click Pipeline: Multi-modal Two-Tower Recommendation")
    
    parser.add_argument("--item_data", type=str, required=True, help="Đường dẫn file JSON metadata của Item")
    parser.add_argument("--user_data", type=str, required=True, help="Đường dẫn file JSON lịch sử của User")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Thư mục lưu kết quả toàn dự án")
    
    args = parser.parse_args()

    phase1_data_dir = os.path.join(args.output_dir, "phase1_data")
    phase2_data_dir = os.path.join(args.output_dir, "phase2_data")
    item_checkpoints_dir = os.path.join(args.output_dir, "checkpoints/item_tower")
    user_checkpoints_dir = os.path.join(args.output_dir, "checkpoints/user_tower")
    
    os.makedirs(phase1_data_dir, exist_ok=True)
    os.makedirs(phase2_data_dir, exist_ok=True)
    os.makedirs(item_checkpoints_dir, exist_ok=True)

    cmd_prep_item = [
        sys.executable, "src/phase1_item_tower/data_pipeline.py",
        "--input", args.item_data,
        "--output_dir", phase1_data_dir
    ]
    run_command(cmd_prep_item, "BƯỚC 1 - Tiền xử lý dữ liệu Sản phẩm")

    cmd_prep_user = [
        sys.executable, "src/phase2_user_tower/data_pipeline.py",
        "--input", args.user_data,
        "--output_dir", phase2_data_dir
    ]
    run_command(cmd_prep_user, "BƯỚC 2 - Cắt Leave-One-Out dữ liệu Người dùng")

    cmd_train_item = [
        sys.executable, "src/phase1_item_tower/train.py",
        "--data_dir", phase1_data_dir,
        "--save_dir", item_checkpoints_dir,

        "--epochs", "100",
    ]
    run_command(cmd_train_item, "BƯỚC 3 - Huấn luyện Tháp Sản phẩm (Item Tower)")

    cmd_prep_user = [
        sys.executable, "src/phase1_item_tower/inference.py",
        "--data_dir", phase1_data_dir,
        "--checkpoint_path", item_checkpoints_dir,
        "--output_dir", phase2_data_dir
    ]
    run_command(cmd_prep_user, "BƯỚC 4 - Inference tháp sản phẩm")

    cmd_train_user = [
        sys.executable, "src/phase2_user_tower/train.py",
        "--data_dir", phase2_data_dir,
        "--checkpoint_path", item_checkpoints_dir,
        "--save_dir", user_checkpoints_dir,
    
        "--epochs", "20",
        "--batch_size", "64",
        "--max_seq_len", "50",
        "--num_layers", "2",
        "--nhead", "4",
        "--lr", "1e-4"
    ]
    run_command(cmd_train_user, "BƯỚC 5 - Huấn luyện Tháp Chuỗi (User Tower - SASRec)")

    user_model_path = os.path.join(user_checkpoints_dir, "user_tower.pth")
    
    cmd_test_user = [
        sys.executable, "src/phase2_user_tower/test.py",
        "--data_dir", phase2_data_dir,
        "--checkpoint_path", user_model_path,

        "--max_seq_len", "50",
        "--num_layers", "2",
        "--nhead", "4",
        "--batch_size", "64" 
    ]
    run_command(cmd_test_user, "BƯỚC 6 - Đánh giá Mô hình trên Tập Test (Full Ranking)")

    print(f"\nTẤT CẢ ĐÃ XONG! Mọi kết quả, configs và mô hình được lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()