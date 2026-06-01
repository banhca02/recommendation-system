import os
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from models import UserTower
from losses import SymmetricExplicitContrastiveLoss
from user_dataset import UserHistoryDataset, collate_fn_user
from extract_review_features import extract_review_features

def train_user_tower(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- BẮT ĐẦU HUẤN LUYỆN USER TOWER TẠI {device.type.upper()} ---")

    item_dict_path = os.path.join(args.data_dir, 'item_embeddings.pt')
    print(f"Đang tải từ điển Item cho Targets/Negatives: {item_dict_path}")
    item_dict = torch.load(item_dict_path, map_location='cpu')
 
    d_model = next(iter(item_dict.values())).shape[0] if item_dict else 256

    review_dict_path = os.path.join(args.data_dir, 'precomputed_review_embs.pt')

    extract_review_features(
        data_path=os.path.join(args.data_dir, 'user_all.pkl'), 
        checkpoint_path=args.checkpoint_path,
        output_path=review_dict_path
    )
    
    print("\n--- Đang chuẩn bị Dữ liệu (Datasets) ---")
    
    train_df = pd.read_pickle(os.path.join(args.data_dir, 'user_train.pkl'))
    val_df = pd.read_pickle(os.path.join(args.data_dir, 'user_val.pkl'))

    all_item_ids = []
    for k in item_dict.keys():
        try:
            all_item_ids.append(int(k))
        except ValueError:
            all_item_ids.append(k) 
            
    print(f"Tổng số Item có sẵn để lấy mẫu Negative: {len(all_item_ids)}")

    train_dataset = UserHistoryDataset(
        user_df=train_df,
        review_dict_path=review_dict_path,
        item_dict_path=item_dict_path,
        max_seq_len=args.max_seq_len,
        pad_value=args.pad_value,
        mask_prob=args.mask_prob,
        mode='train',
        all_item_ids=all_item_ids
    )

    val_dataset = UserHistoryDataset(
        user_df=val_df,
        review_dict_path=review_dict_path,
        item_dict_path=item_dict_path,
        max_seq_len=args.max_seq_len,
        pad_value=args.pad_value,
        mode='val',
        all_item_ids=all_item_ids
    )
    
    print("\n--- Khởi tạo DataLoaders ---")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,        
        num_workers=2,
        pin_memory=True,    
        collate_fn=collate_fn_user
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,      
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn_user
    )

    print(f"SẴN SÀNG HUẤN LUYỆN:")
    print(f"Train Batches: {len(train_loader)} (từ {len(train_dataset)} user sequences)")
    print(f"Val Batches:   {len(val_loader)} (từ {len(val_dataset)} users)")

    user_tower = UserTower(d_model=d_model, nhead=args.nhead, num_layers=args.num_layers).to(device)
    
    criterion = SymmetricExplicitContrastiveLoss(temperature=args.temperature).to(device)
    
    optimizer = optim.AdamW(user_tower.parameters(), lr=args.lr, weight_decay=5e-2)
    grad_scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []

    print("\n--- Tiến hành Vòng lặp Huấn luyện ---")
    for epoch in range(args.epochs):
        user_tower.train() 
        total_train_loss = 0
        loop_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batched_user_dict, pos_item_embs, neg_item_embs in loop_train:
            optimizer.zero_grad(set_to_none=True)

            pos_item_embs = pos_item_embs.to(device)
            neg_item_embs = neg_item_embs.to(device)
            
            with torch.amp.autocast('cuda'):
                user_embs = user_tower(batched_user_dict) 
                loss = criterion(user_embs, pos_item_embs, neg_item_embs)
            
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            total_train_loss += loss.item()
            loop_train.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        user_tower.eval() 
        total_val_loss = 0
        loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        
        with torch.no_grad(): 
            for batched_user_dict, pos_item_embs, neg_item_embs in loop_val:

                pos_item_embs = pos_item_embs.to(device)
                neg_item_embs = neg_item_embs.to(device)
                
                with torch.amp.autocast('cuda'):
                    user_embs = user_tower(batched_user_dict)
                    loss = criterion(user_embs, pos_item_embs, neg_item_embs)
                
                total_val_loss += loss.item()
                loop_val.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            print(f"Val Loss cải thiện từ {best_val_loss:.4f} xuống {avg_val_loss:.4f}.")
            best_val_loss = avg_val_loss 
            
            save_path = os.path.join(args.save_dir, 'user_tower.pth')
            torch.save(user_tower.state_dict(), save_path)
        else:
            print(f" (Val Loss không cải thiện so với {best_val_loss:.4f})")
        print("-" * 50)

    history_data = {"train_loss": train_loss_history, "val_loss": val_loss_history}
    with open(os.path.join(args.save_dir, 'user_training_history.json'), 'w') as f:
        json.dump(history_data, f)

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Huấn luyện User Tower (SASRec + Symmetric Contrastive Loss)")

    # ==========================================
    # ĐƯỜNG DẪN DỮ LIỆU VÀ LƯU TRỮ
    # ==========================================
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Thư mục chứa dữ liệu Phase 2 (user_*.pkl, item_embeddings.pt, precomputed_review_embs.pt)")
    parser.add_argument("--save_dir", type=str, required=True, 
                        help="Thư mục lưu Model Checkpoints và Training History")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Thư mục lưu Model Checkpoints của item tower")

    # ==========================================
    # CẤU HÌNH DATASET & CHUỖI 
    # ==========================================
    parser.add_argument("--max_seq_len", type=int, default=50, 
                        help="Độ dài tối đa của chuỗi lịch sử người dùng (Sliding Window)")
    parser.add_argument("--pad_value", type=int, default=0, 
                        help="Giá trị dùng để đệm (Padding) cho các chuỗi ngắn")
    parser.add_argument("--mask_prob", type=float, default=0.1, 
                        help="Tỷ lệ mask ngẫu nhiên item trong lúc train (Data Augmentation)")

    # ==========================================
    # CẤU HÌNH KIẾN TRÚC MÔ HÌNH 
    # ==========================================
    parser.add_argument("--num_layers", type=int, default=2, 
                        help="Số lớp Transformer Encoder (SASRec)")
    parser.add_argument("--nhead", type=int, default=4, 
                        help="Số lượng Attention Heads")
    
    # ==========================================
    # HYPERPARAMETERS
    # ==========================================
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Kích thước Batch")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Số lượng Epochs huấn luyện")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning Rate (AdamW)")
    parser.add_argument("--temperature", type=float, default=0.07, 
                        help="Nhiệt độ (Temperature) cho InfoNCE / Contrastive Loss")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train_user_tower(args)