import os
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UserTower
from user_dataset import UserHistoryDataset, collate_fn_user

def extract_user_embeddings(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- BẮT ĐẦU TRÍCH XUẤT OFFLINE USER EMBEDDINGS (Device: {device}) ---")

    user_tower = UserTower(d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers).to(device)

    checkpoint_path = os.path.join(args.checkpoint_path, 'user_tower.pth')
    print(f"Đang tải checkpoint từ: {checkpoint_path}")
    user_tower.load_state_dict(torch.load(checkpoint_path, map_location=device))
    user_tower.eval() 

    print("\n--- Creating Inference DataLoader (shuffle=False) ---")

    test_df = pd.read_pickle(os.path.join(args.data_dir, 'user_all.pkl'))
    review_dict_path = os.path.join(args.data_dir, 'precomputed_review_embs.pt')
    item_dict_path = os.path.join(args.data_dir, 'item_embeddings.pt')

    full_dataset = UserHistoryDataset(
        user_df=test_df,
        review_dict_path=review_dict_path,
        item_dict_path=item_dict_path,
        max_seq_len=args.max_seq_len,
        pad_value=0,
        mode='test' 
    )
    
    inference_loader = DataLoader(
        full_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=2, pin_memory=True, collate_fn=collate_fn_user
    )
    
    print(f"DataLoader đã được tạo với {len(inference_loader)} batch (Tổng số người dùng: {len(full_dataset)}).")

    all_user_embs = []

    with torch.no_grad():
        for batched_user_dict, targets in tqdm(inference_loader, desc="Inferencing"):
            batched_user_dict = {k: v.to(device) for k, v in batched_user_dict.items()}
            
            with torch.amp.autocast('cuda'):
                user_embeddings = user_tower(batched_user_dict)
                user_embeddings = torch.nn.functional.normalize(user_embeddings, p=2, dim=1)

                all_user_embs.append(user_embeddings.cpu())

    print("\n--- Inference Complete. Concatenating results... ---")
    user_embeddings_all = torch.cat(all_user_embs, dim=0)
    
    print(f"Final User Embeddings Shape: {user_embeddings_all.shape}")
    print(f"(Expected: [{len(full_dataset)}, 256])")

    print("\nDictionary Mapping...")
    user_dict = {}

    for idx in tqdm(range(len(full_dataset)), desc="Packing"):
        raw_user_id = str(full_dataset.df.iloc[idx]['user_id'])
        user_dict[raw_user_id] = user_embeddings_all[idx]

    output_file = os.path.join(args.output_dir, 'user_embeddings.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(user_dict, output_file)
    
    print(f"\nĐã lưu thành công từ điển Vector Người dùng (User Embeddings) vào:\n {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất User Embeddings Offline")

    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa user_all.pkl, item_embeddings.pt, ...")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Đường dẫn đến file model user_tower.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256, help="Kích thước vector embedding")

    args = parser.parse_args()
    extract_user_embeddings(args)