import os
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UserTower
from user_dataset import UserHistoryDataset, collate_fn_user

def calculate_metrics_full_ranking(user_embeddings, target_items, item_embs_gpu, k_list=[10, 20]):
    scores = torch.matmul(user_embeddings, item_embs_gpu.T)

    scores[:, 0] = -float('inf')

    max_k = max(k_list)
    _, topk_indices = torch.topk(scores, max_k, dim=1) # [Batch, max_k]

    target_embs = target_items.clone().detach().to(scores.device)

    target_sim_scores = torch.matmul(target_embs, item_embs_gpu.T)

    target_indices = target_sim_scores.argmax(dim=1).view(-1, 1)

    hits_matrix = (topk_indices == target_indices) 

    results = {}
    for k in k_list:
        hits_k = hits_matrix[:, :k] 
        
        batch_hits = hits_k.any(dim=1).sum().item()
        nonzero_indices = hits_k.nonzero(as_tuple=False)
        
        ndcg_sum = 0.0
        mrr_sum = 0.0
        
        if len(nonzero_indices) > 0:
            ranks = nonzero_indices[:, 1].float() # 0-based rank
            mrr_sum = (1.0 / (ranks + 1.0)).sum().item()
            ndcg_sum = (1.0 / torch.log2(ranks + 2.0)).sum().item()
            
        results[k] = {
            'hits': batch_hits,
            'ndcg': ndcg_sum,
            'mrr': mrr_sum
        }
        
    return results

def test_user_tower(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- BẮT ĐẦU ĐÁNH GIÁ (TESTING) TẠI {device.type.upper()} ---")

    item_dict_path = os.path.join(args.data_dir, 'item_embeddings.pt')
    print(f"Đang tải từ điển Item cho Ranking: {item_dict_path}")
    item_dict = torch.load(item_dict_path, map_location='cpu')

    print("Đang tạo Ma trận Item phục vụ Full-Ranking...")
    
    max_id = max([int(k) for k in item_dict.keys()]) 
    d_model = next(iter(item_dict.values())).shape[0]
    
    item_embs_gpu = torch.zeros((max_id + 1, d_model), device=device)

    for item_id, emb_tensor in item_dict.items():
        idx = int(item_id)
        item_embs_gpu[idx] = emb_tensor.to(device)

    item_embs_gpu = torch.nn.functional.normalize(item_embs_gpu, p=2, dim=1)

    test_df = pd.read_pickle(os.path.join(args.data_dir, 'user_test.pkl'))
    review_dict_path = os.path.join(args.data_dir, 'precomputed_review_embs.pt')
    
    test_dataset = UserHistoryDataset(
        user_df=test_df,
        review_dict_path=review_dict_path,
        item_dict_path=item_dict_path,
        max_seq_len=args.max_seq_len,
        pad_value=0,
        mode='test' 
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=2, pin_memory=True, collate_fn=collate_fn_user
    )

    user_tower_model = UserTower(d_model=d_model, nhead=args.nhead, num_layers=args.num_layers).to(device)
    print(f"Đang tải trọng số đã huấn luyện từ: {args.checkpoint_path}")
    user_tower_model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    user_tower_model.eval()

    k_list = [1, 5, 10, 20, 50]
    total_metrics = {k: {'hits': 0, 'ndcg': 0, 'mrr': 0} for k in k_list}
    total_samples = 0
    
    print(f"\nBắt đầu đánh giá FULL-RANKING (Quét toàn bộ {item_embs_gpu.size(0)} Items) trên {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batched_user_dict, targets in tqdm(test_loader, desc="Testing"):
            batched_user_dict = {k: v.to(device) for k, v in batched_user_dict.items()}
            
            with torch.amp.autocast('cuda'):
                user_embeddings = user_tower_model(batched_user_dict)
                user_embeddings = torch.nn.functional.normalize(user_embeddings, p=2, dim=1)

                batch_results = calculate_metrics_full_ranking(
                    user_embeddings=user_embeddings, 
                    target_items=targets, 
                    item_embs_gpu=item_embs_gpu, 
                    k_list=k_list
                )

            for k in k_list:
                total_metrics[k]['hits'] += batch_results[k]['hits']
                total_metrics[k]['ndcg'] += batch_results[k]['ndcg']
                total_metrics[k]['mrr']  += batch_results[k]['mrr']
            
            total_samples += targets.size(0)

    if total_samples == 0:
        print("Không có dữ liệu đánh giá.")
        return

    k_values = sorted(total_metrics.keys())
    
    header = f"{'Metric':<10}"
    for k in k_values:
        header += f" | @{k:<8}"
        
    line_len = len(header)
    print("\n" + "-" * line_len)
    print(header)
    print("-" * line_len)
    
    metrics_to_print = [
        ('Hit Rate', 'hits'), 
        ('NDCG', 'ndcg'), 
        ('MRR', 'mrr')
    ]
    
    for display_name, dict_key in metrics_to_print:
        row_str = f"{display_name:<10}"
        for k in k_values:
            avg_val = total_metrics[k][dict_key] / total_samples
            row_str += f" | {avg_val:<9.4f}"
        print(row_str)
        
    print("-" * line_len)
    print(f"Tổng số User Sequences được đánh giá: {total_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình User Tower")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa user_test.pkl, item_embeddings.pt, ...")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Đường dẫn đến file model user_tower.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    
    args = parser.parse_args()
    test_user_tower(args)