import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import ItemTower
from item_dataset import ItemMetadataDataset, collate_fn_item

def extract_fused_item_embeddings(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- BẮT ĐẦU TRÍCH XUẤT OFFLINE ITEM EMBEDDINGS (Device: {device}) ---")

    item_tower = ItemTower(text_in_dim=768, img_in_dim=768, embed_dim=256).to(device)

    checkpoint_path = os.path.join(args.checkpoint_path, 'item_tower.pth')
    print(f"Đang tải checkpoint từ: {checkpoint_path}")
    item_tower.load_state_dict(torch.load(checkpoint_path, map_location=device))
    item_tower.eval() 

    print("\n--- Creating Inference DataLoader (shuffle=False) ---")

    image_pt = os.path.join(args.checkpoint_path, 'precomputed_image_embs.pt')
    text_pt = os.path.join(args.checkpoint_path, 'precomputed_text_embs.pt')

    full_dataset = ItemMetadataDataset(
        data_dir=args.data_dir, 
        filename='item_metadata_encoded.pkl', 
        image_pt_path=image_pt, 
        text_pt_path=text_pt
    )
    
    inference_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size, 
        shuffle=False,    
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn_item
    )
    
    print(f"DataLoader created with {len(inference_loader)} batches (Total items: {len(full_dataset)}).")

    all_fused_embs = []

    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Running Full Inference"):
            text_feats = batch['text_feats']
            img_feats = batch['img_feats']
            
            with torch.amp.autocast('cuda'):
                _, _, fused_emb, _ = item_tower(text_feats, img_feats)

            all_fused_embs.append(fused_emb.cpu())

    print("\n--- Inference Complete. Concatenating results... ---")
    fused_embeddings_all = torch.cat(all_fused_embs, dim=0)
    
    print(f"Final Item Embeddings Shape: {fused_embeddings_all.shape}")
    print(f"(Expected: [{len(full_dataset)}, 256])")

    print("\nDictionary Mapping...")
    item_dict = {}

    for idx in tqdm(range(len(full_dataset)), desc="Packing"):
        raw_item_id = str(full_dataset.df.iloc[idx]['item_idx'])
        item_dict[raw_item_id] = fused_embeddings_all[idx]

    output_file = os.path.join(args.output_dir, 'item_embeddings.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(item_dict, output_file)
    
    print(f"\nĐã lưu thành công từ điển Vector Sản phẩm (Item Embeddings) vào:\n {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất Item Embeddings Offline")

    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục Phase 1 chứa dữ liệu")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Đường dẫn file item_tower.pth")
    parser.add_argument("--output_dir", type=str, required=True, help="Nơi lưu kết quả (ví dụ: data/phase2/item_embeddings.pt)")
    parser.add_argument("--batch_size", type=int, default=256, help="Kích thước batch khi inference")

    args = parser.parse_args()
    extract_fused_item_embeddings(args)