import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pandas as pd

from models import ItemTower
from losses import ContrastiveLoss
from item_dataset import ItemMetadataDataset, collate_fn_item
from extract_image_features import extract_vit_features
from extract_text_features import extract_bert_features

def training(args):
    print(f"Thư mục dữ liệu: {args.data_dir}")
    print(f"Thư mục lưu model: {args.save_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    full_data_path = os.path.join(args.data_dir, 'item_metadata_encoded.pkl')

    image_pt = extract_vit_features(
        file_path=full_data_path, 
        output_path=os.path.join(args.save_dir, 'precomputed_image_embs.pt')
    )

    text_pt = extract_bert_features(
        file_path=full_data_path, 
        output_path=os.path.join(args.save_dir, 'precomputed_text_embs.pt')
    )

    image_pt = os.path.join(args.save_dir, 'precomputed_image_embs.pt')
    text_pt = os.path.join(args.save_dir, 'precomputed_text_embs.pt')

    train_dataset = ItemMetadataDataset(
        data_dir=args.data_dir, 
        filename='item_metadata_train.pkl', 
        image_pt_path=image_pt, 
        text_pt_path=text_pt
    )
    
    val_dataset = ItemMetadataDataset(
        data_dir=args.data_dir, 
        filename='item_metadata_val.pkl', 
        image_pt_path=image_pt, 
        text_pt_path=text_pt
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,       
        num_workers=2,
        pin_memory=True,    
        collate_fn=collate_fn_item
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,      
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn_item
    )

    criterion = ContrastiveLoss(temperature=args.temperature).to(device)
    grad_scaler = torch.amp.GradScaler('cuda')

    item_tower = ItemTower(text_in_dim=768, img_in_dim=768, embed_dim=256).to(device)

    optimizer = optim.Adam(
        list(item_tower.parameters()),
        lr=args.lr,
        weight_decay=1e-3
    )

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    
    print("\n--- Bắt đầu Huấn luyện ---")
    for epoch in range(args.epochs):
        item_tower.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        total_train_loss = 0.0
    
        for batch in loop:
            text_feats = batch['text_feats']
            img_feats = batch['img_feats']
        
            optimizer.zero_grad(set_to_none=True)
        
            with torch.amp.autocast('cuda'):
                txt_emb, img_emb, fused_emb, _ = item_tower(text_feats, img_feats)
            
                loss_1 = criterion(img_emb, txt_emb)
                loss_2 = criterion(fused_emb, txt_emb) 
                loss_3 = criterion(fused_emb, img_emb)  

                total_batch_loss = loss_1 + loss_2 + loss_3 
            
            grad_scaler.scale(total_batch_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        
            total_train_loss += total_batch_loss.item()
            loop.set_postfix(loss=total_batch_loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        item_tower.eval()
        
        loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in loop_val:
                text_feats = batch['text_feats']
                img_feats = batch['img_feats']
                
                with torch.amp.autocast('cuda'):
                    txt_emb, img_emb, fused_emb, _ = item_tower(text_feats, img_feats)
            
                    loss_1 = criterion(img_emb, txt_emb)
                    loss_2 = criterion(fused_emb, txt_emb) 
                    loss_3 = criterion(fused_emb, img_emb)  

                    total_batch_loss = loss_1 + loss_2 + loss_3 
                    
                total_val_loss += total_batch_loss.item()
                loop_val.set_postfix(loss=total_batch_loss.item())
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
 
        if avg_val_loss < best_val_loss:
            print(f"\nVal Loss cải thiện từ {best_val_loss:.4f} xuống {avg_val_loss:.4f}.")
            print(f"Đang lưu model tốt nhất...")
            best_val_loss = avg_val_loss

            torch.save(item_tower.state_dict(), os.path.join(args.save_dir, 'item_tower.pth'))
        else:
            print(f"\n(Val Loss không cải thiện so với {best_val_loss:.4f})")
            
        print("-" * 50) 

    print("--- Huấn luyện Hoàn tất ---")

    history_data = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history
    }
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_data, f)
        
    print(f"Model tốt nhất đạt Val Loss = {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Item Tower cho Recommender System")

    parser.add_argument("--data_dir", type=str, required=True, help="Đường dẫn đến thư mục chứa dữ liệu Phase 1")
    parser.add_argument("--save_dir", type=str, required=True, help="Đường dẫn lưu Model Checkpoints")

    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=100, help="Số lượng Epochs huấn luyện")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning Rate")
    parser.add_argument("--temperature", type=float, default=0.07, help="Nhiệt độ cho Contrastive Loss")

    args = parser.parse_args()

    training(args)