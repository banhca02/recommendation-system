import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class ItemMetadataDataset(Dataset):
    def __init__(self, data_dir, filename='item_metadata_encoded.pkl', image_pt_path=None, text_pt_path=None):
        self.df = pd.read_pickle(os.path.join(data_dir, filename))
        
        self.image_tensors = {}
        if image_pt_path and os.path.exists(image_pt_path):
            print(f"Đang nạp Image Tensors từ {image_pt_path}...")
            self.image_tensors = torch.load(image_pt_path, map_location='cpu')
            
        self.text_tensors = {}
        if text_pt_path and os.path.exists(text_pt_path):
            print(f"Đang nạp Text Tensors từ {text_pt_path}...")
            self.text_tensors = torch.load(text_pt_path, map_location='cpu')
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        item_idx = row['item_idx']
        item_id = str(row['item_id']) 


        text_feat = self.text_tensors.get(item_id, None)
        img_feats = self.image_tensors.get(item_id, None)
        
        return {
            'item_idx': item_idx,
            'text_feat': text_feat,
            'img_feats': img_feats
        }

def collate_fn_item(batch):
    item_indices = torch.tensor([b['item_idx'] for b in batch], dtype=torch.long)

    text_feats_batch = []
    valid_text_dim = None
    text_feats_tensor = None

    for b in batch:
        if b['text_feat'] is not None:
            valid_text_dim = b['text_feat'].shape[0]
            break

    if valid_text_dim is not None:
        for b in batch:
            if b['text_feat'] is not None:
                text_feats_batch.append(b['text_feat'])
            else:
                text_feats_batch.append(torch.zeros(valid_text_dim))
        text_feats_tensor = torch.stack(text_feats_batch) 
    else:
        text_feats_tensor = None
    img_feats_list = [b['img_feats'] for b in batch]

    return {
        'item_indices': item_indices,
        'text_feats': text_feats_tensor,
        'img_feats': img_feats_list
    }