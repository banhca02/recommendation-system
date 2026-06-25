import torch
from torch.utils.data import Dataset
import pandas as pd
import random

class UserHistoryDataset(Dataset):
    def __init__(
        self, 
        user_df, 
        review_dict_path, 
        item_dict_path,      
        max_seq_len=50,      
        pad_value=0,         
        mask_prob=0.1,       
        mode='train', 
        all_item_ids=None
    ):
        self.max_len = max_seq_len
        self.pad_value = pad_value
        self.mask_prob = mask_prob
        self.mode = mode
        self.all_item_ids = all_item_ids or []
        self.samples = []
        
        print(f"Đang tải từ điển Review Embeddings từ: {review_dict_path}...")
        self.review_dict = torch.load(review_dict_path, map_location='cpu')
        self.rev_emb_dim = next(iter(self.review_dict.values())).shape[0] if len(self.review_dict) > 0 else 256 
        self.zero_rev_tensor = torch.zeros(self.rev_emb_dim, dtype=torch.float)

        print(f"Đang tải từ điển Item Embeddings từ: {item_dict_path}...")
        self.item_dict = torch.load(item_dict_path, map_location='cpu')
        self.item_emb_dim = next(iter(self.item_dict.values())).shape[0] if len(self.item_dict) > 0 else 256 
        self.zero_item_tensor = torch.zeros(self.item_emb_dim, dtype=torch.float)
        
        if mode in ['train', 'val']:
            print(f"Đang xử lý dữ liệu cho tập: {mode.upper()} (Max Len: {self.max_len})...")
        else:
            print(f"Đang xử lý dữ liệu để inference (Max Len: {self.max_len})...")    
        
        for _, row in user_df.iterrows():
            if self.mode == 'train':
                item_indices = row['item_indices_seq']
                review_texts = row['review_text_seq'] 
                ratings = row['rating_seq'] 

                user_explicit_negatives = [
                    item_indices[j] for j in range(len(item_indices)) 
                    if ratings[j] < 3
                ]
                
                for i in range(1, len(item_indices)):
                    if ratings[i-1] >= 3:
                        input_indices = item_indices[:i]
                        input_reviews = review_texts[:i] 
                        target_item = item_indices[i] 
                        
                        self.samples.append((
                            input_indices, input_reviews, target_item, user_explicit_negatives
                        ))
            else:
                input_indices = row['input_item_indices_seq']
                input_reviews = row['input_review_text_seq'] 
                input_ratings = row['input_rating_seq']
                target_item = row['target_item_indices_seq'] 
                target_rating = row['target_rating_seq']

                if target_rating >= 3:
                    user_explicit_negatives = []
                    if self.mode == 'val':
                        user_explicit_negatives = [
                            input_indices[j] for j in range(len(input_indices)) 
                            if input_ratings[j] < 3
                        ]
                    self.samples.append((
                        input_indices, input_reviews, target_item, user_explicit_negatives
                    ))
                    
        if self.mode == 'train':
            print(f"Đã sinh ra {len(self.samples)} mẫu trượt cửa sổ. Đang xáo trộn...")
            random.shuffle(self.samples)
        print(f"Đã chuẩn bị xong {len(self.samples)} mẫu.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_indices, raw_reviews_text, target_item, user_explicit_negatives = self.samples[idx]

        input_indices = [int(x) for x in raw_indices]
        input_reviews_text = list(raw_reviews_text)

        if self.mode == 'train':
            seq_len_current = len(input_indices)
            for i in range(seq_len_current):
                if i != seq_len_current - 1 and random.random() < self.mask_prob:
                    input_indices[i] = self.pad_value
                    input_reviews_text[i] = "MASKED" 
        
        seq_len = len(input_indices)
        if seq_len > self.max_len:
            start_idx = seq_len - self.max_len
            input_indices = input_indices[start_idx:]
            input_reviews_text = input_reviews_text[start_idx:]
            seq_len = self.max_len

        padding_len = self.max_len - seq_len

        padded_indices = [self.pad_value] * padding_len + input_indices

        review_tensors = []
        for _ in range(padding_len):
            review_tensors.append(self.zero_rev_tensor)
            
        for text in input_reviews_text:
            if text in self.review_dict:
                review_tensors.append(self.review_dict[text])
            else:
                review_tensors.append(self.zero_rev_tensor)
                
        padded_reviews_matrix = torch.stack(review_tensors)
        
        item_tensors = []
        for _ in range(padding_len):
            item_tensors.append(self.zero_item_tensor)
            
        for item_id in input_indices:
            if item_id in self.item_dict:
                item_tensors.append(self.item_dict[item_id])
            elif str(item_id) in self.item_dict:
                item_tensors.append(self.item_dict[str(item_id)])
            else:
                item_tensors.append(self.zero_item_tensor)
                
        padded_items_matrix = torch.stack(item_tensors)

        transformer_mask = torch.tensor([True] * padding_len + [False] * seq_len, dtype=torch.bool)
        
        user_data_dict = {
            "item_indices": torch.tensor(padded_indices, dtype=torch.long),
            "review_embs": padded_reviews_matrix, 
            "item_embs": padded_items_matrix,      
            "transformer_mask": transformer_mask 
        }

        if target_item in self.item_dict:
            target_tensor = self.item_dict[target_item]
        elif str(target_item) in self.item_dict:
            target_tensor = self.item_dict[str(target_item)]
        else:
            target_tensor = self.zero_item_tensor

        if self.mode in ['train', 'val']:
            if len(user_explicit_negatives) > 0:
                negative_item = random.choice(user_explicit_negatives)
            else:
                negative_item = random.choice(self.all_item_ids)
                while (negative_item in input_indices) or (negative_item == target_item):
                    negative_item = random.choice(self.all_item_ids)

            if negative_item in self.item_dict:
                negative_tensor = self.item_dict[negative_item]
            elif str(negative_item) in self.item_dict:
                negative_tensor = self.item_dict[str(negative_item)]
            else:
                negative_tensor = self.zero_item_tensor
                    
            return user_data_dict, target_tensor, negative_tensor
        else: 
            return user_data_dict, target_tensor

def collate_fn_user(batch):
    user_dicts = [b[0] for b in batch]
    targets = torch.stack([b[1] for b in batch])
    
    batched_user_dict = {
        "item_indices": torch.stack([d["item_indices"] for d in user_dicts]),
        "review_embs": torch.stack([d["review_embs"] for d in user_dicts]),
        "item_embs": torch.stack([d["item_embs"] for d in user_dicts]),
        "transformer_mask": torch.stack([d["transformer_mask"] for d in user_dicts])
    }
    
    if len(batch[0]) == 3: 
        negatives = torch.stack([b[2] for b in batch])
        return batched_user_dict, targets, negatives
    else: 
        return batched_user_dict, targets