import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from phase1_item_tower.models import TextEncoder

def extract_review_features(data_path, checkpoint_path, output_path, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Đang tải Tokenizer và Base Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name).to(device)
    base_model.eval()

    item_tower_checkpoint = os.path.join(checkpoint_path, 'item_tower.pth')
    full_state_dict = torch.load(item_tower_checkpoint, map_location=device)

    text_encoder_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('text_encoder.'):
            new_key = key.replace('text_encoder.', '', 1) 
            text_encoder_state_dict[new_key] = value

    custom_encoder = TextEncoder(in_dim=768, out_dim=256).to(device) 
    custom_encoder.load_state_dict(text_encoder_state_dict)
    custom_encoder.eval()

    print(f"Đang đọc dữ liệu từ: {data_path}...")
    df = pd.read_pickle(data_path)
    
    unique_texts = set()

    text_columns = [
        'review_text_seq', 'review_title_seq'
    ]
    
    print("Đang gom nhóm và lọc các đoạn văn bản trùng lặp...")
    for col in text_columns:
        if col in df.columns:
            for seq in df[col]:
                if isinstance(seq, (list, tuple)):
                    for text_item in seq:
                        if isinstance(text_item, str):
                            clean_text = text_item.strip()
                            if clean_text != "" and clean_text.lower() != "unknown":
                                unique_texts.add(clean_text)
                
                elif isinstance(seq, str):
                    clean_text = seq.strip()
                    if clean_text != "" and clean_text.lower() != "unknown":
                        unique_texts.add(clean_text)

    valid_texts_list = list(unique_texts)
    print(f"Đã chắt lọc được {len(valid_texts_list)} đoạn văn bản/tiêu đề duy nhất để xử lý.")

    text_to_tensor_dict = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(valid_texts_list), batch_size), desc="Extracting Texts"):
            batch_texts = valid_texts_list[i : i + batch_size]
            batch_inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            ).to(device)

            outputs = base_model(**batch_inputs)
            last_hidden = outputs.last_hidden_state
            attention_mask = batch_inputs['attention_mask']

            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())
            sum_embs = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9)
            base_embs = sum_embs / sum_mask  

            with torch.amp.autocast('cuda'):
                final_embs = custom_encoder(base_embs)

            final_embs_cpu = final_embs.cpu()
            for j, text_string in enumerate(batch_texts):
                text_to_tensor_dict[text_string] = final_embs_cpu[j]

    torch.save(text_to_tensor_dict, output_path)
    print(f"\nĐã lưu thành công từ điển mapping Text -> Vector vào:\n {output_path}")