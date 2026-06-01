import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def extract_bert_features(file_path, output_path, model_name='sentence-transformers/all-mpnet-base-v2', batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu trích xuất Text Embedding bằng {model_name} trên {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    df = pd.read_pickle(file_path)

    item_ids = df['item_id'].astype(str).tolist()
    texts = df['text'].tolist()

    texts = [t if isinstance(t, str) and t.strip() != "" else "unknown" for t in texts]

    item_text_tensors = {}

    print(f"Đang xử lý {len(texts)} văn bản...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Batches"):
            batch_ids = item_ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]

            batch_inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            ).to(device)

            outputs = model(**batch_inputs)
            last_hidden = outputs.last_hidden_state
            attention_mask = batch_inputs['attention_mask']

            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())
            sum_embs = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9)
            embs = sum_embs / sum_mask 

            embs_cpu = embs.cpu()
            for j, item_id in enumerate(batch_ids):
                item_text_tensors[item_id] = embs_cpu[j]

    torch.save(item_text_tensors, output_path)
    print(f"Đã lưu vector văn bản gốc (Hidden State) vào: {output_path}")
