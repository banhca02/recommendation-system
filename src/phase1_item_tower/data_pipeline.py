import json
import pandas as pd
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def parse_item_metadata_json(file_path):
    print(f"Đang đọc file JSON thông tin sản phẩm: {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    
    for item_id, features in tqdm(data.items(), desc="Đang trích xuất Text & Image"):
        record = {'item_id': str(item_id)}

        img_val = features.get('img', [])
        if img_val == "" or img_val is None:
            img_val = []
        record['img'] = img_val
        
        text_val = features.get('text', "")
        if text_val is None:
            text_val = ""
        record['text'] = str(text_val)
            
        records.append(record)
        
    df = pd.DataFrame(records)
    df['item_idx'] = range(1, len(df) + 1)

    cols = ['item_idx', 'item_id', 'text', 'img']
    df = df[cols]
    
    print(f"\nĐã tải thông tin cho {len(df)} sản phẩm.")
    print(f"Các trường dữ liệu thu được: {list(df.columns)}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="File JSON Item Metadata")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục lưu data Pha 1")

    parser.add_argument("--val_size", type=float, default=0.1, help="Tỷ lệ tập Validation (mặc định 0.1 ~ 10%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed để cố định kết quả chia tách")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df_clean = parse_item_metadata_json(args.input)

    print(f"\nĐang chia tập dữ liệu với tỷ lệ Validation là {args.val_size} (Seed: {args.seed})...")
    df_train, df_val = train_test_split(df_clean, test_size=args.val_size, random_state=args.seed)

    full_path = os.path.join(args.output_dir, 'item_metadata_encoded.pkl')
    train_path = os.path.join(args.output_dir, 'item_metadata_train.pkl')
    val_path = os.path.join(args.output_dir, 'item_metadata_val.pkl')

    df_clean.to_pickle(full_path)
    df_train.to_pickle(train_path)
    df_val.to_pickle(val_path)
    
    print(f"\nHoàn tất tiền xử lý Pha 1. Dữ liệu đã được chia và lưu tại thư mục: {args.output_dir}")
    print(f"File Train (Dùng huấn luyện): {os.path.basename(train_path)} ({len(df_train)} items)")
    print(f"File Val (Dùng đánh giá loss): {os.path.basename(val_path)} ({len(df_val)} items)")
    print(f"File Tổng (Dùng inference cho user tower): {os.path.basename(full_path)} ({len(df_clean)} items)")