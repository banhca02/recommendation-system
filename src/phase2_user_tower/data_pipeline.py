import json
import pandas as pd
import os
import argparse
from tqdm import tqdm

def parse_standard_json(file_path):
    print(f"Đang đọc và phân tích file JSON: {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    
    for user_id, interactions in tqdm(data.items(), desc="Đóng gói chuỗi (Sequence)"):
        if not interactions or not isinstance(interactions, list):
            continue

        user_record = {'user_id': user_id}
        item_seq = []
        
        all_keys = set()
        for inter in interactions:
            all_keys.update(inter.keys())

        if 'item' in all_keys:
            all_keys.remove('item')

        aux_seqs = {k: [] for k in all_keys}

        for inter in interactions:
            if 'item' not in inter:
                continue            
            item_seq.append(inter['item'])
            for k in all_keys:
                val = inter.get(k)
                if val is None or val == "":
                    if 'text' in str(k).lower() or 'title' in str(k).lower():
                        val = ""
                    else:
                        val = 5.0 
                
                aux_seqs[k].append(val)

        if len(item_seq) > 0:
            user_record['item_indices_seq'] = item_seq
   
            for k, v_list in aux_seqs.items():
                clean_key = str(k).strip().replace(" ", "_")
                user_record[f"{clean_key}_seq"] = v_list
                
            records.append(user_record)
            
    df = pd.DataFrame(records)
    print(f"Đã tạo thành công DataFrame với {len(df)} người dùng.")
    print(f"Các cột thu được: {list(df.columns)}")
    return df

def create_leave_one_out_splits(df, output_dir, item_col='item_indices_seq'):
    aux_cols = [c for c in df.columns if c.endswith('_seq') and c != item_col]
    
    train_records = []
    val_records = []
    test_records = []
    
    print("\nĐang tiến hành cắt Leave-One-Out...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if len(row[item_col]) < 5:
            continue
            
        base_info = {'user_id': row['user_id']}
        
        # 1. TẬP TRAIN (0 -> n-2)
        train_dict = base_info.copy()
        train_dict[item_col] = row[item_col][:-2]
        for col in aux_cols:
            train_dict[col] = row[col][:-2]
        train_records.append(train_dict)
        
        # 2. TẬP VALIDATION (Input: 0 -> n-2 | Target: n-1)
        val_dict = base_info.copy()
        val_dict[f'input_{item_col}'] = row[item_col][:-2]
        val_dict[f'target_{item_col}'] = row[item_col][-2]
        for col in aux_cols:
            val_dict[f'input_{col}'] = row[col][:-2]
            val_dict[f'target_{col}'] = row[col][-2]
        val_records.append(val_dict)
        
        # 3. TẬP TEST (Input: 0 -> n-1 | Target: n)
        test_dict = base_info.copy()
        test_dict[f'input_{item_col}'] = row[item_col][:-1]
        test_dict[f'target_{item_col}'] = row[item_col][-1]
        for col in aux_cols:
            test_dict[f'input_{col}'] = row[col][:-1]
            test_dict[f'target_{col}'] = row[col][-1]
        test_records.append(test_dict)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(train_records).to_pickle(os.path.join(output_dir, 'user_train.pkl'))
    pd.DataFrame(val_records).to_pickle(os.path.join(output_dir, 'user_val.pkl'))
    pd.DataFrame(test_records).to_pickle(os.path.join(output_dir, 'user_test.pkl'))
    
    print(f"Hoàn tất! Các file Train/Val/Test đã chia được lưu tại: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2 User Tower Data Preprocessing")
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn tới file JSON lịch sử tương tác của User")
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục lưu các file .pkl (Train/Val/Test)")
    
    args = parser.parse_args()

    df = parse_standard_json(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    total_path = os.path.join(args.output_dir, 'user_all.pkl')
    df.to_pickle(total_path)
    print(f"\nĐã lưu tập dữ liệu TỔNG (chưa chia cắt) vào: {total_path}")

    create_leave_one_out_splits(df, args.output_dir)