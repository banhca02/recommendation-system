import os
import pandas as pd
import torch
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import timm
from torchvision import transforms

def download_single_image(url, local_filepath, item_id):
    if not os.path.exists(local_filepath):
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(local_filepath)
    return item_id, local_filepath

def extract_vit_features(file_path, output_path, image_save_dir="./downloaded_images", model_name='vit_base_patch16_224', batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Bắt đầu tải ảnh về thư mục: {image_save_dir}...")
    os.makedirs(image_save_dir, exist_ok=True)
    df = pd.read_pickle(file_path)
    
    download_tasks = []
    item_local_images = {}
    
    for index, row in df.iterrows():
        item_id = str(row['item_id'])
        img_urls = row['img']
        
        if not img_urls or not isinstance(img_urls, list) or len(img_urls) == 0:
            continue
            
        item_local_images[item_id] = [] 
        
        for i, url in enumerate(img_urls):
            local_filename = f"{item_id}_{i}.jpg"
            local_filepath = os.path.join(image_save_dir, local_filename)
            download_tasks.append((url, local_filepath, item_id))
            
    print(f"Tổng số ảnh cần quét/tải: {len(download_tasks)} ảnh. Đang tiến hành tải song song...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(download_single_image, task[0], task[1], task[2]): task 
            for task in download_tasks
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Tiến độ tải Batch"):
            try:
                res_item_id, res_filepath = future.result()
                item_local_images[res_item_id].append(res_filepath)
            except Exception as e:
                print(e)
                continue

    print(f"\nBắt đầu trích xuất đặc trưng ảnh bằng {model_name} trên {device}...")

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.to(device)
    model.eval()

    VIT_OUT_DIM = model.num_features 

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    item_image_tensors = {}

    with torch.no_grad():
        all_image_tasks = []
        for item_id, local_paths in item_local_images.items():
            for img_path in local_paths:
                all_image_tasks.append((item_id, img_path))

        for i in tqdm(range(0, len(all_image_tasks), batch_size), desc="Trích xuất Vector (Batched)"):
            batch_tasks = all_image_tasks[i : i + batch_size]
            
            valid_tensors = []
            valid_item_ids = []

            for item_id, img_path in batch_tasks:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img)
                    valid_tensors.append(img_tensor)
                    valid_item_ids.append(item_id)
                except Exception as e:
                    continue

            if len(valid_tensors) > 0:
                batch_tensor = torch.stack(valid_tensors).to(device)
                feats = model(batch_tensor).cpu() 
                
                for j, item_id in enumerate(valid_item_ids):
                    if item_id not in item_image_tensors:
                        item_image_tensors[item_id] = []
                    item_image_tensors[item_id].append(feats[j])

        for item_id in item_image_tensors:
            item_image_tensors[item_id] = torch.stack(item_image_tensors[item_id])

    print(f"\nĐã trích xuất thành công {len(item_image_tensors)} sản phẩm có ảnh.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(item_image_tensors, output_path)
    print(f"Đã lưu vector vào: {output_path}")
