import torch
from transformers import ViTModel, ViTImageProcessor
from tqdm import tqdm
import os
import json
import requests
from PIL import Image
from io import BytesIO
import gzip

def fetch_image(url):
    """Downloads an image from a URL and converts it to RGB."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None

def read_jsonl(file_path):
    """Smart reader that handles both raw .jsonl and compressed .jsonl.gz files."""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            while True:
                try:
                    line = f.readline()
                    if not line: 
                        break
                    yield line
                except EOFError:
                    print(f"\n[WARNING] Hit a broken line! {file_path} is an incomplete download.")
                    print("Salvaging the data we successfully read so far...")
                    break 
    except gzip.BadGzipFile:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line

def extract_and_save_features():
    print("Loading ViT Model and Processor...")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # --- UPDATED PATH ---
    mapping_path = 'dataset/amazon_beauty/id2asin.json'
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Cannot find {mapping_path}. Run train.py first to generate the dictionary!")
        
    with open(mapping_path, 'r') as f:
        id2asin = {int(k): v for k, v in json.load(f).items()}
    
    asin2id = {v: k for k, v in id2asin.items()}
    print(f"Total unique items to process: {len(id2asin)}")

    # --- UPDATED PATH (Added .gz) ---
    meta_file_path = 'meta_Beauty_and_Personal_Care.jsonl.gz' 
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f"Cannot find {meta_file_path}. Make sure it is in the same folder as this script!")
    
    item_features = {}
    zero_tensor = torch.zeros(768) 
    
    print("Scanning local JSONL file and extracting features...")
    
    for line in tqdm(read_jsonl(meta_file_path), desc="Processing Images"):
        try:
            data = json.loads(line)
            asin = data.get('parent_asin')
            
            if asin in asin2id:
                integer_id = asin2id[asin]
                if integer_id not in item_features:
                    images_list = data.get('images', [])
                    if images_list and len(images_list) > 0:
                        img_data = images_list[0]
                        img_url = img_data.get('hi_res') or img_data.get('large')
                        
                        if img_url:
                            img = fetch_image(img_url)
                            if img:
                                inputs = processor(images=img, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    outputs = model(**inputs)
                                    cls_feature = outputs.last_hidden_state[:, 0, :].cpu().squeeze(0)
                                item_features[integer_id] = cls_feature
        except json.JSONDecodeError:
            continue

    print("\nFilling in missing/broken images with zero-tensors...")
    for integer_id in id2asin.keys():
        if integer_id not in item_features:
            item_features[integer_id] = zero_tensor

    item_features[0] = zero_tensor 

    vocab_size = max(id2asin.keys()) + 1
    feature_matrix = torch.zeros((vocab_size, 768))
    
    for idx, tensor in item_features.items():
        feature_matrix[idx] = tensor

    # --- UPDATED PATH ---
    save_path = 'dataset/amazon_beauty/item_image_features.pt'
    torch.save(feature_matrix, save_path)
    print(f"\nSuccessfully saved feature matrix of shape {feature_matrix.shape} to {save_path}")

if __name__ == "__main__":
    extract_and_save_features()
