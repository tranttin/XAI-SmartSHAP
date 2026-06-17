import json
import pandas as pd
import os
import random

def extract_arxiv_from_local(json_path, output_csv, limit=5000, sampling_rate=0.05):
    """
    Đọc file JSON 5.2GB trực tiếp từ Drive theo từng dòng.
    Lấy mẫu ngẫu nhiên để bộ dữ liệu trải dài từ cũ đến mới.
    """
    if not os.path.exists(json_path):
        print(f"[!] Không tìm thấy file gốc tại: {json_path}")
        print("[*] Hãy đảm bảo anh đã upload file 'arxiv-metadata-oai-snapshot.json' vào đúng thư mục.")
        return

    print(f"[*] Bắt đầu trích xuất từ file local: {json_path}")
    extracted_data = []
    count = 0
    total_scanned = 0

    # Đọc file line-by-line để tiết kiệm RAM
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            
            total_scanned += 1
            
            # Lấy mẫu ngẫu nhiên để tránh chỉ lấy các bài báo quá cũ ở đầu file
            if random.random() > sampling_rate:
                continue

            try:
                data = json.loads(line)
                categories = data.get('categories', '')
                
                # Lọc: Chỉ lấy lĩnh vực Computer Science (cs.)
                if 'cs.' in categories:
                    extracted_data.append({
                        'title': data.get('title', '').replace('\n', ' ').strip(),
                        'abstract': data.get('abstract', '').replace('\n', ' ').strip(),
                        'categories': categories,
                        'update_date': data.get('update_date', '')
                    })
                    count += 1
                    
                    if count % 500 == 0:
                        print(f"    - Đã lấy được {count} bài (Đã quét {total_scanned} dòng)...")
            except Exception as e:
                continue

    # Chuyển sang DataFrame và xáo trộn ngẫu nhiên một lần nữa
    df = pd.DataFrame(extracted_data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Lưu kết quả CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n[OK] Hoàn thành! Đã trích xuất {len(df)} bài báo.")
    print(f"[*] File kết quả: {output_csv}")
    print(f"[*] Kích thước: {os.path.getsize(output_csv) / 1024**2:.2f} MB")

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Anh hãy kiểm tra xem tên file JSON trên Drive có khớp không
JSON_INPUT_PATH = "/content/drive/MyDrive/Colab/XAI/arxiv-metadata-oai-snapshot.json"
CSV_OUTPUT_PATH = "/content/drive/MyDrive/Colab/XAI/data/arxiv/papers.csv"

# Chạy trích xuất
extract_arxiv_from_local(JSON_INPUT_PATH, CSV_OUTPUT_PATH, limit=5000, sampling_rate=0.05)