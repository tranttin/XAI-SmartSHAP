import os
import pandas as pd
import re


class DataLoader:
    def __init__(self, base_path="."):
        """
        base_path là thư mục gốc của project XAI.
        Ví dụ:
        - Local:  C:\\Users\\ADMIN\\PycharmProjects\\XAI
        - Colab:  /content/drive/MyDrive/Colab/XAI
        """
        self.base_path = base_path

    def load_dataset(self, name):
        """
        Tải dữ liệu từ data/{name}/papers.csv
        Ví dụ:
        - data/arxiv/papers.csv
        - data/hartzbyte/papers.csv
        - data/s2orc/papers.csv
        """
        path = os.path.join(self.base_path, "data", name, "papers.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path}")

        print(f"[*] Đang đọc file: {path}")
        df = pd.read_csv(path)

        required_cols = ["abstract"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"File {path} thiếu cột bắt buộc: {col}. "
                    f"Các cột hiện có: {list(df.columns)}"
                )

        # Chuẩn hóa nhẹ
        df = df.dropna(subset=["abstract"]).copy()
        df["abstract"] = df["abstract"].astype(str).str.strip()
        df = df[df["abstract"].str.len() > 0]

        # Nếu thiếu title thì tạo title rỗng để code sau không gãy
        if "title" not in df.columns:
            df["title"] = ""

        # Nếu thiếu paper_id thì tự sinh
        if "paper_id" not in df.columns:
            df["paper_id"] = [f"{name}_{i}" for i in range(len(df))]

        print(f"[*] Loaded {len(df)} papers from dataset: {name}")
        return df.reset_index(drop=True)

    @staticmethod
    def preprocess_abstract(text, min_len=5):
        """
        Tách abstract thành danh sách câu.
        Bản này đủ dùng cho thí nghiệm sentence-level.
        """
        if not isinstance(text, str):
            return []

        text = re.sub(r"\s+", " ", text).strip()

        # Tách câu dựa trên . ! ?
        sentences = re.split(r"(?<=[.!?])\s+", text)

        return [
            s.strip()
            for s in sentences
            if len(s.strip()) > min_len
        ]