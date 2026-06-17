from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import re

OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "papers.csv"

DATASET_NAME = "sentence-transformers/s2orc"
SUBSET_NAME = "title-abstract-pair"

N_PAPERS = 20000
MIN_ABSTRACT_WORDS = 80
SEED = 42


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("[*] Loading S2ORC by streaming...")
    print("[*] This will NOT download the full dataset.")

    ds = load_dataset(
        DATASET_NAME,
        SUBSET_NAME,
        split="train",
        streaming=True
    )

    # Tránh lấy tuần tự quá thiên lệch
    ds = ds.shuffle(buffer_size=10000, seed=SEED)

    rows = []
    seen_titles = set()

    pbar = tqdm(total=N_PAPERS)

    for ex in ds:
        title = clean_text(ex.get("title", ""))
        abstract = clean_text(ex.get("abstract", ""))

        if not title or not abstract:
            continue

        if len(abstract.split()) < MIN_ABSTRACT_WORDS:
            continue

        key = title.lower()
        if key in seen_titles:
            continue

        seen_titles.add(key)

        rows.append({
            "paper_id": f"s2orc_{len(rows)}",
            "title": title,
            "abstract": abstract
        })

        pbar.update(1)

        if len(rows) >= N_PAPERS:
            break

    pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved {len(df)} papers to: {OUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()