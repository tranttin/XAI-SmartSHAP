import os
import re
import json
import time
import pandas as pd
from typing import List, Optional

# =========================
# CẤU HÌNH
# =========================
INPUT_CSV = "data/arxiv/papers.csv"
OUTPUT_CSV = "data/arxiv/papers_enriched.csv"

MODE = "openai"   # mock | openai

# DÁN KEY THẲNG VÀO ĐÂY
API_KEY = "sk-proj-ULOcQRno3W4D7yXP_Nm2uQwF4q3PlgbsBUGgjoA"

OPENAI_MODEL = "gpt-4.1-mini"
LIMIT = None
SLEEP_SECONDS = 0.2
SAVE_EVERY = 100
MAX_RETRIES = 3

# =========================
# TIỆN ÍCH TEXT
# =========================
def normalize_whitespace(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_sentence_split(text: str) -> List[str]:
    """
    Tách câu đơn giản cho abstract tiếng Anh.
    Không hoàn hảo nhưng đủ ổn cho pipeline ban đầu.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    protected = {
        "e.g.": "EGTOKEN",
        "i.e.": "IETOKEN",
        "et al.": "ETALTOKEN",
        "Fig.": "FIGTOKEN",
        "Eq.": "EQTOKEN",
        "Dr.": "DRTOKEN",
        "Mr.": "MRTOKEN",
        "Ms.": "MSTOKEN",
        "Prof.": "PROFTOKEN",
    }

    for k, v in protected.items():
        text = text.replace(k, v)

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z(\[])', text)
    parts = [p.strip() for p in parts if p.strip()]

    restored = []
    for p in parts:
        for k, v in protected.items():
            p = p.replace(v, k)
        restored.append(normalize_whitespace(p))

    return restored


def mock_refine_sentence(sentence: str) -> str:
    sentence = normalize_whitespace(sentence)
    sentence = sentence.replace(" ,", ",").replace(" .", ".")
    return sentence


# =========================
# OPENAI API
# =========================
def openai_refine_sentence(
    sentence: str,
    model: str = OPENAI_MODEL,
    max_retries: int = MAX_RETRIES
) -> str:
    """
    Gọi OpenAI API để viết lại câu theo hướng:
    - giữ nguyên nghĩa kỹ thuật
    - ngắn gọn hơn nếu có thể
    - không thêm thông tin mới
    - trả đúng 1 câu
    """
    if not API_KEY or API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        raise ValueError("Chưa dán API key thật vào biến API_KEY.")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Thiếu thư viện openai. Cài bằng: pip install --upgrade openai") from e

    client = OpenAI(api_key=API_KEY)

    prompt = (
        "Rewrite the following scientific sentence into a cleaner and slightly more concise form "
        "while preserving its technical meaning exactly. "
        "Do not add new information. Return only one rewritten sentence.\n\n"
        f"Sentence: {sentence}"
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
            )

            text = getattr(resp, "output_text", None)
            text = text.strip() if text else ""
            return normalize_whitespace(text) if text else sentence

        except Exception as e:
            last_error = e
            wait_time = min(2 ** attempt, 5)
            print(f"[!] API lỗi lần {attempt + 1}/{max_retries}: {e}")
            time.sleep(wait_time)

    raise RuntimeError(f"OpenAI refine failed after {max_retries} retries: {last_error}")


def refine_sentence(sentence: str, mode: str = MODE) -> str:
    if mode == "mock":
        return mock_refine_sentence(sentence)
    if mode == "openai":
        return openai_refine_sentence(sentence)
    raise ValueError(f"Unsupported mode: {mode}")


# =========================
# XỬ LÝ MỖI ABSTRACT
# =========================
def process_abstract(abstract: str, mode: str = MODE) -> dict:
    abstract = normalize_whitespace(abstract)
    sentences = simple_sentence_split(abstract)

    refined_sentences = []
    statuses = []

    for sent in sentences:
        try:
            refined = refine_sentence(sent, mode=mode)
            status = "ok"
        except Exception as e:
            print(f"[!] Lỗi refine sentence, fallback về câu gốc: {e}")
            refined = sent
            status = "fallback"

        refined_sentences.append(refined)
        statuses.append(status)

        if mode == "openai" and SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)

    return {
        "sentences_json": json.dumps(sentences, ensure_ascii=False),
        "sentences_refined_json": json.dumps(refined_sentences, ensure_ascii=False),
        "sentence_status_json": json.dumps(statuses, ensure_ascii=False),
        "n_sentences": len(sentences),
    }


# =========================
# MAIN
# =========================
def enrich_papers_csv(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
    mode: str = MODE,
    limit: Optional[int] = LIMIT,
) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Không tìm thấy file input: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = {"title", "abstract", "categories", "update_date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    if limit is not None:
        df = df.iloc[:limit].copy()
    else:
        df = df.copy()

    results = []
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total = len(df)
    print(f"[*] Bắt đầu enrich {total} dòng | mode={mode}")
    print(f"[*] Model: {OPENAI_MODEL}")

    for _, row in df.iterrows():
        title = normalize_whitespace(row.get("title", ""))
        abstract = normalize_whitespace(row.get("abstract", ""))
        categories = normalize_whitespace(row.get("categories", ""))
        update_date = normalize_whitespace(row.get("update_date", ""))

        if not abstract:
            enriched = {
                "title": title,
                "abstract": abstract,
                "categories": categories,
                "update_date": update_date,
                "sentences_json": json.dumps([], ensure_ascii=False),
                "sentences_refined_json": json.dumps([], ensure_ascii=False),
                "sentence_status_json": json.dumps([], ensure_ascii=False),
                "n_sentences": 0,
            }
        else:
            processed = process_abstract(abstract, mode=mode)
            enriched = {
                "title": title,
                "abstract": abstract,
                "categories": categories,
                "update_date": update_date,
                **processed,
            }

        results.append(enriched)

        current = len(results)
        if current % 20 == 0 or current == total:
            print(f"    - Đã xử lý {current}/{total}")

        if current % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)
            print(f"    - Đã lưu tạm: {output_csv}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

    print(f"\n[OK] Hoàn thành: {output_csv}")
    print(f"[*] Số dòng: {len(out_df)}")
    print(f"[*] Cột đầu ra: {list(out_df.columns)}")


if __name__ == "__main__":
    enrich_papers_csv()