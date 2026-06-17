# XAI-RecSys: Explainable Scientific Paper Recommender Systems Evaluation

Dự án này cung cấp một framework đánh giá định lượng và toàn diện các phương pháp giải thích (Explainable AI - XAI) ở cấp độ câu (sentence-level) áp dụng cho hệ thống gợi ý bài báo khoa học. Hệ thống lõi sử dụng mô hình Bi-Encoder và Cross-Encoder kết hợp cùng các biến thể tối ưu của thuật toán gán mức độ đóng góp dựa trên lý thuyết trò chơi (Shapley/Banzhaf Values).

---

## 📌 Các Tính Năng Chính

* **Pipeline Xử Lý Dữ Liệu Tự Động:** Trích xuất, lọc và lấy mẫu ngẫu nhiên dữ liệu lớn từ kho lưu trữ arXiv (lọc riêng lĩnh vực Computer Science) mà không gây tràn bộ nhớ (RAM).
* **Làm Giàu Dữ Liệu (Data Enrichment):** Sử dụng LLM tích hợp (OpenAI API) để chuẩn hóa và tinh gọn cấu trúc câu trong các đoạn tóm tắt (Abstract) nhưng vẫn giữ nguyên ngữ nghĩa kỹ thuật.
* **Framework Gợi Ý Kết Hợp (Dual-Encoder):** Tích hợp cả cơ chế tương đồng Cosine qua Bi-encoder (`BAAI/bge-small-en`) và tái xếp hạng độ tương quan sâu qua Cross-encoder (`BAAI/bge-reranker-base`).
* **Đa Dạng Phương Pháp Giải Thích (Explainers):** Hỗ trợ từ các thuật toán cổ điển (KernelSHAP, LIME, RISE, LOO) cho đến các biến thể nghiên cứu tối ưu nâng cao (`SmartShap`, `SmartShap-Sentence`, `AdaptiveSmartShapLOO`, `ScreenerSmartShap`).
* **Hệ Thống Đánh Giá Định Lượng Toàn Diện:** Đo lường tự động các chỉ số khắt khe về độ tin cậy (Faithfulness bao gồm Comprehensiveness & Sufficiency), độ tương đồng phân hạng (Rank Agreement), sai số trực tiếp (MAE, RMSE) và độ ổn định thực nghiệm (Stability).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
XAI/
├── data/
│   └── arxiv/
│       ├── papers.csv              # Dữ liệu bài báo thô sau trích xuất
│       └── papers_enriched.csv     # Dữ liệu đã được làm giàu/phân rã câu qua LLM
├── runs/                           # Thư mục lưu vết log kết quả thực nghiệm
├── crawl.py                        # Script trích xuất cấu trúc dữ liệu từ file JSON gốc
├── chunk.py                        # Script phân rã câu và gọi OpenAI API làm giàu dữ liệu
├── loader.py                       # Bộ nạp dữ liệu và tiền xử lý chuỗi văn bản
├── models.py                       # Kiến trúc Recommender Model và các thuật toán XAI Explainer
├── xai_evaluator.py                # Module tính toán các chỉ số đánh giá định lượng XAI
├── main.py                         # File thực thi chạy thực nghiệm cấu hình tổng thể
└── README.md                       # Tài liệu hướng dẫn dự án
