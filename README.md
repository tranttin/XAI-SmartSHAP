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


## 📊 Tham Số Cấu Hình Hệ Thống (`main.py`)

Anh có thể điều chỉnh linh hoạt các tham số dưới đây khi thực thi pipeline nghiên cứu:

| Tham số | Kiểu dữ liệu | Giá trị mặc định | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `--base_path` | `str` | `None` | Đường dẫn tuyệt đối đến thư mục root của dự án (Local hoặc Google Drive). |
| `--dataset` | `str` | `arxiv` | Tên thư mục chứa dữ liệu trong mục `data/` (ví dụ: `arxiv`, `hartzbyte`, `s2orc`). |
| `--mode` | `str` | `smartshap` | Chọn thuật toán giải thích cần chạy (`smartshap`, `lime`, `loo`, `rise`, `all`,...). |
| `--all_modes` | `str` | `baseline_kernel,smartshap,...` | Danh sách các phương pháp giải thích sẽ chạy song song nếu đặt `--mode all`. |
| `--samples` | `int` | `5` | Số lượng mẫu dữ liệu bài báo được bốc ngẫu nhiên đưa vào quy trình đánh giá. |
| `--ns_smart` | `int` | `200` | Số lượng mẫu liên minh tổ hợp (Coalition) tối đa cho các biến thể SmartSHAP. |
| `--ns_base` | `int` | `1000` | Số lượng mẫu tổ hợp cho mô hình nền tảng lý thuyết chuẩn (Baseline Kernel). |
| `--max_segments`| `int` | `12` | Số lượng câu tối đa được chọn lọc giữ lại trong module `smartshap_sentence`. |
| `--min_sentences`| `int` | `2` | Số lượng câu tối thiểu của abstract để hệ thống chấp nhận đánh giá (tránh skip mẫu quá ngắn). |
| `--query` | `str` | `None` | Câu truy vấn cố định cho mọi mẫu. Nếu bỏ trống, hệ thống tự động lấy tiêu đề bài báo làm query. |
| `--faith_k` | `int` | `1` | Số lượng câu top-k quan trọng nhất được trích xuất để đo đạc độ tin cậy. |
| `--stability_runs`| `int` | `3` | Số lần chạy lặp thực nghiệm độc lập (thay đổi seed) để đánh giá sai số ổn định. |
| `--use_abs_faith`| `flag` | `False` | Nếu bật, hệ thống sẽ lấy giá trị tuyệt đối (Absolute values) của attribution để xếp hạng câu. |

---

## 📈 Ý Nghĩa Các Chỉ Số Đánh Giá Đầu Ra

Kết quả đánh giá định lượng sẽ được hiển thị trực quan dưới dạng bảng thống kê trung bình (`AVG RESULTS`) tại Terminal và đồng thời ghi log trực tiếp vào bộ lưu trữ **TensorBoard** bên trong thư mục `runs/`:

### 1. Nhóm chỉ số Độ Tin Cậy (Faithfulness)
* **Avg Comprehensiveness (Tính toàn diện):** Đo lường mức sụt giảm điểm số dự đoán của mô hình gợi ý khi loại bỏ đi top-$k$ câu quan trọng nhất. 
  * *Ý nghĩa:* Giá trị càng **lớn** càng tốt, chứng minh explainer đã tìm đúng các câu thực sự mang tính quyết định đến kết quả gợi ý.
* **Avg Sufficiency (Tính đầy đủ):** Đo mức giữ vững điểm số dự đoán khi *chỉ giữ lại* duy nhất các câu được chọn là quan trọng nhất và loại bỏ phần còn lại.
  * *Ý nghĩa:* Giá trị càng **nhỏ** càng tốt (hoặc âm), thể hiện bản thân các câu được chọn đã đủ để đại diện cho toàn bộ văn bản.

### 2. Nhóm chỉ số So Sánh với Baseline Chuẩn
* **Avg Rank Agreement vs Baseline:** Hệ số tương quan thứ tự Spearman giữa vector gán trọng số của phương pháp hiện tại so với thuật toán nền tảng chuẩn high-budget (`baseline_kernel` với lượng sample tổ hợp rất lớn).
  * *Ý nghĩa:* Càng tiến gần về `1.0` thể hiện thứ tự phân hạng phân phối câu quan trọng càng chính xác tuyệt đối so với lý thuyết trò chơi chuẩn gốc.
* **Avg MAE / RMSE:** Sai số tuyệt đối trung bình (MAE) và căn sai số bình phương trung bình (RMSE) về mặt trị số attribution trực tiếp đối chiếu với Baseline.
  * *Ý nghĩa:* Giá trị càng **nhỏ** càng tốt.

### 3. Nhóm chỉ số Ổn Định và Hiệu Năng
* **Avg Stability (Độ ổn định thực nghiệm):** Tính toán giá trị trung bình tương quan Spearman giữa mọi cặp phân phối kết quả của các phiên chạy lặp lại khi thay đổi hạt giống ngẫu nhiên (`seed`).
  * *Ý nghĩa:* Càng cao càng tốt (tiến về `1.0`), cho thấy thuật toán không bị ảnh hưởng quá mức bởi yếu tố lấy mẫu ngẫu nhiên.
* **Avg Runtime / CPU Time per paper:** Thời gian xử lý trung bình (tính bằng giây) trên mỗi bài báo.
  * *Ý nghĩa:* Đo lường chi phí tài nguyên và tốc độ tính toán thực tế của phương pháp giải thích.
"""
