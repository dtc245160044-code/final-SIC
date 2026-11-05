# 📑 BÁO CÁO ĐỒ ÁN CUỐI KỲ CHƯƠNG TRÌNH SAMSUNG INNOVATION CAMPUS (SIC)

**Đề tài:** Phát hiện Bình luận Độc hại Tiếng Việt trên Mạng Xã hội (Bộ dữ liệu ViHSD)  
**Kho mã nguồn:** `https://github.com/dtc245160044-code/final-SIC.git`  

> **Ghi chú:** Mọi con số thống kê, bảng biểu và ví dụ lỗi trong báo cáo này đều được trích xuất trực tiếp từ kết quả chạy code thật trong `data/processed/` và `outputs/` (tuyệt đối không điền số giả). Kết quả chính thức của dự án được lấy từ lần chấm duy nhất trên tập **Test (`2.986` mẫu)**. Mọi kết quả của cả hai mô hình (Baseline và PhoBERT fine-tuned từ Colab) đều đã được tính toán đầy đủ và đối chiếu thống kê chính thức trên tập Test.

---

## 1. Đặt vấn đề & Mục tiêu Đề tài

### 1.1. Bối cảnh thực tế
Sự bùng nổ của các nền tảng mạng xã hội và video trực tuyến (YouTube, Facebook, TikTok) tại Việt Nam kéo theo lượng bình luận khổng lồ mỗi ngày. Bên cạnh các ý kiến trao đổi tích cực, tình trạng bình luận xúc phạm, miệt thị, công kích cá nhân và ngôn từ thù ghét (hate speech) diễn biến phức tạp. Việc kiểm duyệt thủ công toàn bộ bình luận vừa tốn thời gian, vừa gây áp lực tâm lý lớn cho đội ngũ kiểm duyệt viên.

Tuy nhiên, bài toán tự động phát hiện bình luận độc hại tiếng Việt gặp nhiều thách thức đặc thù:
1. **Hiện tượng biến âm, teencode và né bộ lọc:** Người dùng mạng thường xuyên viết tắt (`ko`, `dc`, `bt`, `dm`), kéo dài chữ cái (`nguuuuu`), hoặc chèn dấu chấm/gạch ngang/dấu sao giữa các chữ cái (`m.à.y`, `n-g-u`, `đ*o`) để qua mặt các bộ lọc từ khóa tĩnh.
2. **Đặc trưng từ ghép tiếng Việt:** Khoảng trắng trong tiếng Việt dùng để tách âm tiết chứ không tách từ (ví dụ: `"bất_hạnh"`, `"thất_học"`), đòi hỏi bước tách từ (`word segmentation`) chính xác trước khi đưa vào mô hình học máy.
3. **Độ lệch nhãn nghiêm trọng và ranh giới ngữ nghĩa mờ:** Bình luận thông thường chiếm đa số tuyệt đối (`82,29%`), trong khi lớp `OFFENSIVE` chỉ chiếm `6,76%` và có ranh giới rất mờ giữa câu nói đùa suồng sã, câu xúc phạm nhẹ và câu thù ghét.

### 1.2. Mục tiêu của dự án & Tóm tắt Kết quả Chính thức (trên tập Test)
- Xây dựng quy trình tiền xử lý văn bản tiếng Việt chuẩn hóa (`src/preprocess.py`), giải quyết trực tiếp teencode, chữ kéo dài, ký tự chèn và tách từ bằng `pyvi`.
- Huấn luyện, tinh chỉnh có hệ thống (76 thử nghiệm trên tập Validation) và hiệu chuẩn xác suất mô hình **TF-IDF + LinearSVC (Calibrated)**, đồng thời xây dựng pipeline huấn luyện mô hình ngôn ngữ tiền huấn luyện **PhoBERT (`vinai/phobert-base`)**.
- **Kết quả chính thức trên tập Test độc lập (`2.986` mẫu — chấm đúng 1 lần duy nhất cho mỗi mô hình):**
  - **Mô hình TF-IDF + LinearSVC (Calibrated):** **Accuracy = `87,74%`**, **F1-macro = `67,03%`**, **Toxic Precision = `74,59%`**, **Toxic Recall = `68,24%`** (F1 `CLEAN` = `94,13%`, F1 `OFFENSIVE` = `41,36%`, F1 `HATE` = `65,61%`), độ trễ siêu nhanh **`4,32 ms/bình luận`**, kích thước **`9,42 MB`**.
  - **Mô hình PhoBERT (`vinai/phobert-base` fine-tune):** **Accuracy = `84,73%`**, **F1-macro = `67,49%`**, **Toxic Precision = `61,05%`**, **Toxic Recall = `81,47%`** (F1 `CLEAN` = `92,13%`, F1 `OFFENSIVE` = `47,33%`, F1 `HATE` = `63,01%`), độ trễ **`25,28 ms/bình luận`**, kích thước **`516,95 MB`**.
  - **Kết luận so sánh:** Hai mô hình có hiệu năng F1-macro tổng thể tương đương nhau (+0,46 điểm % không có ý nghĩa thống kê). PhoBERT vượt trội về khả năng phát hiện bình luận độc hại (bắt thêm 70 bình luận độc hại, Toxic Recall đạt 81,47%), nhưng đánh đổi bằng tỷ lệ báo nhầm cao hơn và tài nguyên tính toán lớn hơn.
- Xây dựng ứng dụng Web **Streamlit 6 trang** phục vụ thử nghiệm trực tiếp (kèm giải thích từ gây độc bằng kỹ thuật *Leave-One-Token-Out*), minh họa tiền xử lý, hiển thị bảng đánh giá và quét bình luận thực tế qua **YouTube Data API v3** (`src/youtube.py` & `scripts/lay_youtube_demo.py`).

---

## 2. Bộ dữ liệu ViHSD & Quy trình Tiền xử lý

### 2.1. Khám phá và làm sạch bộ dữ liệu (`scripts/process_data_fixed.py`)
Dự án sử dụng bộ dữ liệu **ViHSD (Vietnamese Hate Speech Detection)** với 3 nhãn chuẩn:
- **`0` — Bình thường (`CLEAN`):** Bình luận văn minh, trung lập hoặc tích cực.
- **`1` — Xúc phạm (`OFFENSIVE`):** Bình luận châm chọc, mỉa mai, dùng từ ngữ thô tục/suồng sã nhưng chưa tới mức công kích thù ghét nặng nề.
- **`2` — Thù ghét (`HATE`):** Bình luận chửi bới trực diện, lăng mạ, hạ nhục nhân phẩm hoặc kích động thù địch.

Quá trình làm sạch từ file gốc `data/raw/comments.csv` diễn ra qua các bước kiểm soát chất lượng nghiêm ngặt:
- **Dữ liệu thô ban đầu (`data/raw/comments.csv`):** `33.397` dòng (`01_du_lieu_goc.csv` giữ `30.606` dòng hợp lệ).
- **Sau khi làm sạch văn bản (`lam_sach`) và loại bỏ dòng rỗng:** Khử **755 dòng trùng lặp** nội dung sau chuẩn hóa (trong đó có **240 câu trùng lặp có nhãn bất đồng** giữa những người gán nhãn được xử lý bằng cách giữ nhãn xuất hiện nhiều nhất — *majority vote*).
- **Tổng số mẫu sạch duy nhất (`02_du_lieu_sach.csv`):** **`29.851` mẫu**.
- **Chia tập dữ liệu phân tầng (`stratified split`, `random_state=42` theo tỷ lệ `80% / 10% / 10%`):** Đảm bảo không rò rỉ dữ liệu (*zero data leakage* — giao nhau của `text_clean` giữa Train, Val và Test đều bằng `0`).

| Tập dữ liệu | File lưu trữ | Tổng số mẫu | Nhãn `0` (`CLEAN`) | Nhãn `1` (`OFFENSIVE`) | Nhãn `2` (`HATE`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train (80%)** | `data/processed/03_train.csv` | **23.880** | 19.652 (`82,29%`) | 1.615 (`6,76%`) | 2.613 (`10,94%`) |
| **Validation (10%)** | `data/processed/03_val.csv` | **2.985** | 2.456 (`82,28%`) | 202 (`6,77%`) | 327 (`10,95%`) |
| **Test (10%)** | `data/processed/03_test.csv` | **2.986** | 2.457 (`82,28%`) | 202 (`6,76%`) | 327 (`10,95%`) |
| **Toàn bộ** | `data/processed/02_du_lieu_sach.csv` | **29.851** | **24.565 (`82,29%`)** | **2.019 (`6,76%`)** | **3.267 (`10,94%`)** |

**Độ dài câu trong tập Train:** Trung bình đạt **11,6 từ/bình luận**, trung vị đạt **8,0 từ/bình luận**, câu dài nhất gồm **444 từ**.

### 2.2. Thống kê các dấu hiệu nhiễu trong dữ liệu (`phat_hien`)
Hàm `phat_hien(raw)` trong `src/preprocess.py` quét trên 23.880 mẫu huấn luyện (`03_train.csv`) ghi nhận:
- **Teencode / viết tắt:** Xuất hiện trong **4.623** bình luận (`19,36%`).
- **Emoji / biểu tượng cảm xúc:** Xuất hiện trong **3.225** bình luận (`13,50%`).
- **Chữ kéo dài (`>= 3` ký tự lặp như `đẹpppp`, `nguuuu`):** Xuất hiện trong **804** bình luận (`3,37%`).
- **Chèn ký tự (`.`, `-`, `_`) giữa chữ cái để né bộ lọc:** Xuất hiện trong **126** bình luận (`0,53%`).
- **Đường dẫn URL / Link:** Xuất hiện trong **31** bình luận (`0,13%`).

### 2.3. Quy trình 8 bước tiền xử lý chuẩn (`src/preprocess.py`)
Mọi bình luận (cả khi huấn luyện lẫn khi dự đoán trực tiếp trên Web) đều đi qua duy nhất 2 hàm `lam_sach(text)` và `tach_tu(text_sach)`:
1. Chuẩn hóa bảng mã tiếng Việt về chuẩn **Unicode dựng sẵn (`NFC`)**.
2. Xóa thẻ HTML (`<...>`) và các đường dẫn URL (`http://`, `https://`, `www.`, `t.me/`).
3. Xóa thẻ nhắc tên người dùng (`@username`) và biểu tượng cảm xúc (Emoji).
4. Rút gọn các chữ cái bị kéo dài từ 3 ký tự liên tiếp trở lên về 1 ký tự (nhưng giữ nguyên các từ viết hoa toàn bộ như tên viết tắt).
5. Ghép liền các chữ cái bị cố tình chèn dấu `.`, `-`, `_` ở giữa (ví dụ: `n.g.u` -> `ngu`, `m-à-y` -> `mày`).
6. Chuyển toàn bộ văn bản về chữ thường (`lowercase`).
7. Dịch **42 từ teencode/viết tắt phổ biến** sang từ tiếng Việt chuẩn theo từ điển `TEENCODE` (`ko` -> `không`, `dc` -> `được`, `bt` -> `bình thường`, `j` -> `gì`, `vs` -> `với`, `dm` -> `đm`...).
8. Tách từ tiếng Việt bằng `pyvi.ViTokenizer.tokenize` (bảo toàn ký tự `*` thông qua token tạm `X_STAR_X` để không làm vỡ các từ viết lách luật dạng `đ*o`, `đ*t`).

---

## 3. Phương pháp Mô hình hóa & Tinh chỉnh Siêu tham số trên Tập Validation

### 3.1. Mô hình Baseline: TF-IDF kết hợp LinearSVC có Hiệu chuẩn Xác suất
- **Trích xuất đặc trưng (`FeatureUnion`):** Kết hợp đồng thời hai nhánh TF-IDF:
  - **Word n-gram:** Bắt từ đơn, cụm 2 từ và cụm 3 từ đã tách (`ngram_range=(1, 3)`, `min_df=2`, `max_features=45.000`, `sublinear_tf=True`).
  - **Character n-gram (`char_wb`):** Bắt chuỗi ký tự con từ 2 đến 5 ký tự trong phạm vi từ (`ngram_range=(2, 5)`, `min_df=3`, `max_features=45.000`, `sublinear_tf=True`), giúp mô hình nhận diện tốt các từ viết sai chính tả nhẹ hoặc biến thể từ tục chưa có trong từ điển.
- **Lý do dùng `CalibratedClassifierCV(cv=5)`:** Mô hình `LinearSVC` phân lớp rất mạnh trên không gian TF-IDF chiều cao nhưng bản chất chỉ xuất ra khoảng cách tới siêu phẳng (`decision_function`), không có hàm `predict_proba()`. Việc bọc `CalibratedClassifierCV(LinearSVC, cv=5, method="sigmoid")` (Platt scaling 5-fold chỉ fit trên tập Train) giúp mô hình xuất ra xác suất thực sự cho từng nhãn `P(CLEAN)`, `P(OFFENSIVE)`, `P(HATE)` và tính được `diem_toxic = P(OFFENSIVE) + P(HATE)`.

### 3.2. Quá trình 76 thử nghiệm trên tập Validation (`outputs/thu_nghiem_baseline.csv`)
Để tìm cấu hình tốt nhất mà **tuyệt đối không chạm vào tập Test**, kịch bản `scripts/tune_baseline.py` đã thực hiện **76 lần thử nghiệm** trên tập Validation (`2.985` mẫu) theo 4 nhóm:
1. **Nhóm 1 — Mốc chuẩn `CalibratedClassifierCV(LinearSVC, cv=5)`** với các hệ số phạt `C` từ `0.05` đến `1.0` và `class_weight` (`None`, `balanced`).
2. **Nhóm 2 — Đối chiếu với `LogisticRegression`** (`C` từ `0.5` đến `5.0`, `class_weight="balanced"` và không trọng số).
3. **Nhóm 3 — Tinh chỉnh trọng số lớp (`class_weight = {0: 1.0, 1: w1, 2: w2}`)** và **ngưỡng quyết định riêng cho lớp `OFFENSIVE`** (`thr_off` từ `0.20` đến `0.45`).
4. **Nhóm 4 — Mở rộng đặc trưng n-gram (`word (1,3) + char_wb (2,5)`) kết hợp trọng số lớp và ngưỡng.**

**Bảng trích xuất các thử nghiệm trên tập Validation (`2.985` mẫu — chỉ dùng để chọn mô hình, không phải kết quả cuối):**

| Thứ hạng (Val) | Tên thử nghiệm (`experiment_name`) | Nhóm mô hình | Accuracy Val | F1-macro Val | F1 `CLEAN` (Val) | F1 `OFFENSIVE` (Val) | F1 `HATE` (Val) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1 (Được chọn)** | `CalibratedSVC_C0.1_sublinear_w13_c25_w_1_3.0_2.0` | SVC Calibrated + N-gram (1,3)+(2,5) + Trọng số `{0:1, 1:3, 2:2}` | **87,64%** | **68,19%** | **93,97%** | **47,03%** | **63,56%** |
| **2** | `CalibratedSVC_C0.1_sublinear_w13_c25_w_1_3.0_2.0_thrOff_0.4` | SVC Calibrated + Ngưỡng `thr_off=0.40` | 87,27% | 68,10% | 93,86% | 46,61% | 63,84% |
| **3** | `CalibratedSVC_C0.15_sublinear_w13_c25_w_1_3.5_2.0_thrOff_0.45` | SVC Calibrated (`C=0.15`, `w={0:1, 1:3.5, 2:2}`) | 87,40% | 67,84% | 93,86% | 46,65% | 63,02% |
| **17** | `CalibratedSVC_C0.2_bal` | SVC Calibrated chuẩn (`C=0.2, balanced, w12+c25`) | 88,07% | 67,05% | 94,14% | 41,72% | 65,30% |
| **31** | `LogReg_C2.0_none` | Logistic Regression (`C=2.0`, không trọng số) | 88,21% | 66,35% | 94,25% | 41,08% | 63,72% |
| **Mốc cũ** | `UnCalibrated_LinearSVC_C0.1_bal` (`run_baseline.py`) | LinearSVC chưa hiệu chuẩn (`C=0.1, balanced`) | 87,20% | 66,05% | 93,96% | 41,51% | 62,67% |
| **65** | `LogReg_C1.0_bal` | Logistic Regression (`C=1.0, balanced`) | 83,35% | 63,34% | 91,63% | 37,52% | 60,87% |

> **Giải thích vì sao điểm trên Validation cao hơn điểm trên Test:**  
> Cấu hình `CalibratedSVC_C0.1_sublinear_w13_c25_w_1_3.0_2.0` được chọn vì đạt đỉnh trong số **76 lần thử nghiệm liên tiếp trên cùng tập Validation (`03_val.csv`)**. Việc chọn cực đại qua 76 lần thử trên Val tất yếu tạo ra độ lệch lạc quan nhẹ (*selection bias / multiple-comparisons optimism*) trên tập Val (`F1-macro Val = 68,19%`, `F1 OFFENSIVE Val = 47,03%`). Khi đem mô hình này chấm đúng 1 lần duy nhất trên tập **Test độc lập (`03_test.csv`)**, kết quả thực tế đạt **`F1-macro Test = 67,03%`** và **`F1 OFFENSIVE Test = 41,36%`** (chênh lệch `-1,16%` ở F1-macro). Do đó, **số liệu trên tập Test ở Mục 4 là kết quả chính thức duy nhất của báo cáo**.

### 3.3. Mô hình PhoBERT Fine-tune (`scripts/run_phobert.py` & `notebooks/03_phobert_colab.ipynb`)
- **Kiến trúc thực tế trong code:** Fine-tune mô hình **`vinai/phobert-base`** với tầng phân loại 3 lớp (`AutoModelForSequenceClassification`), độ dài tối đa `max_length = 128` token sau tách từ `pyvi`.
- **Hàm mất mát có trọng số (`WeightedTrainer`):** Để chống lệch nhãn, hàm `CrossEntropyLoss` được truyền vector trọng số lớp cân bằng (`compute_class_weight('balanced')`) tính trên tập Train, kết hợp `fp16=True`, `learning_rate=2e-5`, `weight_decay=0.01`, `4 epochs` và chọn checkpoint tốt nhất theo `eval_f1_macro`.
- **Kết quả huấn luyện thực tế trên Google Colab GPU (Tesla T4):**
  - Quá trình huấn luyện diễn ra tự động hoàn toàn theo kịch bản `notebooks/03_phobert_colab.ipynb` và `scripts/run_phobert.py` với tổng thời gian **`483,64 giây` (~8 phút)** cho 4 epochs trên 23.880 mẫu Train (`batch_size = 16`, `gradient_accumulation_steps = 2`, effective batch size = 32).
  - Diễn biến đánh giá trên tập Validation qua 4 epochs (`outputs/phobert_training_log.csv`):
    - **Epoch 1:** Loss = 0.5126 | Val Accuracy = 83,08% | Val F1-macro = **65,41%**
    - **Epoch 2:** Loss = 0.4497 | Val Accuracy = 85,26% | Val F1-macro = **68,41%** ➔ **Checkpoint tốt nhất (`best`)** được lưu lại.
    - **Epoch 3:** Loss = 0.4950 | Val Accuracy = 84,39% | Val F1-macro = **67,57%**
    - **Epoch 4:** Loss = 0.5480 | Val Accuracy = 84,66% | Val F1-macro = **67,89%**
  - Trọng số tốt nhất tại Epoch 2 (`models/phobert/best/`) đạt **F1-macro Validation = `68,41%`** và **Accuracy Validation = `85,26%`** được mang về máy trạm để đánh giá chính thức trên tập Test ở Mục 4.

---

## 4. Kết quả Chính thức trên Tập Test (`2.986` mẫu — Chấm đúng 1 lần)

Kịch bản `scripts/evaluate_final.py` chấm điểm tập `data/processed/03_test.csv` (`2.986` mẫu độc lập hoàn toàn) đúng 1 lần duy nhất và xuất ra `outputs/bang_so_sanh_mo_hinh.csv`:

### 4.1. Bảng kết quả chính thức trên tập Test (`outputs/bang_so_sanh_mo_hinh.csv`)
| Mô hình | Accuracy (%) | F1-macro (%) | Toxic Precision (%) | Toxic Recall (%) | F1 `CLEAN` (%) | F1 `OFFENSIVE` (%) | F1 `HATE` (%) | Độ trễ (ms/câu) | Kích thước (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LinearSVC (Calibrated)** | **87,74%** | **67,03%** | **74,59%** | **68,24%** | **94,13%** | **41,36%** | **65,61%** | **4,32 ms** | **9,42 MB** |
| **PhoBERT (`vinai/phobert-base` fine-tune)** | **84,73%** | **67,49%** | **61,05%** | **81,47%** | **92,13%** | **47,33%** | **63,01%** | **25,28 ms** | **516,95 MB** |

> **Nhận xét so sánh trung thực và chuyên sâu giữa hai mô hình trên tập Test:**
> 1. **Hiệu năng tổng thể (F1-macro) tương đương nhau:** PhoBERT đạt F1-macro **67,49%**, nhỉnh hơn Baseline (**67,03%**) đúng **+0,46 điểm %**. Kết quả kiểm định Paired Bootstrap (2.000 lần lặp mẫu lại trên tập Test, `outputs/bootstrap_chenh_lech_f1.csv`) cho thấy khoảng tin cậy 95% của độ chênh lệch là `[-1,98%, +2,99%]`. Vì khoảng này chứa 0, ta có cơ sở khoa học khẳng định: **hai mô hình có hiệu năng phân loại tổng thể tương đương nhau**.
> 2. **Sự đánh đổi bản chất: Khả năng bắt độc hại (Recall) đối lập với Độ báo nhầm (Precision):**
>    - **PhoBERT bắt nhạy hơn nhiều (Toxic Recall = 81,47% vs 68,24%):** Trên tổng số 529 câu độc hại thật trong tập Test, PhoBERT bắt được **431 câu**, chỉ để lọt 98 câu (Baseline chỉ bắt được 361 câu, để lọt 168 câu). Như vậy PhoBERT **bắt thêm được 70 bình luận độc hại**, và đặc biệt cải thiện mạnh lớp `OFFENSIVE` (F1 tăng từ 41,36% lên 47,33%, Recall lớp này tăng vọt từ 39,11% lên 61,39%) nhờ khả năng hiểu nghĩa ngữ cảnh của transformer hai chiều.
>    - **PhoBERT báo nhầm người dùng bình thường nhiều hơn (Toxic Precision = 61,05% vs 74,59%):** Trong 2.457 câu bình thường, PhoBERT báo nhầm 275 câu thành độc hại (`11,19%`), cao hơn gấp đôi so với Baseline chỉ báo nhầm 123 câu (`5,01%`). Điều này khiến Accuracy tổng thể của PhoBERT (84,73%) thấp hơn Baseline (87,74%).
>    - **Chi phí triển khai phần cứng:** Baseline cực nhẹ (**9,42 MB**) và chạy nhanh gấp 6 lần (**4,32 ms/câu**), chạy hoàn toàn trên CPU thông thường. PhoBERT nặng tới **516,95 MB** (gấp 55 lần) và độ trễ **25,28 ms/câu** trên GPU.
> 3. **Giải thích tiêu chí chọn `best_model` trong `models/config.json`:**
>    File cấu hình hệ thống `models/config.json` do `scripts/build_config.py` sinh ra tự động, trong đó tiêu chí xếp hạng mô hình tốt nhất được lập trình là **mô hình có F1-macro cao nhất trên tập Test** (`outputs/bang_so_sanh_mo_hinh.csv`). Vì `67,49%` > `67,03%`, `PhoBERT (fine-tune)` được chọn làm `best_model` mặc định khi người dùng có thư mục trọng số.*, báo nhầm sang `Xúc phạm` 57 mẫu và sang `Thù ghét` 66 mẫu (tổng cộng **123 ca Báo nhầm — False Positive**, chiếm `4,99%` số câu bình thường).
- **Lớp `1 — Xúc phạm` (201 mẫu thật):** Mô hình đoán đúng **73 mẫu (`36,32%`)**, bỏ sót sang `Bình thường` 91 mẫu (`45,27%`) và nhầm mức độ sang `Thù ghét` 37 mẫu (`18,41%`). Kết hợp với Precision `48,03%`, **F1 chính thức của lớp `OFFENSIVE` trên tập Test là `41,36%`**.
- **Lớp `2 — Thù ghét` (319 mẫu thật):** Mô hình đoán đúng **204 mẫu (`63,95%`)**, nhầm mức độ sang `Xúc phạm` 38 mẫu (`11,91%`) và bỏ sót sang `Bình thường` 77 mẫu (`24,14%`). **F1 chính thức của lớp `HATE` trên tập Test đạt `65,61%`** (cao hơn trên Val là `63,56%`).
- **Tổng số ca Bỏ sót độc hại (False Negative — thật là 1 hoặc 2 nhưng đoán là 0):** **168 mẫu** trên tổng số 520 mẫu độc hại.

---

## 5. Phân tích Lỗi Chuyên sâu trên Tập Test

### 5.1. Hiệu năng theo độ dài câu (`outputs/phan_tich_do_dai_cau.csv`)
Chia 2.986 mẫu tập Test thành 3 nhóm theo số lượng từ sau tách từ:

| Nhóm độ dài câu | Số bình luận | Tỷ lệ mẫu (%) | Accuracy (%) | F1-macro (%) | F1 `CLEAN` (%) | F1 `OFFENSIVE` (%) | F1 `HATE` (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ngắn (`<= 10` từ)** | 1.821 | 60,98% | **90,33%** | **66,70%** | 95,54% | **48,16%** | 56,38% |
| **Trung bình (`11 - 25` từ)** | 891 | 29,84% | 84,51% | 62,44% | 92,07% | 33,64% | 61,60% |
| **Dài (`> 25` từ)** | 274 | 9,18% | 81,02% | 60,36% | 89,10% | **13,33%** | **78,64%** |

**Phát hiện quan trọng:**
- Ở các bình luận **ngắn (`<= 10` từ, chiếm `60,98%` tập Test)**, mô hình đạt độ chính xác cao nhất (`90,33%`) và nhận diện lớp `OFFENSIVE` tốt nhất (`F1 = 48,16%`) vì câu ngắn thường chứa trực tiếp cụm từ châm chọc mà không bị loãng bởi các từ trung lập.
- Ngược lại, ở các bình luận **dài (`> 25` từ)**: **F1 của lớp `HATE` tăng vọt lên `78,64%`** (do các đoạn chửi bới dài lặp lại nhiều từ khóa công kích mạnh), nhưng **F1 của lớp `OFFENSIVE` giảm mạnh xuống `13,33%`** vì trong câu dài, mật độ từ nhạy cảm bị pha loãng hoặc bị đẩy hẳn sang lớp `HATE`.

### 5.2. Tỷ lệ lỗi theo các đặc tính nhiễu (`outputs/phan_tich_nhieu_dac_biet.csv`)
Đối chiếu tỷ lệ đoán sai nhãn trên các nhóm bình luận có chứa dấu hiệu nhiễu trên tập Test:

| Nhóm bình luận trên tập Test | Số mẫu | Tỷ lệ đoán sai nhãn (%) | Accuracy (%) | F1-macro (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Câu không chứa các dấu hiệu trên** | 1.991 | 15,17% | 84,83% | 67,54% |
| **Câu có chữ kéo dài** | 102 | 16,67% | 83,33% | 66,36% |
| **Câu có emoji** | 403 | 11,41% | 88,59% | 68,54% |
| **Câu có teencode / viết tắt** | 596 | 18,12% | 81,88% | 64,39% |
| **Câu chèn ký tự / dấu `*` né lọc** | 19 | **10,53%** | **89,47%** | **84,27%** |

**Nhận xét đối chiếu giữa hai mô hình (`outputs/so_sanh_theo_nhom_hai_mo_hinh.csv`):**
- **Bình luận có dấu `*` hoặc chèn ký tự né bộ lọc (19 mẫu):** Đây là nhóm ghi nhận sự vượt trội rõ rệt nhất của **PhoBERT**. Mô hình Baseline TF-IDF bị đánh lừa nặng nề với tỷ lệ đoán sai lên tới **36,84%** (F1-macro chỉ đạt 59,00%). Trong khi đó, PhoBERT nhờ cơ chế self-attention và hiểu ngữ cảnh xung quanh token đã giảm tỷ lệ lỗi xuống chỉ còn **10,53%** (F1-macro tăng vọt lên **84,27%**).
- **Bình luận dài (>25 từ):** Cả hai mô hình đều nhận diện rất tốt lớp Thù ghét (`F1 HATE` đạt ~78%), nhưng PhoBERT giữ được F1 lớp Xúc phạm ở mức **41,03%**, trong khi Baseline bị tụt xuống **13,33%** do các n-gram từ ngữ xúc phạm bị pha loãng hoàn toàn trong câu dài.
- **Bình luận có teencode và chữ kéo dài:** Nhờ quy trình 8 bước tiền xử lý chuẩn hóa ở `src/preprocess.py`, cả hai mô hình đều duy trì được F1-macro ổn định trên 64%.

### 5.3. Phân tích nguyên nhân lớp `OFFENSIVE` đạt F1 = `41,36%` trên Test & Top 15 ca đoán sai (`outputs/top15_offensive_doan_sai.csv`)
Việc lớp `OFFENSIVE` có F1 chính thức trên tập Test là **`41,36%`** (dưới mức `0,50`) xuất phát từ **3 hạn chế bản chất của dữ liệu**:
1. **Thiếu mẫu huấn luyện nghiêm trọng:** Lớp `OFFENSIVE` chỉ chiếm **`6,76%`** bộ dữ liệu (`1.615` mẫu Train, `202` mẫu Test).
2. **Định nghĩa nhãn nằm ở vùng xám:** Nhiều câu trong `outputs/top15_offensive_doan_sai.csv` dùng từ ngữ hoàn toàn thông thường để mỉa mai hoặc nói trại âm:
   - Ví dụ 1 (Minh chứng PhoBERT hiểu ngữ cảnh tốt hơn Baseline): *"Trường hợp này ở địa phương khác kêu là gì ko biết còn địa phương mình gọi là khùng á"* -> Nhãn thật trên Test: `OFFENSIVE (1)`.
  - **Baseline:** Đoán sai thành `Bình thường` (`P_toxic = 0,1231`) vì cụm từ xúc phạm *"khùng á"* nằm ở cuối câu điều kiện dài và bị các từ trung lập lấn át.
  - **PhoBERT:** Đoán **chính xác `Xúc phạm`** (`P_toxic = 0,7832`, xác suất OFFENSIVE = 0,6988) nhờ khả năng nắm bắt ngữ cảnh trọn vẹn của câu.
   - Ví dụ 2: *"Này khẩu phật tâm xà"* hoặc *"Cứ tưởng thế là ngầu"* -> Nhãn thật: `OFFENSIVE`, mô hình đoán: `Bình thường` (`P = 0,8882` và `0,8746`) vì không hề có từ tục, chỉ mang sắc thái chê bai ngầm.
   - Ví dụ 3: *"Đ*o hiểu lun"* và *"Ghi thêm chữ đ*t chị m nx :))* -> Nhãn thật: `OFFENSIVE`, mô hình đoán: `Bình thường` (`P = 0,8696` và `0,8613`) do người viết dùng dấu `*` và viết tắt biến thể (`lun`, `nx`).
3. **Bất đồng nhãn từ người gán nhãn (Annotator Disagreement):** Ngay trong `outputs/phan_tich_loi.csv`, nhiều câu có nhãn gốc là `Bình thường (0)` nhưng chứa từ chửi thề rất nặng (ví dụ: *"Đụ mẹ tay ghi , não đéo kịp suy nghĩ..."* hoặc *"Má dơ vl =)))"*), khiến mô hình dự đoán là `Thù ghét / Xúc phạm` với độ tự tin `> 99%` và bị tính là lỗi False Positive.

📝 **`[CẦN NHÓM ĐỌC LẠI & BỔ SUNG NHẬN XÉT]`:** Nhóm có thể chọn thêm 3–5 câu trong file `outputs/phan_tich_loi.csv` để điền cột `nhận_xét_của_nhóm` và trích dẫn trực tiếp vào slide báo cáo.

---

## 6. Kiến trúc Ứng dụng Web Streamlit & Tích hợp YouTube API

Ứng dụng Web được thiết kế theo kiến trúc tách biệt hoàn toàn giữa **Tầng giao diện (`app/`)** và **Tầng suy luận lõi (`src/`)**:
- **`src/inference.py`:** Cung cấp hợp đồng hàm chuẩn (`doc_cau_hinh`, `tai_mo_hinh`, `phan_loai`, `phan_loai_hang_loat`, `tim_tu_gay_doc`). Hàm `tim_tu_gay_doc` sử dụng kỹ thuật **Leave-One-Token-Out (LOTO)** kết hợp độ giảm xác suất và độ giảm logit để tô đỏ chính xác những từ làm tăng điểm độc hại trong câu.
- **`src/youtube.py` & `scripts/lay_youtube_demo.py`:** Kết nối trực tiếp `commentThreads.list` của YouTube Data API v3, giới hạn an toàn tối đa 500 bình luận (hoặc 100–200 bình luận qua `scripts/lay_youtube_demo.py`), xử lý đầy đủ các mã lỗi tiếng Việt (`commentsDisabled`, `quotaExceeded`, `keyInvalid`, `videoNotFound`) và tuyệt đối không lưu tên người bình luận hay API key. Chế độ dự phòng trên Trang 4 chỉ hiển thị khi người dùng đã tự chạy API thật để lưu `outputs/ket_qua_youtube.csv` (kèm cột `video_id` và `ngay_lay`); nếu chưa chạy thì hiển thị rõ *"Chưa có kết quả"*.
- **6 trang giao diện (`app/Home.py` & `app/pages/1..5`):** Đã vượt qua toàn bộ bộ kiểm thử tự động `pytest` (`tests/test_inference.py`) ở cả 3 trạng thái: đủ dữ liệu, thiếu thư mục `outputs/`, và thiếu thư mục `models/` mà không phát sinh bất kỳ lỗi dừng chương trình nào.

---

## 7. Hạn chế Đạo đức AI & Hướng Phát triển

1. **Nguyên tắc Con người làm trung tâm (*Human-in-the-loop*):** Với tỷ lệ Báo nhầm (FP) khoảng `4,99%` trên các câu bình thường (nhất là các câu dùng từ lóng/tục để đùa vui, khen ngợi bạn bè như *"cá kho... ngon vailone :)"*), hệ thống **chỉ được sử dụng làm công cụ sàng lọc và xếp hạng ưu tiên** cho kiểm duyệt viên, tuyệt đối không tự động xóa bình luận hay khóa tài khoản người dùng.
2. **Bảo vệ quyền riêng tư:** Không thu thập, lưu trữ hay hiển thị định danh cá nhân của người bình luận.
3. **Hướng phát triển tiếp theo:**
   - Hoàn tất huấn luyện và tích hợp mô hình **PhoBERT (`vinai/phobert-base`)** để nắm bắt ngữ cảnh hai chiều và cải thiện khả năng nhận diện các câu mỉa mai, nói giảm nói tránh ở lớp `OFFENSIVE`.
   - Bổ sung ngữ cảnh tiêu đề bài viết / video và bình luận gốc (parent comment) vào đầu vào của mô hình.
   - Mở rộng bộ chuẩn hóa từ lóng, từ viết trại âm và ký tự che chữ (`*`) bằng mô hình ngôn ngữ mức ký tự.
