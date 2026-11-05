# 📑 BÁO CÁO ĐỒ ÁN CUỐI KỲ CHƯƠNG TRÌNH SAMSUNG INNOVATION CAMPUS (SIC)

**Đề tài:** Phát hiện Bình luận Độc hại Tiếng Việt trên Mạng Xã hội (Bộ dữ liệu ViHSD)  
**Kho mã nguồn:** `https://github.com/dtc245160044-code/final-SIC.git`  

> **Ghi chú:** Mọi con số thống kê, bảng biểu và ví dụ lỗi trong báo cáo này đều được trích xuất trực tiếp từ kết quả chạy code thật trong `data/processed/` và `outputs/` (tuyệt đối không điền số giả). Các mục đánh dấu **`[CẦN NHÓM ĐỌC LẠI / BỔ SUNG KHI CÓ PHOBERT]`** là vị trí dành cho nhóm rà soát nhận xét cá nhân hoặc điền thêm dòng PhoBERT sau khi chạy xong notebook Colab `notebooks/03_phobert_colab.ipynb`.

---

## 1. Đặt vấn đề & Mục tiêu Đề tài

### 1.1. Bối cảnh thực tế
Sự bùng nổ của các nền tảng mạng xã hội và video trực tuyến (YouTube, Facebook, TikTok) tại Việt Nam kéo theo lượng bình luận khổng lồ mỗi ngày. Bên cạnh các ý kiến trao đổi tích cực, tình trạng bình luận xúc phạm, miệt thị, công kích cá nhân và ngôn từ thù ghét (hate speech) diễn biến phức tạp. Việc kiểm duyệt thủ công toàn bộ bình luận vừa tốn thời gian, vừa gây áp lực tâm lý lớn cho đội ngũ kiểm duyệt viên.

Tuy nhiên, bài toán tự động phát hiện bình luận độc hại tiếng Việt gặp nhiều thách thức đặc thù:
1. **Hiện tượng biến âm, teencode và né bộ lọc:** Người dùng mạng thường xuyên viết tắt (`ko`, `dc`, `bt`, `dm`), kéo dài chữ cái (`nguuuuu`), hoặc chèn dấu chấm/gạch ngang/dấu sao giữa các chữ cái (`m.à.y`, `n-g-u`, `đ*o`) để qua mặt các bộ lọc từ khóa tĩnh.
2. **Đặc trưng từ ghép tiếng Việt:** Khoảng trắng trong tiếng Việt dùng để tách âm tiết chứ không tách từ (ví dụ: `"bất_hạnh"`, `"thất_học"`), đòi hỏi bước tách từ (`word segmentation`) chính xác trước khi đưa vào mô hình học máy.
3. **Độ lệch nhãn nghiêm trọng và ranh giới ngữ nghĩa mờ:** Bình luận thông thường chiếm đa số tuyệt đối (>80%), trong khi ranh giới giữa câu nói đùa suồng sã, câu xúc phạm nhẹ (`OFFENSIVE`) và câu thù ghét (`HATE`) rất mỏng manh.

### 1.2. Mục tiêu của dự án
- Xây dựng quy trình tiền xử lý văn bản tiếng Việt chuẩn hóa (`src/preprocess.py`), giải quyết trực tiếp teencode, chữ kéo dài, ký tự chèn và tách từ bằng `pyvi`.
- Huấn luyện, tinh chỉnh có hệ thống (76 thử nghiệm trên tập Validation) và hiệu chuẩn xác suất mô hình học máy **TF-IDF + LinearSVC (Calibrated)**, đồng thời xây dựng pipeline huấn luyện mô hình ngôn ngữ tiền huấn luyện **PhoBERT (`vinai/phobert-base-v2`)**.
- Đánh giá khách quan một lần duy nhất trên tập **Test (2.986 mẫu)** và thực hiện phân tích lỗi chuyên sâu (theo loại lỗi FP/FN, theo độ dài câu, theo đặc tính nhiễu và phân tích riêng lớp `OFFENSIVE`).
- Xây dựng ứng dụng Web **Streamlit 6 trang** phục vụ thử nghiệm trực tiếp (kèm giải thích từ gây độc bằng kỹ thuật *Leave-One-Token-Out*), minh họa tiền xử lý, hiển thị bảng đánh giá và quét bình luận thực tế qua **YouTube Data API v3**.

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
| **Train (80%)** | `data/processed/03_train.csv` | **23.880** | 19.726 (`82,60%`) | 1.606 (`6,73%`) | 2.548 (`10,67%`) |
| **Validation (10%)** | `data/processed/03_val.csv` | **2.985** | 2.466 (`82,61%`) | 201 (`6,73%`) | 318 (`10,65%`) |
| **Test (10%)** | `data/processed/03_test.csv` | **2.986** | 2.466 (`82,59%`) | 201 (`6,73%`) | 319 (`10,68%`) |
| **Toàn bộ** | `data/processed/02_du_lieu_sach.csv` | **29.851** | **24.658 (`82,60%`)** | **2.008 (`6,73%`)** | **3.185 (`10,67%`)** |

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

## 3. Phương pháp Mô hình hóa & Tinh chỉnh Siêu tham số

### 3.1. Mô hình Baseline: TF-IDF kết hợp LinearSVC có Hiệu chuẩn Xác suất
- **Trích xuất đặc trưng (`FeatureUnion`):** Kết hợp đồng thời hai nhánh TF-IDF:
  - **Word n-gram:** Bắt từ đơn, cụm 2 từ và cụm 3 từ đã tách (`ngram_range=(1, 3)`, `min_df=2`, `max_features=45.000`, `sublinear_tf=True`).
  - **Character n-gram (`char_wb`):** Bắt chuỗi ký tự con từ 2 đến 5 ký tự trong phạm vi từ (`ngram_range=(2, 5)`, `min_df=3`, `max_features=45.000`, `sublinear_tf=True`), giúp mô hình nhận diện tốt các từ viết sai chính tả nhẹ hoặc biến thể từ tục chưa có trong từ điển.
- **Lý do dùng `CalibratedClassifierCV(cv=5)`:** Mô hình `LinearSVC` phân lớp rất mạnh trên không gian TF-IDF chiều cao nhưng bản chất chỉ xuất ra khoảng cách tới siêu phẳng (`decision_function`), không có hàm `predict_proba()`. Việc bọc `CalibratedClassifierCV(LinearSVC, cv=5, method="sigmoid")` (Platt scaling 5-fold ngay trên tập Train) giúp mô hình xuất ra xác suất thực sự cho từng nhãn `P(CLEAN)`, `P(OFFENSIVE)`, `P(HATE)` và tính được `diem_toxic = P(OFFENSIVE) + P(HATE)`.

### 3.2. Kết quả 76 thử nghiệm tinh chỉnh trên tập Validation (`outputs/thu_nghiem_baseline.csv`)
Để tìm cấu hình tốt nhất mà **tuyệt đối không chạm vào tập Test**, kịch bản `scripts/tune_baseline.py` đã thực hiện **76 lần thử nghiệm** trên tập Validation (`2.985` mẫu) theo 4 nhóm:
1. **Nhóm 1 — Mốc chuẩn `CalibratedClassifierCV(LinearSVC, cv=5)`** với các hệ số phạt `C` từ `0.05` đến `1.0` và `class_weight` (`None`, `balanced`).
2. **Nhóm 2 — Đối chiếu với `LogisticRegression`** (`C` từ `0.5` đến `5.0`, `class_weight="balanced"` và không trọng số).
3. **Nhóm 3 — Tinh chỉnh trọng số lớp (`class_weight = {0: 1.0, 1: w1, 2: w2}`)** và **ngưỡng quyết định riêng cho lớp `OFFENSIVE`** (`thr_off` từ `0.20` đến `0.45`).
4. **Nhóm 4 — Mở rộng đặc trưng n-gram (`word (1,3) + char_wb (2,5)`) kết hợp trọng số lớp và ngưỡng.**

**Bảng trích xuất các thử nghiệm tiêu biểu trên tập Validation (`2.985` mẫu):**

| Thứ hạng | Tên thử nghiệm (`experiment_name`) | Nhóm mô hình | Accuracy Val | F1-macro Val | F1 `CLEAN` | F1 `OFFENSIVE` | F1 `HATE` |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1 (Tốt nhất)** | `CalibratedSVC_C0.1_sublinear_w13_c25_w_1_3.0_2.0` | SVC Calibrated + N-gram (1,3)+(2,5) + Trọng số `{0:1, 1:3, 2:2}` | **87,64%** | **68,19%** | **93,97%** | **47,03%** | **63,56%** |
| **2** | `CalibratedSVC_C0.1_sublinear_w13_c25_w_1_3.0_2.0_thrOff_0.4` | SVC Calibrated + Ngưỡng `thr_off=0.40` | 87,27% | 68,10% | 93,86% | 46,61% | 63,84% |
| **3** | `CalibratedSVC_C0.15_sublinear_w13_c25_w_1_3.5_2.0_thrOff_0.45` | SVC Calibrated (`C=0.15`, `w={0:1, 1:3.5, 2:2}`) | 87,40% | 67,84% | 93,86% | 46,65% | 63,02% |
| **17** | `CalibratedSVC_C0.2_bal` | SVC Calibrated chuẩn (`C=0.2, balanced, w12+c25`) | 88,07% | 67,05% | 94,14% | 41,72% | 65,30% |
| **31** | `LogReg_C2.0_none` | Logistic Regression (`C=2.0`, không trọng số) | 88,21% | 66,35% | 94,25% | 41,08% | 63,72% |
| **Mốc cũ** | `UnCalibrated_LinearSVC_C0.1_bal` (`run_baseline.py`) | LinearSVC chưa hiệu chuẩn (`C=0.1, balanced`) | 87,20% | 66,05% | 93,96% | 41,51% | 62,67% |
| **65** | `LogReg_C1.0_bal` | Logistic Regression (`C=1.0, balanced`) | 83,35% | 63,34% | 91,63% | 37,52% | 60,87% |

**Nhận xét lựa chọn mô hình:**
- Cấu hình **`CalibratedSVC_C0.1_sublinear_w13_c25_w_1_3.0_2.0`** đạt **F1-macro cao nhất trên Validation (`68,19%`)**, cải thiện **`+2,14` điểm % F1-macro** và **`+5,52` điểm % F1 trên nhãn `OFFENSIVE`** (từ `41,51%` lên `47,03%`) so với baseline gốc nhờ gán trọng số mẫu `{0: 1.0, 1: 3.0, 2: 2.0}` vừa đủ trước khi đưa vào `CalibratedClassifierCV`.
- Mô hình này được lưu chính thức tại `models/baseline_calibrated.pkl` (giữ nguyên `models/baseline_LinearSVC_C0.1.pkl` để đối chiếu).
- **Lưu ý phương pháp luận:** Vì 76 lần thử nghiệm đều được so sánh trên tập Validation nên con số `68,19%` có thể hơi lạc quan nhẹ trên Val; do đó năng lực thực tế của mô hình được chốt bằng lần chấm duy nhất trên tập **Test** ở Mục 4.

### 3.3. Mô hình PhoBERT Fine-tune (`scripts/run_phobert.py` & `notebooks/03_phobert_colab.ipynb`)
- **Kiến trúc:** Fine-tune mô hình `vinai/phobert-base-v2` với tầng phân loại 3 lớp (`AutoModelForSequenceClassification`), độ dài tối đa `max_length = 128` token sau tách từ `pyvi`.
- **Hàm mất mát có trọng số (`WeightedCrossEntropyTrainer`):** Để chống lệch nhãn, hàm `CrossEntropyLoss` được truyền vector trọng số lớp nghịch đảo tần suất xuất hiện trên tập Train (`w_c = N / (3 * N_c)`), kết hợp `fp16=True`, `learning_rate=2e-5`, `weight_decay=0.01`, `4 epochs` và chọn checkpoint tốt nhất theo `eval_f1_macro`.
- 📝 **`[CẦN NHÓM ĐỌC LẠI / BỔ SUNG KHI CÓ PHOBERT]`:** Sau khi chạy xong `notebooks/03_phobert_colab.ipynb` trên Google Colab GPU và giải nén về `models/phobert/best/`, nhóm chạy `python scripts/evaluate_final.py` và điền bổ sung chỉ số F1-macro Val / Test của PhoBERT vào đây.

---

## 4. Kết quả Đánh giá Chính thức trên Tập Test (`2.986` mẫu)

Kịch bản `scripts/evaluate_final.py` chấm điểm tập `data/processed/03_test.csv` (`2.986` mẫu độc lập hoàn toàn) đúng 1 lần duy nhất và xuất ra `outputs/bang_so_sanh_mo_hinh.csv`:

### 4.1. Bảng tổng hợp các chỉ số trên tập Test
| Mô hình | Accuracy (%) | F1-macro (%) | Toxic Precision (%) | Toxic Recall (%) | F1 `CLEAN` (%) | F1 `OFFENSIVE` (%) | F1 `HATE` (%) | Độ trễ (ms/câu) | Kích thước (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LinearSVC (Calibrated)** | **87,74%** | **67,03%** | **74,59%** | **68,24%** | **94,13%** | **41,36%** | **65,61%** | **5,89 ms** | **9,42 MB** |
| *PhoBERT (fine-tune)* | *`[Chờ Colab]`* | *`[Chờ Colab]`* | *`[Chờ Colab]`* | *`[Chờ Colab]`* | *`[Chờ Colab]`* | *`[Chờ Colab]`* | *`[Chờ Colab]`* | *`[Chờ Colab]`* | *`[Chờ Colab]`* |

*(Trong đó **Toxic Precision** và **Toxic Recall** đo hiệu năng phát hiện bình luận độc hại gộp chung hai lớp `{1: OFFENSIVE, 2: HATE}` so với lớp `{0: CLEAN}`).*

### 4.2. Phân tích Ma trận nhầm lẫn trên tập Test (`outputs/ma_tran_nham_lan_TF_IDF_LinearSVC_Calibrated.png`)
Trên tổng số **2.986** mẫu của tập Test:
- **Lớp `0 — Bình thường` (2.466 mẫu thật):** Mô hình đoán đúng **2.343 mẫu (`95,01%`)**, báo nhầm sang `Xúc phạm` 57 mẫu và sang `Thù ghét` 66 mẫu (tổng cộng **123 ca Báo nhầm — False Positive**, chiếm `4,99%` số câu bình thường).
- **Lớp `1 — Xúc phạm` (201 mẫu thật):** Mô hình đoán đúng **73 mẫu (`36,32%`)**, bỏ sót sang `Bình thường` 91 mẫu (`45,27%`) và nhầm mức độ sang `Thù ghét` 37 mẫu (`18,41%`).
- **Lớp `2 — Thù ghét` (319 mẫu thật):** Mô hình đoán đúng **204 mẫu (`63,95%`)**, nhầm mức độ sang `Xúc phạm` 38 mẫu (`11,91%`) và bỏ sót sang `Bình thường` 77 mẫu (`24,14%`).
- **Tổng số ca Bỏ sót độc hại (False Negative — thật là 1 hoặc 2 nhưng đoán là 0):** **168 mẫu** trên tổng số 520 mẫu độc hại (tức bắt trúng **352/520 = `67,69%`** bình luận độc hại; nếu tính macro recall trên 2 lớp độc hại đạt **`68,24%`**).

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
| **Câu không chứa các dấu hiệu nhiễu** | 1.991 | 12,10% | 87,90% | 68,07% |
| **Câu có chữ kéo dài** | 102 | **7,84%** | **92,16%** | **69,20%** |
| **Câu có emoji** | 403 | 9,43% | 90,57% | 62,37% |
| **Câu có teencode / viết tắt** | 596 | **15,60%** | 84,40% | 61,69% |
| **Câu chèn ký tự / dấu `*` né lọc** | 19 | **36,84%** | 63,16% | 59,00% |

**Nhận xét:**
- Nhờ hàm `fix_keo_dai` trong `src/preprocess.py`, các câu có chữ kéo dài được chuẩn hóa rất tốt và đạt tỷ lệ lỗi thấp nhất (`7,84%`).
- Các câu chứa **teencode (`15,60%` lỗi)** và đặc biệt là **câu chèn ký tự / dấu `*` né bộ lọc (`36,84%` lỗi)** vẫn là nhóm khó nhất vì người dùng tạo ra các biến thể viết tắt ngoài danh sách 42 từ hoặc dùng dấu `*` che mất nguyên âm (`đ*o`, `đ*t`) khiến từ vựng lệch khỏi từ điển chuẩn.

### 5.3. Phân tích nguyên nhân lớp `OFFENSIVE` đạt F1 thấp (`41,36%`) & Top 15 ca đoán sai (`outputs/top15_offensive_doan_sai.csv`)
Việc lớp `OFFENSIVE` có F1 dưới `0,50` (trên Val đạt `47,03%`, trên Test đạt `41,36%`) xuất phát từ **3 hạn chế bản chất của dữ liệu**:
1. **Thiếu mẫu huấn luyện nghiêm trọng:** Lớp `OFFENSIVE` chỉ chiếm **`6,73%`** bộ dữ liệu (`1.606` mẫu Train, `201` mẫu Test).
2. **Định nghĩa nhãn nằm ở vùng xám:** Nhiều câu trong `outputs/top15_offensive_doan_sai.csv` dùng từ ngữ hoàn toàn thông thường để mỉa mai hoặc nói trại âm:
   - Ví dụ 1: *"Trường hợp này ở địa phương khác kêu là gì ko biết còn địa phương mình gọi là khùng á"* -> Nhãn thật: `OFFENSIVE`, mô hình đoán: `Bình thường` (`P = 0,8769`) vì cả câu chỉ có từ *"khùng"* nằm cuối câu điều kiện dài.
   - Ví dụ 2: *"Này khẩu phật tâm xà"* hoặc *"Cứ tưởng thế là ngầu"* -> Nhãn thật: `OFFENSIVE`, mô hình đoán: `Bình thường` (`P = 0,8882` và `0,8746`) vì không hề có từ tục, chỉ mang sắc thái chê bai ngầm.
   - Ví dụ 3: *"Đ*o hiểu lun"* và *"Ghi thêm chữ đ*t chị m nx :))* -> Nhãn thật: `OFFENSIVE`, mô hình đoán: `Bình thường` (`P = 0,8696` và `0,8613`) do người viết dùng dấu `*` và viết tắt biến thể (`lun`, `nx`).
3. **Bất đồng nhãn từ người gán nhãn (Annotator Disagreement):** Ngay trong `outputs/phan_tich_loi.csv`, nhiều câu có nhãn gốc là `Bình thường (0)` nhưng chứa từ chửi thề rất nặng (ví dụ: *"Đụ mẹ tay ghi , não đéo kịp suy nghĩ..."* hoặc *"Má dơ vl =)))"*), khiến mô hình dự đoán là `Thù ghét / Xúc phạm` với độ tự tin `> 99%` và bị tính là lỗi False Positive.

📝 **`[CẦN NHÓM ĐỌC LẠI & BỔ SUNG NHẬN XÉT]`:** Nhóm có thể chọn thêm 3–5 câu trong file `outputs/phan_tich_loi.csv` để điền cột `nhận_xét_của_nhóm` và trích dẫn trực tiếp vào slide báo cáo.

---

## 6. Kiến trúc Ứng dụng Web Streamlit & Tích hợp YouTube API

Ứng dụng Web được thiết kế theo kiến trúc tách biệt hoàn toàn giữa **Tầng giao diện (`app/`)** và **Tầng suy luận lõi (`src/`)**:
- **`src/inference.py`:** Cung cấp hợp đồng hàm chuẩn (`doc_cau_hinh`, `tai_mo_hinh`, `phan_loai`, `phan_loai_hang_loat`, `tim_tu_gay_doc`). Hàm `tim_tu_gay_doc` sử dụng kỹ thuật **Leave-One-Token-Out (LOTO)** kết hợp độ giảm xác suất và độ giảm logit để tô đỏ chính xác những từ làm tăng điểm độc hại trong câu.
- **`src/youtube.py`:** Kết nối trực tiếp `commentThreads.list` của YouTube Data API v3, giới hạn an toàn tối đa 500 bình luận, xử lý đầy đủ các mã lỗi tiếng Việt (`commentsDisabled`, `quotaExceeded`, `keyInvalid`, `videoNotFound`) và tuyệt đối không lưu tên người bình luận hay API key.
- **6 trang giao diện (`app/Home.py` & `app/pages/1..5`):** Đã vượt qua toàn bộ bộ kiểm thử tự động `pytest` (`tests/test_inference.py`) ở cả 3 trạng thái: đủ dữ liệu, thiếu thư mục `outputs/`, và thiếu thư mục `models/` mà không phát sinh bất kỳ lỗi dừng chương trình nào.

---

## 7. Hạn chế Đạo đức AI & Hướng Phát triển

1. **Nguyên tắc Con người làm trung tâm (*Human-in-the-loop*):** Với tỷ lệ Báo nhầm (FP) khoảng `4,99%` trên các câu bình thường (nhất là các câu dùng từ lóng/tục để đùa vui, khen ngợi bạn bè như *"cá kho... ngon vailone :)"*), hệ thống **chỉ được sử dụng làm công cụ sàng lọc và xếp hạng ưu tiên** cho kiểm duyệt viên, tuyệt đối không tự động xóa bình luận hay khóa tài khoản người dùng.
2. **Bảo vệ quyền riêng tư:** Không thu thập, lưu trữ hay hiển thị định danh cá nhân của người bình luận.
3. **Hướng phát triển tiếp theo:**
   - Hoàn tất huấn luyện và tích hợp mô hình **PhoBERT (`vinai/phobert-base-v2`)** để nắm bắt ngữ cảnh hai chiều và cải thiện khả năng nhận diện các câu mỉa mai, nói giảm nói tránh ở lớp `OFFENSIVE`.
   - Bổ sung ngữ cảnh tiêu đề bài viết / video và bình luận gốc (parent comment) vào đầu vào của mô hình.
   - Mở rộng bộ chuẩn hóa từ lóng, từ viết trại âm và ký tự che chữ (`*`) bằng mô hình ngôn ngữ mức ký tự.
