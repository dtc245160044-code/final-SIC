# 🎤 KỊCH BẢN THUYẾT TRÌNH DEMO 5 PHÚT & CẨM NANG PHẢN BIỆN HỘI ĐỒNG

Tài liệu này giúp nhóm tự tin trình bày đồ án trong **5 phút** trên ứng dụng Web Streamlit (`streamlit run app/Home.py`), kèm theo **4 câu demo trích nguyên văn từ tập Test (`data/processed/03_test.csv`)**, **10 câu hỏi hội đồng thường gặp** và **Danh sách kiểm tra trước giờ thi**.

---

## ⏱️ PHẦN 1: KỊCH BẢN TRÌNH BÀY 5 PHÚT TRÊN WEB

### 🔹 Phút 0:00 – 0:45 | Mở đầu tại Trang 0 (`Home.py` — Tổng quan dự án)
- **Thao tác trên màn hình:** Mở trang chủ **🛡️ Tổng quan dự án**. Chỉ vào 4 thẻ KPI và sơ đồ quy trình.
- **Lời thoại gợi ý:**
  > *"Xin chào Hội đồng. Nhóm chúng em thực hiện đề tài **Phát hiện bình luận độc hại tiếng Việt** trên bộ dữ liệu chuẩn **ViHSD** gồm 3 nhãn: **Bình thường (0)**, **Xúc phạm (1)** và **Thù ghét (2)**.*  
  > *Từ 33.397 dòng thô ban đầu, sau khi làm sạch và khử trùng lặp nội dung, bộ dữ liệu gồm **29.851 bình luận duy nhất**, chia phân tầng 80/10/10 (23.880 Train, 2.985 Val, 2.986 Test). Điểm thách thức lớn nhất của dữ liệu là nhãn **Bình thường chiếm tới 82,60%**, trong khi nhãn **Xúc phạm chỉ chiếm 6,73%**.*  
  > *Trên tập Test độc lập (chấm đúng 1 lần duy nhất), mô hình **TF-IDF + LinearSVC (Calibrated)** đạt **Accuracy 87,74%** và **F1-macro 67,03%** với tốc độ suy luận chỉ khoảng **5,89 ms/bình luận**."*

---

### 🔹 Phút 0:45 – 1:30 | Quy trình Làm sạch Tiếng Việt tại Trang 2 (`2_Tien_xu_ly.py`)
- **Thao tác trên màn hình:** Chuyển sang trang **🧹 Tiền xử lý tiếng Việt**, bấm **Chạy chuẩn hóa** trên câu thật trích từ ViHSD để hiện khung đối chiếu Trước / Sau và bảng 8 câu thật từ `outputs/truoc_sau_xu_ly.csv`.
- **Lời thoại gợi ý:**
  > *"Vì bình luận mạng xã hội Việt Nam chứa rất nhiều nhiễu — cụ thể trên tập Train có **19,36% câu chứa teencode** và **13,50% câu chứa emoji** — nhóm xây dựng hàm `lam_sach` và `tach_tu` dùng chung cho cả lúc huấn luyện lẫn lúc chạy Web: tự động xóa link/emoji, rút gọn chữ kéo dài, ghép chữ bị chèn dấu chấm/gạch ngang né bộ lọc, dịch 42 từ teencode phổ biến và tách từ ghép bằng `pyvi`."*

---

### 🔹 Phút 1:30 – 3:15 | Điểm nhấn Demo trực tiếp 4 Câu Thật từ Tập Test tại Trang 1 (`1_Thu_nghiem.py`)
- **Thao tác trên màn hình:** Mở trang **🧪 Thử nghiệm**. Lần lượt bấm **4 nút mẫu** (đều là câu trích nguyên văn từ `data/processed/03_test.csv`) rồi bấm **Phân tích** để hội đồng thấy cả **điểm mạnh** lẫn **hạn chế thật** của mô hình:

#### ✅ Câu Demo 1: Bình luận khen ngợi có dấu chấm kéo dài & emoji (Mô hình đoán ĐÚNG)
- **Câu nhập (bấm nút `[Mẫu 1: Khen ngợi (Test #CLEAN)]`):**
  ```text
  Bác Trần Dần trả lời câu hỏi cuối rất hay....rất sáng suốt 👍
  ```
- **Kết quả mô hình chạy thật:**
  - **Nhãn thật trên Test:** `Bình thường (0)` ➔ **Mô hình dự đoán:** `🟢 Bình thường` (Độ tin cậy: **`95,31%`**, Điểm độc hại: **`4,69%`**).
  - **Sau tiền xử lý:** `bác trần dần trả_lời câu hỏi cuối rất hay ... . rất sáng_suốt` (đã tự động xóa emoji `👍` và nối từ ghép `trả_lời`, `sáng_suốt`).

#### ✅ Câu Demo 2: Câu có Teencode công kích (Mô hình đoán ĐÚNG & Tô đỏ từ gây độc)
- **Câu nhập (bấm nút `[Mẫu 2: Teencode công kích (Test #HATE)]`):**
  ```text
  Bán hàng đàng hoàng phải ko, bán hàng lương thiện đúng không, dám cho đối chứng ko, công an cồng đầu chứ ở đó mà xạo xạo, qua mặt thiên hạ hả bưởi
  ```
- **Kết quả mô hình chạy thật:**
  - **Sau tiền xử lý:** Các từ viết tắt `ko` được chuẩn hóa thành `không`, tách từ ghép `đàng_hoàng`, `lương_thiện`, `đối_chứng`, `công_an`, `qua_mặt`, `thiên_hạ`.
  - **Nhãn thật trên Test:** `Thù ghét (2)` ➔ **Mô hình dự đoán:** `🔴 Thù ghét` (Độ tin cậy: **`78,34%`**, **Điểm độc hại tổng hợp = `91,25%`**).
  - **Từ bị tô đỏ (Leave-One-Token-Out):** `công_an`, `đầu`, `xạo`.

#### ⚠️ Câu Demo 3: Câu Mỉa mai / nói giảm nói tránh ở lớp `OFFENSIVE` (Mô hình BỎ SÓT — False Negative)
- **Câu nhập (bấm nút `[Mẫu 3: Mỉa mai ngầm (Test #OFFENSIVE)]` — trích từ `outputs/top15_offensive_doan_sai.csv`):**
  ```text
  Trường hợp này ở địa phương khác kêu là gì ko biết còn địa phương mình gọi là khùng á
  ```
- **Kết quả mô hình chạy thật:**
  - **Nhãn thật trên Test:** `Xúc phạm (1)` ➔ **Mô hình dự đoán:** `🟢 Bình thường` (Xác suất Bình thường: **`87,69%`**, Điểm độc hại: **`12,31%`**).
  - **Điểm nhấn thuyết trình:** Giải thích thẳng thắn với Hội đồng: *"Câu này trong tập Test có nhãn thật là OFFENSIVE, nhưng người viết dùng cách nói vòng vo, chỉ cài chữ 'khùng á' ở cuối câu nên mô hình TF-IDF bị bỏ sót sang Bình thường. Đây là minh họa trực tiếp vì sao lớp OFFENSIVE chỉ đạt F1 41,36% trên tập Test."*

#### ⚠️ Câu Demo 4: Câu dùng từ tục để khen món ăn (Mô hình BÁO NHẦM — False Positive)
- **Câu nhập (bấm nút `[Mẫu 4: Từ tục khen đồ ăn (Test #FP)]` — trích từ `outputs/phan_tich_loi.csv`):**
  ```text
  Địt mẹ, ông thử ăn cá kho đi, càng hâm lại nó ăn lại càng ngon vailone :)
  ```
- **Kết quả mô hình chạy thật:**
  - **Nhãn thật trên Test:** `Bình thường (0)` (người viết đang khen món cá kho ngon).
  - **Mô hình dự đoán:** `🟠 Xúc phạm (OFFENSIVE)` (**Điểm độc hại = `98,52%`**) và tô đỏ các từ `địt_mẹ`, `vailone`.
  - **Điểm nhấn thuyết trình:** *"Đây là ca Báo nhầm thực tế từ tập Test: khi người dùng dùng từ cảm thán suồng sã để khen món ăn, mô hình bị đánh lừa bởi từ khóa nhạy cảm."*

---

### 🔹 Phút 3:15 – 4:15 | Kết quả Đánh giá Chính thức & Phân tích Lỗi tại Trang 3 (`3_Danh_gia_mo_hinh.py`)
- **Thao tác trên màn hình:** Mở trang **📈 Đánh giá mô hình**, lướt qua 3 tab: **So sánh mô hình** -> **Ma trận nhầm lẫn** -> **Phân tích lỗi (FP/FN)** và mở 3 hộp phân tích chuyên sâu ở cuối tab 3.
- **Lời thoại gợi ý:**
  > *"Nhóm tuân thủ nghiêm ngặt nguyên tắc chống rò rỉ dữ liệu: 76 lần thử nghiệm tinh chỉnh chỉ thực hiện trên tập Validation (`F1-macro Val = 68,19%`). Tập Test được chấm đúng 1 lần duy nhất làm kết quả chính thức: đạt **Accuracy 87,74%**, **F1-macro 67,03%**, trong đó **F1 lớp CLEAN đạt 94,13%**, **HATE đạt 65,61%**, và **OFFENSIVE đạt 41,36%** (Validation cao hơn Test 1,16% do đã tinh chỉnh 76 lần trên Val).*  
  > *Qua phân tích chuyên sâu trên tập Test, nhóm phát hiện: câu càng dài (>25 từ) thì F1 lớp HATE càng cao (78,64%) nhưng F1 lớp OFFENSIVE giảm còn 13,33%; đồng thời các câu chèn dấu `*` né lọc có tỷ lệ lỗi cao nhất (36,84%)."*

---

### 🔹 Phút 4:15 – 5:00 | Quét Bình luận YouTube & Đạo đức AI (`4_YouTube.py` & `5_Han_che_dao_duc.py`)
- **Thao tác trên màn hình:** Mở trang **📺 Phân loại YouTube**, dán link video + API key (hoặc chuyển sang chế độ dự phòng đọc file `outputs/ket_qua_youtube.csv` đã được nhóm chạy trước bằng `python scripts/lay_youtube_demo.py`). Kết thúc ở trang **⚖️ Hạn chế & Đạo đức**.
- **Lời thoại gợi ý:**
  > *"Ứng dụng tích hợp YouTube Data API v3 cho phép lấy bình luận công khai của một video bất kỳ, phân loại và lưu kèm `video_id` và thời점 lấy (`ngay_lay`). Để bảo vệ quyền riêng tư và đạo đức AI, hệ thống tuyệt đối **không lưu tên người bình luận**, **không lưu API key**, và chỉ đóng vai trò công cụ gợi ý sàng lọc chứ không tự động xóa bình luận hay khóa tài khoản."*

---

## ❓ PHẦN 2: 10 CÂU HỎI HỘI ĐỒNG HAY HỎI & GỢI Ý TRẢ LỜI

### Câu 1: Tại sao nhóm dùng F1-macro làm thước đo chính để chọn mô hình mà không dùng Accuracy?
- **Trả lời:** Vì bộ dữ liệu ViHSD bị lệch nhãn rất nặng: nhãn `0 (Bình thường)` chiếm tới **`82,60%`**, trong khi nhãn `1 (Xúc phạm)` chỉ chiếm **`6,73%`** và nhãn `2 (Thù ghét)` chiếm **`10,67%`**. Nếu chỉ nhìn vào Accuracy, một mô hình đoán tất cả mọi câu là "Bình thường" cũng đã đạt Accuracy `82,60%` nhưng không bắt được bất kỳ bình luận độc hại nào (`F1-macro = 30,15%`). Chỉ số **F1-macro** tính trung bình cộng không trọng số F1 của cả 3 lớp, buộc mô hình phải làm tốt trên cả hai lớp thiểu số (`OFFENSIVE` và `HATE`).

### Câu 2: `LinearSVC` không có hàm `predict_proba()`, vậy làm sao nhóm lấy được xác suất từng nhãn và Điểm độc hại trên Web?
- **Trả lời:** Nhóm sử dụng kỹ thuật hiệu chuẩn xác suất **`CalibratedClassifierCV(LinearSVC(C=0.1), cv=5, method='sigmoid')`** (Platt Scaling 5-fold cross-validation chỉ fit trên tập Train). Kỹ thuật này khớp một hàm Logistic trên điểm `decision_function` của SVM, giúp xuất ra phân phối xác suất chuẩn hóa trên 3 nhãn. Khi đó, **Điểm độc hại (`diem_toxic`)** được tính bằng tổng xác suất của 2 nhãn độc hại: `P(Xúc phạm) + P(Thù ghét)`.

### Câu 3: Tại sao F1 trên tập Validation là `68,19%` (lớp `OFFENSIVE` là `47,03%`) nhưng trên tập Test lại là `67,03%` (lớp `OFFENSIVE` là `41,36%`)?
- **Trả lời:** Vì trên tập Validation, nhóm đã chạy **76 lần thử nghiệm tinh chỉnh** (`outputs/thu_nghiem_baseline.csv`) gồm chọn trọng số lớp `{0: 1.0, 1: 3.0, 2: 2.0}`, ngưỡng quyết định và n-gram `word (1,3) + char_wb (2,5)`. Việc chọn mô hình tốt nhất qua 76 lần thử trên cùng tập Val tạo ra độ lạc quan nhẹ trên Val. Khi đánh giá đúng 1 lần duy nhất trên tập **Test độc lập (`2.986` mẫu)**, F1-macro thực tế là **`67,03%`** và F1 lớp `OFFENSIVE` là **`41,36%`** (trong khi F1 lớp `HATE` trên Test đạt **`65,61%`**, cao hơn trên Val là `63,56%`). Nhóm lấy số liệu trên tập Test làm kết quả chính thức.

### Câu 4: Làm sao nhóm tính và tô đỏ được "từ gây độc" trên giao diện Web đối với cả SVM và PhoBERT?
- **Trả lời:** Nhóm cài đặt hàm `tim_tu_gay_doc()` trong `src/inference.py` theo phương pháp **Leave-One-Token-Out (LOTO)**: với một câu đã tách từ gồm $N$ từ, hệ thống tạo $N$ biến thể câu mà mỗi biến thể lược bỏ đi đúng 1 từ, sau đó chạy mô hình để tính độ sụt giảm của Điểm độc hại (`do_giam_prob`) và độ sụt giảm Logit (`do_giam_logit`). Từ nào khi bỏ đi làm điểm độc hại giảm từ `0.03` trở lên (hoặc logit giảm `>= 0.35` đối với câu có nhiều từ chửi làm bão hòa xác suất ở `0.9999`) sẽ được tô đỏ.

### Câu 5: Tại sao phải dùng `pyvi` để tách từ tiếng Việt? Code PhoBERT của nhóm dùng bản nào?
- **Trả lời:** Trong tiếng Việt, đơn vị từ thường gồm 2 âm tiết trở lên (ví dụ: `"xúc_phạm"`, `"thất_học"`). Nếu tách theo khoảng trắng đơn thuần, từ `"thất_học"` bị xé thành hai âm tiết rời `"thất"` và `"học"`, làm nhiễu biểu diễn TF-IDF. Hơn nữa, trong `scripts/run_phobert.py` và `notebooks/03_phobert_colab.ipynb`, nhóm sử dụng mô hình **`vinai/phobert-base`** — vốn được huấn luyện tiền kỳ trên văn bản tiếng Việt đã tách từ bằng dấu gạch dưới `_` (`pyvi`).

### Câu 6: Trong hàm `tach_tu()` của `src/preprocess.py`, tại sao lại phải thay dấu `*` bằng `X_STAR_X` trước khi gọi `ViTokenizer.tokenize`?
- **Trả lời:** Nhiều bình luận lách luật dùng dấu `*` giữa từ (như `đ*o`, `đ*t`). Bộ tách từ `ViTokenizer` mặc định coi dấu `*` là dấu câu và tách rời thành 3 token `đ * o`. Việc tạm thời đổi `*` thành `X_STAR_X` giúp `ViTokenizer` giữ nguyên cả cụm từ `đ*o` thành một token duy nhất để đặc trưng `char_wb (2,5)` vẫn bắt được tiền tố/hậu tố của từ đó.

### Câu 7: Nhóm đảm bảo không bị rò rỉ dữ liệu (Data Leakage) giữa Train, Val và Test bằng cách nào?
- **Trả lời:**
  1. Ở bước tiền xử lý (`scripts/process_data_fixed.py`), nhóm khử toàn bộ các câu trùng nhau theo cột `text_clean` trước khi chia tập; kiểm tra bằng `scripts/check_data.py` xác nhận giao nhau giữa Train, Val và Test bằng `0`.
  2. Bộ biến đổi `TfidfVectorizer` và bộ hiệu chuẩn `CalibratedClassifierCV(cv=5)` chỉ được `.fit()` trên `03_train.csv`.
  3. Toàn bộ 76 lần thử nghiệm chọn mô hình chỉ chấm trên `03_val.csv`. Tập `03_test.csv` chỉ được mở ra chấm đúng 1 lần ở `scripts/evaluate_final.py`.

### Câu 8: Khi người dùng dán link YouTube vào Trang 4, nếu video tắt bình luận, sai API key hoặc mất mạng thì ứng dụng xử lý thế nào?
- **Trả lời:** Trong `src/youtube.py`, nhóm bắt riêng từng mã lỗi của YouTube Data API v3 (`commentsDisabled`, `quotaExceeded`, `keyInvalid`, `videoNotFound`, `requests.RequestException`) và hiển thị thông báo tiếng Việt rõ ràng hướng dẫn cách xử lý, tuyệt đối không để văng lỗi kỹ thuật (traceback) ra màn hình. Ngoài ra, nhóm có script `scripts/lay_youtube_demo.py` để lấy trước 100–200 bình luận YouTube thật bằng API key của nhóm (lưu kèm `video_id` và `ngay_lay` vào `outputs/ket_qua_youtube.csv`) dùng làm chế độ dự phòng khi phòng thi mất mạng; nếu chưa chạy script này thì chế độ dự phòng hiện rõ *"Chưa có kết quả"*.

### Câu 9: Tại sao nhóm tách `requirements.txt` và `requirements-train.txt`?
- **Trả lời:** Để chạy ứng dụng Web Streamlit và mô hình Baseline TF-IDF, chỉ cần các thư viện nhẹ trong `requirements.txt` (cài đặt rất nhanh và ghim đúng `scikit-learn==1.9.1` để mở file `.pkl` an toàn). Trong khi đó, các gói huấn luyện học sâu nặng hàng GB (`torch`, `transformers`, `datasets`, `accelerate`) được tách sang `requirements-train.txt` phục vụ khi chạy PhoBERT.

### Câu 10: Hệ thống này có thể áp dụng để tự động xóa bình luận ngay lập tức trên mạng xã hội không?
- **Trả lời:** **Không nên tự động xóa hoàn toàn.** Như nhóm đã trình bày ở Trang 5 (Hạn chế & Đạo đức), trên tập Test mô hình vẫn có tỷ lệ Báo nhầm (False Positive) khoảng `4,99%` ở các câu dùng từ lóng/tục để trêu đùa bạn bè hoặc khen ngợi. Nếu tự động xóa sẽ xâm phạm quyền tự do biểu đạt của người dùng. Giải pháp đúng đắn là mô hình **Human-in-the-loop**: dùng Điểm độc hại (`diem_toxic`) để lọc và đẩy các bình luận có nguy cơ cao lên đầu danh sách cho kiểm duyệt viên xác nhận.

---

## ✅ PHẦN 3: DANH SÁCH KIỂM TRA TRƯỚC GIỜ THI (10 MỤC — MỤC 7 BẢN VẼ)

Trước khi bước vào phòng bảo vệ 15 phút, nhóm hãy tích kiểm tra đủ 10 mục sau:

- [x] **1. Chạy thử toàn bộ trên máy sẽ mang đi thi:** Mở PowerShell tại thư mục dự án, chạy `streamlit run app/Home.py`, bấm qua cả **6 trang** (Tổng quan, Thử nghiệm, Tiền xử lý, Đánh giá mô hình, YouTube, Hạn chế & đạo đức) không trang nào báo lỗi đỏ.
- [x] **2. Kiểm tra bộ test tự động:** Chạy `python -m pytest tests/test_inference.py -v` đảm bảo xanh `4 passed`.
- [x] **3. Thử nghiệm 4 câu mẫu thật từ tập Test trên Trang 1:** Bấm cả 4 nút mẫu (câu khen ngợi, câu teencode công kích, câu mỉa mai FN, câu từ tục khen món ăn FP), kiểm tra thẻ kết quả hiện đúng màu, thanh xác suất cộng lại bằng `100%` và các từ gây độc được tô đỏ.
- [x] **4. Thử các trường hợp biên trên Trang 1:** Bấm Phân tích khi ô trống, nhập toàn emoji `😂😂😂` — ứng dụng hiện cảnh báo nhẹ nhàng, không sập.
- [ ] **5. Tạo file dự phòng YouTube thật bằng API key của nhóm:** Trước buổi thi, chạy `python scripts/lay_youtube_demo.py`, nhập link video YouTube công khai + API key của nhóm để tạo `outputs/ket_qua_youtube.csv` (có `video_id` và `ngay_lay` thật) phòng trường hợp phòng thi mất Internet.
- [x] **6. Kiểm tra đồng nhất số liệu chính thức (tập Test):** Số liệu Accuracy (`87,74%`), F1-macro (`67,03%`), F1 OFFENSIVE (`41,36%`) trên Trang 0 (`Home.py`), Trang 3 (`3_Danh_gia_mo_hinh.py`), `README.md` và `reports/bao_cao_du_an.md` khớp nhau 100% (vì đều đọc từ `outputs/bang_so_sanh_mo_hinh.csv`).
- [x] **7. Kiểm tra bảo mật:** Không có mã API key YouTube nào nằm trong code hay trong `git log`.
- [x] **8. Kiểm tra phiên bản thư viện:** Máy thuyết trình đã cài đúng `scikit-learn==1.9.1` trong `requirements.txt` để mở `models/baseline_calibrated.pkl` mượt mà.
- [x] **9. Chuẩn bị sẵn cửa sổ trình duyệt:** Mở sẵn tab `http://localhost:8501` ở mức thu phóng (zoom) `100%` hoặc `110%`, chế độ giao diện sáng (Light mode) để máy chiếu hiển thị rõ nét nhất.
- [x] **10. Phân công thành viên:** Thống nhất rõ ai trình bày phần Dữ liệu & Tiền xử lý, ai demo trực tiếp trên Web, và ai trả lời câu hỏi về Mô hình & Phân tích lỗi.
