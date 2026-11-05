# 🎤 KỊCH BẢN THUYẾT TRÌNH DEMO 5 PHÚT & CẨM NANG PHẢN BIỆN HỘI ĐỒNG

Tài liệu này giúp nhóm tự tin trình bày đồ án trong **5 phút** trên ứng dụng Web Streamlit (`streamlit run app/Home.py`), kèm theo **4 câu demo thực tế** (có sẵn kết quả chạy thật của mô hình), **10 câu hỏi hội đồng thường gặp** và **Danh sách kiểm tra trước giờ thi**.

---

## ⏱️ PHẦN 1: KỊCH BẢN TRÌNH BÀY 5 PHÚT TRÊN WEB

### 🔹 Phút 0:00 – 0:45 | Mở đầu tại Trang 0 (`Home.py` — Tổng quan dự án)
- **Thao tác trên màn hình:** Mở trang chủ **🛡️ Tổng quan dự án**. Chỉ vào 4 thẻ KPI và sơ đồ quy trình.
- **Lời thoại gợi ý:**
  > *"Xin chào Hội đồng. Nhóm chúng em thực hiện đề tài **Phát hiện bình luận độc hại tiếng Việt** trên bộ dữ liệu chuẩn **ViHSD** gồm 3 nhãn: **Bình thường (0)**, **Xúc phạm (1)** và **Thù ghét (2)**.*  
  > *Từ 33.397 dòng thô ban đầu, sau khi làm sạch và khử trùng lặp nội dung, bộ dữ liệu gồm **29.851 bình luận duy nhất**, chia phân tầng 80/10/10 (23.880 Train, 2.985 Val, 2.986 Test). Điểm thách thức lớn nhất của dữ liệu là nhãn **Bình thường chiếm tới 82,6%**, trong khi nhãn **Xúc phạm chỉ chiếm 6,73%**.*  
  > *Trên tập Test độc lập, mô hình **TF-IDF + LinearSVC (Calibrated)** đạt **Accuracy 87,74%** và **F1-macro 67,03%** với tốc độ suy luận chỉ khoảng **6 ms/bình luận**."*

---

### 🔹 Phút 0:45 – 1:30 | Quy trình Làm sạch Tiếng Việt tại Trang 2 (`2_Tien_xu_ly.py`)
- **Thao tác trên màn hình:** Chuyển sang trang **🧹 Tiền xử lý tiếng Việt**, bấm vào một câu mẫu có cả link, emoji, chữ kéo dài và teencode để hiện 3 hộp so sánh.
- **Lời thoại gợi ý:**
  > *"Vì bình luận mạng xã hội Việt Nam chứa rất nhiều nhiễu — cụ thể trên tập Train có **19,36% câu chứa teencode** và **13,50% câu chứa emoji** — nhóm xây dựng hàm `lam_sach` và `tach_tu` dùng chung cho cả lúc huấn luyện lẫn lúc chạy Web: tự động xóa link/emoji, rút gọn chữ kéo dài, ghép chữ bị chèn dấu chấm/gạch ngang né bộ lọc, dịch 42 từ teencode phổ biến và tách từ ghép bằng `pyvi`."*

---

### 🔹 Phút 1:30 – 3:15 | Điểm nhấn Demo trực tiếp 4 Câu Thực tế tại Trang 1 (`1_Thu_nghiem.py`)
- **Thao tác trên màn hình:** Mở trang **🧪 Thử nghiệm trực tiếp**. Lần lượt bấm/nhập **4 câu demo thực tế** dưới đây để hội đồng thấy cả **điểm mạnh** lẫn **hạn chế thật** của mô hình:

#### ✅ Câu Demo 1: Bình luận tích cực (Mô hình đoán ĐÚNG)
- **Câu nhập (bấm nút `[😊 Mẫu 1: Bình thường]`):**
  ```text
  Video hôm nay chia sẻ kiến thức rất bổ ích, cảm ơn tác giả nhiều nhé!
  ```
- **Kết quả mô hình chạy thật:**
  - **Nhãn dự đoán:** `🟢 Bình thường (CLEAN)` (Độ tin cậy: **`83,43%`**, Điểm độc hại: **`16,57%`**).
  - **Sau tiền xử lý:** `video hôm_nay chia_sẻ kiến_thức rất bổ_ích , cảm_ơn tác_giả nhiều nhé !`
  - **Điểm nhấn thuyết trình:** Các từ ghép 2 âm tiết (`hôm_nay`, `chia_sẻ`, `kiến_thức`, `bổ_ích`, `cảm_ơn`, `tác_giả`) được `pyvi` nối dấu gạch dưới chính xác.

#### ✅ Câu Demo 2: Câu Teencode + Chèn ký tự né bộ lọc (Mô hình bắt ĐÚNG & Tô đỏ từ gây độc)
- **Câu nhập (bấm nút `[🔤 Mẫu 4: Teencode + Chèn ký tự]`):**
  ```text
  m.à.y n.g.u vcl đm ko bt j hết
  ```
- **Kết quả mô hình chạy thật:**
  - **Sau tiền xử lý:** `mày ngu vcl đm không bình_thường gì hết` (đã tự động nối `m.à.y` -> `mày`, `n.g.u` -> `ngu`, dịch `ko` -> `không`, `bt` -> `bình_thường`, `j` -> `gì`).
  - **Nhãn dự đoán:** `🟠 Xúc phạm (OFFENSIVE)` (`52,20%` Xúc phạm, `47,79%` Thù ghét -> **Điểm độc hại tổng hợp = `100%`**).
  - **Từ bị tô đỏ (Leave-One-Token-Out):** `mày`, `ngu`, `vcl`, `đm`.
  - **Điểm nhấn thuyết trình:** Nếu không có bước ghép ký tự chèn `RE_CHEN_KT` trong `src/preprocess.py`, mô hình túi từ thông thường sẽ bỏ lọt `m.à.y n.g.u`.

#### ⚠️ Câu Demo 3: Câu Mỉa mai ngầm không dùng từ tục (Mô hình BỎ SÓT — False Negative)
- **Câu nhập (bấm nút `[🎭 Mẫu 3: Mỉa mai (Khó)]`):**
  ```text
  Trình độ cỡ này mà cũng đòi lên mạng dạy đời người khác à? Hài hước thật.
  ```
- **Kết quả mô hình chạy thật:**
  - **Nhãn dự đoán:** `🟢 Bình thường (CLEAN)` (Xác suất Bình thường: **`66,55%`**, nhưng Điểm độc hại đã nhích lên **`33,45%`** nhờ bắt được từ `dạy_đời`).
  - **Điểm nhấn thuyết trình:** Giải thích thẳng thắn với Hội đồng: *"Câu này hoàn toàn không dùng từ chửi thề mà mỉa mai bằng từ vựng lịch sự. Mô hình TF-IDF dựa trên cụm từ nên mới chỉ đẩy điểm độc hại lên 33,45% và vẫn xếp vào Bình thường. Đây chính là lý do nhóm luôn đặt dòng cảnh báo 'Mô hình có thể sai với câu mỉa mai' ngay dưới kết quả."*

#### ⚠️ Câu Demo 4: Câu dùng từ tục để khen món ăn (Mô hình BÁO NHẦM — False Positive)
- **Câu nhập (trích từ tập Test thật trong `outputs/phan_tich_loi.csv`):**
  ```text
  Địt mẹ, ông thử ăn cá kho đi, càng hâm lại nó ăn lại càng ngon vailone :)
  ```
- **Kết quả mô hình chạy thật:**
  - **Nhãn thật trong ViHSD:** `Bình thường (0)` (người viết đang khen món cá kho ngon).
  - **Nhãn mô hình dự đoán:** `🟠 Xúc phạm (OFFENSIVE)` (**Điểm độc hại = `98,52%`**) và tô đỏ 2 từ `địt_mẹ`, `vailone`.
  - **Điểm nhấn thuyết trình:** *"Đây là ví dụ thực tế từ tập Test cho thấy hạn chế của mô hình học theo từ khóa: khi người dùng dùng từ cảm thán suồng sã để khen món ăn, mô hình bị đánh lừa bởi từ khóa nhạy cảm và báo nhầm."*

---

### 🔹 Phút 3:15 – 4:15 | Kết quả Đánh giá & Phân tích Lỗi tại Trang 3 (`3_Danh_gia_mo_hinh.py`)
- **Thao tác trên màn hình:** Mở trang **📈 Đánh giá mô hình**, lướt qua 3 tab: **So sánh mô hình** -> **Ma trận nhầm lẫn** -> **Phân tích lỗi (FP/FN)** và mở 3 hộp phân tích chuyên sâu ở cuối tab 3.
- **Lời thoại gợi ý:**
  > *"Nhóm tuân thủ nghiêm ngặt nguyên tắc chống rò rỉ dữ liệu: 76 lần thử nghiệm tinh chỉnh chỉ thực hiện trên tập Validation (nâng F1-macro Val từ 66,05% lên 68,19%). Tập Test chỉ được chấm đúng 1 lần duy nhất, đạt **F1-macro 67,03%**, trong đó **F1 lớp CLEAN đạt 94,13%**, **HATE đạt 65,61%**, và **OFFENSIVE đạt 41,36%**.*  
  > *Qua phân tích chuyên sâu trên tập Test, nhóm phát hiện: câu càng dài (>25 từ) thì F1 lớp HATE càng cao (78,64%) nhưng F1 lớp OFFENSIVE giảm còn 13,33%; đồng thời các câu chèn dấu `*` né lọc có tỷ lệ lỗi cao nhất (36,84%)."*

---

### 🔹 Phút 4:15 – 5:00 | Quét Bình luận YouTube & Đạo đức AI (`4_YouTube.py` & `5_Han_che_dao_duc.py`)
- **Thao tác trên màn hình:** Mở trang **🎬 Phân tích bình luận YouTube**, bấm nút **"📂 Dùng dữ liệu mẫu có sẵn"** (hoặc dán link YouTube nếu có mạng) để hiện biểu đồ tròn, thanh trượt lọc điểm độc hại và nút tải CSV. Kết thúc ở trang **⚖️ Hạn chế & Đạo đức**.
- **Lời thoại gợi ý:**
  > *"Ứng dụng tích hợp YouTube Data API v3 cho phép quét tối đa 500 bình luận của một video bất kỳ, lọc theo ngưỡng điểm độc hại để hỗ trợ người kiểm duyệt. Để bảo vệ quyền riêng tư và đạo đức AI, hệ thống tuyệt đối **không lưu tên người bình luận**, **không lưu API key**, và chỉ đóng vai trò công cụ gợi ý sàng lọc chứ không tự động xóa bình luận hay khóa tài khoản."*

---

## ❓ PHẦN 2: 10 CÂU HỎI HỘI ĐỒNG HAY HỎI & GỢI Ý TRẢ LỜI

### Câu 1: Tại sao nhóm dùng F1-macro làm thước đo chính để chọn mô hình mà không dùng Accuracy?
- **Trả lời:** Vì bộ dữ liệu ViHSD bị lệch nhãn rất nặng: nhãn `0 (Bình thường)` chiếm tới **`82,60%`**, trong khi nhãn `1 (Xúc phạm)` chỉ chiếm **`6,73%`** và nhãn `2 (Thù ghét)` chiếm **`10,67%`**. Nếu chỉ nhìn vào Accuracy, một mô hình đoán tất cả mọi câu là "Bình thường" cũng đã đạt Accuracy `82,60%` nhưng không bắt được bất kỳ bình luận độc hại nào (`F1-macro = 30,15%`). Chỉ số **F1-macro** tính trung bình cộng không trọng số F1 của cả 3 lớp, buộc mô hình phải làm tốt trên cả hai lớp thiểu số (`OFFENSIVE` và `HATE`).

### Câu 2: `LinearSVC` không có hàm `predict_proba()`, vậy làm sao nhóm lấy được xác suất từng nhãn và Điểm độc hại trên Web?
- **Trả lời:** Nhóm sử dụng kỹ thuật hiệu chuẩn xác suất **`CalibratedClassifierCV(LinearSVC(C=0.1), cv=5, method='sigmoid')`** (Platt Scaling 5-fold cross-validation chỉ fit trên tập Train). Kỹ thuật này khớp một hàm Logistic trên điểm `decision_function` của SVM, giúp xuất ra phân phối xác suất chuẩn hóa trên 3 nhãn. Khi đó, **Điểm độc hại (`diem_toxic`)** được tính bằng tổng xác suất của 2 nhãn độc hại: `P(Xúc phạm) + P(Thù ghét)`.

### Câu 3: Tại sao F1 của nhãn `OFFENSIVE` chỉ đạt `41,36%` trên tập Test? Nhóm đã thử cách gì để cải thiện?
- **Trả lời:** Trong Bước 2 (`outputs/thu_nghiem_baseline.csv`), nhóm đã thực hiện 76 lần thử nghiệm gồm tinh chỉnh trọng số lớp (`{0: 1.0, 1: 3.0, 2: 2.0}`), chỉnh ngưỡng ưu tiên riêng cho lớp `OFFENSIVE` và thêm đặc trưng `word (1,3) + char_wb (2,5)`, giúp nâng F1 `OFFENSIVE` trên tập Val từ `41,51%` lên `47,03%`. Tuy nhiên, trên tập Test chỉ số này đạt `41,36%` do 3 nguyên nhân bản chất của dữ liệu:
  1. Số mẫu `OFFENSIVE` quá ít (chỉ `6,73%` toàn bộ dữ liệu).
  2. Nhiều câu `OFFENSIVE` dùng từ vựng thông thường để mỉa mai hoặc nói giảm nói tránh (như *"Này khẩu phật tâm xà"*, *"Cứ tưởng thế là ngầu"*).
  3. Ranh giới giữa `CLEAN` – `OFFENSIVE` – `HATE` ngay trong bộ dữ liệu gốc có sự bất đồng cao giữa những người gán nhãn (240 văn bản trùng lặp trong dữ liệu thô được gán nhiều nhãn khác nhau).

### Câu 4: Làm sao nhóm tính và tô đỏ được "từ gây độc" trên giao diện Web đối với cả SVM và PhoBERT?
- **Trả lời:** Nhóm cài đặt hàm `tim_tu_gay_doc()` trong `src/inference.py` theo phương pháp **Leave-One-Token-Out (LOTO)**: với một câu đã tách từ gồm $N$ từ, hệ thống tạo $N$ biến thể câu mà mỗi biến thể lược bỏ đi đúng 1 từ, sau đó chạy mô hình để tính độ sụt giảm của Điểm độc hại (`do_giam_prob`) và độ sụt giảm Logit (`do_giam_logit`). Từ nào khi bỏ đi làm điểm độc hại giảm từ `0.03` trở lên (hoặc logit giảm `>= 0.35` đối với câu có nhiều từ chửi làm bão hòa xác suất ở `0.9999`) sẽ được tô đỏ. Cách này hoạt động nhất quán cho mọi mô hình mà không phụ thuộc cấu trúc bên trong.

### Câu 5: Tại sao phải dùng `pyvi` để tách từ tiếng Việt? Nếu bỏ bước tách từ thì sao?
- **Trả lời:** Trong tiếng Việt, đơn vị từ thường gồm 2 âm tiết trở lên (ví dụ: `"xúc_phạm"`, `"thất_học"`, `"dạy_đời"`). Nếu tách theo khoảng trắng đơn thuần, từ `"thất_học"` bị xé thành hai âm tiết rời `"thất"` (trong *thất bại*, *nội thất*) và `"học"` (trong *học tập*), làm nhiễu biểu diễn TF-IDF. Hơn nữa, mô hình `vinai/phobert-base-v2` được huấn luyện tiền kỳ trên văn bản đã tách từ bằng gạch dưới `_`, nên bắt buộc đầu vào phải qua `tach_tu()` mới khớp từ điển BPE của PhoBERT.

### Câu 6: Trong hàm `tach_tu()` của `src/preprocess.py`, tại sao lại phải thay dấu `*` bằng `X_STAR_X` trước khi gọi `ViTokenizer.tokenize`?
- **Trả lời:** Nhiều bình luận lách luật dùng dấu `*` giữa từ (như `đ*o`, `đ*t`). Bộ tách từ `ViTokenizer` mặc định coi dấu `*` là dấu câu và tách rời thành 3 token `đ * o`. Việc tạm thời đổi `*` thành `X_STAR_X` giúp `ViTokenizer` giữ nguyên cả cụm từ `đ*o` thành một token duy nhất để đặc trưng `char_wb (2,5)` vẫn bắt được tiền tố/hậu tố của từ đó.

### Câu 7: Nhóm đảm bảo không bị rò rỉ dữ liệu (Data Leakage) giữa Train, Val và Test bằng cách nào?
- **Trả lời:**
  1. Ở bước tiền xử lý (`scripts/process_data_fixed.py`), nhóm khử toàn bộ các câu trùng nhau theo cột `text_clean` trước khi chia tập; kiểm tra bằng `scripts/check_data.py` xác nhận giao nhau giữa Train, Val và Test bằng `0`.
  2. Bộ biến đổi `TfidfVectorizer` và bộ hiệu chuẩn `CalibratedClassifierCV(cv=5)` chỉ được `.fit()` trên `03_train.csv`.
  3. Toàn bộ 76 lần thử nghiệm chọn mô hình chỉ chấm trên `03_val.csv`. Tập `03_test.csv` chỉ được mở ra chấm đúng 1 lần ở `scripts/evaluate_final.py`.

### Câu 8: Khi người dùng dán link YouTube vào Trang 4, nếu video tắt bình luận, sai API key hoặc mất mạng thì ứng dụng xử lý thế nào?
- **Trả lời:** Trong `src/youtube.py`, nhóm bắt riêng từng mã lỗi của YouTube Data API v3 (`commentsDisabled`, `quotaExceeded`, `keyInvalid`, `videoNotFound`, `requests.RequestException`) và hiển thị thông báo tiếng Việt rõ ràng hướng dẫn cách xử lý, tuyệt đối không để văng lỗi kỹ thuật (traceback) ra màn hình. Ngoài ra, trên giao diện luôn có nút **"📂 Dùng dữ liệu mẫu có sẵn"** đọc từ `outputs/ket_qua_youtube.csv` để demo ngay cả khi phòng thi mất kết nối Internet.

### Câu 9: Tại sao nhóm tách `requirements.txt` và `requirements-train.txt`?
- **Trả lời:** Để chạy ứng dụng Web Streamlit và mô hình Baseline TF-IDF, chỉ cần các thư viện nhẹ trong `requirements.txt` (cài đặt rất nhanh và ghim đúng `scikit-learn==1.9.1` để mở file `.pkl` an toàn). Trong khi đó, các gói huấn luyện học sâu nặng hàng GB (`torch`, `transformers`, `datasets`, `accelerate`) được tách sang `requirements-train.txt` phục vụ khi chạy PhoBERT.

### Câu 10: Hệ thống này có thể áp dụng để tự động xóa bình luận ngay lập tức trên mạng xã hội không?
- **Trả lời:** **Không nên tự động xóa hoàn toàn.** Như nhóm đã trình bày ở Trang 5 (Hạn chế & Đạo đức), trên tập Test mô hình vẫn có tỷ lệ Báo nhầm (False Positive) khoảng `4,99%` ở các câu dùng từ lóng/tục để trêu đùa bạn bè hoặc khen ngợi. Nếu tự động xóa sẽ xâm phạm quyền tự do biểu đạt của người dùng. Giải pháp đúng đắn là mô hình **Human-in-the-loop**: dùng Điểm độc hại (`diem_toxic`) để lọc và đẩy các bình luận có nguy cơ cao lên đầu danh sách cho kiểm duyệt viên xác nhận.

---

## ✅ PHẦN 3: DANH SÁCH KIỂM TRA TRƯỚC GIỜ THI (10 MỤC — MỤC 7 BẢN VẼ)

Trước khi bước vào phòng bảo vệ 15 phút, nhóm hãy tích kiểm tra đủ 10 mục sau:

- [x] **1. Chạy thử toàn bộ trên máy sẽ mang đi thi:** Mở PowerShell tại thư mục dự án, chạy `streamlit run app/Home.py`, bấm qua cả **6 trang** (Tổng quan, Thử nghiệm, Tiền xử lý, Đánh giá mô hình, YouTube, Hạn chế & đạo đức) không trang nào báo lỗi đỏ.
- [x] **2. Kiểm tra bộ test tự động:** Chạy `python -m pytest tests/test_inference.py -v` đảm bảo xanh `4 passed`.
- [x] **3. Thử nghiệm 4 câu mẫu trên Trang 1:** Bấm cả 4 nút mẫu (câu bình thường, câu chửi thẳng, câu mỉa mai, câu teencode), kiểm tra thẻ kết quả hiện đúng màu, thanh xác suất cộng lại bằng `100%` và các từ gây độc được tô đỏ.
- [x] **4. Thử các trường hợp biên trên Trang 1:** Bấm Phân tích khi ô trống, nhập toàn emoji `😂😂😂` — ứng dụng hiện cảnh báo nhẹ nhàng, không sập.
- [x] **5. Kiểm tra phương án dự phòng mất mạng ở Trang 4 (YouTube):** Bấm nút **"📂 Dùng dữ liệu mẫu có sẵn"** để chắc chắn bảng 120 bình luận và biểu đồ tròn hiện ra ngay lập tức không cần Internet.
- [x] **6. Kiểm tra đồng nhất số liệu:** Số liệu Accuracy (`87,74%`) và F1-macro (`67,03%`) trên Trang 0 (`Home.py`), Trang 3 (`3_Danh_gia_mo_hinh.py`), `README.md` và `reports/bao_cao_du_an.md` khớp nhau 100% (vì đều đọc từ `outputs/bang_so_sanh_mo_hinh.csv`).
- [x] **7. Kiểm tra bảo mật:** Không có mã API key YouTube nào nằm trong code hay trong `git log`.
- [x] **8. Kiểm tra phiên bản thư viện:** Máy thuyết trình đã cài đúng `scikit-learn==1.9.1` trong `requirements.txt` để mở `models/baseline_calibrated.pkl` mượt mà.
- [x] **9. Chuẩn bị sẵn cửa sổ trình duyệt:** Mở sẵn tab `http://localhost:8501` ở mức thu phóng (zoom) `100%` hoặc `110%`, chế độ giao diện sáng (Light mode) để máy chiếu hiển thị rõ nét nhất.
- [x] **10. Phân công thành viên:** Thống nhất rõ ai trình bày phần Dữ liệu & Tiền xử lý, ai demo trực tiếp trên Web, và ai trả lời câu hỏi về Mô hình & Phân tích lỗi.
