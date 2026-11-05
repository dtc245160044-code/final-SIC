# 🛡️ Phát hiện Bình luận Độc hại Tiếng Việt trên Mạng Xã hội (ViHSD)

Dự án xây dựng hệ thống **Phát hiện và Phân loại Bình luận Độc hại Tiếng Việt** theo 3 mức độ trên bộ dữ liệu chuẩn **ViHSD (Vietnamese Hate Speech Detection)**, kết hợp quy trình tiền xử lý đặc thù cho văn bản mạng xã hội Việt Nam (chuẩn hóa teencode, chữ kéo dài, ký tự chèn né bộ lọc, tách từ bằng `pyvi`) và ứng dụng Web trực quan hóa gồm 6 trang bằng **Streamlit**.

---

## 🎯 1. Tổng quan Bài toán & Bộ dữ liệu

Hệ thống phân loại mỗi bình luận tiếng Việt vào một trong **3 nhãn**:
- **`0` — Bình thường (`CLEAN`):** Bình luận văn minh, trao đổi trung lập hoặc tích cực.
- **`1` — Xúc phạm (`OFFENSIVE`):** Bình luận có từ ngữ suồng sã, châm chọc, mỉa mai hoặc xúc phạm nhẹ.
- **`2` — Thù ghét (`HATE`):** Bình luận chửi bới nặng nề, lăng mạ, công kích cá nhân/nhóm hoặc đe dọa.

### Thống kê bộ dữ liệu sau làm sạch & khử trùng lặp (`data/processed/`)
Từ **33.397** dòng dữ liệu thô ban đầu (`data/raw/comments.csv`), sau khi chuẩn hóa, loại bỏ dòng rỗng và khử các câu trùng lặp nội dung (giữ nhãn theo đa số), bộ dữ liệu còn **29.851** mẫu duy nhất và được chia phân tầng (`stratified`, `seed=42`) theo tỷ lệ `80% / 10% / 10%`:

| Tập dữ liệu | File lưu trữ | Tổng số mẫu | `0 = Bình thường` | `1 = Xúc phạm` | `2 = Thù ghét` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train (Huấn luyện)** | `data/processed/03_train.csv` | **23.880** | 19.726 (`82,60%`) | 1.606 (`6,73%`) | 2.548 (`10,67%`) |
| **Validation (Tinh chỉnh)** | `data/processed/03_val.csv` | **2.985** | 2.466 (`82,61%`) | 201 (`6,73%`) | 318 (`10,65%`) |
| **Test (Kiểm tra cuối)** | `data/processed/03_test.csv` | **2.986** | 2.466 (`82,59%`) | 201 (`6,73%`) | 319 (`10,68%`) |
| **Tổng cộng** | | **29.851** | **24.658 (`82,60%`)** | **2.008 (`6,73%`)** | **3.185 (`10,67%`)** |

---

## 📊 2. Kết quả Thực nghiệm trên Tập Test (`outputs/bang_so_sanh_mo_hinh.csv`)

> **Nguyên tắc đánh giá:** Toàn bộ quá trình chọn mô hình và tinh chỉnh siêu tham số (76 thử nghiệm trong `outputs/thu_nghiem_baseline.csv`) chỉ sử dụng tập **Validation**. Tập **Test (2.986 mẫu)** chỉ được chấm đúng 1 lần duy nhất ở bước đánh giá cuối cùng (`scripts/evaluate_final.py`).

| Mô hình | Accuracy (%) | F1-macro (%) | Toxic Precision (%) | Toxic Recall (%) | F1 CLEAN (%) | F1 OFFENSIVE (%) | F1 HATE (%) | Độ trễ (ms/câu) | Kích thước (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LinearSVC (Calibrated)** | **87,74%** | **67,03%** | **74,59%** | **68,24%** | **94,13%** | **41,36%** | **65,61%** | **5,89 ms** | **9,42 MB** |

*(Khi bổ sung thư mục `models/phobert/best/` từ Colab và chạy lại `python scripts/evaluate_final.py`, dòng kết quả của `PhoBERT (fine-tune)` sẽ tự động được cập nhật vào bảng trên và trên giao diện Web).*

---

## 📂 3. Cấu trúc Thư mục Dự án

```text
final-SIC/
├── app/                                # Ứng dụng Web Streamlit (6 trang)
│   ├── Home.py                         # Trang 0: Tổng quan dự án, KPI & kiến trúc pipeline
│   ├── ui.py                           # Thư viện giao diện dùng chung & kiểm tra file an toàn
│   └── pages/
│       ├── 1_Thu_nghiem.py             # Trang 1: Thử nghiệm trực tiếp & tô đỏ từ gây độc (LOTO)
│       ├── 2_Tien_xu_ly.py             # Trang 2: Minh họa từng bước tiền xử lý & bảng đối chiếu
│       ├── 3_Danh_gia_mo_hinh.py       # Trang 3: Bảng so sánh, Ma trận nhầm lẫn & Phân tích lỗi FP/FN
│       ├── 4_YouTube.py                # Trang 4: Quét bình luận YouTube theo link (có chế độ dự phòng)
│       └── 5_Han_che_dao_duc.py        # Trang 5: Hạn chế kỹ thuật & Đạo đức AI
├── data/
│   └── processed/                      # Dữ liệu chuẩn đã làm sạch & tách từ (03_train/val/test.csv)
├── models/                             # Chứa trọng số mô hình (được gitignore các file nặng >50MB)
│   ├── config.json                     # File cấu hình sinh tự động từ scripts/build_config.py
│   ├── baseline_LinearSVC_C0.1.pkl     # Mô hình Baseline gốc
│   ├── baseline_calibrated.pkl         # Mô hình Baseline tốt nhất (CalibratedClassifierCV, cv=5)
│   └── phobert/best/                   # (Tùy chọn) Trọng số PhoBERT fine-tuned tải về từ Colab
├── notebooks/                          # Jupyter Notebooks phân tích & huấn luyện
│   ├── 01_du_lieu_va_tien_xu_ly.ipynb  # Khám phá dữ liệu & kiểm tra hàm tiền xử lý
│   ├── 02_baseline_tfidf.ipynb         # Thực nghiệm TF-IDF + LinearSVC
│   ├── 03_phobert.ipynb                # Notebook PhoBERT cục bộ
│   └── 03_phobert_colab.ipynb          # Notebook tự chứa chạy PhoBERT trên Google Colab GPU
├── outputs/                            # Toàn bộ bảng biểu CSV & biểu đồ PNG sinh từ code chạy thật
│   ├── bang_so_sanh_mo_hinh.csv        # Bảng tổng hợp kết quả chính thức trên tập Test
│   ├── thu_nghiem_baseline.csv         # Lịch sử 76 thử nghiệm tinh chỉnh Baseline trên tập Val
│   ├── phan_tich_loi.csv               # 70 mẫu lỗi FP/FN điển hình trên tập Test
│   ├── phan_tich_do_dai_cau.csv        # Phân tích F1 theo độ dài câu (ngắn / trung bình / dài)
│   ├── phan_tich_nhieu_dac_biet.csv    # Phân tích tỷ lệ lỗi trên câu teencode, chèn ký tự, emoji
│   ├── top15_offensive_doan_sai.csv    # Top 15 câu OFFENSIVE bị đoán sai kèm nhận xét
│   ├── truoc_sau_xu_ly.csv             # 20 câu mẫu minh họa trước & sau tiền xử lý
│   ├── ket_qua_youtube.csv             # 120 bình luận mẫu phân loại thật (dùng khi mất mạng)
│   └── *.png                           # Các biểu đồ phân bố nhãn, ma trận nhầm lẫn, so sánh mô hình
├── reports/                            # Báo cáo tổng kết & kịch bản thuyết trình
│   ├── bao_cao_du_an.md                # Báo cáo toàn văn dự án (số liệu trích xuất 100% từ file thật)
│   └── kich_ban_demo.md                # Kịch bản thuyết trình 5 phút + 10 câu hỏi phản biện
├── scripts/                            # Các kịch bản chạy bằng dòng lệnh (nguồn chuẩn duy nhất)
│   ├── check_data.py                   # Kiểm tra tính toàn vẹn của data/processed/
│   ├── run_baseline.py                 # Chạy mô hình Baseline gốc trên tập Val
│   ├── tune_baseline.py                # Chạy 76 thử nghiệm tinh chỉnh Baseline trên tập Val
│   ├── run_phobert.py                  # Huấn luyện PhoBERT (hỗ trợ cờ --smoke kiểm tra nhanh)
│   ├── build_config.py                 # Tự động quét đĩa và sinh models/config.json
│   └── evaluate_final.py               # Chấm tập Test đúng 1 lần, xuất bảng & biểu đồ cho Web
├── src/                                # Thư viện lõi của hệ thống
│   ├── preprocess.py                   # Hàm chuẩn: lam_sach, tach_tu, phat_hien (42 từ teencode)
│   ├── inference.py                    # Hợp đồng hàm suy luận: doc_cau_hinh, tai_mo_hinh, phan_loai...
│   └── youtube.py                      # Kết nối YouTube Data API v3 an toàn (tối đa 500 bình luận)
├── tests/
│   └── test_inference.py               # Bộ kiểm thử pytest (hợp đồng hàm, câu biên & 6 trang Web)
├── requirements.txt                    # Các gói gọn nhẹ để chạy Web Demo & Baseline (đã ghim phiên bản)
└── requirements-train.txt              # Các gói nặng (torch, transformers...) để huấn luyện PhoBERT
```

---

## ⚙️ 4. Hướng dẫn Cài đặt Môi trường

Yêu cầu **Python 3.11** (khuyến nghị chạy trên Windows PowerShell hoặc môi trường ảo `venv`).

### 4.1. Cài đặt để chạy Web Demo, Baseline & Kiểm thử (nhẹ, nhanh)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> **Lưu ý quan trọng:** File `requirements.txt` đã ghim cứng `scikit-learn==1.9.1` và `pandas==3.0.5` để đảm bảo nạp file `models/baseline_calibrated.pkl` không gặp cảnh báo lệch phiên bản.

### 4.2. Cài đặt đầy đủ nếu muốn huấn luyện hoặc chạy suy luận với PhoBERT
```powershell
pip install -r requirements-train.txt
```

---

## 🚀 5. Lệnh Chạy Từng Bước

Trên Windows PowerShell, hãy bật UTF-8 trước khi chạy các script in tiếng Việt:
```powershell
$env:PYTHONIOENCODING="utf-8"
```

### Bước 1: Kiểm tra dữ liệu đã tiền xử lý
```powershell
python scripts/check_data.py
```

### Bước 2: Huấn luyện & tinh chỉnh mô hình Baseline (chỉ dùng Train + Val)
```powershell
# Chạy baseline gốc
python scripts/run_baseline.py

# Chạy 76 thử nghiệm tinh chỉnh & hiệu chuẩn xác suất (lưu models/baseline_calibrated.pkl)
python scripts/tune_baseline.py
```

### Bước 3: Huấn luyện PhoBERT (trên Google Colab GPU hoặc máy có GPU)
- Mở file `notebooks/03_phobert_colab.ipynb` trên **Google Colab** (chọn Runtime **T4 GPU**), hoặc chạy kiểm tra nhanh cục bộ bằng cờ `--smoke`:
```powershell
python scripts/run_phobert.py --smoke
```

### Bước 4: Cập nhật cấu hình & Đánh giá cuối cùng trên tập Test
```powershell
python scripts/build_config.py
python scripts/evaluate_final.py
```

### Bước 5: Chạy bộ kiểm thử tự động (`pytest`)
```powershell
python -m pytest tests/test_inference.py -v
```

### Bước 6: Khởi động ứng dụng Web Streamlit
```powershell
streamlit run app/Home.py
```
Ứng dụng sẽ tự động mở tại địa chỉ `http://localhost:8501`.

---

## 📦 6. Hướng dẫn Đặt Mô hình (`models/`) từ Google Drive / Colab

Vì các file trọng số mô hình có dung lượng lớn nên không đẩy trực tiếp lên Git (`.gitignore` bỏ qua `models/`). Khi tải mô hình về máy cá nhân, bạn đặt đúng vào đường dẫn sau:

1. **Mô hình Baseline (bắt buộc để chạy Web ngay):**
   - Đặt file `baseline_calibrated.pkl` vào thư mục `models/baseline_calibrated.pkl` (hoặc chạy lệnh `python scripts/tune_baseline.py` mất ~2 phút để tự sinh file này trên máy).
2. **Mô hình PhoBERT (tùy chọn — sau khi chạy xong `03_phobert_colab.ipynb`):**
   - Giải nén file `phobert_ket_qua_colab.zip` tải từ Google Drive vào gốc dự án sao cho thư mục `models/phobert/best/` chứa `config.json`, `model.safetensors` (hoặc `pytorch_model.bin`), `vocab.txt`, `bpe.codes`.
   - Giải nén thêm `outputs/phobert_val_predictions.csv` và `outputs/phobert_test_predictions.csv` vào thư mục `outputs/`.
   - Chạy lại:
     ```powershell
     python scripts/build_config.py
     python scripts/evaluate_final.py
     ```
   - Giao diện Web sẽ tự động nhận diện thêm `PhoBERT (fine-tune)` trong hộp chọn mô hình mà không cần sửa bất kỳ dòng code nào.

---

## 🔑 7. Hướng dẫn Lấy YouTube Data API v3 Key (Cho Trang 4 — YouTube)

Trang **🎬 Phân tích bình luận YouTube** hỗ trợ 2 chế độ:
- **Chế độ dữ liệu mẫu có sẵn (không cần mạng / không cần API key):** Bấm nút **"📂 Dùng dữ liệu mẫu có sẵn"** để nạp ngay 120 bình luận từ `outputs/ket_qua_youtube.csv`.
- **Chế độ tải bình luận trực tiếp từ YouTube:**
  1. Truy cập [Google Cloud Console](https://console.cloud.google.com/).
  2. Tạo một Project mới -> vào mục **APIs & Services** -> **Library**.
  3. Tìm kiếm **"YouTube Data API v3"** và bấm **Enable**.
  4. Vào mục **APIs & Services** -> **Credentials** -> bấm **Create Credentials** -> chọn **API key**.
  5. Sao chép mã API key (bắt đầu bằng `AIza...`) và dán vào ô **"🔑 YouTube Data API v3 Key"** trên giao diện Web.

> 🔒 **Cam kết bảo mật & quyền riêng tư:**
> - Mã API key chỉ lưu tạm trong phiên trình duyệt (`password input`), **tuyệt đối không ghi ra file, log hay lịch sử Git**.
> - Hệ thống **không lưu tên người bình luận (`authorDisplayName`) hay ID kênh**, chỉ tải nội dung bình luận công khai, số lượt thích và thời gian đăng (tối đa 500 bình luận/video).
