# Hướng dẫn chạy dự án

Dự án này bao gồm các kịch bản (scripts) để tiền xử lý dữ liệu và huấn luyện mô hình. Bạn nên chạy từ thư mục gốc của dự án.

## Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## Các bước thực hiện

1. **Kiểm tra dữ liệu:**
```bash
python scripts/check_data.py
```

2. **Tiền xử lý dữ liệu:**
```bash
python scripts/process_data_fixed.py
```

3. **Huấn luyện Baseline (ví dụ Logistic Regression, SVM, v.v.):**
```bash
python scripts/create_baseline.py
python scripts/run_baseline.py
```

4. **Huấn luyện PhoBERT:**
```bash
python scripts/create_phobert.py
python scripts/run_phobert.py
```
