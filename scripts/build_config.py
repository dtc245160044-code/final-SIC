import os
import json
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

BASELINE_NAME = "TF-IDF + LinearSVC (Calibrated)"
PHOBERT_NAME = "PhoBERT (fine-tune)"


def _has_phobert(models_dir: str) -> bool:
    for sub in [os.path.join(models_dir, "phobert", "best"), os.path.join(models_dir, "phobert")]:
        cfg = os.path.join(sub, "config.json")
        w_safe = os.path.join(sub, "model.safetensors")
        w_bin = os.path.join(sub, "pytorch_model.bin")
        if os.path.isfile(cfg) and (os.path.isfile(w_safe) or os.path.isfile(w_bin)):
            return True
    return False


def build_config() -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    available = []

    baseline_path = os.path.join(MODELS_DIR, "baseline_calibrated.pkl")
    fallback_baseline = os.path.join(MODELS_DIR, "baseline_LinearSVC_C0.1.pkl")
    tfidf_lr_path = os.path.join(MODELS_DIR, "tfidf_lr.pkl")

    if os.path.isfile(baseline_path) or os.path.isfile(fallback_baseline) or os.path.isfile(tfidf_lr_path):
        available.append(BASELINE_NAME)

    if _has_phobert(MODELS_DIR):
        available.append(PHOBERT_NAME)

    best_model = available[0] if available else ""

    # Nếu đã có bảng so sánh kết quả thật trên tập Test, chọn mô hình có F1-macro cao nhất trong số các mô hình hiện có
    comp_csv = os.path.join(OUTPUTS_DIR, "bang_so_sanh_mo_hinh.csv")
    if os.path.isfile(comp_csv) and available:
        try:
            df = pd.read_csv(comp_csv)
            f1_col = "F1-macro (%)" if "F1-macro (%)" in df.columns else ("F1-macro" if "F1-macro" in df.columns else None)
            if f1_col and "Mô hình" in df.columns:
                df_avail = df[df["Mô hình"].isin(available)].copy()
                if not df_avail.empty:
                    df_avail = df_avail.sort_values(by=f1_col, ascending=False)
                    best_model = str(df_avail.iloc[0]["Mô hình"])
        except Exception as e:
            print(f"Lưu ý: Không đọc được {comp_csv} để chọn best_model ({e}). Dùng mặc định: {best_model}")

    config = {
        "label_names": ["Bình thường", "Xúc phạm", "Thù ghét"],
        "label_codes": ["CLEAN", "OFFENSIVE", "HATE"],
        "toxic_ids": [1, 2],
        "best_model": best_model,
        "max_len": 128,
        "models_available": available,
        "is_demo_data": False,
    }

    config_path = os.path.join(MODELS_DIR, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Đã tạo {config_path}:")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return config


if __name__ == "__main__":
    build_config()

