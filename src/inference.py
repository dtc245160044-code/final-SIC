import os
import time
import json
import pickle
import numpy as np
import pandas as pd

from src.preprocess import lam_sach, tach_tu

try:
    import streamlit as st
    _cache_resource = st.cache_resource(show_spinner=False)
except Exception:
    def _cache_resource(func):
        return func

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

BASELINE_NAME = "TF-IDF + LinearSVC (Calibrated)"
PHOBERT_NAME = "PhoBERT (fine-tune)"

_LOCAL_MODEL_CACHE = {}


def _detect_available_models(models_dir: str = MODELS_DIR) -> list[str]:
    avail = []
    for fname in ["baseline_calibrated.pkl", "tfidf_lr.pkl", "baseline_LinearSVC_C0.1.pkl"]:
        if os.path.isfile(os.path.join(models_dir, fname)):
            avail.append(BASELINE_NAME)
            break
    for sub in [os.path.join(models_dir, "phobert", "best"), os.path.join(models_dir, "phobert")]:
        cfg = os.path.join(sub, "config.json")
        w_safe = os.path.join(sub, "model.safetensors")
        w_bin = os.path.join(sub, "pytorch_model.bin")
        if os.path.isfile(cfg) and (os.path.isfile(w_safe) or os.path.isfile(w_bin)):
            avail.append(PHOBERT_NAME)
            break
    return avail


def doc_cau_hinh(models_dir: str | None = None) -> dict:
    """Đọc models/config.json. Nếu thiếu file hoặc thư mục models/, trả về cấu hình an toàn dựa trên file thật."""
    mdir = models_dir or MODELS_DIR
    cfg_path = os.path.join(mdir, "config.json")
    default_cfg = {
        "label_names": ["Bình thường", "Xúc phạm", "Thù ghét"],
        "label_codes": ["CLEAN", "OFFENSIVE", "HATE"],
        "toxic_ids": [1, 2],
        "best_model": "",
        "max_len": 128,
        "models_available": [],
        "is_demo_data": False,
        "config_missing": True,
    }
    disk_avail = _detect_available_models(mdir)
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Chỉ giữ các mô hình thật sự có file trên đĩa để tránh lỗi khi thiếu file
            listed = data.get("models_available", [])
            real_avail = [m for m in listed if m in disk_avail]
            for m in disk_avail:
                if m not in real_avail:
                    real_avail.append(m)
            data["models_available"] = real_avail
            if data.get("best_model") not in real_avail:
                data["best_model"] = real_avail[0] if real_avail else ""
            data.setdefault("label_names", default_cfg["label_names"])
            data.setdefault("label_codes", default_cfg["label_codes"])
            data.setdefault("toxic_ids", default_cfg["toxic_ids"])
            data.setdefault("max_len", 128)
            data.setdefault("is_demo_data", False)
            data["config_missing"] = False
            return data
        except Exception:
            pass

    default_cfg["models_available"] = disk_avail
    default_cfg["best_model"] = disk_avail[0] if disk_avail else ""
    return default_cfg


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


@_cache_resource
def tai_mo_hinh(ten_mo_hinh: str, models_dir: str | None = None):
    """Nạp mô hình từ đĩa (chỉ nạp 1 lần nhờ cache). Thiếu file thì trả về None, không báo lỗi đỏ."""
    mdir = models_dir or MODELS_DIR
    cache_key = (ten_mo_hinh, mdir)
    if cache_key in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[cache_key]

    if not ten_mo_hinh:
        return None

    # 1. Nhóm Baseline TF-IDF
    if ten_mo_hinh in [BASELINE_NAME, "TF-IDF + Logistic Regression", "Baseline"]:
        for fname in ["baseline_calibrated.pkl", "tfidf_lr.pkl", "baseline_LinearSVC_C0.1.pkl"]:
            fpath = os.path.join(mdir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "rb") as f:
                        obj = pickle.load(f)
                    loaded = {"type": "sklearn", "model": obj, "path": fpath}
                    _LOCAL_MODEL_CACHE[cache_key] = loaded
                    return loaded
                except Exception:
                    return None
        return None

    # 2. Nhóm PhoBERT
    if "PhoBERT" in ten_mo_hinh or "phobert" in ten_mo_hinh.lower():
        for sub in [os.path.join(mdir, "phobert", "best"), os.path.join(mdir, "phobert")]:
            cfg_file = os.path.join(sub, "config.json")
            w_safe = os.path.join(sub, "model.safetensors")
            w_bin = os.path.join(sub, "pytorch_model.bin")
            if os.path.isfile(cfg_file) and (os.path.isfile(w_safe) or os.path.isfile(w_bin)):
                try:
                    import torch
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    tokenizer = AutoTokenizer.from_pretrained(sub)
                    model = AutoModelForSequenceClassification.from_pretrained(sub)
                    model.to(device)
                    model.eval()
                    loaded = {
                        "type": "phobert",
                        "model": model,
                        "tokenizer": tokenizer,
                        "device": device,
                        "path": sub,
                    }
                    _LOCAL_MODEL_CACHE[cache_key] = loaded
                    return loaded
                except Exception:
                    return None
        return None

    return None


def _du_doan_xac_suat(danh_sach_seg: list[str], model_bundle: dict, max_len: int = 128) -> np.ndarray:
    """Nhận danh sách câu đã tách từ (text_seg), trả về ma trận xác suất (N, 3)."""
    if not danh_sach_seg:
        return np.zeros((0, 3), dtype=np.float64)

    mtype = model_bundle["type"]
    obj = model_bundle["model"]

    if mtype == "sklearn":
        # Trường hợp lưu tuple (tfidf, lr)
        if isinstance(obj, tuple) and len(obj) == 2:
            vec, clf = obj
            X = vec.transform(danh_sach_seg)
            if hasattr(clf, "predict_proba"):
                return np.asarray(clf.predict_proba(X), dtype=np.float64)
            dec = clf.decision_function(X)
            return _softmax_np(dec)
        # Trường hợp lưu Pipeline chuẩn
        if hasattr(obj, "predict_proba"):
            return np.asarray(obj.predict_proba(danh_sach_seg), dtype=np.float64)
        dec = obj.decision_function(danh_sach_seg)
        return _softmax_np(dec)

    if mtype == "phobert":
        import torch

        tokenizer = model_bundle["tokenizer"]
        model = model_bundle["model"]
        device = model_bundle["device"]
        batch_size = 32
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(danh_sach_seg), batch_size):
                batch_texts = danh_sach_seg[i : i + batch_size]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    raise ValueError(f"Không hỗ trợ loại mô hình: {mtype}")


def phan_loai(binh_luan: str, ten_mo_hinh: str | None = None, models_dir: str | None = None) -> dict:
    """
    Phân loại 1 bình luận theo quy trình: lam_sach -> tach_tu -> mô hình -> xác suất.
    Trả về dict đúng hợp đồng mục 5 của ban_ve_giao_dien.md.
    """
    cfg = doc_cau_hinh(models_dir=models_dir)
    label_names = cfg.get("label_names", ["Bình thường", "Xúc phạm", "Thù ghét"])
    toxic_ids = cfg.get("toxic_ids", [1, 2])
    max_len = int(cfg.get("max_len", 128))

    if binh_luan is None or not isinstance(binh_luan, str):
        binh_luan = str(binh_luan or "")

    raw_strip = binh_luan.strip()
    if not raw_strip:
        return {
            "error": "Bình luận đang để trống. Vui lòng nhập nội dung cần phân tích.",
            "binh_luan": binh_luan,
            "sau_xu_ly": "",
            "nhan": "",
            "nhan_id": -1,
            "do_tin_cay": 0.0,
            "diem_toxic": 0.0,
            "xac_suat_tung_nhan": {ten: 0.0 for ten in label_names},
            "mo_hinh": ten_mo_hinh or cfg.get("best_model", ""),
            "do_tre_ms": 0.0,
        }

    avail = cfg.get("models_available", [])
    chosen_model = ten_mo_hinh if (ten_mo_hinh and ten_mo_hinh in avail) else cfg.get("best_model", "")
    if not chosen_model and avail:
        chosen_model = avail[0]

    bundle = tai_mo_hinh(chosen_model, models_dir=models_dir) if chosen_model else None

    t0 = time.perf_counter()
    sach = lam_sach(raw_strip)
    seg = tach_tu(sach) if sach else ""

    if not seg.strip():
        do_tre = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "error": "Sau khi làm sạch (bỏ emoji, link, ký tự đặc biệt), bình luận không còn từ nào để phân tích.",
            "binh_luan": binh_luan,
            "sau_xu_ly": "",
            "nhan": label_names[0],
            "nhan_id": 0,
            "do_tin_cay": 1.0,
            "diem_toxic": 0.0,
            "xac_suat_tung_nhan": {
                label_names[0]: 1.0,
                label_names[1]: 0.0,
                label_names[2]: 0.0,
            },
            "mo_hinh": chosen_model or cfg.get("best_model", ""),
            "do_tre_ms": do_tre,
        }

    if bundle is None:
        return {
            "error": "Chưa có mô hình trong thư mục models/. Hãy chạy script huấn luyện trước.",
            "binh_luan": binh_luan,
            "sau_xu_ly": seg,
            "nhan": "",
            "nhan_id": -1,
            "do_tin_cay": 0.0,
            "diem_toxic": 0.0,
            "xac_suat_tung_nhan": {ten: 0.0 for ten in label_names},
            "mo_hinh": chosen_model or "Chưa có mô hình",
            "do_tre_ms": 0.0,
        }

    probs = _du_doan_xac_suat([seg], bundle, max_len=max_len)[0]
    do_tre = round((time.perf_counter() - t0) * 1000, 2)

    k = int(np.argmax(probs))
    diem_toxic = float(np.sum([probs[idx] for idx in toxic_ids if idx < len(probs)]))

    return {
        "binh_luan": binh_luan,
        "sau_xu_ly": seg,
        "nhan": label_names[k],
        "nhan_id": k,
        "do_tin_cay": round(float(probs[k]), 4),
        "diem_toxic": round(diem_toxic, 4),
        "xac_suat_tung_nhan": {
            label_names[i]: round(float(probs[i]), 4) for i in range(len(label_names))
        },
        "mo_hinh": chosen_model,
        "do_tre_ms": do_tre,
    }


def tim_tu_gay_doc(
    binh_luan: str,
    ten_mo_hinh: str | None = None,
    models_dir: str | None = None,
    nguong_giam: float = 0.03,
) -> list[dict]:
    """
    Giải thích mức từ (Leave-One-Token-Out):
    Bỏ lần lượt từng từ trong câu đã tách từ rồi tính lại điểm độc hại (diem_toxic).
    Từ nào khi bỏ đi làm diem_toxic giảm nhiều nhất (>= nguong_giam) thì được đánh dấu là từ gây độc.
    """
    kq_goc = phan_loai(binh_luan, ten_mo_hinh=ten_mo_hinh, models_dir=models_dir)
    if kq_goc.get("error") or not kq_goc.get("sau_xu_ly"):
        return []

    cfg = doc_cau_hinh(models_dir=models_dir)
    toxic_ids = cfg.get("toxic_ids", [1, 2])
    max_len = int(cfg.get("max_len", 128))
    chosen_model = kq_goc["mo_hinh"]
    bundle = tai_mo_hinh(chosen_model, models_dir=models_dir)
    if bundle is None:
        return []

    tokens = kq_goc["sau_xu_ly"].split()
    if len(tokens) == 0:
        return []

    diem_goc = float(kq_goc["diem_toxic"])
    if len(tokens) == 1:
        return [{"tu": tokens[0], "do_giam_toxic": round(diem_goc, 4), "gay_doc": diem_goc >= 0.5}]

    # Tạo danh sách câu đã lược bỏ từng từ một (Leave-One-Token-Out)
    cau_bo_tu = [" ".join(tokens[:i] + tokens[i + 1 :]) for i in range(len(tokens))]
    probs_loo = _du_doan_xac_suat(cau_bo_tu, bundle, max_len=max_len)

    def _logit(p_val: float) -> float:
        p_clip = float(np.clip(p_val, 1e-5, 1.0 - 1e-5))
        return float(np.log(p_clip / (1.0 - p_clip)))

    logit_goc = _logit(diem_goc)
    ket_qua_tu = []
    for i, tu in enumerate(tokens):
        p_loo = probs_loo[i]
        toxic_loo = float(np.sum([p_loo[idx] for idx in toxic_ids if idx < len(p_loo)]))
        do_giam_prob = diem_goc - toxic_loo
        do_giam_logit = logit_goc - _logit(toxic_loo)
        ket_qua_tu.append(
            {
                "tu": tu,
                "do_giam_toxic": round(float(do_giam_prob), 4),
                "do_giam_logit": round(float(do_giam_logit), 4),
                "gay_doc": bool(
                    diem_goc >= 0.30 and (do_giam_prob >= nguong_giam or do_giam_logit >= 0.35)
                ),
            }
        )
    return ket_qua_tu


def phan_loai_hang_loat(
    danh_sach: list[str] | pd.DataFrame,
    ten_mo_hinh: str | None = None,
    models_dir: str | None = None,
) -> pd.DataFrame:
    """
    Phân loại hàng loạt danh sách bình luận hoặc DataFrame (cột 'text').
    Thêm các cột: sau_xu_ly, nhan, do_tin_cay, diem_toxic.
    """
    cfg = doc_cau_hinh(models_dir=models_dir)
    label_names = cfg.get("label_names", ["Bình thường", "Xúc phạm", "Thù ghét"])
    toxic_ids = cfg.get("toxic_ids", [1, 2])
    max_len = int(cfg.get("max_len", 128))

    if isinstance(danh_sach, pd.DataFrame):
        df = danh_sach.copy()
        if "text" not in df.columns:
            df["text"] = df.iloc[:, 0].astype(str)
    else:
        df = pd.DataFrame({"text": [str(x) if x is not None else "" for x in danh_sach]})

    if df.empty:
        df["sau_xu_ly"] = []
        df["nhan"] = []
        df["do_tin_cay"] = []
        df["diem_toxic"] = []
        return df

    df["sau_xu_ly"] = df["text"].apply(lambda s: tach_tu(lam_sach(str(s))))
    avail = cfg.get("models_available", [])
    chosen_model = ten_mo_hinh if (ten_mo_hinh and ten_mo_hinh in avail) else cfg.get("best_model", "")
    if not chosen_model and avail:
        chosen_model = avail[0]

    bundle = tai_mo_hinh(chosen_model, models_dir=models_dir) if chosen_model else None
    if bundle is None:
        raise RuntimeError("Chưa có mô hình trong thư mục models/. Hãy chạy script huấn luyện trước.")

    valid_mask = df["sau_xu_ly"].str.strip().str.len() > 0
    probs_all = np.zeros((len(df), len(label_names)), dtype=np.float64)
    probs_all[:, 0] = 1.0  # Câu rỗng sau làm sạch mặc định là Bình thường (0)

    if valid_mask.any():
        valid_texts = df.loc[valid_mask, "sau_xu_ly"].tolist()
        probs_valid = _du_doan_xac_suat(valid_texts, bundle, max_len=max_len)
        probs_all[valid_mask.values, :] = probs_valid

    pred_ids = probs_all.argmax(axis=1)
    df["nhan"] = [label_names[i] for i in pred_ids]
    df["do_tin_cay"] = np.max(probs_all, axis=1).round(4)
    df["diem_toxic"] = probs_all[:, toxic_ids].sum(axis=1).round(4)
    for idx, ten_nhan in enumerate(label_names):
        df[f"xac_suat_{idx}"] = probs_all[:, idx].round(4)
    return df
