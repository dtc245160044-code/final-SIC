import os
import re
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocess import phat_hien
from src.inference import tai_mo_hinh, _du_doan_xac_suat, BASELINE_NAME, PHOBERT_NAME
from scripts.build_config import build_config

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

LABEL_NAMES_VI = ["Bình thường", "Xúc phạm", "Thù ghét"]
LABEL_CODES = ["CLEAN", "OFFENSIVE", "HATE"]
TOXIC_IDS = [1, 2]

print("=== GIAI ĐOẠN B: ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (CHẤM ĐÚNG 1 LẦN) ===")
test_path = os.path.join(ROOT_DIR, "data", "processed", "03_test.csv")
test_df = pd.read_csv(test_path, keep_default_na=False)
test_df = test_df[test_df["text_seg"] != ""].reset_index(drop=True)

X_test = test_df["text_seg"].tolist()
y_test = test_df["label"].astype(int).values
print(f"Số mẫu tập Test: {len(test_df):,}")


def do_do_tre_ms(bundle: dict, texts: list[str], n: int = 100) -> float:
    """Đo thời gian suy luận trung bình cho 1 bình luận (ms/bình luận)."""
    if bundle is None:
        return 0.0
    mau = texts[: min(n, len(texts))]
    _du_doan_xac_suat(mau[:3], bundle)
    t0 = time.perf_counter()
    for t in mau:
        _du_doan_xac_suat([t], bundle)
    return round((time.perf_counter() - t0) / len(mau) * 1000, 2)


def tinh_dung_luong_mb(path_str: str) -> float:
    if not path_str or not os.path.exists(path_str):
        return 0.0
    if os.path.isfile(path_str):
        return round(os.path.getsize(path_str) / (1024 * 1024), 2)
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(path_str)
        for f in files
    )
    return round(total / (1024 * 1024), 2)


ket_qua_mo_hinh = []
du_doan_proba = {}

# 1. Đánh giá Baseline (TF-IDF + LinearSVC Calibrated)
baseline_bundle = tai_mo_hinh(BASELINE_NAME)
baseline_pred_csv = os.path.join(OUTPUTS_DIR, "baseline_test_predictions.csv")

if baseline_bundle is not None:
    if os.path.isfile(baseline_pred_csv):
        print(f"Đọc dự đoán Test đã lưu của {BASELINE_NAME} từ {baseline_pred_csv}...")
        df_b_pred = pd.read_csv(baseline_pred_csv)
        proba_base = df_b_pred[["prob_0", "prob_1", "prob_2"]].values
    else:
        print(f"Đang chạy dự đoán Test lần đầu cho {BASELINE_NAME}...")
        proba_base = _du_doan_xac_suat(X_test, baseline_bundle)
        df_b_out = test_df.copy()
        df_b_out["pred"] = proba_base.argmax(axis=1)
        for i in range(3):
            df_b_out[f"prob_{i}"] = proba_base[:, i]
        df_b_out.to_csv(baseline_pred_csv, index=False)

    latency_base = do_do_tre_ms(baseline_bundle, X_test, n=100)
    size_base = tinh_dung_luong_mb(baseline_bundle["path"])
    du_doan_proba[BASELINE_NAME] = (proba_base, latency_base, size_base)

# 2. Đánh giá PhoBERT (nếu có thư mục models/phobert/best hoặc file outputs/phobert_test_predictions.csv)
phobert_bundle = tai_mo_hinh(PHOBERT_NAME)
phobert_pred_csv = os.path.join(OUTPUTS_DIR, "phobert_test_predictions.csv")
phobert_meta_json = os.path.join(OUTPUTS_DIR, "phobert_meta.json")

if phobert_bundle is not None or os.path.isfile(phobert_pred_csv):
    if os.path.isfile(phobert_pred_csv):
        print(f"Đọc dự đoán Test của {PHOBERT_NAME} từ {phobert_pred_csv}...")
        df_p_pred = pd.read_csv(phobert_pred_csv)
        proba_pho = df_p_pred[["prob_0", "prob_1", "prob_2"]].values
    else:
        print(f"Đang chạy dự đoán Test cho {PHOBERT_NAME}...")
        proba_pho = _du_doan_xac_suat(X_test, phobert_bundle)
        df_p_out = test_df.copy()
        df_p_out["pred"] = proba_pho.argmax(axis=1)
        for i in range(3):
            df_p_out[f"prob_{i}"] = proba_pho[:, i]
        df_p_out.to_csv(phobert_pred_csv, index=False)

    latency_pho = do_do_tre_ms(phobert_bundle, X_test, n=25) if phobert_bundle else 0.0
    size_pho = tinh_dung_luong_mb(phobert_bundle["path"]) if phobert_bundle else 0.0
    if os.path.isfile(phobert_meta_json):
        try:
            with open(phobert_meta_json, "r", encoding="utf-8") as fm:
                meta_p = json.load(fm)
            if size_pho == 0.0:
                size_pho = float(meta_p.get("model_size_mb", 0.0))
        except Exception:
            pass
    du_doan_proba[PHOBERT_NAME] = (proba_pho, latency_pho, size_pho)
else:
    print("ℹ️ Chưa có PhoBERT (models/phobert/best hoặc outputs/phobert_test_predictions.csv). Chỉ chấm Baseline hiện có.")

# Tính bảng chỉ số cho từng mô hình trên tập Test
for ten_mh, (proba, do_tre, mb) in du_doan_proba.items():
    y_pred = proba.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    _, _, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)

    y_true_toxic = np.isin(y_test, TOXIC_IDS).astype(int)
    y_pred_toxic = np.isin(y_pred, TOXIC_IDS).astype(int)
    tox_prec = precision_score(y_true_toxic, y_pred_toxic, zero_division=0)
    tox_rec = recall_score(y_true_toxic, y_pred_toxic, zero_division=0)

    hang = {
        "Mô hình": ten_mh,
        "Accuracy (%)": round(float(acc * 100), 2),
        "F1-macro (%)": round(float(f1_mac * 100), 2),
        "Toxic Precision (%)": round(float(tox_prec * 100), 2),
        "Toxic Recall (%)": round(float(tox_rec * 100), 2),
        "F1 CLEAN (%)": round(float(f1_per_class[0] * 100), 2),
        "F1 OFFENSIVE (%)": round(float(f1_per_class[1] * 100), 2),
        "F1 HATE (%)": round(float(f1_per_class[2] * 100), 2),
        "Độ trễ (ms/bình luận)": round(float(do_tre), 2),
        "Kích thước (MB)": round(float(mb), 2),
    }
    ket_qua_mo_hinh.append(hang)

    print(f"\n--- KẾT QUẢ TRÊN TẬP TEST: {ten_mh} ---")
    print(classification_report(y_test, y_pred, target_names=[f"{LABEL_CODES[i]} ({i})" for i in range(3)], digits=4))

    # Vẽ Ma trận nhầm lẫn (Số lượng + % theo hàng) đúng chuẩn bản vẽ
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_pt = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES_VI, yticklabels=LABEL_NAMES_VI, ax=ax[0])
    ax[0].set_title(f"{ten_mh}: Số lượng (Test)")
    ax[0].set_xlabel("Mô hình đoán")
    ax[0].set_ylabel("Nhãn thật")

    sns.heatmap(cm_pt, annot=True, fmt=".1f", cmap="Oranges", xticklabels=LABEL_NAMES_VI, yticklabels=LABEL_NAMES_VI, ax=ax[1])
    ax[1].set_title(f"{ten_mh}: % theo hàng (Recall)")
    ax[1].set_xlabel("Mô hình đoán")
    ax[1].set_ylabel("Nhãn thật")

    plt.tight_layout()
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", ten_mh).strip("_")
    cm_out_path = os.path.join(OUTPUTS_DIR, f"ma_tran_nham_lan_{safe_name}.png")
    plt.savefig(cm_out_path, dpi=180)
    plt.close()
    print(f"Đã lưu Ma trận nhầm lẫn: {cm_out_path}")

# Lưu bảng so sánh mô hình
df_ss = pd.DataFrame(ket_qua_mo_hinh)
comp_csv_path = os.path.join(OUTPUTS_DIR, "bang_so_sanh_mo_hinh.csv")
df_ss.to_csv(comp_csv_path, index=False)
print(f"\n=== BẢNG SO SÁNH MÔ HÌNH TRÊN TẬP TEST ({comp_csv_path}) ===")
print(df_ss.to_string(index=False))

# Vẽ biểu đồ so sánh mô hình (outputs/so_sanh_mo_hinh.png)
chi_so_ve = ["Accuracy (%)", "F1-macro (%)", "Toxic Precision (%)", "Toxic Recall (%)", "F1 OFFENSIVE (%)", "F1 HATE (%)"]
df_melt = df_ss.melt(id_vars="Mô hình", value_vars=chi_so_ve, var_name="Chỉ số", value_name="Giá trị (%)")
plt.figure(figsize=(10, 4.5))
sns.barplot(data=df_melt, x="Chỉ số", y="Giá trị (%)", hue="Mô hình", palette="Set2")
plt.ylim(0, 105)
plt.ylabel("%")
plt.xlabel("")
plt.title("So sánh hiệu năng trên tập Test (2.986 bình luận)")
plt.xticks(rotation=12)
plt.legend(loc="upper right")
plt.tight_layout()
comp_png_path = os.path.join(OUTPUTS_DIR, "so_sanh_mo_hinh.png")
plt.savefig(comp_png_path, dpi=180)
plt.close()
print(f"Đã lưu biểu đồ so sánh: {comp_png_path}")

# Chọn mô hình tốt nhất theo F1-macro (%) trên tập Test
best_row = df_ss.sort_values(by="F1-macro (%)", ascending=False).iloc[0]
ten_tot_nhat = str(best_row["Mô hình"])
proba_tot, _, _ = du_doan_proba[ten_tot_nhat]
y_pred_tot = proba_tot.argmax(axis=1)

# ==============================================================================
# MỤC 7: XUẤT outputs/phan_tich_loi.csv (TỐI THIỂU 30 FP VÀ 30 FN CÓ CHỦ ĐÍCH)
# ==============================================================================
loi_df = test_df.copy()
loi_df["nhãn_thật"] = [LABEL_NAMES_VI[i] for i in y_test]
loi_df["nhãn_đoán"] = [LABEL_NAMES_VI[i] for i in y_pred_tot]
loi_df["xác_suất_đoán"] = proba_tot.max(axis=1).round(4)
loi_df["xác_suất_toxic"] = proba_tot[:, TOXIC_IDS].sum(axis=1).round(4)

that_toxic = np.isin(y_test, TOXIC_IDS)
doan_toxic = np.isin(y_pred_tot, TOXIC_IDS)
loi_df["loại_lỗi"] = np.select(
    [(~that_toxic) & doan_toxic, that_toxic & (~doan_toxic)],
    ["FP", "FN"],
    default="",
)

dh_series = loi_df["text"].apply(phat_hien).apply(pd.Series)
loi_df = pd.concat([loi_df, dh_series], axis=1)
loi_df["so_tu"] = loi_df["text_seg"].str.split().str.len()

TU_PHU_DINH = {"không", "chẳng", "chưa", "đâu", "chả", "đừng", "khỏi"}
TU_DUA_CUOI = ["haha", "hihi", "kkk", "=))", ":))", ":v", "😂", "🤣", "😅", "😆", "lol", "vl", "vcl"]
TU_TUC_THONG_DUNG = ["đm", "đcm", "vl", "vcl", "đéo", "cc", "clm", "vãi", "ngu", "chó", "đĩ", "lồn", "cặc"]
TU_MIA_MAI = ["dạy đời", "thượng đẳng", "nhất bạn", "chịu rồi", "bố đời", "tự nhục", "khôn thế", "giỏi quá", "chuyên gia", "thánh"]


def goi_y_nguyen_nhan_chi_tiet(r) -> str:
    ly_do = []
    raw_low = str(r["text"]).lower()
    clean_words = set(str(r["text_clean"]).split())

    if r["loại_lỗi"] == "FP":
        if any(w in raw_low for w in TU_DUA_CUOI) and any(t in raw_low for t in TU_TUC_THONG_DUNG):
            ly_do.append("từ tục/lóng dùng trong ngữ cảnh đùa vui hoặc cảm thán")
        elif any(t in raw_low for t in TU_TUC_THONG_DUNG):
            ly_do.append("bắt nhầm từ khóa nhạy cảm dù câu không nhằm công kích")
        if any(w in raw_low for w in TU_DUA_CUOI):
            ly_do.append("có biểu tượng cười/trêu đùa (=)), :v, haha)")
    elif r["loại_lỗi"] == "FN":
        if any(m in raw_low for m in TU_MIA_MAI) or ("?" in raw_low and not any(t in raw_low for t in TU_TUC_THONG_DUNG)):
            ly_do.append("mỉa mai/châm biếm ngầm không dùng từ chửi trực diện")
        elif not any(t in raw_low for t in TU_TUC_THONG_DUNG):
            ly_do.append("nói giảm nói tránh hoặc xúc phạm bằng từ vựng thông thường")

    if r.get("chen_kt", 0) > 0 or "*" in raw_low:
        ly_do.append("chèn ký tự/dấu * để né bộ lọc")
    if r.get("teencode", 0) > 0:
        ly_do.append("có teencode/viết tắt")
    if r.get("keo_dai", 0) > 0:
        ly_do.append("chữ kéo dài")
    if r.get("emoji", 0) > 0:
        ly_do.append("có emoji")
    if r.get("so_tu", 0) <= 5:
        ly_do.append("câu rất ngắn (<=5 từ), thiếu ngữ cảnh")
    if TU_PHU_DINH & clean_words:
        ly_do.append("có từ phủ định dễ làm đảo nghĩa")
    if (r["loại_lỗi"] == "FP" and r["xác_suất_toxic"] >= 0.85) or (r["loại_lỗi"] == "FN" and r["xác_suất_toxic"] <= 0.12):
        ly_do.append("mô hình tự tin cao ngược với nhãn gốc (nhãn gốc có thể gây tranh cãi)")

    if not ly_do:
        ly_do.append("thiếu ngữ cảnh hội thoại bên ngoài bình luận")
    return "; ".join(ly_do)


loi_df["gợi_ý_nguyên_nhân"] = loi_df.apply(goi_y_nguyen_nhan_chi_tiet, axis=1)

fp_all = loi_df[loi_df["loại_lỗi"] == "FP"].sort_values(by="xác_suất_toxic", ascending=False)
fn_all = loi_df[loi_df["loại_lỗi"] == "FN"].sort_values(by="xác_suất_toxic", ascending=True)

n_fp_total = len(fp_all)
n_fn_total = len(fn_all)
print(f"\nTổng số lỗi nhóm Toxic trên Test: FP (Báo nhầm) = {n_fp_total} | FN (Bỏ sót) = {n_fn_total}")

# Lấy tối thiểu 30 FP và 30 FN (ưu tiên đa dạng nguyên nhân + độ tự tin sai cao nhất)
fp_selected = fp_all.head(max(30, min(35, len(fp_all))))
fn_selected = fn_all.head(max(30, min(35, len(fn_all))))

xuat_loi = pd.concat([fp_selected, fn_selected], ignore_index=True)
xuat_loi = xuat_loi.rename(columns={"text": "bình_luận", "text_seg": "sau_xử_lý"})
cot_xuat = [
    "bình_luận",
    "sau_xử_lý",
    "nhãn_thật",
    "nhãn_đoán",
    "loại_lỗi",
    "xác_suất_toxic",
    "gợi_ý_nguyên_nhân",
]
xuat_loi = xuat_loi[cot_xuat].copy()
xuat_loi["nhận_xét_của_nhóm"] = ""

phan_tich_loi_path = os.path.join(OUTPUTS_DIR, "phan_tich_loi.csv")
xuat_loi.to_csv(phan_tich_loi_path, index=False)
print(f"Đã lưu {len(xuat_loi)} mẫu lỗi ({len(fp_selected)} FP + {len(fn_selected)} FN) vào: {phan_tich_loi_path}")

# ==============================================================================
# MỤC 9: PHÂN TÍCH BỔ SUNG CHO BÁO CÁO
# 9.1. F1-macro theo độ dài câu
# ==============================================================================
loi_df["nhom_do_dai"] = pd.cut(
    loi_df["so_tu"],
    bins=[0, 10, 25, 10000],
    labels=["Ngắn (<=10 từ)", "Trung bình (11-25 từ)", "Dài (>25 từ)"],
)

rows_len = []
for nhom_ten, grp in loi_df.groupby("nhom_do_dai", observed=False):
    if len(grp) == 0:
        continue
    yt_g = grp["label"].values
    yp_g = y_pred_tot[grp.index]
    f1_m = f1_score(yt_g, yp_g, average="macro", zero_division=0)
    acc_g = accuracy_score(yt_g, yp_g)
    _, _, f1_c, _ = precision_recall_fscore_support(yt_g, yp_g, labels=[0, 1, 2], zero_division=0)
    rows_len.append(
        {
            "Nhóm độ dài câu": str(nhom_ten),
            "Số bình luận": len(grp),
            "Tỷ lệ mẫu (%)": round(len(grp) / len(loi_df) * 100, 2),
            "Accuracy (%)": round(float(acc_g * 100), 2),
            "F1-macro (%)": round(float(f1_m * 100), 2),
            "F1 CLEAN (%)": round(float(f1_c[0] * 100), 2),
            "F1 OFFENSIVE (%)": round(float(f1_c[1] * 100), 2),
            "F1 HATE (%)": round(float(f1_c[2] * 100), 2),
        }
    )

df_len = pd.DataFrame(rows_len)
len_csv_path = os.path.join(OUTPUTS_DIR, "phan_tich_do_dai_cau.csv")
df_len.to_csv(len_csv_path, index=False)
print("\n=== 9.1. HIỆU NĂNG THEO ĐỘ DÀI CÂU TRÊN TẬP TEST ===")
print(df_len.to_string(index=False))

plt.figure(figsize=(7.5, 4))
sns.barplot(data=df_len, x="Nhóm độ dài câu", y="F1-macro (%)", hue="Nhóm độ dài câu", palette="Blues_d", legend=False)
plt.ylim(0, 100)
plt.title(f"F1-macro theo độ dài câu trên tập Test ({ten_tot_nhat})")
plt.ylabel("F1-macro (%)")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "f1_theo_do_dai_cau.png"), dpi=160)
plt.close()

# ==============================================================================
# 9.2. Tỷ lệ lỗi với câu có teencode / chèn ký tự / chữ kéo dài / emoji
# ==============================================================================
loi_df["sai_nhan"] = (y_test != y_pred_tot).astype(int)
nhom_nhieu = [
    ("Câu có teencode/viết tắt", loi_df["teencode"] > 0),
    ("Câu chèn ký tự / dấu * né lọc", (loi_df["chen_kt"] > 0) | (loi_df["text"].str.contains(r"\*", regex=True))),
    ("Câu có chữ kéo dài", loi_df["keo_dai"] > 0),
    ("Câu có emoji", loi_df["emoji"] > 0),
    ("Câu không chứa các dấu hiệu trên", (loi_df["teencode"] == 0) & (loi_df["chen_kt"] == 0) & (loi_df["keo_dai"] == 0) & (loi_df["emoji"] == 0)),
]

rows_nhieu = []
for ten_nhom, mask in nhom_nhieu:
    sub = loi_df[mask]
    if len(sub) == 0:
        continue
    yt_s = sub["label"].values
    yp_s = y_pred_tot[sub.index]
    rows_nhieu.append(
        {
            "Nhóm bình luận": ten_nhom,
            "Số mẫu trên Test": len(sub),
            "Tỷ lệ lỗi đoán sai nhãn (%)": round(float(sub["sai_nhan"].mean() * 100), 2),
            "Accuracy (%)": round(float((1 - sub["sai_nhan"].mean()) * 100), 2),
            "F1-macro (%)": round(float(f1_score(yt_s, yp_s, average="macro", zero_division=0) * 100), 2),
        }
    )

df_nhieu = pd.DataFrame(rows_nhieu)
nhieu_csv_path = os.path.join(OUTPUTS_DIR, "phan_tich_nhieu_dac_biet.csv")
df_nhieu.to_csv(nhieu_csv_path, index=False)
print("\n=== 9.2. TỶ LỆ LỖI THEO ĐẶC TÍNH NHIỄU TRÊN TẬP TEST ===")
print(df_nhieu.to_string(index=False))

# ==============================================================================
# 9.3. Top 15 câu OFFENSIVE (nhãn 1) bị đoán sai nhiều nhất kèm nhận xét
# ==============================================================================
off_wrong = loi_df[(loi_df["label"] == 1) & (y_pred_tot != 1)].copy()
off_wrong["xac_suat_nhan_sai"] = proba_tot[off_wrong.index, y_pred_tot[off_wrong.index]].round(4)
off_wrong["xac_suat_offensive"] = proba_tot[off_wrong.index, 1].round(4)
off_wrong = off_wrong.sort_values(by="xac_suat_nhan_sai", ascending=False).head(15)

rows_off15 = []
for _, r in off_wrong.iterrows():
    nh_doan = r["nhãn_đoán"]
    if nh_doan == "Bình thường":
        nx = f"Bỏ sót sang Bình thường (P={r['xac_suat_nhan_sai']}): {r['gợi_ý_nguyên_nhân']}."
    else:
        nx = f"Nhầm mức độ sang Thù ghét (P={r['xac_suat_nhan_sai']}): Câu chứa từ công kích mạnh khiến mô hình xếp lên mức HATE."
    rows_off15.append(
        {
            "bình_luận": r["text"],
            "sau_xử_lý": r["text_seg"],
            "nhãn_thật": "Xúc phạm (OFFENSIVE)",
            "nhãn_đoán": nh_doan,
            "xác_suất_nhãn_đoán": r["xac_suat_nhan_sai"],
            "xác_suất_offensive": r["xac_suat_offensive"],
            "nhận_xét_phân_tích": nx,
        }
    )

df_off15 = pd.DataFrame(rows_off15)
off15_csv_path = os.path.join(OUTPUTS_DIR, "top15_offensive_doan_sai.csv")
df_off15.to_csv(off15_csv_path, index=False)
print(f"\nĐã lưu 15 câu OFFENSIVE bị đoán sai điển hình vào: {off15_csv_path}")

# MỤC 8: Cập nhật lại models/config.json
print("\n--- CẬP NHẬT models/config.json ---")
build_config()
