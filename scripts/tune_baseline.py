import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

SEED = 42
np.random.seed(SEED)

print("=== BƯỚC 2: THỬ NGHIỆM CẢI THIỆN BASELINE TRÊN TẬP VAL (KHÔNG ĐỤNG TEST) ===")
train = pd.read_csv("data/processed/03_train.csv", keep_default_na=False)
val = pd.read_csv("data/processed/03_val.csv", keep_default_na=False)

train = train[train["text_seg"] != ""].reset_index(drop=True)
val = val[val["text_seg"] != ""].reset_index(drop=True)

X_train, y_train = train["text_seg"], train["label"].values
X_val, y_val = val["text_seg"], val["label"].values

print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")

def make_features(word_ngram=(1, 2), char_ngram=(2, 5), min_df_w=3, min_df_c=5, sublinear=False):
    tfidf_w = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram,
        min_df=min_df_w,
        max_df=0.9,
        sublinear_tf=sublinear,
    )
    tfidf_c = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram,
        min_df=min_df_c,
        max_df=0.9,
        sublinear_tf=sublinear,
    )
    return FeatureUnion([("word_tfidf", tfidf_w), ("char_tfidf", tfidf_c)])

results = []
best_f1 = -1.0
best_name = ""
best_params = ""
best_pipeline = None
best_val_preds = None

def evaluate_and_log(group_name, exp_name, params_str, y_true, y_pred, pipeline_obj=None):
    global best_f1, best_name, best_params, best_pipeline, best_val_preds
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    prec, rec, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    row = {
        "nhom_thu_nghiem": group_name,
        "ten_thu_nghiem": exp_name,
        "tham_so": params_str,
        "accuracy_val": round(float(acc), 4),
        "f1_macro_val": round(float(f1_mac), 4),
        "precision_clean": round(float(prec[0]), 4),
        "recall_clean": round(float(rec[0]), 4),
        "f1_clean": round(float(f1_cls[0]), 4),
        "precision_offensive": round(float(prec[1]), 4),
        "recall_offensive": round(float(rec[1]), 4),
        "f1_offensive": round(float(f1_cls[1]), 4),
        "precision_hate": round(float(prec[2]), 4),
        "recall_hate": round(float(rec[2]), 4),
        "f1_hate": round(float(f1_cls[2]), 4),
    }
    results.append(row)
    print(
        f"[{exp_name}] F1-macro={f1_mac:.4f} | Acc={acc:.4f} | "
        f"OFFENSIVE(P={prec[1]:.3f}, R={rec[1]:.3f}, F1={f1_cls[1]:.3f}) | "
        f"HATE(F1={f1_cls[2]:.3f})"
    )
    if pipeline_obj is not None and f1_mac > best_f1:
        best_f1 = f1_mac
        best_name = exp_name
        best_params = params_str
        best_pipeline = pipeline_obj
        best_val_preds = y_pred

# Vectorizer gốc (như Bước 1) để dùng chung cho Nhóm 1, 2, 3
base_feats = make_features(word_ngram=(1, 2), char_ngram=(2, 5), min_df_w=3, min_df_c=5, sublinear=False)
print("\n--- Đang trích xuất đặc trưng TF-IDF cơ sở (word 1-2, char_wb 2-5) ---")
X_tr_base = base_feats.fit_transform(X_train)
X_va_base = base_feats.transform(X_val)

# Mốc 0: LinearSVC_C0.1 gốc (chưa calibrate, để đối chiếu trong bảng)
clf_raw = LinearSVC(C=0.1, class_weight="balanced", random_state=SEED, max_iter=2000)
clf_raw.fit(X_tr_base, y_train)
evaluate_and_log(
    "0_Moc_goc_chua_xac_suat",
    "LinearSVC_C0.1_uncalibrated",
    "word(1,2)+char_wb(2,5), LinearSVC(C=0.1, class_weight=balanced), khong co predict_proba",
    y_val,
    clf_raw.predict(X_va_base),
    pipeline_obj=None,  # Không chọn làm baseline_calibrated vì không có predict_proba
)

# ==============================================================================
# NHÓM 1: CalibratedClassifierCV(LinearSVC, cv=5) làm mốc chuẩn có xác suất
# ==============================================================================
print("\n--- NHÓM 1: CalibratedClassifierCV(LinearSVC, cv=5) ---")
for c_val in [0.05, 0.1, 0.2, 0.5, 1.0]:
    for cw_label, cw in [("balanced", "balanced"), ("None", None)]:
        base_svc = LinearSVC(C=c_val, class_weight=cw, random_state=SEED, max_iter=2000)
        cal_svc = CalibratedClassifierCV(estimator=base_svc, cv=5, method="sigmoid")
        cal_svc.fit(X_tr_base, y_train)
        pipe = Pipeline([("tfidf", base_feats), ("clf", cal_svc)])
        preds = cal_svc.predict(X_va_base)
        evaluate_and_log(
            "1_Calibrated_LinearSVC",
            f"CalibratedSVC_C{c_val}_cw-{cw_label}",
            f"word(1,2)+char_wb(2,5), CalibratedClassifierCV(LinearSVC(C={c_val}, class_weight={cw_label}), cv=5)",
            y_val,
            preds,
            pipeline_obj=pipe,
        )

# ==============================================================================
# NHÓM 2: So sánh với LogisticRegression
# ==============================================================================
print("\n--- NHÓM 2: LogisticRegression ---")
for c_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
    for cw_label, cw in [("balanced", "balanced"), ("None", None)]:
        lr = LogisticRegression(C=c_val, class_weight=cw, random_state=SEED, max_iter=1000)
        lr.fit(X_tr_base, y_train)
        pipe = Pipeline([("tfidf", base_feats), ("clf", lr)])
        preds = lr.predict(X_va_base)
        evaluate_and_log(
            "2_LogisticRegression",
            f"LogReg_C{c_val}_cw-{cw_label}",
            f"word(1,2)+char_wb(2,5), LogisticRegression(C={c_val}, class_weight={cw_label})",
            y_val,
            preds,
            pipeline_obj=pipe,
        )

# ==============================================================================
# NHÓM 3: Chỉnh trọng số lớp (class_weight) & ngưỡng quyết định cho OFFENSIVE
# ==============================================================================
print("\n--- NHÓM 3: Trọng số lớp tùy chỉnh & Ngưỡng quyết định cho OFFENSIVE ---")
# 3A. Trọng số lớp vừa phải trong LogisticRegression và CalibratedSVC (giúp cân bằng Precision/Recall của OFFENSIVE)
custom_weights = [
    ("w_1_2.5_2", {0: 1.0, 1: 2.5, 2: 2.0}),
    ("w_1_3_2", {0: 1.0, 1: 3.0, 2: 2.0}),
    ("w_1_3.5_2.5", {0: 1.0, 1: 3.5, 2: 2.5}),
    ("w_1_4_2.5", {0: 1.0, 1: 4.0, 2: 2.5}),
    ("w_1_5_3", {0: 1.0, 1: 5.0, 2: 3.0}),
]

for w_name, w_dict in custom_weights:
    # LogisticRegression với trọng số lớp tùy chỉnh
    for c_val in [1.0, 2.0]:
        lr = LogisticRegression(C=c_val, class_weight=w_dict, random_state=SEED, max_iter=1000)
        lr.fit(X_tr_base, y_train)
        pipe = Pipeline([("tfidf", base_feats), ("clf", lr)])
        evaluate_and_log(
            "3_Trong_so_va_Nguong",
            f"LogReg_C{c_val}_{w_name}",
            f"word(1,2)+char_wb(2,5), LogisticRegression(C={c_val}, class_weight={w_dict})",
            y_val,
            lr.predict(X_va_base),
            pipeline_obj=pipe,
        )

    # CalibratedClassifierCV(LinearSVC) có truyền sample_weight vừa phải khi fit
    sw = compute_sample_weight(class_weight=w_dict, y=y_train)
    base_svc = LinearSVC(C=0.1, random_state=SEED, max_iter=2000)
    cal_svc = CalibratedClassifierCV(estimator=base_svc, cv=5, method="sigmoid")
    cal_svc.fit(X_tr_base, y_train, sample_weight=sw)
    pipe_svc = Pipeline([("tfidf", base_feats), ("clf", cal_svc)])
    evaluate_and_log(
        "3_Trong_so_va_Nguong",
        f"CalibratedSVC_C0.1_sw-{w_name}",
        f"word(1,2)+char_wb(2,5), CalibratedSVC(LinearSVC(C=0.1), cv=5, sample_weight={w_dict})",
        y_val,
        cal_svc.predict(X_va_base),
        pipeline_obj=pipe_svc,
    )

# 3B. Điều chỉnh ngưỡng quyết định (Threshold / Prior adjustment) trên xác suất của CalibratedSVC_C0.1_cw-balanced
cal_svc_bal = CalibratedClassifierCV(
    estimator=LinearSVC(C=0.1, class_weight="balanced", random_state=SEED, max_iter=2000),
    cv=5,
    method="sigmoid",
)
cal_svc_bal.fit(X_tr_base, y_train)
val_probs_bal = cal_svc_bal.predict_proba(X_va_base)

for th_off in [0.20, 0.25, 0.30, 0.35]:
    for th_hate in [0.30, 0.35, 0.40]:
        # Quy tắc ngưỡng ưu tiên: nếu prob_2 >= th_hate hoặc prob_1 >= th_off
        preds_th = np.zeros(len(y_val), dtype=int)
        for i, p in enumerate(val_probs_bal):
            if p[2] >= th_hate and p[2] >= p[1]:
                preds_th[i] = 2
            elif p[1] >= th_off:
                preds_th[i] = 1
            elif p[2] >= th_hate:
                preds_th[i] = 2
            else:
                preds_th[i] = int(np.argmax(p))
        evaluate_and_log(
            "3_Trong_so_va_Nguong",
            f"CalibratedSVC_C0.1_thresh_off{th_off}_hate{th_hate}",
            f"CalibratedSVC(C=0.1, cw=balanced) + threshold(off>={th_off}, hate>={th_hate})",
            y_val,
            preds_th,
            pipeline_obj=None,  # Ghi nhận kết quả thử ngưỡng vào CSV
        )

# ==============================================================================
# NHÓM 4: Thử thêm n-gram và sublinear_tf
# ==============================================================================
print("\n--- NHÓM 4: Thử nghiệm đặc trưng TF-IDF (n-gram, sublinear_tf) ---")
feature_configs = [
    ("sublinear_w12_c25", (1, 2), (2, 5), 3, 5, True),
    ("sublinear_w13_c25", (1, 3), (2, 5), 2, 5, True),
    ("sublinear_w13_c26", (1, 3), (2, 6), 2, 5, True),
    ("rawtf_w13_c26", (1, 3), (2, 6), 2, 5, False),
]

for f_name, w_ng, c_ng, m_w, m_c, sub_tf in feature_configs:
    feats = make_features(word_ngram=w_ng, char_ngram=c_ng, min_df_w=m_w, min_df_c=m_c, sublinear=sub_tf)
    X_tr_f = feats.fit_transform(X_train)
    X_va_f = feats.transform(X_val)

    # 4a. CalibratedSVC C=0.1, C=0.2 (natural calibration)
    for c_val in [0.1, 0.2]:
        cal = CalibratedClassifierCV(
            estimator=LinearSVC(C=c_val, class_weight="balanced", random_state=SEED, max_iter=2000),
            cv=5,
            method="sigmoid",
        )
        cal.fit(X_tr_f, y_train)
        pipe = Pipeline([("tfidf", feats), ("clf", cal)])
        evaluate_and_log(
            "4_Dac_trung_TFIDF",
            f"CalibratedSVC_C{c_val}_{f_name}",
            f"word{w_ng}+char_wb{c_ng}, min_df=({m_w},{m_c}), sublinear_tf={sub_tf}, CalibratedSVC(C={c_val}, cw=balanced)",
            y_val,
            cal.predict(X_va_f),
            pipeline_obj=pipe,
        )

    # 4b. CalibratedSVC C=0.1 với sample_weight vừa phải {0:1, 1:2.0, 2:1.5}, {0:1, 1:2.5, 2:2.0}, {0:1, 1:3.0, 2:2.0}
    for w_tag, w_d in [
        ("w_1_2.0_1.5", {0: 1.0, 1: 2.0, 2: 1.5}),
        ("w_1_2.5_2.0", {0: 1.0, 1: 2.5, 2: 2.0}),
        ("w_1_3.0_2.0", {0: 1.0, 1: 3.0, 2: 2.0}),
    ]:
        sw = compute_sample_weight(class_weight=w_d, y=y_train)
        cal_sw = CalibratedClassifierCV(
            estimator=LinearSVC(C=0.1, random_state=SEED, max_iter=2000),
            cv=5,
            method="sigmoid",
        )
        cal_sw.fit(X_tr_f, y_train, sample_weight=sw)
        pipe_sw = Pipeline([("tfidf", feats), ("clf", cal_sw)])
        evaluate_and_log(
            "4_Dac_trung_TFIDF",
            f"CalibratedSVC_C0.1_{f_name}_{w_tag}",
            f"word{w_ng}+char_wb{c_ng}, sublinear_tf={sub_tf}, CalibratedSVC(C=0.1, sample_weight={w_d})",
            y_val,
            cal_sw.predict(X_va_f),
            pipeline_obj=pipe_sw,
        )

    # 4c. LogisticRegression với class_weight tùy chỉnh trên đặc trưng mới
    for w_tag, w_d in [("w_1_2.5_2.0", {0: 1.0, 1: 2.5, 2: 2.0}), ("w_1_3.0_2.0", {0: 1.0, 1: 3.0, 2: 2.0})]:
        lr_f = LogisticRegression(C=2.0, class_weight=w_d, random_state=SEED, max_iter=1000)
        lr_f.fit(X_tr_f, y_train)
        pipe_lr = Pipeline([("tfidf", feats), ("clf", lr_f)])
        evaluate_and_log(
            "4_Dac_trung_TFIDF",
            f"LogReg_C2.0_{f_name}_{w_tag}",
            f"word{w_ng}+char_wb{c_ng}, sublinear_tf={sub_tf}, LogisticRegression(C=2.0, class_weight={w_d})",
            y_val,
            lr_f.predict(X_va_f),
            pipeline_obj=pipe_lr,
        )

# Lưu toàn bộ lịch sử thử nghiệm ra CSV
df_res = pd.DataFrame(results)
csv_out = "outputs/thu_nghiem_baseline.csv"
df_res.to_csv(csv_out, index=False, encoding="utf-8")
print(f"\nĐã lưu toàn bộ {len(df_res)} thử nghiệm vào: {csv_out}")

print("\n=== TOP 10 THỬ NGHIỆM THEO F1-MACRO TRÊN TẬP VAL ===")
top10 = df_res.sort_values("f1_macro_val", ascending=False).head(10)
print(
    top10[
        [
            "ten_thu_nghiem",
            "f1_macro_val",
            "accuracy_val",
            "precision_offensive",
            "recall_offensive",
            "f1_offensive",
            "f1_hate",
        ]
    ].to_string(index=False)
)

print(f"\n=> MÔ HÌNH ĐƯỢC CHỌN LƯU: {best_name} (F1-macro Val = {best_f1:.4f})")
print(f"   Tham số: {best_params}")

target_names = ["CLEAN (0)", "OFFENSIVE (1)", "HATE (2)"]
print("\nClassification Report trên Val của mô hình được chọn:")
print(classification_report(y_val, best_val_preds, target_names=target_names, digits=4))

# Vẽ Confusion Matrix trên Val cho mô hình được chọn
cm = confusion_matrix(y_val, best_val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.title(f"Confusion Matrix (Val) - {best_name}")
plt.ylabel("Thực tế")
plt.xlabel("Dự đoán")
plt.tight_layout()
cm_path = "outputs/confusion_matrix_baseline_calibrated_val.png"
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Đã lưu Confusion Matrix (Val) vào: {cm_path}")

# Lưu mô hình tốt nhất vào models/baseline_calibrated.pkl
model_path = "models/baseline_calibrated.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_pipeline, f)
print(f"Đã lưu mô hình vào: {model_path} (vẫn giữ nguyên models/baseline_LinearSVC_C0.1.pkl)")

