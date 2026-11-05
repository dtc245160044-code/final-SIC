import os
import time
import json
import inspect
import argparse
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["WANDB_DISABLED"] = "true"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

parser = argparse.ArgumentParser(description="Huấn luyện PhoBERT trên tập ViHSD đã tiền xử lý")
parser.add_argument("--smoke", action="store_true", help="Chạy thử nhanh (300 train, 100 val, 100 test, 1 epoch)")
parser.add_argument("--batch_size", type=int, default=16, help="per_device_train_batch_size (mặc định: 16, giảm xuống 8 nếu hết VRAM)")
parser.add_argument("--grad_accum", type=int, default=2, help="gradient_accumulation_steps (mặc định: 2, tăng lên 4 nếu giảm batch_size=8)")
parser.add_argument("--epochs", type=int, default=4, help="Số epoch huấn luyện (mặc định: 4)")
parser.add_argument("--model_dir", type=str, default=None, help="Thư mục lưu checkpoint và model tốt nhất")
parser.add_argument("--out_dir", type=str, default=None, help="Thư mục lưu file dự đoán CSV, biểu đồ và log")
parser.add_argument("--resume", action="store_true", help="Tiếp tục huấn luyện từ checkpoint gần nhất nếu bị ngắt kết nối")
args = parser.parse_args()

model_dir = args.model_dir or ("models/phobert_smoke" if args.smoke else "models/phobert")
out_dir = args.out_dir or ("outputs/smoke" if args.smoke else "outputs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"=== THIẾT BỊ CHẠY: {device_name} (CUDA available: {torch.cuda.is_available()}) ===")

print("--- 1. ĐỌC DỮ LIỆU ---")
train_df = pd.read_csv("data/processed/03_train.csv", keep_default_na=False)
val_df = pd.read_csv("data/processed/03_val.csv", keep_default_na=False)
test_df = pd.read_csv("data/processed/03_test.csv", keep_default_na=False)

train_df = train_df[train_df["text_seg"] != ""].reset_index(drop=True)
val_df = val_df[val_df["text_seg"] != ""].reset_index(drop=True)
test_df = test_df[test_df["text_seg"] != ""].reset_index(drop=True)

if args.smoke:
    train_df = train_df.head(300).reset_index(drop=True)
    val_df = val_df.head(100).reset_index(drop=True)
    test_df = test_df.head(100).reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

classes = np.sort(train_df["label"].unique())
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["label"])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()

print(f"Class weights: {class_weights}")

print("--- 2. TOKENIZATION ---")
MODEL_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_ds = Dataset.from_pandas(train_df[["text_seg", "label"]])
val_ds = Dataset.from_pandas(val_df[["text_seg", "label"]])
test_ds = Dataset.from_pandas(test_df[["text_seg", "label"]])

MAX_LEN = 128

def tokenize_function(examples):
    return tokenizer(examples["text_seg"], truncation=True, max_length=MAX_LEN)

rm_train = ["text_seg", "__index_level_0__"] if "__index_level_0__" in train_ds.column_names else ["text_seg"]
rm_val = ["text_seg", "__index_level_0__"] if "__index_level_0__" in val_ds.column_names else ["text_seg"]
rm_test = ["text_seg", "__index_level_0__"] if "__index_level_0__" in test_ds.column_names else ["text_seg"]

train_encoded = train_ds.map(tokenize_function, batched=True, remove_columns=rm_train)
val_encoded = val_ds.map(tokenize_function, batched=True, remove_columns=rm_val)
test_encoded = test_ds.map(tokenize_function, batched=True, remove_columns=rm_test)

train_encoded.set_format("torch")
val_encoded.set_format("torch")
test_encoded.set_format("torch")

print("--- 3. HUẤN LUYỆN PHOBERT ---")
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions),
    }

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

num_epochs = 1 if args.smoke else args.epochs
eff_batch = max(1, args.batch_size * args.grad_accum)
total_steps = max(1, (len(train_encoded) // eff_batch) * num_epochs)

training_args_kwargs = dict(
    output_dir=model_dir,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    per_device_eval_batch_size=32,
    num_train_epochs=num_epochs,
    warmup_steps=max(1, int(0.1 * total_steps)),
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    logging_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
if "eval_strategy" in valid_params:
    training_args_kwargs["eval_strategy"] = "epoch"
elif "evaluation_strategy" in valid_params:
    training_args_kwargs["evaluation_strategy"] = "epoch"

training_args = TrainingArguments(**training_args_kwargs)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    eval_dataset=val_encoded,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

last_ckpt = None
if args.resume and os.path.isdir(model_dir):
    last_ckpt = get_last_checkpoint(model_dir)
    if last_ckpt:
        print(f"=> Tìm thấy checkpoint gần nhất để tiếp tục: {last_ckpt}")

t0 = time.time()
trainer.train(resume_from_checkpoint=last_ckpt)
train_time_sec = round(time.time() - t0, 2)

best_dir = os.path.join(model_dir, "best")
trainer.save_model(best_dir)
tokenizer.save_pretrained(best_dir)

# Lưu lịch sử huấn luyện (loss + F1-macro Val mỗi epoch)
log_history = trainer.state.log_history
with open(os.path.join(out_dir, "phobert_training_log.json"), "w", encoding="utf-8") as f:
    json.dump(log_history, f, ensure_ascii=False, indent=2)

epoch_rows = {}
for entry in log_history:
    ep = entry.get("epoch")
    if ep is None:
        continue
    ep_key = int(round(float(ep)))
    if ep_key not in epoch_rows:
        epoch_rows[ep_key] = {"epoch": ep_key, "train_loss": None, "eval_loss": None, "eval_f1_macro": None, "eval_accuracy": None}
    if "loss" in entry and "eval_loss" not in entry:
        epoch_rows[ep_key]["train_loss"] = round(float(entry["loss"]), 4)
    if "eval_loss" in entry:
        epoch_rows[ep_key]["eval_loss"] = round(float(entry["eval_loss"]), 4)
    if "eval_f1_macro" in entry:
        epoch_rows[ep_key]["eval_f1_macro"] = round(float(entry["eval_f1_macro"]), 4)
    if "eval_accuracy" in entry:
        epoch_rows[ep_key]["eval_accuracy"] = round(float(entry["eval_accuracy"]), 4)

df_log = pd.DataFrame(sorted(epoch_rows.values(), key=lambda r: r["epoch"]))
df_log.to_csv(os.path.join(out_dir, "phobert_training_log.csv"), index=False)
print("\n=== LOG HUẤN LUYỆN THEO EPOCH (TRÊN TẬP VAL) ===")
print(df_log.to_string(index=False))

print("\n--- 4. ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT TRÊN TẬP VAL ---")
predictions = trainer.predict(val_encoded)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
preds = np.argmax(probs, axis=-1)

val_f1_macro = float(f1_score(val_df["label"], preds, average="macro"))
val_acc = float(accuracy_score(val_df["label"], preds))

val_results = val_df.copy()
val_results["pred"] = preds
for i in range(3):
    val_results[f"prob_{i}"] = probs[:, i]
val_results.to_csv(os.path.join(out_dir, "phobert_val_predictions.csv"), index=False)

target_names = ["CLEAN (0)", "OFFENSIVE (1)", "HATE (2)"]
cm = confusion_matrix(val_df["label"], preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix (PhoBERT) - Val")
plt.ylabel("Thực tế")
plt.xlabel("Dự đoán")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix_phobert_val.png"), dpi=300)
plt.close()

# Tính dung lượng thư mục best model (MB)
model_size_mb = round(
    sum(
        os.path.getsize(os.path.join(r, file))
        for r, _, files in os.walk(best_dir)
        for file in files
    )
    / (1024 * 1024),
    2,
)

meta_info = {
    "model_name": MODEL_NAME,
    "device": device_name,
    "epochs": num_epochs,
    "batch_size": args.batch_size,
    "grad_accum": args.grad_accum,
    "train_time_sec": train_time_sec,
    "model_size_mb": model_size_mb,
    "val_f1_macro": round(val_f1_macro, 4),
    "val_accuracy": round(val_acc, 4),
    "baseline_val_f1_macro_ref": 0.6819,
}
with open(os.path.join(out_dir, "phobert_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta_info, f, ensure_ascii=False, indent=2)

print(f"\nĐã lưu mô hình tốt nhất vào: {best_dir} ({model_size_mb} MB)")
print(f"Thời gian huấn luyện: {train_time_sec} giây")
print(f"Đã lưu dự đoán Val vào: {out_dir}/phobert_val_predictions.csv")
print(f"Đã lưu log huấn luyện vào: {out_dir}/phobert_training_log.csv")
print("\nClassification Report (Val):")
print(classification_report(val_df["label"], preds, target_names=target_names, digits=4))

print("==================================================================")
print(f"=> F1-macro (Val) của PhoBERT : {val_f1_macro:.4f}")
print(f"=> F1-macro (Val) của Baseline: 0.6819 (CalibratedSVC ở Bước 2)")
diff = val_f1_macro - 0.6819
print(f"=> Chênh lệch trên Val        : {diff:+.4f} ({'PhoBERT cao hơn' if diff >= 0 else 'Baseline cao hơn'})")
print("==================================================================")

print("\n--- 5. XUẤT XÁC SUẤT DỰ ĐOÁN TRÊN TẬP TEST (CHƯA CHẤM ĐIỂM TEST) ---")
test_preds = trainer.predict(test_encoded)
test_probs = torch.nn.functional.softmax(torch.tensor(test_preds.predictions), dim=-1).numpy()
test_results = test_df.copy()
test_results["pred"] = np.argmax(test_probs, axis=-1)
for i in range(3):
    test_results[f"prob_{i}"] = test_probs[:, i]
test_results.to_csv(os.path.join(out_dir, "phobert_test_predictions.csv"), index=False)
print(f"Đã lưu xác suất dự đoán Test vào: {out_dir}/phobert_test_predictions.csv (để dành chấm 1 lần ở Bước 4)")
