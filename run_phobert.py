import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset

os.environ["WANDB_DISABLED"] = "true"
os.makedirs("models/phobert", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("--- 1. ĐỌC DỮ LIỆU ---")
train_df = pd.read_csv("data/processed/03_train.csv", keep_default_na=False)
val_df = pd.read_csv("data/processed/03_val.csv", keep_default_na=False)
test_df = pd.read_csv("data/processed/03_test.csv", keep_default_na=False)

train_df = train_df[train_df['text_seg'] != '']
val_df = val_df[val_df['text_seg'] != '']
test_df = test_df[test_df['text_seg'] != '']

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

classes = np.sort(train_df['label'].unique())
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()

print("Class weights:", class_weights)

print("--- 2. TOKENIZATION ---")
MODEL_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = Dataset.from_pandas(train_df[['text_seg', 'label']])
val_ds = Dataset.from_pandas(val_df[['text_seg', 'label']])

MAX_LEN = 128
def tokenize_function(examples):
    return tokenizer(examples["text_seg"], padding="max_length", truncation=True, max_length=MAX_LEN)

train_encoded = train_ds.map(tokenize_function, batched=True, remove_columns=["text_seg", "__index_level_0__"] if "__index_level_0__" in train_ds.column_names else ["text_seg"])
val_encoded = val_ds.map(tokenize_function, batched=True, remove_columns=["text_seg", "__index_level_0__"] if "__index_level_0__" in val_ds.column_names else ["text_seg"])

train_encoded.set_format("torch")
val_encoded.set_format("torch")

print("--- 3. HUẤN LUYỆN PHOBERT ---")
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, predictions, average="macro")}

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

training_args = TrainingArguments(
    output_dir="models/phobert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    seed=SEED,
    logging_steps=100
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    eval_dataset=val_encoded,
    compute_metrics=compute_metrics
)

trainer.train()

print("--- 4. ĐÁNH GIÁ TRÊN TẬP VAL ---")
predictions = trainer.predict(val_encoded)
logits = predictions.predictions
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
preds = np.argmax(probs, axis=-1)

val_results = val_df.copy()
val_results['pred'] = preds
val_results['prob_0'] = probs[:, 0]
val_results['prob_1'] = probs[:, 1]
val_results['prob_2'] = probs[:, 2]
val_results.to_csv("outputs/phobert_val_predictions.csv", index=False)
print("Đã lưu xác suất dự đoán vào outputs/phobert_val_predictions.csv")

target_names = ["CLEAN (0)", "OFFENSIVE (1)", "HATE (2)"]
print("\\nClassification Report:")
print(classification_report(val_df['label'], preds, target_names=target_names))

cm = confusion_matrix(val_df['label'], preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (PhoBERT) - Val')
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_phobert_val.png", dpi=300)
print("Saved outputs/confusion_matrix_phobert_val.png")
