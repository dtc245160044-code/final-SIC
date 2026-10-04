import os, re, unicodedata, json
import pandas as pd
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split

print("Đọc dữ liệu...")
df = pd.read_csv('data/raw/comments.csv')
if 'free_text' in df.columns: df.rename(columns={'free_text': 'text'}, inplace=True)
if 'label_id' in df.columns: df.rename(columns={'label_id': 'label'}, inplace=True)
df['text'] = df['text'].astype(str)
LABEL_NAMES = ['CLEAN', 'OFFENSIVE', 'HATE']
if df['label'].dtype == object: df['label'] = df['label'].map({ten: i for i, ten in enumerate(LABEL_NAMES)})
df['label'] = df['label'].astype(int)
df = df[df['text'] != '#ERROR!']

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import lam_sach, tach_tu
print("Làm sạch...")
df['text_clean'] = df['text'].apply(lam_sach)
df = df[df['text_clean'] != '']

print("Bỏ trùng...")
def get_mode(g):
    m = g.mode()
    if len(m) == 1: return m.iloc[0]
    return -1
labels = df.groupby('text_clean')['label'].apply(get_mode).reset_index()
labels = labels[labels['label'] != -1]
df_sach = df.drop_duplicates(subset=['text_clean'])[['text', 'text_clean']].merge(labels, on='text_clean')

print("Tách từ...")

df_sach['text_seg'] = df_sach['text_clean'].apply(tach_tu)
df_sach = df_sach[df_sach['text_seg'].str.len() > 0].copy().reset_index(drop=True)
df_sach.to_csv('data/processed/02_du_lieu_sach.csv', index=False)

print("Chia tập...")
train_val, test = train_test_split(df_sach, test_size=0.10, stratify=df_sach['label'], random_state=42)
train, val = train_test_split(train_val, test_size=1/9, stratify=train_val['label'], random_state=42)
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

train.to_csv('data/processed/03_train.csv', index=False)
val.to_csv('data/processed/03_val.csv', index=False)
test.to_csv('data/processed/03_test.csv', index=False)

print(f"Train: {len(train)}")
print(f"Val: {len(val)}")
print(f"Test: {len(test)}")


