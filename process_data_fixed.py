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

TEENCODE = {'ko': 'không', 'k': 'không', 'kh': 'không', 'hok': 'không', 'khong': 'không', 'hông': 'không', 'dc': 'được', 'đc': 'được', 'dk': 'được', 'duoc': 'được', 'vs': 'với', 'j': 'gì', 'ntn': 'như thế nào', 'mn': 'mọi người', 'mk': 'mình', 'mik': 'mình', 'bn': 'bao nhiêu', 'bt': 'bình thường', 'đt': 'điện thoại', 'sp': 'sản phẩm', 'ae': 'anh em', 'ad': 'admin', 'cmt': 'bình luận', 'trc': 'trước', 'r': 'rồi', 'z': 'vậy', 'wa': 'quá', 'qa': 'quá', 'thik': 'thích', 'iu': 'yêu', 'bit': 'biết', 'ok': 'ổn', 'oke': 'ổn', 'dm': 'đm', 'dcm': 'đcm', 'đg': 'đang', 'lm': 'làm', 'ng': 'người', 'nc': 'nói chuyện', 'ib': 'nhắn tin', 'tks': 'cảm ơn', 'thanks': 'cảm ơn'}
RE_HTML = re.compile(r'<[^>]+>')
RE_URL = re.compile(r'(https?://\S+|www\.\S+|\bt\.me/\S+)', re.I)
RE_MENTION = re.compile(r'@\w+')
RE_EMOJI = re.compile(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]')
RE_CHEN_KT = re.compile(r'\b\w(?:[\.\-_]\w)+\b')
RE_TU = re.compile(r'\b(\w+)\b')

def fix_keo_dai(match):
    word = match.group(0)
    if word.isupper(): return word
    return re.sub(r'(.)\1{2,}', r'\1', word)

def lam_sach(text):
    t = unicodedata.normalize('NFC', str(text))
    t = RE_HTML.sub(' ', t)
    t = RE_URL.sub(' ', t)
    t = RE_MENTION.sub(' ', t)
    t = RE_EMOJI.sub(' ', t)
    t = re.sub(r'\b\w+\b', fix_keo_dai, t)
    t = RE_CHEN_KT.sub(lambda m: re.sub(r'[\.\-_]', '', m.group()), t)
    t = t.lower()
    t = RE_TU.sub(lambda m: TEENCODE.get(m.group(1), m.group(1)), t)
    return re.sub(r'\s+', ' ', t).strip()

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
def tach_tu(text_sach):
    t = text_sach.replace('*', 'X_STAR_X')
    t = ViTokenizer.tokenize(t)
    return t.replace('X_STAR_X', '*')
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

# Cập nhật notebook 01
notebook_path = "notebooks/01_du_lieu_va_tien_xu_ly.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_process = """# Bỏ trùng dựa trên text_clean
def get_mode(g):
    m = g.mode()
    if len(m) == 1: return m.iloc[0]
    return -1

labels = df.groupby('text_clean')['label'].apply(get_mode).reset_index()
labels = labels[labels['label'] != -1]

df_unique = df.drop_duplicates(subset=['text_clean'])[['text', 'text_clean']]
df_sach = df_unique.merge(labels, on='text_clean')

# Tách từ
df_sach["text_seg"] = df_sach["text_clean"].apply(tach_tu)
df_sach = df_sach[df_sach["text_seg"].str.len() > 0].copy().reset_index(drop=True)
df_sach.to_csv("data/processed/02_du_lieu_sach.csv", index=False)"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def resolve_duplicates" in source:
            cell['source'] = [line + "\\n" for line in cell_process.split("\\n")]
            cell['source'][-1] = cell['source'][-1].replace("\\n", "")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
