import os
import sys
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ui import (
    cai_dat_trang,
    the_chi_so,
    thong_bao_thieu_file,
    hien_chan_trang_canh_bao,
    OUTPUTS_DIR,
    DATA_PROCESSED_DIR,
)

cfg = cai_dat_trang(tieu_de="Tổng quan", icon="🛡️")

st.title("🛡️ Phát hiện bình luận độc hại tiếng Việt")
st.markdown(
    "Hệ thống phân loại bình luận mạng xã hội tiếng Việt thành **3 mức độ**: "
    "**Bình thường (`CLEAN`)**, **Xúc phạm (`OFFENSIVE`)**, và **Thù ghét (`HATE`)** "
    "trên bộ dữ liệu **ViHSD**, hỗ trợ người kiểm duyệt phát hiện nhanh nội dung độc hại."
)

# Đọc số liệu thật từ các file trên đĩa
so_binh_luan_str = "Chưa có file"
phu_de_du_lieu = "Cần chạy tiền xử lý"
for candidate in ["02_du_lieu_sach.csv", "01_du_lieu_goc.csv"]:
    fpath = os.path.join(DATA_PROCESSED_DIR, candidate)
    if os.path.isfile(fpath):
        try:
            df_count = pd.read_csv(fpath, usecols=[0])
            so_binh_luan_str = f"{len(df_count):,}"
            phu_de_du_lieu = f"Đọc từ data/processed/{candidate}"
            break
        except Exception:
            pass

label_names = cfg.get("label_names", ["Bình thường", "Xúc phạm", "Thù ghét"])
so_nhan_str = f"{len(label_names)} nhãn" if label_names else "Chưa có"
phu_de_nhan = " · ".join(label_names)

best_model_str = cfg.get("best_model") or "Chưa có kết quả"
f1_best_str = "Chưa có kết quả"
phu_de_f1 = "Hãy chạy đánh giá mô hình"

comp_csv = os.path.join(OUTPUTS_DIR, "bang_so_sanh_mo_hinh.csv")
val_csv = os.path.join(OUTPUTS_DIR, "thu_nghiem_baseline.csv")

if os.path.isfile(comp_csv):
    try:
        df_comp = pd.read_csv(comp_csv)
        f1_col = "F1-macro (%)" if "F1-macro (%)" in df_comp.columns else ("F1-macro" if "F1-macro" in df_comp.columns else None)
        if f1_col and not df_comp.empty:
            df_sorted = df_comp.sort_values(by=f1_col, ascending=False)
            top_row = df_sorted.iloc[0]
            best_model_str = str(top_row.get("Mô hình", best_model_str))
            val_f1 = float(top_row[f1_col])
            if val_f1 <= 1.0:
                val_f1 *= 100.0
            f1_best_str = f"{val_f1:.2f}%"
            phu_de_f1 = "Đọc từ outputs/bang_so_sanh_mo_hinh.csv (Test)"
    except Exception:
        pass
elif os.path.isfile(val_csv):
    try:
        df_val = pd.read_csv(val_csv)
        if "f1_macro_val" in df_val.columns and not df_val.empty:
            top_val = df_val.sort_values(by="f1_macro_val", ascending=False).iloc[0]
            f1_best_str = f"{float(top_val['f1_macro_val']) * 100:.2f}% (Val)"
            phu_de_f1 = "Đọc từ outputs/thu_nghiem_baseline.csv (Val)"
    except Exception:
        pass

col1, col2, col3, col4 = st.columns(4)
with col1:
    the_chi_so("Số bình luận sạch", so_binh_luan_str, phu_de_du_lieu, "#2563EB")
with col2:
    the_chi_so("Số nhãn phân loại", so_nhan_str, phu_de_nhan, "#16A34A")
with col3:
    the_chi_so("Mô hình tốt nhất", best_model_str, f"Sẵn có: {len(cfg.get('models_available', []))} mô hình", "#D97706")
with col4:
    the_chi_so("F1-macro tốt nhất", f1_best_str, phu_de_f1, "#DC2626")

# Quy trình xử lý
st.markdown(
    """
    <div class="sentinel-card">
        <div style="font-weight:600;color:#0F172A;margin-bottom:8px;">🔄 Quy trình xử lý chuẩn (Pipeline)</div>
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;font-size:0.92rem;">
            <span class="mono-tag" style="background:#EFF6FF;color:#1D4ED8;">1. Dữ liệu thô (ViHSD)</span>
            <span>➔</span>
            <span class="mono-tag" style="background:#EFF6FF;color:#1D4ED8;">2. Làm sạch (NFC, URL, Emoji, Kéo dài, Teencode)</span>
            <span>➔</span>
            <span class="mono-tag" style="background:#EFF6FF;color:#1D4ED8;">3. Tách từ tiếng Việt (PyVi + bảo toàn *)</span>
            <span>➔</span>
            <span class="mono-tag" style="background:#EFF6FF;color:#1D4ED8;">4. Mô hình (TF-IDF CalibratedSVC / PhoBERT)</span>
            <span>➔</span>
            <span class="mono-tag" style="background:#DCFCE7;color:#15803D;">5. Nhãn & Điểm độc hại</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📊 Phân bố nhãn trong bộ dữ liệu")
    phan_bo_img = os.path.join(OUTPUTS_DIR, "phan_bo_nhan.png")
    if os.path.isfile(phan_bo_img):
        st.image(phan_bo_img, caption="Phân bố 3 nhãn trong dữ liệu (đọc từ outputs/phan_bo_nhan.png)", use_container_width=True)
    else:
        thong_bao_thieu_file("outputs/phan_bo_nhan.png", "python scripts/check_data.py")

with right_col:
    st.subheader("🚀 Lối tắt điều hướng nhanh")
    st.markdown(
        "Chọn chức năng bạn muốn trải nghiệm hoặc kiểm tra số liệu:"
    )
    if hasattr(st, "page_link"):
        st.page_link("pages/1_Thu_nghiem.py", label="🧪 Trang 2: Thử nghiệm nhập câu trực tiếp", icon="🧪")
        st.page_link("pages/2_Tien_xu_ly.py", label="🧹 Trang 3: Xem trước / sau tiền xử lý", icon="🧹")
        st.page_link("pages/3_Danh_gia_mo_hinh.py", label="📈 Trang 4: Bảng so sánh, Ma trận & Phân tích lỗi", icon="📈")
        st.page_link("pages/4_YouTube.py", label="📺 Trang 5: Phân loại bình luận YouTube hàng loạt", icon="📺")
        st.page_link("pages/5_Han_che_dao_duc.py", label="⚖️ Trang 6: Hạn chế & Đạo đức AI", icon="⚖️")
    else:
        st.info("Hãy chọn các trang từ **Thanh điều hướng bên trái**.")

    # Tóm tắt nhanh tỷ lệ lệch nhãn từ file thật nếu có
    train_csv = os.path.join(DATA_PROCESSED_DIR, "03_train.csv")
    if os.path.isfile(train_csv):
        try:
            df_tr = pd.read_csv(train_csv, usecols=["label"])
            vc = (df_tr["label"].value_counts(normalize=True).sort_index() * 100).round(2)
            st.markdown("#### 📌 Tỷ lệ nhãn thực tế (Tập Train)")
            for lid, pct in vc.items():
                ten_nh = label_names[int(lid)] if int(lid) < len(label_names) else f"Nhãn {lid}"
                st.write(f"- **{ten_nh} (`{lid}`)**: `{pct}%`")
        except Exception:
            pass

hien_chan_trang_canh_bao()
