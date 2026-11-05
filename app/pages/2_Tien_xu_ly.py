import os
import sys
import html
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ui import cai_dat_trang, thong_bao_thieu_file, hien_chan_trang_canh_bao, OUTPUTS_DIR
from src.preprocess import lam_sach, tach_tu, phat_hien

cai_dat_trang(tieu_de="Tiền xử lý", icon="🧹")

st.title("🧹 Tiền xử lý văn bản tiếng Việt: Trước và Sau")
st.markdown(
    "Bình luận mạng xã hội thường chứa đường dẫn, emoji, chữ kéo dài (`quááááá`), teencode (`ko`, `dc`, `mn`) "
    "và ký tự chèn giữa chữ để né bộ lọc. Trang này minh họa trực tiếp quy trình chuẩn hóa "
    "từ `src/preprocess.py` trên dữ liệu thật của bộ ViHSD."
)

c_in, c_btn = st.columns([4, 1])
with c_in:
    cau_tho = st.text_input(
        "Nhập câu thô cần kiểm tra tiền xử lý (mặc định là câu thật từ bộ dữ liệu ViHSD):",
        value="Đúng là bọn mắt híp lò xo thụt :))) bên việt nam t cái này ra cách đây 10 năm r và bọn t gọi là cái L :)))",
        max_chars=500,
    )
with c_btn:
    st.write("")
    st.write("")
    chay_chuan_hoa = st.button("⚙️ Chạy chuẩn hóa", type="primary", use_container_width=True)

if cau_tho and cau_tho.strip():
    dh = phat_hien(cau_tho)
    van_ban_sach = lam_sach(cau_tho)
    van_ban_seg = tach_tu(van_ban_sach) if van_ban_sach else ""

    col_before, col_after = st.columns(2, gap="large")

    with col_before:
        badges = []
        if dh.get("link", 0) > 0:
            badges.append(f'<span class="mono-tag" style="background:#DBEAFE;color:#1E40AF;">🔗 {dh["link"]} link</span>')
        if dh.get("teencode", 0) > 0:
            badges.append(f'<span class="mono-tag" style="background:#FEF3C7;color:#B45309;">💬 {dh["teencode"]} teencode</span>')
        if dh.get("keo_dai", 0) > 0:
            badges.append(f'<span class="mono-tag" style="background:#FCE7F3;color:#9D174D;">〰️ {dh["keo_dai"]} chữ kéo dài</span>')
        if dh.get("chen_kt", 0) > 0:
            badges.append(f'<span class="mono-tag" style="background:#FEE2E2;color:#B91C1C;">🛡️ {dh["chen_kt"]} chèn ký tự</span>')
        if dh.get("emoji", 0) > 0:
            badges.append(f'<span class="mono-tag" style="background:#E0E7FF;color:#3730A3;">😀 {dh["emoji"]} emoji</span>')

        badges_html = "".join(badges) if badges else '<span class="mono-tag" style="background:#F1F5F9;color:#475569;">Không có dấu hiệu nhiễu đặc biệt</span>'

        st.markdown(
            f"""
            <div class="sentinel-card" style="border-left: 5px solid #64748B;">
                <div style="font-weight:700;color:#334155;margin-bottom:8px;">TRƯỚC XỬ LÝ (RAW) — {len(cau_tho)} ký tự</div>
                <div style="font-size:0.95rem;color:#0F172A;margin-bottom:12px;">{html.escape(cau_tho)}</div>
                <div style="font-size:0.82rem;color:#475569;margin-bottom:4px;"><b>Đã phát hiện:</b></div>
                <div>{badges_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_after:
        chips_html = " ".join(
            f'<span class="mono-tag" style="background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;">{html.escape(tok)}</span>'
            for tok in van_ban_seg.split()
        )
        st.markdown(
            f"""
            <div class="sentinel-card" style="border-left: 5px solid #2563EB;">
                <div style="font-weight:700;color:#1D4ED8;margin-bottom:8px;">SAU LÀM SẠCH + TÁCH TỪ (PYVI) — {len(van_ban_seg)} ký tự</div>
                <div style="font-size:0.92rem;color:#334155;margin-bottom:8px;"><b>Làm sạch:</b> <code>{html.escape(van_ban_sach)}</code></div>
                <div style="font-size:0.82rem;color:#475569;margin-bottom:6px;"><b>Từng token đưa vào mô hình (từ ghép nối bằng <code>_</code>):</b></div>
                <div style="line-height:1.9;">{chips_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
col_table, col_chart = st.columns([1.3, 1], gap="large")

with col_table:
    st.subheader("📋 Bảng đối chiếu Trước & Sau xử lý (trích từ ViHSD)")
    csv_path = os.path.join(OUTPUTS_DIR, "truoc_sau_xu_ly.csv")
    if os.path.isfile(csv_path):
        df_ts = pd.read_csv(csv_path)
        st.dataframe(df_ts, use_container_width=True, hide_index=True)
    else:
        thong_bao_thieu_file("outputs/truoc_sau_xu_ly.csv", "notebooks/01_du_lieu_va_tien_xu_ly.ipynb")

with col_chart:
    st.subheader("📊 Tỷ lệ bình luận chứa dấu hiệu nhiễu")
    img_path = os.path.join(OUTPUTS_DIR, "thong_ke_dau_hieu_ban.png")
    if os.path.isfile(img_path):
        st.image(img_path, caption="Tỷ lệ % bình luận chứa từng dấu hiệu cần xử lý trên toàn bộ dữ liệu", use_container_width=True)
    else:
        thong_bao_thieu_file("outputs/thong_ke_dau_hieu_ban.png", "notebooks/01_du_lieu_va_tien_xu_ly.ipynb")

hien_chan_trang_canh_bao()
