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
    "và ký tự chèn giữa chữ để né bộ lọc (`b.ắ.c k.ỳ`, `l**`). Trang này minh họa trực tiếp quy trình chuẩn hóa "
    "từ `src/preprocess.py`."
)

c_in, c_btn = st.columns([4, 1])
with c_in:
    cau_tho = st.text_input(
        "Nhập câu thô cần kiểm tra tiền xử lý:",
        value="hàng như l** ý mn đừng mua lừa đảo đấy k dùng dc đâu https://shopee.vn/fake-shop 😍",
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
                <div style="background:#F8FAFC;padding:12px;border-radius:8px;border:1px solid #E2E8F0;font-size:0.95rem;color:#0F172A;">
                    {html.escape(cau_tho)}
                </div>
                <div style="margin-top:12px;font-size:0.86rem;color:#475569;">
                    <b>Đã phát hiện:</b><br><div style="margin-top:6px;">{badges_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_after:
        tokens = van_ban_seg.split() if van_ban_seg else []
        chips_html = "".join([f'<span class="token-chip">{html.escape(t)}</span>' for t in tokens]) or "<i>(Rỗng sau khi làm sạch)</i>"

        st.markdown(
            f"""
            <div class="sentinel-card" style="border-left: 5px solid #2563EB;">
                <div style="font-weight:700;color:#1D4ED8;margin-bottom:8px;">SAU LÀM SẠCH & TÁCH TỪ — {len(van_ban_seg)} ký tự ({len(tokens)} token)</div>
                <div style="font-size:0.88rem;color:#475569;margin-bottom:6px;">
                    <b>Sau làm sạch (<code>lam_sach</code>):</b> <code>{html.escape(van_ban_sach)}</code>
                </div>
                <div style="font-size:0.88rem;color:#475569;margin-top:10px;">
                    <b>Từng từ đưa vào mô hình (<code>tach_tu</code> — từ ghép nối bằng <code>_</code>):</b>
                    <div style="margin-top:6px;">{chips_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
col_table, col_chart = st.columns([1.35, 1], gap="large")

with col_table:
    st.subheader("📋 Bảng ví dụ Trước / Sau xử lý")
    truoc_sau_csv = os.path.join(OUTPUTS_DIR, "truoc_sau_xu_ly.csv")
    if os.path.isfile(truoc_sau_csv):
        try:
            df_ts = pd.read_csv(truoc_sau_csv)
            st.dataframe(df_ts, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Không đọc được file `outputs/truoc_sau_xu_ly.csv`: {e}")
    else:
        thong_bao_thieu_file("outputs/truoc_sau_xu_ly.csv", "python scripts/evaluate_final.py")

with col_chart:
    st.subheader("📈 Thống kê dấu hiệu nhiễu toàn bộ dữ liệu")
    ban_img = os.path.join(OUTPUTS_DIR, "thong_ke_dau_hieu_ban.png")
    if os.path.isfile(ban_img):
        st.image(ban_img, caption="Tỷ lệ % bình luận chứa từng dấu hiệu cần xử lý (outputs/thong_ke_dau_hieu_ban.png)", use_container_width=True)
    else:
        thong_bao_thieu_file("outputs/thong_ke_dau_hieu_ban.png", "python scripts/evaluate_final.py")

hien_chan_trang_canh_bao()
