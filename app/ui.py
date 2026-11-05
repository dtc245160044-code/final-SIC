import os
import html
import streamlit as st
from src.inference import doc_cau_hinh

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

MAU_NHAN = {
    "Bình thường": {"border": "#16A34A", "bg": "#DCFCE7", "text": "#15803D", "code": "CLEAN (0)"},
    "CLEAN": {"border": "#16A34A", "bg": "#DCFCE7", "text": "#15803D", "code": "CLEAN (0)"},
    "Xúc phạm": {"border": "#D97706", "bg": "#FEF3C7", "text": "#B45309", "code": "OFFENSIVE (1)"},
    "OFFENSIVE": {"border": "#D97706", "bg": "#FEF3C7", "text": "#B45309", "code": "OFFENSIVE (1)"},
    "Thù ghét": {"border": "#DC2626", "bg": "#FEE2E2", "text": "#B91C1C", "code": "HATE (2)"},
    "HATE": {"border": "#DC2626", "bg": "#FEE2E2", "text": "#B91C1C", "code": "HATE (2)"},
}


def cai_dat_trang(tieu_de: str, icon: str = "🛡️"):
    """Thiết lập cấu hình trang, CSS phong cách Sentinels AI và thanh bên trái."""
    st.set_page_config(
        page_title=f"{tieu_de} — Phát hiện bình luận độc hại tiếng Việt",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stApp {
            background-color: #F7F8FA;
        }
        .sentinel-card {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .sentinel-metric-label {
            font-size: 0.82rem;
            color: #64748B;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 6px;
        }
        .sentinel-metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }
        .sentinel-metric-sub {
            font-size: 0.80rem;
            color: #475569;
            margin-top: 4px;
        }
        .mono-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 6px;
            display: inline-block;
            margin-right: 6px;
            margin-bottom: 4px;
        }
        .token-chip {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.84rem;
            background-color: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            padding: 4px 10px;
            border-radius: 8px;
            display: inline-block;
            margin: 3px 4px 3px 0;
        }
        .token-chip-toxic {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.84rem;
            background-color: #FEE2E2;
            color: #B91C1C;
            border: 1px solid #FCA5A5;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 8px;
            display: inline-block;
            margin: 3px 4px 3px 0;
        }
        .demo-banner {
            background-color: #DC2626;
            color: #FFFFFF;
            padding: 12px 18px;
            border-radius: 10px;
            font-weight: 600;
            margin-bottom: 16px;
            text-align: center;
        }
        .disclaimer-box {
            background-color: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 4px solid #D97706;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.88rem;
            color: #92400E;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cfg = doc_cau_hinh()

    with st.sidebar:
        st.markdown("### 🛡️ ViHSD Toxic Detector")
        st.caption("Hệ thống hỗ trợ phát hiện bình luận độc hại tiếng Việt")
        st.markdown("---")
        avail = cfg.get("models_available", [])
        best = cfg.get("best_model", "")
        if avail:
            st.markdown(f"**Mô hình sẵn sàng ({len(avail)}):**")
            for m in avail:
                badge = "⭐ Tốt nhất" if m == best else "✓ Sẵn sàng"
                st.caption(f"• {m} ({badge})")
        else:
            st.warning("Chưa tìm thấy mô hình trong thư mục `models/`.")
        st.markdown("---")
        st.caption("Nhãn phân loại: `0=Bình thường` · `1=Xúc phạm` · `2=Thù ghét`")

    # Nếu dùng dữ liệu mẫu -> hiện băng đỏ trên mọi trang (Nguyên tắc số 3)
    if cfg.get("is_demo_data", False):
        st.markdown(
            '<div class="demo-banner">⚠️ Đang dùng dữ liệu mẫu, không phải kết quả thật.</div>',
            unsafe_allow_html=True,
        )

    return cfg


def the_chi_so(nhan: str, gia_tri: str, phu_de: str = "", mau_vien: str = "#2563EB"):
    """Vẽ thẻ chỉ số (metric card) nền trắng, bo góc 12px."""
    st.markdown(
        f"""
        <div class="sentinel-card" style="border-top: 4px solid {mau_vien};">
            <div class="sentinel-metric-label">{html.escape(str(nhan))}</div>
            <div class="sentinel-metric-value">{html.escape(str(gia_tri))}</div>
            <div class="sentinel-metric-sub">{html.escape(str(phu_de))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def the_ket_qua_phan_loai(kq: dict, tu_gay_doc: list[dict] | None = None):
    """Vẽ thẻ kết quả phân loại có viền trái đổi màu theo nhãn."""
    nhan = kq.get("nhan", "Bình thường")
    style = MAU_NHAN.get(nhan, MAU_NHAN["Bình thường"])
    do_tin_cay_pt = kq.get("do_tin_cay", 0.0) * 100
    diem_toxic_pt = kq.get("diem_toxic", 0.0) * 100
    sau_xu_ly = kq.get("sau_xu_ly", "")
    mo_hinh = kq.get("mo_hinh", "")
    do_tre = kq.get("do_tre_ms", 0.0)

    # Hiển thị từ đã xử lý, tô đỏ các từ làm tăng điểm độc hại
    if tu_gay_doc:
        chips_html = []
        gay_doc_list = []
        for item in tu_gay_doc:
            tu_esc = html.escape(str(item["tu"]))
            if item.get("gay_doc"):
                gay_doc_list.append(tu_esc)
                chips_html.append(f'<span class="token-chip-toxic" title="Đóng góp độc hại">{tu_esc}</span>')
            else:
                chips_html.append(f'<span class="token-chip">{tu_esc}</span>')
        chuoi_tu_html = "".join(chips_html)
        if gay_doc_list:
            tom_tat_tu_doc = (
                f'<div style="margin-top:8px;font-size:0.86rem;color:#B91C1C;">'
                f'🔴 <b>Từ gây độc phát hiện (bỏ từ làm giảm điểm độc hại):</b> '
                f'<code>{", ".join(gay_doc_list)}</code></div>'
            )
        else:
            tom_tat_tu_doc = (
                '<div style="margin-top:8px;font-size:0.84rem;color:#64748B;">'
                'Không có từ đơn lẻ nào vượt ngưỡng gây độc rõ rệt.</div>'
            )
    else:
        chuoi_tu_html = f"<code>{html.escape(sau_xu_ly)}</code>"
        tom_tat_tu_doc = ""

    st.markdown(
        f"""
        <div class="sentinel-card" style="border-left: 6px solid {style['border']};">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div>
                    <span class="mono-tag" style="background:{style['bg']};color:{style['text']};border:1px solid {style['border']};">
                        ● {html.escape(nhan.upper())} ({style['code']})
                    </span>
                    <span class="mono-tag" style="background:#F1F5F9;color:#334155;">
                        Mô hình: {html.escape(str(mo_hinh))}
                    </span>
                </div>
                <div style="font-weight:700;font-size:1.15rem;color:{style['text']};">
                    Độ tin cậy: {do_tin_cay_pt:.1f}%
                </div>
            </div>
            <div style="margin-top:12px;font-size:0.92rem;color:#1E293B;">
                <b>Điểm độc hại tổng hợp (Xúc phạm + Thù ghét):</b>
                <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:{'#DC2626' if diem_toxic_pt >= 50 else '#16A34A'};">
                    {diem_toxic_pt:.1f}%
                </span>
                <span style="color:#64748B;font-size:0.82rem;margin-left:10px;">(Độ trễ: {do_tre:.1f} ms)</span>
            </div>
            <div style="margin-top:12px;font-size:0.88rem;color:#334155;">
                <b>Sau làm sạch & tách từ:</b>
                <div style="margin-top:6px;">{chuoi_tu_html}</div>
                {tom_tat_tu_doc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hien_chan_trang_canh_bao():
    """Hiển thị dòng nhắc bắt buộc ở cuối mỗi kết quả phân loại (Nguyên tắc số 4)."""
    st.markdown(
        """
        <div class="disclaimer-box">
            ⚠️ <b>Lưu ý:</b> Mô hình có thể sai. Kết quả chỉ mang tính hỗ trợ, không thay thế người kiểm duyệt.
        </div>
        """,
        unsafe_allow_html=True,
    )


def thong_bao_thieu_file(ten_file: str, lenh_goi_y: str = "python scripts/evaluate_final.py"):
    """Hiển thị thông báo thân thiện khi thiếu file kết quả, tuyệt đối không điền số giả."""
    st.info(
        f"📂 **Chưa có kết quả, hãy chạy notebook/script trước.**\n\n"
        f"Hệ thống chưa tìm thấy file `{ten_file}`. "
        f"Hãy chạy lệnh `{lenh_goi_y}` để sinh kết quả thật rồi tải lại trang."
    )

