import os
import sys
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ui import (
    cai_dat_trang,
    the_chi_so,
    thong_bao_thieu_file,
    hien_chan_trang_canh_bao,
    OUTPUTS_DIR,
)
from src.youtube import lay_binh_luan_youtube
from src.inference import phan_loai_hang_loat

cfg = cai_dat_trang(tieu_de="Phân loại YouTube", icon="📺")

st.title("📺 Phân loại bình luận YouTube hàng loạt")
st.markdown(
    "Thu thập bình luận công khai từ một video YouTube qua **YouTube Data API v3** và phân loại tự động. "
    "**Bảo mật & Quyền riêng tư:** API key không lưu ra file/log; hệ thống chỉ giữ nội dung, số lượt thích và thời gian, "
    "**tuyệt đối không lưu tên người bình luận**."
)

models_avail = cfg.get("models_available", [])
best_model = cfg.get("best_model", "")
fallback_csv = os.path.join(OUTPUTS_DIR, "ket_qua_youtube.csv")

che_do = st.radio(
    "Chọn nguồn dữ liệu bình luận:",
    options=[
        "🌐 Gọi trực tiếp YouTube Data API v3 (cần mạng & API key)",
        "📂 Chế độ dự phòng: Đọc file outputs/ket_qua_youtube.csv (khi mất mạng / hết quota)",
    ],
    horizontal=True,
)

df_ket_qua = None

if che_do.startswith("🌐"):
    with st.container():
        c_link, c_key, c_num = st.columns([2.2, 1.5, 1])
        with c_link:
            link_video = st.text_input(
                "Link video YouTube công khai (hoặc Video ID 11 ký tự):",
                placeholder="https://www.youtube.com/watch?v=XXXXXXXXXXX",
            )
        with c_key:
            # Kiểm tra st.secrets một cách an toàn, ưu tiên ô nhập ẩn
            default_secret_key = ""
            try:
                default_secret_key = str(st.secrets.get("YOUTUBE_API_KEY", ""))
            except Exception:
                default_secret_key = ""
            api_key_input = st.text_input(
                "YouTube API Key (ô ẩn, không lưu):",
                value=default_secret_key,
                type="password",
                placeholder="Nhập API key của bạn...",
            )
        with c_num:
            so_luong = st.slider("Số bình luận tối đa:", min_value=20, max_value=500, value=100, step=20)

        c_mod, c_btn = st.columns([2.2, 1])
        with c_mod:
            if models_avail:
                idx_def = models_avail.index(best_model) if best_model in models_avail else 0
                chon_mo_hinh = st.selectbox("Mô hình phân loại:", options=models_avail, index=idx_def)
            else:
                chon_mo_hinh = st.selectbox("Mô hình phân loại:", options=["Chưa có mô hình"], disabled=True)
        with c_btn:
            st.write("")
            st.write("")
            bam_chay = st.button("🚀 Lấy và phân loại", type="primary", use_container_width=True)

    if bam_chay:
        if not models_avail:
            st.warning("Chưa có mô hình trong thư mục `models/`. Hãy chạy script huấn luyện trước.")
        else:
            with st.spinner("Đang tải bình luận từ YouTube và chạy mô hình phân loại..."):
                try:
                    df_raw = lay_binh_luan_youtube(link_video, api_key_input, toi_da=so_luong)
                    if df_raw.empty:
                        st.info("Video này hiện chưa có bình luận công khai nào.")
                    else:
                        df_ket_qua = phan_loai_hang_loat(df_raw, ten_mo_hinh=chon_mo_hinh)
                        os.makedirs(OUTPUTS_DIR, exist_ok=True)
                        df_ket_qua.to_csv(fallback_csv, index=False)
                        st.success(f"Đã lấy và phân loại thành công {len(df_ket_qua)} bình luận! (Đã cập nhật file dự phòng `outputs/ket_qua_youtube.csv`)")
                except (ValueError, RuntimeError) as err:
                    st.warning(f"⚠️ {err}")
                    if os.path.isfile(fallback_csv):
                        st.info("💡 Bạn có thể chuyển sang **Chế độ dự phòng** ở nút chọn phía trên để xem kết quả đã lưu sẵn trong `outputs/ket_qua_youtube.csv`.")
                except Exception as err:
                    st.warning(f"Không thể hoàn tất yêu cầu: {err}")

else:
    st.caption("Đang sử dụng dữ liệu dự phòng từ `outputs/ket_qua_youtube.csv` (dùng khi trình diễn không có kết nối mạng hoặc hết hạn mức API).")
    if os.path.isfile(fallback_csv):
        try:
            df_ket_qua = pd.read_csv(fallback_csv)
            # Đảm bảo không có cột tên người dùng nếu file cũ từng lưu
            for forbidden_col in ["author", "authorDisplayName", "user", "username"]:
                if forbidden_col in df_ket_qua.columns:
                    df_ket_qua = df_ket_qua.drop(columns=[forbidden_col])
        except Exception as e:
            st.warning(f"Không đọc được file `outputs/ket_qua_youtube.csv`: {e}")
    else:
        thong_bao_thieu_file("outputs/ket_qua_youtube.csv", "python scripts/evaluate_final.py")

if df_ket_qua is not None and not df_ket_qua.empty:
    st.markdown("---")
    tong_so = len(df_ket_qua)
    ty_le = (df_ket_qua["nhan"].value_counts(normalize=True) * 100).round(2)

    pt_clean = float(ty_le.get("Bình thường", ty_le.get("CLEAN", 0.0)))
    pt_off = float(ty_le.get("Xúc phạm", ty_le.get("OFFENSIVE", 0.0)))
    pt_hate = float(ty_le.get("Thù ghét", ty_le.get("HATE", 0.0)))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        the_chi_so("Đã phân loại", f"{tong_so} câu", "Bình luận công khai", "#2563EB")
    with m2:
        the_chi_so("Bình thường", f"{pt_clean:.1f}%", "Không độc hại", "#16A34A")
    with m3:
        the_chi_so("Xúc phạm", f"{pt_off:.1f}%", "Từ ngữ nhạy cảm / công kích nhẹ", "#D97706")
    with m4:
        the_chi_so("Thù ghét", f"{pt_hate:.1f}%", "Chửi bới nặng / kích động thù ghét", "#DC2626")

    c_chart, c_top = st.columns([1, 1.4], gap="large")
    with c_chart:
        st.subheader("📊 Tỷ lệ các nhóm bình luận (%)")
        df_chart = pd.DataFrame(
            {
                "Nhãn": ["Bình thường", "Xúc phạm", "Thù ghét"],
                "Tỷ lệ (%)": [pt_clean, pt_off, pt_hate],
            }
        ).set_index("Nhãn")
        st.bar_chart(df_chart)

    with c_top:
        st.subheader("🔥 Top 5 bình luận có điểm độc hại cao nhất")
        cols_show = [c for c in ["text", "nhan", "diem_toxic", "like", "thoi_gian"] if c in df_ket_qua.columns]
        top5 = df_ket_qua.sort_values(by="diem_toxic", ascending=False).head(5)[cols_show]
        st.dataframe(top5, use_container_width=True, hide_index=True)

    with st.expander(f"📋 Xem toàn bộ {tong_so} bình luận đã phân loại"):
        cols_all = [c for c in ["text", "sau_xu_ly", "nhan", "do_tin_cay", "diem_toxic", "like", "thoi_gian"] if c in df_ket_qua.columns]
        st.dataframe(df_ket_qua[cols_all], use_container_width=True, hide_index=True)

    csv_bytes = df_ket_qua.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Tải kết quả phân loại (ket_qua_youtube.csv)",
        data=csv_bytes,
        file_name="ket_qua_youtube.csv",
        mime="text/csv",
    )

hien_chan_trang_canh_bao()
