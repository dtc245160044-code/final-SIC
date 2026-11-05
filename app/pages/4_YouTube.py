import os
import sys
from datetime import datetime
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
from src.youtube import lay_video_id, lay_binh_luan_youtube
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
        "📂 Chế độ dự phòng: Đọc file outputs/ket_qua_youtube.csv (do bạn chạy API thật lưu lại)",
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
            so_luong = st.slider("Số bình luận tối đa:", min_value=20, max_value=500, value=150, step=10)

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
                    vid_id = lay_video_id(link_video)
                    df_raw = lay_binh_luan_youtube(vid_id, api_key_input, toi_da=so_luong)
                    if df_raw.empty:
                        st.info("Video này hiện chưa có bình luận công khai nào.")
                    else:
                        df_ket_qua = phan_loai_hang_loat(df_raw, ten_mo_hinh=chon_mo_hinh)
                        df_ket_qua["video_id"] = vid_id
                        df_ket_qua["ngay_lay"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        os.makedirs(OUTPUTS_DIR, exist_ok=True)
                        df_ket_qua.to_csv(fallback_csv, index=False, encoding="utf-8")
                        st.success(
                            f"Đã lấy và phân loại thành công {len(df_ket_qua)} bình luận từ video `{vid_id}`! "
                            f"(Đã lưu vào `outputs/ket_qua_youtube.csv`)"
                        )
                except (ValueError, RuntimeError) as err:
                    st.warning(f"⚠️ {err}")
                    if os.path.isfile(fallback_csv):
                        st.info(
                            "💡 Bạn có thể chuyển sang **Chế độ dự phòng** ở nút chọn phía trên để xem kết quả đã lưu trong `outputs/ket_qua_youtube.csv`."
                        )
                except Exception as err:
                    st.warning(f"Không thể hoàn tất yêu cầu: {err}")

else:
    st.caption(
        "Chế độ dự phòng chỉ hiển thị kết quả khi bạn đã chạy lấy bình luận YouTube thật bằng API key của mình "
        "(qua trang này hoặc chạy `python scripts/lay_youtube_demo.py`)."
    )
    if os.path.isfile(fallback_csv):
        try:
            df_loaded = pd.read_csv(fallback_csv)
            for forbidden_col in ["author", "authorDisplayName", "user", "username"]:
                if forbidden_col in df_loaded.columns:
                    df_loaded = df_loaded.drop(columns=[forbidden_col])
            if "video_id" not in df_loaded.columns or "ngay_lay" not in df_loaded.columns:
                st.warning(
                    "📂 **Chưa có kết quả YouTube thật** (file `outputs/ket_qua_youtube.csv` thiếu thông tin `video_id` / `ngay_lay`). "
                    "Hãy chạy `python scripts/lay_youtube_demo.py` bằng API key của bạn trước."
                )
            else:
                df_ket_qua = df_loaded
                vid_info = str(df_ket_qua["video_id"].iloc[0]) if not df_ket_qua.empty else "—"
                ngay_info = str(df_ket_qua["ngay_lay"].iloc[0]) if not df_ket_qua.empty else "—"
                st.info(f"📌 Đang hiển thị dữ liệu YouTube thật — **Video ID:** `{vid_id if 'vid_id' in locals() else vid_info}` | **Ngày lấy:** `{ngay_info}`")
        except Exception as e:
            st.warning(f"Không đọc được file `outputs/ket_qua_youtube.csv`: {e}")
    else:
        thong_bao_thieu_file("outputs/ket_qua_youtube.csv", "python scripts/lay_youtube_demo.py")

if df_ket_qua is not None and not df_ket_qua.empty:
    st.markdown("---")
    tong_so = len(df_ket_qua)
    ty_le = (df_ket_qua["nhan"].value_counts(normalize=True) * 100).round(2)

    pt_clean = float(ty_le.get("Bình thường", ty_le.get("CLEAN", 0.0)))
    pt_off = float(ty_le.get("Xúc phạm", ty_le.get("OFFENSIVE", 0.0)))
    pt_hate = float(ty_le.get("Thù ghét", ty_le.get("HATE", 0.0)))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        the_chi_so("Đã phân loại", f"{tong_so} câu", "Bình luận YouTube công khai", "#2563EB")
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
        cols_show = [c for c in ["text", "nhan", "diem_toxic", "like", "thoi_gian", "video_id", "ngay_lay"] if c in df_ket_qua.columns]
        top5 = df_ket_qua.sort_values(by="diem_toxic", ascending=False).head(5)[cols_show]
        st.dataframe(top5, use_container_width=True, hide_index=True)

    with st.expander(f"📋 Xem toàn bộ {tong_so} bình luận đã phân loại"):
        cols_all = [
            c
            for c in ["text", "sau_xu_ly", "nhan", "do_tin_cay", "diem_toxic", "like", "thoi_gian", "video_id", "ngay_lay"]
            if c in df_ket_qua.columns
        ]
        st.dataframe(df_ket_qua[cols_all], use_container_width=True, hide_index=True)

    csv_bytes = df_ket_qua.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Tải kết quả phân loại (ket_qua_youtube.csv)",
        data=csv_bytes,
        file_name="ket_qua_youtube.csv",
        mime="text/csv",
    )

hien_chan_trang_canh_bao()
