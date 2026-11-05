import os
import sys
import re
import html
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ui import cai_dat_trang, thong_bao_thieu_file, hien_chan_trang_canh_bao, OUTPUTS_DIR

cfg = cai_dat_trang(tieu_de="Đánh giá mô hình", icon="📈")

st.title("📈 Đánh giá & Phân tích lỗi mô hình")
st.markdown(
    "Toàn bộ số liệu dưới đây được đọc trực tiếp từ kết quả chạy thật trong thư mục `outputs/` "
    "(tuyệt đối không điền số mẫu)."
)

tab_ss, tab_cm, tab_loi = st.tabs([
    "1️⃣ So sánh mô hình",
    "2️⃣ Ma trận nhầm lẫn",
    "3️⃣ Phân tích lỗi (FP / FN)",
])

# ==============================================================================
# TAB 1: SO SÁNH MÔ HÌNH
# ==============================================================================
with tab_ss:
    st.subheader("📊 Bảng so sánh các mô hình trên tập Test")
    comp_csv = os.path.join(OUTPUTS_DIR, "bang_so_sanh_mo_hinh.csv")
    comp_png = os.path.join(OUTPUTS_DIR, "so_sanh_mo_hinh.png")

    if os.path.isfile(comp_csv):
        try:
            df_comp = pd.read_csv(comp_csv)
            f1_col = "F1-macro (%)" if "F1-macro (%)" in df_comp.columns else ("F1-macro" if "F1-macro" in df_comp.columns else None)
            best_idx = df_comp[f1_col].idxmax() if (f1_col and not df_comp.empty) else None

            def _highlight_best(row):
                if best_idx is not None and row.name == best_idx:
                    return ["background-color: #DCFCE7; font-weight: 600;" for _ in row]
                return ["" for _ in row]

            st.dataframe(
                df_comp.style.apply(_highlight_best, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("💡 Dòng tô nền xanh nhạt là mô hình đạt **F1-macro** cao nhất trên tập Test.")

            with open(comp_csv, "rb") as f_dl:
                st.download_button(
                    label="📥 Tải bảng so sánh (bang_so_sanh_mo_hinh.csv)",
                    data=f_dl.read(),
                    file_name="bang_so_sanh_mo_hinh.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.warning(f"Không đọc được `outputs/bang_so_sanh_mo_hinh.csv`: {e}")
    else:
        thong_bao_thieu_file("outputs/bang_so_sanh_mo_hinh.csv", "python scripts/evaluate_final.py")

    if os.path.isfile(comp_png):
        st.markdown("#### Biểu đồ trực quan so sánh các chỉ số")
        st.image(comp_png, caption="So sánh các mô hình trên tập Test (outputs/so_sanh_mo_hinh.png)", use_container_width=True)

    # Hiển thị thêm bảng lịch sử tinh chỉnh Baseline trên tập Val (Bước 2) nếu có
    val_exp_csv = os.path.join(OUTPUTS_DIR, "thu_nghiem_baseline.csv")
    if os.path.isfile(val_exp_csv):
        with st.expander("🔎 Xem thêm: Kết quả 76 lần thử nghiệm tinh chỉnh Baseline trên tập Validation (Bước 2)"):
            df_exp = pd.read_csv(val_exp_csv).sort_values(by="f1_macro_val", ascending=False)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: MA TRẬN NHẦM LẪN
# ==============================================================================
with tab_cm:
    st.subheader("🟦 Ma trận nhầm lẫn (Confusion Matrix)")
    # Tìm tất cả ảnh ma trận nhầm lẫn hiện có trong outputs/
    cm_files = {}
    if os.path.isdir(OUTPUTS_DIR):
        for fname in sorted(os.listdir(OUTPUTS_DIR)):
            if fname.startswith("ma_tran_nham_lan_") and fname.endswith(".png"):
                label_display = fname.replace("ma_tran_nham_lan_", "").replace(".png", "").replace("_", " ")
                cm_files[f"{label_display} (Test)"] = os.path.join(OUTPUTS_DIR, fname)
        # Bổ sung ảnh ma trận trên Val nếu người dùng muốn đối chiếu
        for fname in sorted(os.listdir(OUTPUTS_DIR)):
            if fname.startswith("confusion_matrix_") and fname.endswith(".png"):
                label_display = fname.replace("confusion_matrix_", "").replace(".png", "")
                cm_files[f"{label_display} (Validation)"] = os.path.join(OUTPUTS_DIR, fname)

    if cm_files:
        chon_cm = st.selectbox("Chọn mô hình để xem Ma trận nhầm lẫn:", options=list(cm_files.keys()))
        st.image(cm_files[chon_cm], caption=f"File: outputs/{os.path.basename(cm_files[chon_cm])}", use_container_width=True)
        st.markdown(
            """
            <div class="sentinel-card">
                <b>📖 Hướng dẫn đọc Ma trận nhầm lẫn:</b>
                <ul style="margin-bottom:0;margin-top:6px;">
                    <li><b>Hàng ngang (trục dọc):</b> Nhãn thật sự của bình luận trong bộ dữ liệu.</li>
                    <li><b>Cột dọc (trục ngang):</b> Nhãn do mô hình dự đoán.</li>
                    <li><b>Đường chéo chính (trên-trái xuống dưới-phải):</b> Các bình luận được mô hình đoán <b>đúng</b> (bên phải là tỷ lệ % Recall theo từng lớp thật).</li>
                    <li>Các ô nằm ở hàng <code>Xúc phạm / Thù ghét</code> nhưng rơi vào cột <code>Bình thường</code> là các bình luận độc hại bị <b>bỏ sót (False Negative)</b>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        thong_bao_thieu_file("outputs/ma_tran_nham_lan_*.png", "python scripts/evaluate_final.py")

# ==============================================================================
# TAB 3: PHÂN TÍCH LỖI (FP / FN)
# ==============================================================================
with tab_loi:
    st.subheader("🔍 Phân tích lỗi điển hình trên tập Test")
    loi_csv = os.path.join(OUTPUTS_DIR, "phan_tich_loi.csv")

    if os.path.isfile(loi_csv):
        try:
            df_loi = pd.read_csv(loi_csv, keep_default_na=False)
            col_loai = "loại_lỗi" if "loại_lỗi" in df_loi.columns else ("loai_loi" if "loai_loi" in df_loi.columns else None)
            col_text = "bình_luận" if "bình_luận" in df_loi.columns else ("text" if "text" in df_loi.columns else df_loi.columns[0])
            col_seg = "sau_xử_lý" if "sau_xử_lý" in df_loi.columns else ("text_seg" if "text_seg" in df_loi.columns else None)
            col_that = "nhãn_thật" if "nhãn_thật" in df_loi.columns else ("nhan_that" if "nhan_that" in df_loi.columns else "label")
            col_doan = "nhãn_đoán" if "nhãn_đoán" in df_loi.columns else ("nhan_doan" if "nhan_doan" in df_loi.columns else "pred")
            col_goi_y = "gợi_ý_nguyên_nhân" if "gợi_ý_nguyên_nhân" in df_loi.columns else ("goi_y_nguyen_nhan" if "goi_y_nguyen_nhan" in df_loi.columns else None)
            col_nx = "nhận_xét_của_nhóm" if "nhận_xét_của_nhóm" in df_loi.columns else None

            so_luong_hien = st.slider("Số lượng lỗi hiển thị mỗi cột:", min_value=5, max_value=30, value=10, step=5)

            c_fp, c_fn = st.columns(2, gap="large")

            def _ve_danh_sach_the_loi(df_sub: pd.DataFrame, mau_vien: str, nhan_loai: str):
                if df_sub.empty:
                    st.info(f"Không có mẫu lỗi {nhan_loai} nào trong file.")
                    return
                for _, r in df_sub.head(so_luong_hien).iterrows():
                    cau_goc = html.escape(str(r.get(col_text, "")))
                    cau_seg = html.escape(str(r.get(col_seg, ""))) if col_seg else ""
                    nh_that = html.escape(str(r.get(col_that, "")))
                    nh_doan = html.escape(str(r.get(col_doan, "")))
                    goi_y = html.escape(str(r.get(col_goi_y, ""))) if col_goi_y else "—"
                    nhan_xet_raw = str(r.get(col_nx, "")).strip() if col_nx else ""
                    nhan_xet = html.escape(nhan_xet_raw) if nhan_xet_raw else "<i>Chưa có nhận xét (nhóm điền thêm vào outputs/phan_tich_loi.csv)</i>"

                    seg_html = f'<div style="font-size:0.80rem;color:#64748B;margin-top:4px;">Sau xử lý: <code>{cau_seg}</code></div>' if cau_seg else ""
                    st.markdown(
                        f"""
                        <div class="sentinel-card" style="border-left: 5px solid {mau_vien}; padding: 14px 16px;">
                            <div style="font-size:0.93rem;font-weight:600;color:#0F172A;">"{cau_goc}"</div>
                            {seg_html}
                            <div style="margin-top:8px;font-size:0.84rem;">
                                <span class="mono-tag" style="background:#F1F5F9;color:#334155;">Nhãn thật: {nh_that}</span>
                                ➔
                                <span class="mono-tag" style="background:#FEE2E2;color:#B91C1C;">Mô hình đoán: {nh_doan}</span>
                            </div>
                            <div style="margin-top:6px;font-size:0.84rem;color:#334155;">
                                💡 <b>Gợi ý nguyên nhân:</b> {goi_y}
                            </div>
                            <div style="margin-top:4px;font-size:0.84rem;color:#475569;">
                                📝 <b>Nhận xét của nhóm:</b> {nhan_xet}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with c_fp:
                df_fp = df_loi[df_loi[col_loai] == "FP"] if col_loai else pd.DataFrame()
                st.markdown(f"#### 🟠 Báo nhầm — False Positive ({len(df_fp)} mẫu trong CSV)")
                st.caption("Câu thật sự **Bình thường** nhưng bị mô hình báo nhầm là **Độc hại**.")
                _ve_danh_sach_the_loi(df_fp, "#D97706", "FP")

            with c_fn:
                df_fn = df_loi[df_loi[col_loai] == "FN"] if col_loai else pd.DataFrame()
                st.markdown(f"#### 🔴 Bỏ sót — False Negative ({len(df_fn)} mẫu trong CSV)")
                st.caption("Câu thật sự **Độc hại (Xúc phạm / Thù ghét)** nhưng bị mô hình đoán là **Bình thường**.")
                _ve_danh_sach_the_loi(df_fn, "#DC2626", "FN")

            with st.expander("📋 Xem toàn bộ bảng outputs/phan_tich_loi.csv"):
                st.dataframe(df_loi, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Không đọc được `outputs/phan_tich_loi.csv`: {e}")
    else:
        thong_bao_thieu_file("outputs/phan_tich_loi.csv", "python scripts/evaluate_final.py")

hien_chan_trang_canh_bao()
