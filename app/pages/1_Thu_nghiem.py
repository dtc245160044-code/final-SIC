import os
import sys
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ui import cai_dat_trang, the_ket_qua_phan_loai, hien_chan_trang_canh_bao
from src.inference import phan_loai, tim_tu_gay_doc

cfg = cai_dat_trang(tieu_de="Thử nghiệm", icon="🧪")

st.title("🧪 Thử nghiệm phân loại bình luận")
st.markdown(
    "Nhập một bình luận tiếng Việt hoặc chọn câu mẫu trích từ tập **Test (ViHSD)** bên dưới rồi bấm **Phân tích**. "
    "Mọi kết quả đều do mô hình chạy thật theo thời gian thực."
)

# 4 câu trích nguyên văn từ data/processed/03_test.csv
MAU_THU = {
    "Mẫu 1: Khen ngợi (Test #CLEAN)": "Bác Trần Dần trả lời câu hỏi cuối rất hay....rất sáng suốt 👍",
    "Mẫu 2: Teencode công kích (Test #HATE)": "Bán hàng đàng hoàng phải ko, bán hàng lương thiện đúng không, dám cho đối chứng ko, công an cồng đầu chứ ở đó mà xạo xạo, qua mặt thiên hạ hả bưởi",
    "Mẫu 3: Mỉa mai ngầm (Test #OFFENSIVE)": "Trường hợp này ở địa phương khác kêu là gì ko biết còn địa phương mình gọi là khùng á",
    "Mẫu 4: Từ tục khen đồ ăn (Test #FP)": "Địt mẹ, ông thử ăn cá kho đi, càng hâm lại nó ăn lại càng ngon vailone :)",
}

if "text_thu_nghiem" not in st.session_state:
    st.session_state["text_thu_nghiem"] = MAU_THU["Mẫu 1: Khen ngợi (Test #CLEAN)"]

col_input, col_output = st.columns([1.05, 1.15], gap="large")

with col_input:
    st.markdown("#### ✍️ Nhập bình luận cần kiểm tra")
    st.caption("Bấm nút mẫu (trích từ `03_test.csv`) để điền nhanh câu vào ô nhập, sau đó bấm **Phân tích**:")

    btn_cols = st.columns(2)
    for idx, (ten_mau, noi_dung_mau) in enumerate(MAU_THU.items()):
        with btn_cols[idx % 2]:
            if st.button(ten_mau, use_container_width=True):
                st.session_state["text_thu_nghiem"] = noi_dung_mau

    van_ban = st.text_area(
        "Nội dung bình luận (tối đa 500 ký tự):",
        key="text_thu_nghiem",
        height=130,
        max_chars=500,
        placeholder="Nhập câu bình luận tiếng Việt vào đây...",
    )

    models_avail = cfg.get("models_available", [])
    best_model = cfg.get("best_model", "")

    c_mod, c_btn = st.columns([1.5, 1])
    with c_mod:
        if models_avail:
            default_idx = models_avail.index(best_model) if best_model in models_avail else 0
            chon_mo_hinh = st.selectbox("Chọn mô hình:", options=models_avail, index=default_idx)
        else:
            chon_mo_hinh = st.selectbox("Chọn mô hình:", options=["Chưa có mô hình trong models/"], disabled=True)
    with c_btn:
        st.write("")
        st.write("")
        bam_phan_tich = st.button("🔍 Phân tích", type="primary", use_container_width=True)

with col_output:
    st.markdown("#### 📊 Kết quả phân loại")
    if not models_avail:
        st.info("📂 **Chưa có kết quả, hãy chạy notebook/script huấn luyện trước** để tạo mô hình trong `models/`.")
    elif bam_phan_tich or st.session_state.get("text_thu_nghiem"):
        if not van_ban or not van_ban.strip():
            st.warning("Vui lòng nhập nội dung bình luận trước khi bấm Phân tích.")
        elif len(van_ban) > 500:
            st.warning("Bình luận quá dài (vượt quá 500 ký tự). Vui lòng rút gọn lại.")
        else:
            kq = phan_loai(van_ban, ten_mo_hinh=chon_mo_hinh)
            if kq.get("error") and kq.get("nhan_id", -1) == -1:
                st.warning(kq["error"])
            else:
                if kq.get("error"):
                    st.info(kq["error"])
                tu_doc = tim_tu_gay_doc(van_ban, ten_mo_hinh=chon_mo_hinh)
                the_ket_qua_phan_loai(kq, tu_gay_doc=tu_doc)

                st.markdown("**Điểm độc hại tổng hợp (`P(Xúc phạm) + P(Thù ghét)`):**")
                st.progress(min(max(float(kq["diem_toxic"]), 0.0), 1.0))

                st.markdown("**Xác suất từng nhãn:**")
                for ten_nhan, xs in kq.get("xac_suat_tung_nhan", {}).items():
                    c_lbl, c_bar, c_pct = st.columns([1.2, 2.5, 0.7])
                    with c_lbl:
                        st.write(f"{ten_nhan}")
                    with c_bar:
                        st.progress(min(max(float(xs), 0.0), 1.0))
                    with c_pct:
                        st.write(f"`{xs * 100:.1f}%`")

                hien_chan_trang_canh_bao()
