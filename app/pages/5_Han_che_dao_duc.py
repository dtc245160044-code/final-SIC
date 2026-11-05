import os
import sys
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ui import cai_dat_trang, hien_chan_trang_canh_bao, OUTPUTS_DIR, DATA_PROCESSED_DIR

cai_dat_trang(tieu_de="Hạn chế & Đạo đức", icon="⚖️")

st.title("⚖️ Hạn chế của Mô hình & Khía cạnh Đạo đức AI")
st.markdown(
    "Trang này tổng hợp các **hạn chế kỹ thuật** và **nguyên tắc đạo đức** dựa trên số liệu thực nghiệm "
    "và phân tích lỗi thật của dự án trên bộ dữ liệu **ViHSD**."
)

st.info(
    "📝 **Ghi chú cho nhóm:** Các đoạn có đánh dấu **`[CẦN NHÓM ĐỌC LẠI & CHỈNH SỬA]`** bên dưới là bản nháp "
    "được tổng hợp tự động từ số liệu thật trong `data/processed/` và `outputs/`. Nhóm hãy đọc lại và bổ sung "
    "nhận xét riêng của nhóm trước khi bảo vệ."
)

# Đọc số liệu thật từ file để hiển thị động (không hard-code số liệu)
ty_le_txt = "Chưa có file dữ liệu"
train_path = os.path.join(DATA_PROCESSED_DIR, "03_train.csv")
if os.path.isfile(train_path):
    try:
        df_tr = pd.read_csv(train_path, usecols=["label"])
        vc = (df_tr["label"].value_counts(normalize=True).sort_index() * 100).round(2)
        ty_le_txt = (
            f"Nhãn 0 (CLEAN) chiếm `{vc.get(0, 0):.2f}%`, "
            f"Nhãn 1 (OFFENSIVE) chỉ chiếm `{vc.get(1, 0):.2f}%`, "
            f"và Nhãn 2 (HATE) chiếm `{vc.get(2, 0):.2f}%` (trên tổng số `{len(df_tr):,}` mẫu huấn luyện)"
        )
    except Exception:
        pass

thong_ke_loi_txt = "Chưa có file `outputs/phan_tich_loi.csv` (hãy chạy `python scripts/evaluate_final.py`)."
loi_path = os.path.join(OUTPUTS_DIR, "phan_tich_loi.csv")
if os.path.isfile(loi_path):
    try:
        df_loi = pd.read_csv(loi_path, keep_default_na=False)
        col_loai = "loại_lỗi" if "loại_lỗi" in df_loi.columns else "loai_loi"
        n_fp = int((df_loi[col_loai] == "FP").sum())
        n_fn = int((df_loi[col_loai] == "FN").sum())
        thong_ke_loi_txt = f"File `outputs/phan_tich_loi.csv` hiện ghi nhận `{n_fp}` mẫu lỗi Báo nhầm (FP) và `{n_fn}` mẫu lỗi Bỏ sót (FN) tiêu biểu trên tập Test."
    except Exception:
        pass

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        f"""
        <div class="sentinel-card" style="border-left: 5px solid #2563EB;">
            <h4>1. Nguồn dữ liệu & Độ lệch nhãn (Class Imbalance)</h4>
            <ul>
                <li><b>Nguồn dữ liệu:</b> Bộ dữ liệu <b>ViHSD (Vietnamese Hate Speech Detection)</b> gồm các bình luận thu thập từ mạng xã hội Việt Nam và được gán nhãn thủ công bởi nhiều người gán nhãn (crowd-annotated) theo 3 mức: <code>0=CLEAN</code>, <code>1=OFFENSIVE</code>, <code>2=HATE</code>.</li>
                <li><b>Tỷ lệ nhãn thực tế (đọc từ file):</b> {ty_le_txt}.</li>
                <li><b>Ranh giới mờ ở nhãn OFFENSIVE:</b> Trong 240 văn bản trùng lặp xuất hiện nhiều lần ở dữ liệu gốc, nhiều câu được gán đồng thời cả nhãn <code>0</code>, <code>1</code> hoặc <code>2</code>. Điều này cho thấy ranh giới giữa câu nói đùa có từ suồng sã (CLEAN), câu xúc phạm (OFFENSIVE) và câu thù ghét (HATE) mang tính chủ quan cao ngay cả với người gán nhãn.</li>
            </ul>
        </div>

        <div class="sentinel-card" style="border-left: 5px solid #D97706;">
            <h4>2. Các trường hợp mô hình dễ đoán sai</h4>
            <p><i>📝 [CẦN NHÓM ĐỌC LẠI & CHỈNH SỬA — Đối chiếu với các câu cụ thể trong tab Phân tích lỗi]:</i></p>
            <ul>
                <li><b>Nói mỉa mai, châm biếm ngầm (Sarcasm / Irony):</b> Các câu dùng từ ngữ hoàn toàn lịch sự nhưng mang hàm ý chế giễu hoặc hạ thấp người khác thường bị mô hình xếp vào nhóm <i>Bình thường</i> (False Negative).</li>
                <li><b>Từ tục dùng trong ngữ cảnh đùa vui / khen ngợi bạn bè:</b> Nhiều bình luận trên mạng xã hội dùng từ lóng/tục (như <code>vl</code>, <code>đm</code>, <code>bóp dái</code>...) để trêu đùa streamer hoặc khen ngợi nhưng bị mô hình bắt theo từ khóa và báo nhầm thành <i>Độc hại</i> (False Positive).</li>
                <li><b>Thiếu ngữ cảnh hội thoại:</b> Mô hình chỉ đọc từng bình luận rời rạc, không biết nội dung video gốc hay bình luận trước đó đang nói về chủ đề gì.</li>
                <li><b>Biến thể teencode và lách bộ lọc mới:</b> Dù <code>src/preprocess.py</code> đã xử lý 42 từ teencode phổ biến và chữ chèn dấu chấm/gạch ngang, người dùng mạng luôn sáng tạo cách viết trại âm mới chưa có trong từ điển.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="sentinel-card" style="border-left: 5px solid #DC2626;">
            <h4>3. Nguy cơ chặn nhầm & Quyền tự do ngôn luận</h4>
            <p><i>📝 [CẦN NHÓM ĐỌC LẠI & CHỈNH SỬA]:</i></p>
            <ul>
                <li><b>Tác hại của Báo nhầm (False Positive):</b> Nếu hệ thống tự động xóa bình luận hoặc khóa tài khoản ngay khi mô hình báo độc hại, những người dùng đang tranh luận gay gắt nhưng hợp lệ (hoặc dùng phương ngữ, từ cảm thán) sẽ bị <b>chặn nhầm</b>, gây ức chế và ảnh hưởng tới quyền tự do bày tỏ ý kiến.</li>
                <li><b>Tác hại của Bỏ sót (False Negative):</b> Ngược lại, nếu đặt ngưỡng quá cao để tránh báo nhầm, các bình luận công kích cá nhân tinh vi sẽ lọt lưới và gây tổn thương tâm lý cho nạn nhân.</li>
                <li><b>Trạng thái phân tích lỗi hiện tại:</b> {thong_ke_loi_txt}</li>
            </ul>
        </div>

        <div class="sentinel-card" style="border-left: 5px solid #16A34A;">
            <h4>4. Thiên lệch dữ liệu (Bias) & Nguyên tắc sử dụng có trách nhiệm</h4>
            <p><i>📝 [CẦN NHÓM ĐỌC LẠI & CHỈNH SỬA]:</i></p>
            <ul>
                <li><b>Thiên lệch chủ đề (Topic / Entity Bias):</b> Dữ liệu ViHSD tập trung nhiều vào một số luồng livestream, bóng đá và sự kiện xã hội nhất định. Một số tên riêng, địa danh vùng miền hoặc từ viết tắt xuất hiện dày đặc trong các cuộc cãi vã dễ bị mô hình gắn chặt với nhãn độc hại dù được nhắc tới trong câu trung lập.</li>
                <li><b>Bảo vệ quyền riêng tư:</b> Khi thu thập bình luận từ YouTube API, hệ thống tuyệt đối <b>không lưu tên người bình luận hay ID kênh</b>, chỉ phân tích nội dung văn bản công khai.</li>
                <li><b>Nguyên tắc Con người làm trung tâm (Human-in-the-loop):</b> Hệ thống này <b>chỉ đóng vai trò công cụ sàng lọc và gợi ý điểm ưu tiên</b> cho người kiểm duyệt, tuyệt đối không tự động ra quyết định kỷ luật hay xóa nội dung mà không có con người xác nhận.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="sentinel-card">
        <h4>🚀 5. Hướng phát triển tiếp theo</h4>
        <p><i>📝 [CẦN NHÓM ĐỌC LẠI & CHỈNH SỬA]:</i></p>
        <ol style="margin-bottom:0;">
            <li><b>Đưa thêm ngữ cảnh (Context-aware):</b> Kết hợp tiêu đề video/bài viết và bình luận cha (parent comment) làm đầu vào cho PhoBERT thay vì chỉ phân loại câu đơn lẻ.</li>
            <li><b>Cải thiện chất lượng nhãn OFFENSIVE:</b> Rà soát lại các mẫu bất đồng nhãn (label noise) trong tập dữ liệu và bổ sung hướng dẫn gán nhãn rõ ràng hơn giữa <i>từ cảm thán suồng sã</i> và <i>lời lẽ xúc phạm có chủ đích</i>.</li>
            <li><b>Mở rộng chuẩn hóa tiếng Việt:</b> Kết hợp mô hình ngôn ngữ nhỏ hoặc từ điển ngữ âm để chuẩn hóa các kiểu viết trại âm, lách bộ lọc mới ngoài danh sách 42 từ teencode tĩnh.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

hien_chan_trang_canh_bao()
