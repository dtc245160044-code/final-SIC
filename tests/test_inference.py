import os
import sys
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.inference import (
    doc_cau_hinh,
    tai_mo_hinh,
    phan_loai,
    phan_loai_hang_loat,
    tim_tu_gay_doc,
)
from src.youtube import lay_video_id

REQUIRED_KEYS = {
    "binh_luan",
    "sau_xu_ly",
    "nhan",
    "nhan_id",
    "do_tin_cay",
    "diem_toxic",
    "xac_suat_tung_nhan",
    "mo_hinh",
    "do_tre_ms",
}

TWENTY_SAMPLE_SENTENCES = [
    "Video hôm nay chia sẻ kiến thức rất bổ ích, cảm ơn tác giả nhiều nhé!",
    "Đồ ngu dốt biến khỏi đây đi đồ rác rưởi",
    "Cái mặt nhìn thấy ghét, ăn nói vô học thật sự",
    "Thật tuyệt vời! Chúc mừng đội tuyển đã cố gắng hết sức ❤️",
    "m.à.y n.g.u vcl đm ko bt j hết",
    "Sản phẩm dùng ổn trong tầm giá, giao hàng hơi chậm một chút.",
    "Lũ súc vật này sống chỉ tổ chật đất, biến hết đi!",
    "Trình độ cỡ này mà cũng đòi lên mạng dạy đời người khác à? Hài hước thật.",
    "Anh ơi cho em hỏi bài hát ở phút thứ 3 tên là gì vậy ạ?",
    "Đm thằng lừa đảo, trả tiền cho bố mày ngay!",
    "Giọng đọc truyền cảm quá, nghe buổi tối rất dễ ngủ.",
    "Bọn vô ý thức xả rác bừa bãi làm bẩn cả khu phố.",
    "Mày bị điên à mà phát biểu câu ngớ ngẩn thế hả?",
    "Cảm ơn bác sĩ đã tư vấn nhiệt tình cho gia đình em.",
    "Thằng chó này cút ngay khỏi kênh của tao!",
    "Hôm nay trời đẹp quá, đi dạo quanh hồ Tây rất mát.",
    "Ăn hại đái khai, chả làm được tích sự gì cho đời.",
    "Chúc kênh ngày càng phát triển và ra nhiều video hay hơn nữa.",
    "Cái lũ mất dạy, mở mồm ra là chửi bới người khác.",
    "Món này nấu thêm chút hành lá và tiêu xay sẽ dậy mùi thơm hơn.",
]

STREAMLIT_PAGES = [
    "app/Home.py",
    "app/pages/1_Thu_nghiem.py",
    "app/pages/2_Tien_xu_ly.py",
    "app/pages/3_Danh_gia_mo_hinh.py",
    "app/pages/4_YouTube.py",
    "app/pages/5_Han_che_dao_duc.py",
]


def test_edge_cases_phan_loai():
    """Kiểm tra câu rỗng, câu >500 ký tự, toàn emoji, teencode và đầu vào lạ."""
    # 1. Câu rỗng & chỉ khoảng trắng
    for empty_inp in ["", "   ", "\n\t  "]:
        res = phan_loai(empty_inp)
        assert REQUIRED_KEYS.issubset(res.keys())
        assert "error" in res
        assert res["nhan_id"] == -1
        assert res["diem_toxic"] == 0.0

    # 2. Câu quá dài (>500 ký tự)
    long_inp = "Bình luận thử nghiệm rất dài " * 30  # > 800 ký tự
    assert len(long_inp) > 500
    res_long = phan_loai(long_inp)
    assert REQUIRED_KEYS.issubset(res_long.keys())
    assert res_long["nhan_id"] in (0, 1, 2)
    assert 0.0 <= res_long["diem_toxic"] <= 1.0

    # 3. Câu toàn emoji / ký tự đặc biệt
    for emoji_inp in ["😂😂😡🤬😭🔥", "!@#$%^&*()_+-=[]{};':\",./<>?"]:
        res_emo = phan_loai(emoji_inp)
        assert REQUIRED_KEYS.issubset(res_emo.keys())
        assert res_emo["nhan_id"] == 0
        assert res_emo["diem_toxic"] < 0.5

    # 4. Câu teencode và chèn ký tự
    res_tc = phan_loai("m.à.y n.g.u vcl đm ko bt j hết")
    assert REQUIRED_KEYS.issubset(res_tc.keys())
    assert res_tc["nhan_id"] in (1, 2)
    assert res_tc["diem_toxic"] > 0.5

    # 5. Đầu vào lạ (None, số nguyên, số thực)
    for weird_inp in [None, 12345, 3.14159]:
        res_w = phan_loai(weird_inp)
        assert REQUIRED_KEYS.issubset(res_w.keys())
        assert isinstance(res_w["nhan_id"], int)


def test_20_sample_sentences_contract():
    """Kiểm tra 20 câu mẫu trả về đúng khóa, đúng kiểu dữ liệu và xác suất hợp lệ."""
    cfg = doc_cau_hinh()
    assert "models_available" in cfg
    assert len(cfg["models_available"]) >= 1
    ten_mh = cfg["best_model"]

    for cau in TWENTY_SAMPLE_SENTENCES:
        res = phan_loai(cau, ten_mo_hinh=ten_mh)
        assert REQUIRED_KEYS.issubset(res.keys()), f"Thiếu khóa trong kết quả cho câu: {cau}"
        assert isinstance(res["binh_luan"], str)
        assert isinstance(res["sau_xu_ly"], str) and len(res["sau_xu_ly"]) > 0
        assert isinstance(res["nhan"], str) and res["nhan"] in cfg["label_names"]
        assert isinstance(res["nhan_id"], int) and res["nhan_id"] in (0, 1, 2)
        assert isinstance(res["do_tin_cay"], float) and 0.0 <= res["do_tin_cay"] <= 1.0
        assert isinstance(res["diem_toxic"], float) and 0.0 <= res["diem_toxic"] <= 1.0
        assert isinstance(res["xac_suat_tung_nhan"], dict) and len(res["xac_suat_tung_nhan"]) == 3
        assert abs(sum(res["xac_suat_tung_nhan"].values()) - 1.0) < 0.02
        assert isinstance(res["mo_hinh"], str)
        assert isinstance(res["do_tre_ms"], float) and res["do_tre_ms"] >= 0.0

    # Kiểm tra phân loại hàng loạt trên 20 câu
    df_batch = phan_loai_hang_loat(TWENTY_SAMPLE_SENTENCES, ten_mo_hinh=ten_mh)
    assert isinstance(df_batch, pd.DataFrame)
    assert len(df_batch) == 20
    expected_cols = {"text", "sau_xu_ly", "nhan", "do_tin_cay", "diem_toxic"}
    assert expected_cols.issubset(df_batch.columns)

    # Kiểm tra tìm từ gây độc
    tu_doc = tim_tu_gay_doc("Đồ ngu dốt biến khỏi đây đi", ten_mo_hinh=ten_mh)
    assert isinstance(tu_doc, list)
    assert len(tu_doc) >= 1
    assert any(item["gay_doc"] for item in tu_doc)


def test_youtube_url_parser():
    """Kiểm tra hàm tách video ID từ nhiều dạng link YouTube."""
    assert lay_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert lay_video_id("https://youtu.be/dQw4w9WgXcQ?si=12345") == "dQw4w9WgXcQ"
    assert lay_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    with pytest.raises(ValueError):
        lay_video_id("link_khong_hop_le")
    with pytest.raises(ValueError):
        lay_video_id("")


def _run_all_6_pages():
    """Chạy cả 6 trang Streamlit bằng AppTest và đảm bảo không trang nào văng ngoại lệ."""
    for rel_path in STREAMLIT_PAGES:
        abs_path = os.path.join(ROOT_DIR, rel_path)
        at = AppTest.from_file(abs_path, default_timeout=30).run()
        assert len(at.exception) == 0, f"Trang {rel_path} bị lỗi: {at.exception}"


def test_streamlit_6_pages_all_3_states():
    """Kiểm tra cả 6 trang Streamlit chạy mượt ở 3 trạng thái: (1) đủ file, (2) thiếu outputs/, (3) thiếu models/."""
    outputs_dir = os.path.join(ROOT_DIR, "outputs")
    outputs_bak = os.path.join(ROOT_DIR, "outputs_bak_pytest")
    models_dir = os.path.join(ROOT_DIR, "models")
    models_bak = os.path.join(ROOT_DIR, "models_bak_pytest")

    # Trạng thái 1: Đủ cả outputs/ và models/
    _run_all_6_pages()

    # Trạng thái 2: Thiếu thư mục outputs/
    if os.path.exists(outputs_dir):
        os.rename(outputs_dir, outputs_bak)
    try:
        _run_all_6_pages()
    finally:
        if os.path.exists(outputs_bak):
            os.rename(outputs_bak, outputs_dir)

    # Trạng thái 3: Thiếu thư mục models/
    if os.path.exists(models_dir):
        os.rename(models_dir, models_bak)
    try:
        _run_all_6_pages()
    finally:
        if os.path.exists(models_bak):
            os.rename(models_bak, models_dir)
