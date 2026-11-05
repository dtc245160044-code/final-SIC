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

STREAMLIT_PAGES = [
    "app/Home.py",
    "app/pages/1_Thu_nghiem.py",
    "app/pages/2_Tien_xu_ly.py",
    "app/pages/3_Danh_gia_mo_hinh.py",
    "app/pages/4_YouTube.py",
    "app/pages/5_Han_che_dao_duc.py",
]


def _load_20_real_val_sentences() -> list[str]:
    val_path = os.path.join(ROOT_DIR, "data", "processed", "03_val.csv")
    df_val = pd.read_csv(val_path)
    return df_val["text"].dropna().astype(str).head(20).tolist()


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
    long_inp = "Bác Trần Dần trả lời câu hỏi cuối rất hay " * 15  # > 600 ký tự
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

    # 4. Câu thật có teencode từ tập Test ViHSD
    res_tc = phan_loai(
        "Bán hàng đàng hoàng phải ko, bán hàng lương thiện đúng không, dám cho đối chứng ko, công an cồng đầu chứ ở đó mà xạo xạo, qua mặt thiên hạ hả bưởi"
    )
    assert REQUIRED_KEYS.issubset(res_tc.keys())
    assert res_tc["nhan_id"] in (1, 2)
    assert res_tc["diem_toxic"] > 0.5

    # 5. Đầu vào lạ (None, số nguyên, số thực)
    for weird_inp in [None, 12345, 3.14159]:
        res_w = phan_loai(weird_inp)
        assert REQUIRED_KEYS.issubset(res_w.keys())
        assert isinstance(res_w["nhan_id"], int)


def test_20_sample_sentences_contract():
    """Kiểm tra 20 câu thật từ 03_val.csv trả về đúng khóa, đúng kiểu dữ liệu và xác suất hợp lệ."""
    cfg = doc_cau_hinh()
    assert "models_available" in cfg
    assert len(cfg["models_available"]) >= 1
    ten_mh = cfg["best_model"]

    twenty_real_sentences = _load_20_real_val_sentences()
    assert len(twenty_real_sentences) == 20

    for cau in twenty_real_sentences:
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

    # Kiểm tra phân loại hàng loạt trên 20 câu thật
    df_batch = phan_loai_hang_loat(twenty_real_sentences, ten_mo_hinh=ten_mh)
    assert isinstance(df_batch, pd.DataFrame)
    assert len(df_batch) == 20
    expected_cols = {"text", "sau_xu_ly", "nhan", "do_tin_cay", "diem_toxic"}
    assert expected_cols.issubset(df_batch.columns)

    # Kiểm tra tìm từ gây độc trên câu thật từ 03_test.csv
    tu_doc = tim_tu_gay_doc(
        "Địt mẹ, ông thử ăn cá kho đi, càng hâm lại nó ăn lại càng ngon vailone :)",
        ten_mo_hinh=ten_mh,
    )
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
