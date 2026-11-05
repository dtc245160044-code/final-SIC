import os
import sys
import getpass
import argparse
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.youtube import lay_video_id, lay_binh_luan_youtube
from src.inference import doc_cau_hinh, phan_loai_hang_loat


def main():
    parser = argparse.ArgumentParser(
        description="Lấy bình luận thật từ YouTube Data API v3, phân loại và lưu vào outputs/ket_qua_youtube.csv"
    )
    parser.add_argument("--url", type=str, default=None, help="Link video YouTube hoặc Video ID (11 ký tự)")
    parser.add_argument(
        "--so_luong",
        type=int,
        default=None,
        help="Số bình luận tối đa cần lấy (từ 1 đến 200, mặc định 150)",
    )
    parser.add_argument("--mo_hinh", type=str, default=None, help="Tên mô hình phân loại (mặc định: best_model)")
    args = parser.parse_args()

    print("=== LẤY & PHÂN LOẠI BÌNH LUẬN YOUTUBE THẬT (LƯU DỰ PHÒNG CHO WEB DEMO) ===")

    link_video = args.url
    if not link_video:
        link_video = input("1. Nhập link video YouTube công khai (hoặc Video ID 11 ký tự): ").strip()

    video_id = lay_video_id(link_video)
    print(f"   -> Đã nhận diện Video ID: {video_id}")

    api_key = getpass.getpass("2. Nhập YouTube Data API v3 Key (nhập ẩn, không lưu ra file/log): ").strip()
    if not api_key:
        raise ValueError("Bạn chưa nhập API key. Đã hủy thao tác.")

    so_luong = args.so_luong
    if so_luong is None:
        raw_sl = input("3. Số bình luận tối đa muốn lấy (100-200, nhấn Enter để lấy 150): ").strip()
        so_luong = int(raw_sl) if raw_sl else 150
    so_luong = max(1, min(int(so_luong), 200))

    cfg = doc_cau_hinh()
    avail = cfg.get("models_available", [])
    if not avail:
        raise RuntimeError("Chưa có mô hình nào trong thư mục models/. Hãy chạy script huấn luyện trước.")
    chon_mo_hinh = args.mo_hinh if (args.mo_hinh and args.mo_hinh in avail) else cfg.get("best_model", avail[0])

    print(f"\nĐang gọi YouTube Data API v3 để lấy tối đa {so_luong} bình luận từ video '{video_id}'...")
    df_raw = lay_binh_luan_youtube(video_id, api_key=api_key, toi_da=so_luong)
    if df_raw.empty:
        print("Video này hiện không có bình luận công khai nào. Không tạo file.")
        return

    print(f"Đã lấy {len(df_raw)} bình luận thật. Đang phân loại bằng mô hình '{chon_mo_hinh}'...")
    df_kq = phan_loai_hang_loat(df_raw, ten_mo_hinh=chon_mo_hinh)

    ngay_lay = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_kq["video_id"] = video_id
    df_kq["ngay_lay"] = ngay_lay

    out_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ket_qua_youtube.csv")
    df_kq.to_csv(out_path, index=False, encoding="utf-8")

    print("\n=== HOÀN TẤT LƯU FILE DỰ PHÒNG YOUTUBE THẬT ===")
    print(f"- Đường dẫn file : {out_path}")
    print(f"- Video ID       : {video_id}")
    print(f"- Ngày giờ lấy   : {ngay_lay}")
    print(f"- Tổng số câu    : {len(df_kq)}")
    print("- Phân bố nhãn dự đoán:")
    for nhan, dem in df_kq["nhan"].value_counts().items():
        ty_le = dem / len(df_kq) * 100
        print(f"    + {nhan}: {dem} câu ({ty_le:.1f}%)")


if __name__ == "__main__":
    main()
