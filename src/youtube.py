import re
import requests
import pandas as pd

YOUTUBE_COMMENT_THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def lay_video_id(link_hoac_id: str) -> str:
    """Trích xuất video ID (11 ký tự) từ link YouTube hoặc chuỗi ID."""
    if not link_hoac_id or not isinstance(link_hoac_id, str):
        raise ValueError("Link video đang để trống. Hãy dán link dạng https://www.youtube.com/watch?v=XXXXXXXXXXX")

    chuoi = link_hoac_id.strip()
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", chuoi)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", chuoi):
        return chuoi
    raise ValueError(
        "Không nhận ra mã video (Video ID). Hãy dán link YouTube hợp lệ "
        "(ví dụ: https://www.youtube.com/watch?v=XXXXXXXXXXX hoặc https://youtu.be/XXXXXXXXXXX)."
    )


def _giai_thich_loi_youtube(status_code: int, reason: str, raw_msg: str) -> str:
    """Chuyển mã lỗi kỹ thuật của YouTube Data API v3 sang tiếng Việt dễ hiểu."""
    reason_lower = (reason or "").lower()
    msg_lower = (raw_msg or "").lower()

    if "commentsdisabled" in reason_lower or "disabled comments" in msg_lower:
        return "Video này đã tắt tính năng bình luận (commentsDisabled). Hãy thử một video công khai khác."
    if "quotaexceeded" in reason_lower or "dailylimitexceeded" in reason_lower:
        return (
            "Tài khoản API key đã hết hạn mức (quotaExceeded) trong ngày hôm nay. "
            "Bạn có thể dùng chế độ 'Dữ liệu dự phòng (ket_qua_youtube.csv)' bên dưới để tiếp tục demo."
        )
    if "keyinvalid" in reason_lower or "badrequest" in reason_lower and "api key" in msg_lower:
        return "API key không hợp lệ (keyInvalid). Hãy kiểm tra lại mã YouTube Data API v3 của bạn."
    if "videonotfound" in reason_lower or status_code == 404:
        return "Không tìm thấy video (videoNotFound) hoặc video đang ở chế độ riêng tư."
    if "forbidden" in reason_lower or status_code == 403:
        return (
            f"YouTube từ chối truy cập (mã 403 - {reason}). Có thể YouTube Data API v3 chưa được bật "
            "trong Google Cloud Console, video tắt bình luận, hoặc đã hết hạn mức."
        )
    return f"YouTube API trả về lỗi {status_code} ({reason}): {raw_msg[:160]}"


def lay_binh_luan_youtube(
    link: str,
    api_key: str,
    toi_da: int = 100,
    http_get=requests.get,
) -> pd.DataFrame:
    """
    Lấy tối đa `toi_da` (giới hạn <= 500) bình luận gốc của 1 video YouTube công khai.
    Bảo mật & quyền riêng tư:
    - Không ghi API key ra file hay log.
    - Không lưu tên người bình luận (authorDisplayName), chỉ lưu: text, like, thoi_gian.
    """
    if not api_key or not str(api_key).strip():
        raise ValueError("Chưa nhập YouTube API key. Vui lòng nhập API key vào ô mật khẩu hoặc chuyển sang chế độ dự phòng.")

    vid = lay_video_id(link)
    gioi_han = min(max(1, int(toi_da)), 500)

    dong = []
    token = None

    while len(dong) < gioi_han:
        tham_so = {
            "part": "snippet",
            "videoId": vid,
            "maxResults": min(100, gioi_han - len(dong)),
            "textFormat": "plainText",
            "order": "relevance",
            "key": str(api_key).strip(),
        }
        if token:
            tham_so["pageToken"] = token

        try:
            r = http_get(YOUTUBE_COMMENT_THREADS_URL, params=tham_so, timeout=25)
        except requests.RequestException:
            raise RuntimeError(
                "Không kết nối được tới máy chủ YouTube (có thể mất mạng). "
                "Hãy dùng chế độ dự phòng đọc file outputs/ket_qua_youtube.csv."
            )

        if r.status_code != 200:
            reason = "unknownError"
            msg = ""
            try:
                err_json = r.json().get("error", {})
                msg = err_json.get("message", "")
                errors_list = err_json.get("errors", [])
                if errors_list:
                    reason = errors_list[0].get("reason", reason)
            except Exception:
                msg = r.text[:200]
            raise RuntimeError(_giai_thich_loi_youtube(r.status_code, reason, msg))

        du_lieu = r.json()
        for it in du_lieu.get("items", []):
            s = it.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            # Tuyệt đối không lấy authorDisplayName hay authorChannelId để bảo vệ quyền riêng tư
            dong.append(
                {
                    "text": str(s.get("textDisplay", "")).strip(),
                    "like": int(s.get("likeCount", 0) or 0),
                    "thoi_gian": str(s.get("publishedAt", "")),
                }
            )

        token = du_lieu.get("nextPageToken")
        if not token:
            break

    df = pd.DataFrame(dong[:gioi_han], columns=["text", "like", "thoi_gian"])
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df
