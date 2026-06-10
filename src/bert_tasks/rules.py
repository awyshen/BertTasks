from __future__ import annotations

import re
from collections.abc import Callable

from bert_tasks.normalization import normalize_text, normalize_volume
from bert_tasks.schema import ParsedTask


MUSIC_APPS = {
    "QQ音乐": "qq_music_app",
    "qq音乐": "qq_music_app",
    "网易云音乐": "netease_music_app",
    "酷狗音乐": "kugou_music_app",
    "音乐播放器": "default_music_app",
    "默认音乐": "default_music_app",
}

VIDEO_APPS = {
    "本地视频": "default_video_app",
    "视频": "default_video_app",
    "爱奇艺": "iqiyi_video_app",
    "爱艺奇": "iqiyi_video_app",
    "腾讯视频": "tencent_video_app",
    "优酷": "youku_video_app",
}

CONTENT_TYPES = {
    "电影": "movie",
    "影片": "movie",
    "综艺": "program",
    "节目": "program",
    "电视剧": "drama",
    "剧集": "drama",
    "短视频": "short_video",
    "直播": "live",
}

PLACES = [
    "客厅",
    "卧室",
    "主卧",
    "次卧",
    "书房",
    "儿童房",
    "老人房",
    "厨房",
    "阳台",
    "卫生间",
    "洗手间",
    "玄关",
    "餐厅",
    "充电桩",
    "会议室",
    "前台",
]

NEGATIVE_NAV_PLACES = {"充电", "充电桩"}


def _task(user_input: str, intent: str, value: str, params: dict[str, str]) -> ParsedTask:
    return ParsedTask(user_input=user_input, intent=intent, value=value, params=params)


def parse_single_rule(text: str) -> ParsedTask | None:
    text = normalize_text(text)
    if not text:
        return None
    parsers: list[Callable[[str], ParsedTask | None]] = [
        parse_volume,
        parse_music,
        parse_video,
        parse_projector,
        parse_robot,
        parse_assistant,
    ]
    matches = [task for parser in parsers if (task := parser(text)) is not None]
    if len(matches) != 1:
        return None
    return matches[0]


def parse_volume(text: str) -> ParsedTask | None:
    if re.search(r"(静音|安静|音量关掉|声音关掉)", text):
        return _task(text, "volume_control", "speaker", {"volume": "mute"})
    if re.search(r"(调到|调整到|调至|调整至|设置为|设为)(最大|最高|满格|满|最小|最低|[0-9]{1,3}|[零〇一二两三四五六七八九十百]{1,3})%?", text):
        raw = re.search(r"(最大|最高|满格|满|最小|最低|[0-9]{1,3}|[零〇一二两三四五六七八九十百]{1,3})%?", text)
        if raw:
            volume = normalize_volume(raw.group(1))
            if volume is not None:
                return _task(text, "volume_control", "speaker", {"volume": volume})
    if re.search(r"(音量|声音).*(调低|降低|小一点|小点|轻一点|收一收|压低|太大|太吵|低一点)", text) or re.search(r"(小声点|降低点分贝)", text):
        return _task(text, "volume_control", "speaker", {"volume": "down"})
    if re.search(r"(音量|声音).*(调高|提高|大一点|大点|响一点|太小|高一点)", text) or re.search(r"(大点声|提高点分贝)", text):
        return _task(text, "volume_control", "speaker", {"volume": "up"})
    return None


def _clean_song_title(text: str) -> str:
    """Clean song title by removing trailing punctuation and extra characters."""
    text = text.strip()
    text = re.sub(r"[，,。.、；;！!？?]+$", "", text)
    text = re.sub(r"^(歌曲|歌)\s*", "", text)
    return text.strip()


def _clean_singer_name(text: str) -> str:
    """Clean singer name by removing extra characters."""
    text = text.strip()
    text = re.sub(r"[，,。.、；;！!？?]+$", "", text)
    return text.strip()


def parse_music(text: str) -> ParsedTask | None:
    controls = [
        (r"(上一首|上一曲|前一首|切回前一首|再来一遍刚才那歌)", "previous"),
        (r"(下一首|下一曲|换下首|切歌|跳过|换一首)", "next"),
        (r"(暂停.*(音乐|歌曲|放歌)|暂时先不听|先停下)", "pause"),
        (r"(停止.*(音乐|歌曲|放歌)|不要.*(音乐|歌曲|放歌)|别放歌|音乐关了|不想听歌|结束播放歌曲|音乐停下来)", "stop"),
        (r"(继续.*(音乐|歌曲|放歌)|接着放歌|恢复播放|接着听音乐)", "play"),
    ]
    for pattern, control in controls:
        if re.search(pattern, text):
            return _task(text, "music_control", "music_player", {"control": control})

    app = _find_app(text, MUSIC_APPS)
    if app and re.search(r"(打开|启动|开启)", text):
        return _task(text, "music_control", "music_player", {"control": "open", "app": app})
    if app and re.search(r"(关闭|退出|关掉)", text):
        return _task(text, "music_control", "music_player", {"control": "close", "app": app})
    if re.fullmatch(r"(打开|启动|开启)音乐", text):
        return _task(text, "music_control", "music_player", {"control": "open", "app": "default_music_app"})
    if re.fullmatch(r"(关闭|退出|关掉)音乐", text):
        return _task(text, "music_control", "music_player", {"control": "close", "app": "default_music_app"})

    if not re.search(r"(播放|放|来一首|来首|想听|唱)", text):
        return None
    if re.search(r"(视频|电影|综艺|节目|电视剧|剧集|短视频|直播)", text):
        return None
    
    music_text = text
    music_text = re.sub(r"^(请|帮我|给我)\s*", "", music_text)
    music_text = re.sub(r"^(播放|放|来一首|来首|我想听|想听|唱)\s*", "", music_text)
    music_text = re.sub(r"\s*(歌曲|歌|音乐)$", "", music_text)
    
    music_text = re.sub(r"^(\d+首)?\s*", "", music_text)
    music_text = re.sub(r"^(\d+首)?\s*(歌曲|歌)\s*", "", music_text)
    music_text = re.sub(r"^(\d+)首\s*", "", music_text)
    music_text = re.sub(r"^(\d+)首\s*(歌曲|歌)\s*", "", music_text)
    
    music_text = re.sub(r"^(一首|两首|三首|几首|数首)\s*", "", music_text)
    music_text = re.sub(r"^(一首|两首|三首|几首|数首)\s*(歌曲|歌)\s*", "", music_text)
    
    music_text = music_text.strip()
    
    if not music_text:
        return None
    
    singer = None
    song = None
    
    comma_pattern = re.search(r"(.+?)[,，](.+?)的", music_text)
    if comma_pattern:
        song = _clean_song_title(comma_pattern.group(1))
        singer = _clean_singer_name(comma_pattern.group(2))
    
    if singer is None and song is None and "的" in music_text:
        parts = music_text.split("的")
        
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            
            left = re.sub(r"^(\d+首)?", "", left)
            right = re.sub(r"^(\d+首)?", "", right)
            
            if left and right:
                singer = _clean_singer_name(left)
                song = _clean_song_title(right)
            
            elif left and not right:
                song = _clean_song_title(left)
            
            elif right and not left:
                song = _clean_song_title(right)
        
        elif len(parts) > 2:
            singer = _clean_singer_name(parts[0].strip())
            song = _clean_song_title("的".join(parts[1:]).strip())
    
    if singer is None and song is None and ("," in music_text or "，" in music_text):
        sep = "," if "," in music_text else "，"
        parts = music_text.split(sep)
        if len(parts) >= 2:
            song_candidate = _clean_song_title(parts[0].strip())
            singer_candidate = _clean_singer_name(parts[-1].strip())
            
            if song_candidate and singer_candidate:
                song = song_candidate
                singer = singer_candidate
            elif song_candidate:
                song = song_candidate
            elif singer_candidate:
                singer = singer_candidate
    
    if singer is None and song is None:
        clean_text = re.sub(r"^(\d+首)?", "", music_text)
        clean_text = re.sub(r"^(歌曲|歌)\s*", "", clean_text)
        song = _clean_song_title(clean_text)
    
    params: dict[str, str] = {}
    if singer:
        params["singer"] = singer
    if song:
        params["song"] = song
    
    if params:
        return _task(text, "music_control", "music_player", params)
    
    return None


def parse_video(text: str) -> ParsedTask | None:
    app = _find_app(text, VIDEO_APPS)
    if app and re.search(r"(打开|播放|放|启动|开启)", text) and not re.search(r"(电影|综艺|节目|电视剧|剧集|短视频|直播).+", text):
        return _task(text, "app_control", "video_player", {"control": "open", "app": app})
    if app and re.search(r"(关闭|停止|退出|关掉)", text):
        return _task(text, "app_control", "video_player", {"control": "close", "app": app})

    if not re.search(r"(播放|放|打开)", text):
        return None
    if not re.search(r"(视频|电影|综艺|节目|电视剧|剧集|短视频|直播|影片)", text):
        return None
    content_type = "unknown"
    content = text
    content = re.sub(r"^(请|帮我|给我)?(播放|放|打开)", "", content)
    for label, enum in CONTENT_TYPES.items():
        if label in content:
            content_type = enum
            content = content.replace(label, "", 1)
            break
    content = re.sub(r"(视频)$", "", content)
    content = re.sub(r"[，,。.、；;！!？?]+$", "", content.strip())
    if not content:
        return None
    return _task(text, "app_control", "video_player", {"control": "play", "content": content, "content_type": content_type})


def parse_projector(text: str) -> ParsedTask | None:
    if "投影" not in text:
        return None
    if re.search(r"(打开|开启|启动)", text):
        return _task(text, "projector_control", "projector", {"control": "open"})
    if re.search(r"(关闭|关了|关掉|退出)", text):
        return _task(text, "projector_control", "projector", {"control": "close"})
    return None


def parse_robot(text: str) -> ParsedTask | None:
    if re.search(r"(音量|声音|音乐|歌曲|视频|电影|综艺|节目|电视剧|剧集|投影|聊天)", text):
        return None
    if re.search(r"(取消导航|停止导航|不要去.+了)", text):
        return _task(text, "robot_control", "nav", {"control": "cancel"})
    if re.search(r"(停止充电|取消充电|不要充电了)", text):
        return _task(text, "robot_control", "charge", {"control": "stop"})
    if re.search(r"(回去充电|去充电桩|去充电|回充|开始充电)", text):
        return _task(text, "robot_control", "charge", {"control": "start"})

    place = _extract_place(text)
    if place and place not in NEGATIVE_NAV_PLACES and re.search(r"^(请|帮我)?(导航到|去|到|来)", text):
        return _task(text, "robot_control", "nav", {"place": place})
    return None


def parse_assistant(text: str) -> ParsedTask | None:
    if re.search(r"(休息一下|退下吧|先下去吧|睡眠|休眠|待机)", text):
        return _task(text, "assistant_control", "assistant", {"control": "sleep"})
    if re.search(r"(聊一下|聊天模式|打开闲聊|进入聊天|唠嗑|唠嗑模式|闲聊模式)", text):
        return _task(text, "assistant_control", "assistant", {"control": "chat"})
    return None


def _find_app(text: str, mapping: dict[str, str]) -> str | None:
    for label, enum in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        if label in text:
            return enum
    return None


def _extract_place(text: str) -> str | None:
    for place in sorted(PLACES, key=len, reverse=True):
        if place in text:
            return place
    match = re.search(r"(?:导航到|去|到|来)([^，。,.、吧]+)", text)
    if match:
        place = match.group(1)
        place = re.sub(r"(去|吧|那里|这边)$", "", place)
        place = re.sub(r"[，,。.、；;！!？?]+$", "", place.strip())
        
        if 1 <= len(place) <= 8:
            if re.match(r"^(一首|两首|三首|几首|首歌曲|首歌)", place):
                return None
            return place
    return None