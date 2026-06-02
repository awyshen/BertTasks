from __future__ import annotations

from bert_tasks import parse


def first_task(text: str) -> dict:
    result = parse(text)
    assert result["query_type"] == "single_task"
    assert result["source"] == "rule_template"
    return result["tasks"][0]


def test_volume_controls() -> None:
    assert first_task("音量调低一些")["params"] == {"volume": "down"}
    assert first_task("声音大点")["params"] == {"volume": "up"}
    assert first_task("静音")["params"] == {"volume": "mute"}
    assert first_task("声音调整到70%")["params"] == {"volume": "70"}
    assert first_task("音量调到最大")["params"] == {"volume": "100"}


def test_music_controls_and_slots() -> None:
    assert first_task("上一首歌")["params"] == {"control": "previous"}
    assert first_task("切歌")["params"] == {"control": "next"}
    assert first_task("暂停播放歌曲")["params"] == {"control": "pause"}
    assert first_task("不要播放音乐了")["params"] == {"control": "stop"}
    assert first_task("恢复播放")["params"] == {"control": "play"}
    assert first_task("播放蔡琴的渡口")["params"] == {"singer": "蔡琴", "song": "渡口"}
    assert first_task("播放蔡琴的歌")["params"] == {"singer": "蔡琴"}
    assert first_task("播放歌曲渡口")["params"] == {"song": "渡口"}
    assert first_task("打开QQ音乐")["params"] == {"control": "open", "app": "qq_music_app"}
    assert first_task("关闭网易云音乐")["params"] == {"control": "close", "app": "netease_music_app"}


def test_video_controls_and_slots() -> None:
    assert first_task("打开爱奇艺")["params"] == {"control": "open", "app": "iqiyi_video_app"}
    assert first_task("关闭腾讯视频")["params"] == {"control": "close", "app": "tencent_video_app"}
    assert first_task("播放电影变形金刚")["params"] == {
        "control": "play",
        "content": "变形金刚",
        "content_type": "movie",
    }
    assert first_task("播放喜剧人单口季节目")["params"] == {
        "control": "play",
        "content": "喜剧人单口季",
        "content_type": "program",
    }


def test_projector_robot_and_assistant() -> None:
    assert first_task("打开投影仪")["params"] == {"control": "open"}
    assert first_task("投影仪关了")["params"] == {"control": "close"}
    assert first_task("导航到客厅")["params"] == {"place": "客厅"}
    assert first_task("去书房吧")["params"] == {"place": "书房"}
    assert first_task("取消导航")["params"] == {"control": "cancel"}
    assert first_task("回去充电")["params"] == {"control": "start"}
    assert first_task("停止充电")["params"] == {"control": "stop"}
    assert first_task("休息一下")["params"] == {"control": "sleep"}
    assert first_task("进入聊天模式")["params"] == {"control": "chat"}


def test_multi_task() -> None:
    result = parse("到客厅打开投影仪")
    assert result["query_type"] == "multi_task"
    assert result["source"] == "rule_template"
    assert result["tasks"][0]["params"] == {"place": "客厅"}
    assert result["tasks"][1]["params"] == {"control": "open"}

    result = parse("去卧室然后播放蔡琴的歌")
    assert result["query_type"] == "multi_task"
    assert result["source"] == "rule_template"
    assert result["tasks"][0]["params"] == {"place": "卧室"}
    assert result["tasks"][1]["params"] == {"singer": "蔡琴"}


def test_multi_task_common_robot_sequences() -> None:
    result = parse("打开投影仪播放蔡琴的歌")
    assert result["query_type"] == "multi_task"
    assert [task["intent"] for task in result["tasks"]] == ["projector_control", "music_control"]

    result = parse("到客厅把投影打开再打开QQ音乐")
    assert result["query_type"] == "multi_task"
    assert result["tasks"][0]["params"] == {"place": "客厅"}
    assert result["tasks"][1]["params"] == {"control": "open"}
    assert result["tasks"][2]["params"] == {"control": "open", "app": "qq_music_app"}

    result = parse("关闭优酷之后音量调到30")
    assert result["query_type"] == "multi_task"
    assert result["tasks"][0]["params"] == {"control": "close", "app": "youku_video_app"}
    assert result["tasks"][1]["params"] == {"volume": "30"}

    result = parse("去厨房顺便打开爱奇艺")
    assert result["query_type"] == "multi_task"
    assert result["tasks"][0]["params"] == {"place": "厨房"}
    assert result["tasks"][1]["params"] == {"control": "open", "app": "iqiyi_video_app"}


def test_splitter_does_not_over_split_single_tasks() -> None:
    assert first_task("暂停播放歌曲")["params"] == {"control": "pause"}
    assert first_task("播放电影变形金刚")["params"] == {
        "control": "play",
        "content": "变形金刚",
        "content_type": "movie",
    }


def test_unknown_for_unsupported_or_ambiguous() -> None:
    assert parse("") == {"query_type": "unknown", "source": "unknown", "tasks": []}
    assert parse("帮我查一下天气") == {"query_type": "unknown", "source": "unknown", "tasks": []}
    assert parse("播放点东西") == {"query_type": "unknown", "source": "unknown", "tasks": []}
