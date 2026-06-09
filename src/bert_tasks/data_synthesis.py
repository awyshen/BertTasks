from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from bert_tasks.parser import parse


def strip_source(obj: dict) -> dict:
    """Remove source field from structured output for comparison."""
    if isinstance(obj, dict):
        return {k: strip_source(v) for k, v in obj.items() if k != "source"}
    elif isinstance(obj, list):
        return [strip_source(item) for item in obj]
    return obj


def validate_row(row: dict) -> tuple[bool, str]:
    """Validate a data row has required fields."""
    required_fields = {"text", "task_type", "label", "slots", "structured_output"}
    missing = required_fields - set(row.keys())
    if missing:
        return False, f"missing_fields:{','.join(sorted(missing))}"
    
    text = row.get("text", "")
    if len(text) < 3 or len(text) > 100:
        return False, "invalid_length"
    
    if not isinstance(row.get("slots"), dict):
        return False, "invalid_slots_type"
    
    if not isinstance(row.get("structured_output"), dict):
        return False, "invalid_structured_output"
    
    return True, ""


try:
    from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


MULTI_INTENT_TASKS = {
    "volume_down": {
        "label": "volume_control:down",
        "target": {"intent": "volume_control", "value": "speaker", "params": {"volume": "down"}},
        "examples": ["音量调低一些", "声音小一点", "声音太大了"],
    },
    "volume_up": {
        "label": "volume_control:up",
        "target": {"intent": "volume_control", "value": "speaker", "params": {"volume": "up"}},
        "examples": ["音量调高一些", "声音大点", "提高点分贝"],
    },
    "volume_mute": {
        "label": "volume_control:mute",
        "target": {"intent": "volume_control", "value": "speaker", "params": {"volume": "mute"}},
        "examples": ["静音", "音量关掉", "安静"],
    },
    "music_previous": {
        "label": "music_control:previous",
        "target": {"intent": "music_control", "value": "music_player", "params": {"control": "previous"}},
        "examples": ["上一首歌", "上一曲", "切回前一首"],
    },
    "music_next": {
        "label": "music_control:next",
        "target": {"intent": "music_control", "value": "music_player", "params": {"control": "next"}},
        "examples": ["下一首歌", "切歌", "换一首"],
    },
    "music_pause": {
        "label": "music_control:pause",
        "target": {"intent": "music_control", "value": "music_player", "params": {"control": "pause"}},
        "examples": ["暂停播放歌曲", "暂停音乐", "先停下"],
    },
    "music_stop": {
        "label": "music_control:stop",
        "target": {"intent": "music_control", "value": "music_player", "params": {"control": "stop"}},
        "examples": ["停止播放歌曲", "别放歌了", "把音乐关了"],
    },
    "music_resume": {
        "label": "music_control:play",
        "target": {"intent": "music_control", "value": "music_player", "params": {"control": "play"}},
        "examples": ["继续播放音乐", "恢复播放", "接着放歌"],
    },
    "projector_open": {
        "label": "projector_control:open",
        "target": {"intent": "projector_control", "value": "projector", "params": {"control": "open"}},
        "examples": ["打开投影仪", "把投影打开"],
    },
    "projector_close": {
        "label": "projector_control:close",
        "target": {"intent": "projector_control", "value": "projector", "params": {"control": "close"}},
        "examples": ["关闭投影仪", "投影仪关了"],
    },
    "robot_cancel_nav": {
        "label": "robot_control:cancel_nav",
        "target": {"intent": "robot_control", "value": "nav", "params": {"control": "cancel"}},
        "examples": ["取消导航", "不要去客厅了", "停止导航"],
    },
    "robot_start_charge": {
        "label": "robot_control:start_charge",
        "target": {"intent": "robot_control", "value": "charge", "params": {"control": "start"}},
        "examples": ["回去充电", "去充电桩", "去充电"],
    },
    "robot_stop_charge": {
        "label": "robot_control:stop_charge",
        "target": {"intent": "robot_control", "value": "charge", "params": {"control": "stop"}},
        "examples": ["停止充电", "取消充电", "不要充电了"],
    },
    "assistant_sleep": {
        "label": "assistant_control:sleep",
        "target": {"intent": "assistant_control", "value": "assistant", "params": {"control": "sleep"}},
        "examples": ["休息一下", "退下吧", "先下去吧"],
    },
    "assistant_chat": {
        "label": "assistant_control:chat",
        "target": {"intent": "assistant_control", "value": "assistant", "params": {"control": "chat"}},
        "examples": ["我们聊一下吧", "进入聊天模式", "打开闲聊"],
    },
}

SLOT_TASKS = {
    "volume_set": {
        "label": "volume_control:set_volume",
        "slots": ["volume"],
        "target": {"intent": "volume_control", "value": "speaker", "params": {"volume": "<volume>"}},
        "examples": ["音量调到70%", "声音调整到30", "音量调到最大"],
    },
    "music_play_specific": {
        "label": "music_control:play_specific_music",
        "slots": ["singer", "song"],
        "target": {"intent": "music_control", "value": "music_player", "params": {"singer": "<singer>", "song": "<song>"}},
        "examples": ["播放蔡琴的渡口", "播放蔡琴的歌", "播放歌曲渡口"],
    },
    "music_open_app": {
        "label": "music_control:open_app",
        "slots": ["app"],
        "target": {"intent": "music_control", "value": "music_player", "params": {"control": "open", "app": "<app_enum>"}},
        "slot_enums": {"app": {"音乐播放器": "default_music_app", "QQ音乐": "qq_music_app", "网易云音乐": "netease_music_app", "酷狗音乐": "kugou_music_app"}},
        "examples": ["打开音乐播放器", "打开QQ音乐", "启动网易云音乐", "开启酷狗音乐"],
    },
    "music_close_app": {
        "label": "music_control:close_app",
        "slots": ["app"],
        "target": {"intent": "music_control", "value": "music_player", "params": {"control": "close", "app": "<app_enum>"}},
        "slot_enums": {"app": {"音乐播放器": "default_music_app", "QQ音乐": "qq_music_app", "网易云音乐": "netease_music_app", "酷狗音乐": "kugou_music_app"}},
        "examples": ["关闭音乐播放器", "关闭QQ音乐", "退出网易云音乐", "关掉酷狗音乐"],
    },
    "video_open_app": {
        "label": "app_control:open_video_app",
        "slots": ["app"],
        "target": {"intent": "app_control", "value": "video_player", "params": {"control": "open", "app": "<app_enum>"}},
        "slot_enums": {"app": {"本地视频": "default_video_app", "爱奇艺": "iqiyi_video_app", "腾讯视频": "tencent_video_app", "优酷": "youku_video_app"}},
        "examples": ["播放本地视频", "打开爱奇艺", "打开腾讯视频", "启动优酷"],
    },
    "video_close_app": {
        "label": "app_control:close_video_app",
        "slots": ["app"],
        "target": {"intent": "app_control", "value": "video_player", "params": {"control": "close", "app": "<app_enum>"}},
        "slot_enums": {"app": {"本地视频": "default_video_app", "爱奇艺": "iqiyi_video_app", "腾讯视频": "tencent_video_app", "优酷": "youku_video_app"}},
        "examples": ["关闭本地视频", "关闭爱奇艺", "退出腾讯视频", "关掉优酷"],
    },
    "video_play_content": {
        "label": "app_control:play_video_content",
        "slots": ["content", "content_type"],
        "target": {"intent": "app_control", "value": "video_player", "params": {"control": "play", "content": "<content>", "content_type": "<content_type_enum>"}},
        "slot_enums": {"content_type": {"电影": "movie", "影片": "movie", "综艺": "program", "节目": "program", "电视剧": "drama", "剧集": "drama", "短视频": "short_video", "直播": "live", "未知": "unknown"}},
        "examples": ["播放电影变形金刚", "播放喜剧人单口季节目", "播放新闻直播"],
    },
    "robot_nav": {
        "label": "robot_control:navigate_to_place",
        "slots": ["place"],
        "target": {"intent": "robot_control", "value": "nav", "params": {"place": "<place>"}},
        "examples": ["导航到客厅", "去书房吧", "到厨房去"],
    },
}


SYNONYMS = {
    "调高": ["调大", "提高", "放大", "增加"],
    "调低": ["调小", "降低", "减小", "减少"],
    "打开": ["开启", "启动", "点开"],
    "关闭": ["关掉", "关闭", "停止"],
    "播放": ["放", "播"],
    "暂停": ["停下", "暂停一下"],
    "继续": ["接着", "继续播放"],
    "上一首": ["上一曲", "返回上一首"],
    "下一首": ["下一曲", "切换下一首"],
    "导航": ["去", "到", "前往"],
    "充电": ["去充电", "开始充电"],
}


@dataclass
class TaskProgress:
    task_name: str
    task_type: str
    target_count: int
    generated_count: int = 0
    status: str = "pending"
    last_batch_index: int = 0
    failed_attempts: int = 0


class TaskResult:
    def __init__(self, task_name: str, task_type: str, rows: list[dict], success: bool, message: str = ""):
        self.task_name = task_name
        self.task_type = task_type
        self.rows = rows
        self.success = success
        self.message = message
    
    def save(self, output_dir: Path) -> None:
        task_dir = output_dir / "tasks" / self.task_type
        task_dir.mkdir(parents=True, exist_ok=True)
        task_file = task_dir / f"{self.task_name}.jsonl"
        write_jsonl(task_file, self.rows)
        print(f"  Saved {len(self.rows)} rows to {task_file}")


class IncrementalDataGenerator:
    MAX_RETRIES = 3
    
    def __init__(
        self,
        output_dir: Path,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        sleep_seconds: float,
        batch_size: int,
        config: Optional[dict] = None,
        timeout: int = 0,
    ):
        self.output_dir = output_dir
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.sleep_seconds = sleep_seconds
        self.batch_size = batch_size
        self.config = config or {}
        self.timeout = timeout
        
        self.all_rows: list[dict[str, Any]] = []
        self.seen_keys: set[str] = set()
        self.stats = Counter()
        self.task_results: dict[str, TaskResult] = {}
        self.completed_tasks: set[str] = set()
        
        self._init_openai_client()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_data()
        self._load_checkpoint_state()
    
    def _init_openai_client(self) -> None:
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                print(f"  OpenAI client initialized successfully")
            except Exception as e:
                print(f"  Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            print("  OpenAI library not available, falling back to urllib")
            self.client = None
    
    def _load_existing_data(self) -> None:
        all_file = self.output_dir / "all.jsonl"
        if all_file.exists():
            for line in all_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        key = self._dedupe_key(row)
                        if key not in self.seen_keys:
                            self.seen_keys.add(key)
                            self.all_rows.append(row)
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(self.all_rows)} existing rows")
    
    def _dedupe_key(self, row: dict[str, Any]) -> str:
        return f"{row.get('label', '')}\t{self._normalize_for_dedupe(str(row.get('text', '')))}"
    
    def _normalize_for_dedupe(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[，。！？、,.!?;；：:~～]+", "", text)
        replacements = {"ｑｑ": "qq", "qq音乐": "qq音乐", "爱艺奇": "爱奇艺", "洗手间": "卫生间"}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _build_single_prompt(self, task_name: str, task_type: str, spec: dict, count: int, batch_index: int = 1, batch_count: int = 1) -> str:
        schema = {
            "text": "中文用户原始输入",
            "task_type": task_type,
            "label": spec["label"],
            "slots": {slot: "从 text 中抽取的原文 span；没有则为空字符串" for slot in spec.get("slots", [])},
            "structured_output": "不含 source 的目标结构化输出",
        }
        
        output_example = self._build_output_example(task_type, spec)
        slot_policy = self._build_slot_policy(task_type, spec)
        diversity_policy = self._build_diversity_policy(batch_index, batch_count)
        
        return (
            "你是机器人端侧中文意图识别与指令解析数据生成器。\n"
            "你的输出将直接用于训练 BERT 模型，必须严格、可解析、无重复。\n"
            f"任务名称: {task_name}\n"
            f"任务类型: {task_type}\n"
            f"标签: {spec['label']}\n"
            f"当前批次: {batch_index}/{batch_count}\n"
            f"参考例句: {json.dumps(spec['examples'], ensure_ascii=False)}\n"
            f"目标结构模板: {json.dumps(spec.get('target', {'query_type': 'unknown', 'tasks': []}), ensure_ascii=False)}\n"
            f"枚举映射: {json.dumps(spec.get('slot_enums', {}), ensure_ascii=False)}\n"
            f"需要生成 {count} 条高质量、多样化、口语化中文样本。\n"
            "输出硬性要求:\n"
            "1. 只能输出 JSON Lines，每行一个 JSON 对象。\n"
            "2. 每行只能包含这 5 个顶层字段: text, task_type, label, slots, structured_output。\n"
            f"3. task_type 固定为 {task_type}，label 固定为 {spec['label']}。\n"
            "4. text 必须是单条指令，长度3-100字，不要复合句。\n"
            f"{slot_policy}"
            "去重和多样性要求:\n"
            "1. 禁止直接复用参考例句。\n"
            "2. 避免近重复句。\n"
            f"3. {diversity_policy}\n"
            f"JSON 对象 schema: {json.dumps(schema, ensure_ascii=False)}\n"
            f"格式示例: {json.dumps(output_example, ensure_ascii=False)}\n"
        )
    
    def _build_output_example(self, task_type: str, spec: dict) -> dict:
        text = spec["examples"][0]
        structured_output = parse(text)
        structured_output.pop("source", None)
        return {
            "text": text,
            "task_type": task_type,
            "label": spec["label"],
            "slots": {slot: "" for slot in spec.get("slots", [])},
            "structured_output": structured_output,
        }
    
    def _build_slot_policy(self, task_type: str, spec: dict) -> str:
        if task_type == "multi_intent":
            return "slot 要求: slots 必须为 {}，structured_output.tasks[0] 使用目标结构模板。\n"
        
        lines = ["slot 要求:", "1. slots 中的每个非空值都必须是 text 中连续出现的原文片段。"]
        
        if "app" in spec.get("slots", []):
            lines.append("2. app slot: slots.app 写原文 App 名称；structured_output.tasks[0].params.app 写枚举值。")
        if "content_type" in spec.get("slots", []):
            lines.append("2. content_type slot: slots.content_type 写原文中的类型词，params.content_type 写枚举值。")
        if spec["label"] == "music_control:play_specific_music":
            lines.append("2. singer 和 song 可以二者都有，也可以只有其一。")
        if spec["label"] == "volume_control:set_volume":
            lines.append("2. volume 可以是 0-100 数字，也可以是 最大/最小/最高/最低。")
        
        lines.append("")
        return "\n".join(lines) + "\n"
    
    def _build_diversity_policy(self, batch_index: int, batch_count: int) -> str:
        if batch_count <= 1:
            return "混合生成短口语、礼貌完整表达、带自然噪声词的表达。"
        
        ratio = batch_index / batch_count
        if ratio <= 1 / 4:
            return "本批偏短口语表达，例如省略主语、短句、直接命令。"
        if ratio <= 2 / 4:
            return "本批偏礼貌完整表达，例如使用'请/帮我/麻烦你'等礼貌用语。"
        if ratio <= 3 / 4:
            return "本批加入自然噪声词和倒装说法，例如'现在/那个/先/一下/吧'。"
        return "本批生成多样化句式，包括反问句、条件句、祈使句等不同句型。"
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        if self.client is not None:
            return self._call_llm_openai(prompt)
        else:
            return self._call_llm_urllib(prompt)
    
    def _call_llm_openai(self, prompt: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": "你只输出 JSON Lines，不要解释。"},
            {"role": "user", "content": prompt},
        ]
        
        if self.timeout > 0:
            timeout = self.timeout
        else:
            timeout = min(300, 60 + self.batch_size * 4)
        print(f"  Sending request (timeout={timeout}s, batch_size={self.batch_size})")
        
        for attempt in range(self.MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=timeout,
                    max_tokens=8192,
                )
                
                if not completion.choices:
                    print(f"  Empty response from LLM (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(3.0 * (attempt + 1))
                        continue
                    return None
                
                content = completion.choices[0].message.content
                if content is None or content.strip() == "":
                    print(f"  Empty content from LLM (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(3.0 * (attempt + 1))
                        continue
                    return None
                
                return content.strip()
            
            except APITimeoutError:
                print(f"  Request timeout (attempt {attempt + 1}/{self.MAX_RETRIES})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
        
            except APIConnectionError as e:
                print(f"  Connection error: {e} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
        
            except APIStatusError as e:
                print(f"  API Error {e.status_code}: {e.response.json() if e.response else e} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                
                if e.status_code == 429:
                    delay = 3.0 * (2 ** attempt)
                    print(f"  Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    continue
                elif 500 <= e.status_code < 600:
                    if attempt < self.MAX_RETRIES - 1:
                        print(f"  Server error, retrying...")
                        time.sleep(3.0 * (attempt + 1))
                        continue
                else:
                    print(f"  Client error, no retry")
                    return None
        
            except AuthenticationError as e:
                print(f"  Authentication error: {e}")
                return None
        
            except Exception as e:
                print(f"  Unexpected error: {type(e).__name__}: {e} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
        
        print(f"  Failed after {self.MAX_RETRIES} attempts")
        return None
    
    def _call_llm_urllib(self, prompt: str) -> Optional[str]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "你只输出 JSON Lines，不要解释。"},
                {"role": "user", "content": prompt},
            ],
        }
        
        if self.timeout > 0:
            timeout = self.timeout
        else:
            timeout = min(300, 60 + self.batch_size * 4)
        print(f"  Sending request (timeout={timeout}s, batch_size={self.batch_size})")
        
        request = urllib.request.Request(
            url=self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        
        for attempt in range(self.MAX_RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    if "choices" not in data or not data["choices"]:
                        print(f"  Empty response from LLM (attempt {attempt + 1}/{self.MAX_RETRIES})")
                        if attempt < self.MAX_RETRIES - 1:
                            time.sleep(3.0 * (attempt + 1))
                            continue
                        return None
                    
                    content = data["choices"][0]["message"]["content"]
                    if content is None or content.strip() == "":
                        print(f"  Empty content from LLM (attempt {attempt + 1}/{self.MAX_RETRIES})")
                        if attempt < self.MAX_RETRIES - 1:
                            time.sleep(3.0 * (attempt + 1))
                            continue
                        return None
                    
                    return content.strip()
            
            except urllib.error.HTTPError as e:
                print(f"  HTTP Error {e.code}: {e.reason} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                
                if e.code == 429:
                    delay = 3.0 * (2 ** attempt)
                    print(f"  Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    continue
                elif 500 <= e.code < 600:
                    if attempt < self.MAX_RETRIES - 1:
                        print(f"  Server error, retrying...")
                        time.sleep(3.0 * (attempt + 1))
                        continue
                else:
                    print(f"  Client error (HTTP {e.code}), no retry")
                    return None
            
            except urllib.error.URLError as e:
                print(f"  Network error: {e.reason} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
            
            except TimeoutError:
                print(f"  Request timeout (attempt {attempt + 1}/{self.MAX_RETRIES})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
            
            except json.JSONDecodeError as e:
                print(f"  Invalid JSON response from LLM, no retry")
                return None
            
            except Exception as e:
                print(f"  Unexpected error: {type(e).__name__}: {e} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
        
        print(f"  Failed after {self.MAX_RETRIES} attempts")
        return None
    
    def _parse_llm_response(self, content: str) -> list[dict]:
        rows = []
        if not content:
            return rows
        
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            try:
                row = json.loads(line)
                if self._validate_raw_row(row):
                    rows.append(row)
            except json.JSONDecodeError as e:
                self.stats["json_parse_error"] += 1
        return rows
    
    def _validate_raw_row(self, row: dict) -> bool:
        required_fields = ["text", "task_type", "label", "slots", "structured_output"]
        for field in required_fields:
            if field not in row:
                self.stats["missing_field"] += 1
                return False
        
        text = row.get("text", "")
        if len(text) < 3 or len(text) > 100:
            self.stats["invalid_length"] += 1
            return False
        
        if not isinstance(row.get("slots"), dict):
            self.stats["invalid_slots_type"] += 1
            return False
        
        if not isinstance(row.get("structured_output"), dict):
            self.stats["invalid_structured_output"] += 1
            return False
        
        return True
    
    def _normalize_row(self, row: dict) -> dict:
        text = str(row["text"]).strip()
        slots = row.get("slots") or {}
        
        try:
            structured_output = row.get("structured_output") or parse(text)
            structured_output.pop("source", None)
        except:
            structured_output = {"query_type": "unknown", "tasks": []}
        
        return {
            "text": text,
            "task_type": row["task_type"],
            "label": row["label"],
            "tokens": list(text),
            "bio_tags": self._bio_tags(text, slots),
            "slots": slots,
            "structured_output": structured_output,
        }
    
    def _bio_tags(self, text: str, slots: dict) -> list:
        tags = ["O"] * len(text)
        for slot, value in slots.items():
            if not value:
                continue
            start = text.find(value)
            if start < 0:
                continue
            tags[start] = f"B-{slot}"
            for index in range(start + 1, start + len(value)):
                if index < len(tags):
                    tags[index] = f"I-{slot}"
        return tags
    
    def _validate_row(self, row: dict) -> tuple[bool, str]:
        text = row.get("text", "")
        if not text or len(text) < 3 or len(text) > 100:
            return False, "invalid length"
        
        tokens = list(text)
        bio_tags = row.get("bio_tags", [])
        if len(tokens) != len(bio_tags):
            return False, "bad alignment"
        
        if row.get("task_type") == "multi_intent" and any(tag != "O" for tag in bio_tags):
            return False, "multi_intent has BIO tags"
        
        if row.get("task_type") == "slot_filling":
            if not row.get("slots"):
                return False, "slot_filling missing slots"
        
        return True, ""
    
    def _augment_row(self, row: dict) -> list[dict]:
        aug_config = self.config.get("data_augmentation", {})
        if not aug_config.get("enable", False):
            return [row]
        
        augmented = [row]
        text = row["text"]
        
        if random.random() < aug_config.get("synonym_replacement_prob", 0.15):
            for word, synonyms in SYNONYMS.items():
                if word in text:
                    new_text = text.replace(word, random.choice(synonyms), 1)
                    new_row = dict(row)
                    new_row["text"] = new_text
                    new_row["tokens"] = list(new_text)
                    augmented.append(new_row)
                    break
        
        if random.random() < aug_config.get("random_insertion_prob", 0.1):
            insertions = ["请", "帮我", "麻烦你", "可以", "现在", "先"]
            insertion = random.choice(insertions)
            new_text = insertion + text if random.random() > 0.5 else text + "吧"
            new_row = dict(row)
            new_row["text"] = new_text
            new_row["tokens"] = list(new_text)
            augmented.append(new_row)
        
        return augmented
    
    def _save_task_result(self, task_name: str, task_type: str, rows: list[dict]) -> None:
        task_dir = self.output_dir / "tasks" / task_type
        task_dir.mkdir(parents=True, exist_ok=True)
        
        task_file = task_dir / f"{task_name}.jsonl"
        with task_file.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    def _save_task_progress(self, task_key: str, task_name: str, task_type: str, target_count: int, current_count: int, current_batch: int) -> None:
        """保存单个任务的进度"""
        checkpoint_file = self.output_dir / "checkpoint.json"
        
        # 读取现有 checkpoint
        checkpoint = {}
        if checkpoint_file.exists():
            try:
                checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        
        # 更新任务进度
        if "task_progress" not in checkpoint:
            checkpoint["task_progress"] = {}
        
        checkpoint["task_progress"][task_key] = {
            "task_name": task_name,
            "task_type": task_type,
            "target_count": target_count,
            "current_count": current_count,
            "current_batch": current_batch,
            "success": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        checkpoint["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        checkpoint["stats"] = dict(self.stats)
        
        # 保存 checkpoint
        checkpoint_file.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def _append_task_rows(self, task_name: str, task_type: str, new_rows: list[dict]) -> None:
        """增量追加新行到任务文件"""
        task_dir = self.output_dir / "tasks" / task_type
        task_dir.mkdir(parents=True, exist_ok=True)
        task_file = task_dir / f"{task_name}.jsonl"
        
        with task_file.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    def _save_checkpoint(self) -> None:
        checkpoint_file = self.output_dir / "checkpoint.json"
        checkpoint = {
            "generated_count": len(self.all_rows),
            "stats": dict(self.stats),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_progress": {},
            "completed_tasks": [],
        }
        
        for key, result in self.task_results.items():
            checkpoint["task_progress"][key] = {
                "task_name": result.task_name,
                "task_type": result.task_type,
                "row_count": len(result.rows),
                "success": result.success,
                "message": result.message,
            }
            if result.success:
                checkpoint["completed_tasks"].append(key)
        
        checkpoint_file.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        
        all_file = self.output_dir / "all.jsonl"
        with all_file.open("w", encoding="utf-8") as f:
            for row in self.all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    def _load_checkpoint(self) -> Optional[dict]:
        checkpoint_file = self.output_dir / "checkpoint.json"
        if checkpoint_file.exists():
            return json.loads(checkpoint_file.read_text(encoding="utf-8"))
        return None
    
    def _load_checkpoint_state(self) -> None:
        checkpoint = self._load_checkpoint()
        if not checkpoint:
            print("  No checkpoint found, starting fresh")
            return
        
        if "timestamp" in checkpoint:
            print(f"  Resuming from checkpoint created at: {checkpoint['timestamp']}")
        
        if "stats" in checkpoint:
            self.stats = Counter(checkpoint["stats"])
            print(f"  Loaded stats from checkpoint: {dict(self.stats)}")
        
        task_progress = checkpoint.get("task_progress", checkpoint.get("task_results", {}))
        if task_progress:
            tasks_dir = self.output_dir / "tasks"
            for task_key, task_info in task_progress.items():
                task_name = task_info["task_name"]
                task_type = task_info["task_type"]
                task_file = tasks_dir / task_type / f"{task_name}.jsonl"
                
                if task_file.exists():
                    try:
                        rows = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                        result = TaskResult(
                            task_name=task_name,
                            task_type=task_type,
                            rows=rows,
                            success=task_info.get("success", False),
                            message=task_info.get("message", "")
                        )
                        self.task_results[task_key] = result
                        
                        for row in rows:
                            key = self._dedupe_key(row)
                            if key not in self.seen_keys:
                                self.seen_keys.add(key)
                                self.all_rows.append(row)
                    except Exception as e:
                        print(f"  Error loading task {task_key}: {e}")
            
            print(f"  Restored {len(self.task_results)} task results from checkpoint")
        
        if "completed_tasks" in checkpoint:
            self.completed_tasks = set(checkpoint["completed_tasks"])
            print(f"  Found {len(self.completed_tasks)} completed tasks")
        
        if "generated_count" in checkpoint:
            print(f"  Total rows restored: {len(self.all_rows)}")
    
    def _get_task_progress(self, task_name: str, task_type: str) -> Optional[dict]:
        """获取任务进度"""
        checkpoint = self._load_checkpoint()
        if not checkpoint:
            return None
        
        task_key = f"{task_type}_{task_name}"
        task_progress = checkpoint.get("task_progress", {}).get(task_key)
        if not task_progress:
            return None
        
        # 检查任务文件是否存在
        task_file = self.output_dir / "tasks" / task_type / f"{task_name}.jsonl"
        if not task_file.exists():
            return None
        
        # 统计实际行数
        try:
            actual_rows = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            task_progress["current_count"] = len(actual_rows)
        except Exception:
            task_progress["current_count"] = 0
        
        return task_progress
    
    def _is_task_completed(self, task_name: str, task_type: str, target_count: int) -> bool:
        task_file = self.output_dir / "tasks" / task_type / f"{task_name}.jsonl"
        if task_file.exists():
            lines = task_file.read_text(encoding="utf-8").splitlines()
            if len(lines) >= target_count:
                return True
        return False
    
    def process_single_task(
        self,
        task_name: str,
        task_type: str,
        spec: dict,
        target_count: int,
    ) -> TaskResult:
        task_key = f"{task_type}_{task_name}"
        
        # 检查是否已完成
        if self._is_task_completed(task_name, task_type, target_count):
            print(f"  [SKIP] {task_name} already completed ({target_count} rows)")
            task_file = self.output_dir / "tasks" / task_type / f"{task_name}.jsonl"
            rows = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            result = TaskResult(task_name, task_type, rows, success=True, message="Already completed")
            self.task_results[task_key] = result
            return result
        
        # 检查是否有未完成的进度
        task_progress = self._get_task_progress(task_name, task_type)
        if task_progress and not task_progress.get("success", False):
            saved_count = task_progress.get("current_count", 0)
            if saved_count > 0:
                print(f"  Resuming {task_name}: {saved_count}/{target_count} rows already saved")
                
                # 加载已保存的行
                task_file = self.output_dir / "tasks" / task_type / f"{task_name}.jsonl"
                saved_rows = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                
                # 去重
                for row in saved_rows:
                    key = self._dedupe_key(row)
                    if key not in self.seen_keys:
                        self.seen_keys.add(key)
                
                accepted = saved_count
                task_rows = saved_rows
            else:
                accepted = 0
                task_rows = []
        else:
            accepted = 0
            task_rows = []
        
        batch_count = max(1, (target_count + self.batch_size - 1) // self.batch_size)
        failed_batches = 0
        max_failed_batches = 3
        
        print(f"  Processing {task_name}: target={target_count}, batches={batch_count}")
        
        for batch_idx in range(1, batch_count + 1):
            count = min(self.batch_size, target_count - accepted)
            if count <= 0:
                break
            
            prompt = self._build_single_prompt(task_name, task_type, spec, count, batch_idx, batch_count)
            content = self._call_llm(prompt)
            
            if content is None:
                print(f"    Batch {batch_idx} failed - no response")
                failed_batches += 1
                
                if failed_batches >= max_failed_batches:
                    print(f"  {max_failed_batches} consecutive batches failed for {task_name}")
                    print(f"  Current progress: {accepted}/{target_count} rows")
                    result = TaskResult(task_name, task_type, task_rows, success=False, message=f"Failed after {max_failed_batches} failed batches")
                    self.task_results[task_key] = result
                    self._save_checkpoint()
                    return result
                
                continue
            
            raw_rows = self._parse_llm_response(content)
            if not raw_rows:
                print(f"    Batch {batch_idx}: no valid JSONL rows")
                failed_batches += 1
                if failed_batches >= max_failed_batches:
                    print(f"  {max_failed_batches} batches with no valid output, stopping")
                    result = TaskResult(task_name, task_type, task_rows, success=False, message=f"No valid output after {max_failed_batches} batches")
                    self.task_results[task_key] = result
                    self._save_checkpoint()
                    return result
                continue
            
            failed_batches = 0
            
            # 处理本批次数据
            new_rows = []
            for raw_row in raw_rows:
                try:
                    row = self._normalize_row(raw_row)
                    ok, reason = self._validate_row(row)
                    if not ok:
                        self.stats["invalid"] += 1
                        self.stats[f"invalid_{reason}"] = self.stats.get(f"invalid_{reason}", 0) + 1
                        continue
                    
                    key = self._dedupe_key(row)
                    if key in self.seen_keys:
                        self.stats["duplicate"] += 1
                        continue
                    
                    self.seen_keys.add(key)
                    
                    augmented_rows = self._augment_row(row)
                    for aug_row in augmented_rows:
                        aug_key = self._dedupe_key(aug_row)
                        if aug_key not in self.seen_keys:
                            self.seen_keys.add(aug_key)
                            new_rows.append(aug_row)
                            task_rows.append(aug_row)
                            if aug_row is not row:
                                self.stats["augmented"] += 1
                    
                    new_rows.append(row)
                    task_rows.append(row)
                    accepted += 1
                    self.stats["accepted"] += 1
                    
                    if accepted >= target_count:
                        break
                except Exception as e:
                    self.stats["error"] += 1
                    continue
            
            # 立即保存本批次
            if new_rows:
                self._append_task_rows(task_name, task_type, new_rows)
                self._save_task_progress(task_key, task_name, task_type, target_count, accepted, batch_idx)
                print(f"    Batch {batch_idx}: saved {len(new_rows)} new rows, total={accepted}/{target_count}")
            
            if accepted >= target_count:
                break
            
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)
        
        self.all_rows.extend(task_rows)
        
        success = accepted >= target_count
        message = f"Generated {accepted}/{target_count} rows" if success else f"Partial: {accepted}/{target_count} rows"
        result = TaskResult(task_name, task_type, task_rows, success=success, message=message)
        self.task_results[task_key] = result
        self._save_checkpoint()
        
        print(f"  Result: {accepted} rows generated")
        return result
    
    def generate_all(self) -> None:
        minimums = self.config.get("minimums", {})
        per_intent = minimums.get("default_per_intent", 2000)
        per_slot = minimums.get("default_per_slot_filling", 3000)
        
        multi_intent_tasks = list(MULTI_INTENT_TASKS.items())
        slot_tasks = list(SLOT_TASKS.items())
        
        total_tasks = len(multi_intent_tasks) + len(slot_tasks)
        current_task = 0
        completed_tasks = 0
        failed_tasks = 0
        
        print(f"\n{'='*60}")
        print(f"Starting data generation for {total_tasks} tasks")
        print(f"{'='*60}")
        
        for task_name, spec in multi_intent_tasks:
            current_task += 1
            target = minimums.get(task_name, per_intent)
            print(f"\n[{current_task}/{total_tasks}] Processing multi_intent: {task_name} (target={target})")
            
            try:
                result = self.process_single_task(task_name, "multi_intent", spec, target)
                if result.success:
                    completed_tasks += 1
                else:
                    failed_tasks += 1
            except Exception as e:
                print(f"  Error processing {task_name}: {e}")
                failed_tasks += 1
                continue
        
        for task_name, spec in slot_tasks:
            current_task += 1
            target = minimums.get(task_name, per_slot)
            print(f"\n[{current_task}/{total_tasks}] Processing slot_filling: {task_name} (target={target})")
            
            try:
                result = self.process_single_task(task_name, "slot_filling", spec, target)
                if result.success:
                    completed_tasks += 1
                else:
                    failed_tasks += 1
            except Exception as e:
                print(f"  Error processing {task_name}: {e}")
                failed_tasks += 1
                continue
        
        print(f"\n{'='*60}")
        print(f"Generation Summary:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Completed: {completed_tasks}")
        print(f"  Failed: {failed_tasks}")
        print(f"  Total rows: {len(self.all_rows)}")
        print(f"  Stats: {dict(self.stats)}")
        print(f"{'='*60}")
    
    def merge_and_split(self) -> None:
        print("\nMerging task results and splitting...")
        
        all_rows = []
        tasks_dir = self.output_dir / "tasks"
        
        if tasks_dir.exists():
            for task_type_dir in tasks_dir.iterdir():
                if not task_type_dir.is_dir():
                    continue
                for task_file in task_type_dir.glob("*.jsonl"):
                    try:
                        for line in task_file.read_text(encoding="utf-8").splitlines():
                            if line.strip():
                                row = json.loads(line)
                                all_rows.append(row)
                    except Exception as e:
                        print(f"  Error loading {task_file}: {e}")
        
        print(f"  Merged {len(all_rows)} rows from task files")
        
        self.all_rows = all_rows
        self._split_and_write()
    
    def _split_and_write(self) -> None:
        split_config = self.config.get("split", {})
        train_ratio = split_config.get("train", 0.8)
        val_ratio = split_config.get("validation", 0.1)
        
        train_cutoff = int(train_ratio * 10_000)
        val_cutoff = int((train_ratio + val_ratio) * 10_000)
        
        splits = {"train": [], "validation": [], "test": []}
        
        for row in self.all_rows:
            bucket = int(hashlib.sha1(row["text"].encode("utf-8")).hexdigest()[:8], 16) % 10_000
            if bucket < train_cutoff:
                splits["train"].append(row)
            elif bucket < val_cutoff:
                splits["validation"].append(row)
            else:
                splits["test"].append(row)
        
        multi_rows = [r for r in self.all_rows if r["task_type"] == "multi_intent"]
        slot_rows = [r for r in self.all_rows if r["task_type"] == "slot_filling"]
        
        write_jsonl(self.output_dir / "intent_classification.jsonl", self.all_rows)
        write_jsonl(self.output_dir / "multi_intent.jsonl", multi_rows)
        write_jsonl(self.output_dir / "slot_filling.jsonl", slot_rows)
        
        for split_name, split_rows in splits.items():
            split_dir = self.output_dir / "splits" / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            write_jsonl(split_dir / "intent_classification.jsonl", split_rows)
            write_jsonl(split_dir / "multi_intent.jsonl", [r for r in split_rows if r["task_type"] == "multi_intent"])
            write_jsonl(split_dir / "slot_filling.jsonl", [r for r in split_rows if r["task_type"] == "slot_filling"])
        
        label_counts = Counter(row["label"] for row in self.all_rows)
        print(f"\nClass distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(config_path: Optional[str] = None) -> dict:
    default_path = Path(__file__).parent.parent.parent / "configs" / "data_generation.json"
    path = Path(config_path) if config_path else default_path
    
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def validate_config() -> tuple[bool, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("BERT_TASKS_LLM_MODEL")

    if not api_key:
        return False, "OPENAI_API_KEY environment variable is not set"
    if not base_url:
        return False, "OPENAI_BASE_URL environment variable is not set"
    if not model:
        return False, "BERT_TASKS_LLM_MODEL environment variable is not set"

    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM-synthesized training data (incremental mode).")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--model", default=os.getenv("BERT_TASKS_LLM_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=0, help="Request timeout in seconds (0=auto based on batch size)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--skip-config-check", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing task files without generating")
    args = parser.parse_args()

    config = load_config(args.config)
    
    model = args.model or config.get("model") or os.getenv("BERT_TASKS_LLM_MODEL", "gpt-4.1-mini")
    base_url = args.base_url or config.get("base_url") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    temperature = args.temperature if args.temperature != 0.7 else config.get("temperature", 0.7)
    sleep_seconds = args.sleep_seconds if args.sleep_seconds != 2.0 else config.get("sleep_seconds", 2.0)
    batch_size = args.batch_size if args.batch_size != 50 else config.get("batch_size", 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timeout = args.timeout if args.timeout != 0 else config.get("timeout", 0)
    
    print(f"{'='*60}")
    print(f"Data Synthesis Configuration:")
    print(f"  Model: {model}")
    print(f"  Base URL: {base_url}")
    print(f"  Temperature: {temperature}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Request Timeout: {'auto' if timeout == 0 else f'{timeout}s'}")
    print(f"  Sleep Between Batches: {sleep_seconds}s")
    print(f"  Output Directory: {output_dir}")
    print(f"  Merge Only: {args.merge_only}")
    print(f"{'='*60}")

    if not args.merge_only:
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable is not set")
            return
        
        if not args.skip_config_check:
            ok, message = validate_config()
            if not ok:
                print(f"Configuration error: {message}")
                return

    generator = IncrementalDataGenerator(
        output_dir=output_dir,
        model=model,
        api_key=api_key or "",
        base_url=base_url,
        temperature=temperature,
        sleep_seconds=sleep_seconds,
        batch_size=batch_size,
        config=config,
        timeout=timeout,
    )
    
    try:
        if args.merge_only:
            generator.merge_and_split()
        else:
            generator.generate_all()
            generator.merge_and_split()
            
            print(f"\nSuccessfully wrote {len(generator.all_rows)} rows to {output_dir}")
    
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        print("Progress has been saved.")
        generator._save_checkpoint()


if __name__ == "__main__":
    main()