# BertTasks 机器人端侧意图识别与指令解析

BertTasks 面向机器人端侧中文指令理解场景，目标是在本地快速完成高置信的意图识别、指令结构化和必要的槽位抽取。整体策略是：

1. **规则模板优先**：对明确、稳定、风险低的表达直接用规则解析，保证速度和确定性。
2. **BERT 模型服务兜底**：规则未命中时，调用领域内微调后的 BERT 意图分类模型和 slot 抽取模型。
3. **保守拒识**：规则冲突、模型低置信、多任务切分不确定时返回 `unknown`，避免误判。

当前覆盖 `intent_and_instruct.txt` 中的音量、音乐、视频、投影仪、机器人导航/充电、助手控制等任务。

## 工程结构

```text
configs/
  intents.json              # 意图、枚举、训练任务划分、slot 策略
  data_generation.json      # 大模型合成数据配置
  test_generation.json      # 测试用数据合成配置
src/bert_tasks/
  parser.py                 # 规则模板解析与多任务切分
  rules.py                  # 各类意图的规则实现
  runtime.py                # 规则优先 + BERT 兜底运行时
  model.py                  # BERT 分类模型和 slot 模型适配层
  data_synthesis.py         # 大模型数据合成、校验、切分
  service.py                # FastAPI 服务入口
scripts/
  generate_data.py          # 生成 LLM prompt 或调用 LLM 产出数据
  train.py                  # BERT 分类/slot 训练入口骨架
  evaluate.py               # 数据与规则解析评估
  run_tests.py              # 零依赖测试运行器
tests/
  fixtures/                 # 小型训练 smoke fixture
  test_parser.py            # 规则与结构化输出测试
  test_data_synthesis.py    # LLM 数据管线测试
```

## 处理流程

```text
用户输入
  |
  v
文本归一化
  |
  v
规则模板解析
  |-- 命中且无冲突 --> 输出结构化 JSON，source=rule_template
  |
  v
BERT 指令分类模型
  |-- 低置信 --> unknown，source=unknown
  |
  v
分类标签属于 slot-required 任务时进入 BERT token classification
  |-- slot 低置信或 schema 校验失败 --> unknown
  |
  v
结构化组装器输出 JSON
  |-- BERT 兜底成功 --> source=bert_model
```

多任务输入会先做保守切分，例如"到客厅打开投影仪"会拆成"到客厅"和"打开投影仪"。无法稳定切分的复杂输入返回 `unknown`，交给后端大模型处理。

## 任务划分

### 第一阶段：指令分类

第一阶段分类器需要覆盖所有原子任务标签，包括无需 slot 的固定动作类，也包括后续需要 slot 抽取的任务类。这样运行时可以先判断"是不是 slot-required 任务"，再决定是否进入 slot tagger。

无需 slot 的固定动作类，例如：

- 音量调高、调低、静音
- 音乐上一首、下一首、暂停、停止、继续
- 打开/关闭投影仪
- 取消导航、回充、停止充电
- 助手休眠、聊天模式
- unknown 负样本

需要 slot 的分类标签也要进入第一阶段分类训练，例如：

- `volume_control:set_volume`
- `music_control:play_specific_music`
- `music_control:open_app`
- `music_control:close_app`
- `app_control:open_video_app`
- `app_control:close_video_app`
- `app_control:play_video_content`
- `robot_control:navigate_to_place`

第一阶段使用 `intent_classification.jsonl` 训练 BERT sequence classification。

### 第二阶段：Slot 抽取

只有第一阶段分类结果属于 slot-required 标签时，才进入第二阶段 slot 抽取，例如：

- `volume_control:set_volume`：抽取 `volume`
- `music_control:play_specific_music`：抽取 `singer`、`song`
- `music_control:open_app`：抽取音乐类 `app`，如音乐播放器、QQ 音乐、网易云音乐、酷狗音乐
- `music_control:close_app`：抽取音乐类 `app`
- `app_control:open_video_app`：抽取视频类 `app`，如本地视频、爱奇艺、腾讯视频、优酷
- `app_control:close_video_app`：抽取视频类 `app`
- `app_control:play_video_content`：抽取 `content`、`content_type`
- `robot_control:navigate_to_place`：抽取 `place`

第二阶段使用 `slot_filling.jsonl` 训练 BERT token classification，标注格式为字符级 BIO。

## 大模型合成数据

训练数据不使用随机模板生成。`scripts/generate_data.py` 会先为每个任务生成 LLM prompt batch，让大模型产生自然、多样、口语化的中文样本，然后进行：

- JSONL 解析
- slot span 校验
- BIO 标签生成
- schema 校验
- 去重
- 训练/验证/测试集切分

注意：合成数据中的 `structured_output` 不包含 `source` 字段；`source` 只在运行时解析意图和指令时输出，用于标记结果来自规则模板、BERT 模型或拒识。

### 数据合成特性

- **任务级隔离**：每个意图任务独立生成，失败不影响其他任务
- **增量保存**：每个批次生成后立即保存，避免数据丢失
- **断点续跑**：自动从 checkpoint 恢复进度，支持任务中断后继续
- **重试机制**：最多 3 次重试，避免 token 无效消耗
- **数据验证**：验证必要字段、文本长度、格式等

### 环境配置

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export BERT_TASKS_LLM_MODEL=deepseek-v4-pro
```

### 数据合成命令

**完整数据生成流程**：

```bash
PYTHONPATH=src python scripts/generate_data.py \
  --output-dir data/generated \
  --config configs/data_generation.json \
  --batch-size 50 \
  --temperature 0.7
```

**使用自定义配置生成少量测试数据**：

```bash
PYTHONPATH=src python scripts/generate_data.py \
  --output-dir data/test_run \
  --config configs/test_generation.json
```

**断点续跑（自动跳过已完成任务）**：

```bash
PYTHONPATH=src python scripts/generate_data.py \
  --output-dir data/generated \
  --resume
```

**仅合并已生成的任务数据（不调用 LLM）**：

```bash
PYTHONPATH=src python scripts/generate_data.py \
  --output-dir data/generated \
  --merge-only
```

### 数据合成配置

配置文件 `configs/data_generation.json` 示例：

```json
{
  "minimums": {
    "default_per_intent": 2000,
    "default_per_slot_filling": 3000,
    "volume_down": 3000,
    "volume_up": 3000
  },
  "split": {
    "train": 0.8,
    "validation": 0.1
  },
  "data_augmentation": {
    "enable": false
  },
  "model": "deepseek-v4-pro",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "temperature": 0.7,
  "batch_size": 50,
  "sleep_seconds": 2.0,
  "timeout": 0
}
```

**配置说明**：
- `timeout`: API 请求超时时间（秒），0 表示自动根据 batch_size 计算（推荐）
- `batch_size`: 每批请求的样本数，较大的 batch_size 需要更长的超时时间

### 输出目录结构

```text
data/generated/
├── tasks/
│   ├── multi_intent/        # 各意图任务独立存储
│   │   ├── volume_down.jsonl
│   │   ├── volume_up.jsonl
│   │   └── ...
│   └── slot_filling/        # 各槽位任务独立存储
│       ├── volume_set.jsonl
│       └── ...
├── checkpoint.json          # 断点续跑状态
├── all.jsonl                # 合并后全部数据
├── intent_classification.jsonl
├── multi_intent.jsonl
├── slot_filling.jsonl
└── splits/
    ├── train/
    │   ├── intent_classification.jsonl
    │   ├── multi_intent.jsonl
    │   └── slot_filling.jsonl
    ├── validation/
    └── test/
```

## BERT 训练与验证

### 训练命令

**基础训练**：

```bash
PYTHONPATH=src python scripts/train.py \
  --intent-file data/generated/splits/train/intent_classification.jsonl \
  --slot-file data/generated/splits/train/slot_filling.jsonl \
  --validation-intent-file data/generated/splits/validation/intent_classification.jsonl \
  --validation-slot-file data/generated/splits/validation/slot_filling.jsonl \
  --model-name /path/to/bert-base-chinese \
  --output-dir models/bert_tasks \
  --epochs 3 \
  --batch-size 16 \
  --max-length 128
```

**完整训练（带评估）**：

```bash
PYTHONPATH=src python scripts/train.py \
  --intent-file data/generated/splits/train/intent_classification.jsonl \
  --slot-file data/generated/splits/train/slot_filling.jsonl \
  --validation-intent-file data/generated/splits/validation/intent_classification.jsonl \
  --validation-slot-file data/generated/splits/validation/slot_filling.jsonl \
  --test-intent-file data/generated/splits/test/intent_classification.jsonl \
  --test-slot-file data/generated/splits/test/slot_filling.jsonl \
  --model-name /path/to/bert-base-chinese \
  --output-dir models/bert_tasks \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --seed 42 \
  --device auto \
  --evaluate
```

**Smoke 测试（验证数据加载）**：

```bash
PYTHONPATH=src python scripts/train.py --smoke
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--intent-file` | - | 意图分类训练数据路径 |
| `--slot-file` | - | 槽位填充训练数据路径 |
| `--validation-intent-file` | - | 意图分类验证数据路径 |
| `--validation-slot-file` | - | 槽位填充验证数据路径 |
| `--test-intent-file` | - | 意图分类测试数据路径 |
| `--test-slot-file` | - | 槽位填充测试数据路径 |
| `--model-name` | bert-base-chinese | 预训练模型名称或路径 |
| `--output-dir` | models/bert_tasks | 模型输出目录 |
| `--epochs` | 3.0 | 训练轮数 |
| `--batch-size` | 16 | 批大小 |
| `--max-length` | 128 | 最大序列长度 |
| `--learning-rate` | 2e-5 | 学习率 |
| `--weight-decay` | 0.01 | 权重衰减 |
| `--warmup-ratio` | 0.1 | 学习率预热比例 |
| `--early-stopping-patience` | 3 | 早停耐心值 |
| `--seed` | 42 | 随机种子 |
| `--device` | auto | 训练设备 (auto/cpu/cuda) |
| `--smoke` | - | 仅验证数据加载 |
| `--evaluate` | - | 训练后评估测试集 |

### 评估命令

**评估规则解析准确率**：

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --file data/generated/splits/test/intent_classification.jsonl
```

**评估 BERT 模型准确率**：

```bash
PYTHONPATH=src python scripts/evaluate_model.py \
  --classifier-dir models/bert_tasks/intent_classifier \
  --slot-tagger-dir models/bert_tasks/slot_tagger
```

### 模型评测结果

| 模型 | 指标 | 值 |
|------|------|------|
| 意图分类器 | 准确率 | **98%** |
| 意图分类器 | 宏平均 F1 | **97%** |
| 意图分类器 | 加权平均 F1 | **98%** |
| 意图分类器 | 测试样本数 | 5,028 |
| 意图分类器 | 意图类别数 | 23 |
| 槽位标签器 | 准确率 | **98%** |
| 槽位标签器 | 宏平均 F1 | **97%** |
| 槽位标签器 | 加权平均 F1 | **98%** |
| 槽位标签器 | BIO 标签数 | 16 |

## 服务运行

安装服务依赖后启动：

```bash
pip install -e '.[service]'
PYTHONPATH=src python scripts/serve.py
```

请求：

```bash
curl -X POST http://localhost:8000/parse \
  -H 'Content-Type: application/json' \
  -d '{"text":"播放蔡琴的渡口"}'
```

如果部署了 BERT 模型服务目录，可通过环境变量启用兜底模型：

```bash
export BERT_TASKS_CLASSIFIER_DIR=models/intent_classifier
export BERT_TASKS_SLOT_TAGGER_DIR=models/slot_tagger
export BERT_TASKS_INTENT_THRESHOLD=0.9
export BERT_TASKS_SLOT_THRESHOLD=0.9
```

没有配置模型目录时，服务只使用规则模板；规则未命中返回 `unknown`。

## 结果来源标记

所有解析结果都会带 `source` 字段，方便区分来源并统计规则覆盖率和模型兜底率：

- `rule_template`：规则模板解析命中。
- `bert_model`：规则未命中，由 BERT 分类/slot 模型兜底成功。
- `unknown`：规则和模型都未能高置信解析，或输入被拒识。

## 输出格式

单任务：

```json
{
  "query_type": "single_task",
  "source": "rule_template",
  "tasks": [
    {
      "user_input": "声音调整到70%",
      "intent": "volume_control",
      "value": "speaker",
      "params": {
        "volume": "70"
      }
    }
  ]
}
```

多任务：

```json
{
  "query_type": "multi_task",
  "source": "rule_template",
  "tasks": [
    {
      "user_input": "到客厅",
      "intent": "robot_control",
      "value": "nav",
      "params": {
        "place": "客厅"
      }
    },
    {
      "user_input": "打开投影仪",
      "intent": "projector_control",
      "value": "projector",
      "params": {
        "control": "open"
      }
    }
  ]
}
```

拒识：

```json
{
  "query_type": "unknown",
  "source": "unknown",
  "tasks": []
}
```

## 本地验证

安装依赖：

```bash
# 仅安装规则解析运行时
pip install -r requirements-runtime.txt

# 安装服务依赖
pip install -r requirements-service.txt

# 安装 BERT 训练/推理依赖
pip install -r requirements-model.txt

# 安装完整依赖
pip install -r requirements.txt
```

```bash
PYTHONPATH=src python3 scripts/run_tests.py
PYTHONPATH=src python3 -m bert_tasks.cli "到客厅打开投影仪"
PYTHONPATH=src python3 scripts/generate_data.py --dry-run --output-dir /tmp/bert_tasks_prompts --per-task 3 --batch-size 2
PYTHONPATH=src python3 scripts/train.py \
  --intent-file tests/fixtures/intent_classification.jsonl \
  --slot-file tests/fixtures/slot_filling.jsonl \
  --smoke
```

## 完整工作流示例

### 1. 配置环境变量

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export BERT_TASKS_LLM_MODEL=deepseek-v4-pro
```

### 2. 生成训练数据

```bash
# 使用测试配置（小批量）
PYTHONPATH=src python scripts/generate_data.py \
  --output-dir data/test_run \
  --config configs/test_generation.json

# 使用正式配置
PYTHONPATH=src python scripts/generate_data.py \
  --output-dir data/generated \
  --config configs/data_generation.json
```

### 3. 训练模型

```bash
PYTHONPATH=src python scripts/train.py \
  --intent-file data/generated/splits/train/intent_classification.jsonl \
  --slot-file data/generated/splits/train/slot_filling.jsonl \
  --validation-intent-file data/generated/splits/validation/intent_classification.jsonl \
  --validation-slot-file data/generated/splits/validation/slot_filling.jsonl \
  --model-name /path/to/bert-base-chinese \
  --output-dir models/bert_tasks \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --evaluate
```

### 4. 评估模型

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --file data/generated/splits/test/intent_classification.jsonl
```

### 5. 启动服务

```bash
export BERT_TASKS_CLASSIFIER_DIR=models/bert_tasks/intent_classifier
export BERT_TASKS_SLOT_TAGGER_DIR=models/bert_tasks/slot_tagger
PYTHONPATH=src python scripts/serve.py
```