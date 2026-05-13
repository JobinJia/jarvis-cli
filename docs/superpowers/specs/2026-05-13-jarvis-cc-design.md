# jarvis-cc 设计文档

**日期**: 2026-05-13
**作者**: Jobin (with Claude)
**状态**: Draft, 待实现

---

## 1. 目标

为 Claude Code CLI 加一层「贾维斯风格」语音提醒：当 Claude Code 出现需要人类决策的事件时（权限请求 / 闲置等待 / MCP 表单），用接近贾维斯（钢铁侠 AI 管家 J.A.R.V.I.S.）的英式管家口吻、中英双语智能切换地播报一句简短提示。

用户场景：去倒水、走开一会儿、不盯屏幕时不会错过 Claude Code 的关键决策点。

## 2. 非目标

- 不替代 Claude Code 原有的视觉权限弹窗，只是「加一层听觉提醒」。
- 不做语音输入（不是 STT），只做单向 TTS 通知。
- 不在 MVP 内做声音克隆训练（仅 zero-shot），微调留作后续可选项。
- 不发布到 PyPI（结构按可发布组织，但不真发）。
- 不做跨平台 (Linux/Windows)，仅 macOS Apple Silicon。

## 3. 核心约束

| 项 | 约束 |
|---|---|
| Runtime | Python 3.11+ + uv |
| 平台 | macOS 13+, Apple Silicon (M1/M2/M3) |
| 内存 | daemon 闲时 1-2GB (XTTS 模型常驻) |
| 暖启动延迟 | hook 触发到第一个字 ≤ 3s |
| Hook 自身阻塞 | < 10ms (异步 fire-and-forget) |
| 触发事件 | Notification hook 的 `permission_prompt` / `idle_prompt` / `elicitation_dialog`（实现时需要对照 Claude Code 官方 hook 文档核对 payload 字段名，做宽容解析） |
| 触发范围 | 全局 (`~/.claude/settings.json`) |
| LLM 默认 | DeepSeek-Chat，兑底 Ollama Qwen2.5 7B |
| TTS 默认 | XTTS-v2 本地推理 + zero-shot voice cloning |
| 语音语言 | 中英双语自动切换（基于 cwd 项目语言） |
| 并发 | 串行队列 + 10s 同类去重 |

## 4. 整体架构

```
Claude Code ──触发── Notification hook
                       │
                       ▼ (stdin JSON, <10ms)
                  hook_client (一次性进程)
                       │
                       ▼ (unix socket: ~/.jarvis-cc/jarvis.sock)
                  jarvis-daemon (launchd 托管)
                       │
              ┌────────┼────────┬─────────────┐
              ▼        ▼        ▼             ▼
          listener  dedup    queue        health (HTTP)
              │        │        │
              └────────┴────────┘
                       │
                       ▼
                phrase.router ──► LLM provider chain
                       │            (deepseek → ollama)
                       ▼
                tts.engine ──────► TTS provider chain
                       │            (xtts → elevenlabs → say)
                       ▼
                    player (afplay)
                       │
                       ▼
                    扬声器
```

**为什么选 daemon 模型而非每次新进程**：XTTS-v2 模型加载需要 5-10s。daemon 常驻可把暖启动延迟压到 1-2s，符合「贾维斯感」的体验诉求。launchd 托管解决启停、崩溃重启、登录自启。

## 5. 模块拆分

项目结构（src layout，PyPI-ready）：

```
jarvis-cc/
├── pyproject.toml
├── README.md
├── docs/
│   └── superpowers/specs/2026-05-13-jarvis-cc-design.md  (本文档)
├── src/jarvis_cc/
│   ├── __init__.py
│   ├── __main__.py             # python -m jarvis_cc <command>
│   ├── hook_client.py          # 入口 A：Claude Code 调用的瘦客户端
│   ├── daemon/
│   │   ├── __init__.py
│   │   ├── main.py             # 入口 B：daemon 主程序 (asyncio)
│   │   ├── listener.py         # unix socket 监听 → 事件队列
│   │   ├── dedup.py            # 10s 滑动窗口去重
│   │   ├── queue.py            # 单 worker 串行消费
│   │   └── health.py           # HTTP /health (127.0.0.1:9527)
│   ├── phrase/
│   │   ├── __init__.py
│   │   ├── router.py           # LLM provider 路由 + fallback
│   │   ├── prompt.py           # 贾维斯口吻 system prompt + few-shot
│   │   ├── language.py         # 项目语言检测
│   │   ├── templates.py        # LLM 全失败时的兜底模板
│   │   └── providers/
│   │       ├── base.py         # ABC: PhraseProvider
│   │       ├── deepseek.py
│   │       ├── anthropic.py
│   │       ├── openai.py
│   │       └── ollama.py
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── engine.py           # provider 路由
│   │   ├── voice_clone.py      # 参考音频管理
│   │   └── providers/
│   │       ├── base.py         # ABC: TTSProvider
│   │       ├── xtts.py
│   │       ├── elevenlabs.py
│   │       └── say.py          # macOS `say` 兜底
│   ├── player.py               # afplay 包装
│   ├── config.py               # ~/.jarvis-cc/config.toml 加载
│   └── install.py              # CLI: install/uninstall/status/test
├── tests/
│   ├── unit/
│   ├── integration/
│   └── bench/
└── scripts/
    └── com.jobin.jarvis-cc.plist  # launchd 模板
```

**入口注册** (pyproject.toml console_scripts)：

```
jarvis-cc-hook   = jarvis_cc.hook_client:main
jarvis-cc-daemon = jarvis_cc.daemon.main:main
jarvis-cc        = jarvis_cc.install:main
```

`__main__.py` 暴露 `python -m jarvis_cc <subcommand>`，内部转发到对应的入口函数；目的是让用户即便没把 uv 的 bin 加入 PATH，也能用 `python -m jarvis_cc test` 走通流程。

**模块契约**（每个都能独立测试）：

| 模块 | 输入 | 输出 | 副作用 |
|---|---|---|---|
| `hook_client` | stdin JSON | socket 写一行 NDJSON | 无 |
| `daemon.listener` | socket bytes | `Event` dataclass | 入队 |
| `daemon.dedup` | `Event` | `bool`（skip?） | 维护 in-memory 滑动窗口 |
| `daemon.queue` | `Event` | None | 调用下游 phrase + tts + player |
| `phrase.router` | `Event` + `lang` | `str`（贾维斯句） | LLM API call |
| `phrase.language` | `cwd` | `"zh"` \| `"en"` | 读 CLAUDE.md 首 500 字 |
| `tts.engine` | `text` + `lang` + `ref_audio` | `bytes` (wav) | 模型推理 |
| `player` | `bytes` (wav) | None | spawn `afplay` |

## 6. 数据流（典型一次调用）

```
T+0ms     Claude Code 触发 Notification hook
T+5ms     hook_client 读 stdin payload (JSON):
            {
              "session_id": "abc",
              "notification_type": "permission_prompt",
              "tool_name": "Bash",
              "tool_input": {"command": "rm foo.ts"},
              "cwd": "/Users/jiabinbin/myself/blog-valaxy-shuimo"
            }
T+8ms     写入 ~/.jarvis-cc/jarvis.sock 一行 NDJSON
T+10ms    hook_client 退出 ◀ Claude Code 解除阻塞
─────────── 以下在 daemon 内异步发生 ───────────
T+12ms    listener 收到事件，dataclass 化为 Event
T+13ms    dedup.is_duplicate(event)? 10s 内 (cwd, type, tool_name) 哈希存在则跳过
T+15ms    queue.put_nowait(event)
T+20ms    worker.pull → phrase.language.detect_for(cwd) → "zh"
T+30ms    phrase.router.generate(event, lang="zh")
            → 调用 deepseek-chat，prompt:
              system: "你是 Tony Stark 的 AI 管家 J.A.R.V.I.S.。
                       用一句话（< 15 字）用 {lang} 向 Sir 通报以下事件，
                       礼貌、沉稳、略带英式幽默感。不要解释为什么。"
              user: <Event JSON>
T+500ms   LLM 返回："先生，Claude 请求执行删除文件操作。"
T+520ms   tts.engine.synthesize(text, lang="zh",
                                ref_audio=~/.jarvis-cc/voices/jarvis_zh.wav)
T+1800ms  返回 wav bytes
T+1810ms  player.play(wav) → spawn `afplay`
T+1810ms~4000ms  用户听到："先生，Claude 请求执行删除文件操作。"
T+4100ms  worker 拉下一条事件（若有）
```

**总延迟**（暖启动）：触发 → 第一个字 ≈ **1.8 秒**。

## 7. 错误处理与降级

| 失败场景 | 降级行为 |
|---|---|
| daemon 没起 | hook_client 静默失败；launchd 在下次事件前会拉起 |
| socket 不存在 / 权限拒绝 | hook_client 写 `~/.jarvis-cc/missed.log`，daemon 下次启动可选择性回放 |
| LLM API key 缺失 | 启动 warn，运行时直接走 `phrase.templates` |
| LLM 调用 timeout (5s) | fallback 到 Ollama；Ollama 也失败 → templates |
| Ollama 进程没起 | 跳过 ollama provider，标记不可用 |
| XTTS-v2 模型加载失败 | fallback ElevenLabs（若配） → fallback `say -v Daniel` |
| 参考音频文件不存在 | warn + 用 XTTS 自带预训练男声 |
| 队列积压 > 5 | 丢弃最老的事件，保留最新 5 个 |
| daemon OOM / 崩溃 | launchd `KeepAlive=true` 自动重启 |
| afplay 失败 | 写 daemon.log，不影响下一条事件 |

## 8. 配置 (`~/.jarvis-cc/config.toml`)

```toml
[llm]
provider = "deepseek"
fallback = "ollama"

[llm.deepseek]
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-chat"
base_url = "https://api.deepseek.com"
timeout_seconds = 5

[llm.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-haiku-4-5-20251001"

[llm.openai]
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o-mini"

[llm.ollama]
base_url = "http://localhost:11434"
model = "qwen2.5:7b"
timeout_seconds = 10

[tts]
provider = "xtts"
fallback = "say"

[tts.xtts]
model_dir = "~/.jarvis-cc/models/xtts-v2"
ref_audio_zh = "~/.jarvis-cc/voices/jarvis_zh.wav"
ref_audio_en = "~/.jarvis-cc/voices/jarvis_en.wav"
device = "mps"  # mps | cpu

[tts.elevenlabs]
api_key_env = "ELEVENLABS_API_KEY"
voice_id = ""  # 用户填
model = "eleven_turbo_v2_5"

[behavior]
dedup_window_seconds = 10
queue_max_size = 5
# voice_language: auto = phrase.language.detect_for(cwd) 读 CLAUDE.md 首 500 字判定；
#                 zh / en = 强制单语
voice_language = "auto"
events = ["permission_prompt", "idle_prompt", "elicitation_dialog"]
phrase_max_chars = 30

[paths]
socket = "~/.jarvis-cc/jarvis.sock"
log = "~/.jarvis-cc/daemon.log"
missed_log = "~/.jarvis-cc/missed.log"
```

## 9. 安装与卸载

**安装命令**：

```bash
git clone <repo> ~/myself/jarvis-cc
cd ~/myself/jarvis-cc
uv sync
uv run jarvis-cc install
```

`jarvis-cc install` 会做：
1. 创建 `~/.jarvis-cc/` 目录树（`voices/`, `models/`, `logs/`）
2. 写默认 `config.toml`（如果不存在）
3. 修改 `~/.claude/settings.json` 注册 Notification hook（保留原配置，幂等）
4. 写 `~/Library/LaunchAgents/com.jobin.jarvis-cc.plist`
5. 提示用户放参考音频到 `~/.jarvis-cc/voices/`
6. 提示用户配 `DEEPSEEK_API_KEY` 环境变量
7. `launchctl load` plist

**卸载** (`jarvis-cc uninstall`)：反向所有操作，保留用户音频和日志（除非加 `--purge`）。

**Claude Code hook 配置** (会写入 `~/.claude/settings.json`)：

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "jarvis-cc-hook" }
        ]
      }
    ]
  }
}
```

## 10. 测试策略

**单元测试** (`pytest`)：
- `dedup`：构造时间戳，验证 10s 边界
- `phrase.language`：fixture cwd，验证 zh/en/fallback
- `phrase.router`：mock 各 provider，验证 fallback chain
- `phrase.prompt`：固定输入，验证 prompt 构造
- `tts.engine`：mock providers，验证 fallback chain

**集成测试**：
- `tests/integration/test_hook_to_socket.py`：起 mock daemon，调 hook_client，断言 socket 收到正确 NDJSON
- `tests/integration/test_daemon_e2e.py`：起真 daemon（mock LLM + mock TTS），发事件，断言 player 收到正确 wav

**端到端冒烟**（手动）：
```bash
jarvis-cc test --event permission_prompt --tool Bash --lang zh
# 必须听到中文贾维斯句
```

**性能基准**：
- `tests/bench/test_latency.py`：断言冷启动 < 15s、暖启动 < 3s
- `tests/bench/test_concurrency.py`：5 个事件 0.5s 内并发，验证队列顺序正确

## 11. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| Claude Code 升级改了 Notification payload 字段名 | 在 `hook_client` 做宽容解析，未知字段 passthrough；版本检测在 install 时做 |
| XTTS-v2 上游 API 变化 / 模型下架 | 把模型快照存到 `~/.jarvis-cc/models/`，不每次拉取；锁版本 |
| 参考音频版权风险 | 文档明确「仅本地自用、不分发模型/参考音频」；不在仓库内 commit 任何参考音频 |
| 不同项目语言混合（中英混排） | language.detect_for 返回主语言，LLM prompt 允许少量混排 |
| 用户没装 Ollama 但配了 fallback | 启动时 ping，未启用 provider 移出 chain |
| daemon 启动失败影响 Claude Code | hook_client 永远不报错给 stdout，最多写日志 |

**开放问题（实现期可能要决策）**：
- XTTS-v2 在 macOS MPS 后端的稳定性（部分版本有 NaN 问题），可能需要锁定 PyTorch 版本
- DeepSeek-Chat 对「Tony Stark 管家口吻 + 中文」的输出质量需实测，可能需要调 prompt few-shot
- `idle_prompt` 事件在 Claude Code 中触发频率未知，需要实测决定是否要单独 dedup 策略

## 12. 后续路线（不在 MVP 范围）

- **声音微调** (XTTS-v2 fine-tune 或 RVC)：从 zero-shot 7-8 分像 提升到 8-9 分像
- **iOS / Watch 推送**：通过 ntfy.sh / Pushover 把通知转推到手机
- **Web UI**：`localhost:9527` 看实时事件流、调 prompt、试听
- **多用户配置 profile**：工作 / 学习 / 周末 用不同 voice + prompt
- **响应式静音**：检测会议进行中（zoom / meet）自动静音
- **STT 集成**：Jarvis 听见 "yes/no" 直接帮你回答权限请求
