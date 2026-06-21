# jarvis-cli

[English](README.md) | **简体中文**

为 [Claude Code](https://claude.com/claude-code) 和 [Codex CLI](https://github.com/openai/codex) 打造的「贾维斯」语音通知层。

当 Claude Code 或 Codex CLI 需要你的注意时——权限确认、空闲等待、MCP elicitation 对话框、AskUserQuestion 选项、Codex 的 `PermissionRequest`/`agent-turn-complete`——后台守护进程会用一句简短的、英式管家腔调的话提醒你,让你在倒咖啡或离开屏幕时也不会错过这个时刻。一个守护进程同时服务两个客户端。

```
[ Claude Code asks: Allow `rm -rf /` ? ]
                  │
                  ▼
   "Sir, that command appears rather drastic."
```

默认技术栈**完全本地、零成本**:用 Ollama 组织措辞,用 CosyVoice 3 发声。云端 provider(DeepSeek、ElevenLabs)仅作为可选的 fallback。

## 工作原理

```
Claude Code ──Notification / PreToolUse hooks──┐
            └ UserPromptSubmit / PostToolUse ──┤
                                               │
Codex CLI   ──PermissionRequest / PreToolUse ──┤──► jarvis-cli-hook (one-shot, <10ms)
            └ UserPromptSubmit / PostToolUse ──┤
            └ notify (agent-turn-complete) ────┘
                                                │
                                                ▼ Unix socket
                                    jarvis-cli-daemon (launchd, KeepAlive)
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          ▼                                           ▼
                   phrase router                                 TTS engine
              (LLM picks Jarvis line)                       (synthesises audio)
                          │                                           │
                Ollama → DeepSeek                       CosyVoice 3 → XTTS → say
                                                                      │
                                                                      ▼
                                                                   ffplay / afplay
```

Codex CLI 的事件映射与验证步骤见 [`docs/CODEX.md`](docs/CODEX.md);之后切换 provider 的方法见 [`docs/SWITCHING.md`](docs/SWITCHING.md)。

- Hook 对通知是「即发即忘」的(10ms 内返回,绝不阻塞 CC)。
- 守护进程在 launchd 下常驻,崩溃后自动重启。
- 以 `(cwd, type, tool)` 为键的 10 秒滑动窗口去重。
- 有界队列(积压超过 5 条时丢弃最旧的)。
- 根据事件 `cwd` 下的 `CLAUDE.md` / `AGENTS.md` / `README.md` 自动判别中英文。
- 当本地 LLM(Ollama)滑落到云端 fallback 时,贾维斯会出声告知,好让你在开始烧 credit 之前就察觉。
- 同一套 hook + 守护进程上的可选第二职能:装上 `skills` extra 后,`UserPromptSubmit` 还会按轮次检索并注入最相关的已装技能(一次约 10-40ms 的守护进程往返)—— 见[技能治理](#技能治理rag-over-skills)。默认关闭;其余所有事件仍是即发即忘。

## 环境要求

- macOS 13+,Apple Silicon(M1/M2/M3/M4)。未在 Intel 上测试。
- Python 3.11+ 和 [`uv`](https://docs.astral.sh/uv/)。
- 已安装并登录的 Claude Code。
- 至少一个 **LLM** 来源:
  - **Ollama**(推荐,本地且免费),运行 `qwen3:8b` 或类似模型,或
  - **DeepSeek** API key(云端,非常便宜),或
  - Anthropic / OpenAI API key。
- 至少一个 **TTS** 来源:
  - **CosyVoice 3** —— Apache-2.0,本地声音克隆,Apple Silicon Metal 加速(`--extra cosyvoice`,推荐默认),或
  - **XTTS-v2** —— 声音克隆,通过 PyTorch 走 Apple Silicon MPS(`--extra xtts`;权重为 CPML / 非商用),或
  - **ElevenLabs** API key(需 `text_to_speech` 权限,云端),或
  - macOS 自带 `say`(零配置,机械音——同时也是通用 fallback)。

## 安装

```bash
git clone https://github.com/JobinJia/jarvis-cli.git
cd jarvis-cli

# 选择你想要的 TTS 路径;extra 是叠加式的。
uv sync --extra cosyvoice          # 推荐
# uv sync --extra xtts             # 旧路径(CPML 非商用)
# uv sync --extra cosyvoice --extra xtts   # 两个都保留
```

在运行 install **之前**,把至少一个 LLM key 导出到你的 shell rc——它会被烤进 launchd plist,这样后台守护进程才能读到:

```bash
echo 'export DEEPSEEK_API_KEY=sk-...'       >> ~/.zshrc   # 可选,仅当你保留 deepseek 作为 fallback
# (可选)
echo 'export ELEVENLABS_API_KEY=sk_...'     >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY=sk-ant-...'  >> ~/.zshrc
echo 'export OPENAI_API_KEY=sk-...'         >> ~/.zshrc
source ~/.zshrc
```

如果你走**完全本地**路线(Ollama + CosyVoice + `say` fallback),完全不需要任何 API key。

然后:

```bash
uv run jarvis-cli install
```

它会:

1. 创建 `~/.jarvis-cli/{voices,models,logs}/`。
2. 若不存在则写入默认的 `~/.jarvis-cli/config.toml`。
3. 修补 `~/.claude/settings.json`,注册 `Notification`、`PreToolUse`、`UserPromptSubmit`、`PostToolUse` hook,指向项目 venv 中 `jarvis-cli-hook` 的绝对路径。若存在 `~/.codex/`,同样修补 `~/.codex/config.toml`,加入等价的 Codex 生命周期 hook 以及 `notify`(以哨兵注释围栏、幂等;见 [`docs/CODEX.md`](docs/CODEX.md))。
4. 写入 `~/Library/LaunchAgents/com.jobin.jarvis-cli.plist`,并嵌入你的 API key。仅当 CosyVoice 用户同时启用了 XTTS 路径时才需要 `COQUI_TOS_AGREED=1`(XTTS 内部使用 Coqui-TTS)。
5. `launchctl load` 该 plist——守护进程立即启动,并在每次登录时启动。

### TTS 模型准备

**CosyVoice 3**(Apache-2.0,推荐):

```bash
# 下载 Candle 格式权重(磁盘占用约 4.7GB)
uv run hf download spensercai/CosyVoice3-0.5B-Candle \
  --local-dir ~/.jarvis-cli/models/cosyvoice3-0.5b-candle

# 提供一段英文参考音频——10-30 秒你想克隆的声音的干净语音
#(例如从播客或采访里截取)。
# 保存到 ~/.jarvis-cli/voices/jarvis_en.wav(单声道 WAV,推荐约 22050Hz)。
```

把这段参考音频的文字稿填入 `config.toml` 的 `[tts.cosyvoice] ref_text_en`——没有它,CosyVoice 会退回到 `inference_cross_lingual`,这会让短句听起来明显重复。

**XTTS-v2**(旧路径,CPML 非商用):

```bash
# 首次合成调用时权重自动从 HuggingFace 下载(约 2GB)。
# 参考音频的要求同上。
```

现在**重启所有正在运行的 Claude Code 或 Codex CLI 会话**,让它们读到被修补过的配置文件。

> **从旧版本升级?** 重新运行 `uv run jarvis-cli install`,以注册驱动「我回复后停止语音」行为的新 `UserPromptSubmit` 和 `PostToolUse` hook。

## 验证

```bash
uv run jarvis-cli status
# {
#   "queue_size": 0,
#   "queue_capacity": 5,
#   "dropped": 0,
#   "last_text": null
# }
```

发一个合成事件并听听看:

```bash
uv run jarvis-cli test --event permission_prompt --tool Bash
# 首次应在约 5-15 秒内听到一句话(模型加载),
# 之后每次约 3-5 秒
```

端到端触发真实 hook:

```
# 在任意项目里打开 Claude Code,让它做一件
# 不在你自动放行列表里的事,例如:
#   "please run sudo ls /root"
# 当 CC 弹出审批对话框时,你应该听到贾维斯。
```

## 配置

一切都在 `~/.jarvis-cli/config.toml` 里。`install` 之后你得到的默认值:

```toml
[llm]
provider = "ollama"            # 本地、零成本;deepseek 保留作 fallback
fallback = "deepseek"

[llm.deepseek]
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-chat"

[llm.ollama]
base_url = "http://localhost:11434"
model = "qwen3:8b"
timeout_seconds = 30

[tts]
provider = "cosyvoice"         # Apache-2.0 本地声音克隆
fallback = "say"               # macOS 自带,通用兜底

[tts.cosyvoice]
model_dir   = "~/.jarvis-cli/models/cosyvoice3-0.5b-candle"
ref_audio_zh = "~/.jarvis-cli/voices/jarvis_zh.wav"
ref_audio_en = "~/.jarvis-cli/voices/jarvis_en.wav"
ref_text_en = ""               # ref_audio_en 的文字稿——强烈建议填写
n_timesteps = 10               # CFM 采样步数(10 = 库默认)

[tts.xtts]                     # 仅当 [tts] provider = "xtts" 时使用
model_dir   = "~/.jarvis-cli/models/xtts-v2"
ref_audio_zh = "~/.jarvis-cli/voices/jarvis_zh.wav"
ref_audio_en = "~/.jarvis-cli/voices/jarvis_en.wav"
device = "mps"                 # mps | cpu
temperature = 0.5              # < 0.75 默认值 → 多次合成间节奏更稳
speed_short = 1.30             # < 60 字符:略微加速(XTTS 会放慢短句)
speed_long  = 1.00             # ≥ 60 字符:不动(XTTS 长句本就流畅偏快)
short_threshold_chars = 60

[tts.elevenlabs]
api_key_env = "ELEVENLABS_API_KEY"
voice_id = ""                  # 用 ElevenLabs 的话务必设置这个!
model = "eleven_turbo_v2_5"

[behavior]
dedup_window_seconds = 10
queue_max_size = 5
voice_language = "en"          # en | zh | auto
events = ["permission_prompt", "idle_prompt", "elicitation_dialog", "ask_user_question"]
phrase_target_chars = 70
phrase_hard_cap = 120
cancel_on_user_action = true   # 当你在发起事件的 CC 会话里回复时,停止播放

[behavior.privacy]
cloud_redaction = true         # 发送前清洗 HOME 路径与疑似密钥的 token
```

编辑后,重新加载守护进程以生效:

```bash
launchctl unload ~/Library/LaunchAgents/com.jobin.jarvis-cli.plist
launchctl load   ~/Library/LaunchAgents/com.jobin.jarvis-cli.plist
```

### 推荐档(零成本、对开源友好)

这是默认配置。本地 Ollama 组织措辞,本地 CosyVoice 3 发声——两者都是 Apache-2.0,稳态下无任何 API 调用。

```toml
[llm]
provider = "ollama"
fallback = "deepseek"          # 仅在 Ollama 不可达时触发;贾维斯会出声告知

[tts]
provider = "cosyvoice"
fallback = "say"
```

### 云端省钱档

```toml
[llm]
provider = "deepseek"          # 便宜且首字延迟低
fallback = "ollama"

[tts]
provider = "elevenlabs"
fallback = "say"

[tts.elevenlabs]
voice_id = "JBFqnCBsd6RMkjVDRZzb"  # George —— 英式旁白音,非常贾维斯
```

更多声音见 [ElevenLabs 声音库](https://elevenlabs.io/app/voice-library)——把任意声音的 ID 复制进 `voice_id`。你的 EL key 只需要 `text_to_speech` 权限。

### 纯本地飞行模式档

```toml
[llm]
provider = "ollama"
fallback = ""

[tts]
provider = "say"               # macOS 自带
fallback = ""
```

不产生任何网络调用。音质下降;这是你真正的离线底线。

## 日常操作

| 操作 | 命令 |
|---|---|
| 检查守护进程健康 | `uv run jarvis-cli status` |
| 发一个合成事件 | `uv run jarvis-cli test --event permission_prompt --tool Bash` |
| 手动触发贾维斯(由 LLM 措辞) | `uv run jarvis-cli say --reason user-input-requested` |
| 手动触发贾维斯(读出原文) | `uv run jarvis-cli say --text "Sir, shall we proceed?"` |
| 跟踪守护进程日志 | `tail -f ~/.jarvis-cli/daemon.log` |
| 重新加载守护进程 | `launchctl unload ~/Library/LaunchAgents/com.jobin.jarvis-cli.plist && launchctl load ~/Library/LaunchAgents/com.jobin.jarvis-cli.plist` |
| 更新 plist 中的 API key | 重新运行 `uv run jarvis-cli install`(幂等) |
| 卸载(保留数据) | `uv run jarvis-cli uninstall` |
| 卸载(清除数据) | `uv run jarvis-cli uninstall --purge` |

## 故障排查

**完全没有声音。**

- `uv run jarvis-cli status` —— 守护进程可达吗?
- `launchctl list | grep jarvis` —— 服务在运行吗?
- `tail ~/.jarvis-cli/daemon.log` —— 有报错行吗?
- 测试最末端:`say "test"` —— 扬声器正常吗?

**守护进程在跑但 `last_text` 从不变化。** Hook 没能到达 socket。常见原因:

- 你在安装**之后**才加 API key —— 重新运行 `jarvis-cli install` 把它们重新烤进 plist,然后重载守护进程。
- 你的 Claude Code 会话在安装**之前**就已运行 —— 重启 CC,让它重新读取 `~/.claude/settings.json`。
- `cat ~/.claude/settings.json | jq '.hooks.Notification'` 应当显示 `.venv/bin/jarvis-cli-hook` 的绝对路径。若显示的是裸的 `jarvis-cli-hook`,重新运行 install。

**你听到「Sir, the local language model … appears unreachable. I am falling back to the cloud.」** Ollama 要么没运行、模型没拉取,要么请求超时。启动 `ollama serve`,确认 `ollama list` 包含 `config.toml` 里的模型,再试 `curl http://localhost:11434/api/tags`。持续中断期间,该提醒被限流为每五分钟一次。

**CosyVoice 把短句念重了(「Sir Sir, ready ready」)。** 你没填 `[tts.cosyvoice] ref_text_en` —— 没有文字稿,provider 会退回到 `inference_cross_lingual`,它在短句上会幻觉式重复。转写你的 `jarvis_en.wav`(`uvx --from openai-whisper whisper jarvis_en.wav --model tiny --language English`),把清理后的文本粘进配置字段。

**XTTS 流水线以 `isin_mps_friendly` ImportError 崩溃。** `transformers>=5` 移除了 coqui-tts 0.27 仍在导入的符号。`pyproject.toml` 里的 `[xtts]` extra 正是为此把 `transformers<5` 精确锁定 —— 重新运行 `uv sync --extra xtts`。

**`say` 报 `Opening output file failed: fmt?`。** 这是 macOS 的 `say` 二进制在没有显式 `--data-format` 时拒绝写 `.wav`。provider 已替你处理(`--data-format=LEF32@22050`);这条消息意味着你在跑一个旧版守护进程。重新 `uv sync` 并重载。

**ElevenLabs 返回 401 并带 `quota_exceeded`。** 你的免费额度用完了。ElevenLabs 对额度返回 401(而非 402/429)—— 守护进程把它翻译成 `daemon.log` 里一条可读的行(`TTS provider elevenlabs failed: ElevenLabs quota exhausted: …`)。充值、换一个有额度的 key,或切到 CosyVoice / XTTS。

**Ollama 在 qwen3 / R1 类模型上返回空文本。** 确保你的 Ollama 是 0.9+;provider 会自动传 `think: false`。若你钉了旧版 Ollama,请升级。

**贾维斯对我的命令说错了内容。** 内容感知会把 `tool_input`(例如实际的 Bash 命令、文件名)喂进 LLM 提示。若那句话仍显得笼统,检查 `daemon.log` 看 provider 调用是否成功——当 LLM 出错时,守护进程会退回到通用模板。

## 手动触发

Claude Code 只对工具权限确认、空闲等待和 MCP elicitation 触发它的 Notification hook。有些场景不在其中——最典型的是助手主动发起的提问(`AskUserQuestion`,现在走 `PreToolUse` hook)。两种模式:

**由 LLM 措辞** —— 给模型一个上下文标签,让它写出那句话:

```bash
uv run jarvis-cli say --reason "user-input-requested"
# 听到:"Sir, your input is awaited."
```

**读出这段原文** —— 完全绕过 LLM(更快、可预测,适合念出实际问题):

```bash
uv run jarvis-cli say --text "Sir, shall this repository be made public or private?"
# 听到:<原文>
# 默认 --lang en;用 --lang zh 切换声音/发音
```

**为单次调用覆盖声音** —— 适合在不改配置的情况下 A/B 试听候选声音:

```bash
uv run jarvis-cli say \
  --text "Sir, sample line for voice tasting." \
  --voice onwK4e9ZLuTAKqWW03F9        # Daniel,更低沉的英式男声
# 下一次不带 --voice 的 `say` 会回到配置默认声音
```

当激活的 TTS provider 是 ElevenLabs 时,`--voice` 是一个 ElevenLabs `voice_id`;当激活的是 `say` 时,它是 macOS 的 `say` 声音名(如 `Karen`、`Daniel`、`Tingting`)。CosyVoice 和 XTTS 都会忽略该覆盖——它们从参考音频克隆,而非从命名声音。

所有模式都搭载在 `idle_prompt` 事件上,并带一个唯一的 `tool_name`(来自 `--reason` 或自动生成的 uuid),这样去重不会折叠连续的调用。

## 技能治理(RAG-over-skills)

随着你安装越来越多的 Claude Code / Codex 技能,每个技能的 `description` 无论你用不用都会被加载进启动提示——上下文随技能数量增长。`skills` extra 隐藏这条长尾,转而**按轮次**呈现合适的技能:`UserPromptSubmit` hook 嵌入你的提示,从本地索引检索最接近的技能,并把匹配到的技能正文作为 `additionalContext` 注入。由于它直接注入正文(而非经由 Skill 工具),即使是从启动列表中隐藏的技能、或位于已禁用插件里的技能,它也能生效。

可选启用、自包含——只用 TTS 的用户不会拉入任何 embedding 相关依赖。

```bash
uv sync --extra skills          # 加入 fastembed(ONNX,无 PyTorch)+ numpy + pyyaml

# 在 ~/.jarvis-cli/config.toml 中启用
# [skills]
# enabled = true

# 预先拉取模型(可断点续传;慢网推荐)
jarvis-cli skills download

jarvis-cli skills status        # 列出发现的技能(不加载模型)
jarvis-cli skills query 帮我提交代码   # 看一个提示会检索到什么

# 一键应用隐藏策略,可回滚
jarvis-cli skills govern --dry-run   # 预览将隐藏 / 禁用什么
jarvis-cli skills govern             # 隐藏独立技能 + 禁用带技能的插件
jarvis-cli skills govern-status      # 当前治理在管什么
jarvis-cli skills restore            # 按 manifest 撤销
```

它如何与守护进程已在运行的 hook 协同:

- **Embedding 模型** —— `jinaai/jina-embeddings-v2-base-zh`(中英双语,ONNX,约 0.64GB,一次性下载到 `~/.jarvis-cli/skills/models`)。选它是为了跨语言召回:中文提示能匹配英文技能描述。热查询约 10-15ms;模型在守护进程启动时预热。
- **检索** —— 在本地索引(`catalog.json` + `vectors.npy`)上做余弦相似度,外加一个小的词法加权,使共享的专有名词(提示里出现 `vercel`/`vue`/`git`)抬升明显匹配项。按分数分级:高置信命中注入技能正文;较弱的给出一行菜单;再低则什么都不做。
- **隐藏长尾** —— `jarvis-cli skills govern` 把策略固化下来,并记录一份 manifest,使 `skills restore` 能精确反向撤销。独立的 `~/.claude/skills/` 会在 `.claude/settings.local.json` 里被设为 `skillOverrides`(`"user-invocable-only"`)—— 从模型启动上下文中剔除,同时保留 `/name` 可用。插件技能无法按单个技能隐藏,所以带技能的插件被整体禁用(其 agent 会先被重新安置到 `~/.claude/agents/`,例如 superpowers 的 `code-reviewer` 得以保留);不含技能的插件则保持不动。无论哪种情况,检索 hook 仍会无视启用状态从磁盘呈现每一个技能。`--keep name1,name2` 可保留一组热点技能可见。

在 `config.toml` 的 `[skills]` 下调阈值与模型(见 `SkillsConfig`)。没有该 extra 时一切退化为空操作:没有模型、没有注入、TTS 不受影响。

## 项目结构

```
src/jarvis_cli/
├── hook_client.py        # one-shot stdin → socket bridge
├── daemon/
│   ├── main.py           # asyncio entrypoint
│   ├── listener.py       # unix-socket server
│   ├── dedup.py          # sliding-window dedup
│   ├── queue.py          # bounded drop-oldest queue
│   └── health.py         # /health on 127.0.0.1:9527
├── phrase/
│   ├── router.py         # LLM chain + on_primary_fallback alert hook
│   ├── language.py       # cwd → 'zh' | 'en'
│   ├── prompt.py         # Jarvis-tone system prompt + few-shot
│   ├── templates.py      # final fallback strings
│   └── providers/        # deepseek, anthropic, openai, ollama
├── tts/
│   ├── engine.py         # primary → fallback
│   └── providers/        # cosyvoice, xtts, elevenlabs, say
├── skills/               # RAG-over-skills (optional `skills` extra)
│   ├── catalog.py        # scan CC+Codex+plugin SKILL.md
│   ├── embedder.py       # fastembed ONNX (lazy)
│   ├── index.py          # catalog.json + vectors.npy
│   ├── retriever.py      # cosine + lexical boost
│   ├── injector.py       # tiered body / menu / none
│   ├── service.py        # daemon-side query + per-session dedup
│   ├── govern.py         # apply/restore the hiding policy (manifest)
│   └── cli.py            # jarvis-cli skills status|query|download|govern|restore
├── player.py             # afplay + ffplay (streaming) wrappers
├── config.py             # TOML loader, dataclass schema
└── install.py            # CLI: install / uninstall / status / test / say
```

更多文档见 [`docs/`](docs/):
- [`docs/CODEX.md`](docs/CODEX.md) —— Codex CLI 事件映射、自动修补内部细节、验证步骤。
- [`docs/SWITCHING.md`](docs/SWITCHING.md) —— provider/档位切换配方(XTTS ⇄ CosyVoice ⇄ ElevenLabs ⇄ `say`)。

`tests/` 下有 321+ 单元 + 集成测试。用 `uv run pytest` 运行。

## 发版

通过推送 `v*` tag 来发版。[`release` workflow](.github/workflows/release.yml)
会校验 tag 与 `pyproject.toml` 里的 `version` 一致,用 `uv build` 构建 sdist +
wheel,并发布一个带产物和自动生成 release notes 的 GitHub Release。

不发 PyPI:`cosyvoice` extra 通过直链 URL 安装 wheel(`allow-direct-references`),
而 PyPI 不接受直链依赖。分发方式是源码树加上每个 GitHub Release 上挂的产物。

```bash
# 改版本号,同步 lockfile,提交
$EDITOR pyproject.toml                       # 例如 0.4.0 → 0.4.1
uv lock
git commit -am "chore(release): bump 0.4.0 → 0.4.1"

# 打 tag 并推送 —— 剩下的交给 workflow 构建并发布
git tag -a v0.4.1 -m "v0.4.1 —— <亮点>"
git push origin main v0.4.1
```

tag 与版本号不一致时守卫会让构建失败,所以贴错号的版本永远发不出去。

## 许可

项目代码为 MIT —— 见 `LICENSE`。

第三方模型权重各有自己的许可;项目把选择哪条路径的控制权留给用户:

- **CosyVoice 3**(`spensercai/CosyVoice3-0.5B-Candle`、`cosyvoice3.rs`)—— **Apache-2.0**。可商用。
- **XTTS-v2**(`coqui/XTTS-v2`)—— **CPML,非商用**。模型权重本身禁止商业使用;保留 `[xtts]` extra 仅供个人 / 研究部署。
- **ElevenLabs / DeepSeek / Anthropic / OpenAI** —— 受各 provider 自身 ToS 约束。

声音样本、录制的模型以及合成的音频受各自条款约束——切勿把真人的参考音频或生成的声音克隆提交到本仓库。
