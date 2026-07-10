# jarvis-cli 产品迭代路线图

> 本文档是 jarvis-cli 的产品视角汇总:我们是什么、在业界处于什么位置、现有方案怎么做得更深、以及把它当产品迭代时值得加的功能。
> 调研时点:2026-06。状态更新:2026-07-03(streaming/情感化/事件扩展/plugin Phase 1-2 已落地)。来源见文末。

---

## 1. 现状定位:我们是什么

**一句话**:为 Claude Code / Codex CLI 打造的 **Jarvis 语音通知层 + 每轮智能上下文注入层**。一个本地常驻 daemon 同时服务两个客户端。

现有能力(本会话已迭代到的状态):

```
hook(<10ms, fire-and-forget) ──unix socket──▶ daemon(launchd KeepAlive)
                                                 ├─ phrase router:本地优先多级免费 fallback
                                                 │    ollama → siliconflow → zhipu → deepseek → 模板
                                                 ├─ TTS engine: CosyVoice3 / XTTS(Bettany克隆) / Piper / say,支持流式播放
                                                 └─ 每轮注入(UserPromptSubmit):
                                                      skills RAG  ┐
                                                      MCP routing ┴─ gate(整词门控) → LLM verifier(三态) → 注入/澄清
```

两个差异化内核:
1. **人格化措辞**:不是读 Claude 原文,而是用英式管家口吻重新组织一句话(LLM 生成)。
2. **通知层 ✕ 工具路由合一**:同一个 hook+daemon 既做语音提醒,又在每轮 prompt 上做 skills/MCP 的 RAG 检索 + LLM 验证注入。这是市面上没有人结合的组合。

---

## 2. 业界坐标 & 差异化

| 维度 | 业界现状(2026) | 我们的位置 |
|---|---|---|
| 语音**输入** | CC voice mode(2026-03)已标配,Voxtral 等本地 STT 成熟 | 不做(输入侧),专注**输出/环境感知** |
| 完成/等待**提醒** | 赛道拥挤:echook(26 hooks、chime+voice、webhook、statusline)、claude-sounds、Claude Notifier(VSCode) | 多数是**静态音效或读原文**;我们是**人格化 LLM 措辞 + 声音克隆** |
| 工具/skill 选择 | RAG-MCP 论文:语义检索选子集 **-50% token、+3x 准确率**;但静态 embedding "缺细粒度、忽略多轮状态" | 我们已用 **LLM verifier 三态**补足细粒度(超越纯 embedding);多轮状态是下一步 |
| 成本 | 多依赖云端 API | **本地优先、零成本默认**,免费云端多级兜底 |

**护城河**:人格化 + 本地零成本 + 免费 fallback 链 + 通知/路由合一。
**差距**(竞品有我们没有,2026-07 更新):hook 事件覆盖 15/26(T1+T2 已扩)、无双向语音、安装仍需 PyPI + daemon bootstrap(plugin Phase 3-4 未完)。streaming 管线、情感化语音、plugin 化分发(Phase 1-2)已闭环。(cost/token 可观测、statusline 这类视觉功能不在我们的目标内 —— 见 §5。)

---

## 3. 优化方向(把现有能力做深)

### 3.1 意图路由(skills/MCP)
- **现状**:无状态 `gate(整词门控+cosine) → LLM verifier(confirmed/none/unclear) → 注入/澄清`。已领先纯静态 embedding(verifier 提供细粒度区分)。
- **前沿可借鉴**:
  - **history-aware / 多轮状态**(ACE-Router、MTRouter):用会话最近 prompt + 已注入历史改进当轮路由,减少跨轮重复 verify、提升模糊请求的判定。我们现在每轮独立,丢了上下文。
  - **two-stage / 分层检索**:先粗筛候选域再精排,top-k 控制 context。我们 top_k=5 已轻量,但可在 MCP registry 变大时引入。
  - **reasoning-aware reranking**(MemReranker):候选排序时带入意图推理而非纯余弦。

### 3.2 Streaming 管线 ✅ 已落地(2026-07,经实听迭代定型)

- **最终架构**:LLM token 流 → `chunker.py` 句级分块(首块可子句切)→ XTTS **按段整块解码**(`_split_for_gpt` ≤240 字符/段,段内 `inference_stream` 攒完再交付)→ **PCMPlayer**(sounddevice/PortAudio 回调模式,200ms 块,1s 预缓冲,断流补静音)。`tts.pcm_playback = true`。
- **被数据否决的方案**(勿凭直觉回退,见 memory `project_xtts_audio_architecture`):
  - chunk 级实时流:daemon 内解码 RTF 随负载在 0.6–1.7 波动,RTF>1 时必然饿死(逐词半词)
  - daemon 起的 ffplay/SDL:重启后间歇绑定到不可闻输出设备(正常退出但零声音)
  - 10ms 回调块:Python 回调抢 GIL,撕裂爆音
- **同期修复**:ffplay `-ac`→`-ch_layout mono`(流式此前从未真正工作)、GPT 250 字符静默截断(自切分)、功放唤醒吞头(0.25s 静音前导)、`stream_chunk_size=10` 两头皆输(保持 20)。
- **诊断探针长期保留**:每段 RTF 日志 + 播放欠载计数(DEBUG)。
- **剩余调优**:段大小均衡(~120 字符,超长句按子句切)以缩短段间停顿。

### 3.3 情感化语音 ✅ 已落地(2026-07)

- `types.py` `Emotion` + `EVENT_EMOTION` 映射(warm/grave/pleased/gentle/sardonic/neutral),`prompt.py` 统一 `_EMOTION_CLAUSES` 注入语气 → 所有 TTS 引擎从文字层受益。
- ElevenLabs:emotion → `voice_settings` preset。**XTTS:emotion → 韵律映射**(语速 ×0.92~1.05、温度 ±0.08,clamp [0.3, 0.85])——比原计划的多 embedding 方案更轻,流式/批处理两路径统一生效。
- 流式管线同样穿透 emotion(phrase_stream → 逐句 TTS → 回退合成)。
- 剩余:CosyVoice instruct 接口(需上游)、XTTS 多 embedding(按需)。

### 3.4 性能

已落地(2026-07 累计):
- ✅ `_on_query` skills/mcp 双 pipeline `asyncio.gather` 并行;检索热路径单次 tokenize。
- ✅ Ollama `keep_alive=30m`(措辞 + verifier),模型常驻免冷载。
- ✅ daemon 启动预热:XTTS 模型+latents、Piper en 声音、天气缓存(provider `prewarm()` 钩子)。
- ✅ 出队超龄丢弃(`stale_event_max_age_seconds=60`):积压不再播过期通知。
- ✅ XTTS `stream_chunk_size=10` + ffplay 低延迟 flags + 单 ffplay 会话:首音频与句间衔接。
- ✅ 取消即停:XTTS 解码线程带 stop 信号,cancel 不再白烧 GPU。

进行中 / 待做(按优先级):
1. **措辞预取流水线**:播放事件 N 时预措辞 N+1,积压场景每条 −1~2s。
2. **chunker 首块提前切分**:首块允许逗号切分,首音频 −0.5~1s(需试听)。
3. **热路径计时日志**:phrase/TTS 首字节/总时长分段计时,数据驱动下一步。
4. **sounddevice 进程内播放**:替代 ffplay 子进程,起播趋零、取消即时(大改动)。
5. **措辞模型 A/B**(qwen3:4b vs 8b):首句 −0.5~1s,质量需试听。

---

## 4. 新功能 Roadmap(产品迭代)

### Quick wins(低成本、补竞品差距)

| 功能 | 说明 | 价值 | 估时 |
|---|---|---|---|
| ✅ **Hook 事件扩展** | 7→15/26,T1+T2 全部落地(T2 opt-in) | 已完成 | — |
| ✅ **情感化 prompt 重构** | `_EMOTION_CLAUSES` 统一,全 TTS 引擎受益 | 已完成 | — |
| ✅ **报错语音** | `PostToolUseFailure` → 凝重措辞 | 已完成 | — |
| ✅ **完成播报** | `Stop`/`SubagentStop` 事件 | 已完成 | — |
| ✅ **webhook / 远程通知** | fire-and-forget POST(Bark/ntfy/Slack/Discord) | 已完成 | — |

#### Hook 事件扩展实施清单 ✅ T1+T2 已全部落地(T1 默认开启,T2 opt-in)

每个新事件的实现模式完全一致:`hook_client.py` 加 elif → `types.py` 加类型 → `templates.py` 加中英模板 → `prompt.py` 加 clause → `install.py` 注册。

**Tier 1 — 高价值(~2h)**:

| CC Hook | 通知类型 | Jarvis 语音示例 |
|---|---|---|
| `PreCompact` | `context_compacting` | "Sir, the conversation context is about to be compressed." |
| `RateLimitError` | `rate_limited` | "Sir, we've hit the rate limit — a brief intermission." |
| `SubagentStart` | `subagent_spawned` | "Sir, a sub-agent has been dispatched." |
| `MaxTurnsReached` | `max_turns_reached` | "Sir, the turn limit has been reached — Claude has stopped." |

**Tier 2 — 中价值(~1.5h)**:

| CC Hook | 通知类型 | Jarvis 语音示例 |
|---|---|---|
| `APIError` | `api_error` | "Sir, the API has returned an error." |
| `SessionStop` | `session_end` | "Until next time, sir." |
| `PostCompact` | `context_compacted` | "Context compacted, sir. We carry on." |
| `ContextWindowOverflow` | `context_overflow` | "Sir, the context window is full." |

### 中等投入

| 功能 | 说明 | 估时 |
|---|---|---|---|
| ✅ **Streaming 管线** | 已落地,详见 §3.2 | — |
| ✅ **ElevenLabs 情感 preset** | 已落地,连同 XTTS 韵律映射,详见 §3.3 | — |
| **Plugin 化分发 Phase 3-4** | 升级 UX(`doctor` 诊断、say-only 检测)+ marketplace 提交;Phase 1-2 已完成 | 1-2 天 |
| **多 agent 协同感知** | orchestrate 场景播报各 subagent 进度(`subagent_spawned` 事件已打底) | — |
| **会话记忆 / 主动总结** | 长任务结束语音总结"这轮改了 X、跑了 Y";跨会话上下文(MemTool 思路) | — |
| **history-aware 路由** | 见 §3.1,把多轮上下文喂进 gate/verifier | — |

#### Plugin 化分发方案

**当前安装摩擦**(按严重度):CosyVoice wheel 来自 GitHub URL(PyPI 拒绝) > 模型下载 ~7GB > 六步手动流程 > hooks 绝对路径 > launchd 手动管理。

**两层架构**:

```
Layer 1: CC Plugin (轻量,自动注册 hooks)
├─ .claude-plugin/plugin.json     # 元数据
├─ hooks/hooks.json               # ${CLAUDE_PLUGIN_ROOT} 路径,CC 自动注册
├─ hooks/jarvis-hook.sh           # 薄 wrapper → jarvis-cli-hook
└─ scripts/install-daemon.sh      # 一键 bootstrap

Layer 2: PyPI 包 (核心引擎)
├─ jarvis-cli                     # core: httpx + loguru + say 兜底,零下载开箱即用
├─ jarvis-cli[piper]              # +Piper TTS ~15MB
├─ jarvis-cli[cosyvoice]          # 单独 pip install cosyvoice3
├─ jarvis-cli[xtts]               # +PyTorch + coqui-tts
└─ jarvis-cli[skills]             # +fastembed + jina 模型 ~640MB
```

**分阶段**:
1. PyPI 发布(剥离 cosyvoice 直接 URL,core 只含轻量依赖)— 1-2 天
2. CC Plugin 包装(`hooks.json` + bootstrap 脚本)— 1-2 天
3. 升级 UX(`SessionStart` 检测 say-only 并建议升级,`jarvis-cli doctor` 诊断)— 1 天
4. 提交官方 marketplace — 持续

### 愿景 / 大投入
| 功能 | 说明 |
|---|---|
| **双向对话** | 从单向通知 → 语音问答("Jarvis,刚才那个报错怎么回事") |
| **跨设备** | daemon 已多客户端,扩到手机/手表的环境感知 |

---

## 5. 优先级建议(下一步)

按 **价值 ÷ 成本 × 差异化** 排序(2026-07-03 刷新;原第一/二梯队已全部落地):

### 第一梯队:性能第二波(见 §3.4)

1. **措辞预取流水线** + **热路径计时日志**——积压场景每条 −1~2s,计时数据驱动后续。
2. **chunker 首块提前切分**——首音频再 −0.5~1s(需试听把关)。

### 第二梯队:产品增量

3. **多 agent 协同感知**——orchestrate 重度使用场景,`subagent_spawned` 事件已打底,差播报内容与节流策略。
4. **Plugin 化分发 Phase 3-4**——`jarvis-cli doctor`、say-only 升级提示、marketplace 提交。
5. **中文语音重做(TODO,2026-07-11 挂起,退回英文)**——第一轮尝试整体失败,需要重新设计:
   - 第一轮的否决记录(别再走回头路,探针数据见 memory `zh-voice-architecture`):
     克隆路线(XTTS/CosyVoice + 英文 Bettany 参考)必带洋腔;CosyVoice cross_lingual 必复读;
     zero_shot 锚定止住复读但把英音克隆进普通话("伦敦中文腔");Piper huayan/chaowen、
     macOS 婷婷 AI 味重——全部被耳测否决。用户的验收标准:自然真人感的标准普通话男声。
   - 候选待验证:Kokoro-82M v1.1-zh(本地、Apache-2.0、预置母语音色、实测 RTF 0.18-0.25;
     模型与 zm_010/zm_025/zm_031 已在 `~/.jarvis-cli/models/kokoro-zh/`,音色未经用户认可)。
     其它未探索路线:fish-speech / IndexTTS-2 本地跑 MPS 的可行性、云端 TTS(Edge 等,用户未点头)、
     用户提供中文男声样本走已验证的 CosyVoice 克隆管线。
   - 可复用的基建(这轮沉淀,与最终方案无关都能用):TTS 引擎按语言路由(`provider_zh`)、
     时长守卫按语言 fallback(`fallback_cps_zh`)、Piper g2pW 路径修复、CosyVoice prewarm。
   - 环境坑:HF 直连在本机走系统代理会断(用 hf-mirror + curl);`kokoro`+`misaki[zh]`+`click`
     已入 venv 但未进 extras,resync 会丢(同 cosyvoice3 wheel)。

### 第三梯队:做深(按需)

5. **sounddevice 进程内播放**(§3.4)——起播趋零、取消即时。
6. History-aware 路由、会话记忆 / 主动总结。
7. XTTS 多 embedding、CosyVoice instruct(需上游)。

> 视觉/纯数据类功能(状态栏、成本面板)不进主干:jarvis 的主轴是语音/听觉提醒,这类活交给 ccstatusline 等专门工具。

---

## 6. 原则 / 不可破坏的约束

- **local-first 零成本默认**:任何新功能不得让默认路径依赖付费云。
- **hook 永远 fire-and-forget(<10ms)**:不在 hook 里做阻塞工作,重活留给 daemon。
- **隐私**:发往云端的内容必须经 `cloud_redaction`;本地优先本身就是隐私护城河。
- **退化优雅**:任何 provider/模型缺失都应静默降级(已是现状),不打断主流程。

---

## 来源

- [Claude Code 分层 agentic 架构(2026)](https://www.digitalapplied.com/blog/claude-code-leak-agentic-architecture-lessons-2026) · [Claude Code Docs](https://code.claude.com/docs/en/overview) · [hooks 生命周期](https://github.com/disler/claude-code-hooks-mastery)
- [RAG-MCP:检索式工具选择 -50% token/+3x 准确](https://arxiv.org/html/2505.03275v1) · [ACE-Router / ToolACE-MCP:history-aware 路由](https://arxiv.org/pdf/2601.08276) · [MTRouter:多轮成本感知路由](https://arxiv.org/html/2604.23530v1) · [MemTool](https://arxiv.org/pdf/2507.21428)
- [最快 TTS API 2026](https://smallest.ai/blog/top-fastest-text-to-speech-apis-in-2026) · [Mistral Voxtral 开源流式 TTS](https://www.marktechpost.com/2026/03/28/mistral-ai-releases-voxtral-tts-a-4b-open-weight-streaming-speech-model-for-low-latency-multilingual-voice-generation/) · [本地 TTS 云端质量](https://picovoice.ai/blog/local-text-to-speech-with-cloud-quality/)
- [echook(竞品:26 hooks 音频通知)](https://github.com/ChanMeng666/claude-code-audio-hooks) · [claude-sounds](https://daveschumaker.net/claude-sounds-better-notifications-for-claude-code/) · [codeburn / CCometixLine / awesome-claude-code](https://github.com/jqueryscript/awesome-claude-code)
