# jarvis-cli 产品迭代路线图

> 本文档是 jarvis-cli 的产品视角汇总:我们是什么、在业界处于什么位置、现有方案怎么做得更深、以及把它当产品迭代时值得加的功能。
> 调研时点:2026-06。来源见文末。

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
**差距**(竞品有我们没有):hook 事件覆盖少(4/26)、无 webhook 远程通知、无 cost/token 可观测、无 plugin 化分发、无自然语言安装。

---

## 3. 优化方向(把现有能力做深)

### 3.1 意图路由(skills/MCP)
- **现状**:无状态 `gate(整词门控+cosine) → LLM verifier(confirmed/none/unclear) → 注入/澄清`。已领先纯静态 embedding(verifier 提供细粒度区分)。
- **前沿可借鉴**:
  - **history-aware / 多轮状态**(ACE-Router、MTRouter):用会话最近 prompt + 已注入历史改进当轮路由,减少跨轮重复 verify、提升模糊请求的判定。我们现在每轮独立,丢了上下文。
  - **two-stage / 分层检索**:先粗筛候选域再精排,top-k 控制 context。我们 top_k=5 已轻量,但可在 MCP registry 变大时引入。
  - **reasoning-aware reranking**(MemReranker):候选排序时带入意图推理而非纯余弦。

### 3.2 TTS / 延迟
- **现状**:CosyVoice3 / XTTS(Bettany 克隆)/ Piper,已有 `play_stream` 流式。
- **可演进**:
  - **streaming-first**:措辞 LLM token 流 → TTS 流 → 播放流,重叠而非串行(业界 TTFA 已到 40–150ms)。
  - **更快模型**:Voxtral(Mistral 4B 开源、~70ms、可量化本地跑)、Cartesia Sonic(40ms)作为可选 provider。
  - **情感/语气随事件**:报错用凝重语气、完成用轻快语气。

### 3.3 性能(本会话已落地)
- ✅ `_on_query` 的 skills/mcp 两条 pipeline 已 `asyncio.gather` 并行。
- ✅ 检索热路径单次 tokenize、whole-word 列表合并。
- 继续:为热路径加轻量 profiling/计时日志,数据驱动下一步。

---

## 4. 新功能 Roadmap(产品迭代)

### Quick wins(低成本、补竞品差距)
| 功能 | 说明 | 价值 |
|---|---|---|
| **更多 hook 事件** | 现只用 4 个,CC 有 ~26 个生命周期。接入 `PostToolUseFailure`(报错语音)、`Stop`(完成播报)、`PreCompact`(上下文压缩提醒)、`SubagentStop`、rate-limit alert | 高,几乎零架构改动 |
| **报错语音** | 工具失败时 Jarvis 用凝重口吻提示"Sir, the build failed on…" | 高,刚需 |
| **token/成本播报 + statusline** | 会话花费语音/状态栏(对标 codeburn 8k⭐、CCometixLine) | 中高 |
| **webhook / 远程通知** | 离开电脑时推手机/IM(对标 echook) | 中 |

### 中等投入
| 功能 | 说明 |
|---|---|
| **会话记忆 / 主动总结** | 长任务结束语音总结"这轮改了 X、跑了 Y";跨会话上下文(MemTool 思路) |
| **plugin 化分发** | 打包成 CC plugin(skills+hooks+MCP+monitor),支持自然语言安装("tell your AI to install") |
| **可观测性** | 事件流监控/面板(对标 multi-agent-observability 1.5k⭐、claude-code-otel) |
| **history-aware 路由** | 见 3.1,把多轮上下文喂进 gate/verifier |

### 愿景 / 大投入
| 功能 | 说明 |
|---|---|
| **双向对话** | 从单向通知 → 语音问答("Jarvis,刚才那个报错怎么回事") |
| **多 agent 协同感知** | orchestrate 场景下播报各 subagent 进度 |
| **跨设备** | daemon 已多客户端,扩到手机/手表的环境感知 |

---

## 5. 优先级建议(下一步)

按 **价值 ÷ 成本 × 差异化** 排序,建议下一轮迭代先打三个 quick win:

1. **报错语音**(`PostToolUseFailure` → 凝重措辞)——刚需、零架构改动、人格化优势直接体现。
2. **完成播报**(`Stop` 事件)——补齐和 echook 等的基本对位。
3. **成本播报 + statusline**——高频痛点,生态里最热(codeburn 8k⭐)。

这三个都复用现有 hook→daemon→phrase→TTS 管线,不碰核心架构,且把"人格化"卖点扩展到更多时刻。history-aware 路由与 streaming TTS 作为第二梯队的"做深"项。

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
