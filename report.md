# CL-bench Life Harness 优化报告

## 摘要

本项目面向 CL-bench Life 实训任务，目标是在**固定基础模型 `nex-agi/Nex-N2-Pro`** 的前提下，通过优化 agent harness 提升模型在长上下文生活任务上的稳定性和通过率。本项目不训练模型，不修改 dev/test 原始题面、上下文或 rubrics，所有改动均集中在 harness 侧，包括任务解析、路由、分块与检索、结构化中间态、确定性工具、LLM 查询规划、输出校验、失败恢复和实验记录。

我的核心思路是：**不让 LLM 直接面对长、碎、乱的原始生活上下文，而是先通过 harness 将上下文转化为可观测、可调试、可验证的结构化状态，再让 LLM 基于可靠证据完成语义判断和最终表达。** 因此，系统采用 route-aware hybrid 设计：

- Python 负责确定性环节：解析、索引、计数、日期/金额/周期计算、表格渲染、格式检查、引用校验和 trace 记录。
- LLM 负责语义环节：模糊判断、人物关系、情绪/立场、跨片段归纳、查询规划和最终自然语言表达。
- deterministic solver 只作为工具层处理可验证子问题，不作为所有任务的默认解法。
- 每次改动都在 trace 和 dev 实验中记录其修复的失败环节，例如上下文定位、事实抽取、推理、计算、格式或最终校验。

最终 harness 从单次“长上下文 + prompt”调用，演进为一个 workflow-based system：**任务路由 → 结构化中间态 → 证据补充 → LLM 作答 → hard verifier → bounded repair → trace 归因**。

代码库见：https://github.com/YZA114514/NexAU_practice.git

---

## 1. 项目目标与设计原则

CL-bench Life 的任务通常并不要求复杂专业知识，而是要求模型从真实生活风格的上下文中学习并完成任务。这些上下文往往长、碎、乱，可能来自群聊、私聊、论坛、游戏日志、观看历史、订阅记录、健康记录、多文档碎片或版本修订。模型失败常常不是因为完全不会回答，而是因为漏掉细节、误读人物、算错数量、混淆文档来源、输出格式不满足要求，或者在长推理后生成空内容。

因此，本项目的目标不是单纯改写 system prompt，而是把 harness 作为一个工程系统优化。设计原则如下：

1. **结构化优先**：尽量把原始上下文转为表格、事件、人物状态、候选矩阵、computed values 等结构化状态。
2. **工具承担确定性子任务**：对计数、日期、金额、周期、字段抽取、exact list 等可验证任务，优先使用 Python 工具。
3. **LLM 承担语义判断**：对人物动机、情绪、立场、模糊排序、跨文档 claim 对齐等任务，使用 LLM，但限制其基于结构化证据作答。
4. **workflow 级门控**：query planner、deterministic renderer 和 answer repair 并非全局开启，而是根据任务形态和风险选择性启用。
5. **可观测与可回滚**：每题记录 route、workflow、structured state、retrieval evidence、LLM 调用、verifier 结果、repair 过程和 judge 结果，便于失败归因和回滚。

---

## 2. CL-bench Life 任务特点与失败模式

Dev set 的 9 类任务可以归纳为以下几种上下文形态。不同形态的失败风险不同，因此需要不同的 harness 组件。

| 类别 | 典型上下文 | 主要失败风险 | 合适的 harness 组件 |
| --- | --- | --- | --- |
| Game Logs | 游戏日志、PGN、RSS、JSON replay | 长日志计数、重复事件、XML/文本混合 | 结构化 parser、事件表、确定性计数 |
| Digital Footprints & Daily-Life Records | 浏览、观看、听歌、App/设备记录 | 时间线定位、主题聚类、噪声过滤 | event table、date/topic clustering |
| Self-Tracking Trajectories | 健康、训练、订单、订阅 | 周期推断、数值计算、计划表 | recurring items、calendar renderer、calculator |
| Community Interactions | 论坛、Reddit、评论树 | commenter 识别、别名、误归因、楼层关系 | commenter registry、attribution table |
| Group Conversations & Meeting Transcripts | 群聊、会议 | speaker 全局统计、角色、情绪、行动项 | speaker activity、topic timeline、LLM 语义判断 |
| Private Conversations | 私聊、亲友/工作沟通 | 隐含意图、冲突、承诺、未闭环事项 | dialogue state、intent/open-loop extraction |
| Creation & Revision Histories | 草稿、版本、修订记录 | 版本差异、保留/删除项、候选项混淆 | multi-doc matrix、canonical item table |
| Personal Information Fragments | 个人信息碎片 | 实体合并、亲属关系、时间条件 | entity profile、claim-source matrix |
| Public Information Fragments | 公开资料碎片 | claim 对齐、候选过滤、来源引用 | document table、candidate matrix |

这些任务不能只依赖“top-k retrieval + 长 prompt”。例如，评论区题需要识别谁是真正的 commenter，而不是正文里出现的任意大写短语；群聊题需要全局统计 speaker 和议题状态，而不能只看命中的几个片段；订阅题需要把订单历史转化为未来配送计划，而不能让 LLM 直接心算周期；多文档 exact-list 题需要保留 item 的原文形式和来源 section，不能让模型自由改写。

---

## 3. 总体系统流程

当前 harness 的主入口为：

```text
NexAU-main/examples/clbench_life/run_predictions.py
```

主流程如下：

```text
load JSONL
  -> parse_task
  -> route_task
  -> chunk_context
  -> retrieve initial evidence
  -> build_deterministic_findings
  -> build_deterministic_answer if safe
  -> select_workflow
  -> build_structured_state
  -> optional LLM query planner
  -> merge query evidence
  -> render prompt / deterministic answer
  -> call NexAU Agent
  -> hard verifier
  -> bounded answer repair
  -> write prediction JSONL and per-task trace
```

输出严格遵守题目要求：

```json
{
  "id": 0,
  "prediction": "...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

其中 `id` 是输入 JSONL 的 0-based 行号，最后一条 assistant 内容与 `prediction` 一致。

---

## 4. 任务解析与路由设计

### 4.1 `parse_task`：处理 `<|TASK|>` 与多轮上下文

CL-bench Life 的输入中，首条 user message 通常包含：

```text
<长上下文>
<|TASK|>
<第一问>
```

多轮任务还会包含若干 user/assistant 历史，最终只评测最后一轮回答。因此，`parse_task` 的目标是稳定拆分：

- 原始 context；
- 多轮 history；
- 最终 final task；
- 对话中可能继承的约束。

早期如果简单把 `messages[-1]["content"]` 当作最终任务，会把长 context 也混入检索 query，导致 retrieval 失效。因此修复 `<|TASK|>` 后任务提取和多轮 final-task 识别，是整个 harness 的基础。

### 4.2 第一层路由：数据形态路由 `route_task`

`clbench_life_harness.task_router.route_task` 会构造一个 probe，包括：

- 当前 task 文本；
- context 前 80k 字符；
- context 末尾 20k 字符。

这样可以在不把全部长上下文放入 prompt 的情况下捕捉头尾常见格式标记。它返回：

```text
TaskRoute(route, task_type, confidence, signals)
```

主要触发信号如下：

| 触发信号 | route | 目的 |
| --- | --- | --- |
| `runescape.com/m=adventurers-log`、`recent events for:` | `quant_log_solver` | RuneScape 日志事件统计 |
| PGN tags：`[GameId]`、`[Termination]`、`[Event]`，且有 `1/2-1/2` | `exact_compute_solver` | 棋局 draw/termination 分析 |
| `nnet.replay.tracker`、`gameloop` | `exact_compute_solver` | StarCraft replay timeline |
| `HKQuantityTypeIdentifierStepCount` | `quant_log_solver` | 健康步数记录 |
| `purchase_log`、`hero_id`、`players` | `exact_compute_solver` | Dota 购买日志计算 |
| YouTube/Netflix/watch history markers | `quant_log_solver` | 观看历史时间线 |
| `match_id`、`kills`、`deaths`、`assists` | `quant_log_solver` | 游戏比赛统计 |
| bank/credit/transaction markers | `financial_solver` | 金融流水 |
| `<doc>`、document/draft markers | `multi_doc_solver` | 多文档/版本比较 |
| Reddit/thread markers | `thread_tree_solver` | 评论树 |
| speaker turn pattern | `dialogue_social_solver` | 对话/会议 |
| fallback | `general_evidence_solver` | 通用证据任务 |

这一层的作用是从数据格式中发现可用的确定性工具机会。例如 PGN、RSS、JSON replay 不应直接交给 LLM 读全文，而应先被解析成结构化事件。

### 4.3 第二层路由：workflow 路由 `select_workflow`

`clbench_life_harness.workflows.select_workflow` 根据 context subcategory、第一层 route 和 task operator 选择 workflow：

| Workflow | 覆盖范围 | 主要中间态 |
| --- | --- | --- |
| `structured_log_workflow` | 游戏日志、健康记录、观看/听歌历史、订阅、金融流水 | events、aggregates、computed values、calendar |
| `thread_tally_workflow` | Reddit、论坛、博客评论、社区互动 | commenter registry、comment nodes、stance/tally |
| `dialogue_state_workflow` | 群聊、私聊、会议记录 | speaker activity、speaker profile、issues、sentiment evidence |
| `multi_doc_matrix_workflow` | 多文档、版本修订、公开/个人碎片 | document table、claim-doc matrix、canonical item matrix |
| `general_evidence_workflow` | 兜底任务 | retrieval evidence、query evidence |

如果 metadata 不完整，则用 task operator 回退：

- `count_or_aggregate` / `timeline_or_status` -> `structured_log_workflow`
- `compare_or_diff` / `recommendation_or_selection` -> `multi_doc_matrix_workflow`
- `social_inference` -> `dialogue_state_workflow`
- 其他 -> `general_evidence_workflow`

### 4.4 Task operator

`route_task_operator` 从任务文本中抽取操作类型：

| operator | 触发词 | 用途 |
| --- | --- | --- |
| `count_or_aggregate` | count、total、average、top、rank、sort | 计数/聚合 |
| `compare_or_diff` | compare、difference、changed、revision | 比较/版本差异 |
| `quote_or_extract` | quote、snippet、exact | 引用/抽取 |
| `social_inference` | who、support、tension、relationship、feel | 社交推断 |
| `recommendation_or_selection` | recommend、should I、best、focus | 推荐/选择 |
| `timeline_or_status` | when、timeline、before、after、status | 时间线/状态 |

operator 不直接决定答案，而是影响 workflow fallback、prompt constraint 和 verifier 风险提示。

---

## 5. 结构化中间态设计

`StructuredState` 是 harness 的核心中间态。它统一记录：

```text
workflow
context_type
task_operator
tables
evidence_table
computed_values
entity_profiles
timeline
candidate_answers
uncertain_items
parser_confidence
trigger_signals
warnings
fallback_recommendation
```

这个设计的目的不是让表格一次性解决所有任务，而是让每道题的中间过程可检查、可复现、可归因。

### 5.1 通用 EvidenceRow

初始 retrieval hit 和 query-planner retrieval hit 会被转换为 `EvidenceRow`：

- `source`：chunk id 或 workflow source；
- `span`：字符范围；
- `entity`：抽出的 speaker、user、item 或 document；
- `time`：时间戳；
- `claim`：压缩后的片段内容；
- `quote`：第一条非空原文行或关键原文短句；
- `tags`：workflow、matched terms 或 evidence type；
- `confidence`：检索得分、parser confidence 或 neighbor 标记。

这样，prompt 中的证据不再是散乱文本，而是带来源、范围和标签的 evidence pack。

### 5.2 Workflow-specific tables

不同 workflow 会构造不同表：

| Workflow | 关键表 | 目的 |
| --- | --- | --- |
| structured log | `events`、`aggregates`、`recurring_items`、`subscription_cadence_plan`、`planning_calendar` | 事件、计数、订阅周期、未来计划 |
| thread tally | `comments`、`stances`、`tally`、`commenter_registry`、`commenter_attribution` | 评论解析、用户归因、发帖计数 |
| dialogue state | `turns`、`issues`、`speaker_activity`、`speaker_sentiment_evidence` | speaker 统计、议题、情绪/态度候选证据 |
| multi-doc matrix | `documents`、`claim_doc_matrix`、`candidate_matrix`、`canonical_item_matrix` | 多文档候选项与来源 |

### 5.3 Prompt 渲染策略

并不是所有表都会完整塞入 prompt。`render_structured_state_for_prompt` 会根据 workflow 和表类型限制行数并调整优先级。例如：

- `speaker_activity`：展示最相关的 speaker 统计；
- `speaker_sentiment_evidence`：展示候选语义证据；
- `commenter_attribution`：优先展示，避免 commenter misattribution；
- `subscription_cadence_plan`：完整性优先，因为漏项会直接导致失败；
- `canonical_item_matrix`：对多文档 exact-list 任务优先展示。

这种策略在 recall 和 prompt 长度之间折中：模型看到的是关键结构，而不是原始长上下文的重复复制。

---

## 6. 四类核心 Workflow

### 6.1 `structured_log_workflow`

覆盖 Game Logs、Digital Footprints、Self-Tracking Trajectories、部分 financial/quant/exact route。典型上下文包括 PGN、RSS/XML、watch history、health XML、游戏 replay、订阅记录、购买日志等。

处理流程：

```text
parse events
  -> normalize fields
  -> aggregate by entity/time/category
  -> compute values
  -> render evidence or deterministic answer if exact
  -> LLM explain if semantic explanation is needed
```

典型中间态包括：

- `events`：时间、主体、事件类型、原文；
- `aggregates`：按日期、用户、类别、item 的统计；
- `computed_values`：总数、top-k、周期、差值；
- `planning_calendar`：未来月份或时间窗口；
- `subscription_cadence_plan`：商品订阅频率和配送计划。

这一 workflow 的原则是：**Python 负责解析和计算，LLM 负责解释和表达。** 对 PGN draw、poker winnings、step count、纯字段抽取等 exact-computation 任务，可以保留 deterministic renderer；但对趋势解释、偏好判断、异常归纳等任务，仍交给 LLM 基于 computed state 作答。

### 6.2 `thread_tally_workflow`

覆盖论坛、Reddit、博客评论、社区互动等任务。它解决的问题不是一般长文本检索，而是：

- 谁是真正 commenter；
- 谁只是正文中被提到的人；
- 用户别名或显示名如何归并；
- first time / post count / reply relation 如何确定；
- stance、identity quote、misattribution 如何判断。

关键表包括：

- `commenter_registry`：commenter、first_seen、post_count、display_name；
- `commenter_attribution`：answer-facing canonical name、answer_post_count、evidence；
- `comments`：comment node、user、timestamp、score、text；
- `stances`：用户对对象的立场；
- `tally`：投票或支持计数。

这个 workflow 中，Python 负责 commenter 抽取、计数和用户名 grounding；LLM 负责 stance、identity quote、sentiment 和模糊归因。task 59 的改进说明，底层 commenter registry 的准确性直接决定最终答案上限。

### 6.3 `dialogue_state_workflow`

覆盖群聊、私聊、会议记录等任务。它用于处理人物多、时间线乱、态度变化和隐含承诺较多的上下文。

关键表包括：

- `speaker_activity`：speaker 发言数和活跃程度；
- `turns`：speaker turn 与邻近上下文；
- `issues`：讨论议题和状态；
- `speaker_sentiment_evidence`：与情绪、立场、盈利/亏损、满意/不满有关的候选证据；
- `open_loops`：承诺、未完成事项、待确认问题。

这一 workflow 中不能把 speaker count 当成最终答案。比如 task 361 中，“最活跃且与话题相关的用户”不等同于 raw message count top-k。因此，系统让 Python 提供全局统计和候选证据，让 LLM 做最后的语义排序和解释。

### 6.4 `multi_doc_matrix_workflow`

覆盖 Creation & Revision Histories、Personal Information Fragments、Public Information Fragments 等多文档或碎片化资料任务。核心问题是防止 source mixing 和 exact-list 改写。

关键表包括：

- `documents`：doc id、title、section；
- `claim_doc_matrix`：claim 出现在哪些文档；
- `candidate_matrix`：候选对象是否满足各条件；
- `canonical_item_matrix`：exact item 的原文形式、来源 section 和状态；
- `section_items`：每个 section 中的 item 列表。

对 exact-list 任务，query planner 和开放式 repair 可能伤害结果，因为 LLM 会改写、扩写或漏掉 item。因此，这类任务默认关闭 planner，并优先使用 canonical renderer。对 semantic comparison 任务，则允许 LLM 抽象 claim 和判断语义等价，但 Python 保证 document boundary 和 matrix 结构。

---

## 7. LLM Query Planner

早期系统主要依赖 BM25-like retrieval + neighbor expansion。但一些语义任务需要更主动的证据定位，例如身份争议、情绪原因、quote 选择、stance 判断。完全手写规则会过拟合；完全开放式 agent loop 又慢且难复现。因此加入了一个有边界的 LLM query planner。

### 7.1 Planner 输入与输出

`build_query_planner_prompt` 要求 LLM：

- 不要回答任务；
- 只输出 targeted retrieval queries；
- 使用 strict JSON；
- 默认最多生成 6 条 query；
- query 尽量使用原文中的名字、时间戳、短语、实体或字段；
- 不查询内部表名。

输出 schema：

```json
{
  "queries": [
    {
      "purpose": "why this evidence is needed",
      "query": "short retrieval query with exact names, dates, phrases, entities, or fields",
      "expected_fields": ["field_or_fact_to_extract"],
      "quote_required": true,
      "priority": 1
    }
  ],
  "coverage_items": [
    {"kind": "person|item|date|field|section|event|other", "name": "...", "required": true}
  ],
  "output_schema": ["column or section that the final answer should include"]
}
```

`parse_query_plan_response` 会做 JSON 解析和字段校验，非法输出不会进入执行阶段。

### 7.2 Planner 执行方式

`execute_query_plan` 由 Python 执行 LLM 生成的 query：

1. 对每个 query 调用本地 retrieval；
2. 每个 query 取 top-k，默认每 query 4 个 hit；
3. 对命中的 chunk 加前后 neighbor，默认邻居为 1；
4. 去重并限制总命中数，默认最多 28；
5. 生成 `llm_query_evidence` 表，记录 query id、rank、chunk id、行范围、matched terms、purpose 和 snippet。

然后 `merge_query_execution_into_state` 将结果写回 `StructuredState`：

- `llm_query_plan`
- `llm_query_coverage`
- `llm_query_output_schema`
- `llm_query_evidence`

### 7.3 Workflow-level gate

Query planner 并非全局开启。当前策略为：

| 情况 | 是否启用 | 原因 |
| --- | --- | --- |
| 高置信 deterministic answer 已存在 | 否 | 不需要额外 LLM 查询 |
| `structured_log_workflow` | 默认关闭 | 日志/计算题应靠 parser 和计算 |
| `dialogue_state_workflow` | 通常开启 | 需要语义证据、quote 和人物状态 |
| 精确 speaker activity 且已有 `speaker_activity` | 关闭 | 全局计数表更可靠 |
| `thread_tally_workflow` | 条件启用 | 仅 identity、quote、stance、sentiment 等语义任务启用 |
| `multi_doc_matrix_workflow` exact-list | 关闭 | 避免引导模型改写 canonical list |
| `multi_doc_matrix_workflow` semantic comparison | 可开启 | 需要补充 claim/差异证据 |
| `general_evidence_workflow` | 开启 | 缺少专门结构时补充证据 |

这一门控来自实验观察：planner 对 dialogue/thread 任务有帮助，但对 multi-doc exact-list 可能造成负收益。

---

## 8. Deterministic Tools 与泛化控制

### 8.1 使用 deterministic tools 的原因

CL-bench Life 中存在大量半结构化或结构化生活数据，例如 PGN、RSS、watch history、health XML、Dota purchase log、订阅记录和评论区 registry。这类任务如果让 LLM 直接从长上下文中读、算、排，很容易出现：

- 长 reasoning 后空输出；
- 计数错误；
- 漏掉重复项；
- 把噪声当事件；
- 输出格式不稳定。

因此，系统实现了一批 deterministic parser/renderer，但它们被定位为 **tool layer**，不是最终系统本身。

### 8.2 DeterministicAnswer contract

每个 deterministic answer 遵守统一 contract：

```text
kind
summary
content
confidence
trigger_signals
parsed_state_summary
skipped_or_uncertain_items
should_fallback_to_llm
tool_name
llm_role
answer_source
deterministic_core
answer_is_exact
llm_review_recommended
```

该 contract 回答三件事：

1. 为什么工具被触发；
2. 它解析到了什么状态；
3. 是否有不确定项，是否需要 fallback 或 LLM review。

这也是避免 hard coding 的关键：工具按数据形态、confidence 和 exactness 触发，而不是按 task id 触发。

### 8.3 已实现工具类型

| 工具/类型 | 触发方式 | 解决的问题 | 泛化边界 |
| --- | --- | --- | --- |
| RuneScape log parser | Adventurer log/RSS/text event markers | pets、level-up、quest、drop 计数 | 适合同格式日志，不适合语义分类模糊题 |
| PGN draw parser | PGN tags 和 draw result | draw game、termination、last plies | 适合 PGN 结构，不解释复杂棋理 |
| StarCraft replay parser | replay tracker markers | build order/timeline | 适合事件流 |
| Dota purchase parser | purchase JSON | 时间窗口、购买项 | 适合结构化购买日志 |
| watch history pivot | watch history markers | 主题疲劳/回归时间线 | 适合明确观看历史 pivot |
| Subscribe & Save renderer | recurring item/cadence state | 订阅日历和配送计划 | 适合周期订单，不泛化到任意购物推荐 |
| thread commenter registry renderer | commenter attribution table | 用户/别名/发帖次数 | 适合评论者 registry，不做 stance 语义判断 |

### 8.4 确定性与泛化之间的平衡

确定性工具的收益是减少长上下文 reasoning、提高计数和日期计算稳定性、保证输出格式，并支持离线回归测试。但风险是规则可能过窄，parser 错误可能强烈误导最终答案，或者将模糊语义任务错误地确定化。

因此当前原则是：

- 对可验证子问题使用 deterministic tools；
- 对需要理解的子问题只生成 evidence/candidate，不直接给最终结论；
- 每个 tool 暴露 confidence、trigger_signals 和 uncertain_items；
- 当 confidence 不足或 `answer_is_exact=false` 时，交给 LLM 基于 structured state 作答。

---

## 9. Verifier 与 Bounded Repair

### 9.1 Task constraints

`clbench_life_harness.constraints` 从 task 文本中抽取硬约束：

- 是否要求 JSON、table、list；
- 是否要求 quote；
- 是否有 required count；
- 是否禁止 invented usernames；
- 是否有 must-include facets；
- 是否可能要求 exact item 或 canonical wording。

这些不是官方 rubrics，但能抓到很多提交前显然不合格的输出。

### 9.2 Hard verifier

`verify_answer_hard` 检查以下内容：

| 检查 | 目的 |
| --- | --- |
| `non_empty_answer` | 防止空输出 |
| `valid_json` / `markdown_table_present` / `list_items_present` | 检查格式 |
| `required_count_met` | 检查要求数量 |
| `quotes_present` / `quotes_verbatim_in_context_or_state` | 防止伪造引用 |
| `usernames_grounded` | 防止发明用户 |
| `numbers_have_structured_context` | 检查数字是否有结构化依据 |
| `workflow_selected` | 检查是否落到过宽泛 workflow |
| `evidence_or_candidate_state_present` | 检查是否有中间状态 |
| `top_speakers_covered` | speaker activity 风险提示 |
| `sentiment_evidence_not_nei` | 有证据却写 NEI 的风险提示 |
| `commenter_registry_covered` | 检查评论者 registry 覆盖 |
| `subscription_cadence_plan_covered` | 检查订阅计划项覆盖 |
| truncation checks | 检查输出是否疑似截断 |

Hard verifier 不替代 GPT-5.1 judge，而是提交前硬错误检查和失败归因工具。

### 9.3 Bounded answer repair

Repair 只修 verifier 明确指出的错误，并且有最大尝试次数：

- 参数：`--answer-repair-max-attempts`，默认 2；
- 每轮 repair 后重新运行 verifier；
- 如果没有 repairable failures，停止；
- 如果 repair 返回空输出或框架错误，停止；
- trace 记录每轮失败项、prompt 长度、attempts 和 stop reason。

Repair 不是无限 self-reflection。它的原则是：

```text
Only fix the listed verifier errors.
Do not rewrite correct parts.
Do not add unsupported claims.
Do not paraphrase exact items or quotes.
```

对 soft rank/sentiment 任务还有特殊处理：

- `top_speakers_covered` 只作为 warning，不强制按 raw message count 改写答案；
- `speaker_sentiment_evidence` 是 candidate evidence，不是固定答案；
- 如果某个 speaker 有情绪证据但答案写 NEI，repair 可以要求补充或替换为 1-5 rating。

---

## 10. Prompt 与模型调用配置

### 10.1 Answer prompt 结构

最终 answer prompt 包含：

1. system instruction；
2. final task；
3. task constraints；
4. workflow structured state；
5. deterministic findings / candidate answers；
6. evidence pack；
7. output requirements。

关键要求包括：

- 不编造没有证据的事实；
- quote 必须来自原文或 structured state；
- exact count/calculation 使用 computed values；
- 对 `commenter_attribution` 使用 `answer_post_count`；
- 对 `speaker_sentiment_evidence` 只当候选证据，不当固定排序。

### 10.2 推理配置与空输出修复

早期实验发现，模型会出现 reasoning-only 空输出：`content_len=0`，但 `reasoning_content_len` 很长，并且 completion tokens 被耗尽。该问题不是写文件或解析问题，而是模型服务返回了大量 reasoning 但没有 final content。

针对该问题，默认配置改为：

- `thinking_mode=off`；
- prompt 前加 `/no_think`；
- `max_tokens=12000`；
- `request_timeout` 可设为 900；
- `model_call_attempts` 支持多次；
- 指数退避：`retry_backoff_base`、`retry_backoff_max`；
- trace 记录 content/reasoning 长度、usage 和 truncation reason。

这一改动的意义是先让系统失败可见、可复现，避免长时间等待后空输出且无法归因。

---

## 11. 可观测性与 Trace

每道题都会生成 trace，主要内容包括：

- `task_route`：第一层数据形态路由；
- `workflow_decision`：第二层 workflow；
- `task_constraints`；
- chunk 数、原始 retrieval hits、neighbor-expanded hits；
- `structured_state`；
- deterministic answer contract；
- query planner prompt、attempts、status、raw preview；
- LLM 原始返回字段摘要；
- truncation diagnostics；
- hard verification；
- answer repair rounds；
- latency 和 token usage。

Trace 支持以下失败归因：

| 失败环节 | 观察对象 |
| --- | --- |
| 上下文定位 | retrieval hits、query evidence |
| 事实抽取 | structured tables、evidence rows |
| 计算 | computed values、deterministic parser |
| 语义推理 | LLM answer 与 evidence 的关系 |
| 格式 | constraints、hard verifier |
| 失败恢复 | repair attempts 和 stop reason |
| 服务异常 | reasoning/content length、timeout、usage |

---

## 12. 迭代改进记录

| 版本 | 动机 | 改动 | 预期影响 | 验证 |
| --- | --- | --- | --- | --- |
| v1.4 | 多题长时间等待和空输出 | 关闭 thinking、加入 `/no_think`、记录 reasoning/content 长度 | 区分模型调用异常和任务复杂 | task 221 复测，trace 可见 reasoning-only |
| v1.9 | 9 题 smoke 覆盖不足，相关信息常在相邻 chunk | 加 `--neighbor-chunks`，构造 45 题分层 dev 样本 | 提升证据召回，扩大验证覆盖 | dev sample 覆盖 9 子类各 5 题 |
| v1.10 | 多轮 final user 可能重复整段 context，导致任务解析错误 | 修复 `<|TASK|>` 后任务提取；加入 route trace | 避免把长 context 当 task | test row 6 task length 从约 205k 降到 160 |
| v1.12 | 需要从 prompt 转向可验证系统 | 加 constraints、structured state、hard verifier 初版 | 让失败可归因 | dry-run trace 和 compile 检查 |
| v1.13 | RuneScape 日志任务空输出 | 加 RuneScape parser 和 deterministic renderer | 长日志计数稳定化 | task 9 GPT-5.1 judge `16/16` |
| v1.14 | PGN draw 任务空输出 | 加 PGN parser | 棋局结构化抽取稳定 | task 11 GPT-5.1 judge `7/7` |
| v1.15 | deterministic solver 有过拟合风险 | 加 solver contract、confidence、uncertain items、tests | 明确工具边界 | 离线回归测试 |
| v1.21 | 需要完整 workflow-based harness | 加 4 类 workflow、tables、hard verifier、backoff | 从散点工具变系统 | compile + tests + dry trace |
| v1.22 | 需要更可信 dev 评估 | 跑 45 题分层 dev，用 GPT-5.1 judge | 找全局瓶颈 | `6/45` solved，rubric `64.71%` |
| v1.23 | aggregate score 不足以指导修复 | 审计最差失败、near miss 和各 workflow 失败 | 定位根因而非表面错误 | 形成优先级：speaker、commenter、calendar、canonical matrix |
| v1.24 | task 339 订阅计划漏项 | 加 `subscription_cadence_plan` 和 calendar renderer | 订单周期任务稳定化 | task 339 `19/19` |
| v1.25 | 纯 Python 对语义证据不够灵活 | 加 LLM query planner 和 one-shot repair | 语义检索更主动 | 代表性 4 题 rubric 从 `41/86` 到 `55/86` |
| v1.26 | 全局 planner/repair 过宽，thread parser 有伪 commenter | planner 策略门控，thread candidate 过滤 | 降低多文档 exact-list 被改写风险 | task 59 registry 去掉伪 commenter |
| v1.27 | commenter 归因和订阅 renderer 仍不稳 | 加 `commenter_attribution` renderer，稳定 Subscribe renderer | 修复两类可复用 workflow gap | task 59 `11/11`，task 339 `19/19` |
| v1.28 | 模糊 rank 不应被确定性表强修 | 加 `speaker_sentiment_evidence`，repair 对 soft rank 放松 | Python 给证据，LLM 做语义判断 | tests `22 passed`，代表 4 题 `63/86` |
| v1.29 | 时间有限，需要冻结可解释版本 | 写 README/report，保留复现命令 | 工程交付完整 | compile pass，tests `22 passed` |

---

## 13. Dev 实验结果

### 13.1 45 题分层实验

实验设置：

- dev set：9 个官方子类各 5 题；
- 预测模型：`nex-agi/Nex-N2-Pro`；
- judge：OpenRouter `openai/gpt-5.1`；
- 并发：prediction 5，judge 5。

Prediction 结果：

- processed：45；
- error_records：0；
- empty prediction：0；
- suspected truncation：0；
- 平均延迟：`48.62s`。

Judge 结果：

- solved tasks：`6/45`；
- task pass rate：`13.33%`；
- passed rubrics：`431/666`；
- rubric pass rate：`64.71%`；
- parse errors：0。

失败类型：

| failure type | count |
| --- | ---: |
| final_validation | 36 |
| fact_extraction | 32 |
| reasoning | 19 |
| context_location | 15 |
| format | 13 |
| calculation | 1 |

解释：task-level pass rate 低，但 rubric-level pass rate 中等，说明很多答案不是完全错，而是漏掉若干 rubrics 或格式/字段不严格。后续优化应优先增强中间态质量和 final validation，而不是继续增加 prompt 长度。

### 13.2 代表性 4 题实验

选择 task 59、144、339、361 作为代表性失败/目标：

| task | 类型 | 结果 | 说明 |
| --- | --- | --- | --- |
| 59 | thread commenter registry | `11/11` | commenter attribution + renderer 生效 |
| 144 | multi-doc exact list | `8/20` | canonical item matrix 仍不足，exact list renderer 待加强 |
| 339 | Subscribe & Save calendar | `19/19` | cadence plan + deterministic calendar renderer 生效 |
| 361 | speaker activity/sentiment | `25/36` | 全局 speaker 表有帮助，但 fuzzy rank 仍依赖语义判断 |

总体：`63/86` rubrics，`2/4` tasks solved。

这组实验确认了系统边界：

- 可确定的日历/评论归因问题能靠 workflow tools 明显提升；
- 多文档 exact-list 需要更严格的 canonical renderer；
- 模糊 rank/sentiment 不适合靠 deterministic 排序硬修。

---

## 14. 失败根因与修复案例

### 14.1 task 339：Subscribe & Save calendar

失败现象：模型倾向于回答“信息不足”，或漏掉具体商品 cadence。

根因定位：

- 订单记录存在，context location 不是主要问题；
- 失败主要发生在事实抽取和中间状态构建：order history 没有被转化为 recurring item/cadence table；
- 最终生成阶段也容易漏掉必须出现的商品。

修复：

- 提取 `recurring_items`；
- 生成 `subscription_cadence_plan`；
- 构造未来月份 `planning_calendar`；
- 使用 `subscribe_save_calendar_renderer` 输出最终计划。

验证：GPT-5.1 judge 达到 `19/19` rubrics。

泛化性：该机制针对“订单历史 -> 周期计划”这一任务形态，而非某个 task id。风险是部分 cadence 规则仍可能对商品名/包装规格敏感，后续应改进为更通用的间隔估计。

### 14.2 task 59：thread commenter registry

失败现象：

- 模型错把正文短语当 commenter；
- 对 `thomas oak` 和 `Michael Brown` 等误归因/别名处理不稳定；
- post count 使用 displayed count 而不是 answer-facing canonical count。

根因定位：

- 失败发生在事实抽取和归因，不是最终表达；
- thread parser 对 header 边界过宽。

修复：

- 收紧 commenter candidate 过滤；
- 保留合法 ellipsis display name；
- 增加 `commenter_registry`；
- 增加 `commenter_attribution`；
- renderer 使用 `answer_post_count`。

验证：task 59 GPT-5.1 judge 达到 `11/11`。

泛化性：适用于评论区 registry、别名、误归因类任务；不直接解决所有 stance/reasoning 任务，这些仍需 LLM 判断。

### 14.3 task 361：speaker activity and sentiment

失败现象：

- 模型能给出表，但 top users 和 sentiment rating 部分不符合 rubrics；
- 如果按 raw message count 强修，可能替换掉有语义相关证据的 speaker。

根因定位：

- 需要区分“最活跃”和“最活跃且与特定话题相关”；
- 这是语义 rank 问题，不是纯计数问题。

修复方向：

- 增加 `speaker_activity`，提供全局 message count；
- 增加 `speaker_sentiment_evidence`，提供候选 quote、polarity 和 rating_hint；
- repairer 不再强制按 raw count top 10 改写用户集；
- 对有 evidence 却写 NEI 的情况进行软修复。

验证：代表性 run 中该任务仍未全通过，但 rubric 得分保持较高。结论是方向正确，但 dialogue state 仍需更强的 entity、open-loop 和 role 建模。

### 14.4 task 144：multi-doc exact list

失败现象：

- 输出中混入 extras；
- 漏掉 required items；
- category/section drift。

根因定位：

- 多文档 exact-list 需要 canonical item state；
- LLM query planner 和开放式 repair 有时会让模型重写列表，反而破坏精确性。

修复：

- 对 exact-list multi-doc 默认关闭 query planner；
- 对 exact-list multi-doc 默认不做开放式 LLM repair；
- 增加 `canonical_item_matrix`。

后续方向：

- 为 multi-doc exact-list 加 strict schema renderer；
- 对每个 item 记录 section/source/status；
- 最终按 schema 输出，而不是让 LLM 自由组织。

---

## 15. 泛化能力与风险控制

### 15.1 为什么不是 task-id hard coding

当前主要触发条件来自：

- 数据格式标记；
- metadata context_subcategory；
- task operator；
- workflow state 是否存在；
- parser confidence；
- query planner policy。

例如 Subscribe renderer 依赖 `subscription_cadence_plan` 和 `planning_calendar`，不是依赖 task id；thread renderer 依赖 `commenter_attribution`，不是依赖某个固定问题编号。

### 15.2 仍然存在的风险

- 某些 parser 对格式变化敏感；
- 商品 cadence 规则可能对 dev 样例适配较多；
- query planner 可能在 semantic task 上增加噪声证据；
- hard verifier 对 rubrics 的覆盖有限，只能检查硬错误；
- 多文档 canonical list 还不够强；
- 对 dialogue state 的人物意图、关系和 open-loop 建模仍不足。

### 15.3 风险缓解机制

- 每个 deterministic answer 记录 trigger 和 confidence；
- low confidence 时 fallback 给 LLM；
- `uncertain_items` 进入 trace；
- query planner 有 workflow policy gate；
- repair 有最大次数，且不强修 soft rank；
- tests 覆盖 parser、workflow、planner gating 和 verifier。

---

## 16. 复现方式

### 16.1 环境变量

```powershell
$env:LLM_MODEL="nex-agi/Nex-N2-Pro"
$env:LLM_BASE_URL="https://northgate.xiaobei.top/v1"
$env:LLM_API_KEY="<your-key>"
```

如需 GPT-5.1 judge：

```powershell
$env:JUDGE_MODEL="openai/gpt-5.1"
$env:JUDGE_BASE_URL="https://openrouter.ai/api/v1"
$env:JUDGE_API_KEY="<your-openrouter-key>"
```

### 16.2 本地检查

```powershell
.\NexAU-main\.venv\Scripts\python.exe -m compileall .\clbench_life_harness .\NexAU-main\examples\clbench_life
```

```powershell
.\NexAU-main\.venv\Scripts\python.exe -m pytest .\tests\test_structured_solvers.py -q
```

当前结果：`22 passed`。

### 16.3 Dev 45 题实验

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\workflow_dev_45_v1\predictions.jsonl" `
  --trace-dir ".\runs\workflow_dev_45_v1\traces" `
  --ids-file ".\runs\dev_samples\ids_5_per_subcategory_seed20260606.json" `
  --concurrency 5 `
  --model-call-attempts 3 `
  --retry-backoff-base 2 `
  --retry-backoff-max 30 `
  --request-timeout 900 `
  --max-retries 1 `
  --continue-on-error
```

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\judge_rubrics.py `
  --dev-input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --predictions ".\runs\workflow_dev_45_v1\predictions.jsonl" `
  --output ".\runs\workflow_dev_45_v1\judgements_gpt51_v1.jsonl" `
  --ids-file ".\runs\dev_samples\ids_5_per_subcategory_seed20260606.json" `
  --concurrency 5 `
  --max-tokens 8000 `
  --request-timeout 900 `
  --max-retries 2 `
  --thinking-mode auto
```

### 16.4 Test 输出命令

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\实训题目\CL-bench-Life-test.jsonl" `
  --output ".\test_predictions_你的姓名.jsonl" `
  --trace-dir ".\runs\final_test_v1\traces" `
  --concurrency 5 `
  --query-planner auto `
  --answer-repair auto `
  --answer-repair-max-attempts 2 `
  --request-timeout 900 `
  --model-call-attempts 2 `
  --retry-backoff-base 2 `
  --retry-backoff-max 10 `
  --max-retries 1 `
  --continue-on-error
```

如果中断，可追加：

```powershell
--resume
```

---

## 17. 总结

本次优化将 CL-bench Life harness 从单次生成式调用推进为 workflow-based system：

- 路由层识别数据形态和任务操作；
- workflow 层构造可检查的中间态；
- deterministic tools 解决可验证的计数、日期、日历和结构化日志问题；
- LLM query planner 用有限、可追踪的方式补充语义证据；
- hard verifier 和 bounded repair 在最终输出前检查硬约束；
- trace 记录让失败可以被归因、复现和回滚。

当前版本已经解决了 reasoning-only 空输出不可观测、长日志心算、订阅周期、评论者归因等问题，但多文档 exact-list 和复杂 dialogue semantic state 仍是主要短板。下一步最值得改进的是增强 `multi_doc_matrix_workflow` 的 strict canonical renderer，以及增强 `dialogue_state_workflow` 的 entity、open-loop 和 role state，而不是继续堆更长 prompt 或增加孤立的 task-specific solver。

总体而言，这个 harness 的核心不是让 LLM 更努力地读长上下文，而是通过 route-aware workflow 把混乱生活上下文转化为 LLM 能可靠使用的结构化状态；Python 负责让信息可靠，LLM 负责让信息有意义。
