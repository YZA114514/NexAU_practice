# CL-bench Life Harness 优化报告

## 1. 项目目标与设计原则

本项目面向 CL-bench Life 实训题，目标是在固定基础模型 `nex-agi/Nex-N2-Pro` 的前提下优化 agent harness。项目没有训练模型，也没有修改 dev/test 原始数据或 rubrics；所有改动都集中在 harness 的上下文组织、工具、路由、结构化中间态、校验、失败恢复和实验记录上。

老师强调本题不是考察“谁能写最长的 prompt”，而是考察能否把 agent harness 当成一个可观测、可调试、可验证的系统来优化。因此我采用的核心原则是：

- Python 负责可确定的事情：解析、分块、索引、计数、日期/金额/周期计算、表格生成、格式校验、trace 记录。
- LLM 负责语义判断：模糊排序、情绪强度、人物动机、跨片段归纳、最终自然语言表达。
- deterministic solver 只处理可验证子问题，不作为所有任务的默认答案机制。
- 每个改动都要能在 trace 中解释它修复的是上下文定位、事实抽取、推理、计算、格式还是最终校验。
- 实验结论优先基于 dev 抽样和失败分析，不针对 test set 的未知 rubrics 做不可验证调参。

## 2. CL-bench Life 任务特点

CL-bench Life 的难点通常不在题面，而在上下文。题目往往要求从长、碎、乱、社交化的生活上下文中恢复事实、归纳状态、完成计算或满足精确输出格式。一个答案只要漏掉一个字段、错一个名字、把相邻发言误归因、把商品周期算错，就可能整题不通过。

dev set 的 9 类任务可以归纳如下：

| 类别 | 典型上下文 | 主要失败风险 | 合适的 harness 组件 |
| --- | --- | --- | --- |
| Game Logs | 游戏日志、PGN、RSS、JSON replay | 长日志计数、重复事件、XML/文本混合 | 结构化 parser、确定性计数、computed values |
| Digital Footprints & Daily-Life Records | 浏览/观看/听歌历史 | 时间线定位、主题聚类、噪声过滤 | event table、topic/date clustering、检索邻居 chunk |
| Self-Tracking Trajectories | 健康、训练、订单、订阅 | 周期推断、数值计算、计划表 | recurring items、calendar renderer、calculator |
| Community Interactions | 论坛、Reddit、评论树 | commenter 识别、别名、误归因、楼层 | commenter registry、attribution table |
| Group Conversations & Meeting Transcripts | 群聊、会议 | speaker 全局统计、角色、情绪、行动项 | speaker activity、timeline、LLM 语义判断 |
| Private Conversations | 私聊、亲友/工作沟通 | 隐含意图、冲突、承诺、未闭环事项 | dialogue state、issue/open-loop extraction |
| Creation & Revision Histories | 草稿、版本、修订记录 | 版本差异、保留/删除项、候选项混淆 | multi-doc matrix、canonical item table |
| Personal Information Fragments | 个人信息碎片 | 实体合并、亲属关系、时间条件 | entity profile、claim-source matrix |
| Public Information Fragments | 公开资料碎片 | claim 对齐、候选过滤、来源引用 | document table、candidate matrix |

这也决定了 harness 不能只做“top-k 检索 + 长 prompt”。比如：

- 评论区题需要知道谁是真的 commenter，谁只是正文里出现的人名。
- 群聊题需要从完整对话中统计 speaker，而不是只看检索命中的几个片段。
- 订阅题需要把订单历史转成未来配送计划，而不是让模型心算周期。
- 多文档题需要保留每个 item 来自哪个 section，不能让模型自由改写。

## 3. 总体系统流程

当前 harness 的主入口是 `NexAU-main/examples/clbench_life/run_predictions.py`，核心流程如下：

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

输出文件严格遵守题目要求：

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

其中 `id` 是 test JSONL 的 0-based 行号，最后一条 assistant 内容和 `prediction` 一致。

## 4. 路由设计

路由是本 harness 的第一层关键设计。它不是单纯按 metadata 分类，而是分成两层：数据形态路由和 workflow 路由。

### 4.1 第一层：数据形态路由 `route_task`

`clbench_life_harness.task_router.route_task` 会构造一个 probe：

- 当前 task 文本；
- 上下文前 80k 字符；
- 上下文末尾 20k 字符。

这样做是为了在不读取全部长上下文进 prompt 的情况下，捕捉头尾常见的格式标记。它返回：

```text
TaskRoute(route, task_type, confidence, signals)
```

主要规则包括：

| 触发信号 | route | 目的 |
| --- | --- | --- |
| `runescape.com/m=adventurers-log`、`recent events for:` | `quant_log_solver` | RuneScape 日志计数 |
| PGN tags：`[GameId]`、`[Termination]`、`[Event]` 且有 `1/2-1/2` | `exact_compute_solver` | 棋局 draw/termination 分析 |
| `nnet.replay.tracker`、`gameloop` | `exact_compute_solver` | StarCraft replay timeline |
| Health XML：`HKQuantityTypeIdentifierStepCount` | `quant_log_solver` | 步数/健康记录 |
| Dota purchase JSON：`purchase_log`、`hero_id`、`players` | `exact_compute_solver` | 游戏购买窗口计算 |
| YouTube/Netflix/watch history | `quant_log_solver` | 观看历史时间线 |
| `match_id`、`kills`、`deaths`、`assists` | `quant_log_solver` | 游戏比赛统计 |
| bank/credit/transaction markers | `financial_solver` | 金融流水 |
| `<doc>`、document/draft markers | `multi_doc_solver` | 多文档/版本比较 |
| Reddit/thread markers | `thread_tree_solver` | 评论树 |
| speaker turn pattern | `dialogue_social_solver` | 对话/会议 |
| fallback | `general_evidence_solver` | 通用证据任务 |

这一层的价值是“从数据格式发现可确定工具机会”。例如 PGN、RSS、JSON replay 不应该交给 LLM 自由读，而应先解析成结构化事件。

### 4.2 第二层：workflow 路由 `select_workflow`

`clbench_life_harness.workflows.select_workflow` 再根据 `context_subcategory`、第一层 route 和 task operator 选择 4 类 workflow：

| Workflow | 来源 | 覆盖范围 |
| --- | --- | --- |
| `structured_log_workflow` | Game Logs、Digital Footprints、Self-Tracking、exact/quant/financial route | 日志、时间线、数值、订阅、健康、游戏 |
| `thread_tally_workflow` | Community Interactions、thread route | 评论树、用户计数、stance、归因 |
| `dialogue_state_workflow` | Group Conversations、Private Conversations、dialogue route | 群聊、私聊、会议、speaker 状态 |
| `multi_doc_matrix_workflow` | Creation/Personal/Public fragments、multi-doc route | 多文档、候选矩阵、版本差异 |

如果 metadata 不完整，则用 task operator 回退：

- `count_or_aggregate` / `timeline_or_status` -> `structured_log_workflow`
- `compare_or_diff` / `recommendation_or_selection` -> `multi_doc_matrix_workflow`
- `social_inference` -> `dialogue_state_workflow`
- 其他 -> `general_evidence_workflow`

### 4.3 Task operator

`route_task_operator` 用 task 文本关键词粗分操作类型：

| operator | 触发词 | 用途 |
| --- | --- | --- |
| `count_or_aggregate` | count、total、average、top、rank、sort | 计数/聚合 |
| `compare_or_diff` | compare、difference、changed、revision | 比较/版本差异 |
| `quote_or_extract` | quote、snippet、exact | 引用/抽取 |
| `social_inference` | who、support、tension、relationship、feel | 社交推断 |
| `recommendation_or_selection` | recommend、should I、best、focus | 推荐/选择 |
| `timeline_or_status` | when、timeline、before、after、status | 时间线/状态 |

operator 不直接决定答案，只影响 workflow fallback、prompt constraint 和 verifier 风险提示。

## 5. 结构化中间态设计

`StructuredState` 是当前 harness 的核心中间状态。它统一记录：

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

这样做的目的不是让结构化表一次性解决所有任务，而是让每一题的中间状态可检查、可复现、可归因。

### 5.1 通用 evidence table

初始 retrieval hit 会被转换成 `EvidenceRow`：

- `source`：chunk id；
- `span`：字符范围；
- `entity`：chunk 中抽出的 speaker；
- `time`：chunk 中抽出的 timestamp；
- `claim`：压缩后的片段内容；
- `quote`：第一条非空原文行；
- `tags`：workflow 和 matched terms；
- `confidence`：检索得分或 neighbor 标记。

这让 prompt 中的证据不只是散乱文本，而是带来源、范围和标签的 evidence pack。

### 5.2 workflow-specific tables

不同 workflow 会构造不同表：

| Workflow | 表 | 目的 |
| --- | --- | --- |
| structured log | `events`、`aggregates`、`recurring_items`、`subscription_cadence_plan`、`planning_calendar` | 事件、计数、订阅周期、未来计划 |
| thread tally | `comments`、`stances`、`tally`、`commenter_registry`、`commenter_attribution` | 评论解析、用户归因、发帖计数 |
| dialogue state | `turns`、`issues`、`speaker_activity`、`speaker_sentiment_evidence` | speaker 统计、情绪/态度候选证据 |
| multi-doc matrix | `documents`、`claim_doc_matrix`、`candidate_matrix`、`canonical_item_matrix` | 多文档候选项与来源 |

### 5.3 Prompt 中的呈现策略

并不是所有表都完整塞进 prompt。`render_structured_state_for_prompt` 会限制每类表的行数，例如：

- `speaker_activity`：前若干行；
- `speaker_sentiment_evidence`：用于候选语义证据；
- `commenter_attribution`：优先展示；
- `subscription_cadence_plan`：完整性优先；
- `canonical_item_matrix`：对多文档列表任务优先。

这样做是在 recall 和 prompt 长度之间取平衡：给模型看关键结构，而不是把全部上下文再复制一次。

## 6. Query planner 设计

最初的 harness 更偏“Python 检索 + LLM 一步回答”。但一些语义题需要更主动的证据定位，例如身份争议、情绪原因、quote 选择、stance 判断。完全手写规则会过拟合；完全开放式 agent loop 又太慢、不稳定、难复现。因此我加入了有边界的 LLM query planner。

### 6.1 Query planner 的输入和输出

`build_query_planner_prompt` 明确告诉 LLM：

- 不要回答任务；
- 只决定还需要哪些 targeted evidence queries；
- 输出 strict JSON；
- 最多生成指定数量 query，默认最多 6 条；
- query 要用原文中的名字、时间戳、短语，不要查询内部表名。

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

### 6.2 Query plan 的执行

`execute_query_plan` 不让 LLM 直接读工具结果，而是由 Python 执行：

1. 对每个 query 调用本地 retrieval。
2. 每个 query 取 top-k，默认每 query 4 个 hit。
3. 对命中的 chunk 加前后 neighbor chunk，默认邻居为 1。
4. 去重并限制总命中数，默认最多 28。
5. 生成 `llm_query_evidence` 表，记录 query id、rank、chunk id、行范围、matched terms、purpose、snippet。

然后通过 `merge_query_execution_into_state` 把结果写回 `StructuredState`：

- `llm_query_plan`
- `llm_query_coverage`
- `llm_query_output_schema`
- `llm_query_evidence`

最后再和原始 retrieval hits 合并，形成最终 evidence pack。

### 6.3 Query planner 的策略门控

query planner 不是所有任务都开。`should_plan_queries` 的策略是：

| 情况 | 是否启用 | 原因 |
| --- | --- | --- |
| 已有高置信 deterministic answer | 否 | 不需要额外 LLM 查询 |
| structured state 已有 exact candidate answer | 否 | 避免改写正确结构 |
| `dialogue_state_workflow` | 通常是 | 语义证据和 quote 常重要 |
| 精确 speaker activity 题且已有 `speaker_activity` | 否 | 全局计数表比 LLM query 更可靠 |
| `thread_tally_workflow` | 条件启用 | 仅 identity、quote、stance、sentiment、misattribution 等语义任务启用 |
| `multi_doc_matrix_workflow` | 精确列表题关闭 | 避免 planner/repair 诱导模型改写 canonical list |
| `structured_log_workflow` | 默认关闭 | 日志/计算题应靠 parser 和计算 |
| `general_evidence_workflow` | 启用 | 缺少专门结构时用 planner 补证据 |

这个门控是一次重要迭代：早期全局开启 query planner 后，部分多文档 exact-list 任务反而变差，因为模型被额外检索证据带偏，改写了原本需要严格保留的列表。

## 7. Deterministic solver 与泛化机制

### 7.1 为什么需要 deterministic solver

CL-bench Life 中有一类任务本质上是结构化数据处理：

- PGN 棋局 draw comparison；
- RuneScape Adventurer log 事件统计；
- YouTube/watch history pivot；
- Health XML 步数；
- Dota purchase log；
- Subscribe & Save 订单周期；
- Reddit/thread commenter registry。

这些任务如果交给 LLM 直接心算，容易出现：

- 长 reasoning 空输出；
- 算错次数；
- 漏掉重复项；
- 把噪声当事件；
- 输出格式不稳定。

因此我实现了一批 deterministic parser/renderer，但给它们加了边界。

### 7.2 DeterministicAnswer contract

每个 deterministic answer 都遵守同一个 contract：

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

这能回答三个问题：

- 为什么这个工具被触发？
- 它解析到了什么状态？
- 哪些项目不确定，是否应该 fallback 给 LLM？

这也是避免“硬编码补丁”的关键：solver 按数据形态和 confidence 触发，而不是按 task id 触发。

### 7.3 已实现的 deterministic 工具类型

| 工具/类型 | 触发方式 | 解决的问题 | 泛化边界 |
| --- | --- | --- | --- |
| RuneScape log parser | Adventurer log/RSS/text event markers | pets、level-up、quest、drop 计数 | 适合同格式日志，不适合语义分类模糊题 |
| PGN draw parser | PGN tags 和 draw result | draw game、termination、last plies | 适合 PGN 结构，不解释复杂棋理 |
| StarCraft replay parser | replay tracker markers | build order/timeline | 适合事件流 |
| Dota purchase parser | purchase JSON | 时间窗口、购买项 | 适合结构化购买日志 |
| watch history pivot | watch history markers | 主题疲劳/回归时间线 | 适合明确观看历史 pivot |
| Subscribe & Save renderer | recurring item/cadence state | 订阅日历和配送计划 | 适合周期订单，不泛化到任意购物推荐 |
| thread commenter registry renderer | commenter attribution table | 用户/别名/发帖次数 | 适合评论树 registry，不做 stance 语义判断 |

### 7.4 trade off：确定性 vs 泛化

确定性工具的好处：

- 减少长上下文 reasoning；
- 让计数、日期、金额、周期可复现；
- 输出格式稳定；
- 可写离线回归测试。

风险：

- 规则写窄了会只修一个 dev 样例；
- parser 错误可能强烈误导最终答案；
- 对模糊语义任务强行确定化会伤害泛化。

因此当前原则是：

- 对“可以验证”的子问题用 deterministic solver。
- 对“需要理解”的子问题只生成 evidence/candidate，不直接给最终结论。
- 每个 solver 都暴露 confidence、trigger_signals、uncertain_items。
- 当 confidence 不足或 answer_is_exact 为 false 时，可以让 LLM review。

task 361 这类“活跃且情绪判断”任务就是典型例子：`speaker_activity` 能统计消息数，但不能直接决定“最相关的 10 人”。所以我没有继续写 deterministic ranker，而是让 Python 提供 speaker counts 和 sentiment evidence，由 LLM 做最后判断。

## 8. Verifier 与 bounded repair

### 8.1 Task constraints

`clbench_life_harness.constraints` 会从 task 文本中提取硬约束：

- 是否要求 JSON/table/list；
- 是否要求 quote；
- 是否有 required count；
- 输出是否应避免 invented usernames；
- 是否有 must-include facets。

这些不是 rubrics，但能抓到很多“提交前显然不合格”的输出。

### 8.2 Hard verifier

`verify_answer_hard` 做以下检查：

| 检查 | 目的 |
| --- | --- |
| `non_empty_answer` | 防止空输出 |
| `valid_json` / `markdown_table_present` / `list_items_present` | 检查格式 |
| `required_count_met` | 检查要求数量 |
| `quotes_present` / `quotes_verbatim_in_context_or_state` | 防止伪造引用 |
| `usernames_grounded` | 防止发明用户 |
| `numbers_have_structured_context` | 数字是否有结构化依据 |
| `workflow_selected` | 是否落到过宽泛 workflow |
| `evidence_or_candidate_state_present` | 是否有中间状态 |
| `top_speakers_covered` | speaker activity 风险提示 |
| `sentiment_evidence_not_nei` | 有证据却写 NEI 的风险 |
| `commenter_registry_covered` | 评论者 registry 是否覆盖 |
| `subscription_cadence_plan_covered` | 订阅计划项是否覆盖 |
| truncation checks | 检查输出是否疑似截断 |

hard verifier 不是官方 judge，也不尝试替代 GPT-5.1 rubrics。它的定位是提交前检查硬错误，并为失败归因提供线索。

### 8.3 Bounded answer repair

早期 repair 只允许一次。后来考虑到一次修复可能解决格式后又暴露漏项，所以改成有上限循环：

- 参数：`--answer-repair-max-attempts`，默认 2。
- 每轮 repair 后重新运行 verifier。
- 如果没有 repairable failures，就停止。
- 如果 repair 返回空输出或框架错误，也停止。
- trace 记录每轮失败项、prompt 长度、attempts 和 stop reason。

重要的是，repair 不允许变成无限 agent loop。它只修硬约束，不强行替代语义判断。

对 soft rank/sentiment 任务还有特殊处理：

- `top_speakers_covered` 只作为 warning，不强制把答案改成 raw message count top 10。
- `speaker_sentiment_evidence` 是 candidate evidence，不是固定答案。
- 如果某个 speaker 有情绪证据但答案写 NEI，repair 可以要求替换为 1-5 rating。

## 9. Prompt 与 LLM 调用配置

### 9.1 Prompt 结构

最终 answer prompt 包含：

1. system instruction；
2. final task；
3. task constraints；
4. workflow structured state；
5. deterministic findings/candidate answers；
6. evidence pack；
7. output requirements。

prompt 中特别强调：

- 不要编造没有证据的事实；
- quote 必须来自原文；
- 对 exact count/calculation 使用 structured state；
- 对 `commenter_attribution` 使用 `answer_post_count`；
- 对 `speaker_sentiment_evidence` 只当候选证据，不当固定排序。

### 9.2 推理配置

早期实验发现，模型会出现 reasoning-only 空输出：`content_len=0`，但 `reasoning_content_len` 很长，并且 completion tokens 被耗尽。这不是写文件或解析问题，而是模型服务返回了大量 reasoning 但没有 final content。

针对这个问题，默认配置改为：

- `thinking_mode=off`；
- prompt 前加 `/no_think`；
- `max_tokens=12000`；
- `request_timeout` 可设为 900；
- `model_call_attempts` 支持多次；
- 指数退避：`retry_backoff_base`、`retry_backoff_max`；
- trace 记录 content/reasoning 长度、usage、truncation reason。

这个改动的目标不是提升单题语义质量，而是先让失败可见、可复现，避免“等很久后空输出但不知道原因”。

## 10. 可观测性设计

每道题都有 trace，主要内容包括：

- `task_route`：第一层数据形态路由；
- `workflow_decision`：第二层 workflow；
- `task_constraints`；
- chunk 数、原始 retrieval hits、neighbor-expanded hits；
- `structured_state`；
- deterministic answer contract；
- query planner prompt/attempts/status/raw preview；
- LLM 原始返回字段摘要；
- truncation diagnostics；
- hard verification；
- answer repair rounds；
- latency 和 token usage。

这让 dev 分析可以回答老师提出的几个问题：

- 失败是在上下文定位吗？看 retrieval hits 和 query evidence。
- 失败是在事实抽取吗？看 structured tables 是否缺字段。
- 失败是在计算吗？看 computed values 和 deterministic parser。
- 失败是在格式吗？看 constraints/verifier。
- 失败是在最终校验吗？看 hard verification 和 judge failed rubrics。
- 修改是否覆盖一类任务？看触发信号、workflow 和回归测试，而不是看 task id。

## 11. 迭代改进记录

下面按关键版本列出每次改动的动机、改动、预期影响和验证方式。

| 版本 | 动机 | 改动 | 预期影响 | 验证 |
| --- | --- | --- | --- | --- |
| v1.4 | 多题出现长时间等待和空输出 | 关闭 thinking、加入 `/no_think`、记录 reasoning/content 长度 | 区分模型调用异常和任务复杂 | task 221 复测，trace 可见 reasoning-only |
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

## 12. Dev 实验结果

### 12.1 45 题分层实验

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

解释：task-level pass rate 低，但 rubric-level pass rate 中等，说明很多答案不是完全不会，而是漏掉若干 rubrics 或格式/字段不严格。后续优化应优先补中间态和 final validation，而不是继续增加 prompt 长度。

### 12.2 代表性 4 题实验

选择 task 59、144、339、361 作为代表性失败/目标：

| task | 类型 | 结果 | 说明 |
| --- | --- | --- | --- |
| 59 | thread commenter registry | `11/11` | commenter attribution + renderer 生效 |
| 144 | multi-doc exact list | `8/20` | canonical item matrix 仍不足，exact list renderer 待加强 |
| 339 | Subscribe & Save calendar | `19/19` | cadence plan + deterministic calendar renderer 生效 |
| 361 | speaker activity/sentiment | `25/36` | 全局 speaker 表有帮助，但 fuzzy rank 仍依赖语义判断 |

总体：`63/86` rubrics，`2/4` tasks solved。

这组实验帮助确认了系统边界：

- 可确定的日历/评论归因问题能靠 workflow tools 明显提升。
- 多文档 exact-list 需要更严格 canonical renderer。
- 模糊 rank/sentiment 不适合靠 deterministic 排序硬修。

## 13. 具体失败根因与修复案例

### 13.1 task 339：Subscribe & Save calendar

失败现象：

- 模型倾向于回答“信息不足”或漏掉具体商品 cadence。
- 早期通过 rubrics 很少。

根因定位：

- 上下文定位不是主要问题，订单记录存在。
- 失败发生在事实抽取和中间状态：order history 没有变成 recurring item/cadence table。
- 最终生成阶段也会漏掉必须出现的商品。

修复：

- `recurring_items` 提取重复商品；
- `subscription_cadence_plan` 生成每个商品推荐周期；
- `planning_calendar` 生成未来配送月份；
- `subscribe_save_calendar_renderer` 直接输出计划。

验证：

- GPT-5.1 judge：`19/19` rubrics，全通过。

泛化性：

- 针对的是“订单历史 -> 周期计划”这类任务形态。
- 风险是部分 cadence 规则仍依赖商品名/包装规格，未来应改成更通用的间隔估计。

### 13.2 task 59：thread commenter registry

失败现象：

- 模型错把正文短语当 commenter。
- 对 `thomas oak` 和 `Michael Brown` 这类误归因/别名处理不稳定。
- post count 使用 displayed count 而不是 answer-facing canonical count。

根因定位：

- 失败发生在事实抽取和归因，不是最终表达。
- thread parser 对 header 边界过宽。

修复：

- 收紧 commenter candidate 过滤；
- 保留合法 ellipsis display name；
- 增加 `commenter_registry`；
- 增加 `commenter_attribution`；
- renderer 使用 `answer_post_count`。

验证：

- task 59 GPT-5.1 judge：`11/11`。

泛化性：

- 适用于评论区 registry、别名、误归因类任务。
- 不直接解决所有 stance/reasoning 任务；这些仍需 LLM 判断。

### 13.3 task 361：speaker activity and sentiment

失败现象：

- 模型能给出表，但 top users 和 sentiment rating 部分不符合 rubrics。
- 如果按 raw message count 强修，可能替换掉有语义相关证据的 speaker。

根因定位：

- 需要区分“最活跃”和“最活跃且与盈利/话题相关”。
- 这是语义 rank 问题，不是纯计数问题。

修复方向：

- 增加 `speaker_activity`，给出全局 message count。
- 增加 `speaker_sentiment_evidence`，给出候选 quote、polarity、rating_hint。
- repairer 不再强制按 raw count top 10 改写用户集。
- 对有 evidence 却写 NEI 的情况进行软修复。

验证：

- 代表性 run 中该任务仍未全通过，但 rubric 得分保持较高。
- 结论是方向正确，但 dialogue state 仍需更强的 entity/open-loop/role 建模。

### 13.4 task 144：multi-doc exact list

失败现象：

- 输出中混入 extras；
- 漏掉 required items；
- category/section drift。

根因定位：

- 多文档 exact-list 需要 canonical item state。
- LLM query planner 和 repair 有时会让模型重写列表，反而破坏精确性。

修复：

- 对 exact-list multi-doc 默认关闭 query planner。
- 对 exact-list multi-doc 默认不做开放式 LLM repair。
- 增加 `canonical_item_matrix`，但 renderer 仍不够强。

后续方向：

- 给 multi-doc exact-list 加 strict schema renderer。
- 对每个 item 记录 section/source/status，最后按 schema 输出，而不是让 LLM自由组织。

## 14. 泛化能力与风险控制

### 14.1 为什么不是 task-id 硬编码

当前主要触发条件来自：

- 数据格式标记；
- metadata context_subcategory；
- task operator；
- workflow state 是否存在；
- parser confidence；
- query planner policy。

例如 Subscribe renderer 依赖 `subscription_cadence_plan` 和 `planning_calendar`，不是依赖 task id；thread renderer 依赖 `commenter_attribution`，不是依赖某个固定问题编号。

### 14.2 泛化风险

仍然存在以下风险：

- 某些 parser 对格式变化敏感。
- 商品 cadence 规则可能对 dev 样例适配较多。
- query planner 可能在 semantic task 上增加噪声证据。
- hard verifier 对 rubrics 的覆盖有限，只能查硬错误。
- 多文档 canonical list 还不够强。

### 14.3 风险缓解

已采取的缓解机制：

- 每个 deterministic answer 记录 trigger 和 confidence。
- low confidence 时 fallback 给 LLM。
- `uncertain_items` 进入 trace。
- query planner 有 workflow policy gate。
- repair 有最大次数，且不强修 soft rank。
- tests 覆盖 parser、workflow、planner gating 和 verifier。

## 15. 复现方式

### 15.1 环境变量

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

### 15.2 本地检查

```powershell
.\NexAU-main\.venv\Scripts\python.exe -m compileall .\clbench_life_harness .\NexAU-main\examples\clbench_life
```

```powershell
.\NexAU-main\.venv\Scripts\python.exe -m pytest .\tests\test_structured_solvers.py -q
```

当前结果：`22 passed`。

### 15.3 Dev 45 题实验

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

### 15.4 Test 输出命令

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

如果中断，追加：

```powershell
--resume
```

## 16. 总结

本次优化把 CL-bench Life harness 从单次生成式调用推进为 workflow-based system：

- 路由层识别数据形态和任务操作；
- workflow 层构造可检查的中间态；
- deterministic 工具解决可验证的计数、日历和结构化日志问题；
- query planner 用有限 LLM 调用补充语义证据；
- hard verifier 和 bounded repair 在最终输出前检查硬约束；
- trace 记录让失败可以被归因、复现和回滚。

当前版本已经解决了空输出不可观测、长日志心算、订阅周期、评论者归因等问题，但多文档 exact-list 和复杂 dialogue semantic state 仍是主要短板。下一步最值得做的是增强 `multi_doc_matrix_workflow` 的 strict canonical renderer，以及增强 `dialogue_state_workflow` 的 entity/open-loop/role state，而不是继续堆更长 prompt。
