# CL-bench Life Harness 优化报告

## 1. 目标与约束

本项目目标是在不修改基础模型 `nex-agi/Nex-N2-Pro`、不修改 dev/test 原始数据和 rubrics 的前提下，优化 CL-bench Life 的 agent harness。评分关注点不仅是 test pass rate，还包括 dev set 错误分析质量、harness 设计质量、泛化能力和工程交付。

我采用的核心思路是：把 harness 从“一次性长 prompt 调用”改造成一套可观测、可调试、可验证的 workflow 系统。CL-bench Life 的题目通常不难读，但上下文长、碎、乱，失败常发生在上下文定位、事实抽取、计算、格式约束和最终校验，而不是单纯发生在模型表达能力上。

## 2. 任务特点分析

CL-bench Life dev set 包含 9 类生活上下文任务，典型特点如下：

| 类别 | 常见上下文 | 主要风险 |
| --- | --- | --- |
| Game Logs | 游戏日志、PGN、RSS-like 事件流 | 长日志计数、重复事件、XML/文本混合解析 |
| Digital Footprints & Daily-Life Records | 浏览记录、观看历史、日常轨迹 | 时间线定位、主题聚类、噪声过滤 |
| Self-Tracking Trajectories | 订单、健康、训练、订阅记录 | 周期推断、数值计算、计划表生成 |
| Community Interactions | Reddit/论坛/评论树 | 用户归属、楼层/时间戳、别名和误归因 |
| Group Conversations & Meeting Transcripts | 群聊、会议、多人协作 | 全局 speaker 统计、角色/情绪/立场判断 |
| Private Conversations | 私聊、亲友/工作沟通 | 含蓄意图、冲突点、承诺和未闭环事项 |
| Creation & Revision Histories | 草稿、修订、版本记录 | 版本差异、保留/删除项、候选项混淆 |
| Personal Information Fragments | 个人信息碎片 | 实体合并、亲属关系、时间和条件约束 |
| Public Information Fragments | 公开资料碎片 | 跨文档 claim 对齐、候选过滤、证据来源 |

因此，harness 需要同时解决两类问题：

- 硬问题：计数、日期、金额、商品频率、列表格式、引用是否来自原文。
- 软问题：模糊排序、情绪强度、人物动机、语义归纳、边界案例判断。

硬问题适合 Python 确定性工具；软问题仍应交给 LLM，但要给它更可靠的候选证据和中间状态。

## 3. Harness 总体设计

当前系统流程为：

```text
parse_task
  -> route_context_type / route_task_operator
  -> select_workflow
  -> build_structured_state
  -> optional LLM query planner
  -> render evidence pack
  -> deterministic answer or LLM answer writer
  -> hard verifier
  -> bounded answer repair
  -> trace / prediction JSONL
```

### 3.1 四类 workflow

我没有把 9 个类别写成 9 个孤立 solver，而是抽象成 4 类可复用 workflow：

| Workflow | 覆盖类别 | 中间态 |
| --- | --- | --- |
| `structured_log_workflow` | 游戏、浏览记录、自我追踪 | event table、aggregates、computed values、recurring items |
| `thread_tally_workflow` | 社区互动 | comments、commenter registry、stance/tally、attribution |
| `dialogue_state_workflow` | 群聊、私聊、会议 | turns、speaker activity、speaker sentiment evidence、timeline |
| `multi_doc_matrix_workflow` | 创作修订、个人碎片、公开碎片 | documents、claim matrix、candidate matrix、canonical item matrix |

这种设计的好处是：失败可以被定位到“路由错了、表没抽出来、表抽对但最终答案漏项、还是格式校验失败”，而不是只能看到一个低分输出。

### 3.2 结构化中间态

`clbench_life_harness.workflows` 中的 `StructuredState` 统一保存：

- workflow 决策；
- evidence table；
- computed values；
- entity profiles；
- timeline；
- candidate answers；
- uncertain items；
- parser confidence；
- workflow-specific tables。

新增或强化的关键表包括：

- `speaker_activity`：从完整对话中统计 speaker 消息数，避免只按检索片段判断活跃度。
- `speaker_sentiment_evidence`：为情绪/态度任务提供候选证据，但不强行确定最终排序。
- `commenter_registry` / `commenter_attribution`：处理评论区展示名、别名、误归因和发帖次数。
- `recurring_items` / `subscription_cadence_plan` / `planning_calendar`：处理订阅、订单周期和配送计划。
- `canonical_item_matrix`：为多文档精确列表任务保留候选项和来源。

### 3.3 Deterministic solver 的边界

我只对可验证子问题使用确定性解法，例如：

- PGN draw comparison；
- RuneScape Adventurer log 计数；
- watch history pivot；
- Subscribe & Save calendar；
- thread commenter registry。

这些 solver 都有触发信号、parser confidence、parsed state summary、uncertain items 和 fallback 字段。这样可以避免“看起来像高分，其实是针对 task id 写补丁”的风险。

对模糊 rank、情绪强度、人物动机等问题，Python 只生成候选表和证据，由 LLM 做最终语义判断。

### 3.4 LLM query planner 与 repairer

为了避免纯 Python 预处理过度僵硬，也避免无约束 agent 循环太慢，当前采用有边界的 LLM 辅助：

- `query_planner.py` 要求 LLM 输出严格 JSON query plan。
- Python 执行查询、取邻近 chunk、去重，并将结果合并进 structured state。
- planner 只在语义证据检索可能有收益的 workflow 中启用；对多文档精确列表、确定性答案和结构化日志默认关闭。
- `answer_repair` 是有最大次数限制的最终修复，默认最多 2 次。
- repair 只处理硬校验发现的格式、漏项、NEI 与证据冲突等问题；对模糊 rank 不强行按确定性表重排。

### 3.5 推理配置

早期实验发现模型会出现 reasoning-only 空输出：`content_len=0`，但 `reasoning_content_len` 很长，并消耗大量 completion tokens。后来默认使用：

- `thinking_mode=off`；
- prompt 级 `/no_think`；
- `max_tokens=12000`；
- 长超时和指数退避重试；
- trace 记录 content/reasoning 长度、usage、truncation reason。

这个配置解决了空输出不可见的问题，也让后续失败更多暴露为 workflow/抽取/校验问题，而不是模型调用异常。

## 4. 可观测性与可验证性

每道题都会写出 trace，主要字段包括：

- 原始任务解析结果；
- route 和 workflow 决策；
- chunk 和 neighbor chunk 命中；
- structured state；
- query planner 输入输出；
- final prompt 摘要；
- LLM 原始返回摘要；
- hard verification；
- repair rounds；
- latency、token usage、错误记录。

`clbench_life_harness.verifier.verify_answer_hard` 目前覆盖：

- 空输出；
- JSON/table/list 形状；
- required count；
- exact quote grounding；
- username grounding；
- 数字/状态一致性 warning；
- suspected truncation；
- workflow-specific completeness warning。

这些 checker 不替代官方 rubrics judge，但能在没有 test rubrics 的情况下提前发现格式、漏项和明显幻觉。

## 5. Dev 实验与错误分析

### 5.1 分层 45 题实验

为了避免只测少数样例，我构造了 9 个官方子类各 5 题的 dev 样本，使用 GPT-5.1 judge 近似官方评测。

Prediction 结果：

- 45 题全部处理完成；
- `error_records=0`；
- 无空输出；
- 无 suspected truncation；
- 平均延迟 `48.62s`。

Judge 结果：

- solved tasks：`6/45`；
- rubric pass rate：`64.71%`；
- parse errors：`0`。

失败类型统计：

| Failure type | Count |
| --- | ---: |
| final_validation | 36 |
| fact_extraction | 32 |
| reasoning | 19 |
| context_location | 15 |
| format | 13 |
| calculation | 1 |

结论：当前主要瓶颈不是空输出或 API 调用失败，而是中间状态还不够强，导致答案经常“部分正确但漏 rubric”。

### 5.2 代表性改进

#### Subscribe & Save calendar

问题：task 339 初始回答倾向于说信息不足，只通过少量 rubric。根因不是 prompt 不够长，而是 order history 没有被转换成商品周期、用量和日历计划。

改动：

- 增加 `recurring_items`；
- 增加 `subscription_cadence_plan`；
- 增加 `planning_calendar`；
- 用 deterministic renderer 输出配送计划。

结果：GPT-5.1 judge 从低分提升到 `19/19` rubrics，全通过。

#### Thread commenter registry

问题：task 59 中评论区展示名、别名和误归因混在一起，模型容易错算用户和发帖次数。

改动：

- 加强评论 header 解析；
- 过滤伪 commenter；
- 建立 `commenter_registry` 和 `commenter_attribution`；
- renderer 使用 `answer_post_count` 区分展示名出现次数和答案归属。

结果：GPT-5.1 judge 达到 `11/11` rubrics，全通过。

#### Long structured logs

问题：部分长日志任务会触发长 reasoning 或空输出。

改动：

- 对 RuneScape RSS/text log、PGN、watch history 等可结构化日志使用 parser；
- parser 输出 counts、events、uncertain items；
- 高置信时直接生成 deterministic answer。

结果：相应 dev 样例从空输出或低分变为稳定输出，并能通过回归测试保护。

### 5.3 仍然存在的失败

#### 多文档精确列表

task 144 类任务需要恢复精确 recipe/category/item list。当前 `canonical_item_matrix` 有帮助，但最终答案仍可能混入额外项、漏 required items 或 category drift。这里不适合开放式 repair 反复改写，更适合增强 canonical renderer 和 strict schema。

#### 模糊活跃度与情绪排序

task 361 类任务包含“最活跃用户”和 sentiment rating。`speaker_activity` 能给出全局计数，但“活跃且相关”不等于纯消息数排序。这里不能用 deterministic 方法强行排好，而应该让 LLM 根据候选表、上下文证据和题目语义做判断。当前改动已经把 repairer 从强制 top-count 修正改成只检查硬约束。

#### 语义关系和隐含意图

私聊、会议和人物关系题仍需要更强的 entity state 和 open-loop tracking。Python 可以抽取 turns、speaker 和显式承诺，但隐含支持、让步、冲突缓和等仍依赖 LLM。

## 6. 泛化性说明

本项目避免直接按 task id 硬编码。已经实现的改动主要按任务形态触发：

- 结构化日志：根据 PGN tags、RSS/XML、事件行、订单记录等数据形态触发。
- 社区评论：根据评论 header、时间戳、展示名和回复结构触发。
- 对话任务：根据 speaker turns、群聊/私聊结构触发。
- 多文档任务：根据文档边界、标题、section 和候选项触发。

确定性工具的泛化边界是“可验证子问题”。当问题涉及语义判断时，工具只提供候选证据和硬事实，不直接替代 LLM。

潜在风险：

- 结构化表过长会挤占 prompt 空间。
- parser 过于自信可能误导模型。
- repairer 可能把软判断误当硬约束。
- 针对 dev 失败写的 renderer 仍需要更多同类样本验证。

为降低这些风险，trace 中保留 parser confidence、uncertain items、fallback recommendation 和 repair rounds，便于后续回滚或调参。

## 7. 工程交付与复现

### 7.1 本地检查

```powershell
.\NexAU-main\.venv\Scripts\python.exe -m compileall .\clbench_life_harness .\NexAU-main\examples\clbench_life
```

```powershell
.\NexAU-main\.venv\Scripts\python.exe -m pytest .\tests\test_structured_solvers.py -q
```

当前结果：`22 passed`。

### 7.2 Dev 实验

45 题分层 dev prediction：

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

GPT-5.1 judge：

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

### 7.3 Test 输出

官方 test set 无 rubrics，本地只能生成提交文件和 trace：

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

如果运行中断，追加 `--resume`。

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

## 8. 总结

本次优化的重点不是写更长 prompt，而是把 harness 拆成可观察、可定位、可验证的系统组件。当前系统已经能稳定生成输出、记录中间态、区分模型调用问题和 workflow 问题，并在若干可结构化任务上通过确定性工具显著提升。

最重要的经验是：CL-bench Life 的失败常常不是单点失败，而是链式失败。上下文定位漏一条、结构化表少一个实体、最终答案漏一列，都会导致整题不通过。因此，harness 的价值在于让每一步都能被检查，并让改动能覆盖一类任务，而不是只修一个样例。

后续最有价值的方向是继续增强 `multi_doc_matrix_workflow` 的 canonical renderer，以及 `dialogue_state_workflow` 的 entity/open-loop state；这两类仍是当前 dev 失败的主要来源。
