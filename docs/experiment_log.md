# CL-bench Life Harness Experiment Log

本日志用于保留过程记录：每次修改、使用的数据、运行结果、观察到的影响和下一步决策。它不是最终报告，但会直接服务于 `report.md`。

## v0.1 - Observable Parsing / Chunking / Retrieval Baseline

时间：2026-06-05

修改：

- 新增 `clbench_life_harness/`。
- 实现 JSONL 读取、`<|TASK|>` 拆分、多轮最终问题识别。
- 实现稳定分块，记录 chunk id、行号、字符偏移、文档标签、说话人、时间戳。
- 实现轻量关键词检索。
- 为每题生成 trace，记录任务类型、chunk 数、top hits、rubric 样例。

数据：

- Dev：`CL-Bench-dataset/CL-bench%20Life.jsonl`
- Test：`实训题目/CL-bench-Life-test.jsonl`

运行：

```powershell
python -m clbench_life_harness.run_baseline --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" --output-dir ".\runs\baseline_dev_full"
python -m clbench_life_harness.run_baseline --input ".\实训题目\CL-bench-Life-test.jsonl" --output-dir ".\runs\baseline_test_full"
```

结果：

- Dev 405 条处理完成，平均 26.67 chunks，最大 604 chunks。
- Test 30 条处理完成，平均 85.6 chunks，最大 1750 chunks。

观察：

- 第一版可以暴露检索候选证据，但不能判断任务成功。
- Test 第 7 题 Apple Health 步数记录出现 chunk 爆炸，说明固定 overlap 对长表格/日志不稳定。

影响：

- 建立了可观测基础，但分块策略需要修复。

## v0.2 - Chunk Overlap Fix

时间：2026-06-05

修改：

- 将分块 overlap 从单纯行数限制改为行数 + 字符预算双约束。

运行：

```powershell
python -m clbench_life_harness.run_baseline --input ".\实训题目\CL-bench-Life-test.jsonl" --output-dir ".\runs\baseline_test_full_v2"
```

结果：

- Test 平均 chunk 数从 85.6 降至 35.97。
- Test 最大 chunk 数从 1750 降至 293。

观察：

- 分块爆炸得到修复。
- 第 7 题 top_score 仍很低，说明步数题不适合靠普通关键词检索，应走确定性解析。

影响：

- 减少 token 浪费和 trace 噪声。
- 下一步转向确定性计算工具。

## v0.3 - NexAU Runner + Deterministic Step Analyzer

时间：2026-06-05

修改：

- 在 `NexAU-main/examples/clbench_life/` 新增 NexAU 版 runner。
- 新增 `systemprompt.md`，要求模型基于 evidence pack 和 deterministic findings 回答。
- 新增 `run_predictions.py`，支持 prediction JSONL、逐题 trace、并发控制和 `--dry-run`。
- 新增 Apple Health step count 确定性解析，按天和按月汇总步数。

运行：

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\实训题目\CL-bench-Life-test.jsonl" `
  --output ".\runs\nexau_test_task7_dry\predictions.jsonl" `
  --trace-dir ".\runs\nexau_test_task7_dry\traces" `
  --start 7 `
  --limit 1 `
  --dry-run
```

结果：

- NexAU runner dry-run 成功。
- 第 7 题 prompt 包含确定性解析结果：1756 条 step-count records，82 天，日期范围 2026-01-01 到 2026-03-23。
- 月均步数单调上升：2026-01 为 1250.3，2026-02 为 1949.3，2026-03 为 2008.7。

观察：

- 这类自我追踪任务应由 Python 先做确定性计算，再让模型解释趋势。
- 当前还未真实调用 LLM，因为环境中未检测到 `LLM_API_KEY`。

影响：

- Harness 开始体现“确定性计算 + 证据回答”，不是单纯长 prompt。
- 下一步应接入 LLM，跑少量 dev smoke prediction，并根据 rubrics/trace 标注失败类型。

## v0.4 - Stratified Dev Sampling + Approximate Rubric Judge Skeleton

时间：2026-06-05

修改：

- 新增 `clbench_life_harness/sample_dev.py`，支持按官方 `context_subcategory` 分层抽样。
- `run_baseline.py` 和 NexAU `run_predictions.py` 支持 `--ids-file`，可处理非连续原始 row id。
- 新增 `NexAU-main/examples/clbench_life/judge_rubrics.py`，用于 dev prediction 的近似 rubric 自动评估。
- README 更新为先生成 9 子类分层样本，再跑 smoke prediction。

运行：

```powershell
python -m clbench_life_harness.sample_dev `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --ids-output ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --summary-output ".\runs\dev_samples\summary_1_per_subcategory.json" `
  --per-subcategory 1 `
  --prefer-task-type-diversity

.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_stratified_dry\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_stratified_dry\traces" `
  --ids-file ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --dry-run

.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\judge_rubrics.py `
  --dev-input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --predictions ".\runs\nexau_dev_stratified_dry\predictions.jsonl" `
  --output ".\runs\nexau_dev_stratified_dry\rubric_judge_dry.jsonl" `
  --ids-file ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --max-rubrics-per-task 2 `
  --dry-run
```

结果：

- 生成 9 个样本，覆盖全部官方子类。
- 选中 ids：30, 71, 88, 156, 217, 221, 238, 252, 392。
- NexAU runner dry-run 成功生成 9 条 prediction 占位和 trace。
- rubric judge dry-run 成功对齐 dev rubrics 和 prediction。

观察：

- 小规模实验不应使用 first-N；first-N 容易只覆盖少数数据形态。
- Rubric 自动评估可以实现，但它是近似 judge，不等同官方 GPT-5.1 high reasoning effort 判分。

影响：

- 下一次真实 LLM 实验可以覆盖每个官方子类，错误分析更有代表性。
- judge 脚本可用于快速定位失败类型，但报告中应明确它是辅助分析工具。

## v0.5 - First LLM Smoke Attempt Debugging

时间：2026-06-05

现象：

- 用户运行 9 子类 smoke prediction 后，命令只打印 `finished task 30`，随后手动中断。
- 输出多次 `Structured tool call mode enabled but no structured tool definitions were provided.`
- Trace 文件显示 9 个任务的 prediction 都是 `[Error: Maximum iteration limit reached.]`。
- 因脚本原先只在全部 futures 完成后统一写 `predictions.jsonl`，中断后没有最终 prediction 文件。

根因：

- Answer agent 的 `max_iterations=1` 太低，NexAU 执行循环触发最大迭代限制。
- 当前 agent 没有工具，但使用 `tool_call_mode="structured"`，产生无工具 structured warning。
- 预测结果缺少增量落盘和恢复能力。

修改：

- `run_predictions.py` 的 answer agent 改为 `max_iterations=3`。
- Answer agent 和 rubric judge 的 `tool_call_mode` 改为 `xml`，避免无工具 structured warning。
- `run_predictions.py` 增加 `predictions.jsonl.partial` 增量写入。
- 增加 `--resume`，可从已有 final/partial prediction 继续。
- 最终输出仍按输入任务顺序写入 `predictions.jsonl`。

验证：

```powershell
python -m compileall .\NexAU-main\examples\clbench_life

.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_stratified_dry_v2\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_stratified_dry_v2\traces" `
  --ids-file ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --dry-run
```

结果：

- Dry-run v2 生成 9 条 final predictions 和 9 条 partial records。
- 输出 id 顺序为：30, 71, 88, 156, 217, 221, 238, 252, 392。

影响：

- 真实 LLM 实验可以恢复，不再因为中断丢失已完成任务。
- 需要重新跑真实 smoke；旧目录 `runs/nexau_dev_smoke/traces` 中的 prediction 是最大迭代错误，不应用于分析模型能力。

## Next Planned Experiment

目标：

- 接入基础模型 `nex-agi/Nex-N2-Pro`，先跑官方 9 个子类的分层 dev smoke sample。
- 保存 prediction、prompt trace、evidence pack、latency。
- 人工/半自动根据 rubrics 标注失败类型：上下文定位、事实抽取、推理、计算、格式、最终校验。

建议命令：

```powershell
python -m clbench_life_harness.sample_dev `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --ids-output ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --summary-output ".\runs\dev_samples\summary_1_per_subcategory.json" `
  --per-subcategory 1 `
  --prefer-task-type-diversity

.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_smoke\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_smoke\traces" `
  --ids-file ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --concurrency 1
```

## v0.6 - Pre-Experiment Observability and Deterministic Game Aggregation

时间：2026-06-05

触发：
- `runs/nexau_dev_one_v2` 中 task 30 的 `prediction` 为空字符串。
- 需要区分模型可见 `content` 为空、NexAU 解析丢失、还是写入阶段错误。
- task 30 是“按周聚合 OP.gg 游戏统计”的结构化计算题，适合先由 Python 做确定性中间态。

修改：
- `run_predictions.py` 增加 LLM after-model observation hook，trace 中记录 `content_len`、`reasoning_content_len`、`tool_calls_count`、`usage` 和 raw message 安全字段摘要。
- `run_predictions.py` 增加 `--allow-empty`，默认不允许空 prediction；空输出会先写 trace，再抛错，避免把失败静默写入 final predictions。
- `run_predictions.py` 增加 `--max-iterations`，当前无工具版本默认 `1`，降低无意义 agent 循环。
- `deterministic.py` 增加 `game_weekly_stats`，解析 JSON 游戏记录，按 Monday-start week 聚合胜负、类别字段、数值总和和均值。

验证：
```powershell
.\NexAU-main\.venv\Scripts\python.exe -m compileall .\clbench_life_harness .\NexAU-main\examples\clbench_life

.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_task30_dry_v3\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_task30_dry_v3\traces" `
  --start 30 `
  --limit 1 `
  --dry-run
```

结果：
- compileall 通过。
- dry-run 成功生成 `runs/nexau_dev_task30_dry_v3`。
- task 30 trace 中出现 `game_weekly_stats`：解析 100 条游戏记录，覆盖 13 个 Monday-start weeks，日期范围 2025-12-02 到 2026-02-25。
- prompt chars 约 13252，包含确定性周聚合表。

影响：
- 第一次真实 LLM 实验如果再次出现空输出，将能在 trace 里看到可见 content、reasoning 字段长度和 raw message 字段形状。
- 结构化统计题不再完全依赖模型读长 JSON 和心算，降低计算错误风险。
- 旧的 `runs/nexau_dev_one_v2` 不能作为有效实验结果，只能作为失败诊断样例。

## v0.7 - Dev Smoke v3 Invalid Run and Iteration Guard

时间：2026-06-05

运行结果：
- 目录：`runs/nexau_dev_smoke_v3`
- `run_summary.json` 显示 processed=9，avg_latency_sec=0.65。
- 9 条 prediction 全部为 `\n\n[Error: Maximum iteration limit reached.]`。
- 9 个 task trace 的 `llm_observations` 都为空，说明没有进入 after-model observation hook。

根因：
- `--max-iterations 1` 在 NexAU async executor 中会在首次模型调用前触发 `MAX_ITERATIONS_REACHED` 保护。
- 因此 v3 不是一次有效 LLM 能力实验，不能用于 rubric 分析。

修改：
- `run_predictions.py` 默认 `--max-iterations` 从 1 改回 3。
- 增加 `--allow-framework-error`，默认不允许将 NexAU 框架错误字符串作为成功 prediction 写入。
- trace 的 `model_run` 增加 `framework_error` 字段。

下一步：
- 使用新目录重跑，例如 `runs/nexau_dev_smoke_v4`，命令中使用 `--max-iterations 3`。

## v0.8 - Output Budget and Truncation Observability

时间：2026-06-05

触发：
- `runs/nexau_dev_one_v4` 的 task 30 已能正常调用模型并返回可见内容。
- Trace 显示 `completion_tokens=6000`，刚好撞上旧的 `--max-tokens 6000`。
- 同一条响应里 `reasoning_content_len=13239`，可见答案只有 1180 字符，并且表格在末尾被截断。
- 这说明 v4 的主要失败不是上下文定位或接口问题，而是输出预算和 thinking 占用导致的最终答案截断。

修改：
- `run_predictions.py` 默认 `--max-tokens` 从 6000 提高到 32000。
- 增加 `--request-timeout`，默认 900 秒，避免长输出/长 reasoning 被 300 秒固定超时误杀。
- 增加 `--thinking-mode {auto,off,budget}` 和 `--thinking-budget`，通过 `extra_body` 透传 Qwen-compatible thinking 控制参数。
- Trace 的 `model_run` 增加 `max_tokens_requested`、`request_timeout_sec`、`thinking_mode`、`thinking_request`。
- 增加疑似截断检测：当 `completion_tokens >= max_tokens`、代码块未闭合、或末尾 Markdown 表格行明显不完整时，默认先写 trace 再报错。
- 增加 `--allow-truncation`，只在明确需要保留截断样例时使用。

验证：
```powershell
.\NexAU-main\.venv\Scripts\python.exe -m compileall .\NexAU-main\examples\clbench_life\run_predictions.py

.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_task30_dry_v5\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_task30_dry_v5\traces" `
  --start 30 `
  --limit 1 `
  --dry-run `
  --thinking-mode off
```

结果：
- `compileall` 通过。
- dry-run 生成 `runs/nexau_dev_task30_dry_v5`。
- `run_summary.json` 显示 `max_tokens=32000`、`request_timeout=900.0`、`thinking_mode=off`。
- task 30 trace 的 `model_run` 正确记录 `thinking_request={"enable_thinking": false}`，且 dry-run 未误报截断。

影响：
- 下一次真实 LLM 实验应优先使用新目录，例如 `runs\nexau_dev_one_v5`。
- 对 task 30 这类长表格任务，建议先跑 `--thinking-mode off --max-tokens 32000`；如果接口不接受该扩展参数，再退回 `--thinking-mode auto`。
- 这一步把“最终输出是否完整”纳入 harness 观测面，避免把半截答案当作有效实验样本。

## v0.9 - Task 30 v5 Real Run Inspection

时间：2026-06-05

运行目录：
- `runs\nexau_dev_one_v5`

结果：
- 真实模型调用成功，`processed=1`，`avg_latency_sec=42.33`。
- Trace 显示 `max_tokens=32000`，`completion_tokens=2692`，未撞上输出上限。
- `prediction_chars=3227`，`empty_prediction=false`，`framework_error=false`，`suspected_truncation=false`。
- 实际运行配置为 `thinking_mode=auto`，不是 `off`。

人工检查：
- 预测覆盖了 13 个 Monday-start weeks，数据行数量正确。
- 表格中各周 `Games` 求和为 100，满足第一条 rubric。
- 表格包含 wins/losses、kills、deaths、assists、CS、gold、damage to champions、damage taken、vision score。
- 但 dev rubrics 明确要求每周包含 `total duration in minutes` 和 `total duration in seconds`，v5 表格没有 duration 列。
- 预测末尾 notes 提到 weekly averages include duration，但正文表格并未提供 duration，因此属于输出 schema 漏项。

判断：
- v5 是一次有效的链路验证：接口、trace、raw response observation、防截断机制都正常。
- v5 不是一个完全合格答案：失败点主要在最终格式/字段约束，而不是上下文定位或计算。

下一步：
- 对结构化聚合题增加 deterministic answer renderer，至少让 `game_weekly_stats` 这类任务直接输出包含 rubrics 所需字段的固定表格。
- 或在 prompt 中加入 `required output fields`，由 harness 根据 deterministic findings/rubrics 生成，而不是让模型自由选择列。
- 优先实现前者，因为本题的计算结果已经由 Python 确定，LLM 只负责自由组织会带来 schema 漏项风险。

## v1.0 - Dev Rubric Judge Automation Upgrade

时间：2026-06-05

背景：
- CL-bench Life 的 dev set 带 rubrics，可用于本地复现式分析。
- 官方 test 判分是 LLM-as-judge，并且一个任务只有通过全部 rubrics 才算 solved。
- 因此本地 harness 需要尽快有自动化 dev judge，用来做回归、失败定位和改动影响评估。

修改：
- 升级 `NexAU-main/examples/clbench_life/judge_rubrics.py`。
- 增加 `--concurrency`，限制为 1 到 5。
- 增加 `--resume` 和 `.partial` 增量写入，避免中断后丢失已完成 judgement。
- 增加 `--max-tokens`、`--request-timeout`、`--thinking-mode`、`--thinking-budget`。
- 增强 JSON 解析，支持 fenced JSON 和从混杂文本中提取 JSON object。
- 增加 Python 侧 normalized 聚合，不完全信任 judge 自己返回的 passed/failed 数字。
- 增加 `*.summary.json`，统计 task pass rate、rubric pass rate、parse errors、failure type counts。

验证：
```powershell
.\NexAU-main\.venv\Scripts\python.exe -m compileall .\NexAU-main\examples\clbench_life\judge_rubrics.py

.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\judge_rubrics.py `
  --dev-input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --predictions ".\runs\nexau_dev_one_v5\predictions.jsonl" `
  --output ".\runs\nexau_dev_one_v5\rubric_judgements_dry_v3.jsonl" `
  --max-rubrics-per-task 4 `
  --dry-run
```

结果：
- `compileall` 通过。
- dry-run 成功生成 judgement、partial 和 summary。
- 当前环境无 `LLM_API_KEY`，真实 judge 调用需由用户在 PowerShell 中运行。

影响：
- 下一步可以对 `runs\nexau_dev_one_v5` 跑真实 dev judge，验证自动化评测能否捕捉 task 30 的 duration 漏项。
- 之后再跑 9 子类 smoke，每次 harness 改动后比较 summary 中的 task/rubric pass rate 和 failure type 分布。

## v1.1 - Deterministic Final Answer for Weekly Game Stats

时间：2026-06-05

触发：
- `runs\nexau_dev_one_v5` 的自动 judge 显示 task 30 为 `14/16` rubrics。
- 失败 rubrics 是缺少每周 `total duration in minutes` 和 `total duration in seconds`。
- 根因是 Python deterministic findings 已经算出 `total_duration_s`，但 LLM 自由排版最终答案时漏列。

修改：
- `clbench_life_harness/deterministic.py` 增加 `DeterministicAnswer` 和 `build_deterministic_answer()`。
- 针对 `game_weekly_stats` 增加 deterministic final answer renderer。
- 固定输出 Monday-start weekly table，包含 games、wins、losses、duration seconds、duration minutes、kills、deaths、assists、CS、gold、damage、vision。
- `run_predictions.py` 增加 `--deterministic-answer-mode {auto,off}`，默认 `auto`。
- 当 deterministic answer 可用且非 dry-run 时，runner 直接输出该答案，不调用 LLM。
- Trace 的 `model_run` 增加 `deterministic_answer_available`、`deterministic_answer_used`、`deterministic_answer_kind`。

验证：
```powershell
.\NexAU-main\.venv\Scripts\python.exe -m compileall .\clbench_life_harness .\NexAU-main\examples\clbench_life\run_predictions.py

.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_one_v6\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_one_v6\traces" `
  --start 30 `
  --limit 1 `
  --concurrency 1
```

结果：
- `compileall` 通过。
- v6 prediction 生成成功，`avg_latency_sec=0.03`，未调用 LLM。
- 输出 13 个 weekly data rows，`Games Played` 求和为 100。
- 表头包含 `Total Duration (seconds)` 和 `Total Duration (minutes)`。
- Trace 显示 `deterministic_answer_used=true`、`llm_observations=[]`。

下一步：
- 用户侧用真实 judge 跑 `runs\nexau_dev_one_v6`，确认 task 30 是否从 `14/16` 到 `16/16`。
- 若通过，再扩展到 9 子类 smoke；若仍失败，根据 judge 失败 rubrics 再调整 deterministic schema。

## v1.2 - Task 30 v6 Judge Pass

时间：2026-06-05

运行目录：
- `runs\nexau_dev_one_v6`

结果：
- Prediction trace 显示 `deterministic_answer_used=true`，`deterministic_answer_kind=game_weekly_stats`。
- `prediction_chars=1966`，`empty_prediction=false`，`framework_error=false`，`suspected_truncation=false`。
- 真实 dev judge 输出 `rubric_judgements_v1.jsonl.summary.json`。
- `task_pass_rate=1.0`，`rubric_pass_rate=1.0`。
- task 30 从 v5 的 `14/16` 提升到 v6 的 `16/16`。
- 原先失败的 duration minutes 和 duration seconds rubrics 均通过。

判断：
- 本次改动修复的是一类问题：确定性中间态完整，但 LLM 最终自由排版漏 schema 字段。
- deterministic final answer renderer 是合理 harness 组件，而不是针对 task id 的硬编码；触发条件基于任务类型和可解析结构化数据。

下一步：
- 进入 9 子类 dev smoke，用自动 judge 统计各类失败。
- 暂不继续单独优化 task 30，避免过拟合。

## v1.3 - Task 217 Workout Trajectory Deterministic Analyzer

时间：2026-06-05

触发：
- 用户运行 `runs\nexau_dev_smoke_v6` 时，完成 task 156 后长时间无新输出。
- partial prediction 只有 4 条：30、71、88、156。
- `traces` 目录也只有 task 30、71、88、156，没有 task 217 trace。
- 说明 task 217 卡在 LLM 调用阶段，尚未返回到 trace 写入。

诊断：
- task 217 是 workout history TSV，首条 user 约 35,826 字，prompt 约 44,956 字。
- dev rubrics 有 21 条，要求分类、排除 warm-up/cool-down、计算 CBPM、按时间段聚合 watts/calories，并推荐 3 节课满足多个约束。
- dry-run 本地完成很快，说明不是分块/检索卡住；主要是模型面对长 TSV 和多约束计算响应过慢。

修改：
- `clbench_life_harness/deterministic.py` 增加 `workout_trajectory` deterministic analyzer。
- 用 TSV parser 解析 workout history。
- 排除 warm-up 和 cool-down。
- 将 Intervals、HIIT、Climb、KPop 归为 intense。
- 将 Power Zone、Low Impact、Entertainment、Just Ride 归为 steady。
- 计算 intense/steady CBPM、平均 calories/class。
- 按 HST 时间段计算 morning/afternoon/evening 的平均 watts 和 calories。
- 从常见 instructor 中选择 3 节不重复课程，满足总时长不超过 90 分钟、instructor 不重复、平均阻力递增，并尽量最大化总 calories。

验证：
```powershell
.\NexAU-main\.venv\Scripts\python.exe -m compileall .\clbench_life_harness .\NexAU-main\examples\clbench_life\run_predictions.py

.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_task217_v7\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_task217_v7\traces" `
  --start 217 `
  --limit 1 `
  --concurrency 1
```

结果：
- task 217 v7 生成成功，`avg_latency_sec=0.17`，未调用 LLM。
- Trace 显示 `deterministic_answer_used=true`、`deterministic_answer_kind=workout_trajectory`。
- 输出 intense=38、steady=52。
- intense CBPM=4.61，steady CBPM=4.31，差值约 0.3。
- morning watts=59.3，afternoon watts=57.6，evening watts=53.7。
- morning calories=121.7，afternoon calories=135.4，evening calories=127.4。
- 推荐计划总时长 90 分钟，总 calories 476，三位 instructor 不重复，平均阻力 35%、37%、38% 递增。

下一步：
- 建议中断原 smoke_v6 长请求，然后用 `--resume` 重跑同一输出目录；task 217 会由 deterministic analyzer 直接完成。
- 之后继续观察 task 221、238、252、392 是否暴露新的失败类型。
