# NexAU Practice: CL-bench Life Harness

本仓库是针对 CL-bench Life 实训题的 NexAU harness 改造版本。目标不是训练或替换基础模型，而是在固定 `nex-agi/Nex-N2-Pro` 的前提下，把 harness 做成可观测、可调试、可验证的系统。

## 目录结构

- `clbench_life_harness/`：CL-bench Life 的解析、路由、分块、检索、结构化中间态、确定性工具、查询规划和硬校验逻辑。
- `NexAU-main/examples/clbench_life/run_predictions.py`：生成 dev/test prediction JSONL 的主入口。
- `NexAU-main/examples/clbench_life/judge_rubrics.py`：dev set 上的 LLM-as-judge 辅助评测脚本。
- `tests/test_structured_solvers.py`：确定性工具、workflow、query planner gating、verifier 的离线回归测试。
- `report.md`：本次 harness 设计、dev 分析、复现方式和风险说明。


## 环境变量

在 PowerShell 中设置基础模型接口：

```powershell
$env:LLM_MODEL="nex-agi/Nex-N2-Pro"
$env:LLM_BASE_URL="https://northgate.xiaobei.top/v1"
$env:LLM_API_KEY="<your-key>"
```

如果需要在 dev set 上复现 GPT-5.1 judge：

```powershell
$env:JUDGE_MODEL="openai/gpt-5.1"
$env:JUDGE_BASE_URL="https://openrouter.ai/api/v1"
$env:JUDGE_API_KEY="<your-openrouter-key>"
```

## 本地检查

在工作区根目录运行：

```powershell
.\NexAU-main\.venv\Scripts\python.exe -m compileall .\clbench_life_harness .\NexAU-main\examples\clbench_life
```

```powershell
.\NexAU-main\.venv\Scripts\python.exe -m pytest .\tests\test_structured_solvers.py -q
```

当前检查结果：`22 passed`。

## Dev 实验复现

先构造 9 个官方子类各 5 题的分层样本：

```powershell
.\NexAU-main\.venv\Scripts\python.exe -m clbench_life_harness.sample_dev `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --ids-output ".\runs\dev_samples\ids_5_per_subcategory_seed20260606.json" `
  --summary-output ".\runs\dev_samples\summary_5_per_subcategory_seed20260606.json" `
  --per-subcategory 5 `
  --seed 20260606 `
  --prefer-task-type-diversity
```

运行 45 题 dev prediction：

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

使用 GPT-5.1 judge 评测：

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

该实验用于错误定位，不代表最终 test score。记录结果为：45 题全部生成，无空输出，GPT-5.1 judge 下 `6/45` 题全通过，rubric pass rate `64.71%`。

## Test 提交文件生成

官方要求 test prediction 文件为 JSONL，逐行对应 `CL-bench-Life-test.jsonl`，每行包含：

- `id`：从 0 开始的行号。
- `prediction`：最终答案正文。
- `messages`：完整对话轨迹，最后一条 assistant 内容和 `prediction` 一致。

生成测试集输出：

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

如果中途被打断，可加 `--resume` 继续：

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
  --continue-on-error `
  --resume
```

## 设计摘要

当前 harness 采用混合 workflow：

- Python 负责确定性部分：任务解析、路由、分块、索引、实体/时间/数字抽取、计数、日历/表格生成、硬约束检查。
- LLM 负责语义部分：模糊排序、情绪判断、证据解释、最终自然语言表达。
- Deterministic solver 只处理可验证子问题，不作为所有任务的默认解法。
- Query planner 只在语义证据检索有收益的 workflow 中启用。
- Repairer 是有上限的最终答案修复，不允许无限循环。

每道题都会输出 trace，记录 workflow、structured state、evidence、LLM 原始返回摘要、hard verification、repair rounds 和失败类型，便于复现实验与定位问题。
