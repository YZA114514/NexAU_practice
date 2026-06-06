# CL-bench Life NexAU Harness

这个目录是面向 CL-bench Life 的 NexAU baseline harness。它复用工作区根目录下的 `clbench_life_harness` 做确定性解析、分块、检索和 trace，再用 NexAU `Agent` 调用基础模型生成答案。

## 环境变量

```powershell
$env:LLM_MODEL="nex-agi/Nex-N2-Pro"
$env:LLM_BASE_URL="https://northgate.xiaobei.top/v1"
$env:LLM_API_KEY="你的 key"
```

不要把 API key 写入代码、README、trace 或 `.env` 文件。临时运行时在当前 PowerShell 会话设置即可；如果需要跨终端保留，请使用系统环境变量管理方式。

如果 `uv` 默认缓存目录异常，可以先设置：

```powershell
$env:UV_CACHE_DIR="D:\大四下\科研\Prof.Xuanjing Huang\实训\.uv-cache"
```

## Smoke Test

从工作区根目录运行：

先按官方 9 个子类做分层抽样。这样小实验不会只覆盖文件前几条：

```powershell
python -m clbench_life_harness.sample_dev `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --ids-output ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --summary-output ".\runs\dev_samples\summary_1_per_subcategory.json" `
  --per-subcategory 1 `
  --prefer-task-type-diversity
```

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_smoke\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_smoke\traces" `
  --ids-file ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --concurrency 1
```

如果暂时没有 API key，可以先用 dry run 检查 prompt、证据包和 trace：

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\实训题目\CL-bench-Life-test.jsonl" `
  --output ".\runs\nexau_test_dry\predictions.jsonl" `
  --trace-dir ".\runs\nexau_test_dry\traces" `
  --limit 1 `
  --dry-run
```

`--dry-run` 不调用模型，输出中的 prediction 只是占位文本；它用于检查中间态，不代表任务成功。

真实 LLM smoke test 需要 `LLM_API_KEY` 已设置。建议先跑 9 子类分层 dev 样本，确认 trace 和 prediction 都正常，再扩大样本：

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_smoke\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_smoke\traces" `
  --ids-file ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --concurrency 1
```

如果中途手动停止，脚本会把已完成记录写到 `predictions.jsonl.partial`。恢复时加 `--resume`：

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_smoke\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_smoke\traces" `
  --ids-file ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --concurrency 1 `
  --resume
```

Dev rubric 自动评估是近似评估，不等同官方 GPT-5.1 judge。它适合用来定位失败类型和回归趋势：

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\judge_rubrics.py `
  --dev-input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --predictions ".\runs\nexau_dev_smoke\predictions.jsonl" `
  --output ".\runs\nexau_dev_smoke\rubric_judgements.jsonl" `
  --ids-file ".\runs\dev_samples\ids_1_per_subcategory.json"
```

## Test 预测文件

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\实训题目\CL-bench-Life-test.jsonl" `
  --output ".\test_predictions_你的姓名.jsonl" `
  --trace-dir ".\runs\nexau_test\traces" `
  --concurrency 5
```

## 当前设计

- Python 负责解析、分块、检索和 trace，保证可复现。
- NexAU Agent 负责基于 evidence pack 生成最终答案。
- 第一版不使用 verifier agent，先通过 dev trace 暴露错误类型。
- 每题 trace 包含任务类型、top evidence chunks、prompt 预览和 prediction，便于定位失败发生在上下文定位、事实抽取、推理、计算、格式还是最终校验。
## Current Dev Experiment Command

`run_predictions.py` now writes LLM response diagnostics into per-task traces and fails fast on empty predictions by default. Use a fresh run directory for the first real dev smoke:

```powershell
.\NexAU-main\.venv\Scripts\python.exe .\NexAU-main\examples\clbench_life\run_predictions.py `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output ".\runs\nexau_dev_smoke_v3\predictions.jsonl" `
  --trace-dir ".\runs\nexau_dev_smoke_v3\traces" `
  --ids-file ".\runs\dev_samples\ids_1_per_subcategory.json" `
  --concurrency 1 `
  --max-iterations 3
```

If a task fails with an empty prediction, inspect its trace first. Only use `--allow-empty` for debugging, not for scored experiment output.
