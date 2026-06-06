# CL-bench Life Harness

这是第一版可观测骨架，目标不是直接追求最高分，而是先让每道题的解析、分块、检索和中间证据可检查、可复现。

## 当前能力

- 读取 dev/test JSONL，并统一解析最终任务。
- 识别单轮 `<|TASK|>` 与多轮最终 user 问题。
- 对上下文做稳定分块，保留行号、字符偏移、文档标签、说话人和时间戳。
- 用轻量关键词检索生成候选证据。
- 为每道题写出 trace，记录任务类型、chunk 数、top hits、rubric 样例和证据预览。

## 运行命令

在工作区根目录运行：

```powershell
python -m clbench_life_harness.run_baseline `
  --input ".\CL-Bench-dataset\CL-bench%20Life.jsonl" `
  --output-dir ".\runs\baseline_dev_preview" `
  --limit 10
```

当前骨架只使用 Python 标准库，因此可以直接用 `python -m` 跑。后续接入 NexAU agent 时优先使用 NexAU 项目的 uv 环境；如果本机默认 uv cache 初始化失败，可以先把 `UV_CACHE_DIR` 指到工作区内的临时目录。

处理 test 预览：

```powershell
python -m clbench_life_harness.run_baseline `
  --input ".\实训题目\CL-bench-Life-test.jsonl" `
  --output-dir ".\runs\baseline_test_preview" `
  --limit 5
```

输出目录中：

- `run_summary.json`：本次运行总览。
- `summary.jsonl`：每题一行的轻量摘要。
- `traces/task_XXXX.json`：每题详细 trace。

## 设计原则

- Python 做确定性解析、分块、索引、计数和格式校验。
- LLM 后续只负责语义判断、证据筛选和最终表达。
- 每次优化都要能通过 trace 说明修复的是上下文定位、事实抽取、推理、计算、格式还是最终校验问题。
