# sglang-multigpu-router-bench

面向“对齐推理优化岗”的 5 天执行计划实现版。

## 目录结构

- `router/`: 入口路由（RR / least-load / cache-aware）
- `workers/`: Worker 启动脚本
- `clients/`: OpenAI-compatible client demo
- `bench/`: workload + 跑分脚本 + 报告
- `configs/`: 集群配置示例
- `results/`: 输出与图表

## Day 1：多卡 serving 跑通（TP / DP）

目标：一条命令多卡启动，并用 OpenAI-compatible client 调通。

示例（按实际模型与显卡配置修改）：

```bash
python -m sglang.launch_server \
  --model-path <MODEL> \
  --tp-size 2 \
  --dp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

并发验证：

```bash
python clients/openai_client_demo.py --base-url http://127.0.0.1:30000/v1 --concurrency 8
```

记录：显存占用、最大并发 OOM 情况、启动耗时。

## Day 2：Router baseline（RR / least-load）+ worker 管理

目标：一个入口 + 多个 worker；路由转发 `/v1/chat/completions`。

要点：

- 同机多 worker（不同端口）
- 健康检查
- 并发限流（semaphore）
- 失败重试（可选）

## Day 3：Cache-Aware Routing

思路：

- 对请求 prompt 生成 prefix 指纹
- Router 维护 worker 最近指纹集合/计数
- 命中度最高优先
- 输出 metrics：`cache_hit_pred`、`route_reason`、`worker_selected`

## Day 4：Benchmark

Workload：

- A 高复用：固定 system prompt + 固定长背景
- B 低复用：随机背景

指标：

- TTFT（p50 / p95）
- 生成吞吐 tokens/s
- 路由预测命中率 vs 实际收益
- Worker 利用率

## Day 5：面试故事线

三张图：

- 架构图：Client → Router → Workers → Metrics
- A/B workload 的 TTFT / throughput
- 命中率与收益相关性

五个常见追问（参考）：

- 为什么高复用收益大
- TP / DP 如何选
- 多节点怎么做
- 长上下文怎么扩
- 为什么 benchmark 公平

## 运行顺序建议

1. 启动多卡 server
2. 启动多 worker（本机多个端口）
3. 启动 router
4. 跑 client 并发验证
5. 跑 benchmark 输出结果

## 依赖

- Python 3.9+
- `pip install fastapi uvicorn httpx pyyaml`

## 快速启动

1) 启动多个 worker（同机多端口）：

```bash
MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct \
TP_SIZE=2 DP_SIZE=1 NUM_WORKERS=2 GPU_IDS=0,1,2,3 \
bash workers/start_workers.sh
```

2) 启动 router：

```bash
python router/app.py --config configs/cluster.yaml --policy rr --port 8088
```

3) 并发验证：

```bash
python clients/openai_client_demo.py --base-url http://127.0.0.1:8088/v1 --concurrency 8
```

## Cache-Aware Routing

```bash
python router/app.py \
  --config configs/cluster.yaml \
  --policy cache-aware \
  --prefix-chars 512 \
  --fallback-policy least-load \
  --fallback-hit-rate-threshold 0.2 \
  --fallback-window 200 \
  --fallback-min-samples 50 \
  --cache-entries 4096
```

低复用边界控制建议：

- 当最近窗口命中率低于阈值（例如 0.2）时，自动回退到 `least-load`/`rr`，避免 cache-aware 在低复用流量上产生额外决策开销。
- 查看 `GET /metrics`，重点关注：
  - `route_decision_avg_ms` / `route_decision_p95_ms`（路由决策时延）
  - `cpu_percent`（路由 CPU 开销）
  - `load_balance_ratio` / `load_balance_span`（负载均衡度）
  - `route_reason_counts` 与 `fallback_ratio`（命中与回退行为）

## Benchmark

高复用（A）：

```bash
python bench/run.py --workload high --output results/bench_high.jsonl
python bench/report.py --input results/bench_high.jsonl --output results/bench_high_report.json
```

低复用（B）：

```bash
python bench/run.py --workload low --output results/bench_low.jsonl
python bench/report.py --input results/bench_low.jsonl --output results/bench_low_report.json
```

扩展 workload（MLSys 评测）：

- `medium`: 中等复用
- `phase-shift`: 前缀分布中途漂移
- `burst`: 突发到达模式
- `mixed-length`: 长短请求混部

## MLSys Artifact（强基线 + 消融 + 稳定性）

矩阵配置：`configs/mlsys_suite.json`

- 强基线：`rr`, `least-load`, `cache-aware-no-fallback`, `cache-aware-fallback`
- 多种 workload：`high/medium/low/phase-shift/burst/mixed-length`
- 多 seed 统计（默认 `0/1/2`）

运行整套实验：

```bash
bash reproduce.sh
```

或手动：

```bash
python bench/suite.py \
  --matrix configs/mlsys_suite.json \
  --model /home/thjiang/models/Qwen2.5-72B-Instruct
```

输出结构（时间戳目录）：

- 每次 run：`*.jsonl`, `*.jsonl.meta.json`, `*_report.json`
- 汇总：`aggregate.json`
- 环境快照：`env_*.txt`

新增核心指标：

- tail：`ttft_ms_p99`, `e2e_ms_p99`
- 路由开销：`route_decision_p99_ms`, `router_cpu_percent`
- 负载均衡：`jain_fairness_requests`, `jain_fairness_tokens`
- 缓存效率：`cache_hit_pred_ratio`, `reuse_depth_est_*`
- 回退行为：`fallback_ratio`

## 需要保留的材料

- 启动命令（含 `--tp-size` / `--dp-size`）
- `nvidia-smi` 截图
- server log（含 tp/dp 配置）
- 并发验证输出
- benchmark 的 JSONL + report

## RL Rollout Router

本仓库现已在原 Serving Router 上增加面向 GRPO/RLVR 的控制平面：

- Group/Version/Sample Metadata 与 Group Barrier；
- Fixed PACK、Fixed EVEN SPLIT、Load-proportional SPLIT；
- 可解释 Adaptive PACK/SPLIT 与 Offline Oracle；
- READY/DRAINING/UPDATING/FAILED 版本状态机；
- 磁盘权重更新适配边界、Checksum 和 Rolling Update 协调；
- Policy Lag ≤ 1 的有界 Rollout Buffer；
- JSONL Trace、确定性 Offline Replay、Prometheus 指标；
- Unit/Integration/Correctness/Fault-injection 测试。

CPU 正确性与 Golden Trace 回放：

```bash
bash scripts/run_correctness.sh
```

启动前检查 GPU 是否被占用：

```bash
python3 scripts/check_gpu_availability.py --gpu-ids 0,1,2,3,4,5
```

双 Rollout Worker 与 Router：

```bash
MODEL_PATH=/path/to/3b-model bash scripts/launch_rollout_workers_6xa6000.sh
bash scripts/launch_rl_router.sh
```

控制接口包括 `/rl/groups`、`/rl/route`、`/rl/responses`、
`/rl/begin-weight-update`、`/rl/commit-weight-update`、`/rl/workers`、
`/rl/policy-versions` 和 `/rl/metrics/prometheus`。

实施状态和尚未满足的真实训练前置条件见
`docs/implementation_status.md`；架构、正确性与实验规范分别见
`docs/architecture.md`、`docs/correctness.md` 和 `docs/experiment_protocol.md`。

### 四卡模式（2 Trainer + 2 Rollout）

当前服务器可使用非连续 GPU ID，默认布局为 Trainer `1,3`、Rollout
`5,7`。启动器会校验四张卡无重复、角色不重叠且空闲：

```bash
MODEL_PATH=/path/to/3b-model bash scripts/start_4gpu_stack.sh
SLIME_LAUNCH_CMD='<locked slime launch command>' \
  bash scripts/launch_training_4xa6000.sh
```

停止服务：

```bash
bash scripts/stop_4gpu_stack.sh
```

完整说明见 `docs/4gpu_deployment.md`。
