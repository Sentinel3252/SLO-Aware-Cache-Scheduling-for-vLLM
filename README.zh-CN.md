# CacheFlow Scheduler

**面向 vLLM V1 的缓存感知等待队列调度插件。**

[English](README.md) · [安装与使用](docs/getting-started.md) · [配置参考](docs/configuration.md) · [贡献指南](CONTRIBUTING.md)

CacheFlow 调整 vLLM 等待队列中请求的处理顺序。排序时会考虑本地 prefix cache
命中、剩余 prefill 工作量、等待时间和 TTFT 目标。它通过 `--scheduler-cls`
接入单个 V1 Engine，token budget、KV 分配、执行和请求生命周期仍由 vLLM 负责。

如果你想比较 RAG、多会话或长短请求混合场景下的调度策略，可以从这里开始。
项目包含六种策略，以及生成 trace、记录流式延迟和汇总报告的工具。每轮只对
前 K 个候选请求评分，保留请求优先级，并默认保护因抢占而回到 FCFS 队首的请求。
调度开销也有对应的指标可查。

## 当前状态

当前版本为 `0.1.0`，还在早期开发阶段。依赖固定为 vLLM `0.28.0`，已有 GPU
测试则使用 `0.23.1` 的一个开发构建，尚未在 GPU 上验证 `0.28.0`。
适配器依赖 vLLM 私有接口，升级前请阅读[兼容性说明](docs/compatibility.md)。
已有测试中，有些指标改善了，也有些变差了，具体见[结果记录](docs/results.md)。

## 无 GPU 快速开始

使用 Python 3.10+，核心策略和仿真无需第三方运行依赖。下面直接从仓库安装：

```bash
git clone https://github.com/Sentinel3252/SLO-Aware-Cache-Scheduling-for-vLLM.git
cd SLO-Aware-Cache-Scheduling-for-vLLM
python -m pip install -e .

cacheflow-generate --workload rag --count 100 --qps 4 --seed 7 --output results/rag.jsonl
cacheflow-simulate --trace results/rag.jsonl --mode fcfs --output results/fcfs.jsonl
cacheflow-simulate --trace results/rag.jsonl --mode caras --output results/caras.jsonl
cacheflow-report --input FCFS=results/fcfs.jsonl --input CARAS=results/caras.jsonl
```

这份报告使用串行 prefill 和无限 prefix cache 的简化模型，适合观察排序行为，
不能用来判断真实 GPU 性能。

回放开始前会校验所有请求及 ID 的唯一性。遇到无效记录时会报告文件路径和行号。
详细字段见 [trace 格式](docs/trace-format.md)。

## 接入 vLLM

在具备兼容 GPU 运行环境的 Linux 主机上，使用同一个 Python 环境：

```bash
python -m pip install -r requirements/vllm.txt
cacheflow-check-vllm
export CACHEFLOW_MODE=caras
export CACHEFLOW_CANDIDATE_WINDOW=16
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --stream-interval 1 \
  --no-async-scheduling \
  --scheduler-cls cacheflow.vllm_adapter.CARASScheduler
```

`cacheflow-check-vllm` 只检查接口。通过后，还需要发送真实请求，与 FCFS 的输出
比较，并在目标模型和 GPU 上校准 cost model。性能比较时，两边使用相同配置。
完整流程见[用户指南](docs/getting-started.md)。

## 策略与边界

| 模式 | 排序依据 |
| --- | --- |
| `fcfs` | 到达时间与请求 ID |
| `cache` | 本地缓存命中 token 数 |
| `residual` | 剩余 prefill token 数 |
| `aging` | 等待时间修正的剩余工作量 |
| `caras` | 估计工作量、aging 与可选 cache credit |
| `slo` | 可行性、slack 与 aging override |

所有模式都保留请求优先级；等待过久的请求只会在当前候选窗口内被优先处理。
`MAX_WAIT_MS` 不是请求等待时长或 SLO 的硬保证。CacheFlow 不实现跨 worker
路由、cache eviction、remote KV 查询或完整内存装箱。

## 参与项目

遇到兼容性问题、文档不清楚的地方，或有新的 workload 测试结果，都可以提交。
请阅读[贡献指南](CONTRIBUTING.md)、[路线图](ROADMAP.md)和[社区规范](CODE_OF_CONDUCT.md)。

```bash
python -m unittest discover -v
```

问题与使用反馈请提交到 [GitHub Issues](https://github.com/Sentinel3252/SLO-Aware-Cache-Scheduling-for-vLLM/issues)。
许可证为 [Apache-2.0](LICENSE)；CacheFlow 是独立项目。
