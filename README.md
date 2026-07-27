<!--[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)-->
<!--[![Publish](https://github.com/refinefuture-ai/refft.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/refinefuture-ai/reft/actions/workflows/docker-publish.yml))-->
<!--![reft cc-new-logo jpg](https://github.com/user-attachments/assets/25f0c2e7-0f64-41e9-979d-ddb0ff932c4d)-->
<!--<center><h3>C++ Native-implemented(without Python/PyTorch) LLM/LM's inference serving and training for High-Performance and Easy-to-Use</h3></center>-->
<!-- <img width="3466" height="1308" alt="3311b35fe62743ee47cb7401294aac34" src="https://github.com/user-attachments/assets/1ae5471a-9ceb-4daa-aa91-6dc596639786" /> -->

<div align="center">

<!-- <img width="1024" height="356" alt="64fd0df0b35999dbc1a2e4b881231767" src="https://github.com/user-attachments/assets/422dfc35-c025-4949-9093-62e01ac920a6" />-->
<!-- <img width="1024" alt="https://refinefuture.ai" src="https://github.com/user-attachments/assets/dd0ade07-5baf-4373-9bce-17235cd5b143" /> -->
<img width="1024" alt="https://refinefuture.ai" src="https://github.com/user-attachments/assets/e166a109-33b4-4e43-9d61-dadf77d12115" />

# refft.cpp

**Compile a LLM/LM — model, ops, inference, serving, API — into a single native executable.**<br/>
No Python. No PyTorch. No runtime dependencies beyond the OS and the GPU/NPU backend.

[![Release](https://img.shields.io/github/v/release/refinefuture-ai/refft.cpp)](https://github.com/refinefuture-ai/refft.cpp/releases)

[**Quick Start**](#quick-start) · [Runtimes](#runtimes) · [Packages](#packages) · [Key Features](#features) · [FAQs](#faqs) · [Legacy](#legacy-archived)

</div>

---

## About

`refft.cpp` is a building tool to compile LLM/LMs' inference and training on the designated cloud-GPU or edge-NPU backends to a native executable including API, inference serving, training, model, ops, etc

- Average **20%+ faster** inference and training than Python/PyTorch-based inference/training (in the same quantization/precision and use cases)
- **0 running dependencies** other than Linux/Android/Mac system and GPU/NPU backends

<p align="center">
	<!-- <img width="1024" height="510" alt="Refft Builder" src="https://github.com/user-attachments/assets/2cdb49b0-6496-46f7-8dbe-997a7430c160" /> -->
	<!-- <img width="1024" alt="Refft Builder" src="https://github.com/user-attachments/assets/9e34ac36-c653-4987-8846-66c7e539b644" /> -->
	<!-- <img width="1024" alt="53759a43ddb8fd6f5494518b309398cc" src="https://github.com/user-attachments/assets/d4371c96-43d4-4f46-ac80-679cd8fac5f2" /> -->
	<img width="968" height="466" alt="efe50873e7a96490e9168bed1e740e35" src="https://github.com/user-attachments/assets/102bdeb2-4c63-4729-84b3-ffaa6d0f8cb4" />
</p>

---

<a name="quick-start"></a>

## :rocket: Quick Start

Three steps: **install → run → chat**. `refft-hexagon` is used as the example runtime;
the other [runtimes](#runtimes) work the same way.

### 1. Install

One command installs the newest release for your platform. No version to pick, no path to configure.

**Linux, or Android from a host with `adb`** (also works inside Termux):

```bash
curl -fsSL https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/install.sh | sh
```

**Windows** (PowerShell):

```powershell
irm https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/install.ps1 | iex
```

It prints `Installation complete` and where the runtime landed:

| Platform | Default location |
|----------|------------------|
| Linux | `~/.local/share/refft-hexagon` &nbsp;(`/opt/refft-hexagon` as root) |
| Android (pushed over `adb`) | `/data/local/tmp/refft-hexagon` |
| Windows | `%LOCALAPPDATA%\Programs\refft-hexagon` |

> Options such as `--prefix`, `--tag` and `--hexagon-version` are documented in
> [refft-hexagon/README.md](refft-hexagon/README.md).

### 2. Run

Everything is one binary — `refft-hexagon` — with three commands: `cli`, `serve`, `bench`.

<details open>
	<summary><b>Linux / Termux</b></summary>

```bash
cd ~/.local/share/refft-hexagon
export LD_LIBRARY_PATH=$PWD/lib
export ADSP_LIBRARY_PATH=$PWD/lib
export REFFT_HEXAGON_MODULE_PATH=$PWD/lib/librefft_hexagon_v73.so   # match the .so in lib/

./bin/refft-hexagon help
./bin/refft-hexagon cli   --model_dir /path/to/model --backend hexagon --prompt "Who are you?" --max_new_tokens 128
./bin/refft-hexagon serve --model_dir /path/to/model --backend hexagon --port 8080
```

</details>

<details>
	<summary><b>Android over <code>adb</code></b></summary>

```bash
adb shell
cd /data/local/tmp/refft-hexagon
export LD_LIBRARY_PATH=$PWD/lib
export ADSP_LIBRARY_PATH=$PWD/lib

./bin/refft-hexagon cli --model_dir /data/local/tmp/model --backend hexagon --prompt "Who are you?" --max_new_tokens 128
```

</details>

<details>
	<summary><b>Windows (PowerShell)</b></summary>

```powershell
cd $env:LOCALAPPDATA\Programs\refft-hexagon
$env:ADSP_LIBRARY_PATH = "$PWD\lib"

.\bin\refft-hexagon.exe cli   --model_dir C:\path\to\model --backend hexagon --prompt "Who are you?" --max_new_tokens 128
.\bin\refft-hexagon.exe serve --model_dir C:\path\to\model --backend hexagon --port 8080
```

</details>

| | |
|---|---|
| **Commands** | `help` · `version` · `cli` · `serve` · `bench serve --prompt "Who are you?"` |
| **Generation** | `--max_new_tokens <n>` — **default 1** for `cli`, 16 for `serve` · `--chat` · `--stream true` · `--print_perf_metrics` |
| **Sampling** | pass `--sampler chain` first, then `--do_sample` · `--temperature` · `--top_k` · `--top_p` · `--min_p` · `--repetition_penalty` · `--ignore_eos` |
| **Models** | `qwen2_5` · `qwen3` · `qwen3_5` (text + image) · `qwen3_moe` — list them with `cli --list_models` |

### 3. Chat with the server

Once `refft-hexagon serve` is up:

```bash
curl http://127.0.0.1:8080/health

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Explain KV cache reuse briefly."}],
    "max_tokens": 128,
    "stream": false
  }'
```

The endpoint is OpenAI-compatible — point any OpenAI client at `http://127.0.0.1:8080/v1`.

---

<a name="using"></a>

## :package: Runtimes and Packages

<a name="runtimes"></a>

### Runtimes

One native executable per backend, all sharing the same commands and the same OpenAI-compatible API:

| Runtime | Target | Status |
|---------|--------|:------:|
| **`refft-hexagon`** | Qualcomm Hexagon NPU (Snapdragon) — Android, arm64 Linux, Windows | ✅ Available |
| **`refft-cuda`** | NVIDIA GPU — x86_64 Linux | 🚧 Coming |
| **`refft-mlx`** | Apple Silicon — macOS arm64 | 🚧 Coming |

<a name="packages"></a>

### Packages

The installer picks the right package automatically. Direct downloads if you prefer:

| Runtime Package | Description |
|-----------------|-------------|
| [`refft-hexagon_android-aarch64`](https://github.com/refinefuture-ai/refft.cpp/releases/latest) | Android arm64 — Hexagon HTP runtime + DSP skel |
| [`refft-hexagon_ubuntu-arm64`](https://github.com/refinefuture-ai/refft.cpp/releases/latest) | arm64 Linux — Snapdragon dev kits |
| [`refft-hexagon_windows-arm64`](https://github.com/refinefuture-ai/refft.cpp/releases/latest) | Windows on Snapdragon |

- Releases are tagged **`vYYYY.MM.DD.NN`** — year, month, day, and the n-th release of that day.
- A Hexagon package ships one variant (`v73`, `v81`, …) per SoC generation.
- Names follow **`refft-<runtime>_<os>-<arch>[-<variant>]`**, so the upcoming
  `refft-cuda_ubuntu-x86_64` and `refft-mlx_macos-arm64` install and run the same way.

---

<a name="features"></a>

## :fire: Key Features

| Feature | What it does |
|---------|--------------|
| **Native Compilation** | Compile the whole inference/training of a LLM/LM into the native executable object |
| **OpenAI-Compatible API** | Seamless integration with existing tools |
| **Custom Training via Plugins** | Data-loader, Optimizer, Model layers, Loss-function |
| **Multi-Modal Support** | Text, vision, audio, etc |
| **Native vRAM mgt** | Native mem mgt instead of GC to lower peak occ-mem and alloc-overhead |
| **Mixed-precision quantization** | FP16, w4a16, w8a16, etc supported per tensor/channel/block |
| **NPU dynamics** | Dynamic shape, MoE, control flow, flexible heterogeneous compute on NPU |

<!--
- **Configurable Continuous Batching Scheduler** -- Efficient request handling with dynamic batching
- **Paged Attention** -- Optimize mem mgt for long sequences and lower memory footprint
- **Flash Attention -- Optimized mem mgt for long sequences and lower memory footprint
-->

---

<a name="faqs"></a>

## :question: FAQs

<details>
	<summary>Why refft.cpp implements all of modeling, serving and training in C++</summary>

It's manly for a better performance and easy-to-use compared to Python/PyTorch-based as well as for scalability on edge-NPU.

</details>

<details>
	<summary>Why Triton is not used in refft.cpp</summary>

Because the Triton models can get up to 78% of the performance of the CUDA models on the H100 and up to 82% on the A100.

[CUDA-Free Inference for LLMs](https://pytorch.org/blog/cuda-free-inference-for-llms/)

</details>

<details>
	<summary>How to support multi-nodes GPU/NPU</summary>

Technically refft.cpp supports multi-nodes inference and training, while multi-nodes haven't been tested due to lacking of HW resources. Please contact us if needed.

</details>

<!--
<details>
	<summary>How to calculate the required GPU vRAM size for a LLM's inference or training</summary>
	
- Inference: If a LLM weights is xB, then 2x(GB) is the minimum vRAM size needed.
- Training: If a LLM weights is xB, then 8x(GB) is the minimus vRAM size needed for full-parameter training and 4x(GB) is the minimum for freeze-SFT, LoRA or RL.
- TP, PP need to be configured per the amount of GPUs(1, 2, 4, 8, 16...). TP*PP= the amout of GPUs.

</details>
-->

<details>
	<summary>Strictly equivalence of computational precision matters the most in LLM/LM's ops and serving optimization</summary>

- https://epoch.ai/gradient-updates/why-benchmarking-is-hard
- https://blog.vllm.ai/2025/10/28/Kimi-K2-Accuracy.html

</details>

---

## Legacy (Archived)

The previous generation — the old QNN / CUDA / MLX packages built around the `refft-cli` /
`refft-server` binaries, and the training flow — is no longer part of the supported surface:

| Archived | Contents |
|----------|----------|
| [docs/legacy/](docs/legacy/README.md) | Retired packages and their usage |
| [docs/legacy/install_and_usage.md](docs/legacy/install_and_usage.md) | The old install and usage guide |
| [docs/legacy/training.md](docs/legacy/training.md) | Training of LLM/LM — datasets, `refft train`, sample output |
| [docs/legacy/build_tools.md](docs/legacy/build_tools.md) | The `refft.cpp` build tools showcase |

CUDA and Apple Silicon are not going away: they return as the single-binary
[`refft-cuda` and `refft-mlx`](#runtimes) runtimes. For anything current, use `refft-hexagon`
as described in [Quick Start](#quick-start).

---

## Contact Us

Please contact us via [haiteng@refinefuture.ai](mailto:haiteng@refinefuture.ai) for commercial uses, technical consulting, sponsorship/partnership opportunities, etc.

## Acknowledgment

`refft.cpp` was inspired by Andrej Karpathy' [llm.c](https://github.com/karpathy/llm.c), and also referred to [HuggingFace](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
