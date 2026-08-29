<!--[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)-->
[![Release](https://img.shields.io/github/v/release/refinefuture-ai/refft.cpp)](https://github.com/refinefuture-ai/refft.cpp/releases)
<!--[![Publish](https://github.com/refinefuture-ai/refft.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/refinefuture-ai/reft/actions/workflows/docker-publish.yml))-->

<!--![reft cc-new-logo jpg](https://github.com/user-attachments/assets/25f0c2e7-0f64-41e9-979d-ddb0ff932c4d)-->

<!--<center><h3>C++ Native-implemented(without Python/PyTorch) LLM/LM's inference serving and training for High-Performance and Easy-to-Use</h3></center>-->

<!-- <img width="3466" height="1308" alt="3311b35fe62743ee47cb7401294aac34" src="https://github.com/user-attachments/assets/1ae5471a-9ceb-4daa-aa91-6dc596639786" /> -->

<p align="center">
	<!-- <img width="1024" height="356" alt="64fd0df0b35999dbc1a2e4b881231767" src="https://github.com/user-attachments/assets/422dfc35-c025-4949-9093-62e01ac920a6" />-->
	<!-- <img width="1024" alt="https://refinefuture.ai" src="https://github.com/user-attachments/assets/dd0ade07-5baf-4373-9bce-17235cd5b143" /> -->
	<img width="1024" alt="https://refinefuture.ai" src="https://github.com/user-attachments/assets/e166a109-33b4-4e43-9d61-dadf77d12115" />
</p>

# About

`refft.cpp` is a building tool to compile LLM/LMs' inference and training on the designated cloud-GPU or edge-NPU backends to a native executable including API, inference serving, training, model, ops, etc

- Average 20%+ faster inference and training than Python/PyTorch-based inference/training(in the same quantization/precision and use cases)

- 0 running dependencies other than Linux/Android/Mac system and GPU/NPU backends

<p align="center">
	<!-- <img width="1024" height="510" alt="Refft Builder" src="https://github.com/user-attachments/assets/2cdb49b0-6496-46f7-8dbe-997a7430c160" /> -->
	<!-- <img width="1024" alt="Refft Builder" src="https://github.com/user-attachments/assets/9e34ac36-c653-4987-8846-66c7e539b644" /> -->
	<!-- <img width="1024" alt="53759a43ddb8fd6f5494518b309398cc" src="https://github.com/user-attachments/assets/d4371c96-43d4-4f46-ac80-679cd8fac5f2" /> -->
	<!-- <img width="968" height="466" alt="efe50873e7a96490e9168bed1e740e35" src="https://github.com/user-attachments/assets/102bdeb2-4c63-4729-84b3-ffaa6d0f8cb4" /> -->
	<img width="995" height="502" alt="55a6248c50bfda5f996bb47c68cced5c" src="https://github.com/user-attachments/assets/f5af405c-57e1-403b-95c5-436f5038b0c4" />
</p>



## :fire: Key Features

- **Native Compilation** --  Compile the whole inference/training of a LLM/LM into the native executable object
- **OpenAI-Compatible API** -- Seamless integration with existing tools
- **Custom Training via Plugins** -- Data-loader, Optimizer, Model layers, Loss-function
- **Multi-Modal Support** -- Text, vision, audio, etc
- **Native vRAM mgt** -- Native mem mgt instead of GC to lower peak occ-mem and alloc-overhead
- **Mixed-precision quantization** -- FP16, w4a16, w8a16, etc supported per tensor/channel/block
- **NPU dynamics** -- enable NPU to support dynamic shape, MoE, control flow, flexible heterogeneous compute

<!--
- **Configurable Continuous Batching Scheduler** -- Efficient request handling with dynamic batching
- **Paged Attention** -- Optimize mem mgt for long sequences and lower memory footprint
- **Flash Attention -- Optimized mem mgt for long sequences and lower memory footprint
-->

***

<a name="using"></a>

# :rocket: Inference of LLM/LM

`refft.cpp` build tools compile a LLM/LM into a single native executable, one per backend: `refft-hexagon` today, with `refft-cuda` and `refft-mlx` coming. They all expose the same commands and the same OpenAI-compatible API, so the Quick Start below uses `refft-hexagon` as the example.

The previous generation of packages, built around the `refft-cli` / `refft-server` binaries, is no longer supported and is archived under [Legacy](#legacy-archived).

<a name="runtimes"></a>

## Runtimes

|        Runtime       |         Target |    Status |
|----------------------|----------------|-----------|
| `refft-hexagon` | Qualcomm Hexagon NPU (Snapdragon) on Android, arm64 Linux and Windows | Available |
| `refft-cuda` | NVIDIA GPU on x86_64 Linux | Coming |
| `refft-mlx` | Apple Silicon on macOS arm64 | Coming |

<a name="packages"></a>

## Packages

The installer picks the right package automatically. Direct downloads if you prefer:

|        Runtime Package       |         Description |
|------------------------------|---------------------|
| [refft-hexagon_android-aarch64](https://github.com/refinefuture-ai/refft.cpp/releases/latest) | Android arm64, Hexagon HTP runtime + DSP skel |
| [refft-hexagon_ubuntu-arm64](https://github.com/refinefuture-ai/refft.cpp/releases/latest) | arm64 Linux, Snapdragon dev kits |
| [refft-hexagon_windows-arm64](https://github.com/refinefuture-ai/refft.cpp/releases/latest) | Windows on Snapdragon |

Releases are tagged `vYYYY.MM.DD.NN`: year, month, day, and the n-th release of that day. A Hexagon package ships one variant (`v73`, `v81`, etc) per SoC generation. Names follow `refft-<runtime>_<os>-<arch>[-<variant>]`, so the upcoming `refft-cuda_ubuntu-x86_64` and `refft-mlx_macos-arm64` install and run the same way.

<a name="quick-start"></a>

## Quick Start

Three steps: install, run, chat.

### 1. Install

One command installs the newest release for your platform. No version to pick, no path to configure.

Linux, or Android from a host with `adb` (also works inside Termux):

```bash
curl -fsSL https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/install.ps1 | iex
```

It prints `Installation complete` and where the runtime landed:

| Platform | Default location |
|----------|------------------|
| Linux | `~/.local/share/refft-hexagon`, or `/opt/refft-hexagon` as root |
| Android, pushed over `adb` | `/data/local/tmp/refft-hexagon` |
| Windows | `%LOCALAPPDATA%\Programs\refft-hexagon` |

Options such as `--prefix`, `--tag` and `--hexagon-version` are documented in [refft-hexagon/README.md](refft-hexagon/README.md).

### 2. Run

Everything is one binary — `refft-hexagon` — with three commands: `cli`, `serve`, `bench`.

The examples use [refinefuture-ai/SmolVLM2-500M-Video-Instruct-REFFT](https://huggingface.co/refinefuture-ai/SmolVLM2-500M-Video-Instruct-REFFT), a Hexagon serving bundle. Download it first:

```bash
hf download refinefuture-ai/SmolVLM2-500M-Video-Instruct-REFFT --local-dir ~/SmolVLM2-500M-Video-Instruct-REFFT
```

`--model` points at the bundle directory, the one holding `serving_manifest.json`.

Linux / Termux, from the folder printed by the installer:

```bash
cd ~/.local/share/refft-hexagon
export LD_LIBRARY_PATH=$PWD/lib
export ADSP_LIBRARY_PATH=$PWD/lib
export REFFT_HEXAGON_MODULE_PATH=$PWD/lib/librefft_hexagon_v73.so   # match the .so in lib/

./bin/refft-hexagon help
./bin/refft-hexagon cli   --model ~/SmolVLM2-500M-Video-Instruct-REFFT/SmolVLM2-500M-Video-Instruct-w4.refft --backend hexagon --prompt "Who are you?" --max_new_tokens 128
./bin/refft-hexagon serve --model ~/SmolVLM2-500M-Video-Instruct-REFFT/SmolVLM2-500M-Video-Instruct-w4.refft --backend hexagon --port 8080
```

Android over `adb`:

```bash
adb push ~/SmolVLM2-500M-Video-Instruct-REFFT/SmolVLM2-500M-Video-Instruct-w4.refft /data/local/tmp/
adb shell
cd /data/local/tmp/refft-hexagon
export LD_LIBRARY_PATH=$PWD/lib
export ADSP_LIBRARY_PATH=$PWD/lib

./bin/refft-hexagon cli --model /data/local/tmp/SmolVLM2-500M-Video-Instruct-w4.refft --backend hexagon --prompt "Who are you?" --max_new_tokens 128
```

Windows (PowerShell):

```powershell
cd $env:LOCALAPPDATA\Programs\refft-hexagon
$env:ADSP_LIBRARY_PATH = "$PWD\lib"

.\bin\refft-hexagon.exe cli   --model $HOME\SmolVLM2-500M-Video-Instruct-REFFT\SmolVLM2-500M-Video-Instruct-w4.refft --backend hexagon --prompt "Who are you?" --max_new_tokens 128
.\bin\refft-hexagon.exe serve --model $HOME\SmolVLM2-500M-Video-Instruct-REFFT\SmolVLM2-500M-Video-Instruct-w4.refft --backend hexagon --port 8080
```

| Topic | Options |
|-------|---------|
| Commands | `help`, `version`, `cli`, `serve`, `bench serve --prompt "Who are you?"` |
| Generation | `--max_new_tokens <n>` (default 1 for `cli`, 16 for `serve`), `--chat`, `--stream true`, `--print_perf_metrics` |
| Sampling | pass `--sampler chain` first, then `--do_sample`, `--temperature`, `--top_k`, `--top_p`, `--min_p`, `--repetition_penalty`, `--ignore_eos` |
| Model | serving bundle from [refinefuture-ai/SmolVLM2-500M-Video-Instruct-REFFT](https://huggingface.co/refinefuture-ai/SmolVLM2-500M-Video-Instruct-REFFT); the binary's built-in builders are listed by `cli --list_models` |

### 3. Chat with the server

Once `refft-hexagon serve` is up:

```bash
curl http://127.0.0.1:8080/health
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "SmolVLM2-500M-Video-Instruct",
    "messages": [{"role": "user", "content": "Explain KV cache reuse briefly."}],
    "max_tokens": 128,
    "stream": false
  }'
```

The endpoint is OpenAI-compatible, so any OpenAI client works by pointing its base URL at `http://127.0.0.1:8080/v1`.

***

# FAQs

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
	https://epoch.ai/gradient-updates/why-benchmarking-is-hard <br/>
	https://blog.vllm.ai/2025/10/28/Kimi-K2-Accuracy.html
</details>

***

<a name="legacy-archived"></a>

# Legacy (Archived)

The previous generation, the old QNN / CUDA / MLX packages built around the `refft-cli` / `refft-server` binaries, the training flow and the build tools showcase, is no longer part of the supported surface and has been archived:

- [docs/legacy/](docs/legacy/README.md) -- retired packages and their usage
- [docs/legacy/install_and_usage.md](docs/legacy/install_and_usage.md) -- the old install and usage guide
- [docs/legacy/training.md](docs/legacy/training.md) -- training of LLM/LM: datasets, `refft train`, sample output
- [docs/legacy/build_tools.md](docs/legacy/build_tools.md) -- the `refft.cpp` build tools showcase

CUDA and Apple Silicon are not going away: they return as the single-binary [`refft-cuda` and `refft-mlx`](#runtimes) runtimes. For anything current, use `refft-hexagon` as described in [Quick Start](#quick-start).

# Contact Us

Please contact us via [haiteng@refinefuture.ai](mailto:haiteng@refinefuture.ai) for commercial uses, technical consulting, sponsorship/partnership opportunities, etc. 

# Acknowledgment

`refft.cpp` was inspired by Andrej Karpathy' [llm.c](https://github.com/karpathy/llm.c), and also referred to [HuggingFace](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
