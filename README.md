<!--[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)-->
[![Release](https://img.shields.io/github/v/release/refinefuture-ai/refft.cpp)](https://github.com/refinefuture-ai/refft.cpp/releases)
[![Build](https://github.com/refinefuture-ai/reft/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/refinefuture-ai/reft/actions/workflows/release.yml)
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
	<img width="968" height="466" alt="efe50873e7a96490e9168bed1e740e35" src="https://github.com/user-attachments/assets/102bdeb2-4c63-4729-84b3-ffaa6d0f8cb4" />
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

## :tada: refft.cpp build tools

<p align="center">
	<a href="https://refinefuture.ai" target="_blank">
		<!--<img width="2736" height="1650" alt="image" src="https://github.com/user-attachments/assets/1002af12-906d-467e-841a-9b63a5b7e45f" />-->
		<!-- <img width="1024" alt="Native Model Compiler" src="https://github.com/user-attachments/assets/13a71287-511e-490b-b262-38902bb60485" /> -->
		<img width="1024" alt="Reft Builder" src="https://github.com/user-attachments/assets/58332e55-fd88-4274-9256-4b6cae57fdeb" />
	</a>
</p>
`Click and jump to the tools webpage`

***

<a name="using"></a>

# :rocket: Inference of LLM/LM

`refft.cpp` build tools can make the executable files as the following examples

## Quick Start

#### Minimal CLI usage:

```bash
./bin/refft-cli --model qwen3 --model_dir /path/to/model --prompt "Who are you?" --max_new_tokens 64
```

If the binary was built for a fixed matrix tuple, users normally do not need to
repeat backend / runner / precision flags.

Useful options:

- `--ignore_eos`
- `--do_sample`
- `--temperature`
- `--top_k`
- `--top_p`
- `--speculative_mode ngram`
- `--speculative_max_draft_tokens 4`
- `--speculative_ngram_size 3`

#### Server Usage

Minimal server usage:

```bash
./bin/refft-server --model qwen3 --model_dir /path/to/model --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

OpenAI-compatible chat request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Explain KV cache reuse briefly."}],
    "max_tokens": 128,
    "stream": false
  }'
```

Full install and usage guide:

- [Install And Usage](docs/install_and_usage.md)


### For QNN
|        Model Package       |         Description |
|----------------------------|---------------------|
| [refft-android-aarch64-qnn-qwen3](https://github.com/refinefuture-ai/refft.cpp/releases/download/github_draft_20260401/android-aarch64-qnn-qwen3-dynamic-fp16.tar.gz) | 0.6B/1.7B/4B/8B/14B/32B supported <br/> FlashAtttion ops supported <br/> Quantization can be set to w4a16, w8a16, w4afp16, w8afp16, fp16 and default is fp16<br/> Tested on OnePlus15/SM8850/16GB-DDR|
| [refft-android-aarch64-qnn-qwen3-moe](https://github.com/refinefuture-ai/refft.cpp/releases/download/github_draft_20260401/android-aarch64-qnn-qwen3moe-fa_moe_hybrid-fp16.tar.gz) | 30B-A3B supported <br/> MoE, FlashAtttion ops supported <br/> TP supported for multi-HTPs backends <br/> Quantization can be set to w4a16, w8a16, w4afp16, w8afp16, fp16 and default is fp16 <br/> Tested on OnePlus15/SM8850/16GB-DDR|


### For Nvidia

|        Model Package       |         Description |
|----------------------------|---------------------|
| [refft-linux-x64-cuda-qwen3-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-linux-x64-cuda-qwen3-20260323.tar.xz) | 0.6B/1.7B/4B/8B/14B/32B supported |
| [refft-ubuntu2404-x64-cuda-qwen3-20260323.deb](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-ubuntu2404-x64-cuda-qwen3-20260323.deb) | 0.6B/1.7B/4B/8B/14B/32B supported |
| [refft-linux-x64-cuda-qwen3-moe-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-linux-x64-cuda-qwen3-moe-20260323.tar.xz) | 30B-A3B/235B-A22B supported |
| [refft-ubuntu2404-x64-cuda-qwen3-moe-20260323.deb](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-ubuntu2404-x64-cuda-qwen3-moe-20260323.deb) | 30B-A3B/235B-A22B supported |

**Note:** Please contact us for multi-nodes support


### For Apple Silicon

|        Model Packcage      |         Description |
|----------------------------|---------------------|
| [refft-macos-arm64-mlx-qwen3-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-macos-arm64-mlx-qwen3-20260323.tar.xz) | 0.6B/1.7B/4B/8B/14B/32B supported |
| [refft-macos-arm64-mlx-qwen3-moe-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-macos-arm64-mlx-qwen3-moe-20260323.tar.xz) | 30B-A3B/235B-A22B supported |


***

<a name="training"></a>

# :rocket: Training of LLM/LM

<details>
	<summary>Download the public datasets or use your own datasets</summary>

```bash
# Exmaple datasets: `CCI-3-HQ`, `Alpaca GPT4` and `FineWeb`

hf download HuggingFaceFW/finepdfs-edu --repo-type=dataset --local-dir ./datasets/HuggingFaceFW/fineweb-edu
hf download BAAI/CCI3-HQ --repo-type=dataset --local-dir ./datasets/BAAI/CCI3-HQ
hf download llamafactory/alpaca_gpt4_en --repo-type=dataset --local-dir ./datasets/llamafactory/alpaca_gpt4_en
```

</details>

<details>
	<summary>Train LLM via Pre-train/full-SFT/freeze-SFT/LoRA/RL</summary>

```bash
mkdir -p output
refft train \
	--cutoff_len 512 \
	--model ./models/Qwen/Qwen3-0.6B \
	--block_size 512 \
	--test_every 200 \
	--batch_size 4 \
	--fine_tuning_type full \
	--weight_decay 0.1 \
	--warmup_steps 100 \
	--lr_scheduler_type step \
	--learning_rate 4e-5 \
	--epochs 100 \
	--learning_rate_decay_frac 0.0 \
	--use_bf16 \
	--stage sft \
	--checkpoint_dir ./output/checkpoints/sft-Qwen3-0.6B-full \
	--save_every 20000 \
	--grad_accumulation_steps 32 \
	--resume \
	--load_pretrained \
    --tensor_parallels 1 \
    --pipeline_parallels 1 \
    --data_parallels 1 \
    --nodes 1 \
    --gpus_per_node 1 \
	--chat_template qwen3 \
	--datasets cci3@./datasets/BAAI/CCI3-HQ/data \
	--datasets alpaca@./datasets/llamafactory/alpaca_gpt4_en/alpaca-gpt4-data-en.json \
	--datasets fineweb@./datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2025-26
```

</details>

<details>
	<summary>Output</summary>

```bash
[1][2025-11-30 09:20:15][I][         train_main.cc: 186]  Reft: v1.0.0, 5301f2a4fb303fd647fe783aa326522efde8ceb4
[1][2025-11-30 09:20:15][I][         train_main.cc: 187]  Build Time: Sun Nov 30 08:37:07 CST 202
[2025-11-30 09:20:15.895] [info] Apply chat template: qwen2
[1][2025-11-30 09:20:15][I][         train_main.cc: 525]  [0/1] Building tokenizer ...
[2025-11-30 09:20:16.102] [info] Vocab size: 151669
[2025-11-30 09:20:16.102] [info] ids: [[9707,1879,0]
[1][2025-11-30 09:20:16][I][sequence_dataloader_builders.cc: 108]  URL: huatuo@/assets/data/huatuo-100.jsonl
 ████████████████████████████████████████▏ 100.0% [ 101/ 101 | 84.1 kHz | 0s<0s] Parsing lines ...
 ████████████████████████████████████████▏ 100.0% [ 100/ 100 | 127.3 kHz | 0s<0s] Loading dataset ...
[1][2025-11-30 09:20:16][I][sequence_dataloader_builders.cc: 108]  URL: alpaca@/assets/data/alpaca-style/reft_ai.json
 ████████████████████████████████████████▏ 100.0% [   5/   5 | 6.6 kHz | 0s<0s] Loading dataset ...
[1][2025-11-30 09:20:16][I][sequence_dataloader_builders.cc:  24]  Dataset has 200 examples in total

[2025-11-30 09:20:16.115] [info] Found the loader for architecture: Qwen3ForCausalLM
[2025-11-30 09:20:16.115] [info] KV cache block size: 512
[2025-11-30 09:20:16.115] [info] KV cache allocator is created
[2025-11-30 09:20:16.124] [info] Creating Qwen model ...
 ████████████████████████████████████████▏ 100.0% [  28/  28 | 12.9 kHz | 0s<0s] Construct blocks ...
 ████████████████████████████████████████▏ 100.0% [ 311/ 311 | 29.3 Hz | 11s<0s]
[2025-11-30 09:20:26.787] [info] Weights are loaded
[1][2025-11-30 09:20:26][I][         train_main.cc: 658]  [0/1] Model loaded
[2025-11-30 09:20:26.801] [info] Last done steps: 0
[1][2025-11-30 09:20:26][I][         train_main.cc: 737]  [0/1] Let's start training now!!!
[1][2025-11-30 09:20:26][I][         train_main.cc: 740]  [0/1] Fine tuning type: full
[1][2025-11-30 09:20:26][I][         train_main.cc: 753]  [0/1] Building trainer ...
[1][2025-11-30 09:20:26][I][         train_main.cc: 831]  [0/1] Trainer is ready
[1][2025-11-30 09:20:26][I][        sft_trainer.cc:  24]  ++++++++++++++++++++++ Training +++++++++++++++++++++++
[1][2025-11-30 09:20:26][I][        sft_trainer.cc:  25]  Start from steps: 0, total steps: 5000, total epochs: 100
[1][2025-11-30 09:20:26][I][        sft_trainer.cc:  27]  Options: ignore_idx: 151643, grad_accumulate_steps: 32
[1][2025-11-30 09:20:26][I][        sft_trainer.cc:  30]  Resuming dataloader to 0 ...
[1][2025-11-30 09:20:27][I][        sft_trainer.cc: 170]  [0/1] [0/5000] loss: 3.64062, lr: 0.0000400, seq_len: 288
[1][2025-11-30 09:20:27][I][        sft_trainer.cc: 170]  [0/1] [1/5000] loss: 2.12500, lr: 0.0000400, seq_len: 384
[1][2025-11-30 09:20:27][I][        sft_trainer.cc: 170]  [0/1] [2/5000] loss: 1.88281, lr: 0.0000400, seq_len: 360
[1][2025-11-30 09:20:27][I][        sft_trainer.cc: 170]  [0/1] [3/5000] loss: 1.42969, lr: 0.0000400, seq_len: 384
[1][2025-11-30 09:20:27][I][        sft_trainer.cc: 170]  [0/1] [4/5000] loss: 1.96875, lr: 0.0000400, seq_len: 512
[1][2025-11-30 09:20:27][I][        sft_trainer.cc: 170]  [0/1] [5/5000] loss: 1.08594, lr: 0.0000400, seq_len: 320
[1][2025-11-30 09:20:27][I][        sft_trainer.cc: 170]  [0/1] [6/5000] loss: 1.39062, lr: 0.0000400, seq_len: 384
[1][2025-11-30 09:20:27][I][        sft_trainer.cc: 170]  [0/1] [7/5000] loss: 1.75781, lr: 0.0000400, seq_len: 384
[1][2025-11-30 09:20:28][I][        sft_trainer.cc: 170]  [0/1] [8/5000] loss: 1.57812, lr: 0.0000400, seq_len: 512
[1][2025-11-30 09:20:28][I][        sft_trainer.cc: 170]  [0/1] [9/5000] loss: 1.12500, lr: 0.0000400, seq_len: 384
```

</details>

<br/>

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


# Contact Us

Please contact us via [haiteng@refinefuture.ai](mailto:haiteng@refinefuture.ai) for commercial uses, technical consulting, sponsorship/partnership opportunities, etc. 

# Acknowledgment

`refft.cpp` was inspired by Andrej Karpathy' [llm.c](https://github.com/karpathy/llm.c), and also referred to [HuggingFace](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer).

