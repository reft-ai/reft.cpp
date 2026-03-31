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
	<img width="1024" height="510" alt="Refft Builder" src="https://github.com/user-attachments/assets/2cdb49b0-6496-46f7-8dbe-997a7430c160" />
	<!-- <img width="1024" alt="Refft Builder" src="https://github.com/user-attachments/assets/9e34ac36-c653-4987-8846-66c7e539b644" /> -->
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


### For QNN
|            Tool            |         Description |
|----------------------------|---------------------|
| android-aarch64-qnn-qwen3 | 0.6B/1.7B/4B/8B/14B/32B supported, Tested on OnePlus15/SM8850/16GB-DDR|
| android-aarch64-qnn-qwen3-moe | 30B-A3B supported <br/> Notes: MoE, FlashAtttion ops supported; TP supported for multi-HTPs backends; Quantization can be set to w4a16, w8a16, w4afp16, w8afp16, fp16 and default is fp16, Tested on OnePlus15/SM8850/16GB-DDR|



<details>
	<summary>Intall & Run</summary>

```bash
# Download model weights
mkdir -p models
hf download Qwen3/Qwen3-30B-A3B-Instruct-2507 --load-dir ./models
adb push ./models/Qwen3-30B-A3B-Instruct-2507 /data/local/tmp/

# Install
tar -xzf android-aarch64-qnn-qwen3moe-fa_moe_hybrid-fp16.tar.gz

# Run CLI
cd android-aarch64-qnn-qwen3moe-fa_moe_hybrid-fp16
ANDROID_DST=/data/local/tmp/refft_release/android-aarch64-qnn-qwen3moe-fa_moe_hybrid-fp16 bash ./scripts/run_cli.sh --model_dir /data/local/tmp/Qwen3-30B-A3B-Instruct-2507 --prompt "Who are you?" --max_new_tokens 1

# Launch server
cd android-aarch64-qnn-qwen3moe-fa_moe_hybrid-fp16
ANDROID_DST=/data/local/tmp/refft_release/android-aarch64-qnn-qwen3moe-fa_moe_hybrid-fp16 bash ./scripts/run_server.sh --model_dir /data/local/tmp/Qwen3-30B-A3B-Instruct-2507 --port 18080
```

</details>


### For Nvidia

|            Tool            |         Description |
|----------------------------|---------------------|
| [refft-linux-x64-cuda-qwen3-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-linux-x64-cuda-qwen3-20260323.tar.xz) | 0.6B/1.7B/4B/8B/14B/32B supported |
| [refft-ubuntu2404-x64-cuda-qwen3-20260323.deb](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-ubuntu2404-x64-cuda-qwen3-20260323.deb) | 0.6B/1.7B/4B/8B/14B/32B supported |
| [refft-linux-x64-cuda-qwen3-moe-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-linux-x64-cuda-qwen3-moe-20260323.tar.xz) | 30B-A3B/235B-A22B supported |
| [refft-ubuntu2404-x64-cuda-qwen3-moe-20260323.deb](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-ubuntu2404-x64-cuda-qwen3-moe-20260323.deb) | 30B-A3B/235B-A22B supported |

<details>
	<summary>Intall & Run</summary>


**Note:** Please contact us for multi-nodes support

```bash
# Install
tar Jxf ./refft-linux-x64-cuda-qwen3-20260323.tar.xz
sudo cp refft-linux-x64-cuda-qwen3-20260323/bin/refft /usr/bin/refft
# or
sudo apt install ./refft-ubuntu2404-x64-cuda-qwen3-20260323.deb

# Download model weights
mkdir -p models
hf download Qwen3/Qwen3-0.6B --load-dir ./models

# Launch server
refft serve \
  --model /workspace/models/Qwen3/Qwen3-0.6B \
  --served_model_name Qwen3-0.6B
```

```text
  ████████████████████████████████████████▏ 100.0% [ 199/ 199 | 476.2 Hz | 0s<0s]  
[2025-11-11 07:02:50.007] [GPUModelRunner(rank=0)][7] [info] Model loaded
[2025-11-11 07:02:50.007] [GPUWorker(rank=0)][7] [info] GPU Mem Info(free=18.4 GB, total=23.5 GB)
[2025-11-11 07:02:50.007] [Serve][1] [info] Available GPU Memory: 14.4 GB
[2025-11-11 07:02:50.007] [Serve][1] [info] max_num_layers: 1
[2025-11-11 07:02:50.008] [Serve][1] [info] All layers have the same page size: 1835008 bytes
[2025-11-11 07:02:50.008] [Serve][1] [info] GPU KV cache: 8438 blocks
[2025-11-11 07:02:50.008] [Serve][1] [info] GPU KV cache size: 540032 tokens
[2025-11-11 07:02:50.008] [Serve][1] [info] Maximum concurrency for 4096 tokens per request: 131.84x
[2025-11-11 07:02:50.008] [Serve][1] [info] The KV cache size required by each layer: 15483797504 bytes
[2025-11-11 07:02:50.008] [GPUModelRunner(rank=0)][7] [info] Initialize KV cache | num_blocks: 8438
[2025-11-11 07:02:50.009] [GPUModelRunner(rank=0)][7] [info] CUDA graph capture sizes: 1
[2025-11-11 07:02:50.152] [GPUModelRunner(rank=0)][7] [info] Graph test begin, size: 1
[2025-11-11 07:02:50.243] [GPUModelRunner(rank=0)][7] [info] Graph test passed, size: 1
[2025-11-11 07:02:50.243] [GPUModelRunner(rank=0)][7] [info] Graph capturing finished in 0 secs, took 0.04 GiB
[2025-11-11 07:02:50.243] [Serve][1] [info] Init engine (profile, create_kv_cache, warmup model) took 0.23 seconds
[2025-11-11 07:02:50.244] [Serve][1] [info] Starting API server ...
[2025-11-11 07:02:50.245] [Serve][1] [info] HTTP server listening on 0.0.0.0:8888 ...
```

```text
[2026-03-19 22:43:13.499] [info] [17261] [main] QnnDsp <V> RPC Memory mapped for Weights 0x7636400000 [1193279488 B] @ VA af000000
[2026-03-19 22:43:13.674] [info] [17261] [main] QnnDsp <V> Successfully copied 1192255488 bytes of weights! Actual bytes to transfer: 1192255488
[2026-03-19 22:43:13.674] [info] [17261] [main] QnnDsp <V> 13979880 isInit 1
[2026-03-19 22:43:13.674] [info] [17261] [main] QnnDsp <V> New serialized binary size = 13979880
[2026-03-19 22:43:13.676] [info] [17261] [main] QnnDsp <V> 40 isInit 1
[2026-03-19 22:43:13.677] [info] [17261] [main] QnnDsp <V> Found transport session 0xb4000079897a9d98 for deviceId 0x0 coreId 0x0 pdId 0x0!
[2026-03-19 22:43:13.677] [info] [17261] [main] QnnDsp <V> Found transport session 0xb4000079897a9d98 for deviceId 0x0 coreId 0x0 pdId 0x0!
[2026-03-19 22:43:13.899] [info] [17261] [main] QnnDsp <V> transport run [status = 0]
[2026-03-19 22:43:13.912] [info] [17261] [main] QnnDsp <V> HtpTransport::graphPrepareDsp done. graph.m_hexNNGraphHandle = 12970367443114328064
[2026-03-19 22:43:14.978] [info] [17261] [main] graph_prepare.cc:762:STAT: crated_byte_estimate=16674200
[2026-03-19 22:43:18.774] [info] [17261] [main] fa_alloc.cc:2437:STAT: alloca_pickle_base=13979648
[2026-03-19 22:43:18.774] [info] [17261] [main] fa_alloc.cc:2438:STAT: alloca_pickle_const_extent=1192255488
[2026-03-19 22:43:18.774] [info] [17261] [main] fa_alloc.cc:2439:STAT: alloca_io_tensor=15187968
[2026-03-19 22:43:18.774] [info] [17261] [main] fa_alloc.cc:2440:STAT: alloca_mempool_ddr=20992
[2026-03-19 22:43:18.774] [info] [17261] [main] fa_alloc.cc:2441:STAT: alloca_mempool_spillfill_near=9109504
[2026-03-19 22:43:18.774] [info] [17261] [main] fa_alloc.cc:2442:STAT: alloca_mempool_spillfill_far=0
[2026-03-19 22:43:18.774] [info] [17261] [main] fa_alloc.cc:2443:STAT: alloca_mempool_const=1192251136
[2026-03-19 22:43:18.774] [info] [17261] [main] fa_alloc.cc:2456:STAT: alloca_est_total=1216573952
[2026-03-19 22:43:18.822] [info] [17261] [main] QnnDsp <V> Async property not supported. Skipping register Async context
[2026-03-19 22:43:18.823] [info] [17261] [main] QnnDsp <V> Wake up free backend (id: 1)'s thread(s)
[2026-03-19 22:43:18.823] [info] [17261] [main] QnnDsp <I> QnnGraph_finalize done. status 0x0
[2026-03-19 22:43:18.823] [info] [17261] [main] QnnDsp <V> Deactivated logger with handle 0x1
[2026-03-19 22:43:18.823] [info] [17261] [main] QnnDsp <V> RouterFastRPC terminateNativeOpValidatorLogs
[2026-03-19 22:43:18.823] [info] [17261] [main] Construct rope ...
[2026-03-19 22:43:18.826] [info] [17261] [main] Construct rope end
[2026-03-19 22:43:18.826] [info] [17261] [main] QnnDsp <I> QnnProfile_create 0xb4000078d4df74e0
[2026-03-19 22:43:18.826] [info] [17261] [main] QnnDsp <V> Deactivated logger with handle 0x1
                                        ▏ 100.0% [   0/   0 | 0.0 Hz | 2s<0s]
[2026-03-19 22:43:20.770] [info] [17261] [main] Engine is created
[2026-03-19 22:43:20.770] [info] [17261] [main] Starting API server ...
[2026-03-19 22:43:20.770] [info] [17261] [main] Served model name: [Qwen3-0.6B]
[2026-03-19 22:43:23.610] [info] [17261] [main] Vocab size: 151669
[2026-03-19 22:43:23.617] [info] [17261] [main] Ids of "Hello world!": [[9707,1879,0]
[2026-03-19 22:43:23.621] [info] [17261] [main] Apply chat template:
{% for message in messages %}<|im_start|>{{message['role']}}
{{message['content']}}<|im_end|>{% endfor %}{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}
[2026-03-19 22:43:23.621] [info] [17261] [main] Create LLM text processor
[2026-03-19 22:43:23.785] [info] [17261] [main] Start server
[2026-03-19 22:43:23.794] [info] [17526] [main] HTTP server is listening on 0.0.0.0:8888 ...
[2026-03-19 22:43:23.795] [info] [17526] [main] ServerMonitor is started.
```

## Connect via CLI

```bash
curl -Ns http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
	"model": "Qwen3-0.6B",
	"messages": [{"role":"user", "content": "<｜begin▁of▁sentence｜><｜User｜>Who are you?<｜Assistant｜><think>\\n"}],
	"max_tokens": 24,
	"temperature": 0.6,
	"stream": true
  }'
```

```text
data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-0.6B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"role":"assistant"},"index":0,"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-0.6B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"Greetings"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-0.6B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"!"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-0.6B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":" I"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-0.6B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"'m"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-0.6B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":" Deep"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-0.6B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"Seek"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

...
```

</details>

### For Apple silicon

|            Tool            |         Description |
|----------------------------|---------------------|
| [refft-macos-arm64-mlx-qwen3-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-macos-arm64-mlx-qwen3-20260323.tar.xz) | 0.6B/1.7B/4B/8B/14B/32B supported |
| [refft-macos-arm64-mlx-qwen3-moe-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-macos-arm64-mlx-qwen3-moe-20260323.tar.xz) | 30B-A3B/235B-A22B supported |

<details>
	<summary>Intall & Run</summary>


**Note:** Please contact us for multi-nodes support

```bash
# Install
tar Jxf ./refft-macos-arm64-mlx-qwen3-20260323.tar.xz
sudo cp refft-macos-arm64-mlx-qwen3-20260323/bin/refft /usr/bin/refft

# Download model weights
mkdir -p models
hf download Qwen3/Qwen3-0.6B --load-dir ./models

# Launch server
refft serve \
  --model /workspace/models/Qwen3/Qwen3-0.6B \
  --served_model_name Qwen3-0.6B
```

</details>

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

