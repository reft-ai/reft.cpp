<!--[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)-->
[![Release](https://img.shields.io/github/v/release/reft-ai/reft.cpp)](https://github.com/reft-ai/reft.cpp/releases)
[![Build](https://github.com/reft-ai/reft/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/reft-ai/reft/actions/workflows/release.yml)
<!--[![Publish](https://github.com/reft-ai/reft.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/reft-ai/reft/actions/workflows/docker-publish.yml))-->

<!--![reft cc-new-logo jpg](https://github.com/user-attachments/assets/25f0c2e7-0f64-41e9-979d-ddb0ff932c4d)-->

<!--<center><h3>C++ Native-implemented(without Python/PyTorch) LLM/LM's inference serving and training for High-Performance and Easy-to-Use</h3></center>-->

<img width="3466" height="1308" alt="3311b35fe62743ee47cb7401294aac34" src="https://github.com/user-attachments/assets/1ae5471a-9ceb-4daa-aa91-6dc596639786" />

# About

`reft.cpp` is a toolkit for both inference serving and training in only one native executable file

- 20%++ faster TTFT/TPOT than any Python/PyTorch-based servings in the same quantization/precision
- 0 running dependencies other than Linux and CUDA(non-CUDA counterpart)
- GPUs and edge-NPU supported
- MoE/Dense LLM and VL supported

<img width="880" height="504" alt="509d61229a92e679b0890c9433a95311" src="https://github.com/user-attachments/assets/c0351135-4b4e-4c2f-aa6b-49d8229dae54" />



## :fire: Key Features

- **Native Compile** -- C++ implement&compile for modeling, serving, training
- **OpenAI-Compatible API** -- Seamless integration with existing tools
- **Custom via Plugins** -- Plugins for custom ops and training algorithms
- **Multi-Modal Support** -- Support combined text, image, audio, etc
- **Native vRAM mgt** -- Native mem mgt instead of GC to lower peak occ-mem and alloc-overhead

<!--
- **Configurable Continuous Batching Scheduler** -- Efficient request handling with dynamic batching
- **Paged Attention** -- Optimize mem mgt for long sequences and lower memory footprint
- **Flash Attention -- Optimized mem mgt for long sequences and lower memory footprint
-->

## Supported Models

- :white_check_mark: : Supported
- :coffee: : In-Progress

### :zap: LLM

  |            Models         |     Nvidia GPU     |        AMD GPU       |    Qualcomm Hexagon  |     Apple Silicon   |
  |:-------------------------:|:------------------:|:--------------------:|:--------------------:|:-------------------:|
  |DeepSeek-V3.2 (685B)       | :coffee: |     :coffee:         |       N/A       |        :coffee:     |
  |DeepSeek-V3/R1 (671B)      | :coffee: |     :coffee:         |       N/A       |        :coffee:     |
  |DeepSeek-OCR               | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Qwen3 (0.6B-8B)            | :white_check_mark: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Qwen3-MoE                  | :white_check_mark: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Qwen3-MoE (30B - 235B)     | :white_check_mark: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Qwen3-Next (80B)           | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Qwen2.5 (0.5B - 72B)       | :white_check_mark: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Llama3.2 (1B, 3B)          | :white_check_mark: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Llama3.1 (8B, 70B)         | :white_check_mark: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Llama3 (8B, 70B)           | :white_check_mark: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Gemma (2B - 7B)            | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |GPT-OSS (20B, 120B)        | :coffee: |     :coffee:         |       N/A       |        :coffee:     |

### :zap: Vision LM

  |           Models          |      Nvidia GPU    |        AMD GPU       |   Qualcomm Hexagon   |     Apple Silicon   |
  |:-------------------------:|:------------------:|:--------------------:|:--------------------:|:-------------------:|
  |Qwen3-VL (2B - 8B)         | :white_check_mark: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Qwen3-VL-MoE (30B - 235B)  | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Qwen2.5-VL (3B, 7B, 72B)   | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Llama4 (Scout, Maverick)   | :coffee: |     :coffee:         |       N/A       |        :coffee:     |
  |Llama3.2-vision (11B, 90B) | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |SAM-3D-Objects             | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |SAM-3D-Body                | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |SAM-2                      | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |SAM                        | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  
	
### :zap: Audio LM

  |           Models          |      Nvidia GPU    |        AMD GPU       |   Qualcomm Hexagon   |    Apple Silicon    |
  |:-------------------------:|:------------------:|:--------------------:|:--------------------:|:-------------------:|
  |Whisper                    | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |OpenVoice2                 | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |SAM-Audio                  | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  |Melo-TTS                   | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |


### :zap: Visual Features

  |           Models          |      Nvidia GPU    |        AMD GPU       |   Qualcomm Hexagon   |    Apple Silicon    |
  |:-------------------------:|:------------------:|:--------------------:|:--------------------:|:-------------------:|
  |DINOv2                     | :coffee: |     :coffee:         |       :coffee:       |        :coffee:     |
  

***

# Download, Install and Run LLM/LM

***To run the LLM/LM on your on-premises/cloud GPUs or Edge NPU, all you need is a Reft .exe and weights file without PyTorch/Python related.***

<br/>

Example model: `Qwen3/Qwen3-4B`

<details>
<summary>
Download reft.exe and Weights file (the reft install packages of reft-cuda/reft-rocm/reft-qc/reft-mac are for the Nvidia/AMD/Qualcomm/Apple GPU/NPU respectively)
</summary>

```shell
curl -fsL https://github.com/reft-ai/reft.cpp/releases/download/v1.0.1/reft-cuda_1.0.1-0ubuntu24.04_amd64.deb

mkdir -p models
hf download Qwen3/Qwen3-4B --load-dir ./models
```

</details>

<details>
<summary>Install and Run</summary>

```shell
sudo apt install -y ./reft-cuda_1.0.1-0ubuntu24.04_amd64.deb
```

**Note:** Please contact us for multi-nodes support

```bash
reft serve \
  --model /workspace/models/Qwen3/Qwen3-4B \
  --served_model_name Qwen3-4B
```

</details>

<details>
<summary>Output</summary>

```sh
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

</details>

<details>
<summary>Connect via CLI</summary>

```shell
curl -Ns http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
	"model": "Qwen3-4B",
	"messages": [{"role":"user", "content": "<｜begin▁of▁sentence｜><｜User｜>Who are you?<｜Assistant｜><think>\\n"}],
	"max_tokens": 24,
	"temperature": 0.6,
	"stream": true
  }'
```

output

```text
data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-4B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"role":"assistant"},"index":0,"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-4B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"Greetings"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-4B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"!"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-4B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":" I"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-4B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"'m"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-4B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":" Deep"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"Qwen3-4B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"Seek"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

...
```

</details>

<br/>

***

#  Training

<details>
<summary>Download the public datasets or use your own datasets</summary>

```shell
# Exmaple datasets: `CCI-3-HQ`, `Alpaca GPT4` and `FineWeb`

hf download HuggingFaceFW/finepdfs-edu --repo-type=dataset --local-dir ./datasets/HuggingFaceFW/fineweb-edu
hf download BAAI/CCI3-HQ --repo-type=dataset --local-dir ./datasets/BAAI/CCI3-HQ
hf download llamafactory/alpaca_gpt4_en --repo-type=dataset --local-dir ./datasets/llamafactory/alpaca_gpt4_en
```

</details>

<details>
<summary>Train LLM via Pre-train/full-SFT/freeze-SFT/LoRA/RL</summary>

```shell
mkdir -p output
reft train \
	--cutoff_len 512 \
	--model ./models/Qwen/Qwen3-4B \
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
	--checkpoint_dir ./output/checkpoints/sft-qwen3-4b-full \
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

```shell
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
	<summary>Why reft.cpp implements all of modeling, serving and training in C++</summary>

It's manly for a better performance and easy-to-use compared to Python/PyTorch-based as well as for scalability on edge-NPU.

</details>

<details>
	<summary>Why Triton is not used in reft.cpp</summary>

Because the Triton models can get up to 78% of the performance of the CUDA models on the H100 and up to 82% on the A100.	
	
[CUDA-Free Inference for LLMs](https://pytorch.org/blog/cuda-free-inference-for-llms/)
</details>

<details>
	<summary>How to support multi-nodes GPU/NPU</summary>
Technically reft.cpp supports multi-nodes inference and training, while multi-nodes haven't been tested due to lacking of HW resources. Please contact us if needed.
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

Please contact us via [ai@reft-ai.com](mailto:ai@reft-ai.com) for commercial uses, technical consulting, sponsorship/partnership opportunities, etc. 

# Acknowledgment

`reft.cpp` was inspired by Andrej Karpathy' [llm.c](https://github.com/karpathy/llm.c), and also referred to [HuggingFace](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer).

