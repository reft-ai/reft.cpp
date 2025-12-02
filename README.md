# reft.cpp
![reft cc-new-logo jpg](https://github.com/user-attachments/assets/25f0c2e7-0f64-41e9-979d-ddb0ff932c4d)


<!--[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)-->
[![Release](https://img.shields.io/github/v/release/reft-ai/reft.cpp)](https://github.com/reft-ai/reft.cpp/releases)
[![Build](https://github.com/reft-ai/reft/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/reft-ai/reft/actions/workflows/release.yml)
<!--[![Publish](https://github.com/reft-ai/reft.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/reft-ai/reft/actions/workflows/docker-publish.yml))-->

# What is `reft.cpp`

A high-performance and easy-to-use LLM/LM serving tools for both Inference and Training. 
  - 20%+ higher Inference and Training performance than other LLM deployments. All of Ops/Ops-Fusion/Ops-Optimization, serving framework and training tools in C++ without Python/PyTorch, inspired by [llm.c](https://github.com/karpathy/llm.c) of Andrej Karpathy.
  - An excutable file of "reft" with model weights is all you need to run the reft-supported LLMs on your GPU(s).

Reft deliveralbes are for enterprises, institutes, individuals, GPU/NPU chipset vendors, AIDC, who are seeking for the higher performance, cost-efficient, easy-to-use of LLM/LMs deployment.


## Supported models

- :white_check_mark: : Done
- :coffee: : To-Do

### LLM

#### Qwen

<table>
	<thead>
		<tr>
			<th>
				<strong>Architecture</strong>
			</th>
			<th>
				<strong>Model</strong>
			</th>
			<th>
				<strong>Device</strong>
			</th>
			<th>
				<strong>Release Tag</strong>
			</th>
			<th>
				<strong>HuggingFace Model</strong>
			</th>
		</tr>
	</thead>
	<tbody>
		<!--------------------- Qwen3/Qwen3-0.6B --------------------->
		<tr>
			<td align="center" rowspan="4">
				<strong>Qwen3</strong>
			</td>
			<td align="center">
				Qwen3-0.6B
			</td>
			<td>
				<table>
					<tr>
						<td>Nvidia GPU</td>
						<td>
							<table>
								<tr><td>RTX 3060(12GB) x1</td></tr>
								<tr><td>RTX 4090(24GB) x1</td></tr>
							</table>
						</td>
					</tr>
					<tr>
						<td>AMD GPU</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Qualcomm Hexagon</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Huawei Asend</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Apple NPU</td>
						<td>TBD</td>
					</tr>
				</table>
			</td>
			<td align="center">
				qwen3-0.6b
			</td>
			<td align="center">
				<a href=https://huggingface.co/Qwen3/Qwen3-0.6B target="_blank">Qwen3-0.6B</a>
			</td>
		</tr>
		<!--------------------- Qwen3/Qwen3-1.7B --------------------->
		<tr>
			<td align="center">
				Qwen3-1.7B
			</td>
			<td>
				<table>
					<tr>
						<td>Nvidia GPU</td>
						<td>
							<table>
								<tr><td>RTX 3060(12GB) x1</td></tr>
								<tr><td>RTX 4090(24GB) x1</td></tr>
							</table>
						</td>
					</tr>
					<tr>
						<td>AMD GPU</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Qualcomm Hexagon</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Huawei Asend</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Apple NPU</td>
						<td>TBD</td>
					</tr>
				</table>
			</td>
			<td align="center">
				qwen3-1.7b
			</td>
			<td align="center">
				<a href=https://huggingface.co/Qwen3/Qwen3-1.7B target="_blank">Qwen3-1.7B</a>
			</td>
		</tr>
		<!--------------------- Qwen3/Qwen3-4B --------------------->
		<tr>
			<td align="center">
				Qwen3-4B <br/>
				Qwen3-4B-Base-2507 <br/>
				Qwen3-4B-Instruct-2507 <br/>
				Qwen3-4B-Thinking-2507
			</td>
			<td>
				<table>
					<tr>
						<td>Nvidia GPU</td>
						<td>
							<table>
								<tr><td>RTX 3060(12GB) x1</td></tr>
								<tr><td>RTX 4090(24GB) x1</td></tr>
							</table>
						</td>
					</tr>
					<tr>
						<td>AMD GPU</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Qualcomm Hexagon</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Huawei Asend</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Apple NPU</td>
						<td>TBD</td>
					</tr>
				</table>
			</td>
			<td align="center">
				qwen3-4b
			</td>
			<td align="center">
				<a href=https://huggingface.co/Qwen3/Qwen3-4B target="_blank">Qwen3-4B</a> <br/>
				<a href=https://huggingface.co/Qwen3/Qwen3-4B-Base-2507 target="_blank">Qwen3-4B-Base-2507</a> <br/>
				<a href=https://huggingface.co/Qwen3/Qwen3-4B-Instruct-2507 target="_blank">Qwen3-4B-Instruct-2507</a> <br/>
				<a href=https://huggingface.co/Qwen3/Qwen3-4B-Thinking-2507 target="_blank">Qwen3-4B-Thinking-2507</a>
			</td>
		</tr>
		<!--------------------- Qwen3/Qwen3-8B --------------------->
		<tr>
			<td align="center">
				Qwen3-8B <br/>
				Qwen3-8B-Base
			</td>
			<td>
				<table>
					<tr>
						<td>Nvidia GPU</td>
						<td>
							<table>
								<tr><td>RTX 3060(12GB) x1</td></tr>
								<tr><td>RTX 4090(24GB) x1</td></tr>
							</table>
						</td>
					</tr>
					<tr>
						<td>AMD GPU</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Qualcomm Hexagon</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Huawei Asend</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Apple NPU</td>
						<td>TBD</td>
					</tr>
				</table>
			</td>
			<td align="center">
				qwen3-8b
			</td>
			<td align="center">
				<a href=https://huggingface.co/Qwen3/Qwen3-8B target="_blank">Qwen3-8B</a> <br/>
				<a href=https://huggingface.co/Qwen3/Qwen3-8B-Base target="_blank">Qwen3-8B-Base</a> <br/>
			</td>
		</tr>
	</tbody>
</table>

#### Llama

<table>
	<thead>
		<tr>
			<th>
				<strong>Architecture</strong>
			</th>
			<th>
				<strong>Model</strong>
			</th>
			<th>
				<strong>Device</strong>
			</th>
			<th>
				<strong>Release Tag</strong>
			</th>
			<th>
				<strong>HuggingFace Model</strong>
			</th>
		</tr>
	</thead>
	<tbody>
		<!--------------------- Llama/Llama-3.2-1B --------------------->
		<tr>
			<td align="center" rowspan="3">
				<strong>Llama</strong>
			</td>
			<td align="center">
				Llama-3.2-1B <br/>
				Llama-3.2-1B-Instruct
			</td>
			<td>
				<table>
					<tr>
						<td>Nvidia GPU</td>
						<td>
							<table>
								<tr><td>RTX 3060(12GB) x1</td></tr>
								<tr><td>RTX 4090(24GB) x1</td></tr>
							</table>
						</td>
					</tr>
					<tr>
						<td>AMD GPU</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Qualcomm Hexagon</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Huawei Asend</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Apple NPU</td>
						<td>TBD</td>
					</tr>
				</table>
			</td>
			<td align="center">
				llama-3.2-1b
			</td>
			<td align="center">
				<a href=https://huggingface.co/meta-llama/Llama-3.2-1B target="_blank">Llama-3.2-1B</a> <br/>
				<a href=https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct target="_blank">Llama-3.2-1B-Instruct</a>
			</td>
		</tr>
		<!--------------------- Llama/Llama-3.2-3B --------------------->
		<tr>
			<td align="center">
				Llama-3.2-3B <br/>
				Llama-3.2-3B-Instruct
			</td>
			<td>
				<table>
					<tr>
						<td>Nvidia GPU</td>
						<td>
							<table>
								<tr><td>RTX 3060(12GB) x1</td></tr>
								<tr><td>RTX 4090(24GB) x1</td></tr>
							</table>
						</td>
					</tr>
					<tr>
						<td>AMD GPU</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Qualcomm Hexagon</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Huawei Asend</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Apple NPU</td>
						<td>TBD</td>
					</tr>
				</table>
			</td>
			<td align="center">
				llama-3.2-3b
			</td>
			<td align="center">
				<a href=https://huggingface.co/meta-llama/Llama-3.2-3B target="_blank">Llama-3.2-3B</a> <br/>
				<a href=https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct target="_blank">Llama-3.2-3B-Instruct</a>
			</td>
		</tr>
		<!--------------------- Llama/Llama-3.1-8B --------------------->
		<tr>
			<td align="center">
				Llama-3.1-8B <br/>
				Llama-3.1-8B-Instruct
			</td>
			<td>
				<table>
					<tr>
						<td>Nvidia GPU</td>
						<td>
							<table>
								<tr><td>RTX 3060(12GB) x1</td></tr>
								<tr><td>RTX 4090(24GB) x1</td></tr>
							</table>
						</td>
					</tr>
					<tr>
						<td>AMD GPU</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Qualcomm Hexagon</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Huawei Asend</td>
						<td>TBD</td>
					</tr>
					<tr>
						<td>Apple NPU</td>
						<td>TBD</td>
					</tr>
				</table>
			</td>
			<td align="center">
				llama-3.1-8b
			</td>
			<td align="center">
				<a href=https://huggingface.co/meta-llama/Llama-3.1-8B target="_blank">Llama-3.1-8B</a> <br/>
				<a href=https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct target="_blank">Llama-3.1-8B-Instruct</a>
			</td>
		</tr>
	</tbody>
</table>


  | Models                | Nvidia GPU | AMD GPU | Qualcomm Hexagon | Huawei Asend | Apple NPU |
  |:---------------------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|
  |[Qwen3-30B-A3B/30B-A3B-Instruct-2507/30B-A3B-Thinking-2507/235B-A22B/235B-A22B/235B-A22B-Instruct-2507/235B-A22B-Thinking-2507](https://huggingface.co/collections/Qwen/qwen3)|:coffee:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)|:coffee:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)|:coffee:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[DeepSeek-R1-Distill-Qwen-1.5/7/14/32B](https://huggingface.co/collections/deepseek-ai/deepseek-r1)|RTX4090(24GB) <br/> A800(80GB) <br/> H200(141GB) <br/>  :white_check_mark: <br/>  [Download](#11-download-reft-docker-image)|:coffee:|:coffee:|:coffee:|:coffee:|
  |[GPT-OSS-20B/120B](https://huggingface.co/collections/deepseek-ai/deepseek-r1)|:coffee:|:coffee:|:coffee:|:coffee:|:coffee:|

	
### Vision LM

  | Models                | Nvidia GPU | AMD GPU | Qualcomm Hexagon | Huawei Asend | Apple NPU |
  |:---------------------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|
  |[SAM](https://github.com/facebookresearch/segment-anything)| :coffee: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[ViT](https://github.com/google-research/vision_transformer)| :coffee: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[Qwen3-VL-2B/4B/4B-Instruct/8B/8B-Instruct/32B](https://huggingface.co/collections/Qwen/qwen3-vl)|:coffee:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[Qwen3-VL-30B-A3B/235B-A22B](https://huggingface.co/collections/Qwen/qwen3-vl)|:coffee:|:coffee:|:coffee:|:coffee:|:coffee:|
  
	
	
### Audio LM

  | Models                | Nvidia GPU | AMD GPU | Qualcomm Hexagon | Huawei Asend | Apple NPU |
  |:---------------------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|
  |[Whisper](https://huggingface.co/collections/openai/whisper-release)| :coffee: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[OpenVoice](https://huggingface.co/myshell-ai/OpenVoice)| :coffee: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[MeloTTS-English/...](https://huggingface.co/myshell-ai/MeloTTS-English)| :coffee: | :coffee: | :coffee: | :coffee: | :coffee: |


# Install and Run LLM

> To run a LLM/LM on your on-premises or cloud GPUs, all you need is a Reft .exe and weights without PyTorch/Python or related.

<br/>

Example model: `Qwen3/Qwen3-4B`

#

## 1. Download reft .exe and weights


### 1.1 Download reft docker image

```bash
# Qwen3/Qwen3-4B
docker pull ghcr.io/reft-ai/reft:qwen3-4b
```

### 1.2 Download model weights from HuggingFace

```bash
mkdir -p models
hf download Qwen3/Qwen3-4B --load-dir ./models
```

## 2. Run LLM

#### Command

```sh
docker run --rm -it --gpus all --net=host --ipc=host \
  -v ./models:/workspace/models ghcr.io/reft-ai/reft:latest \
  reft serve \
  --model /workspace/models/Qwen3/Qwen3-4B \
  --served_model_name Qwen3-4B \
  --chat_template qwen3
```

#### Output

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

### 2.1 Chat via WebUI

Open url: http://127.0.0.1:8888/ui.html


### 2.2 Chat via CLI

##### Command

```sh
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

##### Output

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

# Training

## Download some interesting datasets

```bash
# Exmaple datasets: `CCI-3-HQ` and `Alpaca GPT4`
modelscope download --dataset BAAI/CCI3-HQ --local_dir ./datasets/BAAI/CCI3-HQ
modelscope download --dataset AI-ModelScope/alpaca-gpt4-data-en --local_dir ./datasets/AI-ModelScope/alpaca-gpt4-data-en
modelscope download --dataset AI-ModelScope/chinese-fineweb-edu-v2 --local_dir ./datasets/AI-ModelScope/chinese-fineweb-edu-v2
# optional
modelscope download --dataset HuggingFaceFW/fineweb-edu --local_dir ./datasets/HuggingFaceFW/fineweb-edu
```

## Choose a model and perform fine-tuning on it

```bash
# Example model: Qwen/Qwen3-0.6B
mkdir -p output
docker run -it --rm --gpus all --net=host --ipc=host \
	-v ./models:/workspace/models -v ./output:/output -v ./datasets:/datasets ghcr.io/reft-ai/reft:qwen3-0.6b \
	reft train \
	--cutoff_len 512 \
	--model /workspace/models/Qwen/Qwen3-0.6B \
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
	--checkpoint_dir /output/checkpoints/sft-qwen3-0.6b-full \
	--save_every 20000 \
	--grad_accumulation_steps 32 \
	--resume \
	--load_pretrained \
	--datasets cci3@/datasets/BAAI/CCI3-HQ/data \
	--datasets alpaca@/datasets/AI-ModelScope/alpaca-gpt4-data-en/alpaca-gpt4-data-en.json \
	--datasets fineweb@/datasets/AI-ModelScope/chinese-fineweb-edu-v2/data \
	--datasets fineweb@/datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2025-26
```
### Output

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
