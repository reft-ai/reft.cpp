# reft.cpp
![reft cc-new-logo jpg](https://github.com/user-attachments/assets/25f0c2e7-0f64-41e9-979d-ddb0ff932c4d)


[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Release](https://img.shields.io/github/v/release/reft-ai/reft.cpp)](https://github.com/reft-ai/reft.cpp/releases)
<!--[![Publish](https://github.com/reft-ai/reft.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/reft-ai/reft/actions/workflows/docker-publish.yml))-->

# What is `reft.cpp`

- A high-performance and easy-to-use LLM/LM serving tools for inference, training. All of Ops/Ops-Fusion/Ops-Optimization, serving framework and training tools in C++ without Python/PyTorch, inspired by [llm.c](https://github.com/karpathy/llm.c) of Andrej Karpathy.

- An excutable file of "reft" with model weights is all you need to run the reft-supported model on your GPU(s) with the better performance.

- Our deliveralbes are for enterprises, institutes, individuals, GPU/NPU chipset vendors, AIDC, who are seeking for the better performance, cost-efficient, easy-to-use of LLM/LMs.



## Supported models

- :white_check_mark: : Done
- :coffee: : To-Do

### LLM

#### QWEN

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

#### LLAMA

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
  /workspace/reft serve \
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

