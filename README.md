# reft.cpp
![reft cc-new-logo jpg](https://github.com/user-attachments/assets/25f0c2e7-0f64-41e9-979d-ddb0ff932c4d)


[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Release](https://img.shields.io/github/v/release/reft-ai/reft.cpp)](https://github.com/reft-ai/reft.cpp/releases)
<!--[![Publish](https://github.com/reft-ai/reft.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/reft-ai/reft/actions/workflows/docker-publish.yml))-->

# What is `reft.cpp`

- A high-performance and easy-to-use LLM/LM serving tools for inference, training. All of Ops/Ops-Fusion/Ops-Optimization, serving framework and training tools in C++ without Python/PyTorch, inspired by [llm.c](https://github.com/karpathy/llm.c) of Andrej Karpathy.

- An excutable file of "reft" with model weights is all you need to run the reft-supported model on your GPU(s) with the better performance.

- Our deliveralbes are for enterprises, institutes, individuals, GPU/NPU chipset vendors, AIDC, who are seeking for the better performance, cost-efficient, easy-to-use of LLM/LMs.

----


# Supported Models

## DeepSeek

* ### DeepSeek-R1
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA H200(141GB) X8</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-H200 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1 \
		--served_model_name DeepSeek-R1 \
		--chat_template ds-r1 \
 		--tp_size 8 \
 		--world_size 8
	```
	</details>

* ### DeepSeek-V3
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA H200(141GB) X8</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-V3-H200 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-V3 \
		--served_model_name DeepSeek-V3 \
		--chat_template ds-v3 \
 		--tp_size 8 \
 		--world_size 8
	```
	</details>


* ### DeepSeek-R1-Distill-Qwen-1.5B
  	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 3060(12GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-1.5B-RTX3060 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
		--served_model_name DeepSeek-R1-Distill-Qwen-1.5B \
		--chat_template ds-distill-qwen2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 4090(24GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-1.5B-RTX4090 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
		--served_model_name DeepSeek-R1-Distill-Qwen-1.5B \
		--chat_template ds-distill-qwen2
	```
	</details>
	
* ### DeepSeek-R1-Distill-Qwen-7B

	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 3060(12GB) X2</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-7B-RTX3060 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
		--served_model_name DeepSeek-R1-Distill-Qwen-7B \
		--chat_template ds-distill-qwen2 \
 		--tp_size 2 \
 		--world_size 2
	```
	</details>

	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 4090(24GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-7B-RTX4090 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
		--served_model_name DeepSeek-R1-Distill-Qwen-7B \
		--chat_template ds-distill-qwen2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA A800(80GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-7B-A800 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
		--served_model_name DeepSeek-R1-Distill-Qwen-7B \
		--chat_template ds-distill-qwen2
	```
	</details>

* ### DeepSeek-R1-Distill-Qwen-14B

  	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX4090(24GB) X2</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-14B-RTX4090 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
		--served_model_name DeepSeek-R1-Distill-Qwen-14B \
		--chat_template ds-distill-qwen2 \
 		--tp_size 2 \
 		--world_size 2
	```
	</details>

	<details>
		<summary>&nbsp;&nbsp;NVIDIA H200(141GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-14B-H200 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
		--served_model_name DeepSeek-R1-Distill-Qwen-14B \
		--chat_template ds-distill-qwen2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA A800(80GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-14B-A800 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
		--served_model_name DeepSeek-R1-Distill-Qwen-14B \
		--chat_template ds-distill-qwen2
	```
	</details>

* ### DeepSeek-R1-Distill-Qwen-32B

	<details>
		<summary>&nbsp;&nbsp;NVIDIA H200(141GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-32B-H200 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
		--served_model_name DeepSeek-R1-Distill-Qwen-32B \
		--chat_template ds-distill-qwen2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA A800(80GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-DeepSeek-R1-Distill-Qwen-32B-A800 \
		reft-server \
		--model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
		--served_model_name DeepSeek-R1-Distill-Qwen-32B \
		--chat_template ds-distill-qwen2
	```
	</details>


# Qwen

* ### Qwen2.5-0.5B-Instruct

	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 3060(12GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-0.5B-Instruct-RTX3060 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-0.5B-Instruct \
		--served_model_name Qwen2.5-0.5B-Instruct \
		--chat_template qwen2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 4090(24GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-0.5B-Instruct-RTX4090 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-0.5B-Instruct \
		--served_model_name Qwen2.5-0.5B-Instruct \
		--chat_template qwen2
	```
	</details>

* ### Qwen2.5-1.5B-Instruct

	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 3060(12GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-1.5B-Instruct-RTX3060 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-1.5B-Instruct \
		--served_model_name Qwen2.5-1.5B-Instruct \
		--chat_template qwen2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 4090(24GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-1.5B-Instruct-RTX4090 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-1.5B-Instruct \
		--served_model_name Qwen2.5-1.5B-Instruct \
		--chat_template qwen2
	```
	</details>

* ### Qwen2.5-3B-Instruct

	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 3060(12GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-3B-Instruct-RTX3060 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-3B-Instruct \
		--served_model_name Qwen2.5-3B-Instruct \
		--chat_template qwen2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 4090(24GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-3B-Instruct-RTX4090 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-3B-Instruct \
		--served_model_name Qwen2.5-3B-Instruct \
		--chat_template qwen2
	```
	</details>

* ### Qwen2.5-7B-Instruct

	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 3060(12GB) X2</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-7B-Instruct-RTX3060 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-7B-Instruct \
		--served_model_name Qwen2.5-7B-Instruct \
		--chat_template qwen2 \
 		--tp_size 2 \
 		--world_size 2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 4090(24GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-7B-Instruct-RTX4090 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-7B-Instruct \
		--served_model_name Qwen2.5-7B-Instruct \
		--chat_template qwen2
	```
	</details>

* ### Qwen2.5-14B-Instruct

	<details>
		<summary>&nbsp;&nbsp;NVIDIA RTX 4090(24GB) X2</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-14B-Instruct-RTX4090 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-14B-Instruct \
		--served_model_name Qwen2.5-14B-Instruct \
		--chat_template qwen2 \
 		--tp_size 2 \
 		--world_size 2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA H200(141GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-14B-Instruct-H200 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-14B-Instruct \
		--served_model_name Qwen2.5-14B-Instruct \
		--chat_template qwen2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA A800(80GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-14B-Instruct-A800 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-14B-Instruct \
		--served_model_name Qwen2.5-14B-Instruct \
		--chat_template qwen2
	```
	</details>

* ### Qwen2.5-32B-Instruct
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA H200(141GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-32B-Instruct-H200 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-32B-Instruct \
		--served_model_name Qwen2.5-32B-Instruct \
		--chat_template qwen2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA A800(80GB) X1</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-32B-Instruct-A800 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-32B-Instruct \
		--served_model_name Qwen2.5-32B-Instruct \
		--chat_template qwen2
	```
	</details>

* ### Qwen2.5-72B-Instruct
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA H200(141GB) X2</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-72B-Instruct-H200 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-72B-Instruct \
		--served_model_name Qwen2.5-72B-Instruct \
		--chat_template qwen2 \
 		--tp_size 2 \
 		--world_size 2
	```
	</details>
	
	<details>
		<summary>&nbsp;&nbsp;NVIDIA A800(80GB) X2</summary>
	
	```sh
	docker run --rm -it --gpus all --net=host --ipc=host \
		-v /models:/workspace/models ghcr.io/reft-ai/reft:1.0.0-Qwen2.5-72B-Instruct-A800 \
		reft-server \
		--model /workspace/models/deepseek-ai/Qwen2.5-72B-Instruct \
		--served_model_name Qwen2.5-72B-Instruct \
		--chat_template qwen2 \
		--tp_size 2 \
 		--world_size 2
	```
	</details>


----

## Quick start

> #### To deploy and run a LLM/LM with on-premises or cloud GPUs, all you need with reft.cpp is GPUs+Linux+Docker.

<br/>

Example model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

### 1. Download model weights

```sh
mkdir -p models
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --load-dir ./models
```

### 2. Download and launch `reft` (OpenAI-compatible API server)

#### Command

```sh
docker run --rm -it --gpus all --net=host --ipc=host \
  -v ./models:/workspace/models ghcr.io/reft-ai/reft:latest \
  /workspace/reft serve \
  --model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --served_model_name DeepSeek-R1-Distill-Qwen-1.5B \
  --chat_template ds-distill-qwen2
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

### 3. Start chatting

<details>
	<summary>Chat via CLI</summary>

#### Command

```sh
curl -Ns http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
	"model": "DeepSeek-R1-Distill-Qwen-1.5B",
	"messages": [{"role":"user", "content": "<｜begin▁of▁sentence｜><｜User｜>Who are you?<｜Assistant｜><think>\\n"}],
	"max_tokens": 24,
	"temperature": 0.6,
	"stream": true
  }'
```

#### Output

```text
data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"role":"assistant"},"index":0,"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"Greetings"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"!"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":" I"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"'m"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":" Deep"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"Seek"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}
```

</details>

<details>
	<summary>Chat via WebUI app</summary>

#### 1. Download and install app
[DeepChat](https://deepchat.thinkinai.xyz)

#### 2. Setup an OpenAI provider
<img width="1027" height="631" alt="image" src="https://github.com/user-attachments/assets/47958351-3cd4-4fb7-a806-fa85a3739ccc" />

#### 3. Now, enjoy chatting!
<img width="1027" height="631" alt="image" src="https://github.com/user-attachments/assets/070e916f-7a28-4b48-bfff-e77e031a6c6d" />

</details>


## Supported models

- :white_check_mark: : Done
- :coffee: : To-Do

### LLM

  | Models                | Nvidia GPU | AMD GPU | Hexagon NPU | Moore Threads GPU | MetaX GPU |
  |:---------------------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|
  |[Qwen2.5-0.5/1.5/3/7/14/32/72B(-Instruct)](https://huggingface.co/collections/Qwen/qwen25)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[Qwen2.5-Math-1.5/7/72B(-Instruct)](https://huggingface.co/collections/Qwen/qwen25-math)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[Qwen2.5-Coder-0.5/1.5/3/7/14/32B(-Instruct)](https://huggingface.co/collections/Qwen/qwen25-coder)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[DeepSeek-R1-Distill-Qwen-1.5/7/14/32B](https://huggingface.co/collections/deepseek-ai/deepseek-r1)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
	
### Vision LM

  | Models                | Nvidia GPU | AMD GPU | Hexagon NPU | Moore Threads GPU | MetaX GPU |
  |:---------------------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|
  |[SAM](https://github.com/facebookresearch/segment-anything)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[ViT](https://github.com/google-research/vision_transformer)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
	
	
### Audio LM

  | Models                | Nvidia GPU | AMD GPU | Hexagon NPU | Moore Threads GPU | MetaX GPU |
  |:---------------------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|
  |[Whisper](https://huggingface.co/collections/openai/whisper-release)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[OpenVoice](https://huggingface.co/myshell-ai/OpenVoice)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[MeloTTS-English/...](https://huggingface.co/myshell-ai/MeloTTS-English)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
