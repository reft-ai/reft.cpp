# reft.cpp
![reft cc-new-logo jpg](https://github.com/user-attachments/assets/25f0c2e7-0f64-41e9-979d-ddb0ff932c4d)


[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/reft-ai/reft.cpp)](https://github.com/reft-ai/reft.cpp/releases)
<!--[![Publish](https://github.com/reft-ai/reft.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/reft-ai/reft/actions/workflows/docker-publish.yml))-->

# What is `reft.cpp`
### A high-performance and easy-to-use LLM/LM serving tools for inference, training. All of Ops, serving framework and training tools in C++ without Python/PyTorch, inspired by [llm.c](https://github.com/karpathy/llm.c) of Andrej Karpathy.
### An excutable file of "reft" with model weights are all you need to run the reft-supported model on your GPU(s) with the better performance.

----

## Quick start

#### Run command

```sh
# Run a local model weights downloaded from Hugging Face and launch an OpenAI-compatible API server
docker run --rm -it --gpus all --net=host --ipc=host \
  -v /models:/workspace/models ghcr.io/reft-ai/reft:latest \
  /workspace/reft serve \
  --model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --served_model_name DeepSeek-R1-Distill-Qwen-1.5B \
  --chat_template ds-distill-qwen2
```

#### Output screenshot

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

#### Now you can start chatting with the assistant.

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

#### The output will be as follows.
```text
data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"role":"assistant"},"index":0,"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"Greetings"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"!"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":" I"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"'m"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":" Deep"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}

data: {"id":"d971c92d-8505-4152-b8b3-cf9726e19127","object":"chat.completion.chunk","created":1589478378,"model":"DeepSeek-R1-Distill-Qwen-1.5B","system_fingerprint":"fp_44709d6fcb","choices":[{"delta":{"content":"Seek"},"index":0,"logprobs":null,"finish_reason":""}],"usage":null}
```

## Description

The main goal of `reft.cpp` is to enable xLM inference, training and serving with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Pure C/C++ implementation without any dependencies
- Custom CUDA kernels for running xLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)

## Supported models

- :white_check_mark: : Done
- :coffee: : In progress

### Text-only

  | Models                | Nvidia GPU | AMD GPU | Hexagon NPU | Moore Threads GPU | Meta GPU |
  |:---------------------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|
  |[Qwen2.5-0.5/1.5/3/7/14/32/72B(-Instruct)](https://huggingface.co/collections/Qwen/qwen25)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[Qwen2.5-Math-1.5/7/72B(-Instruct)](https://huggingface.co/collections/Qwen/qwen25-math)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[Qwen2.5-Coder-0.5/1.5/3/7/14/32B(-Instruct)](https://huggingface.co/collections/Qwen/qwen25-coder)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
  |[DeepSeek-R1-Distill-Qwen-1.5/7/14/32B](https://huggingface.co/collections/deepseek-ai/deepseek-r1)|:white_check_mark:|:coffee:|:coffee:|:coffee:|:coffee:|
	
### Vision

  | Models                | Nvidia GPU | AMD GPU | Hexagon NPU | Moore Threads GPU | Meta GPU |
  |:---------------------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|
  |[SAM](https://github.com/facebookresearch/segment-anything)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[ViT](https://github.com/google-research/vision_transformer)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
	
	
### Audio

  | Models                | Nvidia GPU | AMD GPU | Hexagon NPU | Moore Threads GPU | Meta GPU |
  |:---------------------:|:----------:|:-------:|:-----------:|:-----------:|:-----------:|
  |[Whisper](https://huggingface.co/collections/openai/whisper-release)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[OpenVoice](https://huggingface.co/myshell-ai/OpenVoice)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
  |[MeloTTS-English/...](https://huggingface.co/myshell-ai/MeloTTS-English)| :white_check_mark: | :coffee: | :coffee: | :coffee: | :coffee: |
