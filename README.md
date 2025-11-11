# reft.cpp
![reft cc-new-logo jpg](https://github.com/user-attachments/assets/25f0c2e7-0f64-41e9-979d-ddb0ff932c4d)


[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/reft-ai/reft.cpp)](https://github.com/reft-ai/reft.cpp/releases)
<!--[![Publish](https://github.com/reft-ai/reft.cpp/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/reft-ai/reft/actions/workflows/docker-publish.yml))-->

# What is `reft.cpp`
### An xLM inference, training, and serving toolkit in C/C++ without any dependencies, inspired by [llm.c](https://github.com/karpathy/llm.c) and [nanochat](https://github.com/karpathy/nanochat) (both developed by [Karpathy](https://github.com/karpathy)).

## Hot topics

----

## Quick start

Getting started with reft.cpp is straightforward. Here are several ways to install it on your machine:

- Install `reft.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/reft-ai/reft.cpp/releases)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Run a local model weights downloaded from Hugging Face and launch an OpenAI-compatible API server
docker run --rm -it --gpus all --net=host --ipc=host \
  -v /models:/workspace/models ghcr.io/reft-ai/reft:latest \
  /workspace/reft serve \
  --model /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --served_model_name DeepSeek-R1-Distill-Qwen-1.5B \
  --chat_template ds-distill-qwen2
```

Example screenshot
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

Now you can start chatting with the assistant.
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

The output will be as follows.
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

<details>
<summary>Bindings</summary>
</details>

<details>
<summary>UIs</summary>
</details>

<details>
<summary>Tools</summary>
</details>

<details>
<summary>Infrastructure</summary>
</details>

<details>
<summary>Games</summary>
</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |

<!--
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/hexagon/README.md) | Snapdragon |
-->

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?sort=trending) compatible with `reft.cpp`:

- [Trending](https://huggingface.co/models?sort=trending)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

<!--
## [`llama-cli`](tools/main)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Run simple text completion</summary>

    To disable conversation mode explicitly, use `-no-cnv`

    ```bash
    llama-cli -m model.gguf -p "I believe the meaning of life is" -n 128 -no-cnv

    # I believe the meaning of life is to find your own truth and to live in accordance with it. For me, this means being true to myself and following my passions, even if they don't align with societal expectations. I think that's what I love about yoga – it's not just a physical practice, but a spiritual one too. It's about connecting with yourself, listening to your inner voice, and honoring your own unique journey.
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ... 
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>
## Other documentation

- [main (cli)](tools/main/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```
-->

## Dependencies
