# RefineFuture.AI Install And Usage

## Overview

RefineFuture.AI provides one CLI and one OpenAI-compatible server on top of a
shared LM engine. Users only need to choose:

- model family
- model directory
- backend / runner tuple, if they want to override the matrix default

Common workflows are the same across platforms:

1. unpack a release matrix package
2. point it at a model directory
3. run `refft-cli` for local generation or `refft-server` for API serving

## Release Packages

Release package directories follow this format:

- `refft_<platform>-<arch>-<backend>-<model>-<precision>`

Examples:

- `refft_ubuntu-x86_64-cpu-qwen3-bf16` (`internal validation only`, not shipped as a public release)
- `refft_ubuntu-x86_64-cuda-qwen3-all`
- `refft_android-aarch64-qnn-qwen3-fp16`
- `refft_android-aarch64-qnn-qwen3-w4a16`
- `refft_android-aarch64-hexagon-qwen3-fp16`

Runner naming is intentionally omitted from the package directory. For a given
released `platform/arch/backend/model/precision` tuple, there is exactly one
builtin runner in the package. The builtin runner is still documented in the
package-local `INSTALL.md` and `RELEASE.md`.

Each package contains:

- `bin/`
- `lib/`
- `scripts/run_cli.sh`
- `scripts/run_server.sh`
- `scripts/test_cli.sh`
- `scripts/test_server.sh`
- `scripts/bench_serve.py`
- `INSTALL.md`
- `RELEASE.md`

## Model Directory

All supported runners use the same logical model directory layout:

```text
<model_dir>/
  config.json
  model.safetensors
  vocab.json
  tokenizer_config.json
```

## CLI Usage

Minimal CLI usage:

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

## Server Usage

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

## Platform Notes

### Ubuntu CPU

- runner family: `unified`
- precision family: `all`
- typical matrix precision in release: `bf16`

### Ubuntu CUDA

- runner family: `unified`
- precision family: `all`
- typical matrix precision in release: `fp16`

### Android QNN

- runner family: `disagg`
- separate precision-specific runners
- current release variants:
  - `fp16`
  - `w4a16`

### Android Hexagon

- runner family: `unified`
- current release variant:
  - `fp16`

## Validation And Benchmarking

Package-local validation:

```bash
MODEL_DIR=/path/to/model bash ./scripts/test_cli.sh
MODEL_DIR=/path/to/model PORT=18080 bash ./scripts/test_server.sh
```

Package-local server benchmark:

```bash
python3 ./scripts/bench_serve.py \
  --url http://127.0.0.1:18080/v1/chat/completions \
  --model qwen3 \
  --prompt "Summarize release validation for LM backends." \
  --max-new-tokens 128 \
  --requests 4 \
  --ignore-eos
```

The benchmark prints:

- TTFT
- TPOT
- tok/s
- request-level latency summary

## Logging

Default runtime logging is `warning` and above.

Standard mode avoids debug/reference/profiling prints. Profile-specific metrics
and verbose traces should only appear in profile mode or explicit diagnostics.
