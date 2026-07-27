# Legacy packages and usage (archived)

> **These packages are no longer supported.** They are kept here for reference
> only — no new releases, no fixes, no support requests.
>
> The supported runtime is **`refft-hexagon`**. See the
> [main README](../../README.md) to install and run it.

Everything below describes the retired generation of packages, which shipped two
separate binaries (`refft-cli` and `refft-server`) instead of the single
`refft-hexagon` binary.

Archived documents:

- [install_and_usage.md](install_and_usage.md) — full install and usage guide for `refft-cli` / `refft-server`
- [training.md](training.md) — training of LLM/LM: datasets, `refft train`, sample output
- [build_tools.md](build_tools.md) — the `refft.cpp` build tools showcase

## Retired usage

```bash
./bin/refft-cli    --model qwen3 --model_dir /path/to/model --prompt "Who are you?" --max_new_tokens 64
./bin/refft-server --model qwen3 --model_dir /path/to/model --port 8000
```

```bash
curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Explain KV cache reuse briefly."}],
    "max_tokens": 128,
    "stream": false
  }'
```

The full guide for these binaries is archived in
[install_and_usage.md](install_and_usage.md).

## Retired packages

### QNN (Android)

|        Model Package       |         Description |
|----------------------------|---------------------|
| [refft-android-aarch64-qnn-qwen3](https://github.com/refinefuture-ai/refft.cpp/releases/download/github_draft_20260401/android-aarch64-qnn-qwen3-dynamic-fp16.tar.gz) | 0.6B/1.7B/4B/8B/14B/32B supported <br/> FlashAtttion ops supported <br/> Quantization can be set to w4a16, w8a16, w4afp16, w8afp16, fp16 and default is fp16<br/> Tested on OnePlus15/SM8850/16GB-DDR|
| [refft-android-aarch64-qnn-qwen3-moe](https://github.com/refinefuture-ai/refft.cpp/releases/download/github_draft_20260401/android-aarch64-qnn-qwen3moe-fa_moe_hybrid-fp16.tar.gz) | 30B-A3B supported <br/> MoE, FlashAtttion ops supported <br/> TP supported for multi-HTPs backends <br/> Quantization can be set to w4a16, w8a16, w4afp16, w8afp16, fp16 and default is fp16 <br/> Tested on OnePlus15/SM8850/16GB-DDR|

### Nvidia

|        Model Package       |         Description |
|----------------------------|---------------------|
| [refft-linux-x64-cuda-qwen3-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-linux-x64-cuda-qwen3-20260323.tar.xz) | 0.6B/1.7B/4B/8B/14B/32B supported |
| [refft-ubuntu2404-x64-cuda-qwen3-20260323.deb](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-ubuntu2404-x64-cuda-qwen3-20260323.deb) | 0.6B/1.7B/4B/8B/14B/32B supported |
| [refft-linux-x64-cuda-qwen3-moe-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-linux-x64-cuda-qwen3-moe-20260323.tar.xz) | 30B-A3B/235B-A22B supported |
| [refft-ubuntu2404-x64-cuda-qwen3-moe-20260323.deb](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-ubuntu2404-x64-cuda-qwen3-moe-20260323.deb) | 30B-A3B/235B-A22B supported |

### Apple Silicon

|        Model Packcage      |         Description |
|----------------------------|---------------------|
| [refft-macos-arm64-mlx-qwen3-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-macos-arm64-mlx-qwen3-20260323.tar.xz) | 0.6B/1.7B/4B/8B/14B/32B supported |
| [refft-macos-arm64-mlx-qwen3-moe-20260323.tar.xz](https://github.com/refinefuture-ai/refft.cpp/releases/download/20260323/refft-macos-arm64-mlx-qwen3-moe-20260323.tar.xz) | 30B-A3B/235B-A22B supported |

Older archives are still listed on the
[releases page](https://github.com/refinefuture-ai/refft.cpp/releases).
