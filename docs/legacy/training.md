# Training of LLM/LM (archived)

> **Archived.** This flow is no longer part of the supported product surface.
> It is kept for reference only — see the [main README](../../README.md) for the
> current `refft-hexagon` runtime.


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
