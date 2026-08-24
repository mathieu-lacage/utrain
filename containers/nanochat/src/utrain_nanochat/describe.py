"""The container's self-description: phases, order, and config schema.

``version`` is the *contract* version reported to utrain (what ``describe``
advertises), not the version of this Python package.

No phase is `cacheable`, `download` included. `check-cache` has to declare the
exact sha256 of every file a phase would write without doing the work, and
nanochat's pretraining shards are neither hash-pinned upstream nor small enough
to hash speculatively; everything after them is stochastic. So `check-cache`
refuses every phase and utrain reads that as a plain cache miss.

The defaults below are all smaller than nanochat's own, because nanochat's are
sized for the 8xH100 speedrun and utrain hands a container a single GPU.
"""

DESCRIBE = {
    "name": "nanochat",
    "version": "1.0.0",
    "phases": [
        {"name": "download", "label": "Download Pretraining Data"},
        {"name": "tokenizer", "label": "Train BPE Tokenizer"},
        # The metric names are nanochat's own: its training scripts log through
        # wandb, which utrain shadows, so what reaches the TUI is exactly what
        # upstream chose to call things.
        {
            "name": "pretrain",
            "label": "Pretrain Base Model",
            "plots": [
                {"x": "step", "y": "train/loss"},
                {"x": "step", "y": "val/bpb"},
            ],
        },
        {
            "name": "sft",
            "label": "Supervised Fine-Tuning",
            "plots": [
                {"x": "step", "y": "train/loss"},
                {"x": "step", "y": "chatcore_metric"},
            ],
        },
        {
            "name": "rl",
            "label": "Reinforcement Learning (GRPO on GSM8K)",
            "plots": [
                {"x": "step", "y": "reward"},
                {"x": "step", "y": "pass@1"},
            ],
        },
    ],
    "phase_order": ["download", "tokenizer", "pretrain", "sft", "rl"],
    "config_schema": {
        "globals": {
            "groups": [
                {
                    "name": "model",
                    "label": "Model Architecture",
                    "fields": [
                        {
                            "key": "depth",
                            "label": "Transformer Depth",
                            "type": "int",
                            "default": 8,
                            "min": 4,
                            "max": 32,
                            "description": (
                                "Number of layers; width follows from it. nanochat's "
                                "speedrun uses 24 across eight GPUs -- 8 is what one "
                                "GPU finishes in a sitting."
                            ),
                        },
                        {
                            "key": "max_seq_len",
                            "label": "Context Length",
                            "type": "int",
                            "default": 2048,
                            "min": 256,
                            "max": 8192,
                            "description": "Maximum sequence length",
                        },
                        {
                            "key": "vocab_size",
                            "label": "Vocabulary Size",
                            "type": "int",
                            "default": 32768,
                            "min": 4096,
                            "max": 65536,
                            "description": "BPE vocabulary size, trained by the tokenizer phase",
                        },
                    ],
                }
            ]
        },
        "phases": {
            "download": {
                "groups": [
                    {
                        "name": "data",
                        "label": "Pretraining Corpus",
                        "fields": [
                            {
                                "key": "num_shards",
                                "label": "Shards",
                                "type": "int",
                                "default": 8,
                                "min": 1,
                                "max": 6542,
                                "description": (
                                    "ClimbMix shards to fetch, ~250M characters each. "
                                    "GPT-2 capability needs ~170."
                                ),
                            },
                            {
                                "key": "num_workers",
                                "label": "Download Workers",
                                "type": "int",
                                "default": 4,
                                "min": 1,
                                "max": 32,
                            },
                            {
                                "key": "prefetch_tasks",
                                "label": "Prefetch SFT/RL Datasets",
                                "type": "bool",
                                "default": True,
                                "description": (
                                    "Fetch SmolTalk, MMLU and GSM8K now rather than "
                                    "stalling the sft phase hours later."
                                ),
                            },
                        ],
                    }
                ]
            },
            "tokenizer": {
                "groups": [
                    {
                        "name": "tokenizer",
                        "label": "Tokenizer Training",
                        "fields": [
                            {
                                "key": "max_chars",
                                "label": "Training Characters",
                                "type": "int",
                                "default": 2000000000,
                                "min": 1000000,
                                "description": "Characters of corpus the BPE merges are fit on",
                            },
                            {
                                "key": "doc_cap",
                                "label": "Per-Document Cap",
                                "type": "int",
                                "default": 10000,
                                "min": 100,
                                "description": "Maximum characters taken from any one document",
                            },
                        ],
                    }
                ]
            },
            "pretrain": {
                "groups": [
                    {
                        "name": "training",
                        "label": "Training",
                        "fields": [
                            {
                                "key": "device_batch_size",
                                "label": "Device Batch Size",
                                "type": "int",
                                "default": 8,
                                "min": 1,
                                "max": 64,
                                "description": "Lower this first if the GPU runs out of memory",
                            },
                            {
                                "key": "total_batch_size",
                                "label": "Total Batch Size (tokens)",
                                "type": "int",
                                "default": -1,
                                "description": (
                                    "-1 lets nanochat size it from the model. Lowering it "
                                    "cuts the gradient accumulation steps each optimizer "
                                    "step costs, which is what makes a one-GPU run finish."
                                ),
                            },
                            {
                                "key": "target_param_data_ratio",
                                "label": "Data:Param Ratio",
                                "type": "float",
                                "default": 12.0,
                                "description": (
                                    "Tokens per parameter; sets the step count when "
                                    "Training Steps is -1. Chinchilla-optimal is ~20."
                                ),
                            },
                            {
                                "key": "num_iterations",
                                "label": "Training Steps",
                                "type": "int",
                                "default": -1,
                                "description": "-1 derives the count from the data:param ratio",
                            },
                            {
                                "key": "eval_every",
                                "label": "Eval Every N Steps",
                                "type": "int",
                                "default": 250,
                                "min": 1,
                            },
                            {
                                "key": "eval_tokens",
                                "label": "Eval Tokens",
                                "type": "int",
                                "default": 2097152,
                                "min": 65536,
                                "description": (
                                    "Tokens per validation pass. This is the dominant "
                                    "cost of a short run: nanochat's own 41.9M is one "
                                    "eval per 20 of training on a single GPU."
                                ),
                            },
                            {
                                "key": "core_metric_every",
                                "label": "CORE Metric Every N Steps",
                                "type": "int",
                                "default": 2000,
                                "description": (
                                    "-1 disables it. Any other value also evaluates once "
                                    "at the final step, whatever the interval."
                                ),
                            },
                            {
                                "key": "sample_every",
                                "label": "Sample Every N Steps",
                                "type": "int",
                                "default": 2000,
                                "description": (
                                    "Draws sample text into the phase log. -1 disables it; "
                                    "otherwise it also samples at the final step."
                                ),
                            },
                            {
                                "key": "run_eval",
                                "label": "Run base_eval Afterwards",
                                "type": "bool",
                                "default": True,
                                "description": "CORE metric and val bpb once training ends",
                            },
                        ],
                    }
                ]
            },
            "sft": {
                "groups": [
                    {
                        "name": "training",
                        "label": "Training",
                        "fields": [
                            {
                                "key": "device_batch_size",
                                "label": "Device Batch Size",
                                "type": "int",
                                "default": 4,
                                "min": 1,
                                "max": 64,
                            },
                            {
                                "key": "total_batch_size",
                                "label": "Total Batch Size (tokens)",
                                "type": "int",
                                "default": -1,
                                "description": (
                                    "-1 lets nanochat size it from the model. Lowering it "
                                    "cuts the gradient accumulation steps each optimizer "
                                    "step costs, which is what makes a one-GPU run finish."
                                ),
                            },
                            {
                                "key": "num_iterations",
                                "label": "Training Steps",
                                "type": "int",
                                "default": -1,
                                "description": "-1 runs one epoch over the SFT mixture",
                            },
                            {
                                "key": "eval_every",
                                "label": "Eval Every N Steps",
                                "type": "int",
                                "default": 200,
                                "min": 1,
                            },
                            {
                                "key": "eval_tokens",
                                "label": "Eval Tokens",
                                "type": "int",
                                "default": 2097152,
                                "min": 65536,
                                "description": "Tokens per validation pass",
                            },
                            {
                                "key": "chatcore_every",
                                "label": "ChatCORE Every N Steps",
                                "type": "int",
                                "default": 200,
                                "description": "-1 disables it",
                            },
                            {
                                "key": "mmlu_epochs",
                                "label": "MMLU Epochs",
                                "type": "int",
                                "default": 3,
                                "min": 0,
                                "description": (
                                    "Repeats of MMLU in the training mixture, ~100K rows "
                                    "each. With Training Steps at -1 the mixture size is "
                                    "what decides how long the phase runs."
                                ),
                            },
                            {
                                "key": "gsm8k_epochs",
                                "label": "GSM8K Epochs",
                                "type": "int",
                                "default": 4,
                                "min": 0,
                                "description": "Repeats of GSM8K in the mixture, ~8K rows each",
                            },
                            {
                                "key": "run_eval",
                                "label": "Run chat_eval Afterwards",
                                "type": "bool",
                                "default": True,
                                "description": "ChatCORE over the finished SFT model",
                            },
                        ],
                    },
                    {
                        "name": "chat",
                        "label": "Chat (serve)",
                        "fields": [
                            {
                                "key": "temperature",
                                "label": "Temperature",
                                "type": "float",
                                "default": 0.6,
                                "description": "Sampling temperature when a client sends none",
                            },
                            {
                                "key": "top_k",
                                "label": "Top-K",
                                "type": "int",
                                "default": 50,
                                "min": 1,
                            },
                            {
                                "key": "max_new_tokens",
                                "label": "Reply Length",
                                "type": "int",
                                "default": 512,
                                "min": 16,
                                "max": 8192,
                            },
                        ],
                    },
                ]
            },
            "rl": {
                "groups": [
                    {
                        "name": "training",
                        "label": "Training",
                        "fields": [
                            {
                                "key": "num_epochs",
                                "label": "Epochs over GSM8K",
                                "type": "int",
                                "default": 1,
                                "min": 1,
                            },
                            {
                                "key": "num_samples",
                                "label": "Samples per Question",
                                "type": "int",
                                "default": 16,
                                "min": 2,
                                "description": "The group GRPO centres its advantages over",
                            },
                            {
                                "key": "examples_per_step",
                                "label": "Examples per Step",
                                "type": "int",
                                "default": 16,
                                "min": 1,
                            },
                            {
                                "key": "device_batch_size",
                                "label": "Device Batch Size",
                                "type": "int",
                                "default": 8,
                                "min": 1,
                                "max": 64,
                                "description": "Also the highest k reported as pass@k",
                            },
                            {
                                "key": "max_new_tokens",
                                "label": "Rollout Length",
                                "type": "int",
                                "default": 256,
                                "min": 32,
                            },
                            {
                                "key": "eval_every",
                                "label": "Eval Every N Steps",
                                "type": "int",
                                "default": 60,
                                "min": 1,
                            },
                            {
                                "key": "eval_examples",
                                "label": "Eval Examples",
                                "type": "int",
                                "default": 100,
                                "min": 1,
                                "description": (
                                    "GSM8K problems behind each pass@k number. Every one "
                                    "is a full generation, and an eval always runs before "
                                    "the first training step."
                                ),
                            },
                        ],
                    }
                ]
            },
        },
    },
    "can_serve": True,
}
