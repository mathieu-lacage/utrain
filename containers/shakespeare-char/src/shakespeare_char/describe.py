"""The container's self-description: phases, order, and config schema.

``version`` is the *contract* version reported to utrain (what ``describe``
advertises), not the version of this Python package.
"""

DESCRIBE = {
    "name": "shakespeare-char",
    "version": "1.0.0",
    "phases": [
        # `download` is cacheable: it produces the same corpus every time, and
        # says so up front via `check-cache`, so utrain can serve it from the
        # store instead of re-fetching it for every run.
        {"name": "download", "label": "Download Corpus", "cacheable": True},
        {"name": "tokenizer", "label": "Build Vocabulary"},
        # `pretrain` logs three metrics but only two say whether training is
        # going well, so it names them: utrain's TUI opens on those instead of
        # stacking a plot per metric.
        {
            "name": "pretrain",
            "label": "Train Character LM",
            "plots": [
                {"x": "step", "y": "loss"},
                {"x": "step", "y": "bpb"},
            ],
        },
    ],
    "phase_order": ["download", "tokenizer", "pretrain"],
    "config_schema": {
        "globals": {
            "groups": [
                {
                    "name": "model",
                    "label": "Model Architecture",
                    "fields": [
                        {
                            "key": "n_layer",
                            "label": "Layers",
                            "type": "int",
                            "default": 4,
                            "min": 1,
                            "max": 24,
                            "description": "Number of transformer blocks",
                        },
                        {
                            "key": "n_head",
                            "label": "Attention Heads",
                            "type": "int",
                            "default": 4,
                            "min": 1,
                            "max": 16,
                            "description": "Number of attention heads",
                        },
                        {
                            "key": "n_embd",
                            "label": "Embedding Dim",
                            "type": "int",
                            "default": 128,
                            "min": 32,
                            "max": 1024,
                            "description": "Model embedding dimension",
                        },
                        {
                            "key": "block_size",
                            "label": "Context Length",
                            "type": "int",
                            "default": 256,
                            "min": 64,
                            "max": 2048,
                            "description": "Maximum sequence length",
                        },
                    ],
                }
            ]
        },
        "phases": {
            "pretrain": {
                "groups": [
                    {
                        "name": "training",
                        "label": "Training",
                        "fields": [
                            {
                                "key": "max_iters",
                                "label": "Training Steps",
                                "type": "int",
                                "default": 5000,
                                "min": 100,
                                "max": 100000,
                            },
                            {
                                "key": "batch_size",
                                "label": "Batch Size",
                                "type": "int",
                                "default": 64,
                                "min": 1,
                                "max": 512,
                            },
                            {
                                "key": "learning_rate",
                                "label": "Learning Rate",
                                "type": "float",
                                "default": 3e-4,
                            },
                            {
                                "key": "eval_interval",
                                "label": "Eval Every N Steps",
                                "type": "int",
                                "default": 500,
                                "min": 1,
                            },
                            {
                                "key": "generate_len",
                                "label": "Generate Length (serve)",
                                "type": "int",
                                "default": 200,
                                "min": 10,
                                "max": 2000,
                            },
                        ],
                    }
                ]
            }
        },
    },
    "can_serve": True,
}
