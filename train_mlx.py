import json
import random
import subprocess
import sys
from pathlib import Path

# Apple Silicon fine-tuning path (MLX). The unsloth/deepspeed scripts are
# CUDA-only; this one runs LoRA on an M-series Mac. Same task, same base model.
BASE_MODEL = "mlx-community/Qwen3.5-2B-bf16"
DATA_DIR = Path("mlx_data")
ADAPTER_DIR = Path("adapters")
VALID_FRACTION = 0.05
SEED = 42


def build_dataset():
    """Convert data.jsonl (instruction/output) to MLX chat train/valid splits."""
    rows = []
    with open("data.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": ex["instruction"]},
                        {"role": "assistant", "content": ex["output"]},
                    ]
                }
            )

    random.Random(SEED).shuffle(rows)
    n_valid = max(1, int(len(rows) * VALID_FRACTION))
    valid, train = rows[:n_valid], rows[n_valid:]

    DATA_DIR.mkdir(exist_ok=True)
    for name, split in [("train", train), ("valid", valid)]:
        with open(DATA_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(train)} train / {len(valid)} valid examples to {DATA_DIR}/")


def train():
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        BASE_MODEL,
        "--train",
        "--data",
        str(DATA_DIR),
        "--adapter-path",
        str(ADAPTER_DIR),
        "--batch-size",
        "4",
        "--iters",
        "800",
        "--num-layers",
        "16",
        "--learning-rate",
        "1e-4",
        "--max-seq-length",
        "1024",
        "--steps-per-eval",
        "200",
        "--seed",
        str(SEED),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    build_dataset()
    train()
    print(f"\nDone. LoRA adapters in {ADAPTER_DIR}/")
    print("Test:  python -m mlx_lm generate \\")
    print(f"  --model {BASE_MODEL} --adapter-path {ADAPTER_DIR} \\")
    print('  --prompt "Show hidden files here"')
