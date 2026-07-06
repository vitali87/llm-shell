"""Fine-tune Qwen3.5-2B into a shell-command LoRA on a Modal GPU.

Modal settings (CUDA image, /cache HF volume, huggingface secret, RTX-PRO-6000,
spawn-then-get entrypoint) are copied from the quadbit harness. Training uses
unsloth (bf16 LoRA + GGUF export). The training loop itself is a plain PyTorch
loop (also quadbit's style) to avoid HF Trainer / TRL batch-handling churn.

Qwen3.5 is a vision-language model: it MUST be loaded with FastModel (not
FastLanguageModel), and the text tokenizer extracted via processor.tokenizer.

Run (detached so it survives a client disconnect):
    uv run modal run --detach train_modal.py
Then pull the built model:
    modal volume get llm-shell-hf-cache shell-commands-qwen3.5-2b .
    ollama create shell-commands-qwen3.5-2b -f shell-commands-qwen3.5-2b/Modelfile
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent
MODEL = "Qwen/Qwen3.5-2B"
OUT = "shell-commands-qwen3.5-2b"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env(
        {
            "PATH": (
                "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin"
                ":/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
            "HF_HOME": "/cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    # Pin unsloth to the current release and let IT pull the matching trl/peft/
    # transformers. Leaving trl unpinned backtracks unsloth to a year-old build
    # whose unsloth_zoo breaks on modern trl (ConstantLengthDataset import).
    .pip_install(
        "unsloth==2026.6.9",
        "unsloth_zoo",
        "transformers>=5.2.0",  # Qwen3.5 needs transformers v5 (>=5.2)
        "datasets",
        "torchvision",
        "pillow",
        "hf_transfer",
        "sentencepiece",
        "protobuf",
    )
    .add_local_file((ROOT / "data.jsonl").as_posix(), "/root/data.jsonl")
)
app = modal.App("llm-shell-finetune", image=image)
vol = modal.Volume.from_name("llm-shell-hf-cache", create_if_missing=True)


@app.function(
    # quadbit's standard GPU; oversized for 2B (~5GB), swap to L40S to save credits
    gpu="RTX-PRO-6000",
    timeout=86400,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(epochs: int = 3, max_seq_length: int = 1024, quant: str = "q8_0") -> str:
    import random

    import torch
    from datasets import load_dataset
    from unsloth import FastModel

    # Qwen3.5 is vision-language: load with FastModel and take processor.tokenizer.
    model, processor = FastModel.from_pretrained(
        MODEL,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # unsloth advises bf16 (not QLoRA) for Qwen3.5
        load_in_16bit=True,
        full_finetuning=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Text-only task: attach LoRA to language layers, skip the vision tower.
    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
    )
    model.enable_input_require_grads()
    model.train()

    def render(messages, add_generation_prompt):
        # Render without <think> blocks so the shell helper never emits reasoning.
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )

    def encode(example):
        user = [{"role": "user", "content": example["instruction"]}]
        full = user + [{"role": "assistant", "content": example["output"]}]
        prompt_ids = tokenizer(render(user, True)).input_ids
        input_ids = tokenizer(render(full, False)).input_ids[:max_seq_length]
        # Train only on the response: mask the prompt prefix with -100.
        labels = list(input_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return input_ids, labels

    raw = load_dataset("json", data_files="/root/data.jsonl", split="train")
    examples = [encode(r) for r in raw]
    print(f"Encoded {len(examples)} examples", flush=True)

    # Plain PyTorch loop (quadbit style): batch size 1 + grad accumulation.
    dev = "cuda"
    accum = 16
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
    order = list(range(len(examples)))
    step = 0
    print("Starting training...", flush=True)
    for epoch in range(epochs):
        random.Random(1234 + epoch).shuffle(order)
        for idx in order:
            input_ids, labels = examples[idx]
            w = torch.tensor([input_ids], device=dev)
            lbl = torch.tensor([labels], device=dev)
            loss = model(input_ids=w, labels=lbl).loss
            (loss / accum).backward()
            step += 1
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
    # Flush any gradients left over when the example count is not a multiple of accum.
    if step % accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    opt.zero_grad()
    print("Training done.", flush=True)

    out_dir = f"/cache/{OUT}"
    print(f"Exporting GGUF ({quant}) to {out_dir} ...", flush=True)
    model.save_pretrained_gguf(out_dir, tokenizer, quantization_method=quant)

    # Ollama Modelfile (ChatML), matching create_model.sh.
    ggufs = list(Path(out_dir).glob("*.gguf"))
    if not ggufs:
        raise FileNotFoundError(f"No .gguf file was produced in {out_dir}")
    gguf = ggufs[0]
    Path(out_dir, "Modelfile").write_text(
        f"FROM ./{gguf.name}\n\n"
        "PARAMETER temperature 0\n"
        "PARAMETER top_p 0.7\n"
        'PARAMETER stop "<|im_end|>"\n\n'
        'TEMPLATE """\n'
        "{{ if .System }}<|im_start|>system\n"
        "{{ .System }}<|im_end|>{{ end }}<|im_start|>user\n"
        "{{ .Prompt }}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "\n"
        "</think>\n"
        "\n"
        '"""\n'
    )
    vol.commit()
    print(f"Done. {gguf.name} + Modelfile committed to volume.", flush=True)
    return gguf.name


@app.local_entrypoint()
def main(epochs: int = 3, max_seq_length: int = 1024, quant: str = "q8_0") -> None:
    # spawn (not remote): the run survives a local-client disconnect.
    call = run.spawn(epochs=epochs, max_seq_length=max_seq_length, quant=quant)
    print(f"SPAWN_ID {call.object_id}", flush=True)
    # call.get() blocks; if this waiter dies, recover the result via SPAWN_ID.
    print(f"RESULT {call.get()}", flush=True)
