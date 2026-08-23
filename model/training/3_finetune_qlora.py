"""Fine-tune QLoRA Qwen2.5 untuk mode Pelatih (OFFLINE, dijalankan di Kaggle).

SEKALI JALAN, DI LUAR REPO SUBMISSION. Skrip ini TIDAK pernah dipanggil backend
saat demo — rulebook membatasi implementasi AI pada inferensi berparameter statis.
Keluarannya adalah adapter LoRA (puluhan MB) yang di-commit ke ../checkpoints/ lalu
di-load backend di MODE=local.

Implementasi memakai `transformers.Trainer` + PEFT langsung (bukan trl.SFTTrainer),
supaya stabil lintas versi library Kaggle. Masking label: hanya token jawaban
asisten yang dihitung loss-nya (prompt di-mask -100) — setara completion-only SFT.

Pakai (di sel notebook Kaggle):
    !python 3_finetune_qlora.py --train ../../data/sft/train.jsonl \
                               --val   ../../data/sft/val.jsonl \
                               --out   ../checkpoints
"""
from __future__ import annotations

import os
# Paksa 1 GPU. Di Kaggle T4 x2, Trainer otomatis membungkus model dengan DataParallel
# yang BENTROK dengan model terkuantisasi 4-bit (CUBLAS_STATUS_EXECUTION_FAILED).
# Harus di-set sebelum torch meng-inisialisasi CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_BASE = "Qwen/Qwen2.5-3B-Instruct"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True, help="Folder adapter LoRA hasil latih.")

    ap.add_argument("--tipe", nargs="*", default=None,
                    help="Batasi jenis contoh, mis. --tipe coach_sesi coach_turn (default: semua).")

    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=100, help="Checkpoint berkala (sesi Kaggle bisa mati).")
    ap.add_argument("--resume", action="store_true",
                    help="Lanjutkan dari checkpoint terakhir di --out (sesi Kaggle mati di tengah jalan).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Batasi jumlah step (mis. 5 untuk smoke test cepat). -1 = pakai --epochs.")

    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--no-4bit", action="store_true", help="Latih tanpa kuantisasi (butuh GPU besar).")
    return ap.parse_args()


def muat_baris(path: Path, tipe: list[str] | None) -> list[dict]:
    baris = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if tipe:
        baris = [b for b in baris if b["meta"]["tipe"] in tipe]
    if not baris:
        raise SystemExit(f"ERROR: {path} kosong setelah filter --tipe {tipe}.")
    print(f"  {path.name}: {len(baris)} contoh {dict(Counter(b['meta']['tipe'] for b in baris))}")
    return baris


def buat_dataset(baris: list[dict], tokenizer, max_len: int):
    """Ubah {'messages': [...]} -> {input_ids, attention_mask, labels} dengan
    masking completion-only (hanya jawaban asisten yang dihitung loss)."""
    from datasets import Dataset

    def encode(ex):
        msgs = ex["messages"]
        # Render ke teks dulu lalu tokenisasi biasa -> dijamin list[int] (bukan objek
        # tokenizers.Encoding yang tak bisa ditulis Arrow). add_special_tokens=False
        # karena template chat sudah menyertakan token spesialnya sendiri.
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        n_prompt = min(len(prompt), len(full))
        labels = [-100] * n_prompt + full[n_prompt:]
        full, labels = full[:max_len], labels[:max_len]
        return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}

    ds = Dataset.from_list([{"messages": b["messages"]} for b in baris])
    ds = ds.map(encode, remove_columns=ds.column_names)
    # Buang contoh yang seluruh label-nya ter-mask (prompt kepanjangan) -> cegah loss nan.
    ds = ds.filter(lambda e: any(t != -100 for t in e["labels"]))
    return ds


def main() -> int:
    args = parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        DataCollatorForSeq2Seq, Trainer, TrainingArguments,
    )

    if not torch.cuda.is_available():
        print("ERROR: butuh GPU. Di Kaggle: Settings -> Accelerator -> GPU.", file=sys.stderr)
        return 2

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Memuat data:")
    ds_train = buat_dataset(muat_baris(args.train, args.tipe), tokenizer, args.max_seq_len)
    ds_val = (buat_dataset(muat_baris(args.val, args.tipe), tokenizer, args.max_seq_len)
              if args.val and args.val.exists() else None)
    if ds_val is None:
        print("  (tanpa val — eval loss dilewati)")

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        torch_dtype=dtype,
        device_map={"": 0},  # taruh semua di GPU 0 (3B 4-bit muat longgar di T4/P100 16GB)
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        # Eval in-training dimatikan: materialisasi logits atas vocab 152k bikin OOM di
        # T4. Metrik sebenarnya (adapter vs base) dihitung terpisah oleh 4_evaluate.py.
        eval_strategy="no",
        bf16=bf16,
        fp16=not bf16,
        report_to="none",
        seed=args.seed,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        processing_class=tokenizer,
    )

    ada_checkpoint = args.resume and any(args.out.glob("checkpoint-*"))
    if args.resume and not ada_checkpoint:
        print(f"--resume diminta tapi tidak ada checkpoint-* di {args.out}; mulai dari awal.")
    trainer.train(resume_from_checkpoint=ada_checkpoint or None)

    args.out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out))       # adapter_model.safetensors + adapter_config.json
    tokenizer.save_pretrained(str(args.out))

    meta = {
        "base_model": args.base_model,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": TARGET_MODULES},
        "training": {"epochs": args.epochs, "lr": args.lr, "max_seq_len": args.max_seq_len,
                     "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                     "max_steps": args.max_steps, "quantized_4bit": not args.no_4bit, "seed": args.seed},
        "data": {"train": str(args.train), "val": str(args.val) if args.val else None,
                 "n_train": len(ds_train), "n_val": len(ds_val) if ds_val else 0,
                 "tipe": args.tipe or "semua"},
    }
    (args.out / "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nAdapter tersimpan di {args.out}")
    print("Langkah berikutnya: 4_evaluate.py --adapter <folder ini> --mode both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
