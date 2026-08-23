"""Evaluasi mode Pelatih: adapter hasil fine-tune vs base non-fine-tune (OFFLINE).

Tujuannya satu: MEMBUKTIKAN fine-tuning memberi nilai tambah, bukan sekadar
mengklaimnya. Angka dari sini masuk ke proposal §Metodologi.

Metrik (dihitung dari data/sft/test.jsonl — dialog yang tidak pernah dilihat saat latih):

  coach_turn  (pelabelan satu giliran)
    - akurasi & macro-F1 label `teknik` (6 kelas)
    - akurasi label `kualitas` (baik/lemah)
    Macro-F1 dipakai karena distribusi teknik tidak rata; akurasi saja bisa terlihat
    bagus hanya dengan menebak kelas mayoritas.

  coach_sesi  (penilaian seluruh sesi)
    - JSON valid (%): berapa persen keluaran bisa di-parse jadi bentuk EvaluateResponse.
      Ini syarat kepakaian di backend — keluaran cantik tapi tak bisa di-parse = gagal.
    - MAE skor_total & MAE skor per-teknik terhadap rubrik emas.

Semua dekoding greedy (do_sample=False) supaya angka bisa direproduksi.

Pakai:
    python 4_evaluate.py --test ../../data/sft/test.jsonl --adapter ../checkpoints --mode both
    python 4_evaluate.py --test ../../data/sft/test.jsonl --mode base --limit 20   # smoke test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_BASE = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = {"coach_turn": 48, "coach_sesi": 512}


# --------------------------------------------------------------------------- #
# Metrik (tanpa sklearn — transparan & tanpa dependensi tambahan)
# --------------------------------------------------------------------------- #

def macro_f1(pasangan: list[tuple[str, str]]) -> tuple[float, dict[str, float]]:
    """pasangan = [(emas, prediksi)]. Return (macro-F1, F1 per kelas)."""
    kelas = sorted({e for e, _ in pasangan})
    per_kelas: dict[str, float] = {}
    for k in kelas:
        tp = sum(1 for e, p in pasangan if e == k and p == k)
        fp = sum(1 for e, p in pasangan if e != k and p == k)
        fn = sum(1 for e, p in pasangan if e == k and p != k)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_kelas[k] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (sum(per_kelas.values()) / len(per_kelas) if per_kelas else 0.0), per_kelas


def akurasi(pasangan: list[tuple[str, str]]) -> float:
    return sum(1 for e, p in pasangan if e == p) / len(pasangan) if pasangan else 0.0


def mae(pasangan: list[tuple[float, float]]) -> float:
    return sum(abs(e - p) for e, p in pasangan) / len(pasangan) if pasangan else float("nan")


# --------------------------------------------------------------------------- #
# Parsing keluaran model
# --------------------------------------------------------------------------- #

def parse_json_lenient(teks: str) -> dict | None:
    """Ambil objek JSON pertama. Model non-fine-tune sering membungkus dengan prosa."""
    teks = teks.strip()
    teks = re.sub(r"^```(?:json)?|```$", "", teks, flags=re.MULTILINE).strip()
    awal, akhir = teks.find("{"), teks.rfind("}")
    if awal < 0 or akhir <= awal:
        return None
    try:
        obj = json.loads(teks[awal : akhir + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def sesi_valid(obj: dict | None) -> bool:
    """Cek keluaran sesi benar-benar bisa dipakai backend (bentuk EvaluateResponse)."""
    if not obj or not isinstance(obj.get("skor_total"), (int, float)):
        return False
    per = obj.get("per_teknik")
    if not isinstance(per, list) or not per:
        return False
    if not all(isinstance(x, dict) and "teknik" in x and isinstance(x.get("skor"), (int, float)) for x in per):
        return False
    return isinstance(obj.get("saran"), list)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def muat_model(base: str, adapter: Path | None, muat_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        if muat_4bit
        else None
    )
    tok = AutoTokenizer.from_pretrained(str(adapter) if adapter else base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quant, torch_dtype=dtype, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tok


def hasilkan(model, tok, messages: list[dict], max_new_tokens: int) -> str:
    import torch

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Evaluasi
# --------------------------------------------------------------------------- #

def evaluasi(model, tok, contoh: list[dict], label: str) -> dict:
    turn_teknik: list[tuple[str, str]] = []
    turn_kualitas: list[tuple[str, str]] = []
    sesi_total: list[tuple[float, float]] = []
    sesi_per_teknik: list[tuple[float, float]] = []
    n_sesi = n_sesi_valid = 0

    for i, c in enumerate(contoh, 1):
        tipe = c["meta"]["tipe"]
        emas = parse_json_lenient(c["messages"][-1]["content"])
        pred = parse_json_lenient(
            hasilkan(model, tok, c["messages"][:-1], MAX_NEW_TOKENS.get(tipe, 256))
        )

        if tipe == "coach_turn":
            # Prediksi gagal parse tetap dihitung sebagai salah ("__gagal__"), bukan dibuang —
            # kalau tidak, model yang sering ngawur justru terlihat unggul.
            turn_teknik.append((emas["teknik"], (pred or {}).get("teknik", "__gagal__")))
            turn_kualitas.append((emas["kualitas"], (pred or {}).get("kualitas", "__gagal__")))
        elif tipe == "coach_sesi":
            n_sesi += 1
            if not sesi_valid(pred):
                continue
            n_sesi_valid += 1
            sesi_total.append((float(emas["skor_total"]), float(pred["skor_total"])))
            skor_pred = {x["teknik"]: float(x["skor"]) for x in pred["per_teknik"]}
            for x in emas["per_teknik"]:
                if x["teknik"] in skor_pred:
                    sesi_per_teknik.append((float(x["skor"]), skor_pred[x["teknik"]]))

        if i % 20 == 0:
            print(f"  [{label}] {i}/{len(contoh)}", file=sys.stderr)

    f1, f1_kelas = macro_f1(turn_teknik)
    return {
        "n_contoh": len(contoh),
        "coach_turn": {
            "n": len(turn_teknik),
            "akurasi_teknik": round(akurasi(turn_teknik), 4),
            "macro_f1_teknik": round(f1, 4),
            "f1_per_teknik": {k: round(v, 4) for k, v in f1_kelas.items()},
            "akurasi_kualitas": round(akurasi(turn_kualitas), 4),
        },
        "coach_sesi": {
            "n": n_sesi,
            "json_valid_pct": round(100 * n_sesi_valid / n_sesi, 1) if n_sesi else 0.0,
            "mae_skor_total": round(mae(sesi_total), 2),
            "mae_skor_per_teknik": round(mae(sesi_per_teknik), 2),
        },
    }


def cetak_tabel(hasil: dict[str, dict]) -> None:
    """Tabel markdown siap tempel ke proposal."""
    nama = list(hasil)
    print("\n| Metrik | " + " | ".join(nama) + " |")
    print("|---|" + "---|" * len(nama))
    baris = [
        ("Akurasi teknik (turn)", lambda h: f"{h['coach_turn']['akurasi_teknik']:.3f}"),
        ("Macro-F1 teknik (turn)", lambda h: f"{h['coach_turn']['macro_f1_teknik']:.3f}"),
        ("Akurasi kualitas (turn)", lambda h: f"{h['coach_turn']['akurasi_kualitas']:.3f}"),
        ("JSON valid sesi (%)", lambda h: f"{h['coach_sesi']['json_valid_pct']:.1f}"),
        ("MAE skor_total", lambda h: f"{h['coach_sesi']['mae_skor_total']:.2f}"),
        ("MAE skor per-teknik", lambda h: f"{h['coach_sesi']['mae_skor_per_teknik']:.2f}"),
    ]
    for judul, f in baris:
        print(f"| {judul} | " + " | ".join(f(hasil[n]) for n in nama) + " |")
    print("\n(Akurasi/F1/JSON valid: makin tinggi makin baik. MAE: makin rendah makin baik.)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--mode", choices=["base", "adapter", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah contoh (smoke test).")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="Simpan hasil sebagai JSON.")
    args = ap.parse_args()

    contoh = [json.loads(l) for l in args.test.read_text(encoding="utf-8").splitlines() if l.strip()]
    contoh = [c for c in contoh if c["meta"]["tipe"] in ("coach_turn", "coach_sesi")]
    if args.limit:
        contoh = contoh[: args.limit]
    if not contoh:
        print(f"ERROR: tidak ada contoh coach di {args.test}.", file=sys.stderr)
        return 2
    print(f"{len(contoh)} contoh uji {dict(Counter(c['meta']['tipe'] for c in contoh))}")

    if args.mode in ("adapter", "both") and not args.adapter:
        print("ERROR: --mode adapter/both butuh --adapter.", file=sys.stderr)
        return 2

    hasil: dict[str, dict] = {}
    for nama, adapter in [("base", None), ("fine-tuned", args.adapter)]:
        if args.mode == "base" and nama != "base":
            continue
        if args.mode == "adapter" and nama != "fine-tuned":
            continue
        print(f"\nMemuat model: {nama}")
        model, tok = muat_model(args.base_model, adapter, not args.no_4bit)
        hasil[nama] = evaluasi(model, tok, contoh, nama)
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — pembersihan memori opsional
            pass

    cetak_tabel(hasil)
    if args.out:
        args.out.write_text(json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nHasil tersimpan di {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
