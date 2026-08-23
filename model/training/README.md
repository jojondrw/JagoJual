# Model — Fine-tuning (dikerjakan di Kaggle)

> 📘 **Mau langsung jalan?** Ikuti [`TRAINING.md`](TRAINING.md) — runbook langkah demi
> langkah dari dataset sampai adapter yang jalan di backend, lengkap dengan cara
> memastikan tiap langkah benar dan troubleshooting. Dokumen di bawah ini rujukan
> desain & status skrip.

> **Offline & sekali jalan.** Skrip di folder ini TIDAK dipanggil backend saat
> demo (sesuai batasan MVP rulebook: inferensi berparameter statis). Output-nya
> berupa **adapter LoRA** yang di-commit ke `../checkpoints/` lalu di-load backend
> di MODE=local.

## Ringkasan
- **Base:** `Qwen/Qwen2.5-3B-Instruct` (Apache-2.0) — dipilih karena inferensinya muat di GPU demo 6 GB
- **Metode:** QLoRA (4-bit) via `peft` + `trl`
- **Platform:** Kaggle (T4 / P100, ~30 jam/minggu gratis)
- **Fokus fine-tune:** mode **Pelatih** (penilaian teknik). Role-play andalkan base.

## Status skrip

| File | Fungsi | Status |
|---|---|---|
| `build_scenario_matrix.py` | Rencanakan sel dialog seimbang → `data/scenario_matrix.json` | ✅ siap |
| `prompts.py` | Template prompt generate + prompt inferensi (roleplay/coach) | ✅ siap |
| `1_generate_data.py` | Generate + validasi dialog berlabel → `data/dialogs/*.json` | ✅ siap |
| `rubric.py` | Rubrik deterministik: label emas → target skor sesi | ✅ siap |
| `2_prepare_sft.py` | Flatten dialog → contoh SFT, split 80/10/10 → `data/sft/*.jsonl` | ✅ siap |
| `3_finetune_qlora.py` | QLoRA Qwen2.5, fokus mode Pelatih | ✅ siap |
| `4_evaluate.py` | F1 teknik per-giliran + MAE skor sesi, adapter vs base | ✅ siap |

## Langkah data (sudah bisa dijalankan)

```bash
cd model/training

# 1) matriks skenario (300 sel: 150 otomotif + 150 elektronik, seimbang & deterministik)
python build_scenario_matrix.py --per-bidang 150

# 2) generate dialog — client OpenAI-compatible (provider mana pun / server lokal vLLM/Ollama)
export JAGO_LLM_BASE_URL=https://.../v1
export JAGO_LLM_API_KEY=sk-...
export JAGO_LLM_MODEL=qwen2.5-72b-instruct
python 1_generate_data.py --dry-run     # cek prompt tanpa panggil API
python 1_generate_data.py --limit 5     # smoke test
python 1_generate_data.py               # semua (resume otomatis: skip yang sudah ada)

# MULTI-LLM: jalankan lagi dengan --model / JAGO_LLM_MODEL berbeda untuk diversity.
# File yang sudah ada di-skip; hapus file target dulu bila ingin menimpa dengan model lain.

# 3) validasi seluruh dialog (schema Draft-07 + aturan bisnis per-speaker)
python 1_generate_data.py --validate-only

# 4) flatten jadi contoh SFT -> data/sft/{train,val,test}.jsonl + stats.json
python 2_prepare_sft.py --stats-only     # lihat distribusi dulu
python 2_prepare_sft.py
```

Dependensi: stdlib untuk dry-run & validasi ringan. Validasi schema penuh: `pip install jsonschema`.
Semua label bersumber dari `data/taxonomy.json` (single source of truth, selaras `backend/app/schemas.py`).

## Bentuk contoh SFT

`2_prepare_sft.py` menurunkan tiga jenis contoh dari tiap dialog, semuanya format chat
(`{"messages": [system, user, assistant]}`) supaya langsung cocok dengan chat template Qwen:

| Jenis | Per dialog | Output assistant | Gunanya |
|---|---|---|---|
| `coach_sesi` | 1 | JSON `{skor_total, per_teknik, saran}` | **Tugas utama.** Bentuknya persis `EvaluateResponse` di backend. |
| `coach_turn` | 1 per giliran sales | JSON `{teknik, kualitas}` | Pelabelan murni → dasar metrik F1 di `4_evaluate.py`. |
| `roleplay` | 2 (bisa diatur) | teks balasan pelanggan | Porsi kecil; mode Pelanggan mengandalkan base model. |

Target `coach_sesi` **tidak** dibuat dengan memanggil LLM lagi, melainkan diturunkan dari
label emas per-turn lewat rubrik tetap di `rubric.py` (bobot teknik, skor per teknik, pemilihan
saran). Konsekuensi jujurnya: kalimat saran berasal dari bank teks, jadi model belajar **memilih
diagnosis yang tepat**, bukan mengarang saran baru — tulis ini di proposal §Metodologi.

Split 80/10/10 dikelompokkan per `scenario_id` dan distratifikasi per `bidang`, jadi tidak ada
giliran dari satu dialog yang bocor ke dua split. `data/sft/` sengaja **tidak di-commit**
(turunan deterministik; sumber kebenaran tetap `data/dialogs/`).

> **Aturan anotasi penting:** satu giliran sales = satu teknik. Ucapan yang sekaligus closing
> dan upsell harus dipecah jadi dua giliran, kalau tidak rubrik akan menyimpulkan salah satu
> teknik "tidak dipakai" padahal terlihat di transkrip.

## Fine-tune & evaluasi (di Kaggle)

Buka notebook Kaggle, **Settings → Accelerator → GPU** dan **Internet: On**, lalu:

```bash
pip install -r requirements-train.txt

# 1) latih adapter LoRA (fokus mode Pelatih)
python 3_finetune_qlora.py \
    --train ../../data/sft/train.jsonl \
    --val   ../../data/sft/val.jsonl \
    --out   ../checkpoints

# base model default sudah Qwen2.5-3B-Instruct, sama dengan yang dipakai backend.
# Jangan ganti tanpa melatih ulang: adapter tidak bisa dipasang ke base beda ukuran.

# 2) buktikan fine-tune memberi nilai tambah (angka ini masuk proposal)
python 4_evaluate.py --test ../../data/sft/test.jsonl \
                     --adapter ../checkpoints --mode both \
                     --out ../checkpoints/eval.json
```

Pilihan hyperparameter (default): QLoRA 4-bit NF4 + double-quant, LoRA `r=16 alpha=32
dropout=0.05` pada seluruh proyeksi attention + MLP Qwen2, `lr=2e-4` cosine, 3 epoch,
efektif batch 8 (batch 1 × grad-accum 8), `paged_adamw_8bit`, gradient checkpointing.
bf16/fp16 dipilih otomatis (T4 Kaggle tidak mendukung bf16). `--save-steps 100` supaya
sesi Kaggle yang mati tidak menghanguskan seluruh training.

`3_finetune_qlora.py` menulis `training_meta.json` di folder adapter — hyperparameter &
statistik data yang menghasilkan adapter itu, supaya hasil bisa ditelusuri.

**Metrik `4_evaluate.py`** (dihitung di `test.jsonl`, dialog yang tak pernah dilihat saat latih):

| Metrik | Arti |
|---|---|
| Akurasi & macro-F1 teknik | Ketepatan melabeli teknik per giliran. Macro-F1 dipakai karena distribusi teknik tidak rata. |
| Akurasi kualitas | Ketepatan membedakan giliran `baik` vs `lemah`. |
| JSON valid sesi (%) | Berapa persen keluaran bisa di-parse jadi `EvaluateResponse` — syarat kepakaian di backend. |
| MAE skor_total & per-teknik | Selisih rata-rata skor model vs rubrik emas. |

Prediksi yang gagal di-parse **dihitung salah**, bukan dibuang — kalau dibuang, model yang
sering ngawur justru terlihat unggul. Dekoding greedy supaya angkanya bisa direproduksi.

Terakhir: simpan adapter → push HF Hub / download → commit ke `../checkpoints/`
(`.gitignore` sudah mengizinkan `*.safetensors` + `adapter_config.json` di folder itu).
