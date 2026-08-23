# Panduan Training JagoJual (Kaggle) — Runbook

Panduan langkah demi langkah dari **dataset dialog** sampai **adapter LoRA yang jalan di
backend**. Ikuti berurutan; tiap langkah punya cara memastikan hasilnya benar sebelum lanjut.

> **Kenapa di Kaggle, bukan lokal?** GPU demo (RTX 4050, 6 GB) cukup untuk *inferensi*
> 4-bit tapi tidak untuk *training*. Kaggle memberi P100 16 GB / T4×2 gratis ~30 jam
> per minggu. Training dikerjakan sekali di sana, hasilnya (adapter, puluhan MB) dibawa
> pulang ke repo.
>
> **Kenapa training tidak ada di repo submission?** Rulebook membatasi implementasi AI
> pada inferensi berparameter statis — tanpa auto-tuning, bulk testing, atau feedback loop.
> Skrip di folder ini offline dan tidak pernah dipanggil backend saat demo.

---

## Peta alur

```
data/dialogs/*.json          (dialog berlabel — sudah divalidasi)
        │  2_prepare_sft.py
        ▼
data/sft/{train,val,test}.jsonl
        │  3_finetune_qlora.py        ← butuh GPU Kaggle
        ▼
model/checkpoints/           (adapter LoRA + training_meta.json)
        │  4_evaluate.py              ← bukti fine-tune berguna
        ▼
angka untuk proposal + backend MODE=local
```

---

## Langkah 0 — Prasyarat

1. **Akun Kaggle terverifikasi nomor HP.** Tanpa verifikasi, GPU & Internet tidak bisa
   diaktifkan. Cek di *Settings → Phone Verification*.
2. Buat notebook baru → panel kanan:
   - **Accelerator:** `GPU P100` (utama) atau `GPU T4 x2`
   - **Internet:** `On` (dibutuhkan untuk mengunduh base model dari Hugging Face)
3. Pastikan dataset dialog sudah ada di `data/dialogs/` dan **sudah di-push ke GitHub** —
   cara termudah membawanya ke Kaggle adalah meng-clone repo.

**Cek kuota sebelum mulai.** Kuota GPU Kaggle mingguan, dan reset tiap Sabtu 00:00 UTC.
Training 300 dialog ≈ 1 jam di T4/P100; sisakan kuota untuk evaluasi.

---

## Langkah 1 — Siapkan repo & dependensi di Kaggle

Sel pertama notebook:

```python
!git clone https://github.com/Kenrockender/jagojual-aic.git /kaggle/working/jagojual
%cd /kaggle/working/jagojual/model/training
!pip install -q -r requirements-train.txt
```

`requirements-train.txt` **sengaja tidak memasang torch** — image Kaggle sudah membawa
torch yang cocok dengan driver CUDA-nya, dan memasang ulang justru sering merusak.

Verifikasi GPU terbaca:

```python
import torch
print(torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
```

> `bf16: False` itu normal di T4 — skrip otomatis memakai fp16.

---

## Langkah 2 — Validasi dialog

Jangan melatih data yang belum divalidasi; label yang salah akan diserap model tanpa error.

```python
!python 1_generate_data.py --validate-only
```

Yang dicek: JSON Schema Draft-07, label per-speaker (`teknik`+`kualitas` wajib di turn
`sales`, `keberatan` valid di turn `pelanggan`), dan keberatan utama benar-benar muncul.

Harus **0 error** sebelum lanjut. Kalau ada yang gagal, perbaiki file dialognya.

> **Aturan anotasi yang sering terlewat:** satu giliran sales = satu teknik. Ucapan yang
> sekaligus closing dan upsell harus dipecah jadi dua giliran. Kalau digabung, rubrik
> penilaian sesi akan menyimpulkan salah satu teknik "tidak dipakai" padahal jelas terlihat
> di transkrip — dan model belajar mengatakan hal yang tidak benar.

---

## Langkah 3 — Bentuk contoh SFT

Lihat distribusinya dulu tanpa menulis file:

```python
!python 2_prepare_sft.py --stats-only
```

Perhatikan tiga hal di keluarannya:

| Yang dilihat | Sehat kalau | Kalau tidak |
|---|---|---|
| `distribusi teknik` | tiap teknik ratusan contoh, tidak ada yang < 5% total | tambah sel skenario untuk teknik yang jarang |
| `distribusi kualitas` | `lemah` sekitar 20–35% | terlalu sedikit `lemah` → model tak bisa membedakan mutu |
| `val` / `test` | masing-masing ± 10% dialog | kalau 0, dialog < 10 per bidang |

Kalau sudah wajar, tulis filenya:

```python
!python 2_prepare_sft.py
```

Menghasilkan `data/sft/{train,val,test}.jsonl` + `stats.json`. Tiap baris berformat chat
(`{"messages": [system, user, assistant]}`) supaya langsung cocok dengan chat template Qwen.

Tiga jenis contoh dihasilkan per dialog: `coach_sesi` (1, tugas utama), `coach_turn`
(1 per giliran sales), dan `roleplay` (2, porsi kecil). Split 80/10/10 dikelompokkan
per `scenario_id` **dan** distratifikasi per bidang — tidak ada giliran dari satu dialog
yang bocor ke dua split.

> `data/sft/` tidak di-commit: isinya turunan deterministik dari `data/dialogs/`.
> Sumber kebenaran tetap satu.

---

## Langkah 4 — Fine-tune QLoRA

```python
!python 3_finetune_qlora.py \
    --train ../../data/sft/train.jsonl \
    --val   ../../data/sft/val.jsonl \
    --out   ../checkpoints
```

Default yang dipakai dan alasannya:

| Parameter | Nilai | Kenapa |
|---|---|---|
| base model | `Qwen/Qwen2.5-3B-Instruct` | Apache-2.0 (aman untuk lomba), Bahasa Indonesia bagus, muat inferensi 4-bit di GPU demo 6 GB |
| kuantisasi | 4-bit NF4 + double quant | menekan VRAM saat training, dan wajib untuk inferensi di GPU demo 6 GB |
| LoRA | `r=16, alpha=32, dropout=0.05` | kapasitas cukup untuk keluaran terstruktur, adapter tetap kecil |
| target modules | semua proyeksi attention + MLP | melatih MLP juga membantu tugas format JSON |
| lr / scheduler | `2e-4` / cosine | lazim untuk QLoRA |
| epoch | 1 | dataset kecil dan polanya konsisten; lebih dari ini mudah overfit |
| batch efektif | 8 (1 × grad-accum 8) | menahan pemakaian VRAM |
| optimizer | `paged_adamw_8bit` | hemat memori, tahan lonjakan |

**Pantau `loss`.** Turun lalu mendatar = wajar. Kalau `eval_loss` mulai naik sementara
`train_loss` terus turun, itu overfit — potong lewat `--max-steps`.

Selesai training, folder `../checkpoints` berisi adapter + `training_meta.json`
(catatan hyperparameter & statistik data — jangan dihapus, itu jejak reproduksi kamu).

### Base model training dan backend harus sama

Adapter LoRA **tidak bisa** dipasang ke base model yang berbeda ukuran — dimensi
matriksnya tidak cocok dan akan error saat load. Adapter yang sekarang ada di
`model/checkpoints/` dilatih di atas `Qwen/Qwen2.5-3B-Instruct`, jadi backend juga
harus `JAGOJUAL_BASE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct` (sudah jadi default di
`app/config.py`). Kalau suatu saat base-nya diganti, ganti keduanya dan latih ulang.

---

## Langkah 5 — Evaluasi (jangan dilewat)

Ini yang mengubah klaim "kami fine-tune" jadi bukti yang bisa diperiksa ulang.

```python
!python 4_evaluate.py \
    --test ../../data/sft/test.jsonl \
    --adapter ../checkpoints \
    --mode both \
    --out ../checkpoints/eval.json
```

Keluarannya tabel markdown siap tempel:

| Metrik | Arti | Arah baik |
|---|---|---|
| Akurasi teknik (turn) | ketepatan melabeli teknik per giliran | tinggi |
| Macro-F1 teknik | sama, tapi adil untuk kelas minoritas | tinggi |
| Akurasi kualitas | bisa bedakan giliran `baik` vs `lemah` | tinggi |
| JSON valid sesi (%) | keluaran bisa dipakai backend | tinggi |
| MAE skor_total | selisih skor model vs rubrik emas | rendah |

**Yang dicari:** kolom `fine-tuned` menang telak di **JSON valid** dan **macro-F1**. Base
model biasanya sering membalas dengan prosa ("Tentu! Berikut penilaiannya…") sehingga gagal
di-parse — itulah nilai konkret fine-tuning yang bisa kamu tunjukkan ke juri.

Dekoding greedy, jadi angkanya bisa direproduksi. Prediksi yang gagal di-parse dihitung
**salah**, bukan dibuang — kalau dibuang, model yang sering ngawur malah terlihat unggul.

> Kalau hasilnya ternyata fine-tuned **tidak** lebih baik, catat apa adanya berikut dugaan
> sebabnya. Hasil negatif yang jujur dan
> dianalisis lebih bernilai daripada angka yang dipoles.

---

## Langkah 6 — Bawa adapter ke repo

Adapter berukuran puluhan MB, aman masuk git. `.gitignore` sudah mengizinkan
`*.safetensors` + `adapter_config.json` di `model/checkpoints/`.

Dari Kaggle, unduh folder `model/checkpoints` lewat panel *Output*, lalu di mesin lokal:

```bash
git add model/checkpoints
git commit -m "feat: tambah adapter LoRA hasil fine-tune QLoRA Qwen2.5-3B"
git push
```

Sertakan juga `training_meta.json` dan `eval.json` — dua berkas itu yang membuat angka
hasil evaluasi bisa diperiksa ulang, bukan sekadar diklaim.

---

## Langkah 7 — Jalankan dengan model asli

```bash
cd backend
pip install -r requirements.txt -r requirements-model.txt
JAGOJUAL_MODE=local uvicorn app.main:app
```

Badge di UI harus berubah dari **"Mode contoh (tanpa AI)"** jadi **"AI lokal aktif"**.
Kalau masih mock, `JAGOJUAL_MODE` belum terbaca.

Cek log saat start. Kalau muncul:

```
Adapter LoRA tidak ditemukan di ../model/checkpoints — memakai base model tanpa fine-tune.
```

berarti backend jalan dengan base polos; `adapter_config.json` tidak ada di path itu.

---

## Troubleshooting

**`CUDA out of memory` saat training**
Turunkan berurutan: `--max-seq-len 1024` → `--grad-accum 16 --batch-size 1` →
`--lora-r 8` → pindah ke `Qwen2.5-3B-Instruct`. Jangan naikkan batch size untuk
"mempercepat"; grad-accum memberi batch efektif sama dengan memori jauh lebih kecil.

**Sesi Kaggle mati di tengah training**
Sudah diantisipasi: `--save-steps 100` menulis checkpoint berkala ke `--out`. Jalankan
ulang perintah yang sama dengan tambahan `--resume`:

```python
!python 3_finetune_qlora.py --train ... --val ... --out ../checkpoints --resume
```

Kalau ternyata belum ada checkpoint, skrip memberi tahu lalu mulai dari awal (bukan error).
Sesi Kaggle mati otomatis setelah 12 jam / 20 menit idle — jangan tutup tab, dan pakai
*Save Version → Run All* untuk eksekusi latar belakang.

**`TypeError` dari `SFTTrainer` / `SFTConfig`**
API TRL sering berubah antar versi minor. Skrip ini ditulis untuk `trl>=0.12`
(`SFTConfig`, `max_length`, `processing_class`). Kalau Kaggle memasang versi lain:
`!pip install -q "trl==0.12.*"`.

**`bitsandbytes` gagal / tidak mendeteksi CUDA**
Biasanya karena torch terpasang ulang. Restart session, jangan `pip install torch`.

**Keluaran model penuh `{{` atau prosa, JSON gagal di-parse**
Cek prompt latih dan prompt inferensi identik. `model/training/prompts.py` dan
`backend/app/prompts.py` adalah **kontrak berpasangan** — kalau salah satu diubah tanpa
yang lain, model melihat format berbeda dari yang dilatihkan dan mutunya turun tanpa
error yang kelihatan.

**Loss langsung `nan`**
Umumnya fp16 + lr terlalu besar. Turunkan `--lr 1e-4`, atau pakai GPU yang mendukung bf16 (P100/A100).

---

## Checklist sebelum dianggap selesai

- [ ] `1_generate_data.py --validate-only` → 0 error
- [ ] `2_prepare_sft.py --stats-only` → distribusi teknik & kualitas wajar, val/test tidak kosong
- [ ] Training selesai, `eval_loss` tidak menanjak
- [ ] `4_evaluate.py --mode both` dijalankan, fine-tuned unggul (atau ketidakunggulannya dianalisis)
- [ ] `model/checkpoints/` berisi adapter + `training_meta.json` + `eval.json`, sudah di-commit
- [ ] Backend `JAGOJUAL_MODE=local` jalan, badge UI berubah jadi "AI lokal aktif"
- [ ] Base model yang dipakai backend **sama persis** dengan yang dipakai saat training
