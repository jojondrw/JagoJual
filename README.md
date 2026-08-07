# JagoJual: AI Sales Trainer untuk Toko Offline

Simulator latihan percakapan penjualan berbasis AI untuk tenaga sales di Indonesia
(MVP: **otomotif** & **elektronik**). Trainee berlatih ngobrol melawan **pelanggan AI**
yang punya persona berbeda-beda, lalu mendapat **skor + feedback per teknik jualan** dari
**AI pelatih**, semuanya dalam Bahasa Indonesia.

**Pilar AIC COMPFEST 18: Smart Commerce.** Selaras dengan sponsor **WIZ.AI** (voice-AI untuk sales & CS di Asia Tenggara).

---

## Kenapa ini penting

Toko offline melatih sales lewat role-play manual: butuh senior yang jadi "pelanggan
palsu", subyektif, dan tidak scalable. Sales baru sering belajar langsung ke pelanggan
asli, artinya **belajar dari transaksi yang gagal**. JagoJual memindahkan latihan itu ke
lingkungan aman: trainee bisa mengulang skenario sulit (pelanggan sensitif harga, cuma
lihat-lihat, banyak nanya) sebanyak yang dia mau, dan dapat penilaian objektif yang
**konsisten**, bukan mood trainer.

**Kenapa butuh AI beneran (bukan if-else):** menilai apakah sebuah kalimat sales itu
"menggali kebutuhan" vs "langsung closing", dan apakah eksekusinya *baik* atau *lemah*,
butuh pemahaman bahasa yang tidak bisa dicocokkan kata kunci. Itulah komponen yang
**kami fine-tune**.

---

## Bukti fine-tuning berhasil (`model/checkpoints/eval.json`)

Adapter kami diuji melawan base model yang sama pada test set yang sama:

| Metrik (mode Pelatih) | Base Qwen2.5-7B | **Fine-tuned** |
|---|---|---|
| Akurasi label teknik | 76,8% | **100%** |
| Macro-F1 teknik | 0,76 | **1,0** |
| Akurasi label kualitas (baik/lemah) | 82,7% | **100%** |
| MAE skor sesi (0 sampai 100) | 14,23 | **3,57** |
| MAE skor per-teknik | 50,6 | **0,83** |

> **Catatan jujur soal angka 100%.** Nilai sempurna ini mencerminkan **konsistensi data
> sintetik** kami: pola pelabelannya teratur sehingga tugas sangat terpelajari. Pada
> percakapan dunia nyata yang lebih berantakan, kami memperkirakan akurasi lebih rendah.
> Yang valid sebagai bukti adalah **peningkatan relatifnya** (76% ke 100%, MAE 14,23 ke 3,57):
> fine-tuning jelas mengajari model tugas yang tidak dikuasai base model. Validasi dengan
> transkrip asli adalah langkah pengembangan berikutnya.

---

## Arsitektur

```
+----------------------+     HTTP/JSON      +-----------------------+
|  Frontend Next.js 14  | =================> |   Backend FastAPI      |
|  1. pilih skenario    |                    |                        |
|  2. role-play + skor  | <================= |  /api/chat    Pelanggan|--+
|  (opsi: voice STT/TTS)|                    |  /api/evaluate Pelatih |  |
+----------------------+                    +-----------------------+  |
                                                                       v
                                                  +----------------------------------+
                                                  | MODE=local: Qwen2.5-7B (4-bit)    |
                                                  |   + adapter LoRA fine-tuned        |
                                                  | MODE=mock : heuristik (tanpa AI)   |
                                                  +----------------------------------+
```

- **Dua peran model, satu adapter:** *Pelanggan* (role-play persona) & *Pelatih* (skor + saran JSON).
- **MODE=mock** (default): jalan **tanpa GPU/model**, jalur yang dijamin bisa dijalankan panitia di laptop apa pun.
- **MODE=local**: LLM lokal 4-bit + adapter LoRA. Butuh GPU NVIDIA. Inilah "AI beneran"-nya.
- **Voice (opsional):** STT/TTS via Web Speech API browser (id-ID), tanpa server/GPU. Default OFF.

---

## Menjalankan

### Cara tercepat: mock mode (tanpa GPU, untuk juri)

```bash
git clone https://github.com/jojondrw/JagoJual.git && cd JagoJual
docker compose up --build
```

- UI: http://localhost:3000 · API docs: http://localhost:8000/docs

Balasan pelanggan & skor berasal dari heuristik, cukup untuk mendemokan **alur end-to-end**.
UI menampilkan badge **"Mode contoh · tanpa AI"** supaya statusnya tidak tertukar dengan
keluaran model asli.

### AI beneran: MODE=local (butuh GPU NVIDIA)

Dikembangkan & didemokan di **RTX 4050 6GB** (4-bit). Adapter LoRA sudah ada di `model/checkpoints/`.

```bash
cd backend
pip install -r requirements.txt -r requirements-model.txt   # torch: sesuaikan versi CUDA
JAGOJUAL_MODE=local uvicorn app.main:app
```

Bobot base (~5 GB terkuantisasi) diunduh otomatis dari Hugging Face saat start pertama.
Model dimuat **sekali** ke proses. **Tidak ada training, auto-tuning, atau feedback loop
saat demo**; parameter statis sesuai batasan MVP rulebook.

Kalau model/adapter gagal dimuat, API menjawab **503 dengan sebab yang jelas** dan **tidak**
diam-diam berpindah ke mock. Hasil mock bukan penilaian AI, jadi menyajikannya seolah
keluaran model akan menyesatkan.

### Tanpa Docker (dev)

```bash
# Backend
cd backend && python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt && uvicorn app.main:app --reload
# Frontend (terminal lain)
cd frontend && npm install && npm run dev
```

### Tes backend
```bash
cd backend && pip install -r requirements-dev.txt && pytest
```
22 tes: endpoint mock mode + prompt & normalisasi keluaran MODE=local (tanpa GPU, pemuatan model tidak ikut diuji).

---

## Bagaimana AI-nya dibuat (offline, di Kaggle)

Semua di `model/training/`, **tidak pernah dipanggil backend saat demo** (rulebook membatasi
implementasi AI pada inferensi berparameter statis).

```
build_scenario_matrix.py  -> data/scenario_matrix.json   (300 sel skenario)
1_generate_data.py        -> data/dialogs/*.json          (301 dialog berlabel)
rubric.py                 -> target skor sesi deterministik dari label emas
2_prepare_sft.py          -> data/sft/*.jsonl             (split 80/10/10, tidak di-commit)
3_finetune_qlora.py       -> model/checkpoints/           (adapter LoRA, di-commit)
4_evaluate.py             -> eval.json                    (adapter vs base)
```

- **Base:** Qwen2.5-7B-Instruct (Apache-2.0) + LoRA r=16. Dilatih di Kaggle (T4, 4-bit QLoRA).
- **Data:** sintetik & di-ground ke taksonomi teknik/keberatan/persona (`data/README.md`).
- Runbook lengkap: [`model/training/TRAINING.md`](model/training/TRAINING.md).

> **Penting: `backend/app/prompts.py` dan `model/training/prompts.py` adalah kontrak berpasangan.**
> Adapter dilatih pada format prompt di sana. Kalau salah satu diubah tanpa yang lain, model
> melihat format berbeda dari yang dilatihkan dan mutunya turun **tanpa error yang kelihatan**.
> Ubah keduanya bersamaan.

---

## Struktur

```
JagoJual/
├── backend/
│   ├── app/
│   │   ├── main.py        FastAPI: /api/scenarios, /api/chat, /api/evaluate
│   │   ├── prompts.py     prompt inferensi, KEMBAR dengan model/training/prompts.py
│   │   ├── llm.py         MODE=local: Qwen2.5 4-bit + adapter LoRA
│   │   ├── mock.py        MODE=mock: heuristik kata kunci (bukan AI)
│   │   └── scenarios.py   6 skenario latihan (persona pakai label taksonomi)
│   └── tests/             22 tes, jalan tanpa GPU
├── frontend/              Next.js 14 (2 layar: pilih skenario, lalu role-play + rapor + voice)
├── model/
│   ├── training/          OFFLINE (Kaggle): pipeline data, fine-tune, eval
│   └── checkpoints/       adapter LoRA fine-tuned + eval.json (bukti), committed
└── data/                  taksonomi, matriks skenario, 301 dialog berlabel
```

Dokumen pendukung: [`PLAN.md`](PLAN.md) (desain & rasional lengkap, pemetaan ke kriteria penilaian) ·
[`data/README.md`](data/README.md) (taksonomi label, cara data dibuat, etika & lisensi).

---

## Status

| Tahap | Status |
|---|---|
| M0: Scaffold (mock mode jalan) | ✅ |
| M1: Pipeline dataset + 301 dialog berlabel | ✅ |
| M2: Fine-tune QLoRA + evaluasi (Kaggle) | ✅ adapter & eval di-commit |
| M3: Integrasi LLM lokal (MODE=local) | ✅ |
| M4: Polish frontend + voice opsional | ✅ |
| M5: Proposal PDF + video PoW + video inovasi | ⬜ **in progress** |

**Tenggat penyisihan: 25 Agustus 2026, 23.55 WIB**, batas commit/push terakhir *sekaligus*
batas submisi berkas. Deliverable: link repo (public, README + docker compose), video Proof of
Work ≤7 menit (YouTube unlisted, **dilarang di-cut**, hanya fast-forward + voice over), video
inovasi ≤5 menit (public), proposal PDF ≤20 halaman.

### Utang teknis yang diketahui (bukan bug, hal yang mungkin ditanya juri)

| Hal | Status / dampak |
|---|---|
| 6 skenario hardcoded, sementara `scenario_matrix.json` punya 300 sel | Dataset besar, app menawarkan skenario tulisan tangan. Sengaja untuk MVP; scenario generator adalah pengembangan lanjutan. |
| `npm audit` menyisakan temuan high pada Next.js (tertutup hanya dengan Next 16) | Ditunda: app jalan di localhost, tanpa middleware/`next/image`/server actions. Putuskan sebelum submisi. |

---

## Konvensi commit (Conventional Commits, wajib rulebook)
`feat:` fitur baru · `fix:` perbaikan bug · `refactor:` ubah struktur tanpa ubah fungsi · `docs:` dokumentasi
· panduan: https://www.conventionalcommits.org
