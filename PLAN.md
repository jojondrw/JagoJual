# PLAN — AIC COMPFEST 18 · AI Sales Trainer untuk Toko Offline

> Dokumen ini adalah **rencana untuk direview**, belum implementasi.
> Nama kerja: **JagoJual** (alternatif: SigapSales, PramuniagaAI, TokoCoach). Silakan pilih/ganti.

---

## 1. Ringkasan Satu Paragraf

**JagoJual** adalah simulator latihan percakapan penjualan berbasis AI untuk **tenaga sales lintas industri** di Indonesia (otomotif, elektronik, produk finansial, dll). Trainee memilih **bidang + skenario** pelanggan (mis. "otomotif: pelanggan bilang boros bensin"), lalu berlatih percakapan jualan melawan **pelanggan AI**. Di akhir sesi, trainee mendapat **skor + feedback coaching** per teknik jualan (sapaan, gali kebutuhan, atasi keberatan, closing, upsell). Otaknya adalah **satu LLM open-weights yang di-fine-tune (LoRA)** — model yang sama memerankan pelanggan **dan** menilai teknik jualan trainee.

**Prinsip inti:** keterampilan jualan bersifat **universal** (taksonomi teknik sama di semua bidang); yang berbeda hanya **konteks produk & keberatan khas** per bidang. Jadi arsitektur satu, data diperkaya lintas bidang.

**Masuk pilar rulebook:** Smart Commerce — "penerapan AI di sisi konsumen, **sales operasional**, serta transaksi komersial".

---

## 2. Kenapa Ini "Baru" & Aman dari Tuduhan Non-orisinal

| Aspek | Sera (yang di-copy) | StokCerdas (project AIC lama) | **JagoJual (baru)** |
|---|---|---|---|
| Domain | Onboarding asuransi | Forecasting stok ritel | **Pelatihan sales lintas industri** |
| Inti AI | RAG + LLM API (tanpa fine-tune) | Time-series (Chronos) | **LLM open-weights fine-tuned (LoRA)** |
| Interaksi | Tanya-jawab dokumen | Upload CSV → ramalan | **Role-play percakapan + scoring** |
| Pengguna | Staf baru | Pemilik toko | **Sales otomotif / elektronik / finansial / dll** |

**Strategi orisinalitas:** pakai **pola arsitektur** Sera/StokCerdas (Next.js + FastAPI + Docker Compose, folder `model/training/` terpisah) sebagai *referensi struktur*, tapi **kode ditulis ulang** dengan branding, skema, dan logika baru. Repo baru dibuat bersih dalam periode lomba (17 Jun – 25 Agu 2026), commit mengikuti Conventional Commits.

---

## 3. Batasan MVP (Pemetaan Eksplisit ke Rulebook §Teknis 1)

| Batasan Rulebook | Yang KITA lakukan | Yang TIDAK kita lakukan |
|---|---|---|
| **FE: input tunggal → output AI** | Satu layar role-play: ketik pesan → balasan pelanggan AI + kartu feedback. Plus satu layar pilih skenario. | ❌ Dashboard analitik, ❌ auth kompleks, ❌ halaman riwayat |
| **BE: sinkron saja, docker compose** | FastAPI sinkron; model di-load in-process. Jalan via `docker compose up`. | ❌ Background job, ❌ auto data-logging, ❌ DB terdistribusi |
| **Model: core inference, parameter statis** | Adapter LoRA hasil fine-tune **dibekukan**, hanya inferensi saat demo. | ❌ Auto-tuning, ❌ bulk-testing script di repo, ❌ feedback-loop otomatis |

> Fine-tuning dikerjakan **offline di Kaggle** (bukan di repo submission). Adapter hasilnya di-commit & di-load backend. Ini memisahkan "training" (offline, sekali jalan) dari "inference" (statis saat demo) — sesuai aturan.

---

## 4. Arsitektur & Alur

```
                         ┌─────────── OFFLINE (Kaggle) ───────────┐
                         │  generate data → fine-tune LoRA →       │
                         │  evaluasi → simpan adapter (~puluhan MB) │
                         └───────────────────┬─────────────────────┘
                                             │ commit adapter ke repo
                                             ▼
┌────────────┐  pesan trainee   ┌─────────────────────────────────────────┐
│  Frontend  │ ───────────────▶ │            Backend (FastAPI)             │
│ Next.js 14 │                  │  LLM open-weights + adapter LoRA (LOKAL) │
│  Tailwind  │ ◀─────────────── │   • Mode Pelanggan → balasan role-play   │
└────────────┘  balasan + skor  │   • Mode Pelatih   → skor + feedback     │
                                │  (+ mock mode bila GPU/model tak siap)   │
                                └─────────────────────────────────────────┘
                                   Demo: disajikan di RTX 4050 (6GB), 4-bit
```

**Satu model, dua peran** (dibedakan lewat prompt): memerankan pelanggan **dan** menilai teknik jualan. **Fine-tune difokuskan ke mode Pelatih** (penilaian terstruktur — inti inovasi); mode Pelanggan mengandalkan kemampuan role-play base model. Ini menjaga "core inference bersih" (kriteria 25%) sekaligus menghemat data.

---

## 5. Model & Fine-tuning (KEPUTUSAN)

> **Hasil akhir: `Qwen2.5-3B-Instruct`.** Dokumen ini ditulis di awal proyek dan
> merencanakan 7B sebagai pilihan utama. Saat dicoba di GPU demo (RTX 4050 6GB),
> 7B 4-bit memakai VRAM terlalu mepet sehingga generate-nya lambat dan rawan OOM,
> sementara 3B 4-bit cuma butuh sekitar 2,5–3 GB dan jalan stabil. Cadangan itu
> akhirnya jadi pilihan utama, dan itulah yang dilatih serta di-commit di
> `model/checkpoints/`. Bagian di bawah dibiarkan apa adanya sebagai catatan
> pertimbangan awal.

### Pilihan model
- **Rencana awal: `Qwen2.5-7B-Instruct`** — Indonesia bagus, lisensi **Apache-2.0** (aman untuk lomba), muat 4-bit di 4050.
- **Cadangan ringan: `Qwen2.5-3B-Instruct`** — kalau 7B terlalu lambat di 4050 saat demo. **Ini yang akhirnya dipakai.**
- **Alternatif fokus-ID:** SEA-LION v3 / Sahabat-AI (cek lisensi dulu).

### Kenapa LoRA + open-weights (bukan API/IndoBERT)
- **Akurasi:** LLM instruct mampu *reasoning* (menilai kualitas jualan) — jauh di atas classifier kecil.
- **Patuh aturan:** "Model wajib di fine tune" → LoRA = fine-tune sungguhan milik tim.
- **Verifiable & lokal:** adapter di-commit ke repo → juri bisa periksa & jalankan via docker. Tak bergantung API berbayar/koneksi.
- **Gratis:** dilatih di Kaggle (16GB), disajikan di 4050.

### Cara fine-tune (di Kaggle — masuk proposal §Metodologi)
1. **Base:** unduh `Qwen2.5-7B-Instruct` (HF Hub) di notebook Kaggle (P100 16GB / T4×2).
2. **Data:** dataset dialog jualan (lihat §6) diformat jadi contoh **instruksi→output**, **terutama untuk mode Pelatih** (penilaian teknik). Mode Pelanggan cukup sedikit contoh gaya (role-play andalkan base).
3. **QLoRA:** base di-quantize 4-bit (`bitsandbytes`), latih adapter LoRA via `peft` + `trl` `SFTTrainer`. Beberapa ratus–ribu langkah, `lr≈2e-4`, grad checkpointing.
4. **Evaluasi:** ukur akurasi label teknik & kualitas saran mode Pelatih pada test set. Bandingkan vs base non-fine-tune untuk buktikan nilai fine-tuning.
5. **Simpan adapter** → push ke HF Hub / download → **commit ke `model/checkpoints/`**.
6. **Bekukan:** load statis di backend. **Tidak ada** training saat demo.

> Kaggle = latih saja (sesi ephemeral, bukan host). Demo tetap lokal di 4050.

---

## 6. Dataset (Dialog Jualan Sintetik — Buatan Tim)

**Kenapa sintetik?** Tidak ada dataset publik Indonesia untuk **percakapan jualan toko offline**. Review e-commerce domainnya salah (ulasan pasca-beli, bukan dialog, tanpa teknik sales). Rulebook eksplisit mengizinkan **data sintetik**, dan untuk domain seniche ini sintetik yang dibuat sesuai konteks **lebih relevan**.

### Format
**Level dialog** (sumber) → di-*flatten* jadi **contoh SFT** (instruksi→output untuk LLM):

```json
// Level dialog (sumber tunggal)
{
  "scenario_id": "elektronik_harga_01",
  "bidang": "elektronik", "produk": "TV LED 43 inch",
  "persona": {"tipe": "skeptis", "emosi_awal": "ragu"},
  "turns": [
    {"speaker": "pelanggan", "text": "Ini kok lebih mahal dari toko sebelah ya?", "keberatan": "bandingkan_harga", "emosi": "skeptis"},
    {"speaker": "sales", "text": "Betul Pak beda sedikit, tapi sudah termasuk garansi resmi 2 tahun & antar-pasang gratis.", "teknik": "atasi_keberatan", "kualitas": "baik"}
  ]
}

// Diturunkan jadi contoh SFT:
// (a) Mode Pelanggan
{"instruction":"Perankan pelanggan skeptis yang membeli TV, emosi ragu.","input":"Sales: Selamat datang Pak...","output":"Ini kok lebih mahal dari toko sebelah ya?"}
// (b) Mode Pelatih
{"instruction":"Nilai teknik jualan pada respons sales berikut.","input":"Pelanggan: 'Kok mahal?' Sales: 'Betul Pak, tapi termasuk garansi 2 tahun...'","output":"{\"teknik\":\"atasi_keberatan\",\"kualitas\":\"baik\",\"saran\":\"Perkuat dgn bandingkan nilai total.\"}"}
```

### Isi (taksonomi label — SAMA di semua bidang)
- **Teknik sales (6):** `sapa_rapport` · `gali_kebutuhan` · `presentasi_manfaat` · `atasi_keberatan` · `closing` · `upsell`
- **Keberatan pelanggan (6):** `harga` · `bandingkan_kompetitor` · `ragu_kualitas` · `tidak_butuh` · `mau_pikir_pikir` · `cuma_lihat_lihat`

(Kepala "emosi pelanggan" **dibuang** dari MVP — di luar scope inti; emosi awal cukup jadi metadata persona, bukan tugas klasifikasi.)

### Dimensi Bidang — 2 bidang untuk MVP
Taksonomi teknik tetap sama; tiap bidang menyumbang produk, jargon, & keberatan khas.

| Bidang | Status | Konteks | Keberatan khas |
|---|---|---|---|
| **Otomotif (mobil)** | ⭐ **Hero** (fokus demo/video) | Showroom/dealer | boros_bbm, dp_cicilan, bandingkan_merk, harga_jual_kembali |
| **Elektronik & gadget** | ⭐ **Pendukung** | Toko/pramuniaga | garansi, spek, harga_toko_sebelah, awet |
| Kartu kredit · Properti/KPR · Asuransi · FMCG kanvas | ○ **Roadmap** (di proposal, tidak dibangun) | — | — |

> Multi-industri = cerita pertumbuhan di proposal (fleksibilitas arsitektur), **bukan dibangun** di penyisihan. Menjaga MVP tidak overbuilt (kriteria 15%).

### Jumlah
Target **~150 dialog per bidang** × 2 = **~300 dialog** → beberapa ribu contoh SFT (terutama mode Pelatih), split 80/10/10.

### Cara membuat & jaga mutu
1. **Matriks skenario:** vertikal × produk × jenis keberatan × persona.
2. **Generate:** LLM buat dialog per sel, **di-grounding ke framework sales nyata** (SPIN Selling, AIDA, needs-based selling, skrip objection-handling), sekaligus melabeli tiap turn. Buat juga versi "sales lemah" agar model belajar bedanya.
3. **Validasi manusia:** spot-check 10–15% oleh **kenalan berpengalaman sales** (sudah tersedia), koreksi label, seimbangkan kelas. Cantumkan peran validator di proposal (nilai plus Metodologi).

**Kejujuran (ditulis di proposal):** label awal dari LLM → risiko "model hanya sepandai LLM-nya". Mitigasi: grounding + validasi manusia + (bila sempat) transkrip role-play asli. Ini bernilai plus di kriteria Metodologi (15%).

---

## 7. Alur Pengguna (Role-play Loop)

1. Pilih **bidang + skenario** (mis. "Otomotif: pelanggan bilang boros bensin").
2. Pelanggan AI membuka percakapan.
3. Trainee mengetik respons jualan.
4. Backend: LLM (mode Pelanggan) membalas.
5. Ulangi 3–4 beberapa giliran.
6. Tekan "Selesai" → **mode Pelatih** menilai seluruh percakapan → **ringkasan skor per teknik + 2–3 saran konkret**.

> **Feedback di akhir sesi saja** (bukan per-giliran) — satu output matang, lebih sederhana untuk MVP.
> (Langkah 1 & 6 = layar pendukung; 3–4 = "input tunggal → output AI" inti sesuai batasan FE.)

---

## 8. Struktur Repo (Rencana)

```
JagoJual/
├── README.md                 # setup guide + docker compose (WAJIB jelas)
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py           # FastAPI, endpoint sinkron
│       ├── schemas.py
│       ├── scenarios.py      # skenario & persona pelanggan
│       ├── llm.py            # load base + adapter LoRA, inferensi (statis)
│       ├── roleplay.py       # prompt mode Pelanggan
│       ├── coach.py          # prompt mode Pelatih + parse skor
│       └── mock.py           # mock mode (tanpa GPU/model)
├── frontend/                 # Next.js 14 + Tailwind (2 layar)
├── model/
│   ├── training/             # notebook/skrip Kaggle (OFFLINE)
│   │   ├── 1_generate_data.py
│   │   ├── 2_prepare_sft.py
│   │   ├── 3_finetune_qlora.py   # peft + trl (Kaggle)
│   │   ├── 4_evaluate.py
│   │   └── README.md             # cara reproduksi di Kaggle
│   └── checkpoints/          # adapter LoRA (di-commit / dari HF Hub)
└── data/
    ├── dialogs/              # dataset dialog sintetik + label
    └── README.md             # sumber, framework grounding, lisensi
```

---

## 9. Pemetaan ke Kriteria Penilaian (Total 105%)

- **Implementasi & Arsitektur (25%):** LLM fine-tuned (LoRA) sebagai core inference; AI/BE/FE modular; README docker.
- **Orisinalitas & Dampak (20%):** melatih SDM sales (otomotif/elektronik) = masalah nyata & jarang digarap.
- **MVP Ready (15%):** scope pas (2 bidang, fine-tune fokus Pelatih); jujur soal keterbatasan (label seed, model kecil).
- **Proposal (15%):** metodologi kuat — alur pembuatan dataset, langkah QLoRA di Kaggle, evaluasi vs base.
- **Video Promosi (15%):** storytelling "sales naik kelas" (hero: showroom otomotif).
- **Relevansi Tema (10%):** pas di Smart Commerce (sales operasional).
- **Bonus Business/Governance (3.5%):** model bisnis B2B ke dealer/ritel + **bingkai jualan etis** (jujur, tidak manipulatif) + data dianonimkan.

---

## 10. Tahapan Kerja (Draft)

1. **M0 – Setup:** repo, docker skeleton, pilih nama final, siapkan akun Kaggle (verifikasi HP).
2. **M1 – Data:** generate ~300 dialog sintetik (otomotif + elektronik, grounded) + validasi manusia + format SFT.
3. **M2 – Fine-tune (Kaggle):** QLoRA Qwen2.5-3B fokus mode Pelatih, evaluasi vs base, simpan adapter → commit repo.
4. **M3 – Backend:** endpoint sinkron, load base+adapter di 4050, mode Pelanggan & Pelatih, mock mode.
5. **M4 – Frontend:** 2 layar (pilih bidang+skenario → role-play), ringkasan skor akhir sesi.
6. **M5 – Polish:** README, seed skenario, video PoW, proposal.

---

## 11. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Label dari LLM → model "hanya sepandai LLM" | Grounding framework sales + validasi manusia 10–15% + transkrip asli bila sempat |
| 7B lambat/OOM di 4050 (6GB) saat demo | Terbukti terjadi. Turun ke 3B 4-bit (~2,5–3 GB VRAM), plus konteks pendek & jawaban ringkas |
| Sesi Kaggle mati sebelum training selesai | Checkpoint berkala + simpan adapter ke HF Hub; dataset kecil → training singkat |
| Model/GPU tak siap saat panitia menjalankan | **Mock mode** (balasan dari skrip) agar app tetap jalan lokal |
| Kelas tak seimbang di data sintetik | Kontrol distribusi saat generate + weighting saat SFT |

---

## 12. Keputusan Final (Semua Terkunci)

- **Nama kerja:** JagoJual (boleh diganti kapan saja).
- **Bidang:** Otomotif (hero) + Elektronik.
- **Model:** Qwen2.5-3B-Instruct (Apache-2.0), turun dari rencana awal 7B karena VRAM 4050. ✅
- **Arsitektur:** 1 LLM fine-tuned (LoRA), fokus mode Pelatih; role-play andalkan base.
- **Training:** Kaggle (QLoRA) → adapter di repo → demo di 4050 + mock mode.
- **Dataset:** ~300 dialog sintetik (150/bidang), grounded + validasi manusia.
- **Validasi label:** ✅ tersedia kenalan berpengalaman sales.
- **Dibuang:** kepala emosi, auxiliary data publik, bidang finansial, fine-tune role-play, feedback per-giliran.

---

## Referensi

- Base model: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct (Apache-2.0)
- Fine-tuning: `peft` + `trl` (QLoRA) di Kaggle (P100 16GB / T4×2, ~30 jam/minggu)
- Framework grounding generate dialog: SPIN Selling, AIDA, needs-based selling, objection handling
